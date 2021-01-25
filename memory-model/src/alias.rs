use std::{
    collections::{btree_map::Entry as BEntry, hash_map::Entry, BTreeMap, HashMap},
    convert::{Infallible, TryFrom},
    hash::BuildHasher,
    ops::Range,
};

use fxhash::{FxHashMap, FxHashSet};
use slab::Slab;

use crate::{recycle::Recycler, Pointer};

pub trait PointerMap {
    fn with_size(size: u32) -> Self;

    fn size(&self) -> Option<u32>;

    fn for_each<E, F: FnMut(&mut Vec<Pointer>) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        recycler: &mut Recycler<Vec<Pointer>>,
        f: F,
    ) -> Result<(), E>;
}

pub trait Metadata: Copy + Eq {
    type Filter;

    fn alloc() -> Self;

    fn filter_all() -> Self::Filter;

    fn does_invalidate(self, other: Self, filter: &mut Self::Filter) -> bool;
}

impl PointerMap for Vec<Vec<Pointer>> {
    fn with_size(size: u32) -> Self { (0..size).map(|_| Vec::new()).collect() }

    fn size(&self) -> Option<u32> { Some(self.len() as u32) }

    fn for_each<E, F: FnMut(&mut Vec<Pointer>) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        _: &mut Recycler<Vec<Pointer>>,
        mut f: F,
    ) -> Result<(), E> {
        let range = range.start as usize..range.end as usize;
        for byte in &mut self[range] {
            f(byte)?;
        }
        Ok(())
    }
}

impl<B: Default + BuildHasher> PointerMap for HashMap<u32, Vec<Pointer>, B> {
    fn with_size(_: u32) -> Self { Self::default() }

    fn size(&self) -> Option<u32> { None }

    fn for_each<E, F: FnMut(&mut Vec<Pointer>) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        recycler: &mut Recycler<Vec<Pointer>>,
        mut f: F,
    ) -> Result<(), E> {
        for i in range {
            match self.entry(i) {
                Entry::Vacant(vacant) => {
                    let mut vec = recycler.lease();
                    let should_remove = f(&mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                Entry::Occupied(mut entry) => {
                    let should_remove = f(entry.get_mut())?;
                    if should_remove {
                        recycler.put(entry.remove());
                    }
                }
            }
        }
        Ok(())
    }
}

impl PointerMap for BTreeMap<u32, Vec<Pointer>> {
    fn with_size(_: u32) -> Self { Self::default() }

    fn size(&self) -> Option<u32> { None }

    fn for_each<E, F: FnMut(&mut Vec<Pointer>) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        recycler: &mut Recycler<Vec<Pointer>>,
        mut f: F,
    ) -> Result<(), E> {
        for i in range {
            match self.entry(i) {
                BEntry::Vacant(vacant) => {
                    let mut vec = recycler.lease();
                    let should_remove = f(&mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                BEntry::Occupied(mut entry) => {
                    let should_remove = f(entry.get_mut())?;
                    if should_remove {
                        recycler.put(entry.remove());
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MemoryBlock<D, M = BTreeMap<u32, Vec<Pointer>>> {
    memory: M,
    ptr_info: FxHashMap<Pointer, PointerInfo<D>>,
    deallocated: FxHashSet<Pointer>,
    allocations: Slab<Vec<Pointer>>,
    copies: Slab<u32>,
    vec_recycler: Recycler<Vec<Pointer>>,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PointerInfo<D> {
    pub range: Range<u32>,
    pub ptr_ty: PtrType,
    pub owns_allocation: bool,
    pub alloc_id: u32,
    pub meta: D,
    pub copy_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtrType {
    Shared,
    Exclusive,
}

const OK: Result = Ok(());
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    ReborrowInvalidatesSource,
    UseAfterFree(Pointer),
    InvalidPtr(Pointer),
    NotExclusive(Pointer),
    NotShared(Pointer),
    DeallocateNonOwning(Pointer),
    InvalidatesOldMeta(Pointer),
    AllocateRangeOccupied {
        ptr: Pointer,
        range: Range<u32>,
    },
    InvalidForRange {
        ptr: Pointer,
        range: Range<u32>,
    },
    ReborrowSubset {
        ptr: Pointer,
        source: Pointer,
        source_range: Range<u32>,
    },
}

impl<D: Metadata> Default for MemoryBlock<D> {
    fn default() -> Self { Self::new() }
}

impl<D: Metadata> MemoryBlock<D> {
    pub fn new() -> Self { Self::with_size(0) }
}

impl<D: Metadata, M: PointerMap> MemoryBlock<D, M> {
    pub fn with_size(size: u32) -> Self {
        Self {
            memory: M::with_size(size),
            ptr_info: FxHashMap::default(),
            allocations: Slab::new(),
            deallocated: FxHashSet::default(),
            vec_recycler: Recycler::new(),
            copies: Slab::new(),
        }
    }

    fn size(&self) -> Option<u32> { self.memory.size() }

    pub fn info(&self, ptr: Pointer) -> Option<&PointerInfo<D>> { self.ptr_info.get(&ptr) }

    pub fn update_meta(&mut self, ptr: Pointer, f: impl FnOnce(D) -> D) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }

        let info = self.ptr_info.get_mut(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let old_meta = info.meta;
        let meta = f(info.meta);
        if meta.does_invalidate(old_meta, &mut D::filter_all()) {
            return Err(Error::InvalidatesOldMeta(ptr))
        }
        info.meta = meta;
        if old_meta.does_invalidate(meta, &mut D::filter_all()) {
            let copy_id = info.copy_id;
            let range = info.range.clone();
            self.copies[copy_id as usize] -= 1;
            info.copy_id = self.copies.insert(1) as u32;
            let ptr_info = &self.ptr_info;
            self.memory.for_each(range.clone(), &mut self.vec_recycler, |byte| {
                let (pos, _, copy_block_end) = search(ptr, copy_id, byte, &range, ptr_info)?;
                let copy_block_end = copy_block_end();
                byte.swap(pos, copy_block_end);
                Ok(false)
            })?
        }

        OK
    }

    pub fn is_deallocated(&self, ptr: Pointer) -> bool { self.deallocated.contains(&ptr) }

    pub fn recylcer(&mut self) -> &mut Recycler<Vec<Pointer>> { &mut self.vec_recycler }

    pub fn allocate(&mut self, ptr: Pointer, range: Range<u32>) -> Result {
        if let Some(size) = self.size() {
            assert!(range.end <= size);
        }

        self.memory.for_each(range.clone(), &mut self.vec_recycler, |byte| {
            if byte.is_empty() {
                Ok(false)
            } else {
                Err(Error::AllocateRangeOccupied {
                    ptr,
                    range: range.clone(),
                })
            }
        })?;

        match self.ptr_info.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(PointerInfo {
                    meta: Metadata::alloc(),
                    range: range.clone(),
                    ptr_ty: PtrType::Exclusive,
                    owns_allocation: true,
                    alloc_id: u32::try_from(self.allocations.insert(self.vec_recycler.take())).unwrap(),
                    copy_id: self.copies.insert(1) as u32,
                });
            }
            Entry::Occupied(_) => {
                panic!("Could not allocate tracked pointer")
            }
        }

        let _ = self.memory.for_each(range, &mut self.vec_recycler, |byte| {
            byte.push(ptr);
            Ok::<_, Infallible>(false)
        });

        OK
    }

    pub fn reborrow_shared(&mut self, ptr: Pointer, source: Pointer, range: Range<u32>, meta: D) -> Result {
        let mut filter = D::filter_all();
        let filter = &mut filter;

        let source_range = self.reborrow_common(ptr, source, PtrType::Shared, range.clone(), meta, filter)?;

        let ptr_info = &self.ptr_info;
        let copy_id = ptr_info[&source].copy_id;
        self.memory.for_each(range, &mut self.vec_recycler, |byte| {
            let (pos, _, _) = search(source, copy_id, byte, &source_range, ptr_info)?;
            let pos = pos + 1;
            let offset = byte[pos..].iter().position(|ptr| {
                let info = &ptr_info[ptr];
                info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, filter)
            });
            if let Some(offset) = offset {
                byte.truncate(pos + offset);
            }
            byte.push(ptr);
            Ok(false)
        })
    }

    pub fn reborrow_exclusive(&mut self, ptr: Pointer, source: Pointer, range: Range<u32>, meta: D) -> Result {
        let mut filter = D::filter_all();
        let filter = &mut filter;

        let source_range = self.reborrow_common(ptr, source, PtrType::Exclusive, range.clone(), meta, filter)?;

        let ptr_info = &self.ptr_info;
        let copy_id = ptr_info[&source].copy_id;
        self.memory.for_each(range, &mut self.vec_recycler, |byte| {
            let (pos, _, _) = search(source, copy_id, byte, &source_range, ptr_info)?;
            byte.truncate(pos + 1);
            byte.push(ptr);
            Ok(false)
        })
    }

    fn reborrow_common(
        &mut self,
        ptr: Pointer,
        source: Pointer,
        ptr_ty: PtrType,
        range: Range<u32>,
        meta: D,
        filter: &mut D::Filter,
    ) -> Result<Range<u32>> {
        let source_info = self.ptr_info.get(&source).ok_or(Error::InvalidPtr(source))?;

        if !(source_info.range.start <= range.start && range.end <= source_info.range.end) {
            return Err(Error::ReborrowSubset {
                ptr,
                source,
                source_range: source_info.range.clone(),
            })
        }

        if meta.does_invalidate(source_info.meta, filter) {
            return Err(Error::ReborrowInvalidatesSource)
        }

        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }

        let source_range = source_info.range.clone();
        let alloc_id = source_info.alloc_id;

        self.allocations[alloc_id as usize].push(ptr);
        match self.ptr_info.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(PointerInfo {
                    range,
                    owns_allocation: false,
                    alloc_id,
                    ptr_ty,
                    meta,
                    copy_id: self.copies.insert(1) as u32,
                });
            }
            Entry::Occupied(_) => {
                panic!("Could not reborrow already-borrowed pointer")
            }
        }

        Ok(source_range)
    }

    pub fn copy(&mut self, ptr: Pointer, source: Pointer) -> Result {
        if self.deallocated.contains(&source) {
            return Err(Error::UseAfterFree(source))
        }

        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }

        let source_info = self.ptr_info.get(&source).ok_or(Error::InvalidPtr(source))?;

        if source_info.ptr_ty != PtrType::Shared {
            return Err(Error::NotShared(source))
        }

        let info = source_info.clone();
        self.allocations[info.alloc_id as usize].push(ptr);
        let info = match self.ptr_info.entry(ptr) {
            Entry::Vacant(entry) => entry.insert(info),
            Entry::Occupied(_) => panic!(""),
        };

        self.copies[info.copy_id as usize] += 1;

        let range = info.range.clone();
        let ptr_info = &self.ptr_info;
        let copy_id = ptr_info[&source].copy_id;
        self.memory.for_each(range.clone(), &mut self.vec_recycler, |byte| {
            let (pos, _, _) = search(source, copy_id, byte, &range, ptr_info)?;
            byte.insert(pos, ptr);
            Ok(false)
        })
    }

    pub fn deallocate(&mut self, ptr: Pointer) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }

        if !self.ptr_info.get(&ptr).ok_or(Error::InvalidPtr(ptr))?.owns_allocation {
            return Err(Error::DeallocateNonOwning(ptr))
        }

        self.assert_exclusive_inner(ptr, false)?;

        let info = self.ptr_info.remove(&ptr).ok_or(Error::InvalidPtr(ptr))?;

        let copies = &mut self.copies[info.copy_id as usize];
        *copies -= 1;
        if *copies == 0 {
            self.copies.remove(info.copy_id as usize);
        }

        self.deallocated.insert(ptr);
        let mut allocation = self.allocations.remove(info.alloc_id as usize);
        for ptr in allocation.drain(..) {
            if let Some(info) = self.ptr_info.remove(&ptr) {
                let copies = &mut self.copies[info.copy_id as usize];
                *copies -= 1;
                if *copies == 0 {
                    self.copies.remove(info.copy_id as usize);
                }
            }
            self.deallocated.insert(ptr);
        }
        self.vec_recycler.put(allocation);

        OK
    }

    pub fn mark_shared(&mut self, ptr: Pointer) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }
        let info = self.ptr_info.get_mut(&ptr).unwrap();
        info.ptr_ty = PtrType::Shared;
        self.assert_shared(ptr, D::filter_all())?;
        OK
    }

    pub fn mark_exclusive(&mut self, ptr: Pointer) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }
        let info = self.ptr_info.get_mut(&ptr).ok_or(Error::InvalidPtr(ptr))?;

        info.ptr_ty = PtrType::Exclusive;
        self.assert_exclusive(ptr)?;
        OK
    }

    pub fn assert_shared(&mut self, ptr: Pointer, mut filter: D::Filter) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }
        let filter = &mut filter;
        let ptr_info = &self.ptr_info;
        let info = ptr_info.get(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let meta = info.meta;

        let copy_id = ptr_info[&ptr].copy_id;
        self.memory
            .for_each(info.range.clone(), &mut self.vec_recycler, |byte| {
                let (pos, _, _) = search(ptr, copy_id, byte, &info.range, ptr_info)?;
                let pos = pos + 1;
                let offset = byte[pos..].iter().position(|packed_ptr| {
                    let info = &ptr_info[packed_ptr];
                    info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, filter)
                });
                if let Some(offset) = offset {
                    byte.truncate(pos + offset);
                }
                Ok(false)
            })
    }

    pub fn assert_exclusive(&mut self, ptr: Pointer) -> Result { self.assert_exclusive_inner(ptr, true) }

    fn assert_exclusive_inner(&mut self, ptr: Pointer, keep: bool) -> Result {
        if self.deallocated.contains(&ptr) {
            return Err(Error::UseAfterFree(ptr))
        }

        let info = self.ptr_info.get(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        if info.ptr_ty != PtrType::Exclusive {
            return Err(Error::NotExclusive(ptr))
        }

        let ptr_info = &self.ptr_info;
        let copy_id = ptr_info[&ptr].copy_id;
        self.memory
            .for_each(info.range.clone(), &mut self.vec_recycler, |byte| {
                let (pos, copy_block_start, _) = search(ptr, copy_id, byte, &info.range, ptr_info)?;
                let copy_block_start = copy_block_start();
                byte.swap(pos, copy_block_start);
                byte.truncate(pos + usize::from(keep));
                Ok(!keep && byte.is_empty())
            })
    }
}

fn search<'a, D: Metadata + 'a>(
    ptr: Pointer,
    copy_id: u32,
    byte: &'a [Pointer],
    range: &Range<u32>,
    ptr_info: &'a FxHashMap<Pointer, PointerInfo<D>>,
) -> Result<(usize, impl 'a + FnOnce() -> usize, impl 'a + FnOnce() -> usize)> {
    let this_pos = byte
        .iter()
        .rposition(|byte| *byte == ptr)
        .ok_or(Error::InvalidForRange {
            ptr,
            range: range.clone(),
        })?;
    Ok((
        this_pos,
        move || {
            byte[..=this_pos]
                .iter()
                .rposition(|ptr| ptr_info[ptr].copy_id != copy_id)
                .map_or(0, |pos| pos + 1)
        },
        move || {
            byte[this_pos..]
                .iter()
                .rposition(|ptr| ptr_info[ptr].copy_id != copy_id)
                .unwrap_or(byte.len() - 1)
        },
    ))
}
