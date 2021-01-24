use std::{
    collections::{btree_map::Entry as BEntry, hash_map::Entry, BTreeMap, HashMap},
    convert::{Infallible, TryFrom},
    fmt,
    hash::BuildHasher,
    ops::Range,
};

use fxhash::FxHashMap;
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
                    let mut vec = recycler.take();
                    let should_remove = f(&mut vec)?;
                    if !should_remove {
                        vacant.insert(vec);
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
                    let mut vec = recycler.take();
                    let should_remove = f(&mut vec)?;
                    if !should_remove {
                        vacant.insert(vec);
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

pub struct MemoryBlock<M = BTreeMap<u32, Vec<Pointer>>> {
    memory: M,
    ptr_info: FxHashMap<Pointer, PointerInfo>,
    allocations: Slab<Vec<Pointer>>,
    vec_recycler: Recycler<Vec<Pointer>>,
}

#[non_exhaustive]
#[derive(Debug)]
pub struct PointerInfo {
    pub meta: Meta,
    pub owns_allocation: bool,
    pub alloc_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtrType {
    Shared,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Meta {
    pub range: Range<u32>,
    pub ptr_ty: PtrType,
    pub read: bool,
    pub write: bool,
}

const OK: Result = Ok(());
pub type Result<T = (), E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    InvalidPtr(Pointer),
    NotExclusive(Pointer),
    WriteInvalid(Pointer),
    ReadInvalid(Pointer),
    ExclusiveWriteInvalid(Pointer),
    DeallocateNonOwning(Pointer),
    ReborrowReadPermission,
    ReborrowWritePermission,
    AllocateRangeOccupied {
        ptr: Pointer,
        range: Range<u32>,
    },
    InvalidForRange {
        ptr: Pointer,
        range: Range<u32>,
    },
    SharedWriteInvalid {
        ptr: Pointer,
        range: Range<u32>,
    },
    ReborrowSubset {
        ptr: Pointer,
        source: Pointer,
        source_range: Range<u32>,
    },
    InvalidateNoWriteFailed {
        ptr: Pointer,
        range: Range<u32>,
    },
}

impl Default for MemoryBlock {
    fn default() -> Self { Self::new() }
}

impl MemoryBlock {
    pub fn new() -> Self {
        Self {
            memory: <_>::with_size(0),
            ptr_info: FxHashMap::default(),
            allocations: Slab::new(),
            vec_recycler: Recycler::new(),
        }
    }
}
impl<M: PointerMap> MemoryBlock<M> {
    pub fn with_size(size: u32) -> Self {
        Self {
            memory: M::with_size(size),
            ptr_info: FxHashMap::default(),
            allocations: Slab::new(),
            vec_recycler: Recycler::new(),
        }
    }

    fn size(&self) -> Option<u32> { self.memory.size() }

    pub fn info(&self, ptr: Pointer) -> &PointerInfo { &self.ptr_info[&ptr] }

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
                    meta: Meta {
                        range: range.clone(),
                        ptr_ty: PtrType::Exclusive,
                        read: true,
                        write: true,
                    },
                    owns_allocation: true,
                    alloc_id: u32::try_from(self.allocations.insert(self.vec_recycler.take())).unwrap(),
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

    pub fn reborrow(&mut self, ptr: Pointer, source: Pointer, meta: Meta) -> Result {
        let source_info = &self.ptr_info[&source];

        if !(source_info.meta.range.start <= meta.range.start && meta.range.end <= source_info.meta.range.end) {
            return Err(Error::ReborrowSubset {
                ptr,
                source,
                source_range: source_info.meta.range.clone(),
            })
        }

        if u8::from(source_info.meta.read) < u8::from(meta.read) {
            return Err(Error::ReborrowReadPermission)
        }

        if u8::from(source_info.meta.write) < u8::from(meta.write) {
            return Err(Error::ReborrowWritePermission)
        }

        let source_range = source_info.meta.range.clone();
        let alloc_id = source_info.alloc_id;

        match self.ptr_info.entry(ptr) {
            Entry::Vacant(entry) => {
                entry.insert(PointerInfo {
                    meta: meta.clone(),
                    owns_allocation: false,
                    alloc_id,
                });
            }
            Entry::Occupied(_) => {
                panic!("Could not already-borrowed pointer")
            }
        }

        let ptr_info = &self.ptr_info;
        match meta.ptr_ty {
            PtrType::Shared => self
                .memory
                .for_each(meta.range.clone(), &mut self.vec_recycler, |byte| {
                    let pos = 1 + byte
                        .iter()
                        .rposition(|ptr| *ptr == source)
                        .ok_or(Error::ReborrowSubset {
                            ptr,
                            source,
                            source_range: source_range.clone(),
                        })?;

                    let offset = byte[pos..].iter().position(|ptr| {
                        let byte_meta = &ptr_info[ptr].meta;
                        u8::from(byte_meta.write) < u8::from(meta.write)
                            || u8::from(byte_meta.read) < u8::from(meta.read)
                            || byte_meta.ptr_ty == PtrType::Exclusive
                    });
                    if let Some(offset) = offset {
                        byte.truncate(pos + offset);
                    }
                    byte.push(ptr);
                    Ok(false)
                }),
            PtrType::Exclusive => self.memory.for_each(meta.range, &mut self.vec_recycler, |byte| {
                let pos = 1 + byte
                    .iter()
                    .rposition(|ptr| *ptr == source)
                    .ok_or(Error::ReborrowSubset {
                        ptr,
                        source,
                        source_range: source_range.clone(),
                    })?;
                byte.truncate(pos);
                byte.push(ptr);
                Ok(false)
            }),
        }
    }

    pub fn deallocate(&mut self, ptr: Pointer) -> Result {
        if !self.ptr_info[&ptr].owns_allocation {
            return Err(Error::DeallocateNonOwning(ptr))
        }

        self.assert_exclusive_inner(ptr, false)?;

        let info = self.ptr_info.remove(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let mut allocation = self.allocations.remove(info.alloc_id as usize);
        for ptr in allocation.drain(..) {
            self.ptr_info.remove(&ptr);
        }
        self.vec_recycler.put(allocation);

        OK
    }

    pub fn downgrade(&mut self, ptr: Pointer) -> Result {
        let info = self.ptr_info.get_mut(&ptr).unwrap();

        if info.meta.ptr_ty != PtrType::Exclusive {
            return Err(Error::NotExclusive(ptr))
        }

        info.meta.ptr_ty = PtrType::Shared;
        OK
    }

    pub fn assert_exclusive(&mut self, ptr: Pointer) -> Result { self.assert_exclusive_inner(ptr, true) }

    fn assert_exclusive_inner(&mut self, ptr: Pointer, keep: bool) -> Result {
        let info = &self.ptr_info[&ptr];

        if info.meta.ptr_ty != PtrType::Exclusive {
            return Err(Error::NotExclusive(ptr))
        }

        self.memory
            .for_each(info.meta.range.clone(), &mut self.vec_recycler, |byte| {
                let pos = byte
                    .iter()
                    .rposition(|packed_ptr| *packed_ptr == ptr)
                    .ok_or(Error::InvalidForRange {
                        ptr,
                        range: info.meta.range.clone(),
                    })?;
                byte.truncate(pos + usize::from(keep));
                Ok(!keep && byte.is_empty())
            })
    }

    pub fn assert_shared(&mut self, ptr: Pointer) -> Result { self.assert_shared_inner(ptr, true) }

    pub fn assert_shared_inner(&mut self, ptr: Pointer, is_read: bool) -> Result {
        let ptr_info = &self.ptr_info;
        let info = &ptr_info[&ptr];
        let ptr_meta = info.meta.clone();

        self.memory
            .for_each(info.meta.range.clone(), &mut self.vec_recycler, |byte| {
                let pos = byte
                    .iter()
                    .rposition(|packed_ptr| *packed_ptr == ptr)
                    .ok_or(Error::InvalidForRange {
                        ptr,
                        range: info.meta.range.clone(),
                    })?;
                let offset = byte.iter().skip(pos + 1).position(|packed_ptr| {
                    let meta = &ptr_info[packed_ptr].meta;
                    meta.ptr_ty == PtrType::Exclusive
                        || u8::from(meta.read) < u8::from(ptr_meta.read)
                        || u8::from(meta.write || is_read) < u8::from(ptr_meta.write)
                });
                if let Some(offset) = offset {
                    byte.truncate(pos + 1 + offset);
                }
                Ok(false)
            })
    }

    pub fn read(&mut self, ptr: Pointer) -> Result {
        let info = &self.ptr_info[&ptr];
        if !info.meta.read {
            return Err(Error::ReadInvalid(ptr))
        }

        self.assert_shared(ptr)
    }

    pub fn shared_write(&mut self, ptr: Pointer) -> Result {
        let info = &self.ptr_info[&ptr];
        if !info.meta.write {
            return Err(Error::WriteInvalid(ptr))
        }

        self.assert_shared_inner(ptr, false)
    }

    pub fn exclusive_write(&mut self, ptr: Pointer) -> Result {
        let info = &self.ptr_info[&ptr];

        if !info.meta.write {
            return Err(Error::WriteInvalid(ptr))
        }

        self.assert_exclusive(ptr)
    }
}

impl fmt::Debug for MemoryBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBlock {{ memory: {:?}, ptr_info: {:#?} }}",
            self.memory, self.ptr_info,
        )
    }
}
