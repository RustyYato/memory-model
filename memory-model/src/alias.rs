use std::{
    collections::{btree_map::Entry as BEntry, hash_map::Entry, BTreeMap, HashMap},
    convert::TryFrom,
    fmt,
    hash::BuildHasher,
    ops::Range,
};

use fxhash::{FxHashMap, FxHashSet};
use slab::Slab;

use crate::{
    recycle::{Recycle, Recycler},
    Pointer,
};

macro_rules! check_dealloc {
    ($self:ident, $ptr:expr) => {
        if $self.is_deallocated($ptr) {
            return Err(Error::UseAfterFree($ptr))
        }
    };
}

macro_rules! check_range {
    ($self:expr, $ptr:expr, $range:expr) => {{
        let ptr = $ptr;
        let (id, info) = $self.store.get(ptr)?;
        let range = $range.unwrap_or_else(|| info.range.clone());
        $self.memory.for_each(range.clone(), |Stack(byte)| {
            search(ptr, id, byte, &range).map(|_| false)
        })?
    }};
    ($self:expr, $ptr:expr) => {
        check_range!($self, $ptr, None)
    };
}

pub trait PointerMap {
    fn with_size(size: u32) -> Self;

    fn size(&self) -> Option<u32>;

    fn for_each<E, F: FnMut(&mut Stack) -> Result<bool, E>>(&mut self, range: Range<u32>, f: F) -> Result<(), E>;
}

pub trait Metadata: Copy + Eq {
    type Filter;

    fn alloc() -> Self;

    fn filter_all() -> Self::Filter;

    fn does_invalidate(self, other: Self, filter: &mut Self::Filter) -> bool;
}

#[derive(Debug)]
pub struct FixedSizePointerMap {
    inner: Vec<Stack>,
}

#[derive(Default, Debug)]
pub struct HashPointerMap<B> {
    inner: HashMap<u32, Stack, B>,
    recycler: Recycler<Stack>,
}

#[derive(Default, Debug)]
pub struct BTreePointerMap {
    inner: BTreeMap<u32, Stack>,
    recycler: Recycler<Stack>,
}

impl Metadata for () {
    type Filter = ();

    fn alloc() -> Self {}

    fn filter_all() -> Self::Filter {}

    fn does_invalidate(self, (): Self, (): &mut Self::Filter) -> bool { false }
}

impl PointerMap for FixedSizePointerMap {
    fn with_size(size: u32) -> Self {
        Self {
            inner: (0..size).map(|_| Stack::default()).collect(),
        }
    }

    fn size(&self) -> Option<u32> { Some(self.inner.len() as u32) }

    fn for_each<E, F: FnMut(&mut Stack) -> Result<bool, E>>(&mut self, range: Range<u32>, mut f: F) -> Result<(), E> {
        let range = range.start as usize..range.end as usize;
        for byte in &mut self.inner[range] {
            f(byte)?;
        }
        Ok(())
    }
}

impl<B: Default + BuildHasher> PointerMap for HashPointerMap<B> {
    fn with_size(_: u32) -> Self { Self::default() }

    fn size(&self) -> Option<u32> { None }

    fn for_each<E, F: FnMut(&mut Stack) -> Result<bool, E>>(&mut self, range: Range<u32>, mut f: F) -> Result<(), E> {
        for i in range {
            match self.inner.entry(i) {
                Entry::Vacant(vacant) => {
                    let mut vec = self.recycler.lease();
                    let should_remove = f(&mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                Entry::Occupied(mut entry) => {
                    let should_remove = f(entry.get_mut())?;
                    if should_remove {
                        self.recycler.put(entry.remove());
                    }
                }
            }
        }
        Ok(())
    }
}

impl PointerMap for BTreePointerMap {
    fn with_size(_: u32) -> Self { Self::default() }

    fn size(&self) -> Option<u32> { None }

    fn for_each<E, F: FnMut(&mut Stack) -> Result<bool, E>>(&mut self, range: Range<u32>, mut f: F) -> Result<(), E> {
        for i in range {
            match self.inner.entry(i) {
                BEntry::Vacant(vacant) => {
                    let mut vec = self.recycler.lease();
                    let should_remove = f(&mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                BEntry::Occupied(mut entry) => {
                    let should_remove = f(entry.get_mut())?;
                    if should_remove {
                        self.recycler.put(entry.remove());
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct Stack(Vec<u32>);

impl Recycle for Stack {
    type Size = usize;

    fn empty(mut self) -> Self {
        self.0.clear();
        self
    }

    fn size(&self) -> Self::Size { self.0.capacity() }
}

#[derive(Debug)]
pub struct MemoryBlock<D, M = BTreePointerMap> {
    memory: M,
    deallocated: FxHashSet<Pointer>,
    allocations: Slab<FxHashSet<Pointer>>,
    store: PointerStore<D>,
}

struct PointerStore<D> {
    ptr_info: Slab<PointerInfo<D>>,
    counters: FxHashMap<Pointer, u32>,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PointerInfo<D> {
    copies: u32,
    pub owns_allocation: bool,
    pub alloc_id: u32,
    pub range: Range<u32>,
    pub ptr_ty: PtrType,
    pub meta: D,
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
    ReborrowInvalidatesSource {
        ptr: Pointer,
        source: Pointer,
    },
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

impl<D: Copy> PointerStore<D> {
    fn new() -> Self {
        Self {
            counters: Default::default(),
            ptr_info: Default::default(),
        }
    }

    fn alloc(&mut self, ptr: Pointer, info: PointerInfo<D>) -> u32 {
        let id = u32::try_from(self.ptr_info.insert(info)).expect("Tried to create too many pointers");
        self.counters.insert(ptr, id);
        id
    }

    fn dealloc(&mut self, ptr: Pointer) -> Result<PointerInfo<D>> {
        let id = self.counters.remove(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let info = &mut self.ptr_info[id as usize];
        info.copies -= 1;
        Ok(self.ptr_info.remove(id as usize))
    }

    fn drop(&mut self, ptr: Pointer) -> Result<()> {
        let id = self.counters.remove(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let info = self.ptr_info.get_mut(id as usize).ok_or(Error::InvalidPtr(ptr))?;
        info.copies -= 1;
        if info.copies == 0 {
            self.ptr_info.remove(id as usize);
        }
        Ok(())
    }

    fn make_exclusive(&mut self, ptr: Pointer) -> Result<(u32, u32, &mut PointerInfo<D>)> {
        let id = self.counters.get_mut(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let info = &mut self.ptr_info[*id as usize];
        let old_id = *id;
        if info.copies != 1 {
            info.copies -= 1;
            let mut info = info.clone();
            info.copies = 1;
            *id = u32::try_from(self.ptr_info.insert(info)).expect("Tried to create too many pointers");
        }
        Ok((old_id, *id, &mut self.ptr_info[*id as usize]))
    }

    fn get(&self, ptr: Pointer) -> Result<(u32, &PointerInfo<D>)> {
        let id = *self.counters.get(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        Ok((id, &self.ptr_info[id as usize]))
    }

    fn get_mut(&mut self, ptr: Pointer) -> Result<(u32, &mut PointerInfo<D>)> {
        let id = *self.counters.get(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        Ok((id, &mut self.ptr_info[id as usize]))
    }
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
            store: PointerStore::new(),
            allocations: Slab::new(),
            deallocated: FxHashSet::default(),
        }
    }

    pub fn info(&self, ptr: Pointer) -> Result<&PointerInfo<D>> { self.store.get(ptr).map(|(_, info)| info) }

    pub fn is_deallocated(&self, ptr: Pointer) -> bool { self.deallocated.contains(&ptr) }

    pub fn allocate(&mut self, ptr: Pointer, range: Range<u32>) -> Result {
        check_dealloc!(self, ptr);

        if self.store.counters.contains_key(&ptr) {
            panic!("Could not allocate already tracked pointer")
        }

        self.memory.for_each(range.clone(), |Stack(byte)| {
            if byte.is_empty() {
                Ok(false)
            } else {
                Err(Error::AllocateRangeOccupied {
                    ptr,
                    range: range.clone(),
                })
            }
        })?;

        let alloc_id =
            u32::try_from(self.allocations.insert(FxHashSet::default())).expect("Tried to create too many allocations");

        let id = self.store.alloc(ptr, PointerInfo {
            alloc_id,
            copies: 1,
            meta: D::alloc(),
            ptr_ty: PtrType::Exclusive,
            owns_allocation: true,
            range: range.clone(),
        });

        self.memory.for_each(range, |Stack(byte)| {
            byte.push(id);
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
    ) -> Result<(u32, u32, Range<u32>)> {
        check_dealloc!(self, source);
        check_dealloc!(self, ptr);

        if self.store.counters.contains_key(&ptr) {
            panic!("Could not borrow already tracked pointer")
        }

        check_range!(self, source, Some(range.clone()));

        let (source_id, source_info) = self.store.get(source)?;

        if !(source_info.range.start <= range.start && range.end <= source_info.range.end) {
            return Err(Error::ReborrowSubset {
                ptr,
                source,
                source_range: source_info.range.clone(),
            })
        }

        if meta.does_invalidate(source_info.meta, &mut D::filter_all()) {
            return Err(Error::ReborrowInvalidatesSource { ptr, source })
        }

        let source_range = source_info.range.clone();
        let alloc_id = source_info.alloc_id;

        self.allocations[alloc_id as usize].insert(ptr);
        let id = self.store.alloc(ptr, PointerInfo {
            owns_allocation: false,
            copies: 1,
            alloc_id,
            ptr_ty,
            range,
            meta,
        });

        Ok((id, source_id, source_range))
    }

    pub fn reborrow_exclusive(&mut self, ptr: Pointer, source: Pointer, range: Range<u32>, meta: D) -> Result {
        let (id, source_id, source_range) =
            self.reborrow_common(ptr, source, PtrType::Exclusive, range.clone(), meta)?;

        self.memory.for_each(range, |Stack(byte)| {
            let pos = 1 + search(source, source_id, byte, &source_range)?;
            byte.truncate(pos);
            byte.push(id);
            Ok(false)
        })
    }

    pub fn reborrow_shared(&mut self, ptr: Pointer, source: Pointer, range: Range<u32>, meta: D) -> Result {
        let (id, source_id, source_range) = self.reborrow_common(ptr, source, PtrType::Shared, range.clone(), meta)?;
        let ptr_info = &self.store.ptr_info;
        let mut filter = D::filter_all();

        self.memory.for_each(range, |Stack(byte)| {
            let pos = 1 + search(source, source_id, byte, &source_range).unwrap();

            let offset = byte[pos..].iter().position(|&id| {
                let info = &ptr_info[id as usize];
                info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, &mut filter)
            });

            if let Some(offset) = offset {
                byte.truncate(pos + offset);
            }

            byte.push(id);
            Ok(false)
        })
    }

    pub fn update_meta(&mut self, ptr: Pointer, f: impl FnOnce(D) -> D) -> Result {
        check_dealloc!(self, ptr);

        let (id, info) = self.store.get(ptr)?;
        let old_meta = info.meta;
        let meta = f(info.meta);

        if meta.does_invalidate(old_meta, &mut D::filter_all()) {
            return Err(Error::InvalidatesOldMeta(ptr))
        }

        if info.copies != 1 && old_meta.does_invalidate(meta, &mut D::filter_all()) {
            check_range!(self, ptr, Some(info.range.clone()));

            let (old_id, id, info) = self.store.make_exclusive(ptr).unwrap();
            let range = info.range.clone();

            self.memory.for_each(range.clone(), |Stack(byte)| {
                let pos = search(ptr, old_id, byte, &range).unwrap();
                byte.insert(pos + 1, id);
                Ok(false)
            })?
        }

        let info = &mut self.store.ptr_info[id as usize];
        info.meta = meta;

        OK
    }

    pub fn copy(&mut self, ptr: Pointer, source: Pointer) -> Result {
        check_dealloc!(self, source);
        check_dealloc!(self, ptr);

        let (id, info) = self.store.get_mut(source)?;

        if info.ptr_ty != PtrType::Shared {
            return Err(Error::NotShared(source))
        }

        info.copies += 1;
        self.allocations[info.alloc_id as usize].insert(ptr);
        self.store.counters.insert(ptr, id);

        OK
    }

    pub fn deallocate(&mut self, ptr: Pointer) -> Result<FxHashSet<Pointer>> {
        if !self.store.get(ptr)?.1.owns_allocation {
            return Err(Error::DeallocateNonOwning(ptr))
        }

        let info = self.store.dealloc(ptr)?;
        self.deallocated.insert(ptr);
        let mut allocation = self.allocations.remove(info.alloc_id as usize);

        for &ptr in allocation.iter() {
            let _ = self.store.drop(ptr);
            self.deallocated.insert(ptr);
        }

        allocation.insert(ptr);

        Ok(allocation)
    }

    pub fn mark_exclusive(&mut self, ptr: Pointer) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let (old_id, id, info) = self.store.make_exclusive(ptr)?;
        info.ptr_ty = PtrType::Exclusive;
        let info = &self.store.ptr_info[id as usize];
        let range = &info.range;

        self.memory.for_each(range.clone(), |Stack(byte)| {
            let pos = search(ptr, old_id, byte, range).unwrap();
            byte.truncate(pos);
            byte.push(id);
            Ok(false)
        })
    }

    pub fn mark_shared(&mut self, ptr: Pointer) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let (id, info) = self.store.get(ptr)?;

        self.memory.for_each(info.range.clone(), |Stack(byte)| {
            search(ptr, id, byte, &info.range).unwrap();
            Ok(false)
        })?;

        let (_, info) = self.store.get_mut(ptr).unwrap();
        info.ptr_ty = PtrType::Shared;
        self.assert_shared(ptr, D::filter_all())?;
        OK
    }

    pub fn assert_exclusive(&mut self, ptr: Pointer) -> Result { self.assert_exclusive_inner(ptr, true) }

    pub fn assert_shared(&mut self, ptr: Pointer, mut filter: D::Filter) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let filter = &mut filter;
        let (id, info) = self.store.get(ptr)?;
        let ptr_info = &self.store.ptr_info;
        let meta = info.meta;

        self.memory.for_each(info.range.clone(), |Stack(byte)| {
            let pos = 1 + search(ptr, id, byte, &info.range).unwrap();

            let offset = byte[pos..].iter().position(|&id| {
                let info = &ptr_info[id as usize];
                info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, filter)
            });

            if let Some(offset) = offset {
                byte.truncate(pos + offset);
            }

            Ok(false)
        })
    }

    fn assert_exclusive_inner(&mut self, ptr: Pointer, keep: bool) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let (id, info) = self.store.get(ptr)?;

        if info.ptr_ty != PtrType::Exclusive {
            return Err(Error::NotExclusive(ptr))
        }

        self.memory.for_each(info.range.clone(), |Stack(byte)| {
            let pos = search(ptr, id, byte, &info.range).unwrap();
            byte.truncate(pos + usize::from(keep));
            Ok(!keep && byte.is_empty())
        })
    }
}

fn search(ptr: Pointer, ptr_id: u32, byte: &[u32], range: &Range<u32>) -> Result<usize> {
    byte.iter()
        .rposition(|byte| *byte == ptr_id)
        .ok_or(Error::InvalidForRange {
            ptr,
            range: range.clone(),
        })
}

impl<D: fmt::Debug> fmt::Debug for PointerStore<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_map();

        #[derive(Debug)]
        struct PointerData<A, B> {
            copy_id: A,
            info: B,
        }

        for (ptr, &id) in &self.counters {
            f.entry(&ptr, &PointerData {
                copy_id: id,
                info: &self.ptr_info[id as usize],
            });
        }
        f.finish()
    }
}
