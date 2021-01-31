use std::{
    collections::{btree_map::Entry as BEntry, hash_map::Entry, BTreeMap, HashMap},
    convert::TryFrom,
    fmt,
    hash::BuildHasher,
    ops::Range,
};

use fxhash::{FxHashMap, FxHashSet};
use slab::Slab;

use sync_wrapper::SyncWrapper;

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
        $self.memory.for_each(range.clone(), |_, Stack(byte)| {
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

    fn for_each<E, F: FnMut(u32, &mut Stack) -> Result<bool, E>>(&mut self, range: Range<u32>, f: F) -> Result<(), E>;
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

    fn for_each<E, F: FnMut(u32, &mut Stack) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        mut f: F,
    ) -> Result<(), E> {
        for (byte, i) in &mut self.inner[range.start as usize..range.end as usize]
            .iter_mut()
            .zip(range)
        {
            f(i, byte)?;
        }
        Ok(())
    }
}

impl<B: Default + BuildHasher> PointerMap for HashPointerMap<B> {
    fn with_size(_: u32) -> Self { Self::default() }

    fn size(&self) -> Option<u32> { None }

    fn for_each<E, F: FnMut(u32, &mut Stack) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        mut f: F,
    ) -> Result<(), E> {
        for i in range {
            match self.inner.entry(i) {
                Entry::Vacant(vacant) => {
                    let mut vec = self.recycler.lease();
                    let should_remove = f(i, &mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                Entry::Occupied(mut entry) => {
                    let should_remove = f(i, entry.get_mut())?;
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

    fn for_each<E, F: FnMut(u32, &mut Stack) -> Result<bool, E>>(
        &mut self,
        range: Range<u32>,
        mut f: F,
    ) -> Result<(), E> {
        for i in range {
            match self.inner.entry(i) {
                BEntry::Vacant(vacant) => {
                    let mut vec = self.recycler.lease();
                    let should_remove = f(i, &mut *vec)?;
                    if !should_remove {
                        vacant.insert(vec.take());
                    }
                }
                BEntry::Occupied(mut entry) => {
                    let should_remove = f(i, entry.get_mut())?;
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
pub struct MemoryBlock<'env, D, M = BTreePointerMap> {
    memory: M,
    deallocated: FxHashSet<Pointer>,
    allocations: Slab<FxHashSet<Pointer>>,
    store: PointerStore<D>,
    pub trackers: TrackerGroup<'env, D>,
}

#[derive(Debug)]
pub struct TrackerGroup<'env, D> {
    trackers: FxHashMap<Pointer, Vec<Tracker<'env, D>>>,
}

pub struct Tracker<'env, D> {
    pub tag: &'env str,
    #[allow(clippy::type_complexity)]
    event_handler: SyncWrapper<Box<dyn 'env + FnMut(&'env str, Pointer, RawEvent<'_, D>) + Send>>,
}

enum RawEvent<'a, D> {
    Event {
        event: Event,
        ptr: Pointer,
    },
    PoppedOff {
        event: Event,
        drain: &'a [u32],
        info: &'a Slab<PointerInfo<D>>,
    },
}

impl<D> Copy for RawEvent<'_, D> {}
impl<D> Clone for RawEvent<'_, D> {
    fn clone(&self) -> Self { *self }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum Event {
    ReborrowExlusiveInvalidated { pos: u32 },
    MarkExlusiveInvalidated { pos: u32 },
    ReborrowSharedInvalidated { pos: u32 },
    AssertExlusiveInvalidated { pos: u32 },
    AssertSharedInvalidated { pos: u32 },
    DropInvalidated { pos: u32 },
    ReborrowExlusive,
    MarkExlusive,
    ReborrowShared,
    MarkShared,
    AssertExlusive,
    AssertShared,
    Drop,
}

struct PointerStore<D> {
    ptr_info: Slab<PointerInfo<D>>,
    counters: FxHashMap<Pointer, u32>,
}

#[non_exhaustive]
#[derive(Debug)]
pub struct PointerInfo<D> {
    members: FxHashSet<Pointer>,
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

fn vec_retain<T, F: FnMut(&mut T) -> bool>(vec: &mut Vec<T>, mut f: F) {
    let len = vec.len();
    let mut del = 0;
    {
        let v = &mut **vec;

        for i in 0..len {
            if !f(&mut v[i]) {
                del += 1;
            } else if del > 0 {
                v.swap(i - del, i);
            }
        }
    }
    if del > 0 {
        vec.truncate(len - del);
    }
}

impl<'env, D> TrackerGroup<'env, D> {
    pub fn iter(&self) -> impl '_ + Iterator<Item = (Pointer, &'env str)> {
        self.trackers
            .iter()
            .flat_map(|(&ptr, trackers)| trackers.iter().map(move |tracker| (ptr, tracker.tag)))
    }

    pub fn register<F: 'env + FnMut(&'env str, Pointer, Event) + Send>(
        &mut self,
        tag: &'env str,
        ptr: Pointer,
        event_handler: F,
    ) {
        self.trackers
            .entry(ptr)
            .or_default()
            .push(Tracker::new(tag, event_handler));
    }

    pub fn retain<F: FnMut(&mut Tracker<'env, D>) -> bool>(&mut self, ptr: Pointer, f: F) {
        if let Entry::Occupied(mut trackers) = self.trackers.entry(ptr) {
            let t = trackers.get_mut();
            vec_retain(t, f);
            if t.is_empty() {
                trackers.remove();
            }
        }
    }

    pub fn retain_all<F: FnMut(Pointer, &mut Tracker<'env, D>) -> bool>(&mut self, mut f: F) {
        self.trackers.retain(|&k, v| {
            vec_retain(v, |v| f(k, v));
            !v.is_empty()
        })
    }

    pub fn get(&self, ptr: Pointer) -> &[Tracker<'env, D>] { self.trackers.get(&ptr).map_or(&[], Vec::as_slice) }

    fn is_empty(&self) -> bool { self.trackers.is_empty() }

    fn call(&mut self, event: RawEvent<'_, D>) {
        match event {
            RawEvent::Event { ptr, .. } => {
                if let Some(trackers) = self.trackers.get_mut(&ptr) {
                    for tracker in trackers {
                        tracker.call(ptr, event)
                    }
                }
            }
            RawEvent::PoppedOff { .. } => {
                for (&ptr, trackers) in self.trackers.iter_mut() {
                    for tracker in trackers {
                        tracker.call(ptr, event)
                    }
                }
            }
        }
    }
}

impl<'env, D> Tracker<'env, D> {
    pub fn new<F: 'env + FnMut(&'env str, Pointer, Event) + Send>(tag: &'env str, mut event_handler: F) -> Self {
        Self {
            tag,
            event_handler: SyncWrapper::new(Box::new(move |tag: &'env str, ptr, raw_event: RawEvent<'_, D>| {
                match raw_event {
                    RawEvent::Event { event, ptr: _ } => event_handler(tag, ptr, event),
                    RawEvent::PoppedOff { drain, event, info } => {
                        for &id in drain {
                            for &member in info[id as usize].members.iter() {
                                if ptr == member {
                                    event_handler(tag, ptr, event);
                                }
                            }
                        }
                    }
                }
            }) as _),
        }
    }

    fn call(&mut self, ptr: Pointer, event: RawEvent<'_, D>) { self.event_handler.get_mut()(self.tag, ptr, event) }
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
        info.members.insert(ptr);
        Ok(self.ptr_info.remove(id as usize))
    }

    fn drop(&mut self, ptr: Pointer) -> Result<()> {
        let id = self.counters.remove(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let info = self.ptr_info.get_mut(id as usize).ok_or(Error::InvalidPtr(ptr))?;
        info.members.remove(&ptr);
        if info.members.is_empty() {
            self.ptr_info.remove(id as usize);
        }
        Ok(())
    }

    fn make_exclusive(&mut self, ptr: Pointer) -> Result<(u32, u32, &mut PointerInfo<D>)> {
        let id = self.counters.get_mut(&ptr).ok_or(Error::InvalidPtr(ptr))?;
        let info = &mut self.ptr_info[*id as usize];
        let old_id = *id;
        if info.members.len() != 1 {
            info.members.remove(&ptr);
            let info = PointerInfo {
                members: std::iter::once(ptr).collect(),
                alloc_id: info.alloc_id,
                meta: info.meta,
                owns_allocation: info.owns_allocation,
                ptr_ty: info.ptr_ty,
                range: info.range.clone(),
            };
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

impl<'env, D: Metadata> Default for MemoryBlock<'env, D> {
    fn default() -> Self { Self::new() }
}

impl<'env, D: Metadata> MemoryBlock<'env, D> {
    pub fn new() -> Self { Self::with_size(0) }
}

impl<'env, D: Metadata, M: PointerMap> MemoryBlock<'env, D, M> {
    pub fn with_size(size: u32) -> Self {
        Self {
            memory: M::with_size(size),
            store: PointerStore::new(),
            allocations: Slab::new(),
            deallocated: FxHashSet::default(),
            trackers: TrackerGroup {
                trackers: Default::default(),
            },
        }
    }

    pub fn info(&self, ptr: Pointer) -> Result<&PointerInfo<D>> { self.store.get(ptr).map(|(_, info)| info) }

    pub fn is_deallocated(&self, ptr: Pointer) -> bool { self.deallocated.contains(&ptr) }

    pub fn allocate(&mut self, ptr: Pointer, range: Range<u32>) -> Result {
        check_dealloc!(self, ptr);

        if self.store.counters.contains_key(&ptr) {
            panic!("Could not allocate already tracked pointer")
        }

        self.memory.for_each(range.clone(), |_, Stack(byte)| {
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
            members: std::iter::once(ptr).collect(),
            meta: D::alloc(),
            ptr_ty: PtrType::Exclusive,
            owns_allocation: true,
            range: range.clone(),
        });

        self.memory.for_each(range, |_, Stack(byte)| {
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
            members: std::iter::once(ptr).collect(),
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

        let store = &self.store;
        let trackers = &mut self.trackers;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::ReborrowExlusive,
                ptr: source,
            });

            self.memory.for_each(range.clone(), |byte_pos, Stack(byte)| {
                let pos = 1 + search(source, source_id, byte, &source_range)?;
                trackers.call(RawEvent::PoppedOff {
                    event: Event::ReborrowExlusiveInvalidated { pos: byte_pos },

                    drain: &byte[pos..],
                    info: &store.ptr_info,
                });
                Ok(false)
            })?;
        }

        self.memory.for_each(range, |_, Stack(byte)| {
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

        let store = &self.store;
        let trackers = &mut self.trackers;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::ReborrowShared,
                ptr: source,
            });

            self.memory.for_each(range.clone(), |byte_pos, Stack(byte)| {
                let pos = 1 + search(source, source_id, byte, &source_range).unwrap();

                let offset = byte[pos..].iter().position(|&id| {
                    let info = &ptr_info[id as usize];
                    info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, &mut filter)
                });

                if let Some(offset) = offset {
                    trackers.call(RawEvent::PoppedOff {
                        event: Event::ReborrowSharedInvalidated { pos: byte_pos },
                        drain: &byte[pos + offset..],
                        info: &store.ptr_info,
                    });
                }

                Ok(false)
            })?;
        }

        self.memory.for_each(range, |_, Stack(byte)| {
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

        if info.members.len() != 1 && old_meta.does_invalidate(meta, &mut D::filter_all()) {
            check_range!(self, ptr, Some(info.range.clone()));

            let (old_id, id, info) = self.store.make_exclusive(ptr).unwrap();
            let range = info.range.clone();

            self.memory.for_each(range.clone(), |_, Stack(byte)| {
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

        info.members.insert(ptr);
        self.allocations[info.alloc_id as usize].insert(ptr);
        self.store.counters.insert(ptr, id);

        OK
    }

    pub fn deallocate(&mut self, ptr: Pointer) -> Result<FxHashSet<Pointer>> {
        if !self.store.get(ptr)?.1.owns_allocation {
            return Err(Error::DeallocateNonOwning(ptr))
        }

        let trackers = &mut self.trackers;
        let store = &self.store;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::Drop,
                ptr,
            });

            self.memory
                .for_each(store.get(ptr)?.1.range.clone(), |byte_index, Stack(byte)| {
                    trackers.call(RawEvent::PoppedOff {
                        drain: byte,
                        info: &store.ptr_info,
                        event: Event::DropInvalidated { pos: byte_index },
                    });
                    Ok(false)
                })?;
        }

        let info = self.store.dealloc(ptr)?;
        self.deallocated.insert(ptr);
        let mut allocation = self.allocations.remove(info.alloc_id as usize);
        self.trackers.trackers.remove(&ptr);

        for &ptr in allocation.iter() {
            let _ = self.store.drop(ptr);
            self.deallocated.insert(ptr);
            self.trackers.trackers.remove(&ptr);
        }

        allocation.insert(ptr);

        self.memory.for_each(info.range, |_, Stack(byte)| {
            byte.clear();
            Ok(true)
        })?;

        Ok(allocation)
    }

    pub fn mark_exclusive(&mut self, ptr: Pointer) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let (old_id, id, info) = self.store.make_exclusive(ptr)?;
        info.ptr_ty = PtrType::Exclusive;
        let info = &self.store.ptr_info[id as usize];
        let range = &info.range;

        let trackers = &mut self.trackers;
        let ptr_info = &self.store.ptr_info;

        if !trackers.is_empty() {
            let range = &info.range;

            trackers.call(RawEvent::Event {
                event: Event::MarkExlusive,
                ptr,
            });

            self.memory.for_each(range.clone(), |byte_pos, Stack(byte)| {
                let pos = search(ptr, old_id, byte, range).unwrap();

                trackers.call(RawEvent::PoppedOff {
                    event: Event::MarkExlusiveInvalidated { pos: byte_pos },
                    drain: &byte[pos + 1..],
                    info: ptr_info,
                });

                Ok(false)
            })?;
        }

        self.memory.for_each(range.clone(), |_, Stack(byte)| {
            let pos = search(ptr, old_id, byte, range).unwrap();
            byte.truncate(pos);
            byte.push(id);
            Ok(false)
        })
    }

    pub fn mark_shared(&mut self, ptr: Pointer) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let trackers = &mut self.trackers;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::MarkShared,
                ptr,
            });
        }

        let (_, info) = self.store.get_mut(ptr).unwrap();
        info.ptr_ty = PtrType::Shared;
        self.assert_shared(ptr, D::filter_all())?;
        OK
    }

    pub fn assert_exclusive(&mut self, ptr: Pointer) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let (id, info) = self.store.get(ptr)?;

        if info.ptr_ty != PtrType::Exclusive {
            return Err(Error::NotExclusive(ptr))
        }

        let trackers = &mut self.trackers;
        let ptr_info = &self.store.ptr_info;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::AssertExlusive,
                ptr,
            });

            self.memory.for_each(info.range.clone(), |byte_pos, Stack(byte)| {
                let pos = 1 + search(ptr, id, byte, &info.range).unwrap();
                trackers.call(RawEvent::PoppedOff {
                    event: Event::AssertExlusiveInvalidated { pos: byte_pos },
                    drain: &byte[pos..],
                    info: ptr_info,
                });
                Ok(false)
            })?;
        }

        self.memory.for_each(info.range.clone(), |_, Stack(byte)| {
            let pos = 1 + search(ptr, id, byte, &info.range).unwrap();
            byte.truncate(pos);
            Ok(false)
        })
    }

    pub fn assert_shared(&mut self, ptr: Pointer, mut filter: D::Filter) -> Result {
        check_dealloc!(self, ptr);
        check_range!(self, ptr);

        let filter = &mut filter;
        let (id, info) = self.store.get(ptr)?;
        let ptr_info = &self.store.ptr_info;
        let meta = info.meta;

        let trackers = &mut self.trackers;

        if !trackers.is_empty() {
            trackers.call(RawEvent::Event {
                event: Event::AssertShared,
                ptr,
            });

            self.memory.for_each(info.range.clone(), |byte_pos, Stack(byte)| {
                let pos = 1 + search(ptr, id, byte, &info.range).unwrap();

                let offset = byte[pos..].iter().position(|&id| {
                    let info = &ptr_info[id as usize];
                    info.ptr_ty == PtrType::Exclusive || meta.does_invalidate(info.meta, filter)
                });

                if let Some(offset) = offset {
                    trackers.call(RawEvent::PoppedOff {
                        event: Event::AssertSharedInvalidated { pos: byte_pos },
                        drain: &byte[pos + offset..],
                        info: ptr_info,
                    })
                }

                Ok(false)
            })?;
        }

        self.memory.for_each(info.range.clone(), |_, Stack(byte)| {
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

impl<D> fmt::Debug for Tracker<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tracker")
            .field("tag", &self.tag)
            .field("event_handler", &"<closure>")
            .finish()
    }
}
