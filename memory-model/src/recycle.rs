use std::{cmp::Ordering, collections::BinaryHeap, hash::Hash};
use sync_wrapper::SyncWrapper;

pub trait Recycle: Default {
    type Size: Ord + Hash;

    fn empty(self) -> Self;

    fn size(&self) -> Self::Size;
}

pub struct Recycler<T> {
    heap: SyncWrapper<BinaryHeap<Empty<T>>>,
}

struct Empty<T>(T);

pub struct AutoCollect<'a, T: Recycle>(&'a mut Recycler<T>, T);

impl<T: Recycle> std::ops::Deref for AutoCollect<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.1 }
}

impl<T: Recycle> std::ops::DerefMut for AutoCollect<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.1 }
}

impl<T: Recycle> Drop for AutoCollect<'_, T> {
    fn drop(&mut self) { self.0.put(std::mem::take(&mut self.1)) }
}

impl<T: Recycle> Eq for Empty<T> {}
impl<T: Recycle> PartialEq for Empty<T> {
    fn eq(&self, other: &Self) -> bool { self.0.size() == other.0.size() }
}

impl<T: Recycle> PartialOrd for Empty<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<T: Recycle> Ord for Empty<T> {
    fn cmp(&self, other: &Self) -> Ordering { self.0.size().cmp(&other.0.size()) }
}

impl<T: Recycle> Default for Recycler<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Recycle> Recycler<T> {
    pub fn new() -> Self {
        Self {
            heap: SyncWrapper::new(BinaryHeap::new()),
        }
    }

    pub fn clear(&mut self) { self.heap.get_mut().clear() }

    pub fn put(&mut self, item: T) { self.heap.get_mut().push(Empty(item.empty())); }

    pub fn take(&mut self) -> T {
        self.heap
            .get_mut()
            .pop()
            .map_or_else(Default::default, |Empty(vec)| vec)
    }

    pub fn lease(&mut self) -> AutoCollect<'_, T> {
        let lease = self.take();
        AutoCollect(self, lease)
    }
}

impl<T> Recycle for Vec<T> {
    type Size = usize;

    fn empty(mut self) -> Self {
        self.clear();
        self
    }

    fn size(&self) -> Self::Size { self.capacity() }
}
