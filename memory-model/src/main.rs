use memory_model::{
    alias::{MemoryBlock, Metadata},
    Pointer,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum PermissionsFilter {
    Read,
    Write,
    Borrow,
}

impl Metadata for Permissions {
    fn alloc() -> Self {
        Self {
            read: true,
            write: true,
        }
    }

    fn filter_all() -> Self::Filter { PermissionsFilter::Borrow }

    type Filter = PermissionsFilter;

    fn does_invalidate(self, other: Self, filter: &mut Self::Filter) -> bool {
        u8::from(other.read || matches!(filter, PermissionsFilter::Write)) < u8::from(self.read)
            || u8::from(other.write || matches!(filter, PermissionsFilter::Read)) < u8::from(self.write)
    }
}

fn main() {
    let mut memory = MemoryBlock::new();
    let a = Pointer::create();
    let b = Pointer::create();
    let c = Pointer::create();
    let d = Pointer::create();
    let e = Pointer::create();

    memory.allocate(a, 0..4).unwrap();

    memory
        .reborrow_shared(b, a, 0..4, Permissions {
            read: true,
            write: true,
        })
        .unwrap();

    memory
        .reborrow_shared(c, a, 2..4, Permissions {
            read: true,
            write: false,
        })
        .unwrap();

    memory
        .reborrow_shared(d, b, 2..4, Permissions {
            read: false,
            write: true,
        })
        .unwrap();

    assert!(memory.info(b).unwrap().meta.write);
    memory.assert_shared(b, PermissionsFilter::Write).unwrap();

    memory
        .reborrow_shared(e, b, 2..4, Permissions {
            read: false,
            write: true,
        })
        .unwrap();

    assert!(memory.info(b).unwrap().meta.write);
    memory.assert_shared(b, PermissionsFilter::Write).unwrap();

    memory.deallocate(a).unwrap();

    println!("{:#?}", memory);
}
