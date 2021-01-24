use memory_model::{
    alias::{MemoryBlock, Meta, PtrType},
    Pointer,
};

fn main() {
    let mut memory = MemoryBlock::new();
    let a = Pointer::create();
    let b = Pointer::create();
    let c = Pointer::create();

    memory.allocate(a, 0..4).unwrap();

    memory
        .reborrow(b, a, Meta {
            ptr_ty: PtrType::Exclusive,
            range: 0..4,
            read: true,
            write: true,
        })
        .unwrap();

    memory.downgrade(b).unwrap();

    memory
        .reborrow(c, a, Meta {
            ptr_ty: PtrType::Shared,
            range: 2..4,
            read: true,
            write: false,
        })
        .unwrap();

    memory.read(b).unwrap();

    println!("{:?}", memory);
}
