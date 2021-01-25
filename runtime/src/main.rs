use core::panic;
use std::ops::Range;

use compiler::ast::Ast;
use fxhash::FxHashMap;
use memory_model::{
    alias::{Error, MemoryBlock, Metadata, PtrType},
    Pointer,
};

use hashbrown::HashMap;

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

#[derive(Default)]
struct Allocator<'a> {
    name_to_ptr: HashMap<&'a str, Pointer>,
    ptr_to_name: FxHashMap<Pointer, &'a str>,
}

impl<'a> Allocator<'a> {
    fn alloc(&mut self, name: &'a str) -> Pointer {
        let ptr = Pointer::create();

        self.name_to_ptr.insert(name, ptr);
        self.ptr_to_name.insert(ptr, name);

        ptr
    }

    fn dealloc(&mut self, ptr: Pointer) {
        let name = self.ptr_to_name.remove(&ptr).unwrap();
        self.name_to_ptr.remove(name);
    }

    fn name(&self, ptr: Pointer) -> &'a str { self.ptr_to_name[&ptr] }

    fn ptr(&self, name: &str) -> Pointer { self.name_to_ptr[name] }

    fn rename(&mut self, source: &str, name: &'a str) {
        let ptr = self.name_to_ptr.remove(source).unwrap();
        self.name_to_ptr.insert(name, ptr);
        *self.ptr_to_name.get_mut(&ptr).unwrap() = name;
    }
}

#[cold]
#[inline(never)]
fn handle_error<M: memory_model::alias::PointerMap>(
    err: Error,
    ast: Ast<'_>,
    allocator: &Allocator,
    model: &MemoryBlock<Permissions, M>,
) -> Box<dyn std::error::Error> {
    use Error::*;
    match err {
        ReborrowInvalidatesSource { ptr, source } => format!(
            "Could not borrow `{}`, because it invalidated it's source `{}`",
            allocator.name(ptr),
            allocator.name(source)
        )
        .into(),
        UseAfterFree(ptr) => format!("Tried to use `{}` after it was freed", allocator.name(ptr)).into(),
        InvalidPtr(ptr) => format!("Tried to use `{}`, which was never registered", allocator.name(ptr)).into(),
        NotExclusive(ptr) => format!("Tried to use `{}` exclusively, but it is shared", allocator.name(ptr)).into(),
        NotShared(ptr) => format!("Tried to use `{}` as shared, but it is exclusive", allocator.name(ptr)).into(),
        DeallocateNonOwning(ptr) => format!(
            "Tried to deallocate `{}`, but it doesn't own an allocation",
            allocator.name(ptr)
        )
        .into(),
        InvalidatesOldMeta(ptr) => format!(
            "Tried to update the meta data of `{}`, but it would invalidate itself",
            allocator.name(ptr)
        )
        .into(),
        AllocateRangeOccupied { ptr: _, range } => format!(
            "Tried to allocate in range {:?}, but that range is already occupied",
            range
        )
        .into(),
        InvalidForRange { ptr, range } => format!(
            "Tried to use `{}` for the range {:?}, but it is not valid for that range",
            allocator.name(ptr),
            range
        )
        .into(),
        ReborrowSubset {
            ptr,
            source,
            source_range,
        } => format!(
            "Tried to reborrow `{}` from `{1}` for the range {2:?}, but `{1}` is not valid for that range",
            allocator.name(ptr),
            allocator.name(source),
            source_range
        )
        .into(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args().nth(1).unwrap();
    let file = std::fs::read_to_string(file).unwrap();
    let (input, tokens) = compiler::tokens::parse::<dec::base::error::DefaultError<_>>(&file).unwrap();
    if !input.is_empty() {
        println!("{}", input);
        return Err("Could not parse tokens".into())
    }
    let (input, ast) = compiler::ast::parse::<dec::base::error::DefaultError<_>>(&tokens).unwrap();
    if !input.is_empty() {
        println!("{:#?}", ast);
        println!("{:#?}", input);
        return Err("Could not parse ast".into())
    }

    let mut allocator = Allocator::default();
    let mut model = MemoryBlock::<_, FxHashMap<_, _>>::with_size(0);

    for ast in ast {
        macro_rules! try_or_throw {
            ($result:expr) => {
                match $result {
                    Ok(x) => x,
                    Err(e) => return Err(handle_error(e, ast, &allocator, &model)),
                }
            };
        }

        match ast.clone() {
            Ast::Allocate { name, range } => match range {
                Range {
                    start: Some(start),
                    end: Some(end),
                } => {
                    let ptr = allocator.alloc(name);
                    try_or_throw!(model.allocate(ptr, start..end));
                }
                _ => panic!("range bounds not specified"),
            },
            Ast::Drop { name } => {
                let ptr = allocator.ptr(name);
                try_or_throw!(model.deallocate(ptr));
                allocator.dealloc(ptr);
            }
            Ast::Borrow {
                name,
                source,
                is_exclusive,
                range,
                read,
                write,
            } => {
                let ptr = allocator.alloc(name);
                let source = allocator.ptr(source);

                let info = try_or_throw!(model.info(source));
                let source_range = &info.range;
                let range = range.unwrap_or(None..None);
                let start = range.start.unwrap_or(source_range.start);
                let end = range.end.unwrap_or(source_range.end);
                let range = start..end;

                let meta = Permissions { read, write };

                let res = if is_exclusive {
                    model.reborrow_exclusive(ptr, source, range, meta)
                } else {
                    model.reborrow_shared(ptr, source, range, meta)
                };
                try_or_throw!(res);
            }
            Ast::Update {
                name,
                is_exclusive,
                read,
                write,
            } => {
                let ptr = allocator.ptr(name);
                let res = if is_exclusive {
                    model.mark_exclusive(ptr)
                } else {
                    model.mark_shared(ptr)
                };
                try_or_throw!(res);
                let res = model.update_meta(ptr, |_| Permissions { read, write });
                try_or_throw!(res);
            }
            Ast::Write { name, is_exclusive } => {
                let ptr = allocator.ptr(name);
                assert!(try_or_throw!(model.info(ptr)).meta.write);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Write)
                };
                try_or_throw!(res);
            }
            Ast::Read { name, is_exclusive } => {
                let ptr = allocator.ptr(name);
                assert!(try_or_throw!(model.info(ptr)).meta.read);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Read)
                };
                try_or_throw!(res);
            }
            Ast::Move { name, source } => {
                let source_ptr = allocator.ptr(source);

                let info = try_or_throw!(model.info(source_ptr));
                match info.ptr_ty {
                    PtrType::Exclusive => allocator.rename(source, name),
                    PtrType::Shared => try_or_throw!(model.copy(allocator.alloc(name), source_ptr)),
                }
            }
        }

        // println!("{:#?}\n\n", model);
    }

    Ok(())
}
