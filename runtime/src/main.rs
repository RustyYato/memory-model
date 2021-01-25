use core::panic;
use std::ops::Range;

use compiler::ast::Ast;
use fxhash::FxHashMap;
use memory_model::{
    alias::{MemoryBlock, Metadata, PtrType},
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

fn main() {
    let file = std::env::args().nth(1).unwrap();
    let file = std::fs::read_to_string(file).unwrap();
    let (input, tokens) = compiler::tokens::parse::<dec::base::error::DefaultError<_>>(&file).unwrap();
    if !input.is_empty() {
        println!("{}", input);
        return
    }
    let (input, ast) = compiler::ast::parse::<dec::base::error::DefaultError<_>>(&tokens).unwrap();
    if !input.is_empty() {
        println!("{:#?}", ast);
        println!("{:#?}", input);
        return
    }

    let mut allocator = Allocator::default();
    let mut model = MemoryBlock::<_, FxHashMap<_, _>>::with_size(0);

    for ast in ast {
        println!("{:?}", ast);
        match ast {
            Ast::Allocate { name, range } => match range {
                Range {
                    start: Some(start),
                    end: Some(end),
                } => {
                    let ptr = allocator.alloc(name);
                    model.allocate(ptr, start..end).unwrap();
                }
                _ => panic!("range bounds not specified"),
            },
            Ast::Drop { name } => {
                let ptr = allocator.ptr(name);
                model.deallocate(ptr).unwrap();
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

                let info = model.info(source).unwrap();
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
                res.unwrap();
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
                res.unwrap();
                model.update_meta(ptr, |_| Permissions { read, write }).unwrap();
            }
            Ast::Write { name, is_exclusive } => {
                let ptr = allocator.ptr(name);
                assert!(model.info(ptr).unwrap().meta.write);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Write)
                };
                res.unwrap();
            }
            Ast::Read { name, is_exclusive } => {
                let ptr = allocator.ptr(name);
                assert!(model.info(ptr).unwrap().meta.read);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Read)
                };
                res.unwrap();
            }
            Ast::Move { name, source } => {
                let source_ptr = allocator.ptr(source);

                let info = model.info(source_ptr).unwrap();
                match info.ptr_ty {
                    PtrType::Exclusive => allocator.rename(source, name),
                    PtrType::Shared => model.copy(allocator.alloc(name), source_ptr).unwrap(),
                }
            }
        }

        // println!("{:#?}\n\n", model);
    }
}
