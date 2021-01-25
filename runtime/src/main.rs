use core::panic;
use std::ops::Range;

use compiler::ast::AstKind;
use fxhash::FxHashMap;
use memory_model::{
    alias::{MemoryBlock, Metadata, PtrType},
    Pointer,
};

use hashbrown::HashMap;

enum Error<'a> {
    Alias(memory_model::alias::Error),
    InvalidPtr(&'a str),
}

impl From<memory_model::alias::Error> for Error<'_> {
    fn from(err: memory_model::alias::Error) -> Self { Self::Alias(err) }
}

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
    invalidated: HashMap<&'a str, Invalidated>,
}

struct Invalidated {
    span: Range<usize>,
    kind: InvalidKind,
}

enum InvalidKind {
    Freed,
    Moved,
}

impl<'a> Allocator<'a> {
    fn alloc(&mut self, name: &'a str) -> Pointer {
        let ptr = Pointer::create();

        self.invalidated.remove(name);

        self.name_to_ptr.insert(name, ptr);
        self.ptr_to_name.insert(ptr, name);

        ptr
    }

    fn dealloc(&mut self, ptr: Pointer, span: Range<usize>) {
        let name = self.ptr_to_name.remove(&ptr).unwrap();
        self.invalidated.insert(name, Invalidated {
            span,
            kind: InvalidKind::Freed,
        });
        self.name_to_ptr.remove(name);
    }

    fn name(&self, ptr: Pointer) -> &'a str { self.ptr_to_name[&ptr] }

    fn ptr<'n>(&self, name: &'n str) -> Result<Pointer, Error<'n>> {
        self.name_to_ptr.get(name).copied().ok_or(Error::InvalidPtr(name))
    }

    fn rename(&mut self, source: &str, name: &'a str) {
        let ptr = self.name_to_ptr.remove(source).unwrap();
        self.name_to_ptr.insert(name, ptr);
        *self.ptr_to_name.get_mut(&ptr).unwrap() = name;
    }
}

fn span_to_string(span: &Range<usize>, line_offsets: &[usize]) -> String {
    let line_start = match line_offsets.binary_search(&span.start) {
        Ok(x) | Err(x) => x,
    };
    let line_end = match line_offsets.binary_search(&span.end) {
        Ok(x) | Err(x) => x,
    };

    let col_start = span.start - line_offsets[line_start];
    let col_end = span.end - line_offsets[line_end];

    let mut line_end = if line_start + 1 == line_end && col_end == 0 {
        1 + line_start
    } else {
        line_end
    };

    let col_end = if col_end == 0 {
        let col_end = line_offsets[line_end] - line_offsets[line_end - 1] - 1;
        if col_end == 0 {
            line_end -= 1;
            line_offsets[line_end] - line_offsets[line_end - 1]
        } else {
            col_end
        }
    } else {
        col_end
    };

    format!("span({}:{}..{}:{})", 1 + line_start, 1 + col_start, line_end, col_end)
}

#[cold]
#[inline(never)]
fn handle_error(
    err: Error,
    span: std::ops::Range<usize>,
    allocator: &Allocator,
    line_offsets: &[usize],
) -> Box<dyn std::error::Error> {
    use memory_model::alias::Error::*;

    let span = span_to_string(&span, line_offsets);

    let err = match err {
        Error::Alias(err) => err,
        Error::InvalidPtr(ptr) => {
            return match allocator.invalidated.get(ptr) {
                Some(Invalidated {
                    kind: InvalidKind::Freed,
                    span: freed_span,
                }) => format!(
                    "{}: Use of freed pointer `{ptr}`. Note: freed `{ptr}` at {freed_span}",
                    span,
                    ptr = ptr,
                    freed_span = span_to_string(&freed_span, line_offsets)
                )
                .into(),
                Some(Invalidated {
                    kind: InvalidKind::Moved,
                    span: moved_span,
                }) => format!(
                    "{}: Use of moved pointer `{ptr}`. Note: moved `{ptr}` at {moved_span}",
                    span,
                    ptr = ptr,
                    moved_span = span_to_string(&moved_span, line_offsets)
                )
                .into(),
                None => format!("{}: Unknown pointer `{}`", span, ptr).into(),
            }
        }
    };

    match err {
        ReborrowInvalidatesSource { ptr, source } => format!(
            "{}: Could not borrow `{}`, because it invalidated it's source `{}`",
            span,
            allocator.name(ptr),
            allocator.name(source)
        )
        .into(),
        UseAfterFree(ptr) => format!("{}: Tried to use `{}` after it was freed", span, allocator.name(ptr)).into(),
        InvalidPtr(ptr) => format!(
            "{}: Tried to use `{}`, which was never registered",
            span,
            allocator.name(ptr)
        )
        .into(),
        NotExclusive(ptr) => format!(
            "{}: Tried to use `{}` exclusively, but it is shared",
            span,
            allocator.name(ptr)
        )
        .into(),
        NotShared(ptr) => format!(
            "{}: Tried to use `{}` as shared, but it is exclusive",
            span,
            allocator.name(ptr)
        )
        .into(),
        DeallocateNonOwning(ptr) => format!(
            "{}: Tried to deallocate `{}`, but it doesn't own an allocation",
            span,
            allocator.name(ptr)
        )
        .into(),
        InvalidatesOldMeta(ptr) => format!(
            "{}: Tried to update the meta data of `{}`, but it would invalidate itself",
            span,
            allocator.name(ptr)
        )
        .into(),
        AllocateRangeOccupied { ptr: _, range } => format!(
            "{}: Tried to allocate in range {:?}, but that range is already occupied",
            span, range
        )
        .into(),
        InvalidForRange { ptr, range } => format!(
            "{}: Tried to use `{}` for the range {:?}, but it is not valid for that range",
            span,
            allocator.name(ptr),
            range
        )
        .into(),
        ReborrowSubset {
            ptr,
            source,
            source_range,
        } => format!(
            "{}: Tried to reborrow `{}` from `{2}` for the range {3:?}, but `{2}` is not valid for that range",
            span,
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

    let mut sum = 0;
    let line_offsets = std::iter::once(0)
        .chain(file.split('\n').map(|input| {
            let len = input.len() + 1;
            sum += len;
            sum - 2
        }))
        .chain(std::iter::once(file.len()))
        .collect::<Vec<_>>();

    let input = dec::base::indexed::Indexed::new(file.as_str());
    let (input, tokens) = compiler::tokens::parse::<dec::base::error::DefaultError<_>>(input).unwrap();
    if !input.is_empty() {
        println!("{}", input.inner());
        return Err("Could not parse tokens".into())
    }
    let (input, ast) = compiler::ast::parse::<dec::base::error::DefaultError<_>>(&tokens).unwrap();
    if !input.is_empty() {
        println!("{:#?}", ast);
        println!("{:#?}", input);
        return Err("Could not parse ast".into())
    }

    let mut allocator = Allocator::default();
    let mut model = MemoryBlock::<_, memory_model::alias::HashPointerMap<fxhash::FxBuildHasher>>::with_size(0);

    for ast in ast {
        macro_rules! try_or_throw {
            ($result:expr) => {
                match $result {
                    Ok(x) => x,
                    Err(e) => return Err(handle_error(Error::from(e), ast.span, &allocator, &line_offsets)),
                }
            };
        }

        match ast.kind {
            AstKind::Allocate { name, range } => match range {
                Range {
                    start: Some(start),
                    end: Some(end),
                } => {
                    let ptr = allocator.alloc(name);
                    try_or_throw!(model.allocate(ptr, start..end));
                }
                _ => panic!("range bounds not specified"),
            },
            AstKind::Drop { name } => {
                let ptr = try_or_throw!(allocator.ptr(name));
                let mut deallocated = try_or_throw!(model.deallocate(ptr));
                deallocated.sort_unstable();
                deallocated.dedup();
                for ptr in deallocated {
                    allocator.dealloc(ptr, ast.span.clone());
                }
            }
            AstKind::Borrow {
                name,
                source,
                is_exclusive,
                range,
                read,
                write,
            } => {
                let ptr = allocator.alloc(name);
                let source = try_or_throw!(allocator.ptr(source));

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
            AstKind::Update {
                name,
                is_exclusive,
                read,
                write,
            } => {
                let ptr = try_or_throw!(allocator.ptr(name));
                let res = if is_exclusive {
                    model.mark_exclusive(ptr)
                } else {
                    model.mark_shared(ptr)
                };
                try_or_throw!(res);
                let res = model.update_meta(ptr, |_| Permissions { read, write });
                try_or_throw!(res);
            }
            AstKind::Write { name, is_exclusive } => {
                let ptr = try_or_throw!(allocator.ptr(name));
                assert!(try_or_throw!(model.info(ptr)).meta.write);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Write)
                };
                try_or_throw!(res);
            }
            AstKind::Read { name, is_exclusive } => {
                let ptr = try_or_throw!(allocator.ptr(name));
                assert!(try_or_throw!(model.info(ptr)).meta.read);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Read)
                };
                try_or_throw!(res);
            }
            AstKind::Move { name, source } => {
                let source_ptr = try_or_throw!(allocator.ptr(source));
                allocator.invalidated.insert(source, Invalidated {
                    span: ast.span.clone(),
                    kind: InvalidKind::Moved,
                });

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
