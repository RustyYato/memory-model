#![feature(drain_filter)]

use std::ops::Range;

use compiler::ast::{Arg, ArgTy, Ast, AstKind, Attribute};
use fxhash::FxHashMap;
use memory_model::{
    alias::{Event, MemoryBlock, Metadata, PtrType, Tracker},
    Pointer,
};

use hashbrown::{hash_map::Entry, HashMap};

mod error;

enum Error<'a> {
    Alias(memory_model::alias::Error),
    InvalidPtr(&'a str),
    TypeMismatch { arg: Arg<'a>, farg: Arg<'a> },
    NoFunction { func: &'a str },
    NoAttribute { attr: Attribute<'a> },
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

type State<'env> = MemoryBlock<'env, Permissions, memory_model::alias::HashPointerMap<fxhash::FxBuildHasher>>;
type FunctionMap<'a> = HashMap<&'a str, Function<'a>>;

struct Function<'a> {
    args: Vec<Arg<'a>>,
    map: FunctionMap<'a>,
    instrs: Vec<Ast<'a>>,
}

#[derive(Clone, Copy)]
struct FunctionMapRef<'m, 'a> {
    parent: Option<&'m FunctionMapRef<'m, 'a>>,
    inner: &'m FunctionMap<'a>,
}

#[derive(Default, Debug)]
struct Allocator<'a> {
    name_to_ptr: HashMap<&'a str, Pointer>,
    ptr_to_name: FxHashMap<Pointer, &'a str>,
    invalidated: HashMap<&'a str, Invalidated>,
}

#[derive(Debug)]
struct Invalidated {
    span: Range<usize>,
    kind: InvalidKind,
}

#[derive(Debug)]
enum InvalidKind {
    Freed,
    Moved,
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

impl<'a> Allocator<'a> {
    fn alloc(&mut self, name: &'a str) -> Pointer {
        let ptr = Pointer::create();

        self.invalidated.remove(name);

        self.name_to_ptr.insert(name, ptr);
        self.ptr_to_name.insert(ptr, name);

        ptr
    }

    fn dealloc(&mut self, ptr: Pointer, span: Range<usize>) {
        if let Some(name) = self.ptr_to_name.remove(&ptr) {
            self.invalidated.insert(name, Invalidated {
                span,
                kind: InvalidKind::Freed,
            });
            self.name_to_ptr.remove(name);
        }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::env::args().nth(1).unwrap();
    let file = std::fs::read_to_string(file).unwrap();

    let mut sum = 0;
    let line_offsets = std::iter::once(0)
        .chain(file.split('\n').map(|input| {
            let len = input.len() + 1;
            sum += len;
            sum
        }))
        .chain(std::iter::once(file.len()))
        .collect::<Vec<_>>();

    let input = dec::base::indexed::Indexed::new(file.as_str());
    let (input, tokens) = compiler::tokens::parse::<dec::base::error::DefaultError<_>>(input).unwrap();
    if !input.is_empty() {
        println!("{}", input.inner());
        return Err("Could not parse tokens".into())
    }
    let (input, mut instrs) = compiler::ast::parse::<dec::base::error::verbose::VerboseError<_>>(&tokens).unwrap();
    if !input.is_empty() {
        println!("{:#?}", instrs);
        println!("{:#?}", input);
        return Err("Could not parse ast".into())
    }

    let mut model = State::with_size(0);
    let mut allocator = Allocator::default();
    let map = build_function_map(&mut instrs);
    let map = FunctionMapRef {
        inner: &map,
        parent: None,
    };

    run(&instrs, &line_offsets, map, &mut model, &mut allocator)
}

fn build_function_map<'a>(instrs: &mut Vec<Ast<'a>>) -> FunctionMap<'a> {
    let mut map = FunctionMap::default();
    let drain = instrs.drain_filter(|ast| matches!(ast.kind, AstKind::FuncDecl { .. }));

    for func_decl in drain {
        let (name, args, mut instrs) = match func_decl.kind {
            AstKind::FuncDecl { name, args, instr } => (name, args, instr),
            _ => unreachable!(),
        };

        match map.entry(name) {
            Entry::Vacant(entry) => {
                entry.insert(Function {
                    args,
                    map: build_function_map(&mut instrs),
                    instrs,
                });
            }
            Entry::Occupied(_) => panic!("function {} was declared twice!", name),
        }
    }

    map
}

fn run<'a>(
    instrs: &[Ast<'a>],
    line_offsets: &[usize],
    map: FunctionMapRef<'_, 'a>,
    model: &mut State<'a>,
    allocator: &mut Allocator<'a>,
) -> Result<(), Box<dyn std::error::Error>> {
    for ast in instrs {
        macro_rules! try_or_throw {
            ($result:expr) => {
                match $result {
                    Ok(x) => x,
                    Err(e) => {
                        return Err(error::handle_error(
                            Error::from(e),
                            ast.span.clone(),
                            allocator,
                            line_offsets,
                        ))
                    }
                }
            };
            ($result:expr, span = $span:expr) => {
                match $result {
                    Ok(x) => x,
                    Err(e) => {
                        return Err(error::handle_error(
                            Error::from(e),
                            $span,
                            allocator,
                            line_offsets,
                        ))
                    }
                }
            };
        }

        let mut should_track = false;
        for attr in ast.attrs.iter() {
            match attr.path {
                "track" => {
                    should_track = true;
                }
                _ => try_or_throw!(Err(Error::NoAttribute { attr: attr.clone() })),
            }
        }

        let event_handler = event_handler(&ast.span, line_offsets);

        println!("execute {}", Span::new(&ast.span, line_offsets));

        match ast.kind {
            AstKind::FuncDecl { .. } => (),
            AstKind::Allocate { name, ref range } => match *range {
                Range {
                    start: Some(start),
                    end: Some(end),
                } => {
                    let ptr = allocator.alloc(name);
                    try_or_throw!(model.allocate(ptr, start..end));

                    if should_track {
                        eprintln!(
                            "{}: allocated `{}` as {:?}",
                            ShowSpan(&ast.span, line_offsets),
                            name,
                            ptr
                        );
                        model.trackers.push(Tracker::new(name, ptr, event_handler.clone()))
                    }
                }
                _ => panic!("range bounds not specified"),
            },
            AstKind::Drop { name } => {
                let ptr = try_or_throw!(allocator.ptr(name));
                let deallocated = try_or_throw!(model.deallocate(ptr));
                for ptr in deallocated {
                    allocator.dealloc(ptr, ast.span.clone());
                }
            }
            AstKind::Borrow {
                name,
                source,
                is_exclusive,
                ref range,
                read,
                write,
            } => {
                let source_name = source;
                let source = try_or_throw!(allocator.ptr(source));
                let ptr = allocator.alloc(name);

                let info = try_or_throw!(model.info(source));
                let source_range = &info.range;
                let range = range.clone().unwrap_or(None..None);
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

                if should_track {
                    eprintln!(
                        "{}: {} borrow `{}` as {:?} from `{}` ({:?})",
                        ShowSpan(&ast.span, line_offsets),
                        if is_exclusive { "exclusive" } else { "shared" },
                        name,
                        ptr,
                        source_name,
                        source,
                    );
                    model.trackers.push(Tracker::new(name, ptr, event_handler))
                }
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
                    PtrType::Shared => {
                        let ptr = allocator.alloc(name);
                        try_or_throw!(model.copy(ptr, source_ptr))
                    }
                }
            }
            AstKind::FuncCall { name, ref args } => {
                let mut map_ = &map;
                let func = loop {
                    if let Some(func) = map_.inner.get(name) {
                        break func
                    }

                    match map_.parent {
                        Some(parent) => map_ = parent,
                        None => try_or_throw!(Err(Error::NoFunction { func: name })),
                    }
                };
                let map = FunctionMapRef {
                    inner: &func.map,
                    parent: Some(&map),
                };
                let instrs = &*func.instrs;

                assert_eq!(args.len(), func.args.len());

                for (farg, &(ref span, arg)) in func.args.iter().zip(args.iter()) {
                    let ty = match farg.ty {
                        Some(ref ty) => ty,
                        None => continue,
                    };

                    let perm = Permissions {
                        read: ty.read,
                        write: ty.write,
                    };

                    let ptr = try_or_throw!(allocator.ptr(arg));
                    let info = try_or_throw!(model.info(ptr));
                    let is_exclusive = matches!(info.ptr_ty, memory_model::alias::PtrType::Exclusive);
                    if info.meta == perm && is_exclusive == ty.is_exclusive {
                        continue
                    }

                    try_or_throw!(
                        Err(Error::TypeMismatch {
                            arg: Arg {
                                name: arg,
                                span: span.clone(),
                                ty: Some(ArgTy {
                                    is_exclusive,
                                    read: info.meta.read,
                                    write: info.meta.write,
                                })
                            },
                            farg: farg.clone(),
                        }),
                        span = span.clone()
                    );
                }

                let mut func_allocator = Allocator::default();

                for (farg, &(ref span, arg)) in func.args.iter().zip(args.iter()) {
                    let name = farg.name;
                    let source = arg;
                    let span = span.clone();

                    let source_ptr = try_or_throw!(allocator.ptr(source), span = span);
                    allocator.invalidated.insert(source, Invalidated {
                        span: ast.span.clone(),
                        kind: InvalidKind::Moved,
                    });

                    let info = try_or_throw!(model.info(source_ptr), span = span);
                    match info.ptr_ty {
                        PtrType::Exclusive => {
                            let ptr = allocator.name_to_ptr.remove(source).unwrap();
                            allocator.ptr_to_name.remove(&ptr);

                            func_allocator.name_to_ptr.insert(name, ptr);
                            func_allocator.ptr_to_name.insert(ptr, name);
                        }
                        PtrType::Shared => {
                            let ptr = func_allocator.alloc(name);
                            try_or_throw!(model.copy(ptr, source_ptr), span = span)
                        }
                    }
                }

                run(instrs, line_offsets, map, model, &mut func_allocator)?;
            }
        }
    }

    Ok(())
}

fn event_handler(span: &Range<usize>, line_offsets: &[usize]) -> impl FnMut(&str, Pointer, Event) + Clone + Send {
    let span = Span::new(span, line_offsets);
    move |tag: &str, ptr: Pointer, event: Event| println!("{:?}: {:?} ({})", event, ptr, tag)
}

struct ShowSpan<'a>(pub &'a Range<usize>, pub &'a [usize]);
#[derive(Clone, Copy)]
struct Span {
    line_start: usize,
    col_start: usize,
    line_end: usize,
    col_end: usize,
}

impl Span {
    fn new(span: &Range<usize>, line_offsets: &[usize]) -> Self {
        let line_start = match line_offsets.binary_search(&span.start) {
            Ok(x) => x,
            Err(x) => x - 1,
        };
        let line_end = match line_offsets.binary_search(&span.end) {
            Ok(x) => x,
            Err(x) => x - 1,
        };

        let col_start = span.start - line_offsets[line_start];
        let col_end = span.end - line_offsets[line_end];

        Self {
            line_start: 1 + line_start,
            col_start: 1 + col_start,
            line_end: 1 + line_end,
            col_end: 1 + col_end,
        }
    }
}

use std::fmt;

impl fmt::Display for ShowSpan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { Span::new(self.0, self.1).fmt(f) }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "span({}:{}..{}:{})",
            self.line_start, self.col_start, self.line_end, self.col_end
        )
    }
}
