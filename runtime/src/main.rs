#![feature(drain_filter)]
#![allow(unused)]

use std::{convert::TryFrom, ops::Range};

use compiler::ast::{self, Ast, AstKind, Span};
use fxhash::FxHashMap;
use memory_model::{
    alias::{Event, MemoryBlock, Metadata, PtrType},
    Pointer,
};

use hashbrown::{hash_map::Entry, HashMap};

mod error;

#[derive(Debug)]
enum Error<'a> {
    Alias(memory_model::alias::Error),
    InvalidPtr(&'a str),
    TypeMismatch { arg: ast::Arg<'a>, farg: ast::Arg<'a> },
    NoFunction { func: &'a str },
    NoAttribute { attr: ast::Attribute<'a> },
    AllocNoBind { span: ast::Span },
    AllocRangeBoundsNotSpecied { span: ast::Span },
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
    args: Vec<ast::Arg<'a>>,
    map: FunctionMap<'a>,
    instrs: Vec<Ast<'a>>,
    ret_value: Option<ast::Expr<'a>>,
    ret_ty: Option<ast::Type>,
}

#[derive(Clone, Copy)]
struct FunctionMapRef<'m, 'a> {
    parent: Option<&'m FunctionMapRef<'m, 'a>>,
    inner: &'m FunctionMap<'a>,
}

#[derive(Debug)]
enum AllocId<'a> {
    Named(&'a str),
    Temp(u32),
}

#[derive(Default, Debug)]
struct Allocator<'a> {
    temp_to_ptr: FxHashMap<u32, Pointer>,
    name_to_ptr: HashMap<&'a str, Pointer>,
    ptr_to_name: FxHashMap<Pointer, AllocId<'a>>,
    invalidated: HashMap<&'a str, Invalidated>,
}

#[derive(Debug)]
struct Invalidated {
    span: Span,
    kind: InvalidKind,
}

#[derive(Debug)]
enum InvalidKind {
    Freed,
    Moved,
}

impl<'a> From<&'a str> for AllocId<'a> {
    fn from(name: &'a str) -> Self { Self::Named(name) }
}

impl From<u32> for AllocId<'_> {
    fn from(temp: u32) -> Self { Self::Temp(temp) }
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
    fn alloc(&mut self, name: impl Into<AllocId<'a>>) -> Pointer {
        let name = name.into();
        let ptr = Pointer::create();

        match name {
            AllocId::Named(name) => {
                self.invalidated.remove(name);
                self.name_to_ptr.insert(name, ptr);
            }
            AllocId::Temp(temp) => {
                self.temp_to_ptr.insert(temp, ptr);
            }
        }

        self.ptr_to_name.insert(ptr, name);

        ptr
    }

    fn dealloc(&mut self, ptr: Pointer, span: Span) {
        match self.ptr_to_name.remove(&ptr) {
            Some(AllocId::Named(name)) => {
                self.invalidated.insert(name, Invalidated {
                    span,
                    kind: InvalidKind::Freed,
                });
                self.name_to_ptr.remove(name);
            }
            Some(AllocId::Temp(temp)) => {
                self.temp_to_ptr.remove(&temp);
            }
            None => (),
        }
    }

    fn name(&self, ptr: Pointer) -> &'a str {
        match self.ptr_to_name[&ptr] {
            AllocId::Named(name) => name,
            AllocId::Temp(temp) => panic!("tried to get the name of temp {}", temp),
        }
    }

    fn ptr<'n>(&self, name: impl Into<AllocId<'n>>) -> Result<Pointer, Error<'n>> {
        match name.into() {
            AllocId::Named(name) => self.name_to_ptr.get(name).copied().ok_or(Error::InvalidPtr(name)),
            AllocId::Temp(temp) => Ok(self.temp_to_ptr[&temp]),
        }
    }

    fn rename<'n>(&mut self, source: &'n str, name: impl Into<AllocId<'a>>) -> Result<(), Error<'n>> {
        let name = name.into();
        let ptr = self.name_to_ptr.remove(source).ok_or(Error::InvalidPtr(source))?;
        match name {
            AllocId::Named(name) => self.name_to_ptr.insert(name, ptr),
            AllocId::Temp(temp) => self.temp_to_ptr.insert(temp, ptr),
        };
        *self.ptr_to_name.get_mut(&ptr).unwrap() = name;
        Ok(())
    }

    fn rename_into<'n>(
        &mut self,
        other: &mut Self,
        source: impl Into<AllocId<'n>>,
        name: impl Into<AllocId<'a>>,
    ) -> Result<(), Error<'n>> {
        let name = name.into();
        let ptr = match source.into() {
            AllocId::Named(source) => self.name_to_ptr.remove(source).ok_or(Error::InvalidPtr(source))?,
            AllocId::Temp(source) => self.temp_to_ptr.remove(&source).unwrap(),
        };
        match name {
            AllocId::Named(name) => other.name_to_ptr.insert(name, ptr),
            AllocId::Temp(temp) => other.temp_to_ptr.insert(temp, ptr),
        };
        self.ptr_to_name.remove(&ptr);
        other.ptr_to_name.insert(ptr, name);
        Ok(())
    }

    fn remove<'n>(&mut self, name: &'n str) -> Result<(), Error<'n>> {
        let ptr = self.name_to_ptr.remove(name).ok_or(Error::InvalidPtr(name))?;
        self.ptr_to_name.remove(&ptr);
        Ok(())
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
        eprintln!("{}", input.inner());
        return Err("Could not parse tokens".into())
    }
    let (input, mut instrs) = compiler::ast::parse::<dec::base::error::verbose::VerboseError<_>>(&tokens).unwrap();
    if !input.is_empty() {
        eprintln!("{:#?}", instrs);
        eprintln!("{:#?}", input);
        return Err("Could not parse ast".into())
    }

    let mut model = State::with_size(0);
    let mut allocator = Allocator::default();

    let map = match build_function_map(&mut instrs) {
        Ok(x) => x,
        Err((span, e)) => return Err(error::handle_error(e, span, &allocator, &line_offsets)),
    };

    let map = FunctionMapRef {
        inner: &map,
        parent: None,
    };

    run(&instrs, &line_offsets, map, &mut model, &mut allocator)
}

fn build_function_map<'a>(instrs: &mut Vec<Ast<'a>>) -> Result<FunctionMap<'a>, (Span, Error<'a>)> {
    let mut map = FunctionMap::default();
    let drain = instrs.drain_filter(|ast| matches!(ast.kind, AstKind::FuncDecl { .. }));

    for func_decl in drain {
        let (id, args, mut instrs, ret_value, ret_ty) = match func_decl.kind {
            AstKind::FuncDecl {
                id,
                args,
                instrs,
                ret_value,
                ret_ty,
            } => (id, args, instrs, ret_value, ret_ty),
            _ => unreachable!(),
        };

        match map.entry(id.name) {
            Entry::Vacant(entry) => {
                entry.insert(Function {
                    args,
                    map: build_function_map(&mut instrs)?,
                    instrs,
                    ret_value,
                    ret_ty,
                });
            }
            Entry::Occupied(_) => panic!("function {} was declared twice!", id.name),
        }
    }

    Ok(map)
}

#[rustfmt::skip]
macro_rules! mk_try_or_throw {
    ($dollar:tt, $default_span:expr, $allocator:expr, $line_offsets:expr) => {
        macro_rules! try_or_throw {
            ($dollar result:expr) => {
                match $dollar result {
                    Ok(x) => x,
                    Err(e) => {
                        return Err(error::handle_error(
                            Error::from(e),
                            $default_span,
                            $allocator,
                            $line_offsets,
                        ))
                    }
                }
            };
            ($dollar result:expr, span = $dollar span:expr) => {
                match $dollar result {
                    Ok(x) => x,
                    Err(e) => {
                        return Err(error::handle_error(
                            Error::from(e),
                            $dollar span,
                            $allocator,
                            $line_offsets,
                        ))
                    }
                }
            };
        }
    };
}

fn run<'a>(
    instrs: &[Ast<'a>],
    line_offsets: &[usize],
    map: FunctionMapRef<'_, 'a>,
    model: &mut State<'a>,
    allocator: &mut Allocator<'a>,
) -> Result<(), Box<dyn std::error::Error>> {
    for ast in instrs {
        mk_try_or_throw!($, ast.span.clone(), allocator, line_offsets);
        let mut should_track = false;
        for attr in ast.attrs.iter() {
            match attr.path.name {
                "track" => should_track = true,
                _ => try_or_throw!(Err(Error::NoAttribute { attr: attr.clone() })),
            }
        }

        eprintln!("execute {}", DispSpan::new(&ast.span, line_offsets));

        match &ast.kind {
            AstKind::FuncDecl { .. } => (),
            AstKind::Let {
                pat: ast::Pattern::Simple(pat),
                expr: ast::Expr::Simple(expr),
            } => write_expr_to(
                WriteTo::Pat(pat.clone()),
                expr.clone(),
                ast.span.clone(),
                should_track,
                line_offsets,
                map,
                model,
                allocator,
                None,
            )?,
            AstKind::Let {
                pat: ast::Pattern::Tuple { pats, span: pat_span },
                expr: ast::Expr::Tuple { exprs, span: expr_span },
            } => {
                assert_eq!(pats.len(), exprs.len());
                for (pat, expr) in pats.iter().zip(exprs) {
                    write_expr_to(
                        WriteTo::Pat(pat.clone()),
                        expr.clone(),
                        ast.span.clone(),
                        should_track,
                        line_offsets,
                        map,
                        model,
                        allocator,
                        None,
                    )?
                }
            }
            AstKind::Let {
                pat: ast::Pattern::Tuple { pats, span: pat_span },
                expr:
                    ast::Expr::FuncCall {
                        ref id,
                        ref args,
                        span: ref expr_span,
                    },
            } => {
                let (func, mut func_allocator) =
                    func_call(id.clone(), args, expr_span.clone(), line_offsets, map, model, allocator)?;
                match &func.ret_value {
                    Some(ast::Expr::Tuple { span, exprs }) => {
                        assert_eq!(pats.len(), exprs.len());
                        for (pat, expr) in pats.iter().zip(exprs) {
                            write_expr_to(
                                WriteTo::Pat(pat.clone()),
                                expr.clone(),
                                ast.span.clone(),
                                should_track,
                                line_offsets,
                                map,
                                model,
                                allocator,
                                Some(&mut func_allocator),
                            )?;
                        }
                    }
                    _ => panic!(),
                }
            }
            AstKind::Let {
                pat: ast::Pattern::Simple(pat @ ast::SimplePattern::Ignore(_)),
                expr:
                    ast::Expr::FuncCall {
                        ref id,
                        ref args,
                        span: ref expr_span,
                    },
            } => {
                let (func, mut func_allocator) =
                    func_call(id.clone(), args, expr_span.clone(), line_offsets, map, model, allocator)?;
                match &func.ret_value {
                    Some(ast::Expr::Simple(expr)) => write_expr_to(
                        WriteTo::Pat(pat.clone()),
                        expr.clone(),
                        ast.span.clone(),
                        should_track,
                        line_offsets,
                        map,
                        model,
                        allocator,
                        Some(&mut func_allocator),
                    )?,
                    Some(ast::Expr::Tuple { span, exprs }) => {
                        for expr in exprs {
                            write_expr_to(
                                WriteTo::Pat(pat.clone()),
                                expr.clone(),
                                ast.span.clone(),
                                should_track,
                                line_offsets,
                                map,
                                model,
                                allocator,
                                Some(&mut func_allocator),
                            )?
                        }
                    }
                    _ => panic!(),
                }
            }
            AstKind::Let {
                pat: ast::Pattern::Simple(pat @ ast::SimplePattern::Ident(_)),
                expr:
                    ast::Expr::FuncCall {
                        ref id,
                        ref args,
                        span: ref expr_span,
                    },
            } => {
                let (func, mut func_allocator) =
                    func_call(id.clone(), args, expr_span.clone(), line_offsets, map, model, allocator)?;
                match &func.ret_value {
                    Some(ast::Expr::Simple(expr)) => write_expr_to(
                        WriteTo::Pat(pat.clone()),
                        expr.clone(),
                        ast.span.clone(),
                        should_track,
                        line_offsets,
                        map,
                        model,
                        allocator,
                        Some(&mut func_allocator),
                    )?,
                    _ => panic!(),
                }
            }
            AstKind::Let {
                pat: ast::Pattern::Simple(pat),
                expr:
                    ast::Expr::FuncCall {
                        ref id,
                        ref args,
                        span: ref expr_span,
                    },
            } => {
                let (func, func_allocator) =
                    func_call(id.clone(), args, expr_span.clone(), line_offsets, map, model, allocator)?;
                match &func.ret_value {
                    Some(ast::Expr::Simple(expr)) => write_expr_to(
                        WriteTo::Pat(pat.clone()),
                        expr.clone(),
                        ast.span.clone(),
                        should_track,
                        line_offsets,
                        map,
                        model,
                        allocator,
                        None,
                    )?,
                    _ => panic!(),
                }
            }
            AstKind::Let {
                pat: ast::Pattern::Simple(pat),
                expr: ast::Expr::Tuple { span, .. },
            } => {
                panic!("Tuples *must* be destructured")
            }
            AstKind::Let {
                pat: ast::Pattern::Tuple { span, .. },
                expr: ast::Expr::Simple(expr),
            } => {
                panic!("Simple expr is not a tuple")
            }
            AstKind::Drop(id) => {
                let ptr = try_or_throw!(allocator.ptr(id.name));
                let deallocated = try_or_throw!(model.deallocate(ptr));
                for ptr in deallocated {
                    allocator.dealloc(ptr, ast.span.clone());
                }
            }
            &AstKind::Update {
                ref id,
                is_exclusive,
                read,
                write,
            } => {
                let ptr = try_or_throw!(allocator.ptr(id.name));
                let res = if is_exclusive {
                    model.mark_exclusive(ptr)
                } else {
                    model.mark_shared(ptr)
                };
                try_or_throw!(res);
                let res = model.update_meta(ptr, |_| Permissions { read, write });
                try_or_throw!(res);
            }
            &AstKind::Write { ref id, is_exclusive } => {
                let ptr = try_or_throw!(allocator.ptr(id.name));
                assert!(try_or_throw!(model.info(ptr)).meta.write);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Write)
                };
                try_or_throw!(res);
            }
            &AstKind::Read { ref id, is_exclusive } => {
                let ptr = try_or_throw!(allocator.ptr(id.name));
                assert!(try_or_throw!(model.info(ptr)).meta.read);
                let res = if is_exclusive {
                    model.assert_exclusive(ptr)
                } else {
                    model.assert_shared(ptr, PermissionsFilter::Read)
                };
                try_or_throw!(res);
            }
        }
    }

    Ok(())
}

fn func_call<'m, 'a>(
    id: ast::Id<'a>,
    args: &[ast::SimpleExpr<'a>],
    span: Span,
    line_offsets: &[usize],
    map: FunctionMapRef<'m, 'a>,
    model: &mut State<'a>,
    allocator: &mut Allocator<'a>,
) -> Result<(&'m Function<'a>, Allocator<'a>), Box<dyn std::error::Error>> {
    mk_try_or_throw!($, todo!(), allocator, line_offsets);

    let mut map_ = &map;
    let func = loop {
        if let Some(func) = map_.inner.get(id.name) {
            break func
        }

        match map_.parent {
            Some(parent) => map_ = parent,
            None => try_or_throw!(Err(Error::NoFunction { func: id.name })),
        }
    };
    let map = FunctionMapRef {
        inner: &func.map,
        parent: Some(&map),
    };
    let instrs = &*func.instrs;

    assert_eq!(args.len(), func.args.len());

    for (farg, expr) in func.args.iter().zip(args.iter()) {
        // let ty = match farg.ty {
        //     Some(ref ty) => ty,
        //     None => continue,
        // };

        // let perm = Permissions {
        //     read: ty.read,
        //     write: ty.write,
        // };

        // let ptr = try_or_throw!(allocator.ptr(arg));
        // let info = try_or_throw!(model.info(ptr));
        // let is_exclusive = matches!(info.ptr_ty, memory_model::alias::PtrType::Exclusive);
        // if info.meta == perm && is_exclusive == ty.is_exclusive {
        //     continue
        // }

        // try_or_throw!(
        //     Err(Error::TypeMismatch {
        //         arg: ast::Arg {
        //             name: arg,
        //             span: span.clone(),
        //             ty: Some(ast::ArgTy {
        //                 span: span.clone(),
        //                 is_exclusive,
        //                 read: info.meta.read,
        //                 write: info.meta.write,
        //             })
        //         },
        //         farg: farg.clone(),
        //     }),
        //     span = span.clone()
        // );
    }

    let mut func_allocator = Allocator::default();

    for (farg, arg) in func.args.iter().zip(args.iter()) {
        let name = farg.id.name;
        let span = span.clone();
        write_expr_to(
            WriteTo::Temp(0),
            arg.clone(),
            arg.span(),
            false,
            line_offsets,
            map,
            model,
            allocator,
            None,
        )?;

        let source_ptr = try_or_throw!(allocator.ptr(0), span = span);

        let info = try_or_throw!(model.info(source_ptr), span = span);
        match info.ptr_ty {
            PtrType::Exclusive => try_or_throw!(allocator.rename_into(&mut func_allocator, 0, name)),
            PtrType::Shared => {
                let ptr = func_allocator.alloc(name);
                try_or_throw!(model.copy(ptr, source_ptr), span = span)
            }
        }
    }

    run(instrs, line_offsets, map, model, &mut func_allocator)?;

    Ok((func, func_allocator))
}

#[derive(Debug)]
enum WriteTo<'a> {
    Pat(ast::SimplePattern<'a>),
    Target(ast::Id<'a>),
    Temp(u32),
}

fn write_expr_to<'a>(
    to: WriteTo<'a>,
    expr: ast::SimpleExpr<'a>,
    span: Span,
    should_track: bool,
    line_offsets: &[usize],
    map: FunctionMapRef<'_, 'a>,
    model: &mut State<'a>,
    allocator: &mut Allocator<'a>,
    mut source_allocator: Option<&mut Allocator<'a>>,
) -> Result<(), Box<dyn std::error::Error>> {
    mk_try_or_throw!($, span, allocator, line_offsets);

    match expr {
        ast::SimpleExpr::Alloc { span, range } => {
            let (target, ptr) = match to {
                WriteTo::Target(target) | WriteTo::Pat(ast::SimplePattern::Ident(target)) => {
                    (Some(target.name), allocator.alloc(target.name))
                }
                WriteTo::Pat(ast::SimplePattern::Ignore(span)) => {
                    try_or_throw!(Err(Error::AllocNoBind { span }))
                }
                WriteTo::Temp(temp) => (None, allocator.alloc(temp)),
            };

            let (start, end) = match (range.start, range.end) {
                (Some(start), Some(end)) => (start, end),
                (_, _) => {
                    try_or_throw!(Err(Error::AllocRangeBoundsNotSpecied { span: span.clone() }))
                }
            };

            try_or_throw!(model.allocate(ptr, start..end));

            if should_track {
                eprintln!(
                    "\t{}: allocated `{}` as {:?}",
                    ShowSpan(&span, line_offsets),
                    target.unwrap_or("<unnamed>"),
                    ptr
                );
                if let Some(target) = target {
                    model.trackers.register(target, ptr, event_handler)
                }
            }

            Ok(())
        }
        ast::SimpleExpr::Move(source) => {
            let source_ptr = try_or_throw!(source_allocator.as_deref_mut().unwrap_or(allocator).ptr(source.name));
            source_allocator
                .as_deref_mut()
                .unwrap_or(allocator)
                .invalidated
                .insert(source.name, Invalidated {
                    span: span.clone(),
                    kind: InvalidKind::Moved,
                });

            let info = try_or_throw!(model.info(source_ptr));

            match info.ptr_ty {
                PtrType::Exclusive => match to {
                    WriteTo::Target(target) | WriteTo::Pat(ast::SimplePattern::Ident(target)) => {
                        let res = match source_allocator {
                            Some(source_allocator) => source_allocator.rename_into(allocator, source.name, target.name),
                            None => allocator.rename(source.name, target.name),
                        };
                        try_or_throw!(res, span = target.span);
                    }
                    WriteTo::Pat(ast::SimplePattern::Ignore(span)) => {
                        try_or_throw!(source_allocator.unwrap_or(allocator).remove(source.name), span = span)
                    }
                    WriteTo::Temp(temp) => {
                        let res = match source_allocator {
                            Some(source_allocator) => source_allocator.rename_into(allocator, source.name, temp),
                            None => allocator.rename(source.name, temp),
                        };

                        try_or_throw!(res);
                    }
                },
                PtrType::Shared => {
                    let ptr = match to {
                        WriteTo::Target(target) | WriteTo::Pat(ast::SimplePattern::Ident(target)) => {
                            allocator.alloc(target.name)
                        }
                        WriteTo::Pat(ast::SimplePattern::Ignore(span)) => {
                            try_or_throw!(source_allocator.unwrap_or(allocator).remove(source.name), span = span);
                            return Ok(())
                        }
                        WriteTo::Temp(temp) => allocator.alloc(temp),
                    };
                    try_or_throw!(model.copy(ptr, source_ptr))
                }
            }

            Ok(())
        }
        ast::SimpleExpr::Borrow {
            source: source_id,
            is_exclusive,
            range,
            read,
            write,
            span,
        } => borrow(
            match to {
                | WriteTo::Target(id) | WriteTo::Pat(ast::SimplePattern::Ident(id)) => Some(id.name.into()),
                WriteTo::Temp(temp) => Some(temp.into()),
                WriteTo::Pat(ast::SimplePattern::Ignore(_)) => None,
            },
            source_id,
            is_exclusive,
            range,
            read,
            write,
            span,
            should_track,
            line_offsets,
            map,
            model,
            allocator,
        ),
    }
}

fn borrow<'a>(
    target: Option<AllocId<'a>>,
    source_id: ast::Id<'a>,
    is_exclusive: bool,
    range: Option<Range<Option<u32>>>,
    read: bool,
    write: bool,
    span: Span,
    should_track: bool,
    line_offsets: &[usize],
    map: FunctionMapRef<'_, 'a>,
    model: &mut State<'a>,
    allocator: &mut Allocator<'a>,
) -> Result<(), Box<dyn std::error::Error>> {
    mk_try_or_throw!($, span, allocator, line_offsets);

    let source = try_or_throw!(allocator.ptr(source_id.name));
    let target_name = match target {
        Some(AllocId::Named(name)) => name,
        Some(AllocId::Temp(temp)) => "<temp>",
        None => "<ignored>",
    };
    let ptr = match target {
        Some(target) => allocator.alloc(target),
        None => Pointer::create(),
    };

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

    if should_track {
        eprintln!(
            "\t{}: {} borrow `{}` as {:?} from `{}` ({:?})",
            ShowSpan(&span, line_offsets),
            if is_exclusive { "exclusive" } else { "shared" },
            target_name,
            ptr,
            source_id.name,
            source,
        );
        model.trackers.register(target_name, ptr, event_handler);
    }

    Ok(())
}

fn event_handler(tag: &str, ptr: Pointer, event: Event) { println!("\t{:?}: {:?} ({})", event, ptr, tag) }

struct ShowSpan<'a>(pub &'a Span, pub &'a [usize]);
#[derive(Clone, Copy)]
struct DispSpan {
    line_start: usize,
    col_start: usize,
    line_end: usize,
    col_end: usize,
}

impl DispSpan {
    fn new(span: &Span, line_offsets: &[usize]) -> Self {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { DispSpan::new(self.0, self.1).fmt(f) }
}

impl fmt::Display for DispSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "span({}:{}..{}:{})",
            self.line_start, self.col_start, self.line_end, self.col_end
        )
    }
}
