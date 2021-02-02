use std::ops::Range;

use dec::{
    base::{
        error::{CaptureInput, Error, ErrorKind, PResult, ParseError},
        ParseMut, ParseOnce, Tag,
    },
    branch::any,
    combinator::opt,
    map::map,
    seq::{all, range, separated},
    tag::tag,
};

use crate::tokens::{Keyword, Symbol, Token, TokenKind};

pub type Span = Range<usize>;

#[derive(Debug, Clone)]
pub struct Ast<'i> {
    pub span: Span,
    pub attrs: Vec<Attribute<'i>>,
    pub kind: AstKind<'i>,
}

#[derive(Debug, Clone)]
pub enum AstKind<'i> {
    Drop(Id<'i>),
    Let {
        pat: Pattern<'i>,
        expr: Expr<'i>,
    },
    Read {
        id: Id<'i>,
        is_exclusive: bool,
    },
    Write {
        id: Id<'i>,
        is_exclusive: bool,
    },
    Update {
        id: Id<'i>,
        is_exclusive: bool,
        read: bool,
        write: bool,
        valid: bool,
    },
    FuncDecl {
        id: Id<'i>,
        args: Vec<Arg<'i>>,
        instrs: Vec<Ast<'i>>,
        ret_value: Option<Expr<'i>>,
        ret_ty: Option<Type>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Id<'a> {
    pub name: &'a str,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum SimplePattern<'a> {
    Ident(Id<'a>),
    Ignore(Span),
}

#[derive(Debug, Clone)]
pub enum Pattern<'a> {
    Simple(SimplePattern<'a>),
    Tuple { pats: Vec<SimplePattern<'a>>, span: Span },
}

#[derive(Debug, Clone)]
pub enum SimpleExpr<'a> {
    Move(Id<'a>),
    Alloc {
        span: Span,
        range: Range<Option<u32>>,
    },
    Borrow {
        span: Span,
        source: Id<'a>,
        ptr_ty: PointerTy,
        range: Option<Range<Option<u32>>>,
    },
}

#[derive(Debug, Clone)]
pub enum Expr<'a> {
    Simple(SimpleExpr<'a>),
    Tuple {
        exprs: Vec<SimpleExpr<'a>>,
        span: Span,
    },
    FuncCall {
        span: Span,
        id: Id<'a>,
        args: Vec<SimpleExpr<'a>>,
    },
}

#[derive(Debug, Clone)]
pub struct Attribute<'a> {
    pub span: Span,
    pub path: Id<'a>,
}

#[derive(Debug, Clone)]
pub struct Arg<'a> {
    pub id: Id<'a>,
    pub ty: PointerTy<Option<bool>>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Pointer(PointerTy<Option<bool>>),
    Tuple {
        span: Span,
        types: Vec<PointerTy<Option<bool>>>,
    },
}

struct Perm {
    span: Option<Span>,
    read: bool,
    write: bool,
    valid: bool,
}

impl SimplePattern<'_> {
    pub fn span(&self) -> Span {
        use SimplePattern::*;
        match self {
            Ignore(span) | Ident(Id { span, .. }) => span.clone(),
        }
    }
}

impl Pattern<'_> {
    pub fn span(&self) -> Span {
        use Pattern::*;
        match self {
            Simple(pat) => pat.span(),
            Tuple { span, .. } => span.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointerTy<R = bool> {
    pub span: Span,
    pub is_exclusive: bool,
    pub read: R,
    pub write: R,
    pub valid: R,
}

impl<'i> SimpleExpr<'i> {
    pub fn span(&self) -> Span {
        use SimpleExpr::*;
        match self {
            Alloc { span, .. }
            | Move(Id { span, .. })
            | Borrow {
                ptr_ty: PointerTy { span, .. },
                ..
            } => span.clone(),
        }
    }
}

impl<'i> Expr<'i> {
    pub fn span(&self) -> Span {
        use Expr::*;
        match self {
            | Simple(expr) => expr.span(),
            Tuple { span, .. } | FuncCall { span, .. } => span.clone(),
        }
    }
}

impl Type {
    pub fn span(&self) -> Span {
        use Type::*;
        match self {
            Pointer(PointerTy { span, .. }) | Tuple { span, .. } => span.clone(),
        }
    }
}

struct TagIdent;
const IDENT: dec::tag::Tag<TagIdent> = dec::tag::Tag(TagIdent);
struct TagNum;
const NUM: dec::tag::Tag<TagNum> = dec::tag::Tag(TagNum);

impl<'t, 'i> Tag<&'t [Token<'i>]> for TagIdent {
    type Output = Id<'i>;

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token {
                kind: TokenKind::Ident(name),
                ref span,
            }, ref input @ ..] => Ok((input, Id {
                span: span.clone(),
                name,
            })),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for TagNum {
    type Output = (Span, u32);

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token {
                kind: TokenKind::Number(num),
                ref span,
            }, ref input @ ..] => Ok((input, (span.clone(), num))),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for TokenKind<'i> {
    type Output = (Span, Self);

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token { kind, ref span }, ref input @ ..] if *self == kind => Ok((input, (span.clone(), kind))),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for Symbol {
    type Output = (Span, Self);

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token {
                kind: TokenKind::Symbol(symbol),
                ref span,
            }, ref input @ ..]
                if *self == symbol =>
            {
                Ok((input, (span.clone(), symbol)))
            }
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for Keyword {
    type Output = (Span, Self);

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token {
                kind: TokenKind::Keyword(keyword),
                ref span,
            }, ref input @ ..]
                if *self == keyword =>
            {
                Ok((input, (span.clone(), keyword)))
            }
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

pub fn parse<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Vec<Ast<'i>>, E> {
    dec::seq::range(.., parse_ast).parse_once(input)
}

pub fn parse_ast<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, attrs) = range(.., parse_attr).parse_once(input)?;

    let (input, mut ast) = any((
        parse_decl_var,
        parse_drop,
        parse_read,
        parse_write,
        parse_update,
        parse_func_decl,
        parse_func_call,
        //
    ))
    .parse_once(input)?;

    ast.attrs = attrs;

    Ok((input, ast))
}

fn parse_drop<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_drop), id, (end, _sym_semi))) =
        all((tag(Keyword::Drop), IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Drop(id),
    }))
}

fn parse_read<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_read), (_, is_exclusive), id, (end, _sym_semi))) =
        all((tag(Keyword::Read), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Read { id, is_exclusive },
    }))
}

fn parse_write<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_write), (_, is_exclusive), id, (end, _sym_semi))) =
        all((tag(Keyword::Write), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Write { id, is_exclusive },
    }))
}

fn parse_update<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (
        input,
        (
            id,
            _sym_arrow,
            _sym_borrow,
            (_, is_exclusive),
            Perm {
                span: _,
                read,
                write,
                valid,
            },
            (end, _sym_semi),
        ),
    ) = all((
        IDENT,
        tag(Symbol::Arrow),
        tag(Symbol::Borrow),
        parse_modifier,
        parse_permissions,
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: id.span.start..end.end,
        kind: AstKind::Update {
            id,
            is_exclusive,
            read,
            write,
            valid,
        },
    }))
}

fn parse_decl_var<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_let), pat, _sym_eq, expr, (end, _sym_semi))) = all((
        tag(Keyword::Let),
        parse_pattern,
        tag(Symbol::Equal),
        parse_expr,
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;

    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Let { pat, expr },
    }))
}

#[allow(clippy::type_complexity)]
fn parse_generic_tuple<'i: 't, 't, E: ParseError<&'t [Token<'i>]>, P: ParseMut<&'t [Token<'i>], E>>(
    input: &'t [Token<'i>],
    parser: P,
) -> PResult<&'t [Token<'i>], (Span, Vec<P::Output>), E> {
    let (input, ((start, _sym_open_paren), items, (end, _sym_close_paren))) = all((
        tag(Symbol::OpenParen),
        dec::seq::fst(
            dec::seq::separated(.., tag(Symbol::Comma), parser),
            opt(tag(Symbol::Comma)),
        ),
        tag(Symbol::CloseParen),
    ))
    .parse_once(input)?;

    Ok((input, (start.start..end.end, items)))
}

fn parse_pattern<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Pattern<'i>, E> {
    fn parse_simple<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], SimplePattern<'i>, E> {
        any((
            map(IDENT, SimplePattern::Ident),
            map(tag(Keyword::Ignore), |(span, _kw_ignore)| SimplePattern::Ignore(span)),
        ))
        .parse_once(input)
    }

    fn parse_tuple<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Pattern<'i>, E> {
        let (input, (span, pats)) = parse_generic_tuple(input, parse_simple)?;
        Ok((input, Pattern::Tuple { pats, span }))
    }

    any((map(parse_simple, Pattern::Simple), parse_tuple)).parse_once(input)
}

fn parse_borrow<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], SimpleExpr<'i>, E> {
    let (input, (source, rest)) = all((
        IDENT,
        opt(all((
            tag(Symbol::Borrow),
            parse_modifier,
            parse_permissions,
            opt(parse_range),
        ))),
    ))
    .parse_once(input)?;

    let expr = match rest {
        None => SimpleExpr::Move(source),
        Some((
            (borrow_start, _sym_borrow),
            (exc_span, is_exclusive),
            Perm {
                span: perm_span,
                read,
                write,
                valid,
            },
            range,
        )) => {
            let ty_end = perm_span.unwrap_or(exc_span);
            let (end, range) = match range {
                Some((end, range)) => (end, Some(range)),
                None => (ty_end.clone(), None),
            };

            SimpleExpr::Borrow {
                span: source.span.start..end.end,
                source,
                range,
                ptr_ty: PointerTy {
                    span: borrow_start.start..ty_end.end,
                    is_exclusive,
                    read,
                    write,
                    valid,
                },
            }
        }
    };

    Ok((input, expr))
}

fn parse_new<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], SimpleExpr<'i>, E> {
    let (input, ((start, _kw_new), (end, range))) = all((tag(Keyword::New), parse_range)).parse_once(input)?;

    Ok((input, SimpleExpr::Alloc {
        span: start.start..end.end,
        range,
    }))
}

fn parse_simple<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], SimpleExpr<'i>, E> {
    any((parse_borrow, parse_new)).parse_once(input)
}

fn parse_func_call_expr<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Expr<'i>, E> {
    let (input, (id, (end, args))) =
        all((IDENT, |input| parse_generic_tuple(input, parse_simple))).parse_once(input)?;

    Ok((input, Expr::FuncCall {
        span: id.span.start..end.end,
        id,
        args,
    }))
}

fn parse_expr<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Expr<'i>, E> {
    fn parse_tuple<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Expr<'i>, E> {
        let (input, (span, exprs)) = parse_generic_tuple(input, parse_simple)?;
        Ok((input, Expr::Tuple { exprs, span }))
    }

    any((parse_func_call_expr, map(parse_simple, Expr::Simple), parse_tuple)).parse_once(input)
}

fn parse_func_call<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (id, (_, args), (end, _sym_semi))) = all((
        IDENT,
        |input| parse_generic_tuple(input, parse_simple),
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;

    let span = id.span.start..end.end;

    Ok((input, Ast {
        attrs: Vec::new(),
        span: span.clone(),
        kind: AstKind::Let {
            pat: Pattern::Simple(SimplePattern::Ignore(0..0)),
            expr: Expr::FuncCall { span, id, args },
        },
    }))
}

fn parse_func_decl<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_fn), id, args, ret_ty, _sym_open_curly, (instrs, ret_value), (end, _sym_close_curly))) =
        all((
            tag(Keyword::Fn),
            IDENT,
            dec::seq::mid(
                tag(Symbol::OpenParen),
                opt(dec::seq::fst(
                    separated(.., tag(Symbol::Comma), parse_arg),
                    opt(tag(Symbol::Comma)),
                )),
                tag(Symbol::CloseParen),
            ),
            opt(dec::seq::snd(tag(Symbol::Arrow), parse_ty)),
            tag(Symbol::OpenCurly),
            all((range(.., parse_ast), opt(parse_expr))),
            tag(Symbol::CloseCurly),
        ))
        .parse_once(input)?;

    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::FuncDecl {
            id,
            args: args.unwrap_or_else(Vec::new),
            instrs,
            ret_value,
            ret_ty,
        },
    }))
}

fn parse_arg<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Arg<'i>, E> {
    let old_input = input;
    let (input, (id, ty)) = all((IDENT, dec::seq::snd(tag(Symbol::Colon), parse_ty))).parse_once(input)?;

    match ty {
        Type::Pointer(ty) => Ok((input, Arg { id, ty })),
        Type::Tuple { .. } => Err(Error::Error(E::from_input_kind(
            old_input,
            ErrorKind::Custom("tuple types not allows for arguments"),
        ))),
    }
}

fn parse_ty<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Type, E> {
    #[allow(clippy::type_complexity, clippy::unnecessary_wraps)]
    fn parse_permissions_ty<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], (Option<Span>, Option<bool>, Option<bool>, Option<bool>), E> {
        let temp_input = &input[..input.len().min(3)];
        let (found_ignore, pos) = temp_input
            .iter()
            .position(|tok| tok.kind == TokenKind::Keyword(Keyword::Ignore))
            .map_or((false, temp_input.len()), |pos| (true, pos));
        let pos = temp_input.len().min(pos);
        let temp_input = &temp_input[..pos];

        let (temp_input, (read, write, valid)) =
            dec::branch::any_set((tag(Keyword::Read), tag(Keyword::Write), tag(Keyword::Valid)))
                .parse_once(temp_input)?;
        let input = &input[pos - temp_input.len()..];

        let default = [Some(false), None][usize::from(found_ignore)];

        let (span, read) = match read {
            Some((span, _)) => (Some(span), Some(true)),
            None => (None, default),
        };

        let (span, write) = match write {
            Some((s, _)) => (Some(merge_span(s, span)), Some(true)),
            None => (None, default),
        };

        let (span, valid) = match valid {
            Some((s, _)) => (Some(merge_span(s, span)), Some(true)),
            None => (None, default),
        };

        Ok((input, (span, read, write, valid)))
    }

    fn parse_pointer_ty<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], PointerTy<Option<bool>>, E> {
        let (input, ((start, _sym_borrow), (span, is_exclusive), (end, read, write, valid))) =
            all((tag(Symbol::Borrow), parse_modifier, parse_permissions_ty)).parse_once(input)?;

        let end = end.unwrap_or(span);

        Ok((input, PointerTy {
            span: start.start..end.end,
            is_exclusive,
            read,
            write,
            valid,
        }))
    }

    fn parse_tuple<'i, 't, E: ParseError<&'t [Token<'i>]>>(
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Type, E> {
        let (input, (span, types)) = parse_generic_tuple(input, parse_pointer_ty)?;
        Ok((input, Type::Tuple { types, span }))
    }

    any((map(parse_pointer_ty, Type::Pointer), parse_tuple)).parse_once(input)
}

fn parse_modifier<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], (Span, bool), E> {
    match any((tag(Keyword::Shared), tag(Keyword::Exclusive))).parse_once(input)? {
        (input, (span, Keyword::Shared)) => Ok((input, (span, false))),
        (input, (span, Keyword::Exclusive)) => Ok((input, (span, true))),
        _ => unreachable!(),
    }
}

fn merge_span(span: Span, other: Option<Span>) -> Span {
    match other {
        Some(other) => span.start.min(other.start)..span.end.max(other.end),
        None => span,
    }
}

#[allow(clippy::unnecessary_wraps)]
fn parse_permissions<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Perm, E> {
    let (input, (read, write, valid)) =
        dec::branch::any_set((tag(Keyword::Read), tag(Keyword::Write), tag(Keyword::Valid))).parse_once(input)?;

    let (span, read) = match read {
        Some((span, _)) => (Some(span), true),
        None => (None, false),
    };

    let (span, write) = match write {
        Some((s, _)) => (Some(merge_span(s, span)), true),
        None => (None, false),
    };

    let (span, valid) = match valid {
        Some((s, _)) => (Some(merge_span(s, span)), true),
        None => (None, false),
    };

    Ok((input, Perm {
        span,
        read,
        write,
        valid,
    }))
}

#[allow(clippy::type_complexity)]
pub fn parse_range<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], (Span, Range<Option<u32>>), E> {
    let (input, (start, (span, _sym_dot2), end)) = all((opt(NUM), tag(Symbol::Dot2), opt(NUM))).parse_once(input)?;
    match (start, end) {
        (Some((start, s)), Some((end, e))) => Ok((input, (start.start..end.end, Some(s)..Some(e)))),
        (None, Some((end, e))) => Ok((input, (span.start..end.end, None..Some(e)))),
        (Some((start, s)), None) => Ok((input, (start.start..span.end, Some(s)..None))),
        (None, None) => Ok((input, (span, None..None))),
    }
}

pub fn parse_attr<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Attribute<'i>, E> {
    let (input, ((start, _sym_pound), _sym_open_square, path, (end, _syn_close_square))) = all((
        tag(Symbol::Pound),
        tag(Symbol::OpenSquare),
        IDENT,
        tag(Symbol::CloseSquare),
    ))
    .parse_once(input)?;
    Ok((input, Attribute {
        span: start.start..end.end,
        path,
    }))
}
