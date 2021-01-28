use std::ops::Range;

use dec::{
    base::{
        error::{CaptureInput, Error, PResult, ParseError},
        ParseOnce, Tag,
    },
    branch::any,
    combinator::opt,
    map::{map, value},
    seq::{all, range, separated},
    tag::tag,
};

use crate::tokens::{Keyword, Symbol, Token, TokenKind};

type Span = Range<usize>;

#[derive(Debug, Clone)]
pub struct Ast<'i> {
    pub span: Span,
    pub attrs: Vec<Attribute<'i>>,
    pub kind: AstKind<'i>,
}

#[derive(Debug, Clone)]
pub enum AstKind<'i> {
    Allocate {
        name: &'i str,
        range: Range<Option<u32>>,
    },
    Move {
        name: &'i str,
        source: &'i str,
    },
    Read {
        name: &'i str,
        is_exclusive: bool,
    },
    Write {
        name: &'i str,
        is_exclusive: bool,
    },
    Update {
        name: &'i str,
        is_exclusive: bool,
        read: bool,
        write: bool,
    },
    Borrow {
        name: &'i str,
        source: &'i str,
        is_exclusive: bool,
        read: bool,
        write: bool,
        range: Option<Range<Option<u32>>>,
    },
    Drop {
        name: &'i str,
    },
    FuncDecl {
        name: &'i str,
        args: Vec<Arg<'i>>,
        instr: Vec<Ast<'i>>,
        ret_value: Option<Vec<(Span, &'i str)>>,
        ret_ty: Option<Vec<Option<ArgTy>>>,
    },
    FuncCall {
        name: &'i str,
        args: Vec<(Range<usize>, &'i str)>,
    },
}

enum Expr<'a> {
    Range(Range<Option<u32>>),
    Move {
        source: &'a str,
    },
    Borrow {
        source: &'a str,
        is_exclusive: bool,
        read: bool,
        write: bool,
        range: Option<Range<Option<u32>>>,
    },
}

#[derive(Debug, Clone)]
pub struct Attribute<'a> {
    pub span: Span,
    pub path: &'a str,
}

#[derive(Debug, Clone)]
pub struct Arg<'a> {
    pub span: Range<usize>,
    pub name: &'a str,
    pub ty: Option<ArgTy>,
}

#[derive(Debug, Clone)]
pub struct ArgTy {
    pub span: Span,
    pub is_exclusive: bool,
    pub read: bool,
    pub write: bool,
}

struct Perm {
    span: Option<Span>,
    read: bool,
    write: bool,
}

struct TagIdent;
const IDENT: dec::tag::Tag<TagIdent> = dec::tag::Tag(TagIdent);
struct TagNum;
const NUM: dec::tag::Tag<TagNum> = dec::tag::Tag(TagNum);

impl<'t, 'i> Tag<&'t [Token<'i>]> for TagIdent {
    type Output = (Span, &'i str);

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token {
                kind: TokenKind::Ident(ident),
                ref span,
            }, ref input @ ..] => Ok((input, (span.clone(), ident))),
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
        parse_func_call,
        parse_update,
        parse_func_decl,
        //
    ))
    .parse_once(input)?;

    ast.attrs = attrs;

    Ok((input, ast))
}

fn parse_drop<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_drop), (_, name), (end, _sym_semi))) =
        all((tag(Keyword::Drop), IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Drop { name },
    }))
}

fn parse_read<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_read), (_, is_exclusive), (_, name), (end, _sym_semi))) =
        all((tag(Keyword::Read), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Read { name, is_exclusive },
    }))
}

fn parse_write<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_write), (_, is_exclusive), (_, name), (end, _sym_semi))) =
        all((tag(Keyword::Write), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::Write { name, is_exclusive },
    }))
}

fn parse_update<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (
        input,
        ((start, name), _sym_arrow, _sym_borrow, (_, is_exclusive), Perm { span: _, read, write }, (end, _sym_semi)),
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
        span: start.start..end.end,
        kind: AstKind::Update {
            name,
            is_exclusive,
            read,
            write,
        },
    }))
}

fn parse_decl_var<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, _kw_let), (_, name), _sym_eq, expr, (end, _sym_semi))) = all((
        tag(Keyword::Let),
        IDENT,
        tag(Symbol::Equal),
        parse_expr,
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;
    let kind = match expr {
        Expr::Range(range) => AstKind::Allocate { name, range },
        Expr::Move { source } => AstKind::Move { name, source },
        Expr::Borrow {
            source,
            is_exclusive,
            read,
            write,
            range,
        } => AstKind::Borrow {
            name,
            source,
            is_exclusive,
            read,
            write,
            range,
        },
    };

    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind,
    }))
}

fn parse_expr<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Expr<'i>, E> {
    any((
        map(parse_range, Expr::Range),
        map(
            all((
                map(IDENT, |(_, name)| name),
                opt(all((
                    tag(Symbol::Borrow),
                    map(parse_modifier, |(_, is_exclusive)| is_exclusive),
                    parse_permissions,
                    opt(parse_range),
                ))),
            )),
            |(source, rest)| match rest {
                None => Expr::Move { source },
                Some((_sym_borrow, is_exclusive, Perm { span: _, read, write }, range)) => Expr::Borrow {
                    source,
                    is_exclusive,
                    read,
                    write,
                    range,
                },
            },
        ),
    ))
    .parse_once(input)
}

fn parse_func_call<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, ((start, name), _sym_open_paren, args, _sym_close_paren, (end, _sym_semi))) = all((
        IDENT,
        tag(Symbol::OpenParen),
        separated(.., tag(Symbol::Comma), IDENT),
        tag(Symbol::CloseParen),
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;

    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::FuncCall { name, args },
    }))
}

fn parse_func_decl<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (
        input,
        ((start, _kw_fn), (_, name), args, ret_ty, _sym_open_curly, (instr, ret_value), (end, _sym_close_curly)),
    ) = all((
        tag(Keyword::Fn),
        IDENT,
        dec::seq::mid(
            tag(Symbol::OpenParen),
            separated(.., tag(Symbol::Comma), parse_arg),
            tag(Symbol::CloseParen),
        ),
        opt(parse_ret_ty),
        tag(Symbol::OpenCurly),
        all((range(.., parse_ast), opt(parse_ret_value))),
        tag(Symbol::CloseCurly),
    ))
    .parse_once(input)?;

    Ok((input, Ast {
        attrs: Vec::new(),
        span: start.start..end.end,
        kind: AstKind::FuncDecl {
            name,
            args,
            instr,
            ret_value,
            ret_ty,
        },
    }))
}

fn parse_arg<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Arg<'i>, E> {
    let (input, ((span, name), ty)) =
        all((IDENT, opt(dec::seq::snd(tag(Symbol::Colon), parse_arg_ty)))).parse_once(input)?;

    Ok((input, Arg { span, name, ty }))
}

fn parse_arg_ty<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], ArgTy, E> {
    let (input, ((start, _sym_borrow), (span, is_exclusive), Perm { span: end, read, write })) =
        all((tag(Symbol::Borrow), parse_modifier, parse_permissions)).parse_once(input)?;

    let end = end.unwrap_or(span);

    Ok((input, ArgTy {
        span: start.start..end.end,
        is_exclusive,
        read,
        write,
    }))
}

fn parse_ret_ty<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Vec<Option<ArgTy>>, E> {
    dec::seq::snd(
        tag(Symbol::Arrow),
        dec::seq::mid(
            tag(Symbol::OpenParen),
            dec::seq::fst(
                separated(
                    ..,
                    tag(Symbol::Comma),
                    any((map(parse_arg_ty, Some), value(None, tag(TokenKind::Ident("_"))))),
                ),
                opt(tag(Symbol::Comma)),
            ),
            tag(Symbol::CloseParen),
        ),
    )
    .parse_once(input)
}

fn parse_ret_value<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Vec<(Span, &'i str)>, E> {
    dec::seq::mid(
        tag(Symbol::OpenParen),
        dec::seq::fst(separated(.., tag(Symbol::Comma), IDENT), opt(tag(Symbol::Comma))),
        tag(Symbol::CloseParen),
    )
    .parse_once(input)
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

fn parse_permissions<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Perm, E> {
    let (input, is_write) = opt(any((tag(Keyword::Read), tag(Keyword::Write)))).parse_once(input)?;
    match is_write {
        None => Ok((input, Perm {
            span: None,
            read: false,
            write: false,
        })),
        Some((start, Keyword::Write)) => {
            let (input, read) = opt(tag(Keyword::Read)).parse_once(input)?;
            if let Some((end, _kw_read)) = read {
                Ok((input, Perm {
                    span: Some(start.start..end.end),
                    read: true,
                    write: true,
                }))
            } else {
                Ok((input, Perm {
                    span: Some(start),
                    read: false,
                    write: true,
                }))
            }
        }
        Some((start, Keyword::Read)) => {
            let (input, write) = opt(tag(Keyword::Write)).parse_once(input)?;
            if let Some((end, _kw_write)) = write {
                Ok((input, Perm {
                    span: Some(start.start..end.end),
                    read: true,
                    write: true,
                }))
            } else {
                Ok((input, Perm {
                    span: Some(start),
                    read: true,
                    write: false,
                }))
            }
        }
        _ => unreachable!(),
    }
}

pub fn parse_range<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Range<Option<u32>>, E> {
    let (input, (start, _sym_dot2, end)) = all((opt(NUM), tag(Symbol::Dot2), opt(NUM))).parse_once(input)?;
    Ok((input, start.map(|(_, start)| start)..end.map(|(_, end)| end)))
}

pub fn parse_attr<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Attribute<'i>, E> {
    let (input, ((start, _sym_pound), _sym_open_square, (_, path), (end, _syn_close_square))) = all((
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
