#[derive(Debug)]
pub enum Ast<'i> {
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

struct Perm {
    read: bool,
    write: bool,
}

use std::ops::Range;

use dec::{
    base::{
        error::{CaptureInput, Error, PResult, ParseError},
        ParseOnce, Tag,
    },
    branch::any,
    combinator::opt,
    map::{map, value},
    seq::all,
    tag::tag,
};

use crate::tokens::{Keyword, Symbol, Token};

struct TagIdent;
const IDENT: dec::tag::Tag<TagIdent> = dec::tag::Tag(TagIdent);
struct TagNum;
const NUM: dec::tag::Tag<TagNum> = dec::tag::Tag(TagNum);

impl<'t, 'i> Tag<&'t [Token<'i>]> for TagIdent {
    type Output = &'i str;

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match input {
            [Token::Ident(ident), input @ ..] => Ok((input, ident)),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for TagNum {
    type Output = u32;

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token::Number(num), ref input @ ..] => Ok((input, num)),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for Symbol {
    type Output = Self;

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token::Symbol(symbol), ref input @ ..] if *self == symbol => Ok((input, symbol)),
            _ => Err(Error::Error(CaptureInput(input))),
        }
    }
}

impl<'t, 'i> Tag<&'t [Token<'i>]> for Keyword {
    type Output = Self;

    fn parse_tag(
        &self,
        input: &'t [Token<'i>],
    ) -> PResult<&'t [Token<'i>], Self::Output, CaptureInput<&'t [Token<'i>]>, core::convert::Infallible> {
        match *input {
            [Token::Keyword(keyword), ref input @ ..] if *self == keyword => Ok((input, keyword)),
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
    any((
        parse_decl_var,
        parse_drop,
        parse_read,
        parse_write,
        parse_update,
        //
    ))
    .parse_once(input)
}

fn parse_drop<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (_kw_drop, name, _sym_semi)) =
        all((tag(Keyword::Drop), IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast::Drop { name }))
}

fn parse_read<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (_kw_read, is_exclusive, name, _sym_semi)) =
        all((tag(Keyword::Read), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast::Read { name, is_exclusive }))
}

fn parse_write<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (_kw_write, is_exclusive, name, _sym_semi)) =
        all((tag(Keyword::Write), parse_modifier, IDENT, tag(Symbol::SemiColon))).parse_once(input)?;
    Ok((input, Ast::Write { name, is_exclusive }))
}

fn parse_update<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (name, _sym_arrow, _sym_borrow, is_exclusive, Perm { read, write }, _sym_semi)) = all((
        IDENT,
        tag(Symbol::Arrow),
        tag(Symbol::Borrow),
        parse_modifier,
        parse_permissions,
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;
    Ok((input, Ast::Update {
        name,
        is_exclusive,
        read,
        write,
    }))
}

fn parse_decl_var<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Ast<'i>, E> {
    let (input, (_kw_let, name, _sym_eq, expr, _sym_semi)) = all((
        tag(Keyword::Let),
        IDENT,
        tag(Symbol::Equal),
        parse_expr,
        tag(Symbol::SemiColon),
    ))
    .parse_once(input)?;
    let ast = match expr {
        Expr::Range(range) => Ast::Allocate { name, range },
        Expr::Move { source } => Ast::Move { name, source },
        Expr::Borrow {
            source,
            is_exclusive,
            read,
            write,
            range,
        } => Ast::Borrow {
            name,
            source,
            is_exclusive,
            read,
            write,
            range,
        },
    };

    Ok((input, ast))
}

fn parse_expr<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], Expr<'i>, E> {
    any((
        map(parse_range, Expr::Range),
        map(
            all((
                IDENT,
                opt(all((
                    tag(Symbol::Borrow),
                    parse_modifier,
                    parse_permissions,
                    opt(parse_range),
                ))),
            )),
            |(source, rest)| match rest {
                None => Expr::Move { source },
                Some((_sym_borrow, is_exclusive, Perm { read, write }, range)) => Expr::Borrow {
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

fn parse_modifier<'i, 't, E: ParseError<&'t [Token<'i>]>>(input: &'t [Token<'i>]) -> PResult<&'t [Token<'i>], bool, E> {
    any((value(false, tag(Keyword::Shared)), value(true, tag(Keyword::Exclusive)))).parse_once(input)
}

fn parse_permissions<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Perm, E> {
    let (input, is_write) = opt(any((
        value(false, tag(Keyword::Read)),
        value(true, tag(Keyword::Write)),
    )))
    .parse_once(input)?;
    match is_write {
        None => Ok((input, Perm {
            read: false,
            write: false,
        })),
        Some(true) => {
            let (input, read) = opt(tag(Keyword::Read)).parse_once(input)?;
            Ok((input, Perm {
                read: read.is_some(),
                write: true,
            }))
        }
        Some(false) => {
            let (input, write) = opt(tag(Keyword::Write)).parse_once(input)?;
            Ok((input, Perm {
                read: true,
                write: write.is_some(),
            }))
        }
    }
}

pub fn parse_range<'i, 't, E: ParseError<&'t [Token<'i>]>>(
    input: &'t [Token<'i>],
) -> PResult<&'t [Token<'i>], Range<Option<u32>>, E> {
    let (input, (start, _sym_dot2, end)) = all((opt(NUM), tag(Symbol::Dot2), opt(NUM))).parse_once(input)?;
    Ok((input, start..end))
}
