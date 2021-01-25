use dec::base::{
    error::{Error, ErrorKind, PResult, ParseError},
    indexed::Indexed,
    InputSplit, ParseOnce,
};

use std::ops::Range;

type Input<'a> = Indexed<&'a str, usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token<'a> {
    pub span: Range<usize>,
    pub kind: TokenKind<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind<'a> {
    Ident(&'a str),
    Number(u32),
    Symbol(Symbol),
    Keyword(Keyword),
    Whitespace(Whitespace, &'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    OpenParen,
    CloseParen,
    Dot2,
    SemiColon,
    Equal,
    Borrow,
    Arrow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Whitespace {
    LineComment,
    BlockComment,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Let,
    Shared,
    Exclusive,
    Read,
    Write,
    Drop,
}

pub fn parse<'a, E: ParseError<Input<'a>>>(input: Input<'a>) -> PResult<Input<'a>, Vec<Token<'_>>, E> {
    dec::seq::range(.., parse_token).parse_once(input)
}

pub trait SplitOnce: Sized {
    fn split_with<F: FnMut(char) -> bool>(self, splitter: F) -> (Self, Self);
}

impl SplitOnce for &str {
    fn split_with<F: FnMut(char) -> bool>(self, mut splitter: F) -> (Self, Self) {
        self.char_indices()
            .find_map(|(pos, c)| if splitter(c) { Some(pos) } else { None })
            .map_or((self, ""), |pos| self.split_at(pos))
    }
}

pub fn parse_token<'a, E: ParseError<Input<'a>>>(input: Input<'a>) -> PResult<Input<'a>, Token<'_>, E> {
    match parse_token_type(input.inner()) {
        Ok((new_input, kind)) => {
            let offset = input.len() - new_input.len();
            let start = input.pos();
            let input = input.advance(offset).ok().unwrap();
            let end = input.pos();
            Ok((input, Token { span: start..end, kind }))
        }
        Err(Error::Error((new_input, kind))) => {
            let offset = input.len() - new_input.len();
            let input = input.advance(offset).ok().unwrap();
            Err(Error::Error(E::from_input_kind(input, kind)))
        }
        Err(Error::Failure(err)) => match err {},
    }
}

#[allow(clippy::or_fun_call)]
fn parse_token_type(mut input: &str) -> PResult<&str, TokenKind<'_>> {
    loop {
        input = input.trim_start();

        if let Some(new_input) = input.strip_prefix("//") {
            let end = memchr::memchr(b'\n', new_input.as_bytes()).unwrap_or(new_input.len());
            input = &new_input[end..];
        } else {
            break
        }
    }

    match input.split_with(|c: char| !c.is_ascii_digit()) {
        ("", _) => (),
        (number, input) => return Ok((input, TokenKind::Number(number.parse().unwrap()))),
    }

    match input.split_with(|c: char| !c.is_alphanumeric() && c != '_') {
        ("", _) => (),
        (ident, input) => {
            let tok = match ident {
                "let" => TokenKind::Keyword(Keyword::Let),
                "shr" => TokenKind::Keyword(Keyword::Shared),
                "exc" => TokenKind::Keyword(Keyword::Exclusive),
                "read" => TokenKind::Keyword(Keyword::Read),
                "write" => TokenKind::Keyword(Keyword::Write),
                "drop" => TokenKind::Keyword(Keyword::Drop),
                _ => TokenKind::Ident(ident),
            };

            return Ok((input, tok))
        }
    }

    if let Some(input) = input.strip_prefix("..") {
        return Ok((input, TokenKind::Symbol(Symbol::Dot2)))
    }

    if let Some(input) = input.strip_prefix("->") {
        return Ok((input, TokenKind::Symbol(Symbol::Arrow)))
    }

    match input.get(0..1) {
        Some("(") => Ok((&input[1..], TokenKind::Symbol(Symbol::OpenParen))),
        Some(")") => Ok((&input[1..], TokenKind::Symbol(Symbol::CloseParen))),
        Some(";") => Ok((&input[1..], TokenKind::Symbol(Symbol::SemiColon))),
        Some("=") => Ok((&input[1..], TokenKind::Symbol(Symbol::Equal))),
        Some("@") => Ok((&input[1..], TokenKind::Symbol(Symbol::Borrow))),
        _ => Err(Error::Error((input, ErrorKind::Custom("no token found")))),
    }
}
