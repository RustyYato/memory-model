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
        Ok((new_input, (ws_offset, kind))) => {
            let offset = input.len() - new_input.len();
            let start = input.pos() + ws_offset;
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
fn parse_token_type(mut input: &str) -> PResult<&str, (usize, TokenKind<'_>)> {
    let offset = input.len();
    loop {
        input = input.trim_start();

        if let Some(new_input) = input.strip_prefix("//") {
            let end = memchr::memchr(b'\n', new_input.as_bytes()).unwrap_or(new_input.len());
            input = &new_input[end..];
            continue
        }

        if let Some(stripped_input) = input.strip_prefix("/*") {
            enum State {
                PreviousSlash(usize),
                PreviousAsterix(usize),
                New,
            }

            let binput = stripped_input.as_bytes();
            let comment_segment = memchr::memchr2_iter(b'/', b'*', binput)
                .try_fold((0_u32, State::New), |(stack, state), pos| {
                    Ok(match (binput[pos], state) {
                        (b'/', State::PreviousAsterix(last_pos)) if last_pos + 1 == pos => {
                            if let Some(stack) = stack.checked_sub(1) {
                                (stack, State::New)
                            } else {
                                return Err(pos + 1)
                            }
                        }
                        (b'/', _) => (stack, State::PreviousSlash(pos)),
                        (b'*', State::PreviousSlash(last_pos)) if last_pos + 1 == pos => (stack + 1, State::New),
                        (b'*', _) => (stack, State::PreviousAsterix(pos)),
                        _ => unreachable!(),
                    })
                })
                .err();
            input = match comment_segment {
                Some(pos) => &stripped_input[pos..],
                None => "",
            };
            continue
        }

        break
    }

    let offset = offset - input.len();

    match input.split_with(|c: char| !c.is_ascii_digit()) {
        ("", _) => (),
        (number, input) => return Ok((input, (offset, TokenKind::Number(number.parse().unwrap())))),
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

            return Ok((input, (offset, tok)))
        }
    }

    if let Some(input) = input.strip_prefix("..") {
        return Ok((input, (offset, TokenKind::Symbol(Symbol::Dot2))))
    }

    if let Some(input) = input.strip_prefix("->") {
        return Ok((input, (offset, TokenKind::Symbol(Symbol::Arrow))))
    }

    let kind = match input.get(0..1) {
        Some("(") => TokenKind::Symbol(Symbol::OpenParen),
        Some(")") => TokenKind::Symbol(Symbol::CloseParen),
        Some(";") => TokenKind::Symbol(Symbol::SemiColon),
        Some("=") => TokenKind::Symbol(Symbol::Equal),
        Some("@") => TokenKind::Symbol(Symbol::Borrow),
        _ => return Err(Error::Error((input, ErrorKind::Custom("no token found")))),
    };

    Ok((&input[1..], (offset, kind)))
}
