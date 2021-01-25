use dec::base::error::{Error, ErrorKind, PResult, ParseError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token<'a> {
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

pub fn parse<'a, E: ParseError<&'a str>>(input: &'a str) -> PResult<&str, Vec<Token<'_>>, E> {
    use dec::base::ParseOnce;
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

#[allow(clippy::or_fun_call)]
pub fn parse_token<'a, E: ParseError<&'a str>>(mut input: &'a str) -> PResult<&str, Token<'_>, E> {
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
        (number, input) => return Ok((input, Token::Number(number.parse().unwrap()))),
    }

    match input.split_with(|c: char| !c.is_alphanumeric() && c != '_') {
        ("", _) => (),
        (ident, input) => {
            let tok = match ident {
                "let" => Token::Keyword(Keyword::Let),
                "shr" => Token::Keyword(Keyword::Shared),
                "exc" => Token::Keyword(Keyword::Exclusive),
                "read" => Token::Keyword(Keyword::Read),
                "write" => Token::Keyword(Keyword::Write),
                "drop" => Token::Keyword(Keyword::Drop),
                _ => Token::Ident(ident),
            };

            return Ok((input, tok))
        }
    }

    if let Some(input) = input.strip_prefix("..") {
        return Ok((input, Token::Symbol(Symbol::Dot2)))
    }

    if let Some(input) = input.strip_prefix("->") {
        return Ok((input, Token::Symbol(Symbol::Arrow)))
    }

    match input.get(0..1) {
        Some("(") => Ok((&input[1..], Token::Symbol(Symbol::OpenParen))),
        Some(")") => Ok((&input[1..], Token::Symbol(Symbol::CloseParen))),
        Some(";") => Ok((&input[1..], Token::Symbol(Symbol::SemiColon))),
        Some("=") => Ok((&input[1..], Token::Symbol(Symbol::Equal))),
        Some("@") => Ok((&input[1..], Token::Symbol(Symbol::Borrow))),
        _ => Err(Error::Error(E::from_input_kind(
            input,
            ErrorKind::Custom("no token found"),
        ))),
    }
}
