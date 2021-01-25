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
    for ast in ast {
        println!("{:?}", ast);
    }
}
