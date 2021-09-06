/// "Mathematical" binary operations variants
#[derive(Debug, Copy, Clone)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

/// Comparison operations
#[derive(Debug, Copy, Clone)]
pub enum Cmp {
    Eq,
    Ne,
    Le,
    Lt,
    Ge,
    Gt,
}

/// The AST node for expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(String),
    Identifier(String),
    Binop(Binop, Box<Expr>, Box<Expr>),
    Compare(Cmp, Box<Expr>, Box<Expr>),
    IfElse(Box<Expr>, Vec<Expr>, Vec<Expr>),
    Assign(Vec<String>, Vec<Expr>),
    WhileLoop(Box<Expr>, Vec<Expr>),
    Block(Vec<Expr>),
    Call(String, Vec<Expr>),
    GlobalDataAddr(String),
}

pub struct Declaration {
    pub name: String,
    pub params: Vec<String>,
    pub returns: Vec<String>,
    pub body: Vec<Expr>,
}

peg::parser!(pub grammar parser() for str {
    pub rule program() -> Vec<Declaration>
        = (f:function() _ { f })*

    rule function() -> Declaration
        = _ "fn" name:identifier() _
        "(" params:(i:identifier() ** comma()) ")" _
        "->" _
        "(" returns:(i:identifier() ** comma()) _ ")"
        body:block()
        { Declaration {
            name,
            params,
            returns,
            body,
        } }

    rule block() -> Vec<Expr>
        = _ "{" b:(statement() ** _) _ "}" { b }

    rule statement() -> Expr
        = while_loop() / assignment() / expression()

    rule expression() -> Expr
        = if_else()
        / binary_op()

    rule if_else() -> Expr
        = _ "if" e:expression() _ when_true:block() _ "else" when_false:block()
        { Expr::IfElse(Box::new(e), when_true, when_false) }

    rule while_loop() -> Expr
        = _ "while" e:expression() body:block()
        { Expr::WhileLoop(Box::new(e), body) }

    rule assignment() -> Expr
        = assignments:((i:identifier() {i}) ** comma()) _ "=" args:((_ e:expression() _ {e}) ** comma()) {Expr::Assign(assignments, args)}

    rule binary_op() -> Expr = precedence!{
        a:@ _ "==" b:(@) { Expr::Compare(Cmp::Eq, Box::new(a), Box::new(b)) }
        a:@ _ "!=" b:(@) { Expr::Compare(Cmp::Ne, Box::new(a), Box::new(b)) }
        a:@ _ "<"  b:(@) { Expr::Compare(Cmp::Lt, Box::new(a), Box::new(b)) }
        a:@ _ "<=" b:(@) { Expr::Compare(Cmp::Le, Box::new(a), Box::new(b)) }
        a:@ _ ">"  b:(@) { Expr::Compare(Cmp::Gt, Box::new(a), Box::new(b)) }
        a:@ _ ">=" b:(@) { Expr::Compare(Cmp::Ge, Box::new(a), Box::new(b)) }
        --
        a:@ _ "+" b:(@) { Expr::Binop(Binop::Add, Box::new(a), Box::new(b)) }
        a:@ _ "-" b:(@) { Expr::Binop(Binop::Sub, Box::new(a), Box::new(b)) }
        --
        a:@ _ "*" b:(@) { Expr::Binop(Binop::Mul, Box::new(a), Box::new(b)) }
        a:@ _ "/" b:(@) { Expr::Binop(Binop::Div, Box::new(a), Box::new(b)) }
        --
        i:identifier() _ "(" args:((_ e:expression() _ {e}) ** ",") ")" { Expr::Call(i, args) }
        i:identifier() { Expr::Identifier(i) }
        l:literal() { l }
    }

    rule identifier() -> String
        = quiet!{ _ n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.to_owned() } }
        / expected!("identifier")

    rule literal() -> Expr
        = _ n:$(['0'..='9']+"."['0'..='9']+) { Expr::Literal(n.to_owned()) }
        / "&" i:identifier() { Expr::GlobalDataAddr(i) }

    rule comment() -> ()
        = quiet!{"//" [^'\n']*"\n"}

    rule comma() = _ ","

    rule _() =  quiet!{comment() / [' ' | '\t' | '\n']}*
});
