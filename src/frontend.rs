/// The AST node for expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(String),
    Identifier(String),
    Assign(Vec<String>, Vec<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    IfElse(Box<Expr>, Vec<Expr>, Vec<Expr>),
    WhileLoop(Box<Expr>, Vec<Expr>),
    Call(String, Vec<Expr>),
    GlobalDataAddr(String),
}

peg::parser!(pub grammar parser() for str {
    pub rule program() -> Vec<(String, Vec<String>, Vec<String>, Vec<Expr>)>
        = __ f:(function()*) __ { f }

    rule function() -> (String, Vec<String>, Vec<String>, Vec<Expr>)
        = __ "fn" _ name:identifier() _
        "(" params:((_ i:identifier() _ {i}) ** ",") ")" _
        "->" _
        "(" returns:((_ i:identifier() _ {i}) ** ",") ")" _
        "{" __
        stmts:statements()
        _ "}" __
        { (name, params, returns, stmts) }

    rule statements() -> Vec<Expr>
        = s:(statement()*) { s }

    rule statement() -> Expr
        = __ e:expression() __ { e }

    rule expression() -> Expr
        = if_else()
        / while_loop()
        / assignment()
        / binary_op()

    rule if_else() -> Expr
        = "if" _ e:expression() _ "{" _ "\n"
        then_body:statements() _ "}" _ "else" _ "{" _ "\n"
        else_body:statements() _ "}"
        { Expr::IfElse(Box::new(e), then_body, else_body) }

    rule while_loop() -> Expr
        = "while" _ e:expression() _ "{" _ "\n"
        loop_body:statements() _ "}"
        { Expr::WhileLoop(Box::new(e), loop_body) }

    rule assignment() -> Expr
        = assignments:((_ i:identifier() _ {i}) ** ",") _ "=" _ args:((_ e:expression() _ {e}) ** ",") {Expr::Assign(assignments, args)}

    rule binary_op() -> Expr = precedence!{
        a:@ _ "==" _ b:(@) { Expr::Eq(Box::new(a), Box::new(b)) }
        a:@ _ "!=" _ b:(@) { Expr::Ne(Box::new(a), Box::new(b)) }
        a:@ _ "<"  _ b:(@) { Expr::Lt(Box::new(a), Box::new(b)) }
        a:@ _ "<=" _ b:(@) { Expr::Le(Box::new(a), Box::new(b)) }
        a:@ _ ">"  _ b:(@) { Expr::Gt(Box::new(a), Box::new(b)) }
        a:@ _ ">=" _ b:(@) { Expr::Ge(Box::new(a), Box::new(b)) }
        --
        a:@ _ "+" _ b:(@) { Expr::Add(Box::new(a), Box::new(b)) }
        a:@ _ "-" _ b:(@) { Expr::Sub(Box::new(a), Box::new(b)) }
        --
        a:@ _ "*" _ b:(@) { Expr::Mul(Box::new(a), Box::new(b)) }
        a:@ _ "/" _ b:(@) { Expr::Div(Box::new(a), Box::new(b)) }
        --
        i:identifier() _ "(" args:((_ e:expression() _ {e}) ** ",") ")" { Expr::Call(i, args) }
        i:identifier() { Expr::Identifier(i) }
        l:literal() { l }
    }

    rule identifier() -> String
        = quiet!{ n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.to_owned() } }
        / expected!("identifier")

    rule literal() -> Expr
        = n:$(['0'..='9']+"."['0'..='9']+) { Expr::Literal(n.to_owned()) }
        / "&" i:identifier() { Expr::GlobalDataAddr(i) }

    rule comment() -> ()
        = quiet!{"//" [^'\n']*"\n"}


    rule _() =  quiet!{[' ' | '\t']*}

    rule __() =  quiet!{comment()* [' ' | '\t' | '\n']* comment()* [' ' | '\t' | '\n']*}
});
