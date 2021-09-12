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

type NV<T> = non_empty_vec::NonEmpty<T>;

/// The AST node for expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(String),
    Identifier(String),
    Binop(Binop, Box<Expr>, Box<Expr>),
    Compare(Cmp, Box<Expr>, Box<Expr>),
    IfThen(Box<Expr>, Vec<Expr>),
    IfElse(Box<Expr>, Vec<Expr>, Vec<Expr>),
    Assign(NV<String>, NV<Expr>),
    AssignOp(Binop, Box<String>, Box<Expr>),
    WhileLoop(Box<Expr>, Vec<Expr>),
    Block(Vec<Expr>),
    Call(String, Vec<Expr>),
    GlobalDataAddr(String),
    Bool(bool),
    Parentheses(Box<Expr>),
    ArrayGet(String, Box<Expr>),
    ArraySet(String, Box<Expr>, Box<Expr>),
}

fn make_nonempty<T>(v: Vec<T>) -> Option<NV<T>> {
    if v.is_empty() {
        None
    } else {
        Some(unsafe { NV::new_unchecked(v) })
    }
}

#[derive(Debug, Clone)]
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
        //TODO allow for multiple expressions like: a, b, c returned from if/then/else, etc...
        = while_loop() / assignment() / expression()

    rule expression() -> Expr
        = if_then()
        / if_else()
        / while_loop()
        / arrayset()
        / assignment()
        / binary_op()

    rule if_then() -> Expr
        = "if" _ e:expression() then_body:block() "\n"
        { Expr::IfThen(Box::new(e), then_body) }

    rule if_else() -> Expr
        = _ "if" e:expression() _ when_true:block() _ "else" when_false:block()
        { Expr::IfElse(Box::new(e), when_true, when_false) }

    rule while_loop() -> Expr
        = _ "while" e:expression() body:block()
        { Expr::WhileLoop(Box::new(e), body) }

    rule assignment() -> Expr
        = assignments:((i:identifier() {i}) ** comma()) _ "=" args:((_ e:expression() _ {e}) ** comma()) {?
            make_nonempty(assignments)
                .and_then(|assignments| make_nonempty(args)
                .map(|args| Expr::Assign(assignments, args)))
                .ok_or("Cannot assign to/from empty tuple")
        }

    rule arrayset() -> Expr
        = i:identifier() _ "[" idx:expression() "]" _ "=" _ e:expression() {Expr::ArraySet(i, Box::new(idx), Box::new(e))}

    rule binary_op() -> Expr = precedence!{
        a:@ _ "==" b:(@) { Expr::Compare(Cmp::Eq, Box::new(a), Box::new(b)) }
        a:@ _ "!=" b:(@) { Expr::Compare(Cmp::Ne, Box::new(a), Box::new(b)) }
        a:@ _ "<"  b:(@) { Expr::Compare(Cmp::Lt, Box::new(a), Box::new(b)) }
        a:@ _ "<=" b:(@) { Expr::Compare(Cmp::Le, Box::new(a), Box::new(b)) }
        a:@ _ ">"  b:(@) { Expr::Compare(Cmp::Gt, Box::new(a), Box::new(b)) }
        a:@ _ ">=" b:(@) { Expr::Compare(Cmp::Ge, Box::new(a), Box::new(b)) }
        --
        a:@ _ "+" _ b:(@) { Expr::Binop(Binop::Add, Box::new(a), Box::new(b)) }
        i:identifier() _ "+=" _ a:(@) { Expr::AssignOp(Binop::Add, Box::new(i), Box::new(a)) }

        a:@ _ "-" _ b:(@) { Expr::Binop(Binop::Sub, Box::new(a), Box::new(b)) }
        i:identifier() _ "-=" _ a:(@) { Expr::AssignOp(Binop::Sub, Box::new(i), Box::new(a)) }
        --
        a:@ _ "*" _ b:(@) { Expr::Binop(Binop::Mul, Box::new(a), Box::new(b)) }
        i:identifier() _ "*=" _ a:(@) { Expr::AssignOp(Binop::Mul, Box::new(i), Box::new(a)) }

        a:@ _ "/" _ b:(@) { Expr::Binop(Binop::Div, Box::new(a), Box::new(b)) }
        i:identifier() _ "/=" _ a:(@) { Expr::AssignOp(Binop::Div, Box::new(i), Box::new(a)) }
        --
        i:identifier() _ "(" args:((_ e:expression() _ {e}) ** comma()) ")" { Expr::Call(i, args) }
        i:identifier() _ "[" idx:expression() "]" { Expr::ArrayGet(i, Box::new(idx)) }
        i:identifier() { Expr::Identifier(i) }
        l:literal() { l }
        --
        "(" e:expression() ")" { Expr::Parentheses(Box::new(e)) }

    }

    rule identifier() -> String
        = quiet!{ _ n:$((!"true"!"false")['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.to_owned() } }
        / _ "&" i:identifier() _ { "&".to_owned()+&i } //TODO Should this be a seperate type?
        / expected!("identifier")

    rule literal() -> Expr
        = _ n:$(['-']*['0'..='9']+"."['0'..='9']+) { Expr::Literal(n.to_owned()) }
        / "*" i:identifier() { Expr::GlobalDataAddr(i) }
        / _ "true" _ { Expr::Bool(true) }
        / _ "false" _ { Expr::Bool(false) }

    rule comment() -> ()
        = quiet!{"//" [^'\n']*"\n"}

    rule comma() = _ ","

    rule _() =  quiet!{comment() / [' ' | '\t' | '\n']}*
});
