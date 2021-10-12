use crate::validator::ExprType;
use std::fmt::Display;

use std::fmt::Write;

#[derive(Debug, Copy, Clone)]
pub enum Unaryop {
    Not,
}

impl Display for Unaryop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unaryop::Not => write!(f, "!"),
        }
    }
}

/// "Mathematical" binary operations variants
#[derive(Debug, Copy, Clone)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
    LogicalAnd,
    LogicalOr,
    DotAccess,
}

impl Display for Binop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Binop::Add => write!(f, "+"),
            Binop::Sub => write!(f, "-"),
            Binop::Mul => write!(f, "*"),
            Binop::Div => write!(f, "/"),
            Binop::LogicalAnd => write!(f, "&&"),
            Binop::LogicalOr => write!(f, "||"),
            Binop::DotAccess => write!(f, "."),
        }
    }
}

/// Comparison operations
#[derive(Debug, Copy, Clone)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Display for Cmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cmp::Eq => write!(f, "=="),
            Cmp::Ne => write!(f, "!="),
            Cmp::Lt => write!(f, "<"),
            Cmp::Le => write!(f, "<="),
            Cmp::Gt => write!(f, ">"),
            Cmp::Ge => write!(f, ">="),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CodeRef {
    pub pos: usize,
    pub line: Option<usize>,
}

impl CodeRef {
    pub fn new(pos: usize) -> Self {
        CodeRef { pos, line: None }
    }
    pub fn z() -> Self {
        CodeRef { pos: 0, line: None }
    }
    pub fn setup(&mut self, code: &String) {
        if self.line.is_none() {
            self.line = Some(code[..self.pos].matches("\n").count());
        }
    }
}

impl Display for CodeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {}", line)
        } else {
            write!(f, "pos {}", self.pos)
        }
    }
}

type NV<T> = non_empty_vec::NonEmpty<T>;

/// The AST node for expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    LiteralFloat(CodeRef, String),
    LiteralInt(CodeRef, String),
    LiteralBool(CodeRef, bool),
    LiteralString(CodeRef, String),
    LiteralArray(CodeRef, Box<Expr>, usize),
    Identifier(CodeRef, String),
    Binop(CodeRef, Binop, Box<Expr>, Box<Expr>),
    Unaryop(CodeRef, Unaryop, Box<Expr>),
    Compare(CodeRef, Cmp, Box<Expr>, Box<Expr>),
    IfThen(CodeRef, Box<Expr>, Vec<Expr>),
    IfElse(CodeRef, Box<Expr>, Vec<Expr>, Vec<Expr>),
    Assign(CodeRef, NV<Expr>, NV<Expr>),
    AssignOp(CodeRef, Binop, Box<String>, Box<Expr>),
    NewStruct(CodeRef, String, Vec<StructAssignField>),
    WhileLoop(CodeRef, Box<Expr>, Vec<Expr>), //Should this take a block instead of Vec<Expr>?
    Block(CodeRef, Vec<Expr>),
    Call(CodeRef, String, Vec<Expr>),
    GlobalDataAddr(CodeRef, String),
    Parentheses(CodeRef, Box<Expr>),
    ArrayAccess(CodeRef, String, Box<Expr>),
}

impl Expr {
    pub fn get_code_ref(&self) -> &CodeRef {
        match self {
            Expr::LiteralFloat(code_ref, ..) => code_ref,
            Expr::LiteralInt(code_ref, ..) => code_ref,
            Expr::LiteralBool(code_ref, ..) => code_ref,
            Expr::LiteralString(code_ref, ..) => code_ref,
            Expr::LiteralArray(code_ref, ..) => code_ref,
            Expr::Identifier(code_ref, ..) => code_ref,
            Expr::Binop(code_ref, ..) => code_ref,
            Expr::Unaryop(code_ref, ..) => code_ref,
            Expr::Compare(code_ref, ..) => code_ref,
            Expr::IfThen(code_ref, ..) => code_ref,
            Expr::IfElse(code_ref, ..) => code_ref,
            Expr::Assign(code_ref, ..) => code_ref,
            Expr::AssignOp(code_ref, ..) => code_ref,
            Expr::NewStruct(code_ref, ..) => code_ref,
            Expr::WhileLoop(code_ref, ..) => code_ref,
            Expr::Block(code_ref, ..) => code_ref,
            Expr::Call(code_ref, ..) => code_ref,
            Expr::GlobalDataAddr(code_ref, ..) => code_ref,
            Expr::Parentheses(code_ref, ..) => code_ref,
            Expr::ArrayAccess(code_ref, ..) => code_ref,
        }
    }

    pub fn get_code_ref_mut(&mut self) -> &mut CodeRef {
        match self {
            Expr::LiteralFloat(code_ref, ..) => code_ref,
            Expr::LiteralInt(code_ref, ..) => code_ref,
            Expr::LiteralBool(code_ref, ..) => code_ref,
            Expr::LiteralString(code_ref, ..) => code_ref,
            Expr::LiteralArray(code_ref, ..) => code_ref,
            Expr::Identifier(code_ref, ..) => code_ref,
            Expr::Binop(code_ref, ..) => code_ref,
            Expr::Unaryop(code_ref, ..) => code_ref,
            Expr::Compare(code_ref, ..) => code_ref,
            Expr::IfThen(code_ref, ..) => code_ref,
            Expr::IfElse(code_ref, ..) => code_ref,
            Expr::Assign(code_ref, ..) => code_ref,
            Expr::AssignOp(code_ref, ..) => code_ref,
            Expr::NewStruct(code_ref, ..) => code_ref,
            Expr::WhileLoop(code_ref, ..) => code_ref,
            Expr::Block(code_ref, ..) => code_ref,
            Expr::Call(code_ref, ..) => code_ref,
            Expr::GlobalDataAddr(code_ref, ..) => code_ref,
            Expr::Parentheses(code_ref, ..) => code_ref,
            Expr::ArrayAccess(code_ref, ..) => code_ref,
        }
    }

    pub fn setup_ref(&mut self, src_code: &String) {
        //Walk the AST and Set line numbers, etc.. in code ref
        self.get_code_ref_mut().setup(src_code);
        match self {
            Expr::LiteralFloat(_, _) => (),
            Expr::LiteralInt(_, _) => (),
            Expr::LiteralBool(_, _) => (),
            Expr::LiteralString(_, _) => (),
            Expr::LiteralArray(_, _, _) => (),
            Expr::Identifier(_, _) => (),
            Expr::Binop(_, _, a, b) => {
                a.setup_ref(src_code);
                b.setup_ref(src_code);
            }
            Expr::Unaryop(_, _, a) => {
                a.setup_ref(src_code);
            }
            Expr::Compare(_, _, a, b) => {
                a.setup_ref(src_code);
                b.setup_ref(src_code);
            }
            Expr::IfThen(_, a, bv) => {
                a.setup_ref(src_code);
                for b in bv {
                    b.setup_ref(src_code);
                }
            }
            Expr::IfElse(_, a, bv, cv) => {
                a.setup_ref(src_code);
                for b in bv {
                    b.setup_ref(src_code);
                }
                for c in cv {
                    c.setup_ref(src_code);
                }
            }
            Expr::Assign(_, bv, cv) => {
                for b in bv.as_mut_slice() {
                    b.setup_ref(src_code);
                }
                for c in cv.as_mut_slice() {
                    c.setup_ref(src_code);
                }
            }
            Expr::AssignOp(_, _, _, a) => {
                a.setup_ref(src_code);
            }
            Expr::NewStruct(_, _, fields) => {
                for field in fields {
                    field.expr.setup_ref(src_code);
                }
            }
            Expr::WhileLoop(_, a, bv) => {
                a.setup_ref(src_code);
                for b in bv {
                    b.setup_ref(src_code);
                }
            }
            Expr::Block(_, bv) => {
                for b in bv {
                    b.setup_ref(src_code);
                }
            }
            Expr::Call(_, _, bv) => {
                for b in bv {
                    b.setup_ref(src_code);
                }
            }
            Expr::GlobalDataAddr(_, _) => (),
            Expr::Parentheses(_, a) => {
                a.setup_ref(src_code);
            }
            Expr::ArrayAccess(_, _, a) => {
                a.setup_ref(src_code);
            }
        }
    }
}

pub fn setup_coderef(stmts: &mut Vec<Expr>, src_code: &String) {
    for expr in stmts.iter_mut() {
        expr.setup_ref(src_code);
    }
}

//TODO indentation, tests
impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::LiteralFloat(_, s) => write!(f, "{}", s),
            Expr::LiteralInt(_, s) => write!(f, "{}", s),
            Expr::LiteralString(_, s) => write!(f, "\"{}\"", s),
            Expr::LiteralArray(_, e, len) => write!(f, "[{}; {}]", e, len),
            Expr::Identifier(_, s) => write!(f, "{}", s),
            Expr::Binop(_, op, e1, e2) => write!(f, "{} {} {}", e1, op, e2),
            Expr::Unaryop(_, op, e1) => write!(f, "{} {}", op, e1),
            Expr::Compare(_, cmp, e1, e2) => write!(f, "{} {} {}", e1, cmp, e2),
            Expr::IfThen(_, e, body) => {
                writeln!(f, "if {} {{", e)?;
                for expr in body.iter() {
                    writeln!(f, "{}", expr)?;
                }
                write!(f, "}}")?;
                Ok(())
            }
            Expr::IfElse(_, e, body, else_body) => {
                writeln!(f, "if {} {{", e)?;
                for expr in body.iter() {
                    writeln!(f, "{}", expr)?;
                }
                writeln!(f, "}} else {{")?;
                for expr in else_body.iter() {
                    writeln!(f, "{}", expr)?;
                }
                write!(f, "}}")?;
                Ok(())
            }
            Expr::Assign(_, vars, exprs) => {
                for (i, var) in vars.iter().enumerate() {
                    write!(f, "{}", var)?;
                    let len: usize = vars.len().into();
                    if i < len - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, " = ")?;
                for (i, expr) in exprs.iter().enumerate() {
                    write!(f, "{}", expr)?;
                    let len: usize = exprs.len().into();
                    if i < len - 1 {
                        write!(f, ", ")?;
                    }
                }
                Ok(())
            }
            Expr::AssignOp(_, op, s, e) => write!(f, "{} {}= {}", s, op, e),
            Expr::NewStruct(_, struct_name, args) => {
                writeln!(f, "{}{{", struct_name)?;
                for arg in args.iter() {
                    writeln!(f, "{},", arg)?;
                }
                writeln!(f, "}}")?;
                Ok(())
            }
            Expr::WhileLoop(_, eval, block) => {
                writeln!(f, "while {} {{", eval)?;
                for expr in block.iter() {
                    writeln!(f, "{}", expr)?;
                }
                write!(f, "}}")?;
                Ok(())
            }
            Expr::Block(_, block) => {
                for expr in block.iter() {
                    writeln!(f, "{}", expr)?;
                }
                Ok(())
            }
            Expr::Call(_, func_name, args) => {
                //todo print this correctly
                write!(f, "{}(", func_name)?;
                for (i, arg) in args.iter().enumerate() {
                    write!(f, "{}", arg)?;
                    if i < args.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")?;
                Ok(())
            }
            Expr::GlobalDataAddr(_, e) => write!(f, "{}", e),
            Expr::LiteralBool(_, b) => write!(f, "{}", b),
            Expr::Parentheses(_, e) => write!(f, "({})", e),
            Expr::ArrayAccess(_, var, e) => write!(f, "{}[{}]", var, e),
        }
    }
}

pub fn make_nonempty<T>(v: Vec<T>) -> Option<NV<T>> {
    if v.is_empty() {
        None
    } else {
        Some(unsafe { NV::new_unchecked(v) })
    }
}

#[derive(Debug, Clone)]
pub enum Declaration {
    Function(Function),
    Metadata(Vec<String>, String),
    Struct(Struct),
}

impl Display for Declaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Declaration::Function(e) => write!(f, "{}", e),
            Declaration::Metadata(head, body) => {
                for word in head.iter() {
                    write!(f, "{}", word)?;
                }
                writeln!(f, "")?;
                write!(f, "{}", body)?;
                Ok(())
            }
            Declaration::Struct(e) => write!(f, "{}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub name: String,
    pub expr_type: ExprType,
    pub default_to_float: bool,
}

impl Display for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.default_to_float {
            write!(f, "{}", self.name)
        } else {
            write!(f, "{}: {}", self.name, self.expr_type)
        }
    }
}

#[derive(Debug, Clone)]
pub struct StructAssignField {
    pub field_name: String,
    pub expr: Expr,
}

impl Display for StructAssignField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field_name, self.expr)
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Arg>,
    pub returns: Vec<Arg>,
    pub body: Vec<Expr>,
    pub extern_func: bool,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {} (", self.name)?;
        for (i, param) in self.params.iter().enumerate() {
            write!(f, "{}", param)?;
            if i < self.params.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ") -> (")?;
        for ret in self.returns.iter() {
            write!(f, "{}", ret)?;
        }
        writeln!(f, ") {{")?;
        for expr in self.body.iter() {
            writeln!(f, "{}", expr)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<Arg>,
    pub extern_struct: bool,
}

impl Display for Struct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "struct {} {{", self.name)?;
        for param in &self.fields {
            writeln!(f, "{},", param)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

// TODO there must be a better way.
pub fn pretty_indent(code: &str) -> String {
    let mut f = String::from("");
    let mut depth = 0;
    for line in code.lines() {
        if let Some(b_pos) = line.find("}") {
            if let Some(comment) = line.find("//") {
                if comment > b_pos {
                    depth -= 1;
                }
            } else {
                depth -= 1;
            }
        }
        writeln!(f, "{1:0$}{2:}", depth * 4, "", line).unwrap();
        if let Some(b_pos) = line.find("{") {
            if let Some(comment) = line.find("//") {
                if comment > b_pos {
                    depth += 1;
                }
            } else {
                depth += 1;
            }
        }
    }
    f
}

peg::parser!(pub grammar parser() for str {
    pub rule program() -> Vec<Declaration>
        = (d:declaration() _ { d })*

    rule declaration() -> Declaration
        = function()
        / metadata()
        / structdef()

    rule structdef() -> Declaration
        = _ ext:("extern")? _ "struct" name:identifier() _ "{" _ fields:(a:arg() comma() {a})* _ "}" _ {Declaration::Struct(Struct{name, fields, extern_struct: if ext.is_some() {true} else {false}})}

    rule metadata() -> Declaration
        = _ "@" _ headings:(i:(metadata_identifier()** ([' ' | '\t'])) {i}) ([' ' | '\t'])* "\n" body:$[^'@']* "@" _ {Declaration::Metadata(headings, body.join(""))}

    rule metadata_identifier() -> String
        = quiet!{ _ n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.into() } }
        / expected!("identifier")

    rule function() -> Declaration
        = _ ext:("extern")? _  "fn" name:identifier() _
        "(" params:(i:arg() ** comma()) ")" _
        "->" _
        "(" returns:(i:arg() ** comma()) _ ")"
        body:block()
        {
            let mut name = name;
            if let Some(first_param) = params.first() {
                if first_param.name == "self" {
                    name = format!("{}.{}", first_param.expr_type, name)
                    //change func name to struct_name.func_name if first param is self
                }
            }
            Declaration::Function(Function {
            name,
            params,
            returns,
            body,
            extern_func: if ext.is_some() {true} else {false},
        }) }

    rule arg() -> Arg
        = _ i:identifier() _ ":" _ t:type_label() _ { Arg {name: i.into(), expr_type: t.into(), default_to_float: false } }
        / _ pos:position!() i:identifier() _ { Arg {name: i.into(), expr_type: ExprType::F32(CodeRef::new(pos)), default_to_float: true } }

    rule type_label() -> ExprType
        = _ pos:position!() "f32" _ { ExprType::F32(CodeRef::new(pos)) }
        / _ pos:position!() "i64" _ { ExprType::I64(CodeRef::new(pos)) }
        / _ pos:position!() "&[" ty:type_label() "]" _ { ExprType::Array(CodeRef::new(pos), Box::new(ty), None) }
        / _ pos:position!() "&" _ { ExprType::Address(CodeRef::new(pos)) }
        / _ pos:position!() "bool" _ { ExprType::Bool(CodeRef::new(pos)) }
        / _ pos:position!() n:identifier() _ { ExprType::Struct(CodeRef::new(pos), Box::new(n)) }
        / _ pos:position!() "[" _  ty:type_label()  _ ";" _ len:$(['0'..='9']+) _ "]" _ {
            ExprType::Array(CodeRef::new(pos), Box::new(ty), Some(len.parse::<usize>().unwrap()))
        }

    rule block() -> Vec<Expr>
        = _ "{" _ b:(statement() ** _) _ "}" { b }

    rule statement() -> Expr
        //TODO allow for multiple expressions like: a, b, c returned from if/then/else, etc...
        = while_loop() / assignment() / expression()

    rule expression() -> Expr
        = if_then()
        / if_else()
        / while_loop()
        / assignment()
        / op_assignment()
        / binary_op()

    rule if_then() -> Expr
        = _ pos:position!() "if" _ e:expression() then_body:block() "\n"
        { Expr::IfThen(CodeRef::new(pos), Box::new(e), then_body) }

    rule if_else() -> Expr
        = _ pos:position!() "if" e:expression() _ when_true:block() _ "else" when_false:block()
        { Expr::IfElse(CodeRef::new(pos), Box::new(e), when_true, when_false) }

    rule while_loop() -> Expr
        = _ pos:position!() "while" e:expression() body:block()
        { Expr::WhileLoop(CodeRef::new(pos), Box::new(e), body) }

    rule assignment() -> Expr
        = assignments:((binary_op()) ** comma()) _ pos:position!() "=" args:((_ e:expression() _ {e}) ** comma()) {?
            make_nonempty(assignments)
                .and_then(|assignments| make_nonempty(args)
                .map(|args| Expr::Assign(CodeRef::new(pos), assignments, args)))
                .ok_or("Cannot assign to/from empty tuple")
        }


    rule op_assignment() -> Expr
    = a:(binary_op()) _ "+=" _ b:expression() {assign_op_to_assign(Binop::Add, a, b)}
    / a:(binary_op()) _ "-=" _ b:expression() {assign_op_to_assign(Binop::Sub, a, b)}
    / a:(binary_op()) _ "*=" _ b:expression() {assign_op_to_assign(Binop::Mul, a, b)}
    / a:(binary_op()) _ "/=" _ b:expression() {assign_op_to_assign(Binop::Div, a, b)}

    rule binary_op() -> Expr = precedence!{
        a:@ _ pos:position!() "&&" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::LogicalAnd, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "||" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::LogicalOr, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "==" b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Eq, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "!=" b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Ne, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "<"  b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Lt, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "<=" b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Le, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() ">"  b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Gt, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() ">=" b:(@) { Expr::Compare(CodeRef::new(pos), Cmp::Ge, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "+" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::Add, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "-" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::Sub, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "*" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::Mul, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "/" _ b:(@) { Expr::Binop(CodeRef::new(pos), Binop::Div, Box::new(a), Box::new(b)) }
        --
        a:@ pos:position!() "." b:(@) { Expr::Binop(CodeRef::new(pos), Binop::DotAccess, Box::new(a), Box::new(b)) }
        --
        u:unary_op()  { u }
    }

    rule unary_op() -> Expr = precedence!{
        //Having a _ before the () breaks in this case:
        //c = p.x + p.y + p.z
        //(p.x).print()
        pos:position!() i:identifier() "(" args:((_ e:expression() _ {e}) ** comma()) ")" {
            Expr::Call(CodeRef::new(pos), i, args)
        }
        pos:position!() i:identifier() _ "{" args:((_ e:struct_assign_field() _ {e})*) "}" { Expr::NewStruct(CodeRef::new(pos), i, args) }
        pos:position!() i:identifier() _ "[" idx:expression() "]" { Expr::ArrayAccess(CodeRef::new(pos), i, Box::new(idx)) }
        pos:position!() i:identifier() { Expr::Identifier(CodeRef::new(pos), i) }
        l:literal() { l }
        pos:position!() "!" e:expression() { Expr::Unaryop(CodeRef::new(pos),Unaryop::Not, Box::new(e)) }
        --
        pos:position!() "(" e:expression() ")" { Expr::Parentheses(CodeRef::new(pos), Box::new(e)) }
    }

    rule identifier() -> String
        = quiet!{ _ n:$((!"true"!"false")['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']* "::"? ['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.into() } }

    rule literal() -> Expr
        = _ pos:position!() n:$(['-']*['0'..='9']+"."['0'..='9']+) { Expr::LiteralFloat(CodeRef::new(pos), n.into()) }
        / _ pos:position!() n:$(['-']*['0'..='9']+) { Expr::LiteralInt(CodeRef::new(pos), n.into()) }
        / _ pos:position!() "*" i:identifier() { Expr::GlobalDataAddr(CodeRef::new(pos), i) }
        / _ pos:position!() "true" _ { Expr::LiteralBool(CodeRef::new(pos), true) }
        / _ pos:position!() "false" _ { Expr::LiteralBool(CodeRef::new(pos), false) }
        / _ pos:position!() "\"" body:$[^'"']* "\"" _ { Expr::LiteralString(CodeRef::new(pos), body.join("")) }
        / _ pos:position!() "[" _ "\"" repstr:$[^'\"']* "\"" _ ";" _ len:$(['0'..='9']+) _ "]" _ {
            //Temp solution for creating empty strings
            Expr::LiteralString(CodeRef::new(pos), repstr.join("").repeat( len.parse().unwrap()))
        } //[" "; 10]
        / _ pos:position!() "[" _  e:expression()  _ ";" _ len:$(['0'..='9']+) _ "]" _ {

            Expr::LiteralArray(CodeRef::new(pos), Box::new(e), len.parse::<usize>().unwrap())
        }

    rule struct_assign_field() -> StructAssignField
        = _ i:identifier() _ ":" _ e:expression() comma() _ { StructAssignField {field_name: i.into(), expr: e } }

    rule comment() -> ()
        = quiet!{"//" [^'\n']*"\n"}

    rule comma() = _ ","

    rule _() =  quiet!{comment() / [' ' | '\t' | '\n']}*
});

pub fn assign_op_to_assign(op: Binop, a: Expr, b: Expr) -> Expr {
    Expr::Assign(
        *a.clone().get_code_ref(),
        make_nonempty(vec![a.clone()]).unwrap(),
        make_nonempty(vec![Expr::Binop(
            *b.clone().get_code_ref(),
            op,
            Box::new(a),
            Box::new(b),
        )])
        .unwrap(),
    )
}
