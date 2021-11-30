use tracing::error;
use tracing::trace;

use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;

use std::collections::HashSet;
use std::fmt::Display;

use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SarusRange {
    pub start: Option<Box<Expr>>,
    pub end: Option<Box<Expr>>,
}

impl Display for SarusRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start = if let Some(start) = &self.start {
            start.to_string()
        } else {
            "".to_string()
        };
        let end = if let Some(end) = &self.end {
            end.to_string()
        } else {
            "".to_string()
        };
        write!(f, "{}..{}", start, end)
    }
}

#[derive(Debug, Clone)]
pub enum Unaryop {
    Not,
    Negative,
    Slice(SarusRange),
}

impl Display for Unaryop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unaryop::Not => write!(f, "!"),
            Unaryop::Negative => write!(f, "-"),
            Unaryop::Slice(range) => write!(f, "[{}]", range),
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
            Binop::Add => write!(f, " + "),
            Binop::Sub => write!(f, " - "),
            Binop::Mul => write!(f, " * "),
            Binop::Div => write!(f, " / "),
            Binop::LogicalAnd => write!(f, " && "),
            Binop::LogicalOr => write!(f, " || "),
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
            Cmp::Eq => write!(f, " == "),
            Cmp::Ne => write!(f, " != "),
            Cmp::Lt => write!(f, " < "),
            Cmp::Le => write!(f, " <= "),
            Cmp::Gt => write!(f, " > "),
            Cmp::Ge => write!(f, " >= "),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CodeRef {
    pub pos: usize,
    pub line: Option<usize>,
    pub file_index: Option<u64>, //Index that holds the file name / path
}

impl CodeRef {
    pub fn new(pos: usize, code_ctx: &CodeContext) -> Self {
        let line = if pos < code_ctx.code.len() {
            Some(code_ctx.code[..pos].matches('\n').count() + 1)
        } else {
            None
        };
        CodeRef {
            pos,
            line,
            file_index: code_ctx.file_index,
        }
    }
    pub fn z() -> Self {
        CodeRef {
            pos: 0,
            line: None,
            file_index: None,
        }
    }
    pub fn s(&self, t: &Option<Vec<PathBuf>>) -> String {
        let s = if let Some(file_index) = self.file_index {
            if let Some(file_index_table) = t {
                if let Some(file) = file_index_table.get(file_index as usize) {
                    file.to_string_lossy().to_string()
                } else {
                    error!("{} CodeRef File Index out of bounds {}", self, file_index);
                    "".to_string()
                }
            } else {
                "".to_string()
            }
        } else {
            "".to_string()
        };
        if let Some(line) = self.line {
            format!("line {}:{}", s, line)
        } else {
            format!("pos {}:{}", s, self.pos)
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

/// The AST node for expressions.
#[derive(Debug, Clone)]
pub enum Expr {
    LiteralFloat(CodeRef, f32),
    LiteralInt(CodeRef, i64),
    LiteralU8(CodeRef, u8),
    LiteralBool(CodeRef, bool),
    LiteralString(CodeRef, String),
    LiteralArray(CodeRef, Vec<Expr>, usize),
    Identifier(CodeRef, String),
    Binop(CodeRef, Binop, Box<Expr>, Box<Expr>),
    Unaryop(CodeRef, Unaryop, Box<Expr>),
    Compare(CodeRef, Cmp, Box<Expr>, Box<Expr>),
    IfThen(CodeRef, Box<Expr>, Vec<Expr>),
    IfElse(CodeRef, Box<Expr>, Vec<Expr>, Vec<Expr>),
    IfThenElseIf(CodeRef, Vec<(Expr, Vec<Expr>)>),
    IfThenElseIfElse(CodeRef, Vec<(Expr, Vec<Expr>)>, Vec<Expr>),
    Assign(CodeRef, Vec<Expr>, Vec<Expr>),
    NewStruct(CodeRef, String, Vec<StructAssignField>),
    WhileLoop(CodeRef, Box<Expr>, Option<Vec<Expr>>, Vec<Expr>), //Should this take a block instead of Vec<Expr>?
    Block(CodeRef, Vec<Expr>),
    Break(CodeRef),
    Continue(CodeRef),
    Return(CodeRef),
    Call(CodeRef, String, Vec<Expr>, bool),
    GlobalDataAddr(CodeRef, String),
    Parentheses(CodeRef, Box<Expr>),
    ArrayAccess(CodeRef, Box<Expr>, Box<Expr>),
    Declaration(CodeRef, Declaration),
}

impl Expr {
    pub fn debug_get_name(&self) -> String {
        //Is this really the easiest way?
        let s = format!("{:?}", self);
        let end = s.find('(').unwrap_or_else(|| s.len());
        s[0..end].to_string()
    }

    pub fn get_code_ref(&self) -> &CodeRef {
        match self {
            Expr::LiteralFloat(code_ref, ..) => code_ref,
            Expr::LiteralInt(code_ref, ..) => code_ref,
            Expr::LiteralU8(code_ref, ..) => code_ref,
            Expr::LiteralBool(code_ref, ..) => code_ref,
            Expr::LiteralString(code_ref, ..) => code_ref,
            Expr::LiteralArray(code_ref, ..) => code_ref,
            Expr::Identifier(code_ref, ..) => code_ref,
            Expr::Binop(code_ref, ..) => code_ref,
            Expr::Unaryop(code_ref, ..) => code_ref,
            Expr::Compare(code_ref, ..) => code_ref,
            Expr::IfThen(code_ref, ..) => code_ref,
            Expr::IfThenElseIf(code_ref, ..) => code_ref,
            Expr::IfThenElseIfElse(code_ref, ..) => code_ref,
            Expr::IfElse(code_ref, ..) => code_ref,
            Expr::Assign(code_ref, ..) => code_ref,
            Expr::NewStruct(code_ref, ..) => code_ref,
            Expr::WhileLoop(code_ref, ..) => code_ref,
            Expr::Block(code_ref, ..) => code_ref,
            Expr::Break(code_ref, ..) => code_ref,
            Expr::Continue(code_ref, ..) => code_ref,
            Expr::Return(code_ref, ..) => code_ref,
            Expr::Call(code_ref, ..) => code_ref,
            Expr::GlobalDataAddr(code_ref, ..) => code_ref,
            Expr::Parentheses(code_ref, ..) => code_ref,
            Expr::ArrayAccess(code_ref, ..) => code_ref,
            Expr::Declaration(code_ref, ..) => code_ref,
        }
    }

    pub fn get_code_ref_mut(&mut self) -> &mut CodeRef {
        match self {
            Expr::LiteralFloat(code_ref, ..) => code_ref,
            Expr::LiteralInt(code_ref, ..) => code_ref,
            Expr::LiteralU8(code_ref, ..) => code_ref,
            Expr::LiteralBool(code_ref, ..) => code_ref,
            Expr::LiteralString(code_ref, ..) => code_ref,
            Expr::LiteralArray(code_ref, ..) => code_ref,
            Expr::Identifier(code_ref, ..) => code_ref,
            Expr::Binop(code_ref, ..) => code_ref,
            Expr::Unaryop(code_ref, ..) => code_ref,
            Expr::Compare(code_ref, ..) => code_ref,
            Expr::IfThen(code_ref, ..) => code_ref,
            Expr::IfThenElseIf(code_ref, ..) => code_ref,
            Expr::IfThenElseIfElse(code_ref, ..) => code_ref,
            Expr::IfElse(code_ref, ..) => code_ref,
            Expr::Assign(code_ref, ..) => code_ref,
            Expr::NewStruct(code_ref, ..) => code_ref,
            Expr::WhileLoop(code_ref, ..) => code_ref,
            Expr::Block(code_ref, ..) => code_ref,
            Expr::Break(code_ref, ..) => code_ref,
            Expr::Continue(code_ref, ..) => code_ref,
            Expr::Return(code_ref, ..) => code_ref,
            Expr::Call(code_ref, ..) => code_ref,
            Expr::GlobalDataAddr(code_ref, ..) => code_ref,
            Expr::Parentheses(code_ref, ..) => code_ref,
            Expr::ArrayAccess(code_ref, ..) => code_ref,
            Expr::Declaration(code_ref, ..) => code_ref,
        }
    }

    pub fn literal_float(v: f32) -> Self {
        Expr::LiteralFloat(CodeRef::z(), v)
    }
    pub fn literal_int(v: i64) -> Self {
        Expr::LiteralInt(CodeRef::z(), v)
    }
    pub fn literal_bool(v: bool) -> Self {
        Expr::LiteralBool(CodeRef::z(), v)
    }
    pub fn literal_string(s: &str) -> Self {
        Expr::LiteralString(CodeRef::z(), s.to_string())
    }
    pub fn literal_array(v: &Expr, len: usize) -> Self {
        Expr::LiteralArray(CodeRef::z(), vec![v.clone()], len)
    }
    pub fn identifier(s: &str) -> Self {
        Expr::Identifier(CodeRef::z(), s.to_string())
    }
    pub fn binop(op: &Binop, lhs: &Expr, rhs: &Expr) -> Self {
        Expr::Binop(
            CodeRef::z(),
            op.clone(),
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
        )
    }
    pub fn unaryop(op: &Unaryop, lhs: &Expr) -> Self {
        Expr::Unaryop(CodeRef::z(), op.clone(), Box::new(lhs.clone()))
    }
    pub fn compare(op: &Cmp, lhs: &Expr, rhs: &Expr) -> Self {
        Expr::Compare(
            CodeRef::z(),
            op.clone(),
            Box::new(lhs.clone()),
            Box::new(rhs.clone()),
        )
    }
    pub fn if_then(cond: &Expr, then_body: &Vec<Expr>) -> Self {
        Expr::IfThen(CodeRef::z(), Box::new(cond.clone()), then_body.clone())
    }
    pub fn if_else(cond: &Expr, then_body: &Vec<Expr>, else_body: &Vec<Expr>) -> Self {
        Expr::IfElse(
            CodeRef::z(),
            Box::new(cond.clone()),
            then_body.clone(),
            else_body.clone(),
        )
    }
    pub fn if_then_else_if(expr_bodies: &Vec<(Expr, Vec<Expr>)>) -> Self {
        Expr::IfThenElseIf(CodeRef::z(), expr_bodies.clone())
    }
    pub fn if_then_else_if_else(
        expr_bodies: &Vec<(Expr, Vec<Expr>)>,
        else_body: &Vec<Expr>,
    ) -> Self {
        Expr::IfThenElseIfElse(CodeRef::z(), expr_bodies.clone(), else_body.clone())
    }
    pub fn assign(lhs: &Vec<Expr>, rhs: &Vec<Expr>) -> Self {
        Expr::Assign(CodeRef::z(), lhs.clone(), rhs.clone())
    }
    //use assign_op_to_assign(op: Binop, a: Expr, b: Expr) for op_assign
    pub fn new_struct(name: &str, rhs: &Vec<StructAssignField>) -> Self {
        Expr::NewStruct(CodeRef::z(), name.to_string(), rhs.clone())
    }
    pub fn while_loop(cond: &Expr, iter_body: &Option<Vec<Expr>>, body: &Vec<Expr>) -> Self {
        Expr::WhileLoop(
            CodeRef::z(),
            Box::new(cond.clone()),
            iter_body.clone(),
            body.clone(),
        )
    }
    pub fn block(body: &Vec<Expr>) -> Self {
        Expr::Block(CodeRef::z(), body.clone())
    }
    pub fn call(func_name: &str, args: &Vec<Expr>) -> Self {
        Expr::Call(CodeRef::z(), func_name.to_string(), args.clone(), false)
    }
    pub fn parentheses(args: &Expr) -> Self {
        Expr::Parentheses(CodeRef::z(), Box::new(args.clone()))
    }
    pub fn array_access(expr: &Expr, idx_expr: &Expr) -> Self {
        Expr::ArrayAccess(
            CodeRef::z(),
            Box::new(expr.clone()),
            Box::new(idx_expr.clone()),
        )
    }
    pub fn declaration(decl: &Declaration) -> Self {
        Expr::Declaration(CodeRef::z(), decl.clone())
    }
}

//TODO indentation, tests
impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::LiteralFloat(_, s) => write!(f, "{}", s),
            Expr::LiteralInt(_, s) => write!(f, "{}", s),
            Expr::LiteralU8(_, s) => write!(f, "{}u8", s),
            Expr::LiteralString(_, s) => write!(f, "\"{}\"", s),
            Expr::LiteralArray(_, es, len) => {
                if es.len() == 1 {
                    write!(f, "[{}; {}]", es[0], len)
                } else {
                    write!(f, "[")?;
                    for (i, e) in es.iter().enumerate() {
                        if i < es.len() - 1 {
                            write!(f, "{},", e)?;
                        } else {
                            write!(f, "{}", e)?;
                        }
                    }
                    write!(f, "]")
                }
            }
            Expr::Identifier(_, s) => write!(f, "{}", s),
            Expr::Binop(_, op, e1, e2) => write!(f, "{}{}{}", e1, op, e2),
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
            Expr::IfThenElseIf(_, a) => {
                write!(f, "if ")?;
                for (i, (e, body)) in a.iter().enumerate() {
                    writeln!(f, "{} {{", e)?;
                    for expr in body {
                        writeln!(f, "{}", expr)?;
                    }
                    write!(f, "}}")?;
                    if i < a.len() - 1 {
                        write!(f, " else if ")?;
                    } else {
                        writeln!(f)?;
                    }
                }
                Ok(())
            }
            Expr::IfThenElseIfElse(_, a, else_body) => {
                write!(f, "if ")?;
                for (i, (e, body)) in a.iter().enumerate() {
                    writeln!(f, "{} {{", e)?;
                    for expr in body {
                        writeln!(f, "{}", expr)?;
                    }
                    write!(f, "}}")?;
                    if i < a.len() - 1 {
                        write!(f, " else if ")?;
                    }
                }
                writeln!(f, " else {{")?;
                for expr in else_body {
                    writeln!(f, "{}", expr)?;
                }
                writeln!(f, "}}")?;

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
            Expr::NewStruct(_, struct_name, args) => {
                writeln!(f, "{}{{", struct_name)?;
                for arg in args.iter() {
                    writeln!(f, "{},", arg)?;
                }
                writeln!(f, "}}")?;
                Ok(())
            }
            Expr::WhileLoop(_, eval, iter_block, block) => {
                writeln!(f, "while {} ", eval)?;
                if let Some(iter_block) = iter_block {
                    write!(f, "{{")?;
                    for expr in iter_block.iter() {
                        writeln!(f, "{}", expr)?;
                    }
                    write!(f, "}} : ")?;
                }
                write!(f, "{{")?;
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
            Expr::Break(_) => writeln!(f, "break"),
            Expr::Continue(_) => writeln!(f, "continue"),
            Expr::Return(_) => writeln!(f, "return"),
            Expr::Call(_, func_name, args, _) => {
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
            Expr::Declaration(_, e) => write!(f, "{}", e),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Declaration {
    Function(Function),
    Metadata(Vec<String>, String),
    Struct(Struct),
    Include(String), //Naive implementation that will change significantly.
}

impl Display for Declaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Declaration::Function(e) => write!(f, "{}", e),
            Declaration::Metadata(head, body) => {
                for word in head.iter() {
                    write!(f, "{}", word)?;
                }
                writeln!(f)?;
                write!(f, "{}", body)?;
                Ok(())
            }
            Declaration::Struct(e) => write!(f, "{}", e),
            Declaration::Include(path) => writeln!(f, "include {}", path),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub name: String,
    pub expr_type: ExprType,
    pub default_to_float: bool,
    pub closure_arg: Option<Function>,
}

impl Display for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.default_to_float {
            write!(f, "{}", self.name)
        } else if let Some(closure_arg) = &self.closure_arg {
            write!(f, "{}: {}", self.name, closure_arg)
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

/// Comparison operations
#[derive(Debug, Copy, Clone)]
pub enum InlineKind {
    Default,
    Never,
    Always,
    Often,
}

impl Display for InlineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InlineKind::Default => write!(f, ""),
            InlineKind::Never => write!(f, "never_inline"),
            InlineKind::Always => write!(f, "always_inline"),
            InlineKind::Often => write!(f, "inline"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Arg>,
    pub returns: Vec<Arg>,
    pub body: Vec<Expr>,
    pub extern_func: bool,
    pub inline: InlineKind,
}

impl Function {
    pub fn sig_string(&self) -> anyhow::Result<String> {
        let mut f = String::new();
        f.reserve(200);

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
        write!(f, ") {{}}")?;

        Ok(f)
    }

    pub fn external(name: String, params: Vec<Arg>, returns: Vec<Arg>) -> Self {
        Function {
            name,
            params,
            returns,
            body: vec![],
            extern_func: true,
            inline: InlineKind::Never,
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.extern_func {
            write!(f, "extern ")?;
        }
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
        write!(f, ") {{")?;
        if !self.extern_func {
            writeln!(f)?;
            for expr in self.body.iter() {
                writeln!(f, "{}", expr)?;
            }
        }
        write!(f, "}}")?;
        if !self.extern_func {
            writeln!(f)?;
        }
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
        if !self.extern_struct {
            writeln!(f)?;
        }
        for param in &self.fields {
            writeln!(f, "{},", param)?;
        }
        write!(f, "}}")?;
        if !self.extern_struct {
            writeln!(f)?;
        }
        Ok(())
    }
}

// TODO there must be a better way.
pub fn pretty_indent(code: &str) -> String {
    let mut f = String::from("");
    let mut depth = 0i32;
    for line in code.lines() {
        if let Some(b_pos) = line.find('}') {
            if let Some(comment) = line.find("//") {
                if comment > b_pos {
                    depth -= 1;
                }
            } else {
                depth -= 1;
            }
        }
        writeln!(f, "{1:0$}{2:}", depth.max(0) as usize * 4, "", line).unwrap();
        if let Some(b_pos) = line.find('{') {
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

//TODO wish this could be a HashSet
const RESERVED_WORDS: [&str; 5] = ["true", "false", "break", "continue", "return"];

peg::parser!(pub grammar parser(code_ctx: &CodeContext) for str {
    pub rule program() -> Vec<Declaration>
        = (d:declaration() _ { d })*

    rule declaration() -> Declaration
        = function()
        / metadata()
        / structdef()
        / include()

    rule include() -> Declaration
        = _ "include" _ "\"" body:$[^'"']* "\"" { Declaration::Include(body.join("")) }

    rule structdef() -> Declaration
        = _ ext:("extern")? _ "struct" _ name:$(s:identifier() ("::" (ty:type_label() ** "::"))?) _ "{" _ fields:(a:arg() comma() {a})* _ "}" _ {Declaration::Struct(Struct{name: name.to_string(), fields, extern_struct: ext.is_some()})}

    rule metadata() -> Declaration
        = _ "@" _ headings:(i:(metadata_identifier()** ([' ' | '\t'])) {i}) ([' ' | '\t'])* "\n" body:$[^'@']* "@" _ {Declaration::Metadata(headings, body.join(""))}

    rule metadata_identifier() -> String
        = quiet!{ _ n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.into() } }
        / expected!("identifier")

    rule function() -> Declaration
        = _ ext:("extern")? _ inline:function_inline_kind()? _  "fn" _ name:identifier() _
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
            extern_func: ext.is_some(),
            inline: if let Some(inline) = inline {inline} else {InlineKind::Default},
        }) }

    rule closure_definition(name: String) -> Function
        = "|" params:(i:arg() ** comma()) "|" _
         "->" _
         "(" returns:(i:arg() ** comma()) _ ")"
         {
             Function {
             name,
             params,
             returns,
             body: vec![],
             extern_func: false,
             inline: InlineKind::Always,
        } }

    rule closure_declaration(name: String) -> Declaration
        = "|" params:(i:arg() ** comma()) "|" _
        "->" _
        "(" returns:(i:arg() ** comma()) _ ")"
        body:block()
        {
            Declaration::Function(Function {
            name,
            params,
            returns,
            body,
            extern_func: false,
            inline: InlineKind::Always,
        }) }

    rule function_inline_kind() -> InlineKind
    = "inline" {InlineKind::Often}
    / "never_inline" {InlineKind::Never}
    / "always_inline" {InlineKind::Always}

    rule arg() -> Arg
        = _ i:identifier() _ ":" _ c:closure_definition(i) _ { Arg {name: c.name.to_string(), expr_type: ExprType::Void(CodeRef::z()), default_to_float: false, closure_arg: Some(c) } }
        / _ i:identifier() _ ":" _ t:type_label() _ { Arg {name: i, expr_type: t, default_to_float: false, closure_arg: None } }
        / _ pos:position!() i:identifier() _ { Arg {name: i, expr_type: ExprType::F32(CodeRef::new(pos, code_ctx)), default_to_float: true, closure_arg: None } }

    rule type_label() -> ExprType
        = _ pos:position!() "f32" { ExprType::F32(CodeRef::new(pos, code_ctx)) }
        / _ pos:position!() "i64" { ExprType::I64(CodeRef::new(pos, code_ctx)) }
        / _ pos:position!() "u8" { ExprType::U8(CodeRef::new(pos, code_ctx)) }
        / _ pos:position!() "&[" ty:type_label() "]" { ExprType::Array(CodeRef::new(pos, code_ctx), Box::new(ty), ArraySizedExpr::Unsized) }
        / _ pos:position!() "[" ty:type_label() "]" { ExprType::Array(CodeRef::new(pos, code_ctx), Box::new(ty), ArraySizedExpr::Slice) }
        / _ pos:position!() "&" { ExprType::Address(CodeRef::new(pos, code_ctx)) }
        / _ pos:position!() "bool" { ExprType::Bool(CodeRef::new(pos, code_ctx)) }
        / _ pos:position!() n:$(identifier() "::" (type_label() ** "::")) { ExprType::Struct(CodeRef::new(pos, code_ctx), Box::new(n.to_string())) }
        / _ pos:position!() n:identifier() { ExprType::Struct(CodeRef::new(pos, code_ctx), Box::new(n)) }
        / _ pos:position!() "[" _  ty:type_label()  _ ";" _ len:$(['0'..='9']+) _ "]" {
            ExprType::Array(CodeRef::new(pos, code_ctx), Box::new(ty), ArraySizedExpr::Fixed(len.parse::<usize>().unwrap()))
        }

    rule block() -> Vec<Expr>
        = _ "{" _ b:(statement() ** _) _ "}" { b }

    rule statement() -> Expr
        //TODO allow for multiple expressions like: a, b, c returned from if/then/else, etc...
        = expression_declaration()
        / while_loop()
        / assignment()
        / expression()
        / break_()
        / continue_()
        / return_()

    rule break_() -> Expr
        = _ pos:position!() "break" _ {Expr::Break(CodeRef::new(pos, code_ctx))}

    rule continue_() -> Expr
        = _ pos:position!() "continue" _ {Expr::Continue(CodeRef::new(pos, code_ctx))}

    rule return_() -> Expr
        = _ pos:position!() "return" _ {Expr::Return(CodeRef::new(pos, code_ctx))}

    rule expression_declaration() -> Expr
        = _ pos:position!() i:identifier()  _ c:closure_declaration(i) { Expr::Declaration(CodeRef::new(pos, code_ctx), c) }

    rule expression() -> Expr
        = if_then()
        / if_else()
        / if_then_else_if_else()
        / if_then_else_if()
        / while_loop()
        / assignment()
        / op_assignment()
        / binary_op()
        / anon_closure()

    rule anon_closure() -> Expr
        = pos:position!() _ c:closure_declaration(("~anon~".to_string())) { Expr::Declaration(CodeRef::new(pos, code_ctx), c) }

    rule if_then() -> Expr
        = _ pos:position!() "if" _ e:expression() then_body:block() "\n"
        { Expr::IfThen(CodeRef::new(pos, code_ctx), Box::new(e), then_body) }

    rule if_else() -> Expr
        = _ pos:position!() "if" e:expression() _ when_true:block() _ "else" when_false:block()
        { Expr::IfElse(CodeRef::new(pos, code_ctx), Box::new(e), when_true, when_false) }

    rule if_then_else_if() -> Expr
        = _ pos:position!() "if" _ expr_bodies:((_ e:expression() _ b:block() _ {(e, b)}) ** "else if" )
        { Expr::IfThenElseIf(CodeRef::new(pos, code_ctx), expr_bodies) }

    rule if_then_else_if_else() -> Expr
        = _ pos:position!() "if" _ expr_bodies:((_ e:expression() _ b:block() _ {(e, b)}) ** "else if" ) _ "else" when_false:block()
        { Expr::IfThenElseIfElse(CodeRef::new(pos, code_ctx), expr_bodies, when_false) }

    rule while_loop() -> Expr
        = _ pos:position!() "while" e:expression() iter_body:(b:block()_":"{b})? body:block()
        { Expr::WhileLoop(CodeRef::new(pos, code_ctx), Box::new(e), iter_body, body) }

    rule assignment() -> Expr
        = assignments:((binary_op()) ** comma()) _ pos:position!() "=" args:((_ e:expression() _ {e}) ** comma()) {
            Expr::Assign(CodeRef::new(pos, code_ctx), assignments, args)
        }


    rule op_assignment() -> Expr
    = a:(binary_op()) _ "+=" _ b:expression() {assign_op_to_assign(Binop::Add, a, b)}
    / a:(binary_op()) _ "-=" _ b:expression() {assign_op_to_assign(Binop::Sub, a, b)}
    / a:(binary_op()) _ "*=" _ b:expression() {assign_op_to_assign(Binop::Mul, a, b)}
    / a:(binary_op()) _ "/=" _ b:expression() {assign_op_to_assign(Binop::Div, a, b)}

    #[cache]
    rule binary_op() -> Expr = precedence!{
        a:@ _ pos:position!() "&&" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::LogicalAnd, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "||" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::LogicalOr, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "==" _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Eq, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "!=" _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Ne, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "<"  _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Lt, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() "<=" _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Le, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() ">"  _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Gt, Box::new(a), Box::new(b)) }
        a:@ _ pos:position!() ">=" _ b:(@) { Expr::Compare(CodeRef::new(pos, code_ctx), Cmp::Ge, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "+" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::Add, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "-" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::Sub, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "*" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::Mul, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "/" _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::Div, Box::new(a), Box::new(b)) }
        --
        a:@ _ pos:position!() "." _ b:(@) { Expr::Binop(CodeRef::new(pos, code_ctx), Binop::DotAccess, Box::new(a), Box::new(b)) }
        --
        u:unary_op()  { u }
        u:unary()  { u }
    }

    rule unary_op() -> Expr
        //TODO e:unary() "[" idx:expression() "]" is causing a severe performance regression vs using i:identifier() "[" idx:expression() "]"
        = _ pos:position!() e:unary() "[" r:range() "]" { Expr::Unaryop(CodeRef::new(pos, code_ctx),Unaryop::Slice(r), Box::new(e)) }
        / _ pos:position!() e:unary() "[" idx:expression() "]" { Expr::ArrayAccess(CodeRef::new(pos, code_ctx), Box::new(e), Box::new(idx)) }
        / _ pos:position!() "!" e:unary() { Expr::Unaryop(CodeRef::new(pos, code_ctx),Unaryop::Not, Box::new(e)) }
        / _ pos:position!() "-" e:unary() { Expr::Unaryop(CodeRef::new(pos, code_ctx),Unaryop::Negative, Box::new(e)) }

    rule range() -> SarusRange
        = _ se:expression()? _ ".." _ ee:expression()? _ {SarusRange{start: if let Some(se) = se {Some(Box::new(se))} else {None},
                                                                     end: if let Some(ee) = ee {Some(Box::new(ee))} else {None}}}
        //TODO add runtime range with _ se:expression() _ ".." _ ee:expression() _
        //This range should probably be a full on type, where this would return a Expr
        //and the expression type is like range {start: i64, end: i64, end_inclusive: bool, start_is_none: bool, end_is_none: bool}
        //The indexing could be done with either a single i64 or the range type
        //note that building this range struct may be slower than making the range
        //a compile time thing where start and end are just expressions. The downside
        //with this would be that it is not a value that can be passed around

    #[cache]
    rule unary() -> Expr
        //Having a _ before the () breaks in this case:
        //c = p.x + p.y + p.z
        //(p.x).print()
        = _ pos:position!() i:identifier() _macro:("!")? "(" args:((_ e:expression() _ {e}) ** comma()) ")" {
            Expr::Call(CodeRef::new(pos, code_ctx), i, args, _macro.is_some())
        }
        / _ pos:position!() i:identifier() _ "{" args:((_ e:struct_assign_field() _ {e})*) "}" { Expr::NewStruct(CodeRef::new(pos, code_ctx), i, args) }
        / _ pos:position!() i:identifier() { Expr::Identifier(CodeRef::new(pos, code_ctx), i) }
        / _ pos:position!() "(" e:expression() _ ")" { Expr::Parentheses(CodeRef::new(pos, code_ctx), Box::new(e)) }
        / _ pos:position!() "\"" body:$[^'"']* "\"" { Expr::LiteralString(CodeRef::new(pos, code_ctx), body.join("")) }
        / _ pos:position!() "[" e:expression()  _ ";" _ len:$(['0'..='9']+) _ "]" {
            Expr::LiteralArray(CodeRef::new(pos, code_ctx), vec![e], len.parse::<usize>().unwrap())
        }
        / _ pos:position!() "[" exprs:(e:expression() ** comma() {e}) _ "]" {
            let len = exprs.len();
            Expr::LiteralArray(CodeRef::new(pos, code_ctx), exprs, len)
        }
        / l:literal() { l }

    rule identifier() -> String
        = n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']* "::"? ['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*)  {?
            if RESERVED_WORDS.contains(&n) {Err("keyword found, expected identifier")} else {Ok(n.into())}
        }

    rule literal() -> Expr
        = _ pos:position!() n:$(['0'..='9']+) "u8" { Expr::LiteralU8(CodeRef::new(pos, code_ctx), n.parse::<u8>().unwrap()) }
        / _ pos:position!() n:$(['-']?['0'..='9']+"."['0'..='9']+) { Expr::LiteralFloat(CodeRef::new(pos, code_ctx), n.parse::<f32>().unwrap()) }
        / _ pos:position!() n:$(['-']?['0'..='9']+) { Expr::LiteralInt(CodeRef::new(pos, code_ctx), n.parse::<i64>().unwrap()) }
        / _ pos:position!() "*" i:identifier() { Expr::GlobalDataAddr(CodeRef::new(pos, code_ctx), i) }
        / _ pos:position!() "true" { Expr::LiteralBool(CodeRef::new(pos, code_ctx), true) }
        / _ pos:position!() "false" { Expr::LiteralBool(CodeRef::new(pos, code_ctx), false) }

    rule struct_assign_field() -> StructAssignField
        = _ i:identifier() _ ":" _ e:expression() comma() { StructAssignField {field_name: i.into(), expr: e } }

    rule comment() -> ()
        = quiet!{"//" [^'\n']*"\n"}

    rule comma() = _ ","

    rule _() =  quiet!{comment() / [' ' | '\t' | '\n']}*
    rule require_ws() =  quiet!{comment() / [' ' | '\t' | '\n']}+

});

pub fn assign_op_to_assign(op: Binop, a: Expr, b: Expr) -> Expr {
    let b_code_ref = *b.get_code_ref();
    Expr::Assign(
        *a.clone().get_code_ref(),
        vec![a.clone()],
        vec![Expr::Binop(
            b_code_ref,
            op,
            Box::new(a),
            Box::new(Expr::Parentheses(b_code_ref, Box::new(b))),
        )],
    )
}

pub struct CodeContext<'a> {
    pub file_index: Option<u64>,
    pub code: &'a str,
}

//Parse file, adding CodeRef and includes
pub fn parse_with_context(
    code: &str,
    file: &Path,
) -> anyhow::Result<(Vec<Declaration>, Vec<PathBuf>)> {
    let mut ast = Vec::new();
    let mut file_index_table = Vec::new();
    parse_with_context_recursively(
        &mut ast,
        code,
        file,
        &mut file_index_table,
        &mut HashSet::new(),
    )?;
    Ok((ast, file_index_table))
}

//TODO includes here are naive and lack namespaces
pub fn parse_with_context_recursively(
    ast: &mut Vec<Declaration>,
    code: &str,
    file: &Path,
    files_index: &mut Vec<PathBuf>,
    seen_paths: &mut HashSet<String>,
) -> anyhow::Result<()> {
    trace!("parse_with_context {}", file.display());
    files_index.push(file.to_path_buf());
    let code_ctx = CodeContext {
        code,
        file_index: Some((files_index.len() - 1) as u64),
    };
    match parser::program(code, &code_ctx) {
        Ok(mut new_ast) => {
            for decl in &new_ast {
                match decl {
                    Declaration::Include(path_str) => {
                        trace!("Found path_str {}", path_str);
                        let path = Path::new(&path_str);
                        let (new_code, new_file) = if path.is_absolute() {
                            if !seen_paths.insert(path.display().to_string()) {
                                //We have already imported this
                                continue;
                            }
                            trace!("Loading file import at {:?}", path);
                            let new_code = match fs::read_to_string(path) {
                                Ok(new_code) => new_code,
                                Err(e) => {
                                    anyhow::bail!("File import error {} {}", path.display(), e)
                                }
                            };
                            (new_code, path.to_path_buf())
                        } else {
                            let new_path = dunce::canonicalize(file.parent().unwrap().join(path))?;
                            if !seen_paths.insert(new_path.display().to_string()) {
                                //We have already imported this
                                continue;
                            }
                            trace!("Loading file import at {}", new_path.display());
                            let new_code = match fs::read_to_string(new_path.clone()) {
                                Ok(new_code) => new_code,
                                Err(e) => {
                                    anyhow::bail!("File import error {} {}", new_path.display(), e)
                                }
                            };
                            (new_code, new_path)
                        };
                        parse_with_context_recursively(
                            ast,
                            &new_code,
                            &new_file,
                            files_index,
                            seen_paths,
                        )?;
                    }
                    _ => continue,
                }
            }
            ast.append(&mut new_ast);
        }
        Err(err) => anyhow::bail!("{:?} parser {}", file, err),
    }

    Ok(())
}

//No imports or file context
pub fn parse(code: &str) -> anyhow::Result<Vec<Declaration>> {
    let code_ctx = CodeContext {
        code,
        file_index: None,
    };
    Ok(parser::program(&code.replace("\r\n", "\n"), &code_ctx)?)
}
