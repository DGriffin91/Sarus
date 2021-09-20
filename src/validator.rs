use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Declaration, Expr},
    jit::SVariable,
};
use thiserror::Error;

//Reference: https://www.gnu.org/software/libc/manual/html_node/Mathematics.html
//https://docs.rs/libc/0.2.101/libc/
//should this include bessel functions? It seems like they would pollute the name space.

//couldn't get to work (STATUS_ACCESS_VIOLATION):
// "asinh", "acosh", "atanh", "erf", "erfc", "lgamma", "gamma", "tgamma", "exp2", "exp10", "log2"
const STD_1ARG_F: [&str; 21] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "log10", "sqrt", "sinh", "cosh",
    "exp10", "tanh", // libc
    "ceil", "floor", "trunc", "fract", "abs", "round", "float", // built in std
];
const STD_1ARG_I: [&str; 1] = [
    "int", // built in std
];

//couldn't get to work (STATUS_ACCESS_VIOLATION):
// "hypot", "expm1", "log1p"
const STD_2ARG_F: [&str; 4] = [
    "atan2", "pow", // libc
    "min", "max", // built in std
];
const STD_2ARG_I: [&str; 2] = [
    "imin", "imax", // built in std
];

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("Type mismatch; expected {expected}, found {actual}")]
    TypeMismatch { expected: SType, actual: SType },
    #[error("Tuple length mismatch; expected {expected} found {actual}")]
    TupleLengthMismatch { expected: usize, actual: usize },
    #[error("Function \"{0}\" does not exist")]
    UnknownFunction(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SType {
    Void,
    Bool,
    Float,
    Int,
    Address,
    Tuple(Vec<SType>),
}

impl Display for SType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SType::Void => write!(f, "void"),
            SType::Bool => write!(f, "bool"),
            SType::Float => write!(f, "float"),
            SType::Int => write!(f, "int"),
            SType::Address => write!(f, "address"),
            SType::Tuple(inner) => {
                write!(f, "(")?;
                inner
                    .iter()
                    .map(|t| write!(f, "{}, ", t))
                    .collect::<Result<Vec<_>, _>>()?;
                write!(f, ")")
            }
        }
    }
}

impl SType {
    pub fn of(
        expr: &Expr,
        env: &[Declaration],
        variables: &HashMap<String, SVariable>,
    ) -> Result<SType, TypeError> {
        let res = match expr {
            //TODO don't assume all identifiers are floats
            Expr::Identifier(id_name) => {
                if variables.contains_key(id_name) {
                    match variables[id_name] {
                        SVariable::Unknown(_, _) => SType::Address,
                        SVariable::Bool(_, _) => SType::Bool,
                        SVariable::Float(_, _) => SType::Float,
                        SVariable::Int(_, _) => SType::Int,
                        SVariable::Address(_, _) => SType::Address,
                    }
                } else {
                    //This doesn't really make sense.
                    //The validator needs to be aware of previous vars
                    SType::Float
                }
            }
            Expr::LiteralFloat(_) => SType::Float,
            Expr::LiteralInt(_) => SType::Int,
            Expr::Binop(_, l, r) => {
                let lt = SType::of(l, env, variables)?;
                let rt = SType::of(r, env, variables)?;
                if lt == rt {
                    lt
                } else {
                    return Err(TypeError::TypeMismatch {
                        expected: lt,
                        actual: rt,
                    });
                }
            }
            Expr::Compare(_, _, _) => SType::Bool,
            Expr::IfThen(econd, _) => {
                let tcond = SType::of(econd, env, variables)?;
                if tcond != SType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: SType::Bool,
                        actual: tcond,
                    });
                }
                SType::Void
            }
            Expr::IfElse(econd, etrue, efalse) => {
                let tcond = SType::of(econd, env, variables)?;
                if tcond != SType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: SType::Bool,
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| SType::of(e, env, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(SType::Void);
                let tfalse = efalse
                    .iter()
                    .map(|e| SType::of(e, env, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(SType::Void);

                if ttrue == tfalse {
                    ttrue
                } else {
                    return Err(TypeError::TypeMismatch {
                        expected: ttrue,
                        actual: tfalse,
                    });
                }
            }
            Expr::Assign(vars, e) => {
                let tlen = match e.len().into() {
                    1 => SType::of(&e[0], env, variables)?.tuple_size(),
                    n => n,
                };
                if usize::from(vars.len()) != tlen {
                    return Err(TypeError::TupleLengthMismatch {
                        actual: usize::from(vars.len()),
                        expected: tlen,
                    });
                }
                SType::Tuple(
                    e.iter()
                        .map(|e| SType::of(e, env, variables))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            Expr::AssignOp(_, _, e) => SType::of(e, env, variables)?,
            Expr::WhileLoop(_, _) => SType::Void,
            Expr::Block(b) => b
                .iter()
                .map(|e| SType::of(e, env, variables))
                .last()
                .map(Result::unwrap)
                .unwrap_or(SType::Void),
            Expr::Call(fn_name, args) => {
                if let Some(d) = env.iter().find_map(|d| match d {
                    Declaration::Function(func) => {
                        if &func.name == fn_name {
                            Some(func)
                        } else {
                            None
                        }
                    }
                    _ => None,
                }) {
                    if d.params.len() == args.len() {
                        let targs: Result<Vec<_>, _> =
                            args.iter().map(|e| SType::of(e, env, variables)).collect();
                        match targs {
                            Ok(_) => match &d.returns {
                                v if v.is_empty() => SType::Void,
                                v if v.len() == 1 => SType::Float,
                                v => SType::Tuple(vec![SType::Float; v.len()]),
                            },
                            Err(err) => return Err(err),
                        }
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: d.params.len(),
                            actual: args.len(),
                        });
                    }
                } else if STD_1ARG_F.contains(&fn_name.as_str()) {
                    if args.len() == 1 {
                        SType::Float
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: 1,
                            actual: args.len(),
                        });
                    }
                } else if STD_2ARG_F.contains(&fn_name.as_str()) {
                    if args.len() == 2 {
                        SType::Float
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: 2,
                            actual: args.len(),
                        });
                    }
                } else if STD_1ARG_I.contains(&fn_name.as_str()) {
                    if args.len() == 1 {
                        SType::Int
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: 1,
                            actual: args.len(),
                        });
                    }
                } else if STD_2ARG_I.contains(&fn_name.as_str()) {
                    if args.len() == 2 {
                        SType::Int
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: 2,
                            actual: args.len(),
                        });
                    }
                } else {
                    return Err(TypeError::UnknownFunction(fn_name.to_string()));
                }
            }
            Expr::GlobalDataAddr(_) => SType::Float,
            Expr::Bool(_) => SType::Bool,
            Expr::Parentheses(expr) => SType::of(expr, env, variables)?,
            Expr::ArraySet(_, _, e) => SType::of(e, env, variables)?,
            Expr::ArrayGet(_, _) => SType::Float,
        };
        Ok(res)
    }

    pub fn tuple_size(&self) -> usize {
        match self {
            SType::Void => 0,
            SType::Bool | SType::Float | SType::Address | SType::Int => 1,
            SType::Tuple(v) => v.len(),
        }
    }
}

pub fn validate_program(decls: Vec<Declaration>) -> Result<Vec<Declaration>, TypeError> {
    let variables = HashMap::new(); //Previously declared variables, used in jit
    for func in decls.iter().filter_map(|d| match d {
        Declaration::Function(func) => Some(func.clone()),
        _ => None,
    }) {
        for expr in &func.body {
            SType::of(expr, &decls, &variables)?;
        }
    }
    Ok(decls)
}
