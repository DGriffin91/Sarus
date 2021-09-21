use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Declaration, Expr},
    jit::SVariable,
};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("Type mismatch; expected {expected}, found {actual}")]
    TypeMismatch {
        expected: ExprType,
        actual: ExprType,
    },
    #[error("Type mismatch; {s}")]
    TypeMismatchSpecific { s: String },
    #[error("Tuple length mismatch; expected {expected} found {actual}")]
    TupleLengthMismatch { expected: usize, actual: usize },
    #[error("Function \"{0}\" does not exist")]
    UnknownFunction(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
    Void,
    Bool,
    F64,
    I64,
    UnboundedArrayF64,
    UnboundedArrayI64,
    Tuple(Vec<ExprType>),
}

impl Display for ExprType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprType::Void => write!(f, "void"),
            ExprType::Bool => write!(f, "bool"),
            ExprType::F64 => write!(f, "f64"),
            ExprType::I64 => write!(f, "i64"),
            ExprType::UnboundedArrayF64 => write!(f, "&[f64]"),
            ExprType::UnboundedArrayI64 => write!(f, "&[i64]"),
            ExprType::Tuple(inner) => {
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

impl ExprType {
    pub fn of(
        expr: &Expr,
        env: &[Declaration],
        variables: &HashMap<String, SVariable>,
    ) -> Result<ExprType, TypeError> {
        let res = match expr {
            //TODO don't assume all identifiers are floats
            Expr::Identifier(id_name) => {
                if variables.contains_key(id_name) {
                    match variables[id_name] {
                        SVariable::Unknown(_, _) => ExprType::F64, //TODO
                        SVariable::Bool(_, _) => ExprType::Bool,
                        SVariable::F64(_, _) => ExprType::F64,
                        SVariable::I64(_, _) => ExprType::I64,
                        SVariable::UnboundedArrayF64(_, _) => ExprType::UnboundedArrayF64,
                        SVariable::UnboundedArrayI64(_, _) => ExprType::UnboundedArrayI64,
                    }
                } else {
                    //This doesn't really make sense.
                    //The validator needs to be aware of previous vars
                    ExprType::F64
                }
            }
            Expr::LiteralFloat(_) => ExprType::F64,
            Expr::LiteralInt(_) => ExprType::I64,
            Expr::Binop(_, l, r) => {
                let lt = ExprType::of(l, env, variables)?;
                let rt = ExprType::of(r, env, variables)?;
                if lt == rt {
                    lt
                } else {
                    return Err(TypeError::TypeMismatch {
                        expected: lt,
                        actual: rt,
                    });
                }
            }
            Expr::Compare(_, _, _) => ExprType::Bool,
            Expr::IfThen(econd, _) => {
                let tcond = ExprType::of(econd, env, variables)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }
                ExprType::Void
            }
            Expr::IfElse(econd, etrue, efalse) => {
                let tcond = ExprType::of(econd, env, variables)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| ExprType::of(e, env, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void);
                let tfalse = efalse
                    .iter()
                    .map(|e| ExprType::of(e, env, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void);

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
                    1 => ExprType::of(&e[0], env, variables)?.tuple_size(),
                    n => n,
                };
                if usize::from(vars.len()) != tlen {
                    return Err(TypeError::TupleLengthMismatch {
                        actual: usize::from(vars.len()),
                        expected: tlen,
                    });
                }
                ExprType::Tuple(
                    e.iter()
                        .map(|e| ExprType::of(e, env, variables))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            Expr::AssignOp(_, _, e) => ExprType::of(e, env, variables)?,
            Expr::WhileLoop(_, _) => ExprType::Void,
            Expr::Block(b) => b
                .iter()
                .map(|e| ExprType::of(e, env, variables))
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void),
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
                        //TODO make sure types match too
                        let targs: Result<Vec<_>, _> = args
                            .iter()
                            .map(|e| ExprType::of(e, env, variables))
                            .collect();
                        match targs {
                            Ok(_) => match &d.returns {
                                v if v.is_empty() => ExprType::Void,
                                v if v.len() == 1 => v
                                    .first()
                                    .unwrap()
                                    .expr_type
                                    .clone()
                                    .unwrap_or(ExprType::F64),
                                v => {
                                    let mut items = Vec::new();
                                    for arg in v.iter() {
                                        items.push(arg.expr_type.clone().unwrap_or(ExprType::F64));
                                    }
                                    ExprType::Tuple(items)
                                }
                            },
                            Err(err) => return Err(err),
                        }
                    } else {
                        return Err(TypeError::TupleLengthMismatch {
                            expected: d.params.len(),
                            actual: args.len(),
                        });
                    }
                } else {
                    return Err(TypeError::UnknownFunction(fn_name.to_string()));
                }
            }
            Expr::GlobalDataAddr(_) => ExprType::F64,
            Expr::Bool(_) => ExprType::Bool,
            Expr::Parentheses(expr) => ExprType::of(expr, env, variables)?,
            Expr::ArraySet(_, _, e) => ExprType::of(e, env, variables)?,
            Expr::ArrayGet(_, _) => ExprType::F64,
        };
        Ok(res)
    }

    pub fn tuple_size(&self) -> usize {
        match self {
            ExprType::Void => 0,
            ExprType::Bool
            | ExprType::F64
            | ExprType::UnboundedArrayF64
            | ExprType::UnboundedArrayI64
            | ExprType::I64 => 1,
            ExprType::Tuple(v) => v.len(),
        }
    }

    pub fn cranelift_type(
        &self,
        ptr_type: cranelift::prelude::Type,
    ) -> Result<cranelift::prelude::Type, TypeError> {
        match self {
            ExprType::Void => Err(TypeError::TypeMismatchSpecific {
                s: "Void has no cranelift analog".to_string(),
            }),
            ExprType::Bool => Ok(cranelift::prelude::types::B1),
            ExprType::F64 => Ok(cranelift::prelude::types::F64),
            ExprType::I64 => Ok(cranelift::prelude::types::I64),
            ExprType::UnboundedArrayI64 => Ok(ptr_type),
            ExprType::UnboundedArrayF64 => Ok(ptr_type),
            ExprType::Tuple(_) => Err(TypeError::TypeMismatchSpecific {
                s: "Tuple has no cranelift analog".to_string(),
            }),
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
            ExprType::of(expr, &decls, &variables)?;
        }
    }
    Ok(decls)
}
