use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Declaration, Expr, Function},
    jit::{SVariable, StructDef},
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
    #[error("Variable \"{0}\" does not exist")]
    UnknownVariable(String),
    #[error("Struct \"{0}\" does not exist")]
    UnknownStruct(String),
    #[error("Struct \"{0}\" does not have field \"{1}\"")]
    UnknownField(String, String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
    Void,
    Bool,
    F64,
    I64,
    UnboundedArrayF64,
    UnboundedArrayI64,
    Address,
    Tuple(Vec<ExprType>),
    Struct(Box<String>),
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
            ExprType::Address => write!(f, "&"),
            ExprType::Tuple(inner) => {
                write!(f, "(")?;
                inner
                    .iter()
                    .map(|t| write!(f, "{}, ", t))
                    .collect::<Result<Vec<_>, _>>()?;
                write!(f, ")")
            }
            ExprType::Struct(s) => write!(f, "{}", s),
        }
    }
}

fn get_struct_field_type(
    struct_name: &str,
    field_name: &str,
    struct_map: &HashMap<String, StructDef>,
) -> Result<ExprType, TypeError> {
    Ok(if struct_map.contains_key(struct_name) {
        if struct_map[struct_name].fields.contains_key(field_name) {
            struct_map[struct_name].fields[field_name].expr_type.clone()
        } else {
            return Err(TypeError::UnknownField(
                struct_name.to_string(),
                field_name.to_string(),
            ));
        }
    } else {
        return Err(TypeError::UnknownStruct(struct_name.to_string()));
    })
}

impl ExprType {
    pub fn of(
        expr: &Expr,
        env: &[Declaration],
        funcs: &HashMap<String, Function>,
        variables: &HashMap<String, SVariable>,
        constant_vars: &HashMap<String, f64>,
        struct_map: &HashMap<String, StructDef>,
    ) -> Result<ExprType, TypeError> {
        let res = match expr {
            Expr::Identifier(id_name) => {
                if id_name.contains(".") {
                    let parts = id_name.split(".").collect::<Vec<&str>>();
                    if variables.contains_key(parts[0]) {
                        match &variables[parts[0]] {
                            SVariable::Struct(_var_name, struct_name, _var) => {
                                let mut struct_name = struct_name.to_string();
                                for i in 1..parts.len() {
                                    let next_expr =
                                        get_struct_field_type(&struct_name, parts[i], struct_map)?;
                                    if let ExprType::Struct(s) = next_expr.clone() {
                                        struct_name = s.to_string();
                                    }
                                    if i == parts.len() - 1 {
                                        return Ok(next_expr);
                                    }
                                }
                                unreachable!("should have already returned")
                            }
                            _v => {
                                return Err(TypeError::TypeMismatchSpecific {
                                    s: format!("{} is not a Struct", id_name),
                                });
                            }
                        }
                    } else {
                        return Err(TypeError::UnknownVariable(id_name.to_string()));
                    }
                } else if variables.contains_key(id_name) {
                    match &variables[id_name] {
                        SVariable::Unknown(_, _) => ExprType::F64, //TODO
                        SVariable::Bool(_, _) => ExprType::Bool,
                        SVariable::F64(_, _) => ExprType::F64,
                        SVariable::I64(_, _) => ExprType::I64,
                        SVariable::UnboundedArrayF64(_, _) => ExprType::UnboundedArrayF64,
                        SVariable::UnboundedArrayI64(_, _) => ExprType::UnboundedArrayI64,
                        SVariable::Address(_, _) => ExprType::Address,
                        SVariable::Struct(_, structname, _) => {
                            ExprType::Struct(Box::new(structname.to_string()))
                        }
                    }
                } else if constant_vars.contains_key(id_name) {
                    ExprType::F64 //All constants are currently math like PI, TAU...
                } else {
                    return Err(TypeError::UnknownVariable(id_name.to_string()));
                }
            }
            Expr::LiteralFloat(_) => ExprType::F64,
            Expr::LiteralInt(_) => ExprType::I64,
            Expr::LiteralBool(_) => ExprType::Bool,
            Expr::LiteralString(_) => ExprType::UnboundedArrayI64, //TODO change to char
            Expr::Binop(_, l, r) => {
                let lt = ExprType::of(l, env, funcs, variables, constant_vars, struct_map)?;
                let rt = ExprType::of(r, env, funcs, variables, constant_vars, struct_map)?;
                if lt == rt {
                    lt
                } else {
                    return Err(TypeError::TypeMismatch {
                        expected: lt,
                        actual: rt,
                    });
                }
            }
            Expr::Unaryop(_, l) => {
                ExprType::of(l, env, funcs, variables, constant_vars, struct_map)?
            }
            Expr::Compare(_, _, _) => ExprType::Bool,
            Expr::IfThen(econd, _) => {
                let tcond = ExprType::of(econd, env, funcs, variables, constant_vars, struct_map)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }
                ExprType::Void
            }
            Expr::IfElse(econd, etrue, efalse) => {
                let tcond = ExprType::of(econd, env, funcs, variables, constant_vars, struct_map)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| ExprType::of(e, env, funcs, variables, constant_vars, struct_map))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void);
                let tfalse = efalse
                    .iter()
                    .map(|e| ExprType::of(e, env, funcs, variables, constant_vars, struct_map))
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
                    1 => ExprType::of(&e[0], env, funcs, variables, constant_vars, struct_map)?
                        .tuple_size(),
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
                        .map(|e| ExprType::of(e, env, funcs, variables, constant_vars, struct_map))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            Expr::AssignOp(_, _, e) => {
                ExprType::of(e, env, funcs, variables, constant_vars, struct_map)?
            }
            Expr::WhileLoop(_, _) => ExprType::Void,
            Expr::Block(b) => b
                .iter()
                .map(|e| ExprType::of(e, env, funcs, variables, constant_vars, struct_map))
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void),
            Expr::Call(fn_name, args, impl_func) => {
                if *impl_func {
                    if let Some(self_var) = variables.get(&args[0].to_string()) {
                        let e = Expr::Call(
                            format!("{}.{}", self_var.type_name().unwrap(), fn_name),
                            args.to_vec(),
                            false,
                        );
                        return Ok(ExprType::of(
                            &e,
                            env,
                            funcs,
                            variables,
                            constant_vars,
                            struct_map,
                        )?);
                    } else {
                        return Err(TypeError::UnknownVariable(args[0].to_string()));
                    }
                }
                if let Some(d) = funcs.get(fn_name) {
                    if d.params.len() == args.len() {
                        //TODO make sure types match too
                        let targs: Result<Vec<_>, _> = args
                            .iter()
                            .map(|e| {
                                ExprType::of(e, env, funcs, variables, constant_vars, struct_map)
                            })
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
            Expr::Parentheses(expr) => {
                ExprType::of(expr, env, funcs, variables, constant_vars, struct_map)?
            }
            Expr::ArraySet(_, _, e) => {
                ExprType::of(e, env, funcs, variables, constant_vars, struct_map)?
            }
            Expr::ArrayGet(id_name, _) => {
                if variables.contains_key(id_name) {
                    match &variables[id_name] {
                        SVariable::UnboundedArrayF64(_, _) => Ok(ExprType::F64),
                        SVariable::UnboundedArrayI64(_, _) => Ok(ExprType::I64),
                        _ => Err(TypeError::TypeMismatchSpecific {
                            s: format!("{} is not an array", id_name),
                        }),
                    }
                } else {
                    return Err(TypeError::UnknownVariable(id_name.to_string()));
                }?
            }
            Expr::NewStruct(struct_name, _fields) => {
                if struct_map.contains_key(struct_name) {
                    //Need to check field types
                } else {
                    return Err(TypeError::UnknownStruct(struct_name.to_string()));
                }
                ExprType::Struct(Box::new(struct_name.to_string()))
            }
        };
        Ok(res)
    }

    pub fn tuple_size(&self) -> usize {
        match self {
            ExprType::Void => 0,
            ExprType::Bool
            | ExprType::F64
            | ExprType::I64
            | ExprType::Address
            | ExprType::Struct(_)
            | ExprType::UnboundedArrayF64
            | ExprType::UnboundedArrayI64 => 1,
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
            ExprType::Address => Ok(ptr_type),
            ExprType::Struct(_) => Ok(ptr_type),
            ExprType::Tuple(_) => Err(TypeError::TypeMismatchSpecific {
                s: "Tuple has no cranelift analog".to_string(),
            }),
        }
    }
}

pub fn validate_program(
    stmts: &Vec<Expr>,
    env: &[Declaration],
    funcs: &HashMap<String, Function>,
    variables: &HashMap<String, SVariable>,
    constant_vars: &HashMap<String, f64>,
    struct_map: &HashMap<String, StructDef>,
) -> Result<(), TypeError> {
    for expr in stmts {
        ExprType::of(expr, env, funcs, variables, constant_vars, struct_map)?;
    }
    Ok(())
}
