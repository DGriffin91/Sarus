use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Declaration, Expr, Function},
    jit::{SVariable, StructDef},
};
use thiserror::Error;

//TODO Make errors more information rich, also: show line in this file, and line in source
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

#[derive(Debug, Clone)]
pub struct Lval {
    pub expr: Vec<Expr>,
    pub ty: ExprType,
}

impl ExprType {
    pub fn of(
        expr: &Expr,
        lval: &mut Option<Lval>,
        env: &[Declaration],
        funcs: &HashMap<String, Function>,
        variables: &HashMap<String, SVariable>,
        constant_vars: &HashMap<String, f64>,
        struct_map: &HashMap<String, StructDef>,
    ) -> Result<ExprType, TypeError> {
        let res = match expr {
            Expr::Identifier(id_name) => {
                if let Some(lval) = lval {
                    let mut parts = Vec::new();
                    for expr in &lval.expr {
                        if let Expr::Identifier(s) = expr {
                            parts.push(s.as_str());
                        } else {
                            dbg!(&lval);
                            panic!("non identifier found")
                        }
                    }
                    parts.push(id_name);
                    //TODO Refactor?
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
            Expr::LiteralString(_) => ExprType::Address, //TODO change to char
            Expr::Binop(binop, l, r) => match binop {
                crate::frontend::Binop::DotAccess => {
                    //keep on looking at rt until you hit the end or a func
                    //find the result type of fields with struct_map and funcs with Type.func_name() in funcs
                    let lt =
                        ExprType::of(l, lval, env, funcs, variables, constant_vars, struct_map)?;

                    //<<< this is the same as below
                    if let Some(lval) = lval {
                        lval.expr.push((**l).clone());
                        lval.ty = lt;
                    } else {
                        *lval = Some(Lval {
                            expr: vec![(**l).clone()],
                            ty: lt,
                        })
                    }

                    let rt =
                        ExprType::of(r, lval, env, funcs, variables, constant_vars, struct_map)?;

                    let is_binop_dot = if let Expr::Binop(bt, _, _) = **r {
                        if let crate::frontend::Binop::DotAccess = bt {
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !is_binop_dot {
                        //<<< this is the same as above
                        if let Some(lval) = lval {
                            lval.expr.push((**r).clone());
                            lval.ty = rt.clone();
                        } else {
                            *lval = Some(Lval {
                                expr: vec![(**r).clone()],
                                ty: rt.clone(),
                            })
                        }
                    }

                    rt
                }
                _ => {
                    let lt = ExprType::of(
                        l,
                        &mut None,
                        env,
                        funcs,
                        variables,
                        constant_vars,
                        struct_map,
                    )?;
                    let rt = ExprType::of(
                        r,
                        &mut None,
                        env,
                        funcs,
                        variables,
                        constant_vars,
                        struct_map,
                    )?;
                    if lt == rt {
                        lt
                    } else {
                        return Err(TypeError::TypeMismatch {
                            expected: lt,
                            actual: rt,
                        });
                    }
                }
            },
            Expr::Unaryop(_, l) => ExprType::of(
                l,
                &mut None,
                env,
                funcs,
                variables,
                constant_vars,
                struct_map,
            )?,
            Expr::Compare(_, _, _) => ExprType::Bool,
            Expr::IfThen(econd, _) => {
                let tcond = ExprType::of(
                    econd,
                    &mut None,
                    env,
                    funcs,
                    variables,
                    constant_vars,
                    struct_map,
                )?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }
                ExprType::Void
            }
            Expr::IfElse(econd, etrue, efalse) => {
                let tcond = ExprType::of(
                    econd,
                    &mut None,
                    env,
                    funcs,
                    variables,
                    constant_vars,
                    struct_map,
                )?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| {
                        ExprType::of(
                            e,
                            &mut None,
                            env,
                            funcs,
                            variables,
                            constant_vars,
                            struct_map,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void);
                let tfalse = efalse
                    .iter()
                    .map(|e| {
                        ExprType::of(
                            e,
                            &mut None,
                            env,
                            funcs,
                            variables,
                            constant_vars,
                            struct_map,
                        )
                    })
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
                    1 => ExprType::of(
                        &e[0],
                        &mut None,
                        env,
                        funcs,
                        variables,
                        constant_vars,
                        struct_map,
                    )?
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
                        .map(|e| {
                            ExprType::of(
                                e,
                                &mut None,
                                env,
                                funcs,
                                variables,
                                constant_vars,
                                struct_map,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            Expr::AssignOp(_, _, e) => ExprType::of(
                e,
                &mut None,
                env,
                funcs,
                variables,
                constant_vars,
                struct_map,
            )?,
            Expr::WhileLoop(_, _) => ExprType::Void,
            Expr::Block(b) => b
                .iter()
                .map(|e| {
                    ExprType::of(
                        e,
                        &mut None,
                        env,
                        funcs,
                        variables,
                        constant_vars,
                        struct_map,
                    )
                })
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void),
            Expr::Call(fn_name, args) => {
                let fn_name = if let Some(lval) = &lval {
                    format!("{}.{}", lval.ty.to_string(), fn_name)
                } else {
                    fn_name.clone()
                };
                if let Some(func) = funcs.get(&fn_name) {
                    let mut targs = Vec::new();

                    if let Some(lval) = &lval {
                        targs.push(lval.ty.clone());
                    }

                    for e in args {
                        targs.push(ExprType::of(
                            e,
                            &mut None,
                            env,
                            funcs,
                            variables,
                            constant_vars,
                            struct_map,
                        )?);
                    }

                    if func.params.len() != targs.len() {
                        return Err(TypeError::TupleLengthMismatch {
                            //TODO be more specific: function {} expected {} parameters, but {} were given
                            actual: targs.len(),
                            expected: func.params.len(),
                        });
                    }

                    for (i, (targ, param)) in targs.iter().zip(func.params.iter()).enumerate() {
                        let param_type = param.expr_type.as_ref().unwrap_or(&ExprType::F64);
                        if *param_type == *targ {
                            continue;
                        } else {
                            return Err(TypeError::TypeMismatchSpecific {
                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param_type, targ)
                                });
                        }
                    }

                    match &func.returns {
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
                    }
                } else {
                    return Err(TypeError::UnknownFunction(fn_name.to_string()));
                }
            }
            //Expr::ExpressionCall(expr, fn_name, args) => {
            //    let mut args = args.to_vec();
            //    args.insert(0, *expr.to_owned());
            //    ExprType::of(
            //        &Expr::Call(fn_name.to_string(), args, true),
            //        &mut None,
            //        env,
            //        funcs,
            //        variables,
            //        constant_vars,
            //        struct_map,
            //    )?
            //}
            Expr::GlobalDataAddr(_) => ExprType::F64,
            Expr::Parentheses(expr) => ExprType::of(
                expr,
                &mut None,
                env,
                funcs,
                variables,
                constant_vars,
                struct_map,
            )?,
            Expr::ArraySet(_, _, e) => ExprType::of(
                e,
                &mut None,
                env,
                funcs,
                variables,
                constant_vars,
                struct_map,
            )?,
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
        ExprType::of(
            expr,
            &mut None,
            env,
            funcs,
            variables,
            constant_vars,
            struct_map,
        )?;
    }
    Ok(())
}
