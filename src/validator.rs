use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Binop, CodeRef, Expr},
    jit::{Env, SVariable, StructDef},
};
use cranelift::prelude::types;
use thiserror::Error;
use tracing::{error, trace};

//TODO Make errors more information rich, also: show line in this file, and line in source
#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("{c} Type mismatch; expected {expected}, found {actual}")]
    TypeMismatch {
        c: String,
        expected: ExprType,
        actual: ExprType,
    },
    #[error("{c} Type mismatch; {s}")]
    TypeMismatchSpecific { c: String, s: String },
    #[error("{c} Tuple length mismatch; expected {expected} found {actual}")]
    TupleLengthMismatch {
        c: String,
        expected: usize,
        actual: usize,
    },
    #[error("{0} Function \"{1}\" does not exist")]
    UnknownFunction(String, String),
    #[error("{0} Variable \"{1}\" does not exist")]
    UnknownVariable(String, String),
    #[error("{0} Struct \"{1}\" does not exist")]
    UnknownStruct(String, String),
    #[error("{0} Struct \"{1}\" does not have field \"{2}\"")]
    UnknownField(String, String, String),
    #[error("{0} Expression \"{1}\" is not supported")]
    UnsupportedExpr(String, String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArraySizedExpr {
    Unsized,      //size is unknown, just an address with a type
    Sized,        //start of array address is i64 with size.
    Fixed(usize), //size is part of type signature
}

#[derive(Debug, Clone)]
pub enum ExprType {
    Void(CodeRef),
    Bool(CodeRef),
    F32(CodeRef),
    I64(CodeRef),
    Array(CodeRef, Box<ExprType>, ArraySizedExpr),
    Address(CodeRef),
    Tuple(CodeRef, Vec<ExprType>),
    Struct(CodeRef, Box<String>),
}

pub fn f32_t() -> ExprType {
    ExprType::F32(CodeRef::z())
}

pub fn i64_t() -> ExprType {
    ExprType::I64(CodeRef::z())
}

pub fn bool_t() -> ExprType {
    ExprType::Bool(CodeRef::z())
}

pub fn address_t() -> ExprType {
    ExprType::Address(CodeRef::z())
}

pub fn struct_t(name: &str) -> ExprType {
    ExprType::Struct(CodeRef::z(), Box::new(name.to_string()))
}

pub fn array_t(ty: ExprType, size_type: ArraySizedExpr) -> ExprType {
    ExprType::Array(CodeRef::z(), Box::new(ty), size_type)
}

impl PartialEq for ExprType {
    fn eq(&self, other: &Self) -> bool {
        match self {
            ExprType::Void(_) => {
                if let ExprType::Void(_) = other {
                    return true;
                }
            }
            ExprType::Bool(_) => {
                if let ExprType::Bool(_) = other {
                    return true;
                }
            }
            ExprType::F32(_) => {
                if let ExprType::F32(_) = other {
                    return true;
                }
            }
            ExprType::I64(_) => {
                if let ExprType::I64(_) = other {
                    return true;
                }
            }
            ExprType::Array(_, a, sa) => {
                if let ExprType::Array(_, b, sb) = other {
                    return a == b && sa == sb;
                }
            }
            ExprType::Address(_) => {
                if let ExprType::Address(_) = other {
                    return true;
                }
            }
            ExprType::Tuple(_, ta) => {
                if let ExprType::Tuple(_, tb) = other {
                    return ta == tb;
                }
            }
            ExprType::Struct(_, sa) => {
                if let ExprType::Struct(_, sb) = other {
                    return sa == sb;
                }
            }
        }
        false
    }
}

impl Display for ExprType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprType::Void(_) => write!(f, "void"),
            ExprType::Bool(_) => write!(f, "bool"),
            ExprType::F32(_) => write!(f, "f32"),
            ExprType::I64(_) => write!(f, "i64"),
            ExprType::Array(_, ty, size_type) => match size_type {
                ArraySizedExpr::Unsized => write!(f, "&[{}]", ty),
                ArraySizedExpr::Sized => todo!(),
                ArraySizedExpr::Fixed(len) => write!(f, "[{}; {}]", ty, len),
            },
            ExprType::Address(_) => write!(f, "&"),
            ExprType::Tuple(_, inner) => {
                write!(f, "(")?;
                inner
                    .iter()
                    .map(|t| write!(f, "{}, ", t))
                    .collect::<Result<Vec<_>, _>>()?;
                write!(f, ")")
            }
            ExprType::Struct(_, s) => write!(f, "{}", s),
        }
    }
}

fn get_struct_field_type(
    env: &Env,
    parts: Vec<String>,
    lhs_val: &Option<ExprType>,
    code_ref: &CodeRef,
    variables: &HashMap<String, SVariable>,
) -> Result<ExprType, TypeError> {
    let (lhs_val, start) = if let Some(lhs_val) = lhs_val {
        (lhs_val.clone(), 0)
    } else {
        let id_name = &parts[0];
        if let Some(svar) = variables.get(id_name) {
            (svar.expr_type(code_ref).unwrap(), 1)
        } else {
            error!("");
            return Err(TypeError::UnknownVariable(
                code_ref.s(&env.file_idx),
                parts[0].to_string(),
            ));
        }
    };
    match lhs_val {
        ExprType::Struct(_code_ref, struct_name) => {
            let mut struct_name = *struct_name;
            let mut parent_struct_field = if let Some(s) = env.struct_map.get(&struct_name) {
                if let Some(parent_struct_field) = s.fields.get(&parts[start]) {
                    parent_struct_field
                } else {
                    return Err(TypeError::UnknownField(
                        code_ref.s(&env.file_idx),
                        struct_name,
                        parts[start].to_string(),
                    ));
                }
            } else {
                dbg!(&env.struct_map);
                return Err(TypeError::UnknownStruct(
                    code_ref.s(&env.file_idx),
                    struct_name,
                ));
            };

            if parts.len() > 2 {
                for i in start..parts.len() {
                    if let ExprType::Struct(code_ref, _name) = &parent_struct_field.expr_type {
                        parent_struct_field = if let Some(parent_struct_field) =
                            env.struct_map[&struct_name].fields.get(&parts[i])
                        {
                            parent_struct_field
                        } else {
                            return Err(TypeError::UnknownField(
                                code_ref.s(&env.file_idx),
                                struct_name,
                                parts[start].to_string(),
                            ));
                        };
                        struct_name = parent_struct_field.expr_type.to_string().clone();
                    } else {
                        break;
                    }
                }
            }
            Ok(parent_struct_field.expr_type.clone())
        }
        v => {
            error!("");
            Err(TypeError::TypeMismatch {
                c: v.get_code_ref().s(&env.file_idx),
                expected: ExprType::Struct(v.get_code_ref(), Box::new("".to_string())),
                actual: v,
            })
        }
    }
}

impl ExprType {
    pub fn get_code_ref(&self) -> CodeRef {
        *match self {
            ExprType::Array(code_ref, ..) => code_ref,
            ExprType::Void(code_ref) => code_ref,
            ExprType::Bool(code_ref) => code_ref,
            ExprType::F32(code_ref) => code_ref,
            ExprType::I64(code_ref) => code_ref,
            ExprType::Address(code_ref) => code_ref,
            ExprType::Tuple(code_ref, ..) => code_ref,
            ExprType::Struct(code_ref, ..) => code_ref,
        }
    }
    pub fn replace_code_ref(&mut self, new_code_ref: CodeRef) {
        match self {
            ExprType::Array(code_ref, ..) => *code_ref = new_code_ref,
            ExprType::Void(code_ref) => *code_ref = new_code_ref,
            ExprType::Bool(code_ref) => *code_ref = new_code_ref,
            ExprType::F32(code_ref) => *code_ref = new_code_ref,
            ExprType::I64(code_ref) => *code_ref = new_code_ref,
            ExprType::Address(code_ref) => *code_ref = new_code_ref,
            ExprType::Tuple(code_ref, ..) => *code_ref = new_code_ref,
            ExprType::Struct(code_ref, ..) => *code_ref = new_code_ref,
        }
    }

    pub fn width(
        &self,
        ptr_ty: types::Type,
        struct_map: &HashMap<String, StructDef>,
    ) -> Option<usize> {
        match self {
            ExprType::Array(_, ty, size_type) => match size_type {
                ArraySizedExpr::Unsized => None,
                ArraySizedExpr::Sized => todo!(),
                ArraySizedExpr::Fixed(len) => ty.width(ptr_ty, struct_map).map(|width| width * len),
            },
            ExprType::Void(_) => Some(0),
            ExprType::Bool(_) => Some(types::I8.bytes() as usize),
            ExprType::F32(_) => Some(types::F32.bytes() as usize),
            ExprType::I64(_) => Some(types::I64.bytes() as usize),
            ExprType::Address(_) => Some(ptr_ty.bytes() as usize),
            ExprType::Tuple(_code_ref, _expr_types) => None,
            ExprType::Struct(_code_ref, name) => Some(struct_map[&name.to_string()].size),
        }
    }

    pub fn of(
        expr: &Expr,
        env: &Env,
        func_name: &str,
        variables: &HashMap<String, SVariable>,
    ) -> Result<ExprType, TypeError> {
        trace!("of {}", expr);
        let res = match expr {
            Expr::Identifier(code_ref, id_name) => {
                if variables.contains_key(id_name) {
                    variables[id_name].expr_type(code_ref).unwrap()
                } else if let Some(v) = env.constant_vars.get(id_name) {
                    v.expr_type(Some(*code_ref)) //Constants like PI, TAU...
                } else if let Some(_closure) = env.get_inline_closure(func_name, id_name) {
                    ExprType::Void(*code_ref)
                } else {
                    error!("{:#?}", variables);
                    return Err(TypeError::UnknownVariable(
                        code_ref.s(&env.file_idx),
                        id_name.to_string(),
                    ));
                }
            }
            Expr::LiteralFloat(code_ref, _) => ExprType::F32(*code_ref),
            Expr::LiteralInt(code_ref, _) => ExprType::I64(*code_ref),
            Expr::LiteralBool(code_ref, _) => ExprType::Bool(*code_ref),
            Expr::LiteralString(code_ref, _) => ExprType::Address(*code_ref), //TODO change to char
            Expr::LiteralArray(code_ref, e, len) => ExprType::Array(
                *code_ref,
                Box::new(ExprType::of(e, env, func_name, variables)?),
                ArraySizedExpr::Fixed(*len),
            ),
            Expr::Binop(binop_code_ref, binop, lhs, rhs) => match binop {
                crate::frontend::Binop::DotAccess => {
                    let mut path = Vec::new();
                    let mut lhs_val = None;

                    let mut curr_expr = Some(*lhs.to_owned());
                    let mut next_expr = Some(*rhs.to_owned());

                    while let Some(rhs_expr) = curr_expr.clone() {
                        //println!("curr_expr {:?} next_expr {:?}", &curr_expr, &next_expr);
                        //println!("path {:?}", &path);

                        curr_expr = next_expr;
                        next_expr = None;
                        match rhs_expr.clone() {
                            Expr::Call(code_ref, fn_name, args, is_macro) => {
                                if is_macro {
                                    todo!("binop macros not supported yet")
                                }
                                let sval = if path.is_empty() {
                                    if let Some(lhs_val) = lhs_val {
                                        lhs_val
                                    } else {
                                        //This is a lhs call
                                        lhs_val = Some(ExprType::of(
                                            &rhs_expr, env, func_name, variables,
                                        )?);
                                        continue;
                                    }
                                } else if path.len() > 1 || lhs_val.is_some() {
                                    let spath = path
                                        .iter()
                                        .map(|lhs_i: &Expr| lhs_i.to_string())
                                        .collect::<Vec<String>>();
                                    get_struct_field_type(
                                        env, spath, &lhs_val, &code_ref, variables,
                                    )?
                                } else {
                                    ExprType::of(&path[0], env, func_name, variables)?
                                };

                                let fn_name = format!("{}.{}", sval.to_string(), fn_name);

                                if !&env.funcs.contains_key(&fn_name) {
                                    return Err(TypeError::UnknownFunction(
                                        code_ref.s(&env.file_idx),
                                        fn_name,
                                    ));
                                }

                                let params = &env.funcs[&fn_name].params;

                                if params.len() - 1 != args.len() {
                                    return Err(TypeError::TupleLengthMismatch {
                                        //TODO be more specific: function {} expected {} parameters, but {} were given
                                        c: code_ref.s(&env.file_idx),
                                        actual: args.len(),
                                        expected: params.len() - 1,
                                    });
                                }

                                for (i, (param, arg)) in
                                    params.iter().skip(1).zip(args.iter()).enumerate()
                                {
                                    let targ = ExprType::of(arg, env, func_name, variables)?;
                                    if param.expr_type != targ {
                                        return Err(TypeError::TypeMismatchSpecific {
                                                    c: code_ref.s(&env.file_idx),
                                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param.expr_type , targ)
                                                });
                                    }
                                }

                                let returns = &env.funcs[&fn_name].returns;

                                if returns.is_empty() {
                                    lhs_val = Some(ExprType::Void(code_ref))
                                } else if returns.len() == 1 {
                                    lhs_val = Some(returns[0].expr_type.clone())
                                } else {
                                    let mut expr_types = Vec::new();
                                    for arg in returns {
                                        expr_types.push(arg.expr_type.clone())
                                    }
                                    lhs_val = Some(ExprType::Tuple(code_ref, expr_types))
                                }

                                path = Vec::new();
                            }
                            Expr::LiteralString(_code_ref, _) => {
                                lhs_val = Some(ExprType::of(&rhs_expr, env, func_name, variables)?);
                            }
                            Expr::LiteralArray(_code_ref, _, _) => {
                                lhs_val = Some(ExprType::of(&rhs_expr, env, func_name, variables)?);
                            }
                            Expr::Identifier(_code_ref, _i) => path.push(rhs_expr),
                            Expr::Binop(_code_ref, op, lhs, rhs) => {
                                if let Binop::DotAccess = op {
                                    curr_expr = Some(*lhs.clone());
                                    next_expr = Some(*rhs.clone());
                                } else {
                                    return Err(TypeError::UnsupportedExpr(
                                        expr.get_code_ref().s(&env.file_idx),
                                        expr.to_string(),
                                    ));
                                }
                            }

                            Expr::Parentheses(_code_ref, e) => {
                                lhs_val = Some(ExprType::of(&e, env, func_name, variables)?)
                            }
                            Expr::ArrayAccess(code_ref, name, idx_expr) => {
                                match ExprType::of(&idx_expr, env, func_name, variables)? {
                                    ExprType::I64(_code_ref) => (),
                                    e => {
                                        error!("");
                                        return Err(TypeError::TypeMismatch {
                                            c: code_ref.s(&env.file_idx),
                                            expected: ExprType::I64(code_ref),
                                            actual: e,
                                        });
                                    }
                                };
                                if !path.is_empty() {
                                    let mut spath = path
                                        .iter()
                                        .map(|lhs_i: &Expr| lhs_i.to_string())
                                        .collect::<Vec<String>>();
                                    spath.push(name.to_string());
                                    if let ExprType::Array(_code_ref, ty, _len) =
                                        get_struct_field_type(
                                            env, spath, &lhs_val, &code_ref, variables,
                                        )?
                                    {
                                        lhs_val = Some(*ty.to_owned());
                                    }
                                } else {
                                    lhs_val =
                                        Some(ExprType::of(&rhs_expr, env, func_name, variables)?);
                                }

                                path = Vec::new();
                            }
                            Expr::Declaration(..)
                            | Expr::Unaryop(..)
                            | Expr::Compare(..)
                            | Expr::IfThen(..)
                            | Expr::IfElse(..)
                            | Expr::IfThenElseIf(..)
                            | Expr::IfThenElseIfElse(..)
                            | Expr::Assign(..)
                            | Expr::AssignOp(..)
                            | Expr::NewStruct(..)
                            | Expr::WhileLoop(..)
                            | Expr::Block(..)
                            | Expr::GlobalDataAddr(..)
                            | Expr::LiteralFloat(..)
                            | Expr::LiteralInt(..)
                            | Expr::LiteralBool(..) => {
                                return Err(TypeError::UnsupportedExpr(
                                    expr.get_code_ref().s(&env.file_idx),
                                    expr.to_string(),
                                ))
                            }
                        }
                    }
                    if !path.is_empty() {
                        let spath = path
                            .iter()
                            .map(|lhs_i: &Expr| lhs_i.to_string())
                            .collect::<Vec<String>>();
                        lhs_val = Some(get_struct_field_type(
                            env,
                            spath,
                            &lhs_val,
                            binop_code_ref,
                            variables,
                        )?);
                    }

                    if let Some(lhs_val) = lhs_val {
                        return Ok(lhs_val);
                    } else {
                        panic!("No value found");
                    }
                }
                _ => {
                    let lt = ExprType::of(lhs, env, func_name, variables)?;
                    let rt = ExprType::of(rhs, env, func_name, variables)?;
                    if lt == rt {
                        lt
                    } else {
                        error!("");
                        return Err(TypeError::TypeMismatch {
                            c: binop_code_ref.s(&env.file_idx),
                            expected: lt,
                            actual: rt,
                        });
                    }
                }
            },
            Expr::Unaryop(_code_ref, _, l) => ExprType::of(l, env, func_name, variables)?,
            Expr::Compare(code_ref, _, _, _) => ExprType::Bool(*code_ref),
            Expr::IfThen(code_ref, econd, _) => {
                //TODO should we check ExprType::of of every line of body?
                let tcond = ExprType::of(econd, env, func_name, variables)?;
                if tcond != ExprType::Bool(*code_ref) {
                    error!("");
                    return Err(TypeError::TypeMismatch {
                        c: code_ref.s(&env.file_idx),
                        expected: ExprType::Bool(*code_ref),
                        actual: tcond,
                    });
                }
                ExprType::Void(*code_ref)
            }
            Expr::IfThenElseIf(code_ref, expr_bodies) => {
                //TODO should we check ExprType::of of every line of body?
                for (econd, _body) in expr_bodies {
                    let tcond = ExprType::of(econd, env, func_name, variables)?;
                    if tcond != ExprType::Bool(*code_ref) {
                        error!("");
                        return Err(TypeError::TypeMismatch {
                            c: code_ref.s(&env.file_idx),
                            expected: ExprType::Bool(*code_ref),
                            actual: tcond,
                        });
                    }
                }
                ExprType::Void(*code_ref)
            }
            //TODO this should work for IfThenElseIfElse
            Expr::IfThenElseIfElse(code_ref, expr_bodies, else_body) => {
                //TODO should we check ExprType::of of every line of body?
                let mut last_body_type = None;
                for (econd, body) in expr_bodies {
                    let tcond = ExprType::of(econd, env, func_name, variables)?;
                    if tcond != ExprType::Bool(*code_ref) {
                        error!("");
                        return Err(TypeError::TypeMismatch {
                            c: code_ref.s(&env.file_idx),
                            expected: ExprType::Bool(*code_ref),
                            actual: tcond,
                        });
                    }
                    let body_type = body
                        .iter()
                        .map(|e| ExprType::of(e, env, func_name, variables))
                        .collect::<Result<Vec<_>, _>>()?
                        .last()
                        .cloned()
                        .unwrap_or(ExprType::Void(*code_ref));
                    if let Some(slast_body_type) = last_body_type {
                        if body_type == slast_body_type {
                            last_body_type = Some(body_type)
                        } else {
                            error!("");
                            return Err(TypeError::TypeMismatch {
                                c: code_ref.s(&env.file_idx),
                                expected: slast_body_type,
                                actual: body_type,
                            });
                        }
                    } else {
                        last_body_type = Some(body_type)
                    }
                }
                if let Some(slast_body_type) = last_body_type {
                    let else_body_type = else_body
                        .iter()
                        .map(|e| ExprType::of(e, env, func_name, variables))
                        .collect::<Result<Vec<_>, _>>()?
                        .last()
                        .cloned()
                        .unwrap_or(ExprType::Void(*code_ref));
                    if else_body_type != slast_body_type {
                        error!("");
                        return Err(TypeError::TypeMismatch {
                            c: code_ref.s(&env.file_idx),
                            expected: slast_body_type,
                            actual: else_body_type,
                        });
                    }
                    slast_body_type
                } else {
                    ExprType::Void(*code_ref)
                }
            }
            Expr::IfElse(code_ref, econd, etrue, efalse) => {
                //TODO should we check ExprType::of of every line of body?
                let tcond = ExprType::of(econd, env, func_name, variables)?;
                if tcond != ExprType::Bool(*code_ref) {
                    error!("");
                    return Err(TypeError::TypeMismatch {
                        c: code_ref.s(&env.file_idx),
                        expected: ExprType::Bool(*code_ref),
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| ExprType::of(e, env, func_name, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void(*code_ref));
                let tfalse = efalse
                    .iter()
                    .map(|e| ExprType::of(e, env, func_name, variables))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void(*code_ref));

                if ttrue == tfalse {
                    ttrue
                } else {
                    error!("");
                    return Err(TypeError::TypeMismatch {
                        c: code_ref.s(&env.file_idx),
                        expected: ttrue,
                        actual: tfalse,
                    });
                }
            }
            Expr::Assign(code_ref, lhs_exprs, rhs_exprs) => {
                let tlen = match rhs_exprs.len().into() {
                    1 => ExprType::of(&rhs_exprs[0], env, func_name, variables)?.tuple_size(),
                    n => n,
                };
                if usize::from(lhs_exprs.len()) != tlen {
                    return Err(TypeError::TupleLengthMismatch {
                        c: code_ref.s(&env.file_idx),
                        actual: usize::from(rhs_exprs.len()),
                        expected: tlen,
                    });
                }
                let mut rhs_types = Vec::new();
                for (lhs_expr, rhs_expr) in lhs_exprs.iter().zip(rhs_exprs.iter()) {
                    //println!(
                    //    "{}:{} lhs_expr {} = rhs_expr {}",
                    //    file!(),
                    //    line!(),
                    //    lhs_expr,
                    //    rhs_expr
                    //);
                    if let Expr::Identifier(_code_ref, _name) = lhs_expr {
                        //The lhs_expr is just a new var
                        let rhs_type = ExprType::of(rhs_expr, env, func_name, variables)?;
                        rhs_types.push(rhs_type);
                    } else {
                        let lhs_type = ExprType::of(lhs_expr, env, func_name, variables)?;
                        let rhs_type = ExprType::of(rhs_expr, env, func_name, variables)?;

                        if lhs_type != rhs_type {
                            error!("");
                            return Err(TypeError::TypeMismatch {
                                c: lhs_expr.clone().get_code_ref().s(&env.file_idx),
                                expected: lhs_type,
                                actual: rhs_type,
                            });
                        }
                        rhs_types.push(rhs_type);
                    }
                }
                ExprType::Void(*code_ref)
            }
            Expr::AssignOp(_code_ref, _, _, e) => ExprType::of(e, env, func_name, variables)?,
            Expr::WhileLoop(code_ref, idx_expr, stmts) => {
                let idx_type = ExprType::of(idx_expr, env, func_name, variables)?;
                if idx_type != ExprType::Bool(*code_ref) {
                    error!("");
                    return Err(TypeError::TypeMismatch {
                        c: code_ref.s(&env.file_idx),
                        expected: ExprType::Bool(*code_ref),
                        actual: idx_type,
                    });
                }
                for expr in stmts {
                    ExprType::of(expr, env, func_name, variables)?;
                }
                ExprType::Void(*code_ref)
            }

            Expr::Block(code_ref, b) => b
                .iter()
                .map(|e| ExprType::of(e, env, func_name, variables))
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void(*code_ref)),
            Expr::Call(code_ref, fn_name, args, is_macro) => {
                if *is_macro {
                    todo!()
                    // Here the macro can check if the args works and will return
                    // returns = (macros[fn_name])(code_ref, args, env)
                }
                if fn_name == "unsized" {
                    if args.len() != 1 {
                        return Err(TypeError::TupleLengthMismatch {
                            c: code_ref.s(&env.file_idx),
                            actual: args.len(),
                            expected: 1,
                        });
                    }

                    let targ = ExprType::of(&args[0], env, func_name, variables)?;
                    return match targ {
                        ExprType::Array(c, expr, _) => {
                            Ok(ExprType::Array(c, expr, ArraySizedExpr::Unsized))
                        }
                        ExprType::Address(c) => {
                            Ok(ExprType::Array(c, Box::new(targ), ArraySizedExpr::Unsized))
                        }
                        sv => Err(TypeError::TypeMismatchSpecific {
                            c: code_ref.s(&env.file_idx),
                            s: format!("function unsized does not support {}", sv),
                        }),
                    };
                }
                //This could be called from an inline function if so we need to look at its closures

                let func = if let Some((closure, _temp_closure)) =
                    env.get_inline_closure(func_name, fn_name)
                {
                    //func is a closure
                    Some(closure.func)
                } else {
                    env.funcs.get(fn_name).cloned()
                };
                if let Some(func) = func {
                    let mut targs = Vec::new();

                    for e in args {
                        targs.push(ExprType::of(e, env, func_name, variables)?);
                    }

                    if func.params.len() != targs.len() {
                        return Err(TypeError::TupleLengthMismatch {
                            //TODO be more specific: function {} expected {} parameters, but {} were given
                            c: code_ref.s(&env.file_idx),
                            actual: targs.len(),
                            expected: func.params.len(),
                        });
                    }

                    for (i, (targ, param)) in targs.iter().zip(func.params.iter()).enumerate() {
                        let param_type = &param.expr_type;
                        if param_type == targ {
                            continue;
                        } else {
                            error!("");
                            return Err(TypeError::TypeMismatchSpecific {
                                    c: code_ref.s(&env.file_idx),
                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param_type, targ)
                                });
                        }
                    }

                    match &func.returns {
                        v if v.is_empty() => ExprType::Void(*code_ref),
                        v if v.len() == 1 => v.first().unwrap().expr_type.clone(),
                        v => {
                            let mut items = Vec::new();
                            for arg in v.iter() {
                                items.push(arg.expr_type.clone());
                            }
                            ExprType::Tuple(*code_ref, items)
                        }
                    }
                } else {
                    return Err(TypeError::UnknownFunction(
                        code_ref.s(&env.file_idx),
                        fn_name.to_string(),
                    ));
                }
            }
            Expr::GlobalDataAddr(code_ref, _) => ExprType::F32(*code_ref),
            Expr::Parentheses(_code_ref, expr) => ExprType::of(expr, env, func_name, variables)?,
            Expr::ArrayAccess(code_ref, expr, idx_expr) => {
                match ExprType::of(&*idx_expr, env, func_name, variables)? {
                    ExprType::I64(_code_ref) => (),
                    e => {
                        error!("");
                        return Err(TypeError::TypeMismatch {
                            c: code_ref.s(&env.file_idx),
                            expected: ExprType::I64(*code_ref),
                            actual: e,
                        });
                    }
                };

                match ExprType::of(&*expr, env, func_name, variables)? {
                    ExprType::Array(_code_ref, expr_type, _size_type) => *expr_type,
                    _ => {
                        return Err(TypeError::TypeMismatchSpecific {
                            c: code_ref.s(&env.file_idx),
                            s: format!("{} is not an array", expr),
                        })
                    }
                }
            }
            Expr::NewStruct(code_ref, struct_name, _fields) => {
                if env.struct_map.contains_key(struct_name) {
                    //Need to check field types
                } else {
                    return Err(TypeError::UnknownStruct(
                        code_ref.s(&env.file_idx),
                        struct_name.to_string(),
                    ));
                }
                ExprType::Struct(*code_ref, Box::new(struct_name.to_string()))
            }
            Expr::Declaration(code_ref, _declaration) => {
                // TODO check contents
                // checking vars that are in the containing variables should be easy
                // will need to somehow include the func parameters
                ExprType::Void(*code_ref)
            }
        };
        Ok(res)
    }

    pub fn tuple_size(&self) -> usize {
        match self {
            ExprType::Void(_) => 0,
            ExprType::Bool(_)
            | ExprType::F32(_)
            | ExprType::I64(_)
            | ExprType::Address(_)
            | ExprType::Struct(_, _)
            | ExprType::Array(_, _, _) => 1,
            ExprType::Tuple(_, v) => v.len(),
        }
    }

    pub fn cranelift_type(
        &self,
        ptr_type: cranelift::prelude::Type,
        struct_access: bool,
    ) -> Result<cranelift::prelude::Type, TypeError> {
        //TODO add env.file_idx for code_ref errors
        match self {
            ExprType::Void(code_ref) => Err(TypeError::TypeMismatchSpecific {
                c: code_ref.to_string(),
                s: "Void has no cranelift analog".to_string(),
            }),
            ExprType::Bool(_code_ref) => Ok(if struct_access {
                cranelift::prelude::types::I8
            } else {
                cranelift::prelude::types::B1
            }),
            ExprType::F32(_code_ref) => Ok(cranelift::prelude::types::F32),
            ExprType::I64(_code_ref) => Ok(cranelift::prelude::types::I64),
            ExprType::Array(_code_ref, _, _) => Ok(ptr_type),
            ExprType::Address(_code_ref) => Ok(ptr_type),
            ExprType::Struct(_code_ref, _) => Ok(ptr_type),
            ExprType::Tuple(code_ref, _) => Err(TypeError::TypeMismatchSpecific {
                c: code_ref.to_string(),
                s: "Tuple has no cranelift analog".to_string(),
            }),
        }
    }
}
