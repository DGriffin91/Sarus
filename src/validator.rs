use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Binop, CodeRef, Expr},
    jit::{Env, SVariable, StructDef},
};
use cranelift::prelude::types;
use thiserror::Error;

//TODO Make errors more information rich, also: show line in this file, and line in source
#[derive(Debug, Clone, Error)]
pub enum TypeError {
    #[error("{c} Type mismatch; expected {expected}, found {actual}")]
    TypeMismatch {
        c: CodeRef,
        expected: ExprType,
        actual: ExprType,
    },
    #[error("{c} Type mismatch; {s}")]
    TypeMismatchSpecific { c: CodeRef, s: String },
    #[error("{c} Tuple length mismatch; expected {expected} found {actual}")]
    TupleLengthMismatch {
        c: CodeRef,
        expected: usize,
        actual: usize,
    },
    #[error("{0} Function \"{1}\" does not exist")]
    UnknownFunction(CodeRef, String),
    #[error("{0} Variable \"{1}\" does not exist")]
    UnknownVariable(CodeRef, String),
    #[error("{0} Struct \"{1}\" does not exist")]
    UnknownStruct(CodeRef, String),
    #[error("{0} Struct \"{1}\" does not have field \"{2}\"")]
    UnknownField(CodeRef, String, String),
}

#[derive(Debug, Clone)]
pub enum ExprType {
    Void(CodeRef),
    Bool(CodeRef),
    F64(CodeRef),
    I64(CodeRef),
    Array(CodeRef, Box<ExprType>, Option<usize>),
    Address(CodeRef),
    Tuple(CodeRef, Vec<ExprType>),
    Struct(CodeRef, Box<String>),
}

pub fn f64_t() -> ExprType {
    ExprType::F64(CodeRef::z())
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
            ExprType::F64(_) => {
                if let ExprType::F64(_) = other {
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
            ExprType::F64(_) => write!(f, "f64"),
            ExprType::I64(_) => write!(f, "i64"),
            ExprType::Array(_, ty, len) => {
                if let Some(len) = len {
                    write!(f, "&[{}; {}]", ty, len)
                } else {
                    write!(f, "&[{}]", ty)
                }
            }
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
) -> Result<ExprType, TypeError> {
    let (lhs_val, start) = if let Some(lhs_val) = lhs_val {
        (lhs_val.clone(), 0)
    } else {
        (env.variables[&parts[0]].expr_type(code_ref).unwrap(), 1)
    };
    match lhs_val {
        ExprType::Struct(_code_ref, struct_name) => {
            let mut struct_name = *struct_name.to_owned();
            let mut parent_struct_field = if let Some(parent_struct_field) =
                env.struct_map[&struct_name].fields.get(&parts[start])
            {
                parent_struct_field
            } else {
                return Err(TypeError::UnknownField(
                    *code_ref,
                    struct_name,
                    parts[start].to_string(),
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
                                *code_ref,
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
            return Err(TypeError::TypeMismatch {
                c: v.get_code_ref(),
                expected: ExprType::Struct(v.get_code_ref(), Box::new("".to_string())),
                actual: v.clone(),
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
            ExprType::F64(code_ref) => code_ref,
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
            ExprType::F64(code_ref) => *code_ref = new_code_ref,
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
            ExprType::Array(_code_ref, ty, len) => {
                if let Some(len) = len {
                    if let Some(width) = ty.width(ptr_ty, struct_map) {
                        Some(width * len)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ExprType::Void(_) => Some(0),
            ExprType::Bool(_) => Some(types::I8.bytes() as usize),
            ExprType::F64(_) => Some(types::F64.bytes() as usize),
            ExprType::I64(_) => Some(types::I64.bytes() as usize),
            ExprType::Address(_) => Some(ptr_ty.bytes() as usize),
            ExprType::Tuple(_code_ref, _expr_types) => None,
            ExprType::Struct(_code_ref, name) => Some(struct_map[&name.to_string()].size),
        }
    }

    pub fn of(expr: &Expr, env: &Env) -> Result<ExprType, TypeError> {
        let res = match expr {
            Expr::Identifier(code_ref, id_name) => {
                if env.variables.contains_key(id_name) {
                    env.variables[id_name].expr_type(&code_ref).unwrap()
                } else if let Some(v) = env.constant_vars.get(id_name) {
                    v.expr_type(Some(*code_ref)) //Constants like PI, TAU...
                } else {
                    dbg!(&id_name);
                    return Err(TypeError::UnknownVariable(*code_ref, id_name.to_string()));
                }
            }
            Expr::LiteralFloat(code_ref, _) => ExprType::F64(*code_ref),
            Expr::LiteralInt(code_ref, _) => ExprType::I64(*code_ref),
            Expr::LiteralBool(code_ref, _) => ExprType::Bool(*code_ref),
            Expr::LiteralString(code_ref, _) => ExprType::Address(*code_ref), //TODO change to char
            Expr::LiteralArray(code_ref, e, len) => {
                ExprType::Array(*code_ref, Box::new(ExprType::of(e, env)?), Some(*len))
            }
            Expr::Binop(binop_code_ref, binop, lhs, rhs) => match binop {
                crate::frontend::Binop::DotAccess => {
                    let mut path = Vec::new();
                    let mut lhs_val = None;

                    let mut curr_expr = Some(*lhs.to_owned());
                    let mut next_expr = Some(*rhs.to_owned());

                    loop {
                        //println!("curr_expr {:?} next_expr {:?}", &curr_expr, &next_expr);
                        //println!("path {:?}", &path);
                        match curr_expr.clone() {
                            Some(expr) => {
                                curr_expr = next_expr;
                                next_expr = None;
                                match expr.clone() {
                                    Expr::Call(code_ref, fn_name, args) => {
                                        let sval = if path.len() == 0 {
                                            if let Some(lhs_val) = lhs_val {
                                                lhs_val
                                            } else {
                                                //This is a lhs call
                                                lhs_val = Some(ExprType::of(&expr, env)?);
                                                continue;
                                            }
                                        } else if path.len() > 1 || lhs_val.is_some() {
                                            let spath = path
                                                .iter()
                                                .map(|lhs_i: &Expr| lhs_i.to_string())
                                                .collect::<Vec<String>>();
                                            get_struct_field_type(&env, spath, &lhs_val, &code_ref)?
                                        } else {
                                            ExprType::of(&path[0], env)?
                                        };

                                        let fn_name = format!("{}.{}", sval.to_string(), fn_name);

                                        if !&env.funcs.contains_key(&fn_name) {
                                            return Err(TypeError::UnknownFunction(
                                                code_ref,
                                                fn_name.to_string(),
                                            ));
                                        }

                                        let params = &env.funcs[&fn_name].params;

                                        if params.len() - 1 != args.len() {
                                            return Err(TypeError::TupleLengthMismatch {
                                                //TODO be more specific: function {} expected {} parameters, but {} were given
                                                c: code_ref,
                                                actual: args.len(),
                                                expected: params.len() - 1,
                                            });
                                        }

                                        for (i, (param, arg)) in
                                            params.iter().skip(1).zip(args.iter()).enumerate()
                                        {
                                            let targ = ExprType::of(arg, env)?;
                                            if param.expr_type != targ {
                                                return Err(TypeError::TypeMismatchSpecific {
                                                    c: code_ref,
                                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param.expr_type , targ)
                                                });
                                            }
                                        }

                                        let returns = &env.funcs[&fn_name].returns;

                                        if returns.len() == 0 {
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
                                    Expr::LiteralFloat(_code_ref, _) => todo!(),
                                    Expr::LiteralInt(_code_ref, _) => todo!(),
                                    Expr::LiteralBool(_code_ref, _) => todo!(),
                                    Expr::LiteralString(_code_ref, _) => {
                                        lhs_val = Some(ExprType::of(&expr, env)?);
                                    }
                                    Expr::LiteralArray(_code_ref, _, _) => {
                                        lhs_val = Some(ExprType::of(&expr, env)?);
                                    }
                                    Expr::Identifier(_code_ref, _i) => path.push(expr),
                                    Expr::Binop(_code_ref, op, lhs, rhs) => {
                                        if let Binop::DotAccess = op {
                                            curr_expr = Some(*lhs.clone());
                                            next_expr = Some(*rhs.clone());
                                        } else {
                                            todo!();
                                        }
                                    }
                                    Expr::Unaryop(_code_ref, _, _) => todo!(),
                                    Expr::Compare(_code_ref, _, _, _) => todo!(),
                                    Expr::IfThen(_code_ref, _, _) => todo!(),
                                    Expr::IfElse(_code_ref, _, _, _) => todo!(),
                                    Expr::Assign(_code_ref, _, _) => todo!(),
                                    Expr::AssignOp(_code_ref, _, _, _) => todo!(),
                                    Expr::NewStruct(_code_ref, _, _) => todo!(),
                                    Expr::WhileLoop(_code_ref, _, _) => todo!(),
                                    Expr::Block(_code_ref, _) => todo!(),
                                    Expr::GlobalDataAddr(_code_ref, _) => todo!(),
                                    Expr::Parentheses(_code_ref, e) => {
                                        lhs_val = Some(ExprType::of(&e, env)?)
                                    }
                                    Expr::ArrayAccess(code_ref, name, idx_expr) => {
                                        match ExprType::of(&idx_expr, env)? {
                                            ExprType::I64(_code_ref) => (),
                                            e => {
                                                return Err(TypeError::TypeMismatch {
                                                    c: code_ref,
                                                    expected: ExprType::I64(code_ref),
                                                    actual: e,
                                                })
                                            }
                                        };
                                        if path.len() > 0 {
                                            let mut spath = path
                                                .iter()
                                                .map(|lhs_i: &Expr| lhs_i.to_string())
                                                .collect::<Vec<String>>();
                                            spath.push(name.to_string());
                                            if let ExprType::Array(_code_ref, ty, _len) =
                                                get_struct_field_type(
                                                    &env, spath, &lhs_val, &code_ref,
                                                )?
                                            {
                                                lhs_val = Some(*ty.to_owned());
                                            }
                                        } else {
                                            lhs_val = Some(ExprType::of(&expr, env)?);
                                        }

                                        path = Vec::new();
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                    if path.len() > 0 {
                        let spath = path
                            .iter()
                            .map(|lhs_i: &Expr| lhs_i.to_string())
                            .collect::<Vec<String>>();
                        lhs_val = Some(get_struct_field_type(
                            &env,
                            spath,
                            &lhs_val,
                            binop_code_ref,
                        )?);
                    }

                    if let Some(lhs_val) = lhs_val {
                        return Ok(lhs_val);
                    } else {
                        panic!("No value found");
                    }
                }
                _ => {
                    let lt = ExprType::of(lhs, env)?;
                    let rt = ExprType::of(rhs, env)?;
                    if lt == rt {
                        lt
                    } else {
                        return Err(TypeError::TypeMismatch {
                            c: *binop_code_ref,
                            expected: lt,
                            actual: rt,
                        });
                    }
                }
            },
            Expr::Unaryop(_code_ref, _, l) => ExprType::of(l, env)?,
            Expr::Compare(code_ref, _, _, _) => ExprType::Bool(*code_ref),
            Expr::IfThen(code_ref, econd, _) => {
                let tcond = ExprType::of(econd, env)?;
                if tcond != ExprType::Bool(*code_ref) {
                    return Err(TypeError::TypeMismatch {
                        c: *code_ref,
                        expected: ExprType::Bool(*code_ref),
                        actual: tcond,
                    });
                }
                ExprType::Void(*code_ref)
            }
            Expr::IfElse(code_ref, econd, etrue, efalse) => {
                let tcond = ExprType::of(econd, env)?;
                if tcond != ExprType::Bool(*code_ref) {
                    return Err(TypeError::TypeMismatch {
                        c: *code_ref,
                        expected: ExprType::Bool(*code_ref),
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| ExprType::of(e, env))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void(*code_ref));
                let tfalse = efalse
                    .iter()
                    .map(|e| ExprType::of(e, env))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void(*code_ref));

                if ttrue == tfalse {
                    ttrue
                } else {
                    return Err(TypeError::TypeMismatch {
                        c: *code_ref,
                        expected: ttrue,
                        actual: tfalse,
                    });
                }
            }
            Expr::Assign(code_ref, lhs_exprs, rhs_exprs) => {
                let tlen = match rhs_exprs.len().into() {
                    1 => ExprType::of(&rhs_exprs[0], env)?.tuple_size(),
                    n => n,
                };
                if usize::from(lhs_exprs.len()) != tlen {
                    return Err(TypeError::TupleLengthMismatch {
                        c: *code_ref,
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
                        let rhs_type = ExprType::of(rhs_expr, env)?;
                        rhs_types.push(rhs_type);
                    } else {
                        let lhs_type = ExprType::of(lhs_expr, env)?;
                        let rhs_type = ExprType::of(rhs_expr, env)?;

                        if lhs_type != rhs_type {
                            return Err(TypeError::TypeMismatch {
                                c: *lhs_expr.clone().get_code_ref(),
                                expected: lhs_type,
                                actual: rhs_type,
                            });
                        }
                        rhs_types.push(rhs_type);
                    }
                }
                ExprType::Tuple(*code_ref, rhs_types)
            }
            Expr::AssignOp(_code_ref, _, _, e) => ExprType::of(e, env)?,
            Expr::WhileLoop(code_ref, idx_expr, stmts) => {
                let idx_type = ExprType::of(idx_expr, env)?;
                if idx_type != ExprType::Bool(*code_ref) {
                    return Err(TypeError::TypeMismatch {
                        c: *code_ref,
                        expected: ExprType::Bool(*code_ref),
                        actual: idx_type,
                    });
                }
                for expr in stmts {
                    ExprType::of(expr, env)?;
                }
                ExprType::Void(*code_ref)
            }

            Expr::Block(code_ref, b) => b
                .iter()
                .map(|e| ExprType::of(e, env))
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void(*code_ref)),
            Expr::Call(code_ref, fn_name, args) => {
                if let Some(func) = env.funcs.get(fn_name) {
                    let mut targs = Vec::new();

                    for e in args {
                        targs.push(ExprType::of(e, env)?);
                    }

                    if func.params.len() != targs.len() {
                        return Err(TypeError::TupleLengthMismatch {
                            //TODO be more specific: function {} expected {} parameters, but {} were given
                            c: *code_ref,
                            actual: targs.len(),
                            expected: func.params.len(),
                        });
                    }

                    for (i, (targ, param)) in targs.iter().zip(func.params.iter()).enumerate() {
                        let param_type = param.expr_type.clone();
                        if param_type == *targ {
                            continue;
                        } else {
                            return Err(TypeError::TypeMismatchSpecific {
                                    c: *code_ref,
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
                    return Err(TypeError::UnknownFunction(*code_ref, fn_name.to_string()));
                }
            }
            Expr::GlobalDataAddr(code_ref, _) => ExprType::F64(*code_ref),
            Expr::Parentheses(_code_ref, expr) => ExprType::of(expr, env)?,
            Expr::ArrayAccess(code_ref, id_name, idx_expr) => {
                match ExprType::of(&*idx_expr, env)? {
                    ExprType::I64(_code_ref) => (),
                    e => {
                        return Err(TypeError::TypeMismatch {
                            c: *code_ref,
                            expected: ExprType::I64(*code_ref),
                            actual: e,
                        })
                    }
                };
                if env.variables.contains_key(id_name) {
                    match &env.variables[id_name] {
                        SVariable::Array(ty, _len) => Ok(ty.expr_type(code_ref).unwrap()),
                        _ => Err(TypeError::TypeMismatchSpecific {
                            c: *code_ref,
                            s: format!("{} is not an array", id_name),
                        }),
                    }
                } else {
                    dbg!(&id_name);
                    return Err(TypeError::UnknownVariable(*code_ref, id_name.to_string()));
                }?
            }
            Expr::NewStruct(code_ref, struct_name, _fields) => {
                if env.struct_map.contains_key(struct_name) {
                    //Need to check field types
                } else {
                    return Err(TypeError::UnknownStruct(*code_ref, struct_name.to_string()));
                }
                ExprType::Struct(*code_ref, Box::new(struct_name.to_string()))
            }
        };
        Ok(res)
    }

    pub fn tuple_size(&self) -> usize {
        match self {
            ExprType::Void(_) => 0,
            ExprType::Bool(_)
            | ExprType::F64(_)
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
        match self {
            ExprType::Void(code_ref) => Err(TypeError::TypeMismatchSpecific {
                c: *code_ref,
                s: "Void has no cranelift analog".to_string(),
            }),
            ExprType::Bool(_code_ref) => Ok(if struct_access {
                cranelift::prelude::types::I8
            } else {
                cranelift::prelude::types::B1
            }),
            ExprType::F64(_code_ref) => Ok(cranelift::prelude::types::F64),
            ExprType::I64(_code_ref) => Ok(cranelift::prelude::types::I64),
            ExprType::Array(_code_ref, _, _) => Ok(ptr_type),
            ExprType::Address(_code_ref) => Ok(ptr_type),
            ExprType::Struct(_code_ref, _) => Ok(ptr_type),
            ExprType::Tuple(code_ref, _) => Err(TypeError::TypeMismatchSpecific {
                c: *code_ref,
                s: "Tuple has no cranelift analog".to_string(),
            }),
        }
    }
}

pub fn validate_program(stmts: &Vec<Expr>, env: &Env) -> Result<(), TypeError> {
    for expr in stmts {
        ExprType::of(expr, env)?;
    }
    Ok(())
}
