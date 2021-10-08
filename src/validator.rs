use std::{collections::HashMap, fmt::Display};

use crate::{
    frontend::{Binop, Expr},
    jit::{Env, SVariable, StructDef},
    sarus_std_lib,
};
use cranelift::prelude::types;
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
    Array(Box<ExprType>, Option<usize>),
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
            ExprType::Array(ty, len) => {
                if let Some(len) = len {
                    write!(f, "&[{}; {}]", ty, len)
                } else {
                    write!(f, "&[{}]", ty)
                }
            }
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
    env: &Env,
    parts: Vec<String>,
    lhs_val: &Option<ExprType>,
) -> Result<ExprType, TypeError> {
    let (lhs_val, start) = if let Some(lhs_val) = lhs_val {
        (lhs_val.clone(), 0)
    } else {
        (env.variables[&parts[0]].expr_type().unwrap(), 1)
    };
    match lhs_val {
        ExprType::Struct(struct_name) => {
            let mut struct_name = *struct_name.to_owned();
            let mut parent_struct_field = &env.struct_map[&struct_name].fields[&parts[start]];
            if parts.len() > 2 {
                for i in start..parts.len() {
                    if let ExprType::Struct(_name) = &parent_struct_field.expr_type {
                        parent_struct_field = &env.struct_map[&struct_name].fields[&parts[i]];
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
                expected: ExprType::Struct(Box::new("".to_string())),
                actual: v.clone(),
            })
        }
    }
}

impl ExprType {
    pub fn width(
        &self,
        ptr_ty: types::Type,
        struct_map: &HashMap<String, StructDef>,
    ) -> Option<usize> {
        match self {
            ExprType::Array(ty, len) => {
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
            ExprType::Void => Some(0),
            ExprType::Bool => Some(types::I8.bytes() as usize),
            ExprType::F64 => Some(types::F64.bytes() as usize),
            ExprType::I64 => Some(types::I64.bytes() as usize),
            ExprType::Address => Some(ptr_ty.bytes() as usize),
            ExprType::Tuple(_expr_types) => None,
            ExprType::Struct(name) => Some(struct_map[&name.to_string()].size),
        }
    }

    pub fn of(expr: &Expr, env: &Env) -> Result<ExprType, TypeError> {
        let res = match expr {
            Expr::Identifier(id_name) => {
                if env.variables.contains_key(id_name) {
                    env.variables[id_name].expr_type().unwrap()
                } else if env.constant_vars.contains_key(id_name) {
                    ExprType::F64 //All constants are currently math like PI, TAU...
                } else {
                    dbg!(&id_name);
                    return Err(TypeError::UnknownVariable(id_name.to_string()));
                }
            }
            Expr::LiteralFloat(_) => ExprType::F64,
            Expr::LiteralInt(_) => ExprType::I64,
            Expr::LiteralBool(_) => ExprType::Bool,
            Expr::LiteralString(_) => ExprType::Address, //TODO change to char
            Expr::Binop(binop, lhs, rhs) => match binop {
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
                                    Expr::Call(fn_name, args) => {
                                        let sval = if path.len() == 0 {
                                            if let Some(lhs_val) = lhs_val {
                                                lhs_val
                                            } else {
                                                unreachable!("cannot find val type {}", expr)
                                            }
                                        } else if path.len() > 1 {
                                            let spath = path
                                                .iter()
                                                .map(|lhs_i: &Expr| lhs_i.to_string())
                                                .collect::<Vec<String>>();
                                            get_struct_field_type(&env, spath, &lhs_val)?
                                        } else {
                                            ExprType::of(&path[0], env)?
                                        };

                                        let fn_name = format!("{}.{}", sval.to_string(), fn_name);

                                        let params = &env.funcs[&fn_name].params;

                                        if params.len() - 1 != args.len() {
                                            return Err(TypeError::TupleLengthMismatch {
                                                //TODO be more specific: function {} expected {} parameters, but {} were given
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
                                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param.expr_type , targ)
                                                });
                                            }
                                        }

                                        let returns = &env.funcs[&fn_name].returns;

                                        if returns.len() == 0 {
                                            lhs_val = Some(ExprType::Void)
                                        } else if returns.len() == 1 {
                                            lhs_val = Some(returns[0].expr_type.clone())
                                        } else {
                                            let mut expr_types = Vec::new();
                                            for arg in returns {
                                                expr_types.push(arg.expr_type.clone())
                                            }
                                            lhs_val = Some(ExprType::Tuple(expr_types))
                                        }

                                        path = Vec::new();
                                    }
                                    Expr::LiteralFloat(_) => todo!(),
                                    Expr::LiteralInt(_) => todo!(),
                                    Expr::LiteralBool(_) => todo!(),
                                    Expr::LiteralString(_) => {
                                        lhs_val = Some(ExprType::of(&expr, env)?);
                                    }
                                    Expr::Identifier(_) => path.push(expr),
                                    Expr::Binop(op, lhs, rhs) => {
                                        if let Binop::DotAccess = op {
                                            curr_expr = Some(*lhs.clone());
                                            next_expr = Some(*rhs.clone());
                                        } else {
                                            todo!();
                                        }
                                    }
                                    Expr::Unaryop(_, _) => todo!(),
                                    Expr::Compare(_, _, _) => todo!(),
                                    Expr::IfThen(_, _) => todo!(),
                                    Expr::IfElse(_, _, _) => todo!(),
                                    Expr::Assign(_, _) => todo!(),
                                    Expr::AssignOp(_, _, _) => todo!(),
                                    Expr::NewStruct(_, _) => todo!(),
                                    Expr::WhileLoop(_, _) => todo!(),
                                    Expr::Block(_) => todo!(),
                                    Expr::GlobalDataAddr(_) => todo!(),
                                    Expr::Parentheses(e) => lhs_val = Some(ExprType::of(&e, env)?),
                                    Expr::ArrayAccess(name, idx_expr) => {
                                        match ExprType::of(&idx_expr, env)? {
                                            ExprType::I64 => (),
                                            e => {
                                                return Err(TypeError::TypeMismatch {
                                                    expected: ExprType::I64,
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
                                            if let ExprType::Array(ty, _len) =
                                                get_struct_field_type(&env, spath, &lhs_val)?
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
                        lhs_val = Some(get_struct_field_type(&env, spath, &lhs_val)?);
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
                            expected: lt,
                            actual: rt,
                        });
                    }
                }
            },
            Expr::Unaryop(_, l) => ExprType::of(l, env)?,
            Expr::Compare(_, _, _) => ExprType::Bool,
            Expr::IfThen(econd, _) => {
                let tcond = ExprType::of(econd, env)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }
                ExprType::Void
            }
            Expr::IfElse(econd, etrue, efalse) => {
                let tcond = ExprType::of(econd, env)?;
                if tcond != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: tcond,
                    });
                }

                let ttrue = etrue
                    .iter()
                    .map(|e| ExprType::of(e, env))
                    .collect::<Result<Vec<_>, _>>()?
                    .last()
                    .cloned()
                    .unwrap_or(ExprType::Void);
                let tfalse = efalse
                    .iter()
                    .map(|e| ExprType::of(e, env))
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
            Expr::Assign(lhs_exprs, rhs_exprs) => {
                let tlen = match rhs_exprs.len().into() {
                    1 => ExprType::of(&rhs_exprs[0], env)?.tuple_size(),
                    n => n,
                };
                if usize::from(lhs_exprs.len()) != tlen {
                    return Err(TypeError::TupleLengthMismatch {
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
                    if let Expr::Identifier(_name) = lhs_expr {
                        //The lhs_expr is just a new var
                        let rhs_type = ExprType::of(rhs_expr, env)?;
                        rhs_types.push(rhs_type);
                    } else {
                        let lhs_type = ExprType::of(lhs_expr, env)?;
                        let rhs_type = ExprType::of(rhs_expr, env)?;

                        if lhs_type != rhs_type {
                            return Err(TypeError::TypeMismatch {
                                expected: lhs_type,
                                actual: rhs_type,
                            });
                        }
                        rhs_types.push(rhs_type);
                    }
                }
                ExprType::Tuple(rhs_types)
            }
            Expr::AssignOp(_, _, e) => ExprType::of(e, env)?,
            Expr::WhileLoop(idx_expr, stmts) => {
                let idx_type = ExprType::of(idx_expr, env)?;
                if idx_type != ExprType::Bool {
                    return Err(TypeError::TypeMismatch {
                        expected: ExprType::Bool,
                        actual: idx_type,
                    });
                }
                for expr in stmts {
                    ExprType::of(expr, env)?;
                }
                ExprType::Void
            }

            Expr::Block(b) => b
                .iter()
                .map(|e| ExprType::of(e, env))
                .last()
                .map(Result::unwrap)
                .unwrap_or(ExprType::Void),
            Expr::Call(fn_name, args) => {
                if let Some(_) = sarus_std_lib::is_struct_size_call(&fn_name, &env.struct_map) {
                    return Ok(ExprType::I64);
                } else if let Some(func) = env.funcs.get(fn_name) {
                    let mut targs = Vec::new();

                    for e in args {
                        targs.push(ExprType::of(e, env)?);
                    }

                    if func.params.len() != targs.len() {
                        return Err(TypeError::TupleLengthMismatch {
                            //TODO be more specific: function {} expected {} parameters, but {} were given
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
                                    s: format!("function {} expected parameter {} to be of type {} but type {} was found", fn_name, i, param_type, targ)
                                });
                        }
                    }

                    match &func.returns {
                        v if v.is_empty() => ExprType::Void,
                        v if v.len() == 1 => v.first().unwrap().expr_type.clone(),
                        v => {
                            let mut items = Vec::new();
                            for arg in v.iter() {
                                items.push(arg.expr_type.clone());
                            }
                            ExprType::Tuple(items)
                        }
                    }
                } else {
                    return Err(TypeError::UnknownFunction(fn_name.to_string()));
                }
            }
            Expr::GlobalDataAddr(_) => ExprType::F64,
            Expr::Parentheses(expr) => ExprType::of(expr, env)?,
            Expr::ArrayAccess(id_name, idx_expr) => {
                match ExprType::of(&*idx_expr, env)? {
                    ExprType::I64 => (),
                    e => {
                        return Err(TypeError::TypeMismatch {
                            expected: ExprType::I64,
                            actual: e,
                        })
                    }
                };
                if env.variables.contains_key(id_name) {
                    match &env.variables[id_name] {
                        SVariable::Array(ty, _len) => Ok(ty.expr_type().unwrap()),
                        _ => Err(TypeError::TypeMismatchSpecific {
                            s: format!("{} is not an array", id_name),
                        }),
                    }
                } else {
                    dbg!(&id_name);
                    return Err(TypeError::UnknownVariable(id_name.to_string()));
                }?
            }
            Expr::NewStruct(struct_name, _fields) => {
                if env.struct_map.contains_key(struct_name) {
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
            | ExprType::Array(_, _) => 1,
            ExprType::Tuple(v) => v.len(),
        }
    }

    pub fn cranelift_type(
        &self,
        ptr_type: cranelift::prelude::Type,
        struct_access: bool,
    ) -> Result<cranelift::prelude::Type, TypeError> {
        match self {
            ExprType::Void => Err(TypeError::TypeMismatchSpecific {
                s: "Void has no cranelift analog".to_string(),
            }),
            ExprType::Bool => Ok(if struct_access {
                cranelift::prelude::types::I8
            } else {
                cranelift::prelude::types::B1
            }),
            ExprType::F64 => Ok(cranelift::prelude::types::F64),
            ExprType::I64 => Ok(cranelift::prelude::types::I64),
            ExprType::Array(_, _) => Ok(ptr_type),
            ExprType::Address => Ok(ptr_type),
            ExprType::Struct(_) => Ok(ptr_type),
            ExprType::Tuple(_) => Err(TypeError::TypeMismatchSpecific {
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
