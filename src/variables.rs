use crate::frontend::*;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;

use tracing::instrument;
use tracing::trace;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArraySized {
    Unsized, //size is unknown, just an address with a type
    Slice,
    Fixed(Box<SValue>, usize), //size is part of type signature
}

impl ArraySized {
    pub fn expr_type(&self) -> ArraySizedExpr {
        match self {
            ArraySized::Unsized => ArraySizedExpr::Unsized,
            ArraySized::Slice => ArraySizedExpr::Slice,
            ArraySized::Fixed(_, size) => ArraySizedExpr::Fixed(*size),
        }
    }
    pub fn from(builder: &mut FunctionBuilder, size_type: &ArraySizedExpr) -> ArraySized {
        match size_type {
            ArraySizedExpr::Unsized => ArraySized::Unsized,
            ArraySizedExpr::Slice => ArraySized::Slice,
            ArraySizedExpr::Fixed(len) => ArraySized::Fixed(
                Box::new(SValue::I64(
                    builder.ins().iconst::<i64>(types::I64, *len as i64),
                )),
                *len,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SValue {
    Void,
    Unknown(Value),
    Bool(Value),
    F32(Value),
    I64(Value),
    U8(Value),
    Array(Box<SValue>, ArraySized),
    Address(Value),
    Tuple(Vec<SValue>),
    Struct(String, Value),
}

impl Display for SValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SValue::Unknown(_) => write!(f, "unknown"),
            SValue::Bool(_) => write!(f, "bool"),
            SValue::F32(_) => write!(f, "f32"),
            SValue::I64(_) => write!(f, "i64"),
            SValue::U8(_) => write!(f, "u8"),
            SValue::Array(sval, size_type) => match size_type {
                ArraySized::Unsized => write!(f, "&[{}]", sval),
                ArraySized::Slice => write!(f, "[{}]", sval),
                ArraySized::Fixed(_size_val, len) => write!(f, "[{}; {}]", sval, len),
            },
            SValue::Address(_) => write!(f, "&"),
            SValue::Void => write!(f, "void"),
            SValue::Tuple(v) => write!(f, "({})", v.len()),
            SValue::Struct(name, _) => write!(f, "{}", name),
        }
    }
}

impl SValue {
    pub fn from(
        builder: &mut FunctionBuilder,
        expr_type: &ExprType,
        value: Value,
    ) -> anyhow::Result<SValue> {
        Ok(match expr_type {
            ExprType::Void(_code_ref) => SValue::Void,
            ExprType::Bool(_code_ref) => SValue::Bool(value),
            ExprType::F32(_code_ref) => SValue::F32(value),
            ExprType::I64(_code_ref) => SValue::I64(value),
            ExprType::U8(_code_ref) => SValue::U8(value),
            ExprType::Array(_code_ref, ty, size_type) => SValue::Array(
                Box::new(SValue::from(builder, ty, value)?),
                ArraySized::from(builder, size_type),
            ),
            ExprType::Address(_code_ref) => SValue::Address(value),
            ExprType::Tuple(_code_ref, _) => anyhow::bail!("use SValue::from_tuple"),
            ExprType::Struct(_code_ref, name) => SValue::Struct(name.to_string(), value),
        })
    }
    pub fn get_from_variable(
        builder: &mut FunctionBuilder,
        variable: &SVariable,
    ) -> anyhow::Result<SValue> {
        Ok(match variable {
            SVariable::Unknown(_, v) => SValue::Unknown(builder.use_var(*v)),
            SVariable::Bool(_, v) => SValue::Bool(builder.use_var(*v)),
            SVariable::F32(_, v) => SValue::F32(builder.use_var(*v)),
            SVariable::I64(_, v) => SValue::I64(builder.use_var(*v)),
            SVariable::U8(_, v) => SValue::U8(builder.use_var(*v)),
            SVariable::Address(_, v) => SValue::Address(builder.use_var(*v)),
            SVariable::Array(svar, len) => SValue::Array(
                Box::new(SValue::get_from_variable(builder, svar)?),
                len.clone(),
            ),
            SVariable::Struct(_varname, structname, v, _return_struct) => {
                SValue::Struct(structname.to_string(), builder.use_var(*v))
            }
        })
    }
    pub fn replace_value(&self, value: Value) -> anyhow::Result<SValue> {
        Ok(match self {
            SValue::Void => SValue::Void,
            SValue::Bool(_) => SValue::Bool(value),
            SValue::F32(_) => SValue::F32(value),
            SValue::I64(_) => SValue::I64(value),
            SValue::U8(_) => SValue::U8(value),
            SValue::Array(sval, len) => {
                SValue::Array(Box::new(sval.replace_value(value)?), len.clone())
            }
            SValue::Address(_) => SValue::Address(value),
            SValue::Tuple(_values) => anyhow::bail!("use SValue::replace_tuple"),
            //{
            //    let new_vals = Vec::new();
            //    for val in values {
            //        new_vals.push(val.replace_value())
            //    }
            //    SValue::Tuple(_)
            //},
            SValue::Struct(name, _) => SValue::Struct(name.to_string(), value),
            SValue::Unknown(_) => SValue::Unknown(value),
        })
    }
    pub fn expr_type(&self, code_ref: &CodeRef) -> anyhow::Result<ExprType> {
        Ok(match self {
            SValue::Unknown(_) => anyhow::bail!("expression type is unknown"),
            SValue::Bool(_) => ExprType::Bool(*code_ref),
            SValue::F32(_) => ExprType::F32(*code_ref),
            SValue::I64(_) => ExprType::I64(*code_ref),
            SValue::U8(_) => ExprType::U8(*code_ref),
            SValue::Array(sval, size_type) => ExprType::Array(
                *code_ref,
                Box::new(sval.expr_type(code_ref)?),
                size_type.expr_type(),
            ),
            SValue::Address(_) => ExprType::Address(*code_ref),
            SValue::Struct(name, _) => ExprType::Struct(*code_ref, Box::new(name.to_string())),
            SValue::Void => ExprType::Void(*code_ref),
            SValue::Tuple(_) => todo!(),
        })
    }
    pub fn inner(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Unknown(v) => Ok(*v),
            SValue::Bool(v) => Ok(*v),
            SValue::F32(v) => Ok(*v),
            SValue::I64(v) => Ok(*v),
            SValue::U8(v) => Ok(*v),
            SValue::Array(sval, _len) => Ok(sval.inner(ctx)?),
            SValue::Address(v) => Ok(*v),
            SValue::Void => anyhow::bail!("void has no inner {}", ctx),
            SValue::Tuple(v) => anyhow::bail!("inner does not support tuple {:?} {}", v, ctx),
            SValue::Struct(_, v) => Ok(*v),
        }
    }
    pub fn expect_struct(&self, name: &str, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Struct(sname, v) => {
                if sname == name {
                    Ok(*v)
                } else {
                    anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx)
                }
            }
            v => anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SVariable {
    Unknown(String, Variable),
    Bool(String, Variable),
    F32(String, Variable),
    I64(String, Variable),
    U8(String, Variable),
    Array(Box<SVariable>, ArraySized),
    Address(String, Variable),
    Struct(String, String, Variable, bool),
}

impl Display for SVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVariable::Unknown(name, _) => write!(f, "{}", name),
            SVariable::Bool(name, _) => write!(f, "{}", name),
            SVariable::F32(name, _) => write!(f, "{}", name),
            SVariable::I64(name, _) => write!(f, "{}", name),
            SVariable::U8(name, _) => write!(f, "{}", name),
            SVariable::Array(svar, size_type) => match size_type {
                ArraySized::Unsized => write!(f, "&[{}]", svar),
                ArraySized::Slice => write!(f, "[{}]", svar),
                ArraySized::Fixed(_size_val, len) => write!(f, "&[{}; {}]", svar, len),
            },
            SVariable::Address(name, _) => write!(f, "{}", name),
            SVariable::Struct(name, structname, _, _return_struct) => {
                write!(f, "struct {} {}", name, structname)
            }
        }
    }
}

impl SVariable {
    pub fn inner(&self) -> Variable {
        match self {
            SVariable::Unknown(_, v) => *v,
            SVariable::Bool(_, v) => *v,
            SVariable::F32(_, v) => *v,
            SVariable::I64(_, v) => *v,
            SVariable::U8(_, v) => *v,
            SVariable::Array(svar, _len) => svar.inner(),
            SVariable::Address(_, v) => *v,
            SVariable::Struct(_, _, v, _) => *v,
        }
    }
    pub fn expr_type(&self, code_ref: &CodeRef) -> anyhow::Result<ExprType> {
        Ok(match self {
            SVariable::Unknown(_, _) => anyhow::bail!("expression type is unknown"),
            SVariable::Bool(_, _) => ExprType::Bool(*code_ref),
            SVariable::F32(_, _) => ExprType::F32(*code_ref),
            SVariable::I64(_, _) => ExprType::I64(*code_ref),
            SVariable::U8(_, _) => ExprType::U8(*code_ref),
            SVariable::Array(svar, size_type) => ExprType::Array(
                *code_ref,
                Box::new(svar.expr_type(code_ref)?),
                size_type.expr_type(),
            ),
            SVariable::Address(_, _) => ExprType::Address(*code_ref),
            SVariable::Struct(_, name, _, _) => {
                ExprType::Struct(*code_ref, Box::new(name.to_string()))
            }
        })
    }
    pub fn expect_f32(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::F32(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected f32 {}", code_ref, v, ctx),
        }
    }
    pub fn expect_i64(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::I64(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected i64 {}", code_ref, v, ctx),
        }
    }
    pub fn expect_u8(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::U8(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected u8 {}", code_ref, v, ctx),
        }
    }
    pub fn expect_bool(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Bool(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected bool {}", code_ref, v, ctx),
        }
    }
    pub fn expect_array(
        &self,
        code_ref: &CodeRef,
        expect_ty: ExprType,
        expect_size_type: ArraySizedExpr,
        ctx: &str,
    ) -> anyhow::Result<Variable> {
        match self {
            SVariable::Array(svar, size_type) => {
                if size_type.expr_type() != expect_size_type {
                    anyhow::bail!(
                        "{} incorrect length {:?} expected {:?} found {}",
                        code_ref,
                        expect_size_type,
                        size_type,
                        ctx
                    )
                }
                let var_ty = svar.expr_type(code_ref)?;
                if var_ty != expect_ty {
                    anyhow::bail!(
                        "incorrect type {} expected Array{} {}",
                        var_ty,
                        expect_ty,
                        ctx
                    )
                } else {
                    Ok(svar.inner())
                }
            }
            v => anyhow::bail!("incorrect type {} expected Array {}", v, ctx),
        }
    }
    pub fn expect_address(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Address(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Address {}", code_ref, v, ctx),
        }
    }
    pub fn expect_struct(
        &self,
        code_ref: &CodeRef,
        name: &str,
        ctx: &str,
    ) -> anyhow::Result<Variable> {
        match self {
            SVariable::Struct(varname, sname, v, _return_struct) => {
                if sname == name {
                    Ok(*v)
                } else {
                    anyhow::bail!(
                        "{} incorrect type {} expected Struct {} {}",
                        code_ref,
                        varname,
                        name,
                        ctx
                    )
                }
            }
            v => anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx),
        }
    }

    pub fn from(
        builder: &mut FunctionBuilder,
        expr_type: &ExprType,
        name: String,
        var: Variable,
    ) -> anyhow::Result<SVariable> {
        Ok(match expr_type {
            ExprType::Bool(_code_ref) => SVariable::Bool(name, var),
            ExprType::F32(_code_ref) => SVariable::F32(name, var),
            ExprType::I64(_code_ref) => SVariable::I64(name, var),
            ExprType::U8(_code_ref) => SVariable::U8(name, var),
            ExprType::Array(_code_ref, ty, size_type) => SVariable::Array(
                Box::new(SVariable::from(builder, ty, name, var)?),
                ArraySized::from(builder, size_type),
            ),
            ExprType::Address(_code_ref) => SVariable::Address(name, var),
            ExprType::Tuple(_code_ref, _) => anyhow::bail!("use SVariable::from_tuple"),
            ExprType::Struct(_code_ref, name) => {
                SVariable::Struct(name.to_string(), name.to_string(), var, false)
                //last bool is return struct
            }
            ExprType::Void(code_ref) => anyhow::bail!("{} SVariable cannot be void", code_ref),
        })
    }
}

#[instrument(
    level = "info",
    skip(
        builder,
        module,
        func,
        entry_block,
        variables,
        per_scope_vars,
        inline_arg_values
    )
)]
pub fn declare_param_and_return_variables(
    index: &mut usize,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func: &Function,
    entry_block: Block,
    variables: &mut HashMap<String, SVariable>,
    per_scope_vars: &mut HashSet<String>,
    inline_arg_values: &Option<Vec<Value>>,
) -> anyhow::Result<HashSet<String>> {
    let mut return_var_names = HashSet::new();
    //Declare returns
    let entry_block_is_offset = if !func.returns.is_empty() {
        let expr_type = &func.returns[0].expr_type;
        match expr_type {
            ExprType::Struct(code_ref, ..)
            | ExprType::Array(code_ref, _, ArraySizedExpr::Fixed(..) | ArraySizedExpr::Slice) => {
                trace!(
                    "{}: fn is returning struct/array/slice {} declaring var {}",
                    code_ref,
                    expr_type,
                    &func.returns[0].name
                );
                // When calling a function that will return a struct, Rust (or possibly anything using the C ABI),
                // will allocate the stack space needed for the struct that will be returned. This is allocated in
                // the callers frame, then the stack address is passed as a special argument to the first parameter
                // of the callee.
                // https://docs.wasmtime.dev/api/cranelift/prelude/enum.StackSlotKind.html#variant.StructReturnSlot
                // https://docs.wasmtime.dev/api/cranelift_codegen/ir/enum.ArgumentPurpose.html#variant.StructReturn

                let return_arg = &func.returns[0];
                let (name, val) = if let Some(inline_arg_values) = inline_arg_values {
                    (return_arg.name.clone(), inline_arg_values[0])
                } else {
                    let return_param_val = builder.block_params(entry_block)[0];
                    (return_arg.name.clone(), return_param_val)
                };
                return_var_names.insert(name.clone());
                declare_variable(
                    module.target_config().pointer_type(),
                    &return_arg.expr_type,
                    builder,
                    index,
                    &[&name],
                    variables,
                    per_scope_vars,
                    true,
                )?;
                if let Some(var) = variables.get(&name) {
                    builder.def_var(var.inner(), val);
                }
                true
            }
            _ => {
                for arg in &func.returns {
                    return_var_names.insert(arg.name.clone());
                    //declare_variable(
                    //    module.target_config().pointer_type(),
                    //    &arg.expr_type,
                    //    builder,
                    //    index,
                    //    &[&arg.name],
                    //    variables,
                    //    per_scope_vars,
                    //    false,
                    //)?;
                }
                false
            }
        }
    } else {
        false
    };

    //Declare args
    for (i, arg) in func
        .params
        .iter()
        .filter(|a| a.closure_arg.is_none()) //Skip closure args
        .enumerate()
    {
        let val = if let Some(inline_arg_values) = inline_arg_values {
            if entry_block_is_offset {
                inline_arg_values[i + 1]
            } else {
                inline_arg_values[i]
            }
        } else if entry_block_is_offset {
            builder.block_params(entry_block)[i + 1]
        } else {
            builder.block_params(entry_block)[i]
        };
        declare_variable(
            module.target_config().pointer_type(),
            &arg.expr_type,
            builder,
            index,
            &[&arg.name],
            variables,
            per_scope_vars,
            false,
        )?;
        if let Some(var) = variables.get(&arg.name) {
            builder.def_var(var.inner(), val);
        }
    }

    Ok(return_var_names)
}

pub fn declare_variable(
    ptr_type: Type,
    expr_type: &ExprType,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    names: &[&str],
    variables: &mut HashMap<String, SVariable>,
    per_scope_vars: &mut HashSet<String>,
    return_struct: bool,
) -> anyhow::Result<()> {
    let name = *names.first().unwrap();
    if name.contains('.') {
        return Ok(());
    }
    match expr_type {
        ExprType::Void(code_ref) => {
            anyhow::bail!("{} can't assign void type to {}", code_ref, name)
        }
        ExprType::Bool(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::Bool(name.into(), var));
                per_scope_vars.insert(name.into());
                builder.declare_var(var, types::B1);
                *index += 1;
            }
        }
        ExprType::F32(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::F32(name.into(), var));
                per_scope_vars.insert(name.into());
                builder.declare_var(var, types::F32);
                *index += 1;
            }
        }
        ExprType::I64(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::I64(name.into(), var));
                per_scope_vars.insert(name.into());
                builder.declare_var(var, types::I64);
                *index += 1;
            }
        }
        ExprType::U8(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::U8(name.into(), var));
                per_scope_vars.insert(name.into());
                builder.declare_var(var, types::I8);
                *index += 1;
            }
        }
        ExprType::Array(code_ref, ty, size_type) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(
                    name.into(),
                    SVariable::Array(
                        Box::new(SVariable::from(builder, ty, name.to_string(), var)?),
                        ArraySized::from(builder, size_type),
                    ),
                );
                per_scope_vars.insert(name.into());
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Address(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::Address(name.into(), var));
                per_scope_vars.insert(name.into());
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Tuple(_code_ref, expr_types) => {
            if expr_types.len() == 1 {
                //Single nested tuple
                if let ExprType::Tuple(_code_ref, expr_types) = expr_types.first().unwrap() {
                    for (expr_type, sname) in expr_types.iter().zip(names.iter()) {
                        declare_variable(
                            ptr_type,
                            expr_type,
                            builder,
                            index,
                            &[sname],
                            variables,
                            per_scope_vars,
                            return_struct,
                        )?
                    }
                    return Ok(());
                }
            }
            for (expr_type, sname) in expr_types.iter().zip(names.iter()) {
                declare_variable(
                    ptr_type,
                    expr_type,
                    builder,
                    index,
                    &[sname],
                    variables,
                    per_scope_vars,
                    return_struct,
                )?
            }
        }
        ExprType::Struct(code_ref, structname) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(
                    name.into(),
                    SVariable::Struct(name.into(), structname.to_string(), var, return_struct),
                );
                per_scope_vars.insert(name.into());
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
    }
    Ok(())
}
