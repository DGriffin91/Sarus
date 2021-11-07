use crate::frontend::*;
use crate::jit::*;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use std::collections::HashMap;
use std::fmt::Display;

use tracing::instrument;
use tracing::trace;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArraySized {
    Unsized,                   //size is unknown, just an address with a type
    Sized,                     //start of array address is i64 with size.
    Fixed(Box<SValue>, usize), //size is part of type signature
}

impl ArraySized {
    pub fn expr_type(&self) -> ArraySizedExpr {
        match self {
            ArraySized::Unsized => ArraySizedExpr::Unsized,
            ArraySized::Sized => ArraySizedExpr::Sized,
            ArraySized::Fixed(_, size) => ArraySizedExpr::Fixed(*size),
        }
    }
    pub fn from(builder: &mut FunctionBuilder, size_type: &ArraySizedExpr) -> ArraySized {
        match size_type {
            ArraySizedExpr::Unsized => ArraySized::Unsized,
            ArraySizedExpr::Sized => todo!(),
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
            SValue::Array(sval, size_type) => match size_type {
                ArraySized::Unsized => write!(f, "&[{}]", sval),
                ArraySized::Sized => todo!(),
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
            SVariable::Array(svar, size_type) => match size_type {
                ArraySized::Unsized => write!(f, "&[{}]", svar),
                ArraySized::Sized => todo!(),
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
            v => anyhow::bail!("{} incorrect type {} expected Float {}", code_ref, v, ctx),
        }
    }
    pub fn expect_i64(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::I64(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Int {}", code_ref, v, ctx),
        }
    }
    pub fn expect_bool(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Bool(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Bool {}", code_ref, v, ctx),
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
    skip(builder, module, func, entry_block, env, variables, inline_arg_values)
)]
pub fn declare_variables(
    index: &mut usize,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    func: &Function,
    entry_block: Block,
    env: &mut Env,
    variables: &mut HashMap<String, SVariable>,
    inline_arg_values: &Option<Vec<Value>>,
) -> anyhow::Result<()> {
    //Declare returns
    let entry_block_is_offset = if !func.returns.is_empty() {
        if let ExprType::Struct(code_ref, struct_name) = &func.returns[0].expr_type {
            trace!(
                "{}: fn is returning struct {} declaring var {}",
                code_ref,
                struct_name,
                &func.returns[0].name
            );
            // When calling a function that will return a struct, Rust (or possibly anything using the C ABI),
            // will allocate the stack space needed for the struct that will be returned. This is allocated in
            // the callers frame, then the stack address is passed as a special argument to the first parameter
            // of the callee.
            // https://docs.wasmtime.dev/api/cranelift/prelude/enum.StackSlotKind.html#variant.StructReturnSlot
            // https://docs.wasmtime.dev/api/cranelift_codegen/ir/enum.ArgumentPurpose.html#variant.StructReturn

            let return_struct_arg = &func.returns[0];
            let (name, val) = if let Some(inline_arg_values) = inline_arg_values {
                (return_struct_arg.name.clone(), inline_arg_values[0])
            } else {
                let return_struct_param_val = builder.block_params(entry_block)[0];
                (return_struct_arg.name.clone(), return_struct_param_val)
            };
            declare_variable(
                module.target_config().pointer_type(),
                &return_struct_arg.expr_type,
                builder,
                index,
                &[&name],
                variables,
                true,
            )?;
            if let Some(var) = variables.get(&name) {
                builder.def_var(var.inner(), val);
            }
            true
        } else {
            for arg in &func.returns {
                declare_variable(
                    module.target_config().pointer_type(),
                    &arg.expr_type,
                    builder,
                    index,
                    &[&arg.name],
                    variables,
                    false,
                )?;
            }
            false
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
            false,
        )?;
        if let Some(var) = variables.get(&arg.name) {
            builder.def_var(var.inner(), val);
        }
    }

    //Declare body
    for expr in &func.body {
        declare_variables_in_stmt(
            module.target_config().pointer_type(),
            builder,
            index,
            expr,
            func,
            env,
            variables,
        )?;
    }

    Ok(())
}

/// Recursively descend through the AST, translating all implicit
/// variable declarations.
fn declare_variables_in_stmt(
    ptr_type: types::Type,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    expr: &Expr,
    func: &Function,
    env: &mut Env,
    variables: &mut HashMap<String, SVariable>,
) -> anyhow::Result<()> {
    match *expr {
        Expr::Assign(_code_ref, ref to_exprs, ref from_exprs) => {
            if to_exprs.len() == from_exprs.len() {
                for (to_expr, _from_expr) in to_exprs.iter().zip(from_exprs.iter()) {
                    if let Expr::Identifier(_code_ref, name) = to_expr {
                        declare_variable_from_expr(
                            ptr_type,
                            expr,
                            builder,
                            index,
                            &[&name],
                            func,
                            env,
                            variables,
                        )?;
                    }
                }
            } else {
                let mut sto_exprs = Vec::new();
                for to_expr in to_exprs.iter() {
                    if let Expr::Identifier(_code_ref, name) = to_expr {
                        sto_exprs.push(name.as_str());
                    }
                }
                declare_variable_from_expr(
                    ptr_type, expr, builder, index, &sto_exprs, func, env, variables,
                )?;
            }
        }
        Expr::IfElse(_code_ref, ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(ptr_type, builder, index, stmt, func, env, variables)?;
            }
            for stmt in else_body {
                declare_variables_in_stmt(ptr_type, builder, index, stmt, func, env, variables)?;
            }
        }
        Expr::IfThenElseIfElse(_code_ref, ref expr_bodies, ref else_body) => {
            for (_condition, body) in expr_bodies {
                for stmt in body {
                    declare_variables_in_stmt(
                        ptr_type, builder, index, stmt, func, env, variables,
                    )?;
                }
            }
            for stmt in else_body {
                declare_variables_in_stmt(ptr_type, builder, index, stmt, func, env, variables)?;
            }
        }
        Expr::WhileLoop(_code_ref, ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(ptr_type, builder, index, stmt, func, env, variables)?;
            }
        }
        _ => (),
    }
    Ok(())
}

/// Declare a single variable declaration.
fn declare_variable_from_expr(
    ptr_type: Type,
    expr: &Expr,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    names: &[&str],
    func: &Function,
    env: &mut Env,
    variables: &mut HashMap<String, SVariable>,
) -> anyhow::Result<()> {
    match expr {
        Expr::Assign(code_ref, ref _to_exprs, ref from_exprs) => {
            for from_expr in from_exprs.iter() {
                trace!(
                    "{} declare_variable_from_expr Expr::Assign {}",
                    code_ref.s(&env.file_idx),
                    from_expr,
                );

                let expr_type = ExprType::of(from_expr, env, &func.name, variables)?;
                trace!("{:?}", names);
                declare_variable(
                    ptr_type, &expr_type, builder, index, names, variables, false,
                )?;
            }
        }
        expr => {
            anyhow::bail!(
                "{} declare_variable_from_expr encountered non assignment, should be unreachable {}",
                expr.get_code_ref(),
                expr,
            );
        }
    };
    Ok(())
}

fn declare_variable(
    ptr_type: Type,
    expr_type: &ExprType,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    names: &[&str],
    variables: &mut HashMap<String, SVariable>,
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
                builder.declare_var(var, types::B1);
                *index += 1;
            }
        }
        ExprType::F32(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::F32(name.into(), var));
                builder.declare_var(var, types::F32);
                *index += 1;
            }
        }
        ExprType::I64(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::I64(name.into(), var));
                builder.declare_var(var, types::I64);
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
                ); //name.into(), var));
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Address(code_ref) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::Address(name.into(), var));
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
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
    }
    Ok(())
}
