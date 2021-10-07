use crate::frontend::*;
use crate::sarus_std_lib;
use crate::validator::validate_program;
use crate::validator::ExprType;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::ir::ArgumentPurpose;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Display;
use std::slice;

/// The basic JIT class.
pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The data context, which is to data objects what `ctx` is to functions.
    data_ctx: DataContext,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    //CLIF cranelift IR string, by function name
    pub clif: HashMap<String, String>,

    //local variables for each function
    pub variables: HashMap<String, HashMap<String, SVariable>>,
}

impl Default for JIT {
    fn default() -> Self {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            clif: HashMap::new(),
            variables: HashMap::new(),
        }
    }
}

pub fn new_jit_builder() -> JITBuilder {
    JITBuilder::new(cranelift_module::default_libcall_names())
}

impl JIT {
    pub fn from(jit_builder: JITBuilder) -> Self {
        let module = JITModule::new(jit_builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            clif: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// Compile a string in the toy language into machine code.
    pub fn translate(&mut self, prog: Vec<Declaration>) -> anyhow::Result<()> {
        //let mut return_counts = HashMap::new();
        //for func in prog.iter().filter_map(|d| match d {
        //    Declaration::Function(func) => Some(func.clone()),
        //    _ => None,
        //}) {
        //    return_counts.insert(func.name.to_string(), func.returns.len());
        //}

        let mut funcs = HashMap::new();
        for func in prog.iter().filter_map(|d| match d {
            Declaration::Function(func) => Some(func.clone()),
            _ => None,
        }) {
            funcs.insert(func.name.clone(), func);
        }

        let struct_map = create_struct_map(&prog, self.module.target_config().pointer_type())?;

        // First, parse the string, producing AST nodes.
        for d in prog.clone() {
            match d {
                Declaration::Function(func) => {
                    if func.extern_func {
                        //Don't parse contents of std func, it will be empty
                        continue;
                    }
                    ////println!(
                    ////    "name {:?}, params {:?}, the_return {:?}",
                    ////    &name, &params, &the_return
                    ////);
                    //// Then, translate the AST nodes into Cranelift IR.
                    self.codegen(&func, funcs.to_owned(), &prog, &struct_map)?;
                    // Next, declare the function to jit. Functions must be declared
                    // before they can be called, or defined.
                    let id = self
                        .module
                        .declare_function(&func.name, Linkage::Export, &self.ctx.func.signature)
                        .map_err(|e| {
                            anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e)
                        })?;

                    ////println!("ID IS {}", id);
                    // Define the function to jit. This finishes compilation, although
                    // there may be outstanding relocations to perform. Currently, jit
                    // cannot finish relocations until all functions to be called are
                    // defined. For this toy demo for now, we'll just finalize the
                    // function below.
                    self.module
                        .define_function(
                            id,
                            &mut self.ctx,
                            &mut codegen::binemit::NullTrapSink {},
                            &mut codegen::binemit::NullStackMapSink {},
                        )
                        .map_err(|e| {
                            anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e)
                        })?;

                    // Now that compilation is finished, we can clear out the context state.
                    self.module.clear_context(&mut self.ctx);

                    // Finalize the functions which we just defined, which resolves any
                    // outstanding relocations (patching in addresses, now that they're
                    // available).
                    self.module.finalize_definitions();
                }
                _ => continue,
            };
        }

        Ok(())
    }

    pub fn get_func(&mut self, fn_name: &str) -> anyhow::Result<*const u8> {
        match self.module.get_name(fn_name) {
            Some(func) => match func {
                cranelift_module::FuncOrDataId::Func(id) => {
                    Ok(self.module.get_finalized_function(id))
                }
                cranelift_module::FuncOrDataId::Data(_) => {
                    anyhow::bail!("function {} required, data found", fn_name);
                }
            },
            None => anyhow::bail!("No function {} found", fn_name),
        }
    }

    /// Create a zero-initialized data section.
    pub fn create_data(&mut self, name: &str, contents: Vec<u8>) -> anyhow::Result<&[u8]> {
        // The steps here are analogous to `compile`, except that data is much
        // simpler than functions.
        self.data_ctx.define(contents.into_boxed_slice());
        let id = self
            .module
            .declare_data(name, Linkage::Export, true, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        self.module
            .define_data(id, &self.data_ctx)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        self.data_ctx.clear();
        self.module.finalize_definitions();
        let buffer = self.module.get_finalized_data(id);

        Ok(unsafe { slice::from_raw_parts(buffer.0, buffer.1) })
    }

    // Translate from AST nodes into Cranelift IR.
    fn codegen(
        &mut self,
        func: &Function,
        funcs: HashMap<String, Function>,
        prog: &[Declaration],
        struct_map: &HashMap<String, StructDef>,
    ) -> anyhow::Result<()> {
        let ptr_ty = self.module.target_config().pointer_type();

        if func.returns.len() > 0 {
            if let ExprType::Struct(_struct_name) = &func.returns[0].expr_type {
                if func.returns.len() > 1 {
                    anyhow::bail!(
                        "If returning a struct, only 1 return value is currently supported"
                    )
                }
                self.ctx
                    .func
                    .signature
                    .params
                    .push(AbiParam::special(ptr_ty, ArgumentPurpose::StructReturn));
            } else {
                for ret_arg in &func.returns {
                    self.ctx.func.signature.returns.push(AbiParam::new(
                        ret_arg
                            .expr_type
                            .cranelift_type(self.module.target_config().pointer_type(), false)?,
                    ));
                }
            }
        }

        for p in &func.params {
            self.ctx.func.signature.params.push({
                match &p.expr_type {
                    ExprType::F64 => AbiParam::new(types::F64),
                    ExprType::I64 => AbiParam::new(types::I64),
                    ExprType::Array(_ty, _len) => AbiParam::new(ptr_ty),
                    ExprType::Address => AbiParam::new(ptr_ty),
                    ExprType::Void => continue,
                    ExprType::Bool => AbiParam::new(types::B1),
                    ExprType::Struct(_) => AbiParam::new(ptr_ty),
                    ExprType::Tuple(_) => anyhow::bail!("Tuple as parameter not supported"),
                }
            });
        }

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Create the entry block, to start emitting code in.
        let entry_block = builder.create_block();

        // Since this is the entry block, add block parameters corresponding to
        // the function's parameters.
        builder.append_block_params_for_function_params(entry_block);

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);

        // And, tell the builder that this block will have no further
        // predecessors. Since it's the entry block, it won't have any
        // predecessors.
        builder.seal_block(entry_block);
        let constant_vars = sarus_std_lib::get_constants();
        // The toy language allows variables to be declared implicitly.
        // Walk the AST and declare all implicitly-declared variables.

        let mut env = Env {
            prog,
            constant_vars,
            struct_map: struct_map.clone(),
            variables: HashMap::new(),
            ptr_ty,
            funcs,
        };

        println!("declare_variables {}", func.name);
        declare_variables(
            &mut builder,
            &mut self.module,
            &func.params,
            &func.returns,
            &func.body,
            entry_block,
            &mut env,
        )?;

        //Keep function vars around for later debug/print
        self.variables
            .insert(func.name.to_string(), env.variables.clone());

        println!("validate_program {}", func.name);

        //Check every statement, this can catch funcs with no assignment, etc...
        validate_program(&func.body, &env)?;

        println!("FunctionTranslator {}", func.name);
        let ptr_ty = self.module.target_config().pointer_type();

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            func,
            module: &mut self.module,
            ptr_ty,
            env: &env,
        };
        for expr in &func.body {
            trans.translate_expr(expr)?;
        }

        // Set up the return variable of the function. Above, we declared a
        // variable to hold the return value. Here, we just do a use of that
        // variable.
        let mut return_values = Vec::new();
        for ret in func.returns.iter() {
            let return_variable = trans.env.variables.get(&ret.name).unwrap();
            let v =
                match &ret.expr_type {
                    ExprType::F64 => trans
                        .builder
                        .use_var(return_variable.expect_f64("return_variable")?),
                    ExprType::I64 => trans
                        .builder
                        .use_var(return_variable.expect_i64("return_variable")?),
                    ExprType::Array(ty, len) => trans.builder.use_var(
                        return_variable.expect_array(*ty.clone(), *len, "return_variable")?,
                    ),
                    ExprType::Address => trans
                        .builder
                        .use_var(return_variable.expect_address("return_variable")?),
                    ExprType::Void => continue,
                    ExprType::Bool => trans
                        .builder
                        .use_var(return_variable.expect_bool("return_variable")?),
                    ExprType::Tuple(_) => anyhow::bail!("tuple not supported in return"),
                    //We don't actually return structs, they are passed in as StackSlotKind::StructReturnSlot and written to from there
                    ExprType::Struct(_) => continue, //trans.builder.use_var(return_variable.expect_struct(n, "codegen return variables")?)
                };
            return_values.push(v);
        }

        // Emit the return instruction.
        trans.builder.ins().return_(&return_values);

        // Tell the builder we're done with this function.
        trans.builder.finalize();

        //Keep clif around for later debug/print
        self.clif.insert(
            func.name.to_string(),
            trans.builder.func.display(None).to_string(),
        );
        Ok(())
    }

    pub fn add_math_constants(&mut self) -> anyhow::Result<()> {
        for (name, val) in sarus_std_lib::get_constants() {
            self.create_data(&name, val.to_ne_bytes().to_vec())?;
        }
        Ok(())
    }

    pub fn print_clif(&self, show_vars: bool) {
        for (func_name, func_clif) in &self.clif {
            let mut func_clif = func_clif.clone();
            if show_vars {
                for (var_name, var) in &self.variables[func_name] {
                    let clif_var_name = format!("v{}", var.inner().index());
                    func_clif = func_clif
                        .replace(&clif_var_name, &format!("{}~{}", clif_var_name, var_name));
                }
            }
            println!("//{}\n{}", func_name, func_clif);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SValue {
    Void,
    Unknown(Value),
    Bool(Value),
    F64(Value),
    I64(Value),
    Array(Box<SValue>, Option<usize>),
    Address(Value),
    Tuple(Vec<SValue>),
    Struct(String, Value),
}

impl Display for SValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SValue::Unknown(_) => write!(f, "unknown"),
            SValue::Bool(_) => write!(f, "bool"),
            SValue::F64(_) => write!(f, "f64"),
            SValue::I64(_) => write!(f, "i64"),
            SValue::Array(sval, len) => {
                if let Some(len) = len {
                    write!(f, "&[{}; {}]", sval, len)
                } else {
                    write!(f, "&[{}]", sval)
                }
            }
            SValue::Address(_) => write!(f, "&"),
            SValue::Void => write!(f, "void"),
            SValue::Tuple(v) => write!(f, "({})", v.len()),
            SValue::Struct(name, _) => write!(f, "{}", name),
        }
    }
}

impl SValue {
    fn from(expr_type: &ExprType, value: Value) -> anyhow::Result<SValue> {
        Ok(match expr_type {
            ExprType::Void => SValue::Void,
            ExprType::Bool => SValue::Bool(value),
            ExprType::F64 => SValue::F64(value),
            ExprType::I64 => SValue::I64(value),
            ExprType::Array(ty, len) => SValue::Array(Box::new(SValue::from(ty, value)?), *len),
            ExprType::Address => SValue::Address(value),
            ExprType::Tuple(_) => anyhow::bail!("use SValue::from_tuple"),
            ExprType::Struct(name) => SValue::Struct(name.to_string(), value),
        })
    }
    fn get_from_variable(
        builder: &mut FunctionBuilder,
        variable: &SVariable,
    ) -> anyhow::Result<SValue> {
        Ok(match variable {
            SVariable::Unknown(_, v) => SValue::Unknown(builder.use_var(*v)),
            SVariable::Bool(_, v) => SValue::Bool(builder.use_var(*v)),
            SVariable::F64(_, v) => SValue::F64(builder.use_var(*v)),
            SVariable::I64(_, v) => SValue::I64(builder.use_var(*v)),
            SVariable::Address(_, v) => SValue::Address(builder.use_var(*v)),
            SVariable::Array(svar, size) => {
                SValue::Array(Box::new(SValue::get_from_variable(builder, svar)?), *size)
            }
            SVariable::Struct(_varname, structname, v, _return_struct) => {
                SValue::Struct(structname.to_string(), builder.use_var(*v))
            }
        })
    }
    fn replace_value(&self, value: Value) -> anyhow::Result<SValue> {
        Ok(match self {
            SValue::Void => SValue::Void,
            SValue::Bool(_) => SValue::Bool(value),
            SValue::F64(_) => SValue::F64(value),
            SValue::I64(_) => SValue::I64(value),
            SValue::Array(sval, len) => SValue::Array(Box::new(sval.replace_value(value)?), *len),
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
    pub fn expr_type(&self) -> anyhow::Result<ExprType> {
        Ok(match self {
            SValue::Unknown(_) => anyhow::bail!("expression type is unknown"),
            SValue::Bool(_) => ExprType::Bool,
            SValue::F64(_) => ExprType::F64,
            SValue::I64(_) => ExprType::I64,
            SValue::Array(sval, len) => ExprType::Array(Box::new(sval.expr_type()?), *len),
            SValue::Address(_) => ExprType::Address,
            SValue::Struct(name, _) => ExprType::Struct(Box::new(name.to_string())),
            SValue::Void => ExprType::Void,
            SValue::Tuple(_) => todo!(),
        })
    }
    fn inner(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Unknown(v) => Ok(*v),
            SValue::Bool(v) => Ok(*v),
            SValue::F64(v) => Ok(*v),
            SValue::I64(v) => Ok(*v),
            SValue::Array(sval, _len) => Ok(sval.inner(ctx)?),
            SValue::Address(v) => Ok(*v),
            SValue::Void => anyhow::bail!("void has no inner {}", ctx),
            SValue::Tuple(v) => anyhow::bail!("inner does not support tuple {:?} {}", v, ctx),
            SValue::Struct(_, v) => Ok(*v),
        }
    }
    fn expect_f64(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::F64(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected F64 {}", v, ctx),
        }
    }
    fn expect_i64(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::I64(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected I64 {}", v, ctx),
        }
    }
    fn expect_bool(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Bool(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Bool {}", v, ctx),
        }
    }
    fn expect_array(
        &self,
        expect_ty: ExprType,
        expect_len: Option<usize>,
        ctx: &str,
    ) -> anyhow::Result<Value> {
        match self {
            SValue::Array(sval, len) => {
                if expect_len != *len {
                    anyhow::bail!(
                        "incorrect length {:?} expected {:?} found {}",
                        expect_len,
                        len,
                        ctx
                    );
                }
                let var_ty = sval.expr_type()?;
                if var_ty != expect_ty {
                    anyhow::bail!(
                        "incorrect type {} expected Array{} {}",
                        var_ty,
                        expect_ty,
                        ctx
                    )
                } else {
                    sval.inner("expect_array")
                }
            }
            v => anyhow::bail!("incorrect type {} expected UnboundedArrayF64 {}", v, ctx),
        }
    }
    fn expect_address(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Address(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Address {}", v, ctx),
        }
    }
    fn expect_struct(&self, name: &str, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Struct(sname, v) => {
                if sname == name {
                    return Ok(*v);
                } else {
                    anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx)
                }
            }
            v => anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx),
        }
    }
}

pub struct Env<'a> {
    pub prog: &'a [Declaration],
    pub constant_vars: HashMap<String, f64>,
    pub struct_map: HashMap<String, StructDef>,
    pub variables: HashMap<String, SVariable>,
    pub ptr_ty: types::Type,
    pub funcs: HashMap<String, Function>,
}

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    env: &'a Env<'a>,
    func: &'a Function,
    ptr_ty: types::Type,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &Expr) -> anyhow::Result<SValue> {
        //dbg!(&expr);
        match expr {
            Expr::LiteralFloat(literal) => Ok(SValue::F64(
                self.builder.ins().f64const::<f64>(literal.parse().unwrap()),
            )),
            Expr::LiteralInt(literal) => Ok(SValue::I64(
                self.builder
                    .ins()
                    .iconst::<i64>(types::I64, literal.parse().unwrap()),
            )),
            Expr::LiteralString(literal) => self.translate_string(literal),
            Expr::Binop(op, lhs, rhs) => Ok(self.translate_binop(*op, lhs, rhs, false)?.0),
            Expr::Unaryop(op, lhs) => self.translate_unaryop(*op, lhs),
            Expr::Compare(cmp, lhs, rhs) => self.translate_cmp(*cmp, lhs, rhs),
            Expr::Call(name, args) => self.translate_call(name, args, None),
            Expr::GlobalDataAddr(name) => Ok(SValue::Array(
                Box::new(SValue::F64(
                    self.translate_global_data_addr(self.ptr_ty, name),
                )),
                None,
            )),
            Expr::Identifier(name) => {
                match self.env.variables.get(name) {
                    Some(var) => SValue::get_from_variable(&mut self.builder, var),
                    None => Ok(SValue::F64(
                        //TODO Don't assume this is a float (this is used for math const)
                        self.translate_global_data_addr(types::F64, name),
                    )), //Try to load global
                }
            }
            Expr::Assign(to_exprs, from_exprs) => self.translate_assign(to_exprs, from_exprs),
            Expr::AssignOp(_op, _lhs, _rhs) => {
                unimplemented!("currently AssignOp is turned into seperate assign and op")
                //self.translate_math_assign(*op, lhs, rhs),
                //for what this used to look like see:
                //https://github.com/DGriffin91/sarus/tree/cd4bf6472bf02f00ea6037d606842ec84d0ff205
            }
            Expr::NewStruct(struct_name, fields) => self.translate_new_struct(struct_name, fields),
            Expr::IfThen(condition, then_body) => {
                self.translate_if_then(condition, then_body)?;
                Ok(SValue::Void)
            }
            Expr::IfElse(condition, then_body, else_body) => {
                self.translate_if_else(condition, then_body, else_body)
            }
            Expr::WhileLoop(condition, loop_body) => {
                self.translate_while_loop(condition, loop_body)?;
                Ok(SValue::Void)
            }
            Expr::Block(b) => b
                .into_iter()
                .map(|e| self.translate_expr(e))
                .last()
                .unwrap(),
            Expr::LiteralBool(b) => Ok(SValue::Bool(self.builder.ins().bconst(types::B1, *b))),
            Expr::Parentheses(expr) => self.translate_expr(expr),
            Expr::ArrayAccess(name, idx_expr) => {
                self.translate_array_get(name.to_string(), idx_expr, false)
            } //Expr::ArraySet(name, idx_expr, expr) => {
              //    self.translate_array_set(name.to_string(), idx_expr, expr)
              //}
        }
    }

    fn translate_string(&mut self, literal: &String) -> anyhow::Result<SValue> {
        let cstr = CString::new(literal.replace("\\n", "\n").to_string()).unwrap();
        let bytes = cstr.to_bytes_with_nul();
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            types::I8.bytes() * bytes.len() as u32,
        ));
        let stack_slot_address =
            self.builder
                .ins()
                .stack_addr(self.ptr_ty, stack_slot, Offset32::new(0));
        //TODO Is this really how this is done?
        for (i, c) in bytes.iter().enumerate() {
            let v = self.builder.ins().iconst::<i64>(types::I64, *c as i64);
            self.builder.ins().istore8(
                MemFlags::new(),
                v,
                stack_slot_address,
                Offset32::new(i as i32),
            );
        }
        Ok(SValue::Address(stack_slot_address))
    }

    fn translate_binop(
        &mut self,
        op: Binop,
        lhs: &Expr,
        rhs: &Expr,
        get_address: bool,
    ) -> anyhow::Result<(SValue, Option<StructField>)> {
        //println!("{}:{} translate_binop {} {} {}", file!(), line!(), lhs, op, rhs);
        if let Binop::DotAccess = op {
        } else {
            return Ok((self.translate_math_binop(op, lhs, rhs)?, None));
        }

        let mut path = Vec::new();
        let mut lhs_val = None;
        let mut struct_field_def = None;

        let mut curr_expr = Some(lhs);
        let mut next_expr = Some(rhs);

        loop {
            //println!("curr_expr {:?} next_expr {:?}", &curr_expr, &next_expr);
            //println!("path {:?}", &path);
            match curr_expr {
                Some(expr) => {
                    curr_expr = next_expr;
                    next_expr = None;
                    match expr {
                        Expr::Call(fn_name, args) => {
                            if path.len() == 0 {
                                lhs_val = Some(self.translate_call(fn_name, args, lhs_val)?);
                            } else {
                                let sval = if path.len() > 1 {
                                    let spath = path
                                        .iter()
                                        .map(|lhs_i: &Expr| lhs_i.to_string())
                                        .collect::<Vec<String>>();
                                    let (sval_address, struct_def) =
                                        self.get_struct_field_address(spath)?;
                                    if let ExprType::Struct(_name) = struct_def.clone().expr_type {
                                        struct_field_def = Some(struct_def);
                                        sval_address
                                    } else {
                                        //dbg!(&struct_def);
                                        self.get_struct_field(sval_address, &struct_def)?
                                    }
                                } else {
                                    self.translate_expr(&path[0])?
                                };
                                //dbg!(&sval);
                                lhs_val = Some(self.translate_call(fn_name, args, Some(sval))?);
                                path = Vec::new();
                            }
                        }
                        Expr::LiteralFloat(_) => todo!(),
                        Expr::LiteralInt(_) => todo!(),
                        Expr::LiteralBool(_) => todo!(),
                        Expr::LiteralString(_) => lhs_val = Some(self.translate_expr(expr)?),
                        Expr::Identifier(_) => path.push(expr.clone()),
                        Expr::Binop(op, lhs, rhs) => {
                            if let Binop::DotAccess = op {
                                curr_expr = Some(lhs);
                                next_expr = Some(rhs);
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
                        Expr::Parentheses(e) => lhs_val = Some(self.translate_expr(e)?),
                        Expr::ArrayAccess(name, idx_expr) => {
                            if path.len() > 0 {
                                let mut spath = path
                                    .iter()
                                    .map(|lhs_i: &Expr| lhs_i.to_string())
                                    .collect::<Vec<String>>();
                                spath.push(name.to_string());
                                let (sval_address, struct_def) =
                                    self.get_struct_field_address(spath)?;
                                struct_field_def = Some(struct_def.clone());
                                let array_ptr =
                                    if let ExprType::Struct(_name) = struct_def.clone().expr_type {
                                        sval_address
                                    } else {
                                        self.get_struct_field(sval_address, &struct_def)?
                                    };

                                let base_type = match &struct_def.expr_type {
                                    ExprType::Array(ty, _len) => {
                                        ty.cranelift_type(self.ptr_ty, true)?
                                    }
                                    _ => {
                                        anyhow::bail!("can't index type {}", &struct_def.expr_type)
                                    }
                                };

                                //dbg!(&struct_def.expr_type);

                                //TODO support other array types
                                //struct_def.expr_type.width(ptr_ty, &self.env.struct_map)

                                let idx_val = self.translate_expr(idx_expr).unwrap();
                                let idx_val = match idx_val {
                                    SValue::I64(v) => v,
                                    _ => anyhow::bail!("only int supported for array access"),
                                };

                                let field_address_at_ids = self.get_array_address_from_ptr(
                                    base_type.bytes(),
                                    array_ptr.inner("Expr::ArrayAccess")?,
                                    idx_val,
                                );

                                if get_address {
                                    lhs_val = Some(SValue::Address(field_address_at_ids));
                                } else {
                                    let val = self.builder.ins().load(
                                        base_type,
                                        MemFlags::new(),
                                        field_address_at_ids,
                                        Offset32::new(0),
                                    );
                                    lhs_val = match &struct_def.expr_type {
                                        ExprType::Array(ty, _len) => Some(SValue::from(ty, val)?),
                                        _ => {
                                            anyhow::bail!(
                                                "can't index type {}",
                                                &struct_def.expr_type
                                            )
                                        }
                                    };
                                }
                            } else {
                                lhs_val = Some(self.translate_array_get(
                                    name.to_string(),
                                    idx_expr,
                                    get_address,
                                )?);
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
            let (sval_address, struct_def) = self.get_struct_field_address(spath)?;
            if get_address {
                struct_field_def = Some(struct_def);
                lhs_val = Some(sval_address);
            } else if let ExprType::Struct(_name) = struct_def.clone().expr_type {
                struct_field_def = Some(struct_def);
                lhs_val = Some(sval_address);
            } else {
                lhs_val = Some(self.get_struct_field(sval_address, &struct_def)?)
            }
        }

        if let Some(lhs_val) = lhs_val {
            Ok((lhs_val, struct_field_def))
        } else {
            anyhow::bail!("No value found");
        }
    }

    fn translate_math_binop(
        &mut self,
        op: Binop,
        lhs: &Expr,
        rhs: &Expr,
    ) -> anyhow::Result<SValue> {
        let lhs_v = self.translate_expr(lhs)?;
        let rhs_v = self.translate_expr(rhs)?;
        match lhs_v {
            SValue::F64(a) => match rhs_v {
                SValue::F64(b) => Ok(SValue::F64(self.binop_float(op, a, b)?)),
                _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs_v, op, rhs_v),
            },
            SValue::I64(a) => match rhs_v {
                SValue::I64(b) => Ok(SValue::I64(self.binop_int(op, a, b)?)),
                _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs_v, op, rhs_v),
            },
            SValue::Bool(a) => match rhs_v {
                SValue::Bool(b) => Ok(SValue::Bool(self.binop_bool(op, a, b)?)),
                _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs_v, op, rhs_v),
            },
            SValue::Void
            | SValue::Unknown(_)
            | SValue::Array(_, _)
            | SValue::Address(_)
            | SValue::Struct(_, _)
            | SValue::Tuple(_) => {
                anyhow::bail!("operation not supported: {:?} {} {:?}", lhs_v, op, rhs_v)
            }
        }
    }

    fn translate_unaryop(&mut self, op: Unaryop, lhs: &Expr) -> anyhow::Result<SValue> {
        let lhs = self.translate_expr(lhs)?;

        Ok(match lhs {
            SValue::Bool(lhs) => {
                //TODO I'm sure this has absolutely terrible performance
                //thread 'unary_not' panicked at 'not implemented: bool bnot', [...]\cranelift-codegen-0.76.0\src\isa\x64\lower.rs:2375:17
                //SValue::Bool(self.builder.ins().bnot(lhs))
                let i_bool = self.builder.ins().bint(types::I64, lhs);
                let false_const = self.builder.ins().iconst(types::I64, 0);
                SValue::Bool(self.builder.ins().icmp(IntCC::Equal, i_bool, false_const))
            }
            SValue::Void
            | SValue::F64(_)
            | SValue::I64(_)
            | SValue::Unknown(_)
            | SValue::Array(_, _)
            | SValue::Address(_)
            | SValue::Struct(_, _)
            | SValue::Tuple(_) => {
                anyhow::bail!("operation not supported: {:?} {}", lhs, op)
            }
        })
    }

    fn binop_float(&mut self, op: Binop, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        Ok(match op {
            Binop::Add => self.builder.ins().fadd(lhs, rhs),
            Binop::Sub => self.builder.ins().fsub(lhs, rhs),
            Binop::Mul => self.builder.ins().fmul(lhs, rhs),
            Binop::Div => self.builder.ins().fdiv(lhs, rhs),
            Binop::LogicalAnd | Binop::LogicalOr | Binop::DotAccess => {
                anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs)
            }
        })
    }

    fn binop_int(&mut self, op: Binop, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        Ok(match op {
            Binop::Add => self.builder.ins().iadd(lhs, rhs),
            Binop::Sub => self.builder.ins().isub(lhs, rhs),
            Binop::Mul => self.builder.ins().imul(lhs, rhs),
            Binop::Div => self.builder.ins().sdiv(lhs, rhs),
            Binop::LogicalAnd | Binop::LogicalOr | Binop::DotAccess => {
                anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs)
            }
        })
    }

    fn binop_bool(&mut self, op: Binop, lhs: Value, rhs: Value) -> anyhow::Result<Value> {
        Ok(match op {
            Binop::LogicalAnd => self.builder.ins().band(lhs, rhs),
            Binop::LogicalOr => self.builder.ins().bor(lhs, rhs),
            _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs),
        })
    }

    fn translate_cmp(&mut self, cmp: Cmp, lhs: &Expr, rhs: &Expr) -> anyhow::Result<SValue> {
        let lhs = self.translate_expr(lhs).unwrap();
        let rhs = self.translate_expr(rhs).unwrap();
        // if a or b is a float, convert to other to a float
        match lhs {
            SValue::F64(a) => match rhs {
                SValue::F64(b) => Ok(SValue::Bool(self.cmp_float(cmp, a, b))),
                _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
            },
            SValue::I64(a) => match rhs {
                SValue::I64(b) => Ok(SValue::Bool(self.cmp_int(cmp, a, b))),
                _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
            },
            SValue::Bool(a) => match rhs {
                SValue::Bool(b) => Ok(SValue::Bool(self.cmp_bool(cmp, a, b))),
                _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
            },
            SValue::Void
            | SValue::Unknown(_)
            | SValue::Array(_, _)
            | SValue::Address(_)
            | SValue::Struct(_, _)
            | SValue::Tuple(_) => {
                anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, cmp, rhs)
            }
        }
    }

    fn cmp_float(&mut self, cmp: Cmp, lhs: Value, rhs: Value) -> Value {
        let icmp = match cmp {
            Cmp::Eq => FloatCC::Equal,
            Cmp::Ne => FloatCC::NotEqual,
            Cmp::Lt => FloatCC::LessThan,
            Cmp::Le => FloatCC::LessThanOrEqual,
            Cmp::Gt => FloatCC::GreaterThan,
            Cmp::Ge => FloatCC::GreaterThanOrEqual,
        };
        self.builder.ins().fcmp(icmp, lhs, rhs)
    }

    fn cmp_int(&mut self, cmp: Cmp, lhs: Value, rhs: Value) -> Value {
        let icmp = match cmp {
            Cmp::Eq => IntCC::Equal,
            Cmp::Ne => IntCC::NotEqual,
            Cmp::Lt => IntCC::SignedLessThan,
            Cmp::Le => IntCC::SignedLessThanOrEqual,
            Cmp::Gt => IntCC::SignedGreaterThan,
            Cmp::Ge => IntCC::SignedGreaterThanOrEqual,
        };
        self.builder.ins().icmp(icmp, lhs, rhs)
    }

    fn cmp_bool(&mut self, cmp: Cmp, lhs: Value, rhs: Value) -> Value {
        //TODO
        //thread 'logical_operators' panicked at 'not implemented: bool bnot', [...]]\cranelift-codegen-0.76.0\src\isa\x64\lower.rs:2375:17
        //match cmp {
        //    Cmp::Eq => {
        //        let x = self.builder.ins().bxor(lhs, rhs);
        //        self.builder.ins().bnot(x)
        //    }
        //    Cmp::Ne => self.builder.ins().bxor(lhs, rhs),
        //    Cmp::Lt => {
        //        let x = self.builder.ins().bxor(lhs, rhs);
        //        self.builder.ins().band_not(x, lhs)
        //    }
        //    Cmp::Le => {
        //        //There's probably a faster way
        //        let x = self.cmp_bool(Cmp::Eq, lhs, rhs);
        //        let y = self.cmp_bool(Cmp::Lt, lhs, rhs);
        //        self.builder.ins().bor(x, y)
        //    }
        //    Cmp::Gt => {
        //        let x = self.builder.ins().bxor(lhs, rhs);
        //        self.builder.ins().band_not(x, rhs)
        //    }
        //    Cmp::Ge => {
        //        //There's probably a faster way
        //        let x = self.cmp_bool(Cmp::Eq, lhs, rhs);
        //        let y = self.cmp_bool(Cmp::Eq, lhs, rhs);
        //        self.builder.ins().bor(x, y)
        //    }
        //}

        let lhs = self.builder.ins().bint(types::I64, lhs);
        let rhs = self.builder.ins().bint(types::I64, rhs);
        self.cmp_int(cmp, lhs, rhs)
    }

    fn translate_assign(
        &mut self,
        to_exprs: &[Expr],
        from_exprs: &[Expr],
    ) -> anyhow::Result<SValue> {
        // `def_var` is used to write the value of a variable. Note that
        // variables can have multiple definitions. Cranelift will
        // convert them into SSA form for itself automatically.

        //if there are the same number of expressions as there are names
        //eg: `a, b = b, a` then use the first output of each expression
        //But if there is not, use the output of the first expression:
        //eg: `a, b = func_that_outputs_2_floats(1.0)`
        if to_exprs.len() == from_exprs.len() {
            'expression: for (i, to_expr) in to_exprs.iter().enumerate() {
                let val = self.translate_expr(from_exprs.get(i).unwrap())?;
                match to_expr {
                    Expr::Binop(op, lhs, rhs) => {
                        let (struct_field_address, struct_field) =
                            self.translate_binop(*op, lhs, rhs, true)?;

                        if let Some(struct_field_def) = struct_field {
                            self.set_struct_field_at_address(
                                struct_field_address,
                                val,
                                struct_field_def,
                            )?
                        } else {
                            unreachable!()
                        }
                    }
                    Expr::Identifier(name) => {
                        let var = match self.env.variables.get(name) {
                            Some(v) => v,
                            None => anyhow::bail!("variable {} not found", name),
                        };

                        for arg in &self.func.params {
                            if *name == arg.name {
                                if let ExprType::Struct(stuct_name) = &arg.expr_type {
                                    //copy to struct that was passed in as parameter
                                    //TODO should there be specifc syntax for this?
                                    let return_address = self.builder.use_var(var.inner());
                                    copy_to_stack_slot(
                                        self.module.target_config(),
                                        &mut self.builder,
                                        self.env.struct_map[&stuct_name.to_string()].size,
                                        val.expect_struct(
                                            &stuct_name.to_string(),
                                            "translate_assign",
                                        )?,
                                        return_address,
                                        0,
                                    )?;
                                    continue 'expression;
                                }
                            }
                        }

                        if let SVariable::Struct(_var_name, struct_name, _var, return_struct) = var
                        {
                            if *return_struct {
                                //copy to struct, we don't want to return a reference
                                let return_address = self.builder.use_var(var.inner());
                                copy_to_stack_slot(
                                    self.module.target_config(),
                                    &mut self.builder,
                                    self.env.struct_map[struct_name].size,
                                    val.expect_struct(struct_name, "translate_assign")?,
                                    return_address,
                                    0,
                                )?;
                                continue 'expression;
                            }
                        }
                        self.builder
                            .def_var(var.inner(), val.inner("translate_assign")?);
                    }
                    Expr::ArrayAccess(name, idx_expr) => {
                        self.translate_array_set(name.to_string(), idx_expr, &val)?;
                    }
                    _ => {
                        //dbg!(to_expr);
                        todo!()
                    }
                }
            }
            Ok(SValue::Void)
        } else {
            match self.translate_expr(from_exprs.first().unwrap())? {
                SValue::Tuple(values) => {
                    for (i, to_expr) in to_exprs.iter().enumerate() {
                        if let Expr::Binop(_, _, _) = to_expr {
                            todo!()
                            //self.set_struct_field(to_expr, values[i].clone())?
                        } else if let Expr::Identifier(name) = to_expr {
                            let var = match self.env.variables.get(name) {
                                Some(v) => v,
                                None => anyhow::bail!("variable {} not found", name),
                            };
                            self.builder
                                .def_var(var.inner(), values[i].inner("translate_assign")?);
                        } else {
                            todo!()
                        }
                    }

                    Ok(SValue::Void)
                }
                SValue::Void
                | SValue::Unknown(_)
                | SValue::Bool(_)
                | SValue::F64(_)
                | SValue::I64(_)
                | SValue::Array(_, _)
                | SValue::Address(_)
                | SValue::Struct(_, _) => anyhow::bail!("operation not supported {:?}", from_exprs),
            }
        }
    }

    fn translate_array_get(
        &mut self,
        name: String,
        idx_expr: &Expr,
        get_address: bool,
    ) -> anyhow::Result<SValue> {
        //TODO crash if idx_val > ExprType::Array(_, len)
        let variable = match self.env.variables.get(&name) {
            Some(v) => v.clone(),
            None => anyhow::bail!("variable {} not found", name),
        };

        let array_ptr = self.builder.use_var(variable.inner());

        let idx_val = self.translate_expr(idx_expr).unwrap();
        let idx_val = match idx_val {
            SValue::I64(v) => v,
            _ => anyhow::bail!("only int supported for array access"),
        };

        let expr_type = &variable.expr_type()?;
        let base_type = match expr_type {
            ExprType::Array(ty, _len) => ty.cranelift_type(self.ptr_ty, true)?,
            ExprType::Void
            | ExprType::Bool
            | ExprType::F64
            | ExprType::I64
            | ExprType::Address
            | ExprType::Tuple(_)
            | ExprType::Struct(_) => anyhow::bail!("can't index type {}", &expr_type), //TODO get struct width
        };

        let idx_ptr = self.get_array_address_from_ptr(base_type.bytes(), array_ptr, idx_val);

        let val = self
            .builder
            .ins()
            .load(base_type, MemFlags::new(), idx_ptr, Offset32::new(0));

        if get_address {
            Ok(SValue::Address(val))
        } else {
            let sval = match &expr_type {
                ExprType::Array(ty, _len) => SValue::from(ty, val)?,
                ExprType::Void
                | ExprType::Bool
                | ExprType::F64
                | ExprType::I64
                | ExprType::Address
                | ExprType::Tuple(_)
                | ExprType::Struct(_) => anyhow::bail!("can't index type {}", &expr_type),
            };
            Ok(sval)
        }
    }

    fn get_array_address_from_ptr(
        &mut self,
        step_bytes: u32,
        array_ptr: Value,
        idx_val: Value,
    ) -> Value {
        let mult_n = self.builder.ins().iconst(self.ptr_ty, step_bytes as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        idx_ptr
    }

    fn translate_array_set(
        &mut self,
        name: String,
        idx_expr: &Expr,
        val: &SValue,
    ) -> anyhow::Result<SValue> {
        let variable = self.env.variables.get(&name).unwrap();

        let base_type = match variable {
            SVariable::Unknown(_, _) => self.ptr_ty,
            SVariable::Array(ty, _len) => ty.expr_type()?.cranelift_type(self.ptr_ty, true)?,
            SVariable::Address(_, _) => self.ptr_ty,
            SVariable::Struct(_, _, _, _)
            | SVariable::I64(_, _)
            | SVariable::F64(_, _)
            | SVariable::Bool(_, _) => anyhow::bail!("can't index type {}", &variable),
        };

        let array_ptr = self.builder.use_var(variable.inner());

        let idx_val = self.translate_expr(idx_expr)?;
        let idx_val = match idx_val {
            SValue::I64(v) => v,
            _ => anyhow::bail!("only int supported for array access"),
        };
        let mult_n = self
            .builder
            .ins()
            .iconst(self.ptr_ty, base_type.bytes() as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        self.builder.ins().store(
            MemFlags::new(),
            val.inner("array set")?,
            idx_ptr,
            Offset32::new(0),
        );
        Ok(SValue::Void)
    }

    fn translate_if_then(
        &mut self,
        condition: &Expr,
        then_body: &[Expr],
    ) -> anyhow::Result<SValue> {
        let b_condition_value = self.translate_expr(condition)?.expect_bool("if_then")?;

        let then_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // Test the if condition and conditionally branch.
        self.builder.ins().brz(b_condition_value, merge_block, &[]);
        // Fall through to then block.
        self.builder.ins().jump(then_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        for expr in then_body {
            self.translate_expr(expr).unwrap();
        }

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[]);
        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);
        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);
        Ok(SValue::Void)
    }

    fn translate_if_else(
        &mut self,
        condition: &Expr,
        then_body: &[Expr],
        else_body: &[Expr],
    ) -> anyhow::Result<SValue> {
        let b_condition_value = self.translate_expr(condition)?.expect_bool("if_else")?;

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the toy language have a return value.
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.
        let then_value = self.translate_expr(then_body.last().unwrap())?;
        let then_return = match then_value.clone() {
            SValue::Tuple(t) => {
                let mut vals = Vec::new();
                for v in &t {
                    self.builder
                        .append_block_param(merge_block, self.value_type(v.inner("then_return")?));
                    vals.push(v.clone().inner("then_return")?);
                }
                vals
            }
            SValue::Void => vec![],
            sv => {
                let v = sv.inner("then_return")?;
                self.builder
                    .append_block_param(merge_block, self.value_type(v));
                vec![v]
            }
        };

        let else_value = self.translate_expr(else_body.last().unwrap())?;
        let else_return = match else_value.clone() {
            SValue::Tuple(t) => {
                let mut vals = Vec::new();
                for v in &t {
                    vals.push(v.clone().inner("else_return")?);
                }
                vals
            }
            SValue::Void => vec![],
            SValue::Unknown(v) => vec![v],
            SValue::Bool(v) => vec![v],
            SValue::F64(v) => vec![v],
            SValue::I64(v) => vec![v],
            SValue::Array(sval, _len) => vec![sval.inner("translate_if_else")?],
            SValue::Address(v) => vec![v],
            SValue::Struct(_, v) => vec![v],
        };

        if then_value.to_string() != else_value.to_string() {
            anyhow::bail!(
                "if_else return types don't match {:?} {:?}",
                then_value,
                else_value
            )
        }

        // Test the if condition and conditionally branch.
        self.builder.ins().brz(b_condition_value, else_block, &[]);
        // Fall through to then block.
        self.builder.ins().jump(then_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &then_return);

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &else_return);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);

        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block);

        if phi.len() > 1 {
            //TODO the frontend doesn't have the syntax support for this yet
            if let SValue::Tuple(then_tuple) = then_value {
                let mut ret_tuple = Vec::new();
                for (phi_val, sval) in phi.iter().zip(then_tuple.iter()) {
                    ret_tuple.push(sval.replace_value(*phi_val)?)
                }
                Ok(SValue::Tuple(ret_tuple))
            } else {
                anyhow::bail!("expected tuple")
            }
        } else if phi.len() == 1 {
            then_value.replace_value(*phi.first().unwrap())
        } else {
            Ok(SValue::Void)
        }
    }

    fn translate_while_loop(
        &mut self,
        condition: &Expr,
        loop_body: &[Expr],
    ) -> anyhow::Result<SValue> {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);
        self.builder.switch_to_block(header_block);

        let b_condition_value = self.translate_expr(condition)?.expect_bool("while_loop")?;

        self.builder.ins().brz(b_condition_value, exit_block, &[]);
        self.builder.ins().jump(body_block, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        for expr in loop_body {
            self.translate_expr(expr)?;
        }
        self.builder.ins().jump(header_block, &[]);

        self.builder.switch_to_block(exit_block);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        Ok(SValue::Void)
    }

    fn translate_call(
        &mut self,
        fn_name: &str,
        args: &[Expr],
        impl_val: Option<SValue>,
    ) -> anyhow::Result<SValue> {
        let mut fn_name = fn_name.to_string();
        //println!(
        //    "{}:{} translate_call {} {:?} {:?}",
        //    file!(),
        //    line!(),
        //    &fn_name,
        //    &args,
        //    &impl_val
        //);
        let mut arg_values = Vec::new();
        if let Some(impl_sval) = impl_val {
            fn_name = format!("{}.{}", impl_sval.to_string(), fn_name);
            arg_values.push(impl_sval.inner("translate_call")?);
        }

        for expr in args.iter() {
            arg_values.push(self.translate_expr(expr)?.inner("translate_call")?)
        }

        //TODO refactor, we call is_struct_size_call too many times redundantly
        let stack_slot_return =
            if let Some(_) = sarus_std_lib::is_struct_size_call(&fn_name, &self.env.struct_map) {
                None
            } else {
                let returns = &self.env.funcs[&fn_name].returns;
                if returns.len() > 0 {
                    if let ExprType::Struct(name) = &returns[0].expr_type {
                        Some((
                            name.to_string(),
                            self.env.struct_map[&name.to_string()].size,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

        call_with_values(
            &fn_name,
            &arg_values,
            &self.env.funcs,
            &self.env.struct_map,
            &mut self.module,
            &mut self.builder,
            stack_slot_return,
        )
    }

    fn translate_global_data_addr(&mut self, ptr_ty: Type, name: &str) -> Value {
        let sym = self
            .module
            .declare_data(&name, Linkage::Export, true, false)
            .expect("problem declaring data object");
        let local_id = self
            .module
            .declare_data_in_func(sym, &mut self.builder.func);
        let global_val = self.builder.create_global_value(GlobalValueData::Load {
            base: local_id,
            offset: Offset32::new(0),
            global_type: ptr_ty,
            readonly: true,
        });

        //self.builder.ins().symbol_value(ptr_ty, local_id)
        self.builder.ins().global_value(ptr_ty, global_val)
    }

    fn value_type(&self, val: Value) -> Type {
        self.builder.func.dfg.value_type(val)
    }

    fn translate_new_struct(
        &mut self,
        name: &str,
        fields: &[StructAssignField],
    ) -> anyhow::Result<SValue> {
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            self.env.struct_map[name].size as u32,
        ));

        for field in fields.iter() {
            let dst_field_def = &self.env.struct_map[name].fields[&field.field_name];
            let sval = self.translate_expr(&field.expr)?;

            if let SValue::Struct(src_name, src_start_ptr) = sval {
                if src_name != dst_field_def.expr_type.to_string() {
                    anyhow::bail!(
                        "struct {} expected struct {} for field {} but got {} instead",
                        name,
                        dst_field_def.expr_type.to_string(),
                        dst_field_def.name,
                        src_name
                    )
                }

                let stack_slot_address = self.builder.ins().stack_addr(
                    self.ptr_ty,
                    stack_slot,
                    Offset32::new(dst_field_def.offset as i32),
                );

                self.builder.emit_small_memory_copy(
                    self.module.target_config(),
                    stack_slot_address,
                    src_start_ptr,
                    self.env.struct_map[&src_name].size as u64,
                    1,
                    1,
                    true,
                    MemFlags::new(),
                );
            } else {
                let stack_slot_address = self.builder.ins().stack_addr(
                    self.ptr_ty,
                    stack_slot,
                    Offset32::new(dst_field_def.offset as i32),
                );

                let val = if let SValue::Bool(val) = sval {
                    //struct bools are stored as I8
                    self.builder.ins().bint(types::I8, val)
                } else {
                    if sval.to_string() != dst_field_def.expr_type.to_string() {
                        anyhow::bail!(
                            "struct {} expected type {} for field {} but got {} instead",
                            name,
                            dst_field_def.expr_type.to_string(),
                            dst_field_def.name,
                            sval.to_string()
                        )
                    }

                    sval.inner("new_struct")?
                };

                self.builder.ins().store(
                    MemFlags::new(),
                    val,
                    stack_slot_address,
                    Offset32::new(0),
                );
            }
        }
        let stack_slot_address =
            self.builder
                .ins()
                .stack_addr(self.ptr_ty, stack_slot, Offset32::new(0));
        Ok(SValue::Struct(name.to_string(), stack_slot_address))
    }

    fn get_struct_field_location(
        &mut self,
        parts: Vec<String>,
    ) -> anyhow::Result<(StructField, Value, usize)> {
        //println!("get_struct_field_location {:?}", &parts);
        match &self.env.variables[&parts[0]] {
            SVariable::Struct(_var_name, struct_name, var, _return_struct) => {
                let mut parent_struct_field = &self.env.struct_map[struct_name].fields[&parts[1]];
                let base_struct_var_ptr = self.builder.use_var(*var);
                let mut struct_name = struct_name.clone();
                let mut offset = parent_struct_field.offset;
                if parts.len() > 2 {
                    offset = 0;
                    for i in 1..parts.len() {
                        if let ExprType::Struct(_name) = &parent_struct_field.expr_type {
                            parent_struct_field =
                                &self.env.struct_map[&struct_name].fields[&parts[i]];
                            offset += parent_struct_field.offset;
                            struct_name = parent_struct_field.expr_type.to_string().clone();
                        } else {
                            break;
                        }
                    }
                }
                Ok(((*parent_struct_field).clone(), base_struct_var_ptr, offset))
            }
            _ => unreachable!("validator should catch this"),
        }
    }

    fn get_struct_field_address(
        &mut self,
        parts: Vec<String>,
    ) -> anyhow::Result<(SValue, StructField)> {
        let (parent_struct_field_def, base_struct_var_ptr, offset) =
            self.get_struct_field_location(parts)?;
        //println!(
        //    "get_struct_field_address {:?} base_struct_var_ptr {} offset {}",
        //    &parent_struct_field_def, base_struct_var_ptr, offset
        //);

        let offset_v = self.builder.ins().iconst(self.ptr_ty, offset as i64);
        let address = self.builder.ins().iadd(base_struct_var_ptr, offset_v);
        if let ExprType::Struct(name) = &parent_struct_field_def.expr_type {
            //println!("get_struct_field_address ExprType::Struct {}", name);
            //If the struct field is a struct, return address of sub struct
            Ok((
                SValue::Struct(name.to_string(), address),
                parent_struct_field_def,
            ))
        } else {
            //println!("get_struct_field_address SValue::Address");
            //If the struct field is a struct, return address of value
            Ok((SValue::Address(address), parent_struct_field_def))
        }
    }

    fn get_struct_field(
        &mut self,
        field_address: SValue,
        parent_struct_field_def: &StructField,
    ) -> anyhow::Result<SValue> {
        //println!(
        //    "get_struct_field {:?} address {}",
        //    &parent_struct_field_def, field_address
        //);

        match field_address {
            SValue::Address(_) => {
                let mut val = self.builder.ins().load(
                    parent_struct_field_def
                        .expr_type
                        .cranelift_type(self.ptr_ty, true)?,
                    MemFlags::new(),
                    field_address.inner("get_struct_field")?,
                    Offset32::new(0),
                );
                if let ExprType::Bool = parent_struct_field_def.expr_type {
                    let t = self.builder.ins().iconst(types::I8, 1);
                    val = self.builder.ins().icmp(IntCC::Equal, t, val)
                }

                SValue::from(&parent_struct_field_def.expr_type, val)
            }
            //TODO Currently returning struct sub fields as reference.
            //Should we copy, or should there be syntax for copy?
            SValue::Struct(_, _) => Ok(field_address),
            _ => todo!(),
        }
    }

    fn set_struct_field_at_address(
        &mut self,
        address: SValue,
        set_value: SValue,
        struct_field_def: StructField,
    ) -> anyhow::Result<()> {
        if let ExprType::Struct(name) = &struct_field_def.expr_type {
            let src_ptr = set_value.expect_struct(name, "set_struct_field")?;
            self.builder.emit_small_memory_copy(
                self.module.target_config(),
                address.inner("set_struct_field_at_address")?,
                src_ptr,
                struct_field_def.size as u64,
                1,
                1,
                true,
                MemFlags::new(),
            );
            Ok(())
        } else {
            let val = if let SValue::Bool(val) = set_value {
                self.builder.ins().bint(types::I8, val)
            } else {
                set_value.inner("set_struct_field")?
            };
            //If the struct field is not a struct, set copy of value
            self.builder.ins().store(
                MemFlags::new(),
                val,
                address.inner("set_struct_field_at_address")?,
                Offset32::new(0),
            );
            Ok(())
        }
    }
}

fn create_and_copy_to_stack_slot(
    target_config: isa::TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    size: usize,
    src_ptr: Value,
    offset: usize,
) -> anyhow::Result<Value> {
    let stack_slot =
        builder.create_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size as u32));
    let stack_slot_address =
        builder
            .ins()
            .stack_addr(target_config.pointer_type(), stack_slot, Offset32::new(0));

    copy_to_stack_slot(
        target_config,
        builder,
        size,
        src_ptr,
        stack_slot_address,
        offset,
    )
}

fn copy_to_stack_slot(
    target_config: isa::TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    size: usize,
    src_ptr: Value,
    stack_slot_address: Value,
    offset: usize,
) -> anyhow::Result<Value> {
    let offset_v = builder
        .ins()
        .iconst(target_config.pointer_type(), offset as i64);
    let src_ptr_with_offset = builder.ins().iadd(src_ptr, offset_v);
    builder.emit_small_memory_copy(
        target_config,
        stack_slot_address,
        src_ptr_with_offset,
        size as u64,
        1,
        1,
        true,
        MemFlags::new(),
    );

    Ok(stack_slot_address)
}

fn call_with_values(
    name: &str,
    arg_values: &[Value],
    funcs: &HashMap<String, Function>,
    struct_map: &HashMap<String, StructDef>,
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    stack_slot_return: Option<(String, usize)>,
) -> anyhow::Result<SValue> {
    let name = &name.to_string();

    if let Some(struct_name) = sarus_std_lib::is_struct_size_call(name, struct_map) {
        return Ok(sarus_std_lib::translate_size_call(
            builder,
            struct_name,
            struct_map,
        ));
    }

    if !funcs.contains_key(name) {
        anyhow::bail!("function {} not found", name)
    }
    let func = funcs[name].clone();
    if func.params.len() != arg_values.len() {
        anyhow::bail!(
            "function call to {} has {} args, but function description has {}",
            name,
            arg_values.len(),
            func.params.len()
        )
    }

    if func.extern_func {
        if let Some(v) = sarus_std_lib::translate_std(
            module.target_config().pointer_type(),
            builder,
            name,
            arg_values,
        )? {
            return Ok(v);
        }
    }

    let mut sig = module.make_signature();

    let ptr_ty = module.target_config().pointer_type();

    let mut arg_values = Vec::from(arg_values); //in case we need to insert a val for the StructReturnSlot

    for val in arg_values.iter() {
        sig.params
            .push(AbiParam::new(builder.func.dfg.value_type(*val)));
    }

    let stack_slot_address = if let Some((name, size)) = &stack_slot_return {
        //setup StackSlotData to be used as a StructReturnSlot
        let stack_slot = builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::StructReturnSlot,
            *size as u32,
        ));
        //get stack address of StackSlotData
        let stack_slot_address = builder
            .ins()
            .stack_addr(ptr_ty, stack_slot, Offset32::new(0));
        arg_values.insert(0, stack_slot_address);
        sig.params
            .insert(0, AbiParam::special(ptr_ty, ArgumentPurpose::StructReturn));
        Some((name.clone(), stack_slot_address))
    } else {
        None
    };

    if stack_slot_return.is_none() {
        for ret_arg in &func.returns {
            sig.returns.push(AbiParam::new(
                ret_arg.expr_type.cranelift_type(ptr_ty, false)?,
            ));
        }
    }
    let callee = module
        .declare_function(&name, Linkage::Import, &sig)
        .expect("problem declaring function");
    let local_callee = module.declare_func_in_func(callee, &mut builder.func);
    let call = builder.ins().call(local_callee, &arg_values);
    let res = builder.inst_results(call);
    if let Some((name, stack_slot_address)) = stack_slot_address {
        Ok(SValue::Struct(name, stack_slot_address))
    } else if res.len() > 1 {
        Ok(SValue::Tuple(
            res.iter()
                .zip(func.returns.iter())
                .map(|(v, arg)| SValue::from(&arg.expr_type, *v).unwrap())
                .collect::<Vec<SValue>>(),
        ))
    } else if res.len() == 1 {
        Ok(SValue::from(
            &func.returns.first().unwrap().expr_type,
            *res.first().unwrap(),
        )?)
    } else {
        Ok(SValue::Void)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SVariable {
    Unknown(String, Variable),
    Bool(String, Variable),
    F64(String, Variable),
    I64(String, Variable),
    Array(Box<SVariable>, Option<usize>),
    Address(String, Variable),
    Struct(String, String, Variable, bool),
}

impl Display for SVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVariable::Unknown(name, _) => write!(f, "Unknown {}", name),
            SVariable::Bool(name, _) => write!(f, "Bool {}", name),
            SVariable::F64(name, _) => write!(f, "Float {}", name),
            SVariable::I64(name, _) => write!(f, "Int {}", name),
            SVariable::Array(svar, len) => {
                if let Some(len) = len {
                    write!(f, "&[{}; {}]", svar, len)
                } else {
                    write!(f, "&[{}]", svar)
                }
            }
            SVariable::Address(name, _) => write!(f, "Address {}", name),
            SVariable::Struct(name, structname, _, _return_struct) => {
                write!(f, "Struct {} {}", name, structname)
            }
        }
    }
}

impl SVariable {
    fn inner(&self) -> Variable {
        match self {
            SVariable::Unknown(_, v) => *v,
            SVariable::Bool(_, v) => *v,
            SVariable::F64(_, v) => *v,
            SVariable::I64(_, v) => *v,
            SVariable::Array(svar, _len) => svar.inner(),
            SVariable::Address(_, v) => *v,
            SVariable::Struct(_, _, v, _) => *v,
        }
    }
    pub fn expr_type(&self) -> anyhow::Result<ExprType> {
        Ok(match self {
            SVariable::Unknown(_, _) => anyhow::bail!("expression type is unknown"),
            SVariable::Bool(_, _) => ExprType::Bool,
            SVariable::F64(_, _) => ExprType::F64,
            SVariable::I64(_, _) => ExprType::I64,
            SVariable::Array(svar, len) => ExprType::Array(Box::new(svar.expr_type()?), *len),
            SVariable::Address(_, _) => ExprType::Address,
            SVariable::Struct(_, name, _, _) => ExprType::Struct(Box::new(name.to_string())),
        })
    }
    pub fn type_name(&self) -> anyhow::Result<String> {
        Ok(match self {
            SVariable::Unknown(_, _) => anyhow::bail!("Unknown has no type name"),
            SVariable::Bool(_, _) => "bool".to_string(),
            SVariable::F64(_, _) => "f64".to_string(),
            SVariable::I64(_, _) => "i64".to_string(),
            SVariable::Array(svar, len) => {
                if let Some(len) = len {
                    format!("&[{}; {}]", svar.type_name()?, len)
                } else {
                    format!("&[{}]", svar.type_name()?)
                }
            }
            SVariable::Address(_, _) => "&".to_string(),
            SVariable::Struct(_, name, _, _) => name.to_string(),
        })
    }
    fn expect_f64(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::F64(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Float {}", v, ctx),
        }
    }
    fn expect_i64(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::I64(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Int {}", v, ctx),
        }
    }
    fn expect_bool(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Bool(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Bool {}", v, ctx),
        }
    }
    fn expect_array(
        &self,
        expect_ty: ExprType,
        expect_len: Option<usize>,
        ctx: &str,
    ) -> anyhow::Result<Variable> {
        match self {
            SVariable::Array(svar, len) => {
                if expect_len != *len {
                    anyhow::bail!(
                        "incorrect length {:?} expected {:?} found {}",
                        expect_len,
                        len,
                        ctx
                    )
                }
                let var_ty = svar.expr_type()?;
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
    fn expect_address(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Address(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Address {}", v, ctx),
        }
    }
    fn expect_struct(&self, name: &str, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Struct(varname, sname, v, _return_struct) => {
                if sname == name {
                    return Ok(*v);
                } else {
                    anyhow::bail!(
                        "incorrect type {} expected Struct {} {}",
                        varname,
                        name,
                        ctx
                    )
                }
            }
            v => anyhow::bail!("incorrect type {} expected Struct {} {}", v, name, ctx),
        }
    }

    fn from(expr_type: &ExprType, name: String, var: Variable) -> anyhow::Result<SVariable> {
        Ok(match expr_type {
            ExprType::Bool => SVariable::Bool(name, var),
            ExprType::F64 => SVariable::F64(name, var),
            ExprType::I64 => SVariable::I64(name, var),
            ExprType::Array(ty, len) => {
                SVariable::Array(Box::new(SVariable::from(ty, name, var)?), *len)
            }
            ExprType::Address => SVariable::Address(name, var),
            ExprType::Tuple(_) => anyhow::bail!("use SVariable::from_tuple"),
            ExprType::Struct(name) => {
                SVariable::Struct(name.to_string(), name.to_string(), var, false)
                //last bool is return struct
            }
            ExprType::Void => anyhow::bail!("SVariable cannot be void"),
        })
    }
}

fn declare_variables(
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    params: &[Arg],
    returns: &[Arg],
    stmts: &[Expr],
    entry_block: Block,
    env: &mut Env,
) -> anyhow::Result<()> {
    let mut index = 0;

    let entry_block_is_offset = if returns.len() > 0 {
        if let ExprType::Struct(_struct_name) = &returns[0].expr_type {
            // When calling a function that will return a struct, Rust (or possibly anything using the C ABI),
            // will allocate the stack space needed for the struct that will be returned. This is allocated in
            // the callers frame, then the stack address is passed as a special argument to the first parameter
            // of the callee.
            // https://docs.wasmtime.dev/api/cranelift/prelude/enum.StackSlotKind.html#variant.StructReturnSlot
            // https://docs.wasmtime.dev/api/cranelift_codegen/ir/enum.ArgumentPurpose.html#variant.StructReturn

            let return_struct_arg = &returns[0];
            let return_struct_param_val = builder.block_params(entry_block)[0];
            let var = declare_variable(
                module,
                builder,
                &mut env.variables,
                &mut index,
                return_struct_arg,
                true,
            )?;
            if let Some(var) = var {
                builder.def_var(var.inner(), return_struct_param_val);
            }
            true
        } else {
            for arg in returns {
                declare_variable(module, builder, &mut env.variables, &mut index, arg, false)?;
            }
            false
        }
    } else {
        false
    };

    for (i, arg) in params.iter().enumerate() {
        let val = if entry_block_is_offset {
            builder.block_params(entry_block)[i + 1]
        } else {
            builder.block_params(entry_block)[i]
        };
        let var = declare_variable(module, builder, &mut env.variables, &mut index, arg, false)?;
        if let Some(var) = var {
            builder.def_var(var.inner(), val);
        }
    }

    for expr in stmts {
        declare_variables_in_stmt(
            module.target_config().pointer_type(),
            builder,
            &mut index,
            expr,
            env,
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
    env: &mut Env,
) -> anyhow::Result<()> {
    match *expr {
        Expr::Assign(ref to_exprs, ref from_exprs) => {
            if to_exprs.len() == from_exprs.len() {
                for (to_expr, _from_expr) in to_exprs.iter().zip(from_exprs.iter()) {
                    if let Expr::Identifier(name) = to_expr {
                        declare_variable_from_expr(ptr_type, expr, builder, index, &[name], env)?;
                    }
                }
            } else {
                let mut sto_exprs = Vec::new();
                for to_expr in to_exprs.iter() {
                    if let Expr::Identifier(name) = to_expr {
                        sto_exprs.push(name.as_str());
                    }
                }
                declare_variable_from_expr(ptr_type, expr, builder, index, &sto_exprs, env)?;
            }
        }
        Expr::IfElse(ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(ptr_type, builder, index, &stmt, env)?;
            }
            for stmt in else_body {
                declare_variables_in_stmt(ptr_type, builder, index, &stmt, env)?;
            }
        }
        Expr::WhileLoop(ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(ptr_type, builder, index, &stmt, env)?;
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
    env: &mut Env,
) -> anyhow::Result<()> {
    match expr {
        Expr::IfElse(_condition, then_body, _else_body) => {
            //TODO make sure then & else returns match
            declare_variable_from_expr(
                ptr_type,
                then_body.last().unwrap(),
                builder,
                index,
                names,
                env,
            )?;
        }
        expr => {
            let expr_type = ExprType::of(expr, &mut None, &env)?;
            declare_variable_from_type(ptr_type, &expr_type, builder, index, names, env)?;
        }
    };
    Ok(())
}

fn declare_variable_from_type(
    ptr_type: Type,
    expr_type: &ExprType,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    names: &[&str],
    env: &mut Env,
) -> anyhow::Result<()> {
    let name = *names.first().unwrap();
    if name.contains(".") {
        return Ok(());
    }
    match expr_type {
        ExprType::Void => anyhow::bail!("can't assign void type to {}", name),
        ExprType::Bool => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables
                    .insert(name.into(), SVariable::Bool(name.into(), var));
                builder.declare_var(var, types::B1);
                *index += 1;
            }
        }
        ExprType::F64 => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables
                    .insert(name.into(), SVariable::F64(name.into(), var));
                builder.declare_var(var, types::F64);
                *index += 1;
            }
        }
        ExprType::I64 => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables
                    .insert(name.into(), SVariable::I64(name.into(), var));
                builder.declare_var(var, types::I64);
                *index += 1;
            }
        }
        ExprType::Array(ty, len) => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables.insert(
                    name.into(),
                    SVariable::Array(Box::new(SVariable::from(ty, name.to_string(), var)?), *len),
                ); //name.into(), var));
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Address => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables
                    .insert(name.into(), SVariable::Address(name.into(), var));
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Tuple(expr_types) => {
            if expr_types.len() == 1 {
                //Single nested tuple
                if let ExprType::Tuple(expr_types) = expr_types.first().unwrap() {
                    for (expr_type, sname) in expr_types.iter().zip(names.iter()) {
                        declare_variable_from_type(
                            ptr_type,
                            expr_type,
                            builder,
                            index,
                            &[sname],
                            env,
                        )?
                    }
                    return Ok(());
                }
            }
            for (expr_type, sname) in expr_types.iter().zip(names.iter()) {
                declare_variable_from_type(ptr_type, expr_type, builder, index, &[sname], env)?
            }
        }
        ExprType::Struct(structname) => {
            if !env.variables.contains_key(name) {
                let var = Variable::new(*index);
                env.variables.insert(
                    name.into(),
                    SVariable::Struct(name.into(), structname.to_string(), var, false),
                );
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
    }
    Ok(())
}

fn declare_variable(
    module: &mut dyn Module,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    arg: &Arg,
    return_struct: bool,
) -> anyhow::Result<Option<SVariable>> {
    let ptr_ty = module.target_config().pointer_type();
    if !variables.contains_key(&arg.name) {
        let (var, ty) = match &arg.expr_type {
            ExprType::F64 => (
                SVariable::F64(arg.name.clone(), Variable::new(*index)),
                types::F64,
            ),
            ExprType::I64 => (
                SVariable::I64(arg.name.clone(), Variable::new(*index)),
                types::I64,
            ),
            ExprType::Array(ty, len) => (
                SVariable::Array(
                    Box::new(SVariable::from(
                        ty,
                        arg.name.clone(),
                        Variable::new(*index),
                    )?),
                    *len,
                ),
                ptr_ty,
            ),
            ExprType::Address => (
                SVariable::Address(arg.name.clone(), Variable::new(*index)),
                ptr_ty,
            ),
            ExprType::Void => return Ok(None),
            ExprType::Bool => (
                SVariable::Bool(arg.name.clone(), Variable::new(*index)),
                types::B1,
            ),
            ExprType::Tuple(_) => return Ok(None), //anyhow::bail!("single variable tuple not supported"),
            ExprType::Struct(structname) => (
                SVariable::Struct(
                    arg.name.clone(),
                    structname.to_string(),
                    Variable::new(*index),
                    return_struct,
                ),
                ptr_ty,
            ),
        };
        variables.insert(arg.name.clone(), var.clone());
        builder.declare_var(var.inner(), ty);
        *index += 1;
        Ok(Some(var))
    } else {
        Ok(None)
    }
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub size: usize,
    pub name: String,
    pub fields: HashMap<String, StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub offset: usize,
    pub size: usize,
    pub name: String,
    pub expr_type: ExprType,
}

fn create_struct_map(
    prog: &Vec<Declaration>,
    ptr_type: types::Type,
) -> anyhow::Result<HashMap<String, StructDef>> {
    let mut in_structs = HashMap::new();
    for decl in prog {
        if let Declaration::Struct(s) = decl {
            in_structs.insert(s.name.to_string(), s);
        }
    }
    let structs_order = order_structs(&in_structs)?;

    let mut structs: HashMap<String, StructDef> = HashMap::new();

    for struct_name in structs_order {
        let mut largest_field_in_struct = 0;
        let fields_def = &in_structs[&struct_name].fields;
        let mut fields = HashMap::new();
        let mut fields_v = Vec::new();
        let mut struct_size = 0usize;
        for (i, field) in fields_def.iter().enumerate() {
            let (field_size, is_struct) =
                get_field_size(&field.expr_type, &structs, ptr_type, false)?;
            let new_field = StructField {
                offset: struct_size,
                size: field_size,
                name: field.name.to_string(),
                expr_type: field.expr_type.clone(),
            };
            fields.insert(field.name.to_string(), new_field.clone());
            fields_v.push(new_field);

            struct_size += field_size;
            //print!(
            //    "struct_size {} \t {} \t field_size {} \t",
            //    struct_size, field.name, field_size
            //);

            if i < fields_def.len() - 1 {
                //repr(C) alignment see memoffset crate
                let mut field_size: usize;
                let (next_field_size, _is_struct) =
                    get_field_size(&fields_def[i + 1].expr_type, &structs, ptr_type, true)?;
                field_size = next_field_size;
                if is_struct {
                    let (this_field_size, _is_struct) =
                        get_field_size(&fields_def[i].expr_type, &structs, ptr_type, true)?;
                    field_size = field_size.max(this_field_size)
                }
                largest_field_in_struct = largest_field_in_struct.max(field_size);
                let m = struct_size % field_size;
                let padding = if m > 0 { field_size - m } else { m };
                struct_size += padding;
                //print!(
                //    " padding {} \t --- (next_field_size {}) ",
                //    padding, next_field_size
                //);
            }
            //println!("");
        }

        //Padding at end of struct
        if largest_field_in_struct > 0 {
            let m = struct_size % largest_field_in_struct;
            let padding = if m > 0 {
                largest_field_in_struct - m
            } else {
                m
            };
            struct_size += padding;
        }

        structs.insert(
            struct_name.to_string(),
            StructDef {
                size: struct_size,
                name: struct_name.to_string(),
                fields,
            },
        );
    }

    Ok(structs)
}

fn get_field_size(
    expr_type: &ExprType,
    structs: &HashMap<String, StructDef>,
    ptr_type: types::Type,
    max_base_field: bool,
) -> anyhow::Result<(usize, bool)> {
    Ok(match expr_type {
        ExprType::Struct(name) => {
            if max_base_field {
                (
                    get_largest_field_size(0, &expr_type, structs, ptr_type)?,
                    true,
                )
            } else {
                (structs[&name.to_string()].size, true)
            }
        }
        _ => (
            (expr_type.cranelift_type(ptr_type, true)?.bytes() as usize),
            false,
        ),
    })
}

fn get_largest_field_size(
    largest: usize,
    expr_type: &ExprType,
    structs: &HashMap<String, StructDef>,
    ptr_type: types::Type,
) -> anyhow::Result<usize> {
    let mut largest = largest;
    match expr_type {
        t => match t {
            ExprType::Struct(name) => {
                for (_name, field) in &structs[&name.to_string()].fields {
                    let size =
                        get_largest_field_size(largest, &field.expr_type, structs, ptr_type)?;
                    if size > largest {
                        largest = size;
                    }
                }
            }
            _ => {
                let size = t.cranelift_type(ptr_type, true)?.bytes() as usize;
                if size > largest {
                    largest = size;
                }
            }
        },
    }
    Ok(largest)
}

fn order_structs(in_structs: &HashMap<String, &Struct>) -> anyhow::Result<Vec<String>> {
    let mut structs_order = Vec::new();
    let mut last_structs_len = 0usize;
    while structs_order.len() < in_structs.len() {
        for (name, struc) in in_structs {
            let mut can_insert = true;
            for field in &struc.fields {
                match &field.expr_type {
                    ExprType::Void
                    | ExprType::Bool
                    | ExprType::F64
                    | ExprType::I64
                    | ExprType::Array(_, _)
                    | ExprType::Address
                    | ExprType::Tuple(_) => continue,
                    ExprType::Struct(field_struct_name) => {
                        if !in_structs.contains_key(&field_struct_name.to_string()) {
                            anyhow::bail!(
                                "Can't find Struct {} referenced in Struct {} field {}",
                                field_struct_name,
                                struc.name,
                                field.name
                            )
                        }
                        if structs_order.contains(&field_struct_name.to_string()) {
                            continue;
                        } else {
                            can_insert = false;
                        }
                    }
                }
            }
            if can_insert && !structs_order.contains(&name.to_string()) {
                structs_order.push(name.to_string());
            }
        }
        if structs_order.len() > last_structs_len {
            last_structs_len = structs_order.len()
        } else {
            anyhow::bail!("Structs references resulting in loop unsupported")
        }
    }

    Ok(structs_order)
}
