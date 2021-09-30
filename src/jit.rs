use crate::frontend::*;
use crate::sarus_std_lib;
use crate::validator::validate_program;
use crate::validator::ExprType;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
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

    // Translate from toy-language AST nodes into Cranelift IR.
    fn codegen(
        &mut self,
        func: &Function,
        funcs: HashMap<String, Function>,
        env: &[Declaration],
        struct_map: &HashMap<String, StructDef>,
    ) -> anyhow::Result<()> {
        let float = types::F64; //self.module.target_config().pointer_type();

        for p in &func.params {
            self.ctx.func.signature.params.push({
                match &p.expr_type {
                    Some(t) => match t {
                        ExprType::F64 => AbiParam::new(types::F64),
                        ExprType::I64 => AbiParam::new(types::I64),
                        ExprType::UnboundedArrayF64 => {
                            AbiParam::new(self.module.target_config().pointer_type())
                        }
                        ExprType::UnboundedArrayI64 => {
                            AbiParam::new(self.module.target_config().pointer_type())
                        }
                        ExprType::Address => {
                            AbiParam::new(self.module.target_config().pointer_type())
                        }
                        ExprType::Void => continue,
                        ExprType::Bool => AbiParam::new(types::B1),
                        ExprType::Struct(_) => {
                            AbiParam::new(self.module.target_config().pointer_type())
                        }
                        ExprType::Tuple(_) => anyhow::bail!("Tuple as parameter not supported"),
                    },
                    None => AbiParam::new(float),
                }
            });
        }

        for ret_arg in &func.returns {
            self.ctx.func.signature.returns.push(AbiParam::new(
                ret_arg
                    .expr_type
                    .as_ref()
                    .unwrap_or(&ExprType::F64)
                    .cranelift_type(self.module.target_config().pointer_type())?,
            ));
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

        println!("declare_variables {}", func.name);
        let variables = declare_variables(
            float,
            &mut builder,
            &mut self.module,
            &func.params,
            &func.returns,
            &func.body,
            entry_block,
            env,
            &funcs,
            &constant_vars,
            &struct_map,
        )?;

        //Keep function vars around for later debug/print
        self.variables
            .insert(func.name.to_string(), variables.clone());

        println!("validate_program {}", func.name);

        //Check every statement, this can catch funcs with no assignment, etc...
        validate_program(
            &func.body,
            env,
            &funcs,
            &variables,
            &constant_vars,
            &struct_map,
        )?;

        println!("FunctionTranslator {}", func.name);

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            variables,
            constant_vars,
            env,
            funcs,
            struct_map,
            module: &mut self.module,
        };
        for expr in &func.body {
            trans.translate_expr(expr)?;
        }

        // Set up the return variable of the function. Above, we declared a
        // variable to hold the return value. Here, we just do a use of that
        // variable.
        let mut return_values = Vec::new();
        for ret in func.returns.iter() {
            let return_variable = trans.variables.get(&ret.name).unwrap();
            let v = match &ret.expr_type {
                Some(t) => match t {
                    ExprType::F64 => trans
                        .builder
                        .use_var(return_variable.expect_f64("return_variable")?),
                    ExprType::I64 => trans
                        .builder
                        .use_var(return_variable.expect_i64("return_variable")?),
                    ExprType::UnboundedArrayF64 => trans
                        .builder
                        .use_var(return_variable.expect_unbounded_array_f64("return_variable")?),
                    ExprType::UnboundedArrayI64 => trans
                        .builder
                        .use_var(return_variable.expect_unbounded_array_f64("return_variable")?),
                    ExprType::Address => trans
                        .builder
                        .use_var(return_variable.expect_address("return_variable")?),
                    ExprType::Void => continue,
                    ExprType::Bool => trans
                        .builder
                        .use_var(return_variable.expect_bool("return_variable")?),
                    ExprType::Tuple(_) => anyhow::bail!("tuple not supported in return"),
                    ExprType::Struct(_) => anyhow::bail!("returning structs not supported yet"), //TODO support this
                },
                None => trans
                    .builder
                    .use_var(return_variable.expect_f64("return_variable")?),
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
    UnboundedArrayF64(Value),
    UnboundedArrayI64(Value),
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
            SValue::UnboundedArrayF64(_) => write!(f, "&[f64]"),
            SValue::UnboundedArrayI64(_) => write!(f, "&[i64]"),
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
            ExprType::UnboundedArrayF64 => SValue::UnboundedArrayF64(value),
            ExprType::UnboundedArrayI64 => SValue::UnboundedArrayI64(value),
            ExprType::Address => SValue::Address(value),
            ExprType::Tuple(_) => anyhow::bail!("use SValue::from_tuple"),
            ExprType::Struct(name) => SValue::Struct(name.to_string(), value),
        })
    }

    fn inner(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Unknown(v) => Ok(*v),
            SValue::Bool(v) => Ok(*v),
            SValue::F64(v) => Ok(*v),
            SValue::I64(v) => Ok(*v),
            SValue::UnboundedArrayF64(v) => Ok(*v),
            SValue::UnboundedArrayI64(v) => Ok(*v),
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
    fn expect_unbounded_array_f64(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::UnboundedArrayF64(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected UnboundedArrayF64 {}", v, ctx),
        }
    }
    fn expect_unbounded_array_i64(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::UnboundedArrayI64(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected UnboundedArrayI64 {}", v, ctx),
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

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, SVariable>,
    funcs: HashMap<String, Function>,
    struct_map: &'a HashMap<String, StructDef>,
    module: &'a mut JITModule,
    constant_vars: HashMap<String, f64>,
    env: &'a [Declaration],
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &Expr) -> anyhow::Result<SValue> {
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
            Expr::Binop(op, lhs, rhs) => self.translate_binop(*op, lhs, rhs),
            Expr::Unaryop(op, lhs) => self.translate_unaryop(*op, lhs),
            Expr::Compare(cmp, lhs, rhs) => self.translate_cmp(*cmp, lhs, rhs),
            Expr::Call(name, args) => self.translate_call(name, args, None),
            Expr::GlobalDataAddr(name) => Ok(SValue::UnboundedArrayF64(
                self.translate_global_data_addr(self.module.target_config().pointer_type(), name),
            )),
            Expr::Identifier(name) => {
                match self.variables.get(name) {
                    Some(var) => Ok(match var {
                        SVariable::Unknown(_, v) => SValue::Unknown(self.builder.use_var(*v)),
                        SVariable::Bool(_, v) => SValue::Bool(self.builder.use_var(*v)),
                        SVariable::F64(_, v) => SValue::F64(self.builder.use_var(*v)),
                        SVariable::I64(_, v) => SValue::I64(self.builder.use_var(*v)),
                        SVariable::Address(_, v) => SValue::Address(self.builder.use_var(*v)),
                        SVariable::UnboundedArrayF64(_, v) => {
                            SValue::UnboundedArrayF64(self.builder.use_var(*v))
                        }
                        SVariable::UnboundedArrayI64(_, v) => {
                            SValue::UnboundedArrayF64(self.builder.use_var(*v))
                        }
                        SVariable::Struct(_varname, structname, v) => {
                            SValue::Struct(structname.to_string(), self.builder.use_var(*v))
                        }
                    }),
                    None => Ok(SValue::F64(
                        //TODO Don't assume this is a float (this is used for math const)
                        self.translate_global_data_addr(types::F64, name),
                    )), //Try to load global
                }
            }
            Expr::Assign(names, expr) => self.translate_assign(names, expr),
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
            Expr::ArrayGet(name, idx_expr) => self.translate_array_get(name.to_string(), idx_expr),
            Expr::ArraySet(name, idx_expr, expr) => {
                self.translate_array_set(name.to_string(), idx_expr, expr)
            }
        }
    }

    fn translate_string(&mut self, literal: &String) -> anyhow::Result<SValue> {
        let cstr = CString::new(literal.replace("\\n", "\n").to_string()).unwrap();
        let bytes = cstr.to_bytes_with_nul();
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            types::I8.bytes() * bytes.len() as u32,
        ));
        let stack_slot_address = self.builder.ins().stack_addr(
            self.module.target_config().pointer_type(),
            stack_slot,
            Offset32::new(0),
        );
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

    fn translate_binop(&mut self, op: Binop, lhs: &Expr, rhs: &Expr) -> anyhow::Result<SValue> {
        if let Binop::DotAccess = op {
            let mut lval = None;
            let _t = ExprType::of(
                &Expr::Binop(
                    Binop::DotAccess,
                    Box::new(lhs.clone()),
                    Box::new(rhs.clone()),
                ),
                &mut lval,
                self.env,
                &self.funcs,
                &self.variables,
                &self.constant_vars,
                &self.struct_map,
            )?;
            //TODO Refactor, test with array access, make more struct access tests

            if let Some(lval) = lval {
                let mut parts = Vec::new();
                let mut last_expr = None;
                let len = lval.expr.len();
                for (i, expr) in lval.expr.iter().enumerate() {
                    match expr {
                        Expr::Parentheses(e) => {
                            last_expr = Some(self.translate_expr(e)?);
                        }
                        Expr::Identifier(s) | Expr::LiteralString(s) => {
                            parts.push(s.as_str());
                            if i == len - 1 {
                                //if this is the last one
                                last_expr = Some(self.get_struct_field(parts.clone())?);
                            } else if i < len - 1 {
                                //if this is not the last one
                                if let Expr::Call(..) = lval.expr[i + 1] {
                                    //and the next is a call
                                    last_expr = if parts.len() == 1 {
                                        Some(self.translate_expr(&lval.expr[i])?)
                                    } else {
                                        Some(self.get_struct_field(parts.clone())?)
                                    }
                                }
                            }
                            //TODO this can't be used after other Expr types
                        }
                        Expr::Call(name, args) => {
                            last_expr = Some(self.translate_call(name, &args, last_expr)?);
                        }
                        a => {
                            dbg!(a);
                            panic!("non identifier/call found")
                        }
                    }
                }
                if let Some(last_expr) = last_expr {
                    return Ok(last_expr);
                } else {
                    panic!("no last_expr found")
                }
            } else {
                panic!("no lval found")
            }
        }

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
            | SValue::UnboundedArrayF64(_)
            | SValue::UnboundedArrayI64(_)
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
            | SValue::UnboundedArrayF64(_)
            | SValue::UnboundedArrayI64(_)
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
            | SValue::UnboundedArrayF64(_)
            | SValue::UnboundedArrayI64(_)
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

    fn translate_assign(&mut self, names: &[String], expr: &[Expr]) -> anyhow::Result<SValue> {
        // `def_var` is used to write the value of a variable. Note that
        // variables can have multiple definitions. Cranelift will
        // convert them into SSA form for itself automatically.

        //if there are the same number of expressions as there are names
        //eg: `a, b = b, a` then use the first output of each expression
        //But if there is not, use the output of the first expression:
        //eg: `a, b = func_that_outputs_2_floats(1.0)`
        if names.len() == expr.len() {
            for (i, name) in names.iter().enumerate() {
                let val = self.translate_expr(expr.get(i).unwrap())?;
                if name.contains(".") {
                    let parts = name.split(".").collect::<Vec<&str>>();
                    //TODO if val is a struct then does translate_expr above make an unnecessary copy?
                    self.set_struct_field(parts, val)?
                } else {
                    let var = match self.variables.get(name) {
                        Some(v) => v,
                        None => anyhow::bail!("variable {} not found", name),
                    };
                    self.builder
                        .def_var(var.inner(), val.inner("translate_assign")?);
                }
            }
            Ok(SValue::Void)
        } else {
            match self.translate_expr(expr.first().unwrap())? {
                SValue::Tuple(values) => {
                    for (i, name) in names.iter().enumerate() {
                        if name.contains(".") {
                            let parts = name.split(".").collect::<Vec<&str>>();
                            self.set_struct_field(parts, values[i].clone())?
                        } else {
                            let var = match self.variables.get(name) {
                                Some(v) => v,
                                None => anyhow::bail!("variable {} not found", name),
                            };
                            self.builder
                                .def_var(var.inner(), values[i].inner("translate_assign")?);
                        }
                    }

                    Ok(SValue::Void)
                }
                SValue::Void
                | SValue::Unknown(_)
                | SValue::Bool(_)
                | SValue::F64(_)
                | SValue::I64(_)
                | SValue::UnboundedArrayF64(_)
                | SValue::UnboundedArrayI64(_)
                | SValue::Address(_)
                | SValue::Struct(_, _) => anyhow::bail!("operation not supported {:?}", expr),
            }
        }
    }

    fn translate_array_get(&mut self, name: String, idx_expr: &Expr) -> anyhow::Result<SValue> {
        let ptr_ty = self.module.target_config().pointer_type();

        let variable = match self.variables.get(&name) {
            Some(v) => v,
            None => anyhow::bail!("variable {} not found", name),
        };
        let array_ptr = self.builder.use_var(variable.inner());

        let idx_val = self.translate_expr(idx_expr).unwrap();
        let idx_val = match idx_val {
            SValue::F64(v) => self.builder.ins().fcvt_to_uint(ptr_ty, v),
            SValue::I64(v) => v,
            _ => anyhow::bail!("only int and float supported for array access"),
        };
        let mult_n = self.builder.ins().iconst(ptr_ty, types::F64.bytes() as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        let val = self
            .builder
            .ins()
            .load(types::F64, MemFlags::new(), idx_ptr, Offset32::new(0));
        Ok(SValue::F64(val)) //todo, don't assume this is a float
    }

    fn translate_array_set(
        &mut self,
        name: String,
        idx_expr: &Expr,
        expr: &Expr,
    ) -> anyhow::Result<SValue> {
        let ptr_ty = self.module.target_config().pointer_type();

        let new_val = self.translate_expr(expr)?;

        let variable = self
            .variables
            .get(&name)
            .unwrap()
            .expect_unbounded_array_f64("array_set")?;

        let array_ptr = self.builder.use_var(variable);

        let idx_val = self.translate_expr(idx_expr)?;
        let idx_val = match idx_val {
            SValue::F64(v) => self.builder.ins().fcvt_to_uint(ptr_ty, v),
            SValue::I64(v) => v,
            _ => anyhow::bail!("only int and float supported for array access"),
        };
        let mult_n = self.builder.ins().iconst(ptr_ty, types::F64.bytes() as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        self.builder.ins().store(
            MemFlags::new(),
            new_val.inner("array set")?,
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
            SValue::UnboundedArrayF64(v) => vec![v],
            SValue::UnboundedArrayI64(v) => vec![v],
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
            // TODO don't assume these are floats
            Ok(SValue::Tuple(
                phi.iter().map(|v| SValue::F64(*v)).collect::<Vec<SValue>>(),
            ))
        } else if phi.len() == 1 {
            match then_value {
                SValue::Void => Ok(SValue::Void),
                SValue::Unknown(_) => Ok(SValue::Unknown(*phi.first().unwrap())),
                SValue::Bool(_) => Ok(SValue::Bool(*phi.first().unwrap())),
                SValue::F64(_) => Ok(SValue::F64(*phi.first().unwrap())),
                SValue::I64(_) => Ok(SValue::I64(*phi.first().unwrap())),
                SValue::Tuple(_) => anyhow::bail!("not supported"),
                SValue::UnboundedArrayF64(_) => {
                    Ok(SValue::UnboundedArrayF64(*phi.first().unwrap()))
                }
                SValue::UnboundedArrayI64(_) => {
                    Ok(SValue::UnboundedArrayI64(*phi.first().unwrap()))
                }
                SValue::Address(_) => Ok(SValue::Address(*phi.first().unwrap())),
                SValue::Struct(name, _) => Ok(SValue::Struct(name, *phi.first().unwrap())),
            }
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
        let mut arg_values = Vec::new();
        if let Some(impl_func) = impl_val {
            fn_name = format!("{}.{}", impl_func.to_string(), fn_name);
            arg_values.push(impl_func.inner("translate_call")?);
        }

        for expr in args.iter() {
            arg_values.push(self.translate_expr(expr)?.inner("translate_call")?)
        }

        call_with_values(
            &fn_name,
            &arg_values,
            &mut self.funcs,
            &mut self.module,
            &mut self.builder,
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
            self.struct_map[name].size,
        ));

        for field in fields.iter() {
            let dst_field_def = &self.struct_map[name].fields[&field.field_name];
            let sval = self.translate_expr(&field.expr)?;

            if let SValue::Struct(src_name, src_start_ptr) = sval {
                let stack_slot_address = self.builder.ins().stack_addr(
                    self.module.target_config().pointer_type(),
                    stack_slot,
                    Offset32::new(dst_field_def.offset as i32), //TODO this shouldn't need to be *2
                );

                self.builder.emit_small_memory_copy(
                    self.module.target_config(),
                    stack_slot_address,
                    src_start_ptr,
                    self.struct_map[&src_name].size as u64,
                    1,
                    1,
                    true,
                    MemFlags::new(),
                );
            } else {
                let stack_slot_address = self.builder.ins().stack_addr(
                    self.module.target_config().pointer_type(),
                    stack_slot,
                    Offset32::new(dst_field_def.offset as i32),
                );

                self.builder.ins().store(
                    MemFlags::new(),
                    sval.inner("new_struct")?,
                    stack_slot_address,
                    Offset32::new(0),
                );
            }
        }
        let stack_slot_address = self.builder.ins().stack_addr(
            self.module.target_config().pointer_type(),
            stack_slot,
            Offset32::new(0),
        );
        Ok(SValue::Struct(name.to_string(), stack_slot_address))
    }

    fn get_struct_field_location(
        &mut self,
        parts: Vec<&str>,
    ) -> anyhow::Result<(StructField, Value, u32)> {
        match &self.variables[parts[0]] {
            SVariable::Struct(_var_name, struct_name, var) => {
                let mut parent_struct_field = &self.struct_map[struct_name].fields[parts[1]];
                let base_struct_var_ptr = self.builder.use_var(*var);
                let mut struct_name = struct_name;
                let mut offset = parent_struct_field.offset;
                if parts.len() > 2 {
                    offset = 0;
                    for i in 1..parts.len() {
                        if let ExprType::Struct(name) = &parent_struct_field.expr_type {
                            parent_struct_field = &self.struct_map[struct_name].fields[parts[i]];
                            offset += parent_struct_field.offset;
                            struct_name = name;
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

    fn get_struct_field(&mut self, parts: Vec<&str>) -> anyhow::Result<SValue> {
        let (parent_struct_field_def, base_struct_var_ptr, offset) =
            self.get_struct_field_location(parts)?;

        if let ExprType::Struct(name) = &parent_struct_field_def.expr_type {
            //If the struct field is a struct, return copy of struct
            let stack_slot_address = create_and_copy_to_stack_slot(
                self.module.target_config(),
                &mut self.builder,
                parent_struct_field_def.size,
                base_struct_var_ptr,
                offset,
            )?;
            Ok(SValue::Struct(name.to_string(), stack_slot_address))
        } else {
            //If the struct field is not a struct, return copy of value
            let val = self.builder.ins().load(
                parent_struct_field_def
                    .expr_type
                    .cranelift_type(self.module.target_config().pointer_type())?,
                MemFlags::new(),
                base_struct_var_ptr,
                Offset32::new(offset as i32),
            );
            return SValue::from(&parent_struct_field_def.expr_type, val);
        }
    }

    fn set_struct_field(&mut self, parts: Vec<&str>, set_value: SValue) -> anyhow::Result<()> {
        let (parent_struct_field_def, base_struct_var_ptr, offset) =
            self.get_struct_field_location(parts)?;
        if let ExprType::Struct(name) = &parent_struct_field_def.expr_type {
            let src_ptr = set_value.expect_struct(name, "set_struct_field")?;
            let offset_v = self
                .builder
                .ins()
                .iconst(self.module.target_config().pointer_type(), offset as i64);
            let dst_ptr_with_offset = self.builder.ins().iadd(base_struct_var_ptr, offset_v);
            self.builder.emit_small_memory_copy(
                self.module.target_config(),
                dst_ptr_with_offset,
                src_ptr,
                parent_struct_field_def.size as u64,
                1,
                1,
                true,
                MemFlags::new(),
            );
            Ok(())
        } else {
            //If the struct field is not a struct, return copy of value
            self.builder.ins().store(
                MemFlags::new(),
                set_value.inner("set_struct_field")?,
                base_struct_var_ptr,
                Offset32::new(offset as i32),
            );
            Ok(())
        }
    }
}

fn create_and_copy_to_stack_slot(
    target_config: isa::TargetFrontendConfig,
    builder: &mut FunctionBuilder,
    size: u32,
    src_ptr: Value,
    offset: u32,
) -> anyhow::Result<Value> {
    let stack_slot =
        builder.create_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size));
    let stack_slot_address =
        builder
            .ins()
            .stack_addr(target_config.pointer_type(), stack_slot, Offset32::new(0));

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
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
) -> anyhow::Result<SValue> {
    let name = &name.to_string();

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

    for val in arg_values.iter() {
        sig.params
            .push(AbiParam::new(builder.func.dfg.value_type(*val)));
    }

    for ret_arg in &func.returns {
        sig.returns.push(AbiParam::new(
            ret_arg
                .expr_type
                .as_ref()
                .unwrap_or(&ExprType::F64)
                .cranelift_type(ptr_ty)?,
        ));
    }
    let callee = module
        .declare_function(&name, Linkage::Import, &sig)
        .expect("problem declaring function");
    let local_callee = module.declare_func_in_func(callee, &mut builder.func);
    let call = builder.ins().call(local_callee, &arg_values);
    let res = builder.inst_results(call);
    if res.len() > 1 {
        Ok(SValue::Tuple(
            res.iter()
                .zip(func.returns.iter())
                .map(|(v, arg)| {
                    SValue::from(arg.expr_type.as_ref().unwrap_or(&ExprType::F64), *v).unwrap()
                })
                .collect::<Vec<SValue>>(),
        ))
    } else if res.len() == 1 {
        Ok(SValue::from(
            &func
                .returns
                .first()
                .unwrap()
                .expr_type
                .as_ref()
                .unwrap_or(&ExprType::F64),
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
    UnboundedArrayF64(String, Variable),
    UnboundedArrayI64(String, Variable),
    Address(String, Variable),
    Struct(String, String, Variable),
}

impl Display for SVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVariable::Unknown(name, _) => write!(f, "Unknown {}", name),
            SVariable::Bool(name, _) => write!(f, "Bool {}", name),
            SVariable::F64(name, _) => write!(f, "Float {}", name),
            SVariable::I64(name, _) => write!(f, "Int {}", name),
            SVariable::UnboundedArrayF64(name, _) => write!(f, "UnboundedArrayF64 {}", name),
            SVariable::UnboundedArrayI64(name, _) => write!(f, "UnboundedArrayI64 {}", name),
            SVariable::Address(name, _) => write!(f, "Address {}", name),
            SVariable::Struct(name, structname, _) => write!(f, "Struct {} {}", name, structname),
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
            SVariable::UnboundedArrayF64(_, v) => *v,
            SVariable::UnboundedArrayI64(_, v) => *v,
            SVariable::Address(_, v) => *v,
            SVariable::Struct(_, _, v) => *v,
        }
    }
    pub fn type_name(&self) -> anyhow::Result<String> {
        Ok(match self {
            SVariable::Unknown(_, _) => anyhow::bail!("Unknown has no type name"),
            SVariable::Bool(_, _) => "bool".to_string(),
            SVariable::F64(_, _) => "f64".to_string(),
            SVariable::I64(_, _) => "i64".to_string(),
            SVariable::UnboundedArrayF64(_, _) => "&[f64]".to_string(),
            SVariable::UnboundedArrayI64(_, _) => "&[i64]".to_string(),
            SVariable::Address(_, _) => "&".to_string(),
            SVariable::Struct(_, name, _) => name.to_string(),
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
    fn expect_unbounded_array_f64(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::UnboundedArrayF64(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected UnboundedArrayF64 {}", v, ctx),
        }
    }
    fn expect_unbounded_array_i64(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::UnboundedArrayI64(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected UnboundedArrayI64 {}", v, ctx),
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
            SVariable::Struct(varname, sname, v) => {
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
}

fn declare_variables(
    float: types::Type,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    params: &[Arg],
    returns: &[Arg],
    stmts: &[Expr],
    entry_block: Block,
    env: &[Declaration],
    funcs: &HashMap<String, Function>,
    constant_vars: &HashMap<String, f64>,
    struct_map: &HashMap<String, StructDef>,
) -> anyhow::Result<HashMap<String, SVariable>> {
    //TODO we should create a list of the variables here with their expected type, so it can be referenced later
    let mut variables: HashMap<String, SVariable> = HashMap::new();
    let mut index = 0;

    for (i, arg) in params.iter().enumerate() {
        let val = builder.block_params(entry_block)[i];
        let var = declare_variable(module, builder, &mut variables, &mut index, arg);
        if let Some(var) = var {
            builder.def_var(var.inner(), val);
        }
    }

    for arg in returns {
        declare_variable(module, builder, &mut variables, &mut index, arg);
    }

    for expr in stmts {
        declare_variables_in_stmt(
            module.target_config().pointer_type(),
            float,
            builder,
            &mut variables,
            &mut index,
            expr,
            env,
            funcs,
            &constant_vars,
            &struct_map,
        )?;
    }

    Ok(variables)
}

/// Recursively descend through the AST, translating all implicit
/// variable declarations.
fn declare_variables_in_stmt(
    ptr_type: types::Type,
    ty: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    expr: &Expr,
    env: &[Declaration],
    funcs: &HashMap<String, Function>,
    constant_vars: &HashMap<String, f64>,
    struct_map: &HashMap<String, StructDef>,
) -> anyhow::Result<()> {
    match *expr {
        Expr::Assign(ref names, ref exprs) => {
            if exprs.len() == names.len() {
                for (name, expr) in names.iter().zip(exprs.iter()) {
                    declare_variable_from_expr(
                        ptr_type,
                        expr,
                        builder,
                        variables,
                        index,
                        &[name],
                        env,
                        funcs,
                        constant_vars,
                        struct_map,
                    )?;
                }
            } else {
                let mut snames = Vec::new();
                for sname in names.iter() {
                    snames.push(sname.as_str());
                }
                declare_variable_from_expr(
                    ptr_type,
                    expr,
                    builder,
                    variables,
                    index,
                    &snames,
                    env,
                    funcs,
                    constant_vars,
                    struct_map,
                )?;
            }
        }
        Expr::IfElse(ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(
                    ptr_type,
                    ty,
                    builder,
                    variables,
                    index,
                    &stmt,
                    env,
                    funcs,
                    constant_vars,
                    struct_map,
                )?;
            }
            for stmt in else_body {
                declare_variables_in_stmt(
                    ptr_type,
                    ty,
                    builder,
                    variables,
                    index,
                    &stmt,
                    env,
                    funcs,
                    constant_vars,
                    struct_map,
                )?;
            }
        }
        Expr::WhileLoop(ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(
                    ptr_type,
                    ty,
                    builder,
                    variables,
                    index,
                    &stmt,
                    env,
                    funcs,
                    constant_vars,
                    struct_map,
                )?;
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
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    names: &[&str],
    env: &[Declaration],
    funcs: &HashMap<String, Function>,
    constant_vars: &HashMap<String, f64>,
    struct_map: &HashMap<String, StructDef>,
) -> anyhow::Result<()> {
    match expr {
        Expr::IfElse(_condition, then_body, _else_body) => {
            //TODO make sure then & else returns match
            declare_variable_from_expr(
                ptr_type,
                then_body.last().unwrap(),
                builder,
                variables,
                index,
                names,
                env,
                funcs,
                constant_vars,
                struct_map,
            )?;
        }
        expr => {
            let expr_type = ExprType::of(
                expr,
                &mut None,
                &env,
                funcs,
                variables,
                constant_vars,
                struct_map,
            )?;
            declare_variable_from_type(
                ptr_type, &expr_type, builder, variables, index, names, env,
            )?;
        }
    };
    Ok(())
}

fn declare_variable_from_type(
    ptr_type: Type,
    expr_type: &ExprType,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    names: &[&str],
    env: &[Declaration],
) -> anyhow::Result<()> {
    let name = *names.first().unwrap();
    if name.contains(".") {
        return Ok(());
    }
    match expr_type {
        ExprType::Void => anyhow::bail!("can't assign void type to {}", name),
        ExprType::Bool => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::Bool(name.into(), var));
                builder.declare_var(var, types::B1);
                *index += 1;
            }
        }
        ExprType::F64 => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::F64(name.into(), var));
                builder.declare_var(var, types::F64);
                *index += 1;
            }
        }
        ExprType::I64 => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::I64(name.into(), var));
                builder.declare_var(var, types::I64);
                *index += 1;
            }
        }
        ExprType::UnboundedArrayF64 => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::UnboundedArrayF64(name.into(), var));
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::UnboundedArrayI64 => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::UnboundedArrayI64(name.into(), var));
                builder.declare_var(var, ptr_type);
                *index += 1;
            }
        }
        ExprType::Address => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(name.into(), SVariable::Address(name.into(), var));
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
                            variables,
                            index,
                            &[sname],
                            env,
                        )?
                    }
                    return Ok(());
                }
            }
            for (expr_type, sname) in expr_types.iter().zip(names.iter()) {
                declare_variable_from_type(
                    ptr_type,
                    expr_type,
                    builder,
                    variables,
                    index,
                    &[sname],
                    env,
                )?
            }
        }
        ExprType::Struct(structname) => {
            if !variables.contains_key(name) {
                let var = Variable::new(*index);
                variables.insert(
                    name.into(),
                    SVariable::Struct(name.into(), structname.to_string(), var),
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
) -> Option<SVariable> {
    let ptr_ty = module.target_config().pointer_type();
    if !variables.contains_key(&arg.name) {
        let (var, ty) = match &arg.expr_type {
            Some(t) => match t {
                ExprType::F64 => (
                    SVariable::F64(arg.name.clone(), Variable::new(*index)),
                    types::F64,
                ),
                ExprType::I64 => (
                    SVariable::I64(arg.name.clone(), Variable::new(*index)),
                    types::I64,
                ),
                ExprType::UnboundedArrayF64 => (
                    SVariable::UnboundedArrayF64(arg.name.clone(), Variable::new(*index)),
                    ptr_ty,
                ),
                ExprType::UnboundedArrayI64 => (
                    SVariable::UnboundedArrayI64(arg.name.clone(), Variable::new(*index)),
                    ptr_ty,
                ),
                ExprType::Address => (
                    SVariable::Address(arg.name.clone(), Variable::new(*index)),
                    ptr_ty,
                ),
                ExprType::Void => return None,
                ExprType::Bool => (
                    SVariable::Bool(arg.name.clone(), Variable::new(*index)),
                    types::B1,
                ),
                ExprType::Tuple(_) => return None, //anyhow::bail!("single variable tuple not supported"),
                ExprType::Struct(structname) => (
                    SVariable::Struct(
                        arg.name.clone(),
                        structname.to_string(),
                        Variable::new(*index),
                    ),
                    ptr_ty,
                ),
            },
            None => (
                SVariable::F64(arg.name.clone(), Variable::new(*index)),
                types::F64,
            ),
        };
        variables.insert(arg.name.clone(), var.clone());
        builder.declare_var(var.inner(), ty);
        *index += 1;
        Some(var)
    } else {
        None
    }
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub size: u32,
    pub name: String,
    pub fields: HashMap<String, StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub offset: u32,
    pub size: u32,
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
        let mut fields = HashMap::new();
        let mut struct_size = 0u32;
        for field in in_structs[&struct_name].fields.iter() {
            let size = match &field.expr_type {
                Some(t) => match t {
                    ExprType::Void => 0u32,
                    ExprType::Bool => types::B1.bytes(),
                    ExprType::F64 => types::F64.bytes(),
                    ExprType::I64 => types::I64.bytes(),
                    ExprType::UnboundedArrayF64 => ptr_type.bytes(),
                    ExprType::UnboundedArrayI64 => ptr_type.bytes(),
                    ExprType::Address => ptr_type.bytes(),
                    ExprType::Tuple(_) => anyhow::bail!("Tuple in struct not supported"),
                    ExprType::Struct(name) => structs[&name.to_string()].size,
                },
                None => types::F64.bytes(),
            };
            fields.insert(
                field.name.to_string(),
                StructField {
                    offset: struct_size,
                    size,
                    name: field.name.to_string(),
                    expr_type: field.expr_type.as_ref().unwrap_or(&ExprType::F64).clone(),
                },
            );
            struct_size += size;
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

fn order_structs(in_structs: &HashMap<String, &Struct>) -> anyhow::Result<Vec<String>> {
    let mut structs_order = Vec::new();
    let mut last_structs_len = 0usize;
    while structs_order.len() < in_structs.len() {
        for (name, struc) in in_structs {
            let mut can_insert = true;
            for field in &struc.fields {
                match &field.expr_type {
                    Some(t) => match t {
                        ExprType::Void
                        | ExprType::Bool
                        | ExprType::F64
                        | ExprType::I64
                        | ExprType::UnboundedArrayF64
                        | ExprType::UnboundedArrayI64
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
                    },
                    None => continue,
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
