use crate::frontend::*;
use crate::sarus_std_lib;
use crate::sarus_std_lib::SConstant;
use crate::validator::validate_program;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::ir::ArgumentPurpose;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::CString;
use std::fmt::Display;
use std::slice;
use tracing::info;
use tracing::instrument;
use tracing::trace;

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

    //local variables for each function kept around for later debug/print
    pub func_variables: HashMap<String, HashMap<String, SVariable>>,
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
            func_variables: HashMap::new(),
        }
    }
}

pub fn new_jit_builder() -> JITBuilder {
    //let builder = JITBuilder::new(cranelift_module::default_libcall_names());

    //https://github.com/bytecodealliance/wasmtime/issues/2735
    //https://github.com/bytecodealliance/wasmtime-go/issues/53

    //JITBuilder::new(cranelift_module::default_libcall_names())
    let mut flag_builder = settings::builder();
    // On at least AArch64, "colocated" calls use shorter-range relocations,
    // which might not reach all definitions; we can't handle that here, so
    // we require long-range relocation types.
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "true").unwrap();
    //flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder.finish(settings::Flags::new(flag_builder));
    JITBuilder::with_isa(isa, cranelift_module::default_libcall_names())
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
            func_variables: HashMap::new(),
        }
    }

    /// Compile the ast into machine code.
    #[instrument(level = "info", skip(self, prog, src_code))]
    pub fn translate(&mut self, prog: Vec<Declaration>, src_code: String) -> anyhow::Result<()> {
        info!("--------------- translate ---------------");

        let mut prog = prog;

        let struct_map = create_struct_map(&prog, self.module.target_config().pointer_type())?;
        let constant_vars = sarus_std_lib::get_constants(&struct_map);

        let mut funcs = HashMap::new();

        for decl in prog.iter_mut() {
            match decl {
                Declaration::Function(func) => {
                    setup_coderef(&mut func.body, &src_code);
                    funcs.insert(func.name.clone(), func.clone());
                }
                Declaration::Metadata(_, _) => continue,
                Declaration::Struct(_) => continue,
                Declaration::StructMacro(_, _) => continue,
            }
        }

        for d in prog.clone() {
            match d {
                Declaration::Function(func) => {
                    if func.extern_func {
                        //Don't compile the contents of std func, it will be empty
                        trace!(
                            "Function {} is an external function, skipping codegen",
                            func.sig_string()?
                        );
                        continue;
                    }

                    //// Then, translate the AST nodes into Cranelift IR.
                    self.codegen(&func, funcs.to_owned(), &struct_map, &constant_vars)?;
                    // Next, declare the function to jit. Functions must be declared
                    // before they can be called, or defined.
                    let id = self
                        .module
                        .declare_function(&func.name, Linkage::Export, &self.ctx.func.signature)
                        .map_err(|e| {
                            anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e)
                        })?;

                    trace!("cranelift func id is {}", id);
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

        for (name, val) in constant_vars.iter() {
            match val {
                SConstant::I64(n) => self.create_data(&name, (*n).to_ne_bytes().to_vec())?,
                SConstant::F32(n) => self.create_data(&name, (*n).to_ne_bytes().to_vec())?,
                SConstant::Bool(n) => self.create_data(&name, (*n as i8).to_ne_bytes().to_vec())?,
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

    pub fn get_data(&mut self, data_name: &str) -> anyhow::Result<(*const u8, usize)> {
        match self.module.get_name(data_name) {
            Some(func) => match func {
                cranelift_module::FuncOrDataId::Func(_) => {
                    anyhow::bail!("data {} required, function found", data_name);
                }
                cranelift_module::FuncOrDataId::Data(id) => Ok(self.module.get_finalized_data(id)),
            },
            None => anyhow::bail!("No data {} found", data_name),
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
    #[instrument(level = "info", skip(self, func, funcs, struct_map, constant_vars))]
    fn codegen(
        &mut self,
        func: &Function,
        funcs: HashMap<String, Function>,
        struct_map: &HashMap<String, StructDef>,
        constant_vars: &HashMap<String, SConstant>,
    ) -> anyhow::Result<()> {
        info!("{}", func.sig_string()?);
        let ptr_ty = self.module.target_config().pointer_type();

        if func.returns.len() > 0 {
            if let ExprType::Struct(_code_ref, _struct_name) = &func.returns[0].expr_type {
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
                    ExprType::F32(_code_ref) => AbiParam::new(types::F32),
                    ExprType::I64(_code_ref) => AbiParam::new(types::I64),
                    ExprType::Array(_code_ref, _ty, _len) => AbiParam::new(ptr_ty),
                    ExprType::Address(_code_ref) => AbiParam::new(ptr_ty),
                    ExprType::Void(_code_ref) => continue,
                    ExprType::Bool(_code_ref) => AbiParam::new(types::B1),
                    ExprType::Struct(_code_ref, _) => AbiParam::new(ptr_ty),
                    ExprType::Tuple(code_ref, _) => {
                        anyhow::bail!("{} Tuple as parameter not supported", code_ref)
                    }
                }
            });
        }

        trace!("FunctionBuilder::new");
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

        // The Sarus allows variables to be declared implicitly.
        // Walk the AST and declare all implicitly-declared variables.

        let mut env = Env {
            constant_vars: constant_vars.clone(),
            struct_map: struct_map.clone(),
            ptr_ty,
            funcs,
        };

        //println!("declare_variables {}", func.name);

        let mut variables = HashMap::new();

        let mut var_index = 0;
        declare_variables(
            &mut var_index,
            &mut builder,
            &mut self.module,
            &func.params,
            &func.returns,
            &func.body,
            entry_block,
            &mut env,
            &mut variables,
            &mut None,
        )?;

        //Keep function vars around for later debug/print
        self.func_variables
            .insert(func.name.to_string(), variables.clone());

        //println!("validate_program {}", func.name);

        //Check every statement, this can catch funcs with no assignment, etc...
        validate_program(&func.body, &env, &variables)?;

        //println!("FunctionTranslator {}", func.name);
        let ptr_ty = self.module.target_config().pointer_type();

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            func,
            module: &mut self.module,
            ptr_ty,
            env,
            inline_outer_var_aliases: HashMap::new(),
            inline_prefix: Vec::new(),
            inline_variables: HashMap::new(),
            entry_block,
            var_index,
            variables,
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
                ExprType::F32(code_ref) => trans
                    .builder
                    .use_var(return_variable.expect_f32(code_ref, "return_variable")?),
                ExprType::I64(code_ref) => trans
                    .builder
                    .use_var(return_variable.expect_i64(code_ref, "return_variable")?),
                ExprType::Array(code_ref, ty, size_type) => {
                    trans.builder.use_var(return_variable.expect_array(
                        code_ref,
                        *ty.clone(),
                        size_type.clone(),
                        "return_variable",
                    )?)
                }
                ExprType::Address(code_ref) => trans
                    .builder
                    .use_var(return_variable.expect_address(code_ref, "return_variable")?),
                ExprType::Void(_code_ref) => continue,
                ExprType::Bool(code_ref) => trans
                    .builder
                    .use_var(return_variable.expect_bool(code_ref, "return_variable")?),
                ExprType::Tuple(code_ref, _) => {
                    anyhow::bail!("{} tuple not supported in return", code_ref)
                }
                //We don't actually return structs, they are passed in as StackSlotKind::StructReturnSlot and written to from there
                ExprType::Struct(_code_ref, _) => continue, //trans.builder.use_var(return_variable.expect_struct(n, "codegen return variables")?)
            };
            return_values.push(v);
        }

        // Emit the return instruction.
        trans.builder.ins().return_(&return_values);

        //Keep clif around for later debug/print
        self.clif.insert(
            func.name.to_string(),
            trans.builder.func.display(None).to_string(),
        );
        trace!("{}", trans.builder.func.display(None).to_string());

        // Tell the builder we're done with this function.
        trans.builder.finalize();

        Ok(())
    }

    pub fn print_clif(&self, show_vars: bool) {
        for (func_name, func_clif) in &self.clif {
            let mut func_clif = func_clif.clone();
            if show_vars {
                for (var_name, var) in &self.func_variables[func_name] {
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
    fn from(
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
    fn get_from_variable(
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
    fn replace_value(&self, value: Value) -> anyhow::Result<SValue> {
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
    fn inner(&self, ctx: &str) -> anyhow::Result<Value> {
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

pub struct Env {
    pub constant_vars: HashMap<String, SConstant>,
    pub struct_map: HashMap<String, StructDef>,
    pub ptr_ty: types::Type,
    pub funcs: HashMap<String, Function>,
}

/// A collection of state used for translating from Sarus AST nodes
/// into Cranelift IR.
struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    env: Env,
    func: &'a Function,
    ptr_ty: types::Type,

    pub variables: HashMap<String, SVariable>,

    // Aliases to outer vars for use instead of arg/returns
    pub inline_outer_var_aliases: HashMap<String, String>,

    // Stack of prefixes for inline
    // when entering an inline section the func name will be appended
    // when leaving the inline section the func name will be removed
    // these will be joined with the var names like this "func1->func2->var_name"
    pub inline_prefix: Vec<String>,

    // Variables that need to be accessed by inline functions
    pub inline_variables: HashMap<String, SVariable>,

    // so that we can add inline vars
    entry_block: Block,
    var_index: usize,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    #[instrument(name = "expr", skip(self, expr))]
    fn translate_expr(&mut self, expr: &Expr) -> anyhow::Result<SValue> {
        info!(
            "{}: {} | {}",
            expr.get_code_ref(),
            expr,
            expr.debug_get_name(),
        );
        //dbg!(&expr);
        match expr {
            Expr::LiteralFloat(_code_ref, literal) => Ok(SValue::F32(
                self.builder.ins().f32const::<f32>(literal.parse().unwrap()),
            )),
            Expr::LiteralInt(_code_ref, literal) => Ok(SValue::I64(
                self.builder
                    .ins()
                    .iconst::<i64>(types::I64, literal.parse().unwrap()),
            )),
            Expr::LiteralString(_code_ref, literal) => self.translate_string(literal),
            Expr::LiteralArray(code_ref, item, len) => {
                self.translate_array_create(code_ref, item, *len)
            }
            Expr::Binop(_code_ref, op, lhs, rhs) => {
                Ok(self.translate_binop(*op, lhs, rhs, false)?.0)
            }
            Expr::Unaryop(_code_ref, op, lhs) => self.translate_unaryop(*op, lhs),
            Expr::Compare(_code_ref, cmp, lhs, rhs) => self.translate_cmp(*cmp, lhs, rhs),
            Expr::Call(code_ref, name, args, is_macro) => {
                self.translate_call(code_ref, name, args, None, *is_macro)
            }
            Expr::GlobalDataAddr(_code_ref, name) => Ok(SValue::Array(
                Box::new(SValue::F32(
                    self.translate_global_data_addr(self.ptr_ty, name),
                )),
                ArraySized::Unsized,
            )),
            Expr::Identifier(code_ref, name) => {
                if let Ok(svar) = self.get_variable(code_ref, name) {
                    let svar = svar.clone();
                    SValue::get_from_variable(&mut self.builder, &svar)
                } else if let Some(sval) = self.translate_constant(code_ref, name)? {
                    Ok(sval)
                } else {
                    Ok(SValue::F32(
                        //TODO Don't assume this is a float
                        self.translate_global_data_addr(types::F32, name),
                    )) //Try to load global
                }
            }
            Expr::Assign(_code_ref, to_exprs, from_exprs) => {
                self.translate_assign(to_exprs, from_exprs)
            }
            Expr::AssignOp(_code_ref, _op, _lhs, _rhs) => {
                unimplemented!("currently AssignOp is turned into seperate assign and op")
                //self.translate_math_assign(*op, lhs, rhs),
                //for what this used to look like see:
                //https://github.com/DGriffin91/sarus/tree/cd4bf3272bf02f00ea6037d606842ec84d0ff205
            }
            Expr::NewStruct(code_ref, struct_name, fields) => {
                self.translate_new_struct(code_ref, struct_name, fields)
            }
            Expr::IfThen(_code_ref, condition, then_body) => {
                self.translate_if_then(condition, then_body)?;
                Ok(SValue::Void)
            }
            Expr::IfElse(_code_ref, condition, then_body, else_body) => {
                self.translate_if_else(condition, then_body, else_body)
            }
            Expr::IfThenElseIf(code_ref, expr_bodies) => {
                self.translate_if_then_else_if(code_ref, expr_bodies)?;
                Ok(SValue::Void)
            }
            Expr::IfThenElseIfElse(code_ref, expr_bodies, else_body) => {
                self.translate_if_then_else_if_else(code_ref, expr_bodies, else_body)
            }
            Expr::WhileLoop(_code_ref, condition, loop_body) => {
                self.translate_while_loop(condition, loop_body)?;
                Ok(SValue::Void)
            }
            Expr::Block(_code_ref, b) => b
                .into_iter()
                .map(|e| self.translate_expr(e))
                .last()
                .unwrap(),
            Expr::LiteralBool(_code_ref, b) => {
                Ok(SValue::Bool(self.builder.ins().bconst(types::B1, *b)))
            }
            Expr::Parentheses(_code_ref, expr) => self.translate_expr(expr),
            Expr::ArrayAccess(code_ref, name, idx_expr) => {
                let idx_val = self.idx_expr_to_val(idx_expr)?;
                self.translate_array_get_from_var(code_ref, name.to_string(), idx_val, false)
            }
        }
    }

    fn get_variable(&mut self, code_ref: &CodeRef, name: &str) -> anyhow::Result<&SVariable> {
        if self.inline_prefix.len() > 0 {
            //We are in an inline section
            let name = format!("{}->{}", self.inline_prefix.join("->"), name);
            let name = if let Some(alias_name) = self.inline_outer_var_aliases.get(&name) {
                alias_name.as_str()
            } else {
                name.as_str()
            };
            match self.inline_variables.get(name) {
                Some(v) => Ok(v),
                None => anyhow::bail!("{} inline variable {} not found", code_ref, name),
            }
        } else {
            match self.variables.get(name) {
                Some(v) => Ok(v),
                None => anyhow::bail!("{} variable {} not found", code_ref, name),
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

    fn translate_array_create(
        &mut self,
        code_ref: &CodeRef,
        item_expr: &Expr,
        len: usize,
    ) -> anyhow::Result<SValue> {
        trace!(
            "{}: translate_array_create item_expr {} len {}",
            code_ref,
            item_expr,
            len,
        );
        let item_value = self.translate_expr(item_expr)?;

        let item_expr_type = item_value.expr_type(code_ref)?;

        let item_width = match item_expr_type.width(self.ptr_ty, &self.env.struct_map) {
            Some(item_width) => item_width,
            None => anyhow::bail!("{} expression {} has no size", code_ref, item_expr),
        };

        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (item_width * len) as u32,
        ));
        let stack_slot_address =
            self.builder
                .ins()
                .stack_addr(self.ptr_ty, stack_slot, Offset32::new(0));

        for i in 0..len {
            let val = item_value.inner("translate_array_create")?;
            match item_expr_type {
                ExprType::Bool(_) | ExprType::F32(_) | ExprType::I64(_) | ExprType::Address(_) => {
                    self.builder.ins().store(
                        MemFlags::new(),
                        val,
                        stack_slot_address,
                        Offset32::new((i * item_width) as i32),
                    );
                }
                ExprType::Struct(code_ref, _) | ExprType::Array(code_ref, _, _) => {
                    let offset = i * item_width;
                    let offset_v = self.builder.ins().iconst(self.ptr_ty, offset as i64);
                    let stack_slot_offset = self.builder.ins().iadd(stack_slot_address, offset_v);
                    trace!(
                        "{}: emit_small_memory_copy size {} offset {}",
                        code_ref,
                        item_width,
                        offset
                    );
                    self.builder.emit_small_memory_copy(
                        self.module.target_config(),
                        stack_slot_offset,
                        val,
                        item_width as u64,
                        1,
                        1,
                        true,
                        MemFlags::new(),
                    );
                }
                ExprType::Tuple(_, _) | ExprType::Void(_) => anyhow::bail!(
                    "{} cannot assign expression {} to array",
                    code_ref,
                    item_expr
                ),
            }
        }
        let len_val = Box::new(SValue::I64(
            self.builder.ins().iconst(types::I64, len as i64),
        ));
        let ret_val = SValue::Array(
            Box::new(SValue::from(
                &mut self.builder,
                &item_value.expr_type(code_ref)?,
                stack_slot_address,
            )?),
            ArraySized::Fixed(len_val, len),
        );

        Ok(ret_val)
    }

    #[instrument(name = "binop", skip(self, op, lhs, rhs, get_address))]
    fn translate_binop(
        &mut self,
        op: Binop,
        lhs: &Expr,
        rhs: &Expr,
        get_address: bool,
    ) -> anyhow::Result<(SValue, Option<StructField>, bool)> {
        if let Binop::DotAccess = op {
        } else {
            return Ok((self.translate_math_binop(op, lhs, rhs)?, None, false));
        }

        let mut path = Vec::new();
        let mut lhs_val = None;
        let mut struct_field_def = None;
        let mut array_field = false;

        let mut curr_expr = Some(lhs);
        let mut next_expr = Some(rhs);

        let mut log_path = Vec::new();

        loop {
            //println!("curr_expr {:?} next_expr {:?}", &curr_expr, &next_expr);
            //println!("path {:?}", &path);
            match curr_expr {
                Some(expr) => {
                    let debug_name = expr.debug_get_name();
                    if debug_name != "Binop" {
                        log_path.push(debug_name);
                    }
                    curr_expr = next_expr;
                    next_expr = None;
                    array_field = false;
                    match expr {
                        Expr::Call(code_ref, fn_name, args, is_macro) => {
                            if path.len() == 0 {
                                lhs_val =
                                    Some(self.translate_call(
                                        code_ref, fn_name, args, lhs_val, *is_macro,
                                    )?);
                            } else {
                                let sval = if path.len() > 1 || lhs_val.is_some() {
                                    let spath = path
                                        .iter()
                                        .map(|lhs_i: &Expr| lhs_i.to_string())
                                        .collect::<Vec<String>>();
                                    let (sval_address, struct_def) =
                                        self.get_struct_field_address(code_ref, spath, lhs_val)?;
                                    if let ExprType::Struct(_code_ref, _name) =
                                        struct_def.clone().expr_type
                                    {
                                        struct_field_def = Some(struct_def);
                                        sval_address
                                    } else {
                                        //dbg!(&struct_def);
                                        self.get_struct_field(code_ref, sval_address, &struct_def)?
                                    }
                                } else {
                                    self.translate_expr(&path[0])?
                                };
                                //dbg!(&sval);
                                lhs_val = Some(self.translate_call(
                                    code_ref,
                                    fn_name,
                                    args,
                                    Some(sval),
                                    *is_macro,
                                )?);
                                path = Vec::new();
                            }
                        }
                        Expr::LiteralFloat(_code_ref, _) => todo!(),
                        Expr::LiteralInt(_code_ref, _) => todo!(),
                        Expr::LiteralBool(_code_ref, _) => todo!(),
                        Expr::LiteralString(_code_ref, _) => {
                            lhs_val = Some(self.translate_expr(expr)?)
                        }
                        Expr::LiteralArray(_code_ref, _, _) => {
                            lhs_val = Some(self.translate_expr(expr)?)
                        }
                        Expr::Identifier(_code_ref, _) => path.push(expr.clone()),
                        Expr::Binop(_code_ref, op, lhs, rhs) => {
                            if let Binop::DotAccess = op {
                                curr_expr = Some(lhs);
                                next_expr = Some(rhs);
                            } else {
                                todo!();
                            }
                        }
                        Expr::Unaryop(_code_ref, _, _) => todo!(),
                        Expr::Compare(_code_ref, _, _, _) => todo!(),
                        Expr::IfThen(_code_ref, _, _) => todo!(),
                        Expr::IfElse(_code_ref, _, _, _) => todo!(), //TODO, this should actually be possible
                        Expr::IfThenElseIf(_code_ref, _) => todo!(),
                        Expr::IfThenElseIfElse(_code_ref, _, _) => todo!(), //TODO, this should actually be possible
                        Expr::Assign(_code_ref, _, _) => todo!(),
                        Expr::AssignOp(_code_ref, _, _, _) => todo!(),
                        Expr::NewStruct(_code_ref, _, _) => todo!(),
                        Expr::WhileLoop(_code_ref, _, _) => todo!(),
                        Expr::Block(_code_ref, _) => todo!(),
                        Expr::GlobalDataAddr(_code_ref, _) => todo!(),
                        Expr::Parentheses(_code_ref, e) => lhs_val = Some(self.translate_expr(e)?),
                        Expr::ArrayAccess(code_ref, name, idx_expr) => {
                            if path.len() > 0 {
                                let mut spath = path
                                    .iter()
                                    .map(|lhs_i: &Expr| lhs_i.to_string())
                                    .collect::<Vec<String>>();
                                spath.push(name.to_string());
                                let (sval_address, struct_def) =
                                    self.get_struct_field_address(code_ref, spath, lhs_val)?;
                                struct_field_def = Some(struct_def.clone());
                                let array_address = if let ExprType::Struct(_code_ref, _name) =
                                    struct_def.clone().expr_type
                                {
                                    sval_address //TODO should this also just get the address if it's a fixed array?
                                } else {
                                    self.get_struct_field(code_ref, sval_address, &struct_def)?
                                    //struct_of_slices_of_structs fails if this is always sval_address so this is sometimes wanted
                                };
                                array_field = true;
                                let idx_val = self.idx_expr_to_val(idx_expr)?;
                                lhs_val = Some(self.array_get(
                                    array_address.inner("Expr::ArrayAccess")?,
                                    &struct_def.expr_type,
                                    idx_val,
                                    get_address,
                                    true,
                                )?);
                            } else {
                                let idx_val = self.idx_expr_to_val(idx_expr)?;
                                lhs_val = Some(self.translate_array_get_from_var(
                                    code_ref,
                                    name.to_string(),
                                    idx_val,
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
            let (sval_address, struct_def) =
                self.get_struct_field_address(path[0].get_code_ref(), spath, lhs_val)?;
            let code_ref = struct_def.expr_type.get_code_ref();
            if get_address {
                struct_field_def = Some(struct_def);
                lhs_val = Some(sval_address);
            } else if let ExprType::Struct(_code_ref, _name) = struct_def.clone().expr_type {
                struct_field_def = Some(struct_def);
                lhs_val = Some(sval_address);
            } else {
                lhs_val = Some(self.get_struct_field(&code_ref, sval_address, &struct_def)?)
            }
        }

        info!(
            "{{{}}} {}: {}{}{} | {}",
            if get_address { "get_address" } else { "" },
            lhs.get_code_ref(),
            lhs,
            op,
            rhs,
            log_path.join("."),
        );

        if let Some(lhs_val) = lhs_val {
            Ok((lhs_val, struct_field_def, array_field))
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
            SValue::F32(a) => match rhs_v {
                SValue::F32(b) => Ok(SValue::F32(self.binop_float(op, a, b)?)),
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
            | SValue::F32(_)
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

    fn translate_cmp(
        &mut self,
        cmp: Cmp,
        lhs_expr: &Expr,
        rhs_expr: &Expr,
    ) -> anyhow::Result<SValue> {
        let lhs = self.translate_expr(lhs_expr).unwrap();
        let rhs = self.translate_expr(rhs_expr).unwrap();
        // if a or b is a float, convert to other to a float
        match lhs {
            SValue::F32(a) => match rhs {
                SValue::F32(b) => Ok(SValue::Bool(self.cmp_float(cmp, a, b))),
                _ => anyhow::bail!(
                    "{} compare not supported: {:?} {} {:?}",
                    lhs_expr.get_code_ref(),
                    lhs,
                    cmp,
                    rhs
                ),
            },
            SValue::I64(a) => match rhs {
                SValue::I64(b) => Ok(SValue::Bool(self.cmp_int(cmp, a, b))),
                _ => anyhow::bail!(
                    "{} compare not supported: {:?} {} {:?}",
                    lhs_expr.get_code_ref(),
                    lhs,
                    cmp,
                    rhs
                ),
            },
            SValue::Bool(a) => match rhs {
                SValue::Bool(b) => Ok(SValue::Bool(self.cmp_bool(cmp, a, b))),
                _ => anyhow::bail!(
                    "{} compare not supported: {:?} {} {:?}",
                    lhs_expr.get_code_ref(),
                    lhs,
                    cmp,
                    rhs
                ),
            },
            SValue::Void
            | SValue::Unknown(_)
            | SValue::Array(_, _)
            | SValue::Address(_)
            | SValue::Struct(_, _)
            | SValue::Tuple(_) => {
                anyhow::bail!(
                    "{} compare not supported: {:?} {} {:?}",
                    lhs_expr.get_code_ref(),
                    lhs,
                    cmp,
                    rhs
                )
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
        dst_exprs: &[Expr],
        src_exprs: &[Expr],
    ) -> anyhow::Result<SValue> {
        // `def_var` is used to write the value of a variable. Note that
        // variables can have multiple definitions. Cranelift will
        // convert them into SSA form for itself automatically.

        //if there are the same number of expressions as there are names
        //eg: `a, b = b, a` then use the first output of each expression
        //But if there is not, use the output of the first expression:
        //eg: `a, b = func_that_outputs_2_floats(1.0)`
        if dst_exprs.len() == src_exprs.len() {
            'expression: for (i, dst_expr) in dst_exprs.iter().enumerate() {
                let src_sval = self.translate_expr(src_exprs.get(i).unwrap())?;
                match dst_expr {
                    Expr::Binop(_code_ref, op, lhs, rhs) => {
                        let (struct_field_address, struct_field, array_field) =
                            self.translate_binop(*op, lhs, rhs, true)?;

                        if let Some(struct_field_def) = struct_field {
                            self.set_struct_field_at_address(
                                struct_field_address,
                                src_sval,
                                struct_field_def,
                                array_field,
                            )?
                        } else {
                            unreachable!()
                        }
                    }
                    Expr::Identifier(code_ref, name) => {
                        //Can this be done without clone?
                        let dst_svar = self.get_variable(code_ref, name)?.clone();

                        for arg in &self.func.params {
                            if *name == arg.name {
                                /*
                                Should this happen also if the var already exists and has already been initialized?
                                (this can't really be determined at compile time. One option would be to allocate
                                the stack space for all potential vars to be used in a given function. But that
                                could also be excessive. Also even if this copy happens, the stack data from the
                                src_var won't be freed until after the function returns. This could be another
                                reason for having scopes work more like they do in other languages. Then if a stack
                                allocation is being created in a loop, it will be freed on each loop if it hasn't been
                                stored to a var that is outside of the scope of the loop) (How is works also has
                                implications for how aliasing works)
                                */
                                if let ExprType::Struct(code_ref, struct_name) = &arg.expr_type {
                                    trace!(
                                        "{} struct {} is arg {} of this fn {} copying onto memory at {} on assignment",
                                        code_ref,
                                        struct_name,
                                        arg.name,
                                        self.func.name,
                                        arg.name,
                                    );
                                    //copy to struct that was passed in as parameter
                                    let struct_address = self.builder.use_var(dst_svar.inner());
                                    copy_to_stack_slot(
                                        self.module.target_config(),
                                        &mut self.builder,
                                        self.env.struct_map[&struct_name.to_string()].size,
                                        src_sval.expect_struct(
                                            &struct_name.to_string(),
                                            &format!("{} translate_assign", code_ref),
                                        )?,
                                        struct_address,
                                        0,
                                    )?;
                                    continue 'expression;
                                } else if let ExprType::Array(code_ref, expr_type, size_type) =
                                    &arg.expr_type
                                {
                                    match size_type {
                                        ArraySizedExpr::Unsized => (), //Use normal assignment below
                                        ArraySizedExpr::Sized => todo!(),
                                        ArraySizedExpr::Fixed(len) => {
                                            trace!(
                                                "{} array {} is arg {} of this fn {} copying onto memory at {} on assignment",
                                                code_ref,
                                                expr_type,
                                                arg.name,
                                                self.func.name,
                                                arg.name,
                                            );
                                            //copy to array that was passed in as parameter
                                            let array_address =
                                                self.builder.use_var(dst_svar.inner());
                                            copy_to_stack_slot(
                                                self.module.target_config(),
                                                &mut self.builder,
                                                *len * expr_type
                                                    .width(self.ptr_ty, &self.env.struct_map)
                                                    .unwrap(),
                                                src_sval.inner("translate_assign")?,
                                                array_address,
                                                0,
                                            )?;
                                            continue 'expression;
                                        }
                                    }
                                }
                            }
                        }

                        if let SVariable::Struct(var_name, struct_name, var, return_struct) =
                            &dst_svar
                        {
                            //TODO also copy fixed array
                            if *return_struct {
                                trace!(
                                    "{}: struct {} is a return {} of this fn {} copying onto memory at {} on assignment",
                                    code_ref,
                                    struct_name,
                                    var_name,
                                    self.func.name,
                                    var_name,
                                );
                                //copy to struct, we don't want to return a reference
                                let return_address = self.builder.use_var(*var);
                                if !self.env.struct_map.contains_key(struct_name) {
                                    anyhow::bail!(
                                        "{} struct {} does not exist",
                                        code_ref,
                                        struct_name
                                    )
                                }
                                copy_to_stack_slot(
                                    self.module.target_config(),
                                    &mut self.builder,
                                    self.env.struct_map[struct_name].size,
                                    src_sval.expect_struct(&struct_name, "translate_assign")?,
                                    return_address,
                                    0,
                                )?;
                                continue 'expression;
                            }
                        }
                        if dst_svar.expr_type(code_ref)? != src_sval.expr_type(code_ref)? {
                            anyhow::bail!(
                                "{} cannot assign value of type {} to variable {} of type {} ",
                                code_ref,
                                src_sval.expr_type(code_ref)?,
                                dst_svar,
                                dst_svar.expr_type(code_ref)?,
                            )
                        }
                        self.builder
                            .def_var(dst_svar.inner(), src_sval.inner("translate_assign")?);
                    }
                    Expr::ArrayAccess(_code_ref, name, idx_expr) => {
                        let idx_val = self.idx_expr_to_val(idx_expr)?;
                        self.translate_array_set_from_var(name.to_string(), idx_val, &src_sval)?;
                    }
                    _ => {
                        //dbg!(dst_expr);
                        todo!()
                    }
                }
            }
            Ok(SValue::Void)
        } else {
            match self.translate_expr(src_exprs.first().unwrap())? {
                SValue::Tuple(values) => {
                    for (i, dst_expr) in dst_exprs.iter().enumerate() {
                        if let Expr::Binop(_code_ref, _, _, _) = dst_expr {
                            todo!()
                            //self.set_struct_field(dst_expr, values[i].clone())?
                        } else if let Expr::Identifier(code_ref, name) = dst_expr {
                            let var = self.get_variable(code_ref, name)?.inner();
                            self.builder
                                .def_var(var, values[i].inner("translate_assign")?);
                        } else {
                            todo!()
                        }
                    }

                    Ok(SValue::Void)
                }
                SValue::Void
                | SValue::Unknown(_)
                | SValue::Bool(_)
                | SValue::F32(_)
                | SValue::I64(_)
                | SValue::Array(_, _)
                | SValue::Address(_)
                | SValue::Struct(_, _) => anyhow::bail!("operation not supported {:?}", src_exprs),
            }
        }
    }

    fn translate_array_get_from_var(
        &mut self,
        code_ref: &CodeRef,
        name: String,
        idx_val: Value,
        get_address: bool,
    ) -> anyhow::Result<SValue> {
        let variable = self.get_variable(code_ref, &name)?.clone();
        let array_expr_type = &variable.expr_type(code_ref)?;

        match variable {
            SVariable::Array(address, size_type) => {
                let mut bound_check_at_get = true;
                match size_type {
                    ArraySized::Unsized => (),
                    ArraySized::Sized => todo!(),
                    ArraySized::Fixed(sval, _len) => {
                        let b_condition_value = self.builder.ins().icmp(
                            IntCC::SignedGreaterThanOrEqual,
                            idx_val,
                            sval.inner("translate_array_get_from_var")?,
                        );
                        let merge_block = self.exec_if_start(b_condition_value);
                        self.call_panic(code_ref, &format!("{} index out of bounds", code_ref))?;
                        self.exec_if_end(merge_block);
                        bound_check_at_get = false;
                    }
                }
                let array_address = self.builder.use_var(address.inner());
                self.array_get(
                    array_address,
                    array_expr_type,
                    idx_val,
                    get_address,
                    bound_check_at_get,
                )
            }
            _ => anyhow::bail!("{} variable {} is not an array", code_ref, name),
        }
    }

    fn idx_expr_to_val(&mut self, idx_expr: &Expr) -> anyhow::Result<Value> {
        let idx_val = self.translate_expr(idx_expr).unwrap();
        match idx_val {
            SValue::I64(v) => Ok(v),
            _ => anyhow::bail!("only int supported for array access"),
        }
    }

    fn array_get(
        &mut self,
        array_address: Value,
        array_expr_type: &ExprType,
        idx_val: Value,
        get_address: bool,
        check_bounds: bool,
    ) -> anyhow::Result<SValue> {
        let mut width;
        let base_type = match &array_expr_type {
            ExprType::Array(code_ref, ty, size_type) => {
                if check_bounds {
                    match size_type {
                        ArraySizedExpr::Unsized => (),
                        ArraySizedExpr::Sized => todo!(),
                        ArraySizedExpr::Fixed(len) => {
                            //Looks expensive
                            let len_val = self.builder.ins().iconst(types::I64, *len as i64);
                            let b_condition_value = self.builder.ins().icmp(
                                IntCC::SignedGreaterThanOrEqual,
                                idx_val,
                                len_val,
                            );
                            let merge_block = self.exec_if_start(b_condition_value);
                            self.call_panic(
                                code_ref,
                                &format!("{} index out of bounds", code_ref),
                            )?;
                            self.exec_if_end(merge_block);
                        }
                    }
                }
                let c_ty = ty.cranelift_type(self.ptr_ty, true)?;
                width = ty
                    .width(self.ptr_ty, &self.env.struct_map)
                    .unwrap_or(self.ptr_ty.bytes() as usize);
                match *ty.to_owned() {
                    ExprType::Void(_) => (),
                    ExprType::Bool(_) => (),
                    ExprType::F32(_) => (),
                    ExprType::I64(_) => (),
                    ExprType::Array(_, expr_type, size_type) => match size_type {
                        ArraySizedExpr::Unsized => (),
                        ArraySizedExpr::Sized => todo!(),
                        ArraySizedExpr::Fixed(len) => {
                            width =
                                expr_type.width(self.ptr_ty, &self.env.struct_map).unwrap() * len;
                            let array_address_at_idx_ptr =
                                self.get_array_address_from_ptr(width, array_address, idx_val);
                            return Ok(SValue::Array(
                                Box::new(SValue::from(
                                    &mut self.builder,
                                    &expr_type,
                                    array_address_at_idx_ptr,
                                )?),
                                ArraySized::from(&mut self.builder, &size_type),
                            ));
                        }
                    },
                    ExprType::Address(_) => (),
                    ExprType::Tuple(_, _) => (),
                    ExprType::Struct(_code_ref, name) => {
                        //if the items of the array are structs return struct with same start address
                        let base_struct = self.env.struct_map[&name.to_string()].clone();
                        width = base_struct.size;
                        let array_address_at_idx_ptr =
                            self.get_array_address_from_ptr(width, array_address, idx_val);
                        return Ok(SValue::Struct(base_struct.name, array_address_at_idx_ptr));
                    }
                }
                c_ty
            }
            e => {
                anyhow::bail!("{} can't index type {}", e.get_code_ref(), &array_expr_type)
            }
        };

        let array_address_at_idx_ptr =
            self.get_array_address_from_ptr(width, array_address, idx_val);
        if get_address {
            Ok(SValue::Address(array_address_at_idx_ptr))
        } else {
            let val = self.builder.ins().load(
                base_type,
                MemFlags::new(),
                array_address_at_idx_ptr,
                Offset32::new(0),
            );
            match &array_expr_type {
                ExprType::Array(_code_ref, ty, _len) => {
                    Ok(SValue::from(&mut self.builder, ty, val)?)
                }
                e => {
                    anyhow::bail!("{} can't index type {}", e.get_code_ref(), &array_expr_type)
                }
            }
        }
    }

    fn get_array_address_from_ptr(
        &mut self,
        step_bytes: usize,
        array_ptr: Value,
        idx_val: Value,
    ) -> Value {
        let mult_n = self.builder.ins().iconst(self.ptr_ty, step_bytes as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        idx_ptr
    }

    fn translate_array_set_from_var(
        &mut self,
        name: String,
        idx_val: Value,
        val: &SValue,
    ) -> anyhow::Result<SValue> {
        //TODO crash if idx_val > ExprType::Array(_, len)
        let variable = self.get_variable(&CodeRef::z(), &name)?.clone();

        let array_address = self.builder.use_var(variable.inner());

        let array_expr_type = &variable.expr_type(&CodeRef::z())?;

        self.array_set(val, &array_address, array_expr_type, idx_val, true)?;

        Ok(SValue::Void)
    }

    fn array_set(
        &mut self,
        from_val: &SValue,
        array_address: &Value,
        array_expr_type: &ExprType,
        idx_val: Value,
        check_bounds: bool,
    ) -> anyhow::Result<()> {
        let array_address_at_idx_ptr =
            self.array_get(*array_address, array_expr_type, idx_val, true, check_bounds)?;

        match array_address_at_idx_ptr {
            SValue::Void => todo!(),
            SValue::Unknown(_) => todo!(),
            SValue::Bool(_) => todo!(),
            SValue::F32(_) => {}
            SValue::I64(_) => todo!(),
            SValue::Array(_, _) => todo!(),
            SValue::Tuple(_) => todo!(),
            SValue::Address(address) => {
                self.builder.ins().store(
                    MemFlags::new(),
                    from_val.inner("array_set")?,
                    address,
                    Offset32::new(0),
                );
            }
            SValue::Struct(struct_name, struct_address) => {
                copy_to_stack_slot(
                    self.module.target_config(),
                    &mut self.builder,
                    self.env.struct_map[&struct_name].size,
                    from_val.inner("array_set")?,
                    struct_address,
                    0,
                )?;
            }
        }
        Ok(())
    }

    fn translate_if_then(
        &mut self,
        condition: &Expr,
        then_body: &[Expr],
    ) -> anyhow::Result<SValue> {
        let b_condition_value = self.translate_expr(condition)?.inner("if_then")?;

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
        let b_condition_value = self.translate_expr(condition)?.inner("translate_if_else")?;

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // If-else constructs in the Sarus have a return value.
        // In traditional SSA form, this would produce a PHI between
        // the then and else bodies. Cranelift uses block parameters,
        // so set up a parameter in the merge block, and we'll pass
        // the return values to it from the branches.

        // Test the if condition and conditionally branch.
        self.builder.ins().brz(b_condition_value, else_block, &[]);
        // Fall through to then block.
        self.builder.ins().jump(then_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);

        for (i, expr) in then_body.iter().enumerate() {
            if i != then_body.len() - 1 {
                self.translate_expr(expr).unwrap();
            }
        }

        let then_value = self.translate_expr(then_body.last().unwrap())?;
        let then_return = match then_value.clone() {
            SValue::Tuple(t) => {
                let mut vals = Vec::new();
                for v in &t {
                    vals.push(v.inner("then_return")?);
                }
                vals
            }
            SValue::Void => vec![],
            sv => {
                let v = sv.inner("then_return")?;
                vec![v]
            }
        };

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &then_return);

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);

        for (i, expr) in else_body.iter().enumerate() {
            if i != else_body.len() - 1 {
                self.translate_expr(expr).unwrap();
            }
        }

        let else_value = self.translate_expr(else_body.last().unwrap())?;
        let else_return = match else_value.clone() {
            SValue::Tuple(t) => {
                let mut vals = Vec::new();
                for sval in &t {
                    let v = sval.inner("else_return")?;
                    self.builder
                        .append_block_param(merge_block, self.value_type(v));
                    vals.push(v);
                }
                vals
            }
            SValue::Void => vec![],
            sval => {
                let v = sval.inner("else_return")?;
                self.builder
                    .append_block_param(merge_block, self.value_type(v));
                vec![v]
            }
        };

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &else_return);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);

        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block);

        if then_value.to_string() != else_value.to_string() {
            anyhow::bail!(
                "if_else return types don't match {:?} {:?}",
                then_value,
                else_value
            )
        }

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

    fn translate_if_then_else_if(
        &mut self,
        code_ref: &CodeRef,
        condition_bodies: &Vec<(Expr, Vec<Expr>)>,
    ) -> anyhow::Result<SValue> {
        //TODO see how rust or other languages do this, there may be a more efficient way
        trace!(
            "{}: translate_if_then_else_if {:?}",
            code_ref,
            condition_bodies
        );

        let mut b_condition_value;

        let mut eval_blocks = Vec::new();
        let mut branch_blocks = Vec::new();

        for _ in 0..condition_bodies.len() {
            eval_blocks.push(self.builder.create_block());
            branch_blocks.push(self.builder.create_block());
        }

        let merge_block = self.builder.create_block();

        self.builder.ins().jump(eval_blocks[0], &[]);

        for i in 0..condition_bodies.len() {
            self.builder.switch_to_block(eval_blocks[i]);
            self.builder.seal_block(eval_blocks[i]);
            b_condition_value = self
                .translate_expr(&condition_bodies[i].0)?
                .inner("translate_if_then_else_if")?;

            if i < condition_bodies.len() - 1 {
                self.builder
                    .ins()
                    .brz(b_condition_value, eval_blocks[i + 1], &[]);
            } else {
                self.builder.ins().brz(b_condition_value, merge_block, &[]);
            }
            self.builder.ins().jump(branch_blocks[i], &[]);

            self.builder.switch_to_block(branch_blocks[i]);
            self.builder.seal_block(branch_blocks[i]);
            let body = &condition_bodies[i].1;
            for expr in body {
                self.translate_expr(expr).unwrap();
            }
            self.builder.ins().jump(merge_block, &[]);
        }

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);
        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        Ok(SValue::Void)
    }

    fn translate_if_then_else_if_else(
        &mut self,
        code_ref: &CodeRef,
        condition_bodies: &Vec<(Expr, Vec<Expr>)>,
        else_body: &Vec<Expr>,
    ) -> anyhow::Result<SValue> {
        //TODO see how rust or other languages do this, there may be a more efficient way
        trace!(
            "{}: translate_if_then_else_if_else {:?}",
            code_ref,
            condition_bodies
        );

        let mut b_condition_value;

        let mut eval_blocks = Vec::new();
        let mut branch_blocks = Vec::new();
        let mut first_branch_block_value: Option<SValue> = None;

        for _ in 0..condition_bodies.len() {
            eval_blocks.push(self.builder.create_block());
            branch_blocks.push(self.builder.create_block());
        }

        let merge_block = self.builder.create_block();
        let else_block = self.builder.create_block();

        self.builder.ins().jump(eval_blocks[0], &[]);

        for i in 0..condition_bodies.len() {
            //Don't make eval block for else
            self.builder.switch_to_block(eval_blocks[i]);
            self.builder.seal_block(eval_blocks[i]);
            b_condition_value = self
                .translate_expr(&condition_bodies[i].0)?
                .inner("translate_if_then_else_if")?;

            if i < condition_bodies.len() - 1 {
                self.builder
                    .ins()
                    .brz(b_condition_value, eval_blocks[i + 1], &[]);
            } else {
                self.builder.ins().brz(b_condition_value, else_block, &[]);
            }

            self.builder.ins().jump(branch_blocks[i], &[]);

            self.builder.switch_to_block(branch_blocks[i]);
            self.builder.seal_block(branch_blocks[i]);
            let body = &condition_bodies[i].1;
            for (i, expr) in body.iter().enumerate() {
                if i != body.len() - 1 {
                    self.translate_expr(expr).unwrap();
                }
            }
            let branch_block_value = self.translate_expr(body.last().unwrap())?;
            let branch_block_return = match branch_block_value.clone() {
                SValue::Tuple(t) => {
                    let mut vals = Vec::new();
                    for v in &t {
                        vals.push(v.inner("then_return")?);
                    }
                    vals
                }
                SValue::Void => vec![],
                sv => {
                    let v = sv.inner("then_return")?;
                    vec![v]
                }
            };
            if let Some(first_branch_block_value) = &first_branch_block_value {
                if first_branch_block_value.to_string() != branch_block_value.to_string() {
                    anyhow::bail!(
                        "if_else return types don't match {:?} {:?}",
                        first_branch_block_value.to_string(),
                        branch_block_value.to_string()
                    )
                }
            } else {
                first_branch_block_value = Some(branch_block_value)
            }
            self.builder.ins().jump(merge_block, &branch_block_return);
        }

        // ELSE BLOCK //

        self.builder.switch_to_block(else_block);
        self.builder.seal_block(else_block);
        for (i, expr) in else_body.iter().enumerate() {
            if i != else_body.len() - 1 {
                self.translate_expr(expr).unwrap();
            }
        }
        let else_value = self.translate_expr(else_body.last().unwrap())?;
        let else_return = match else_value.clone() {
            SValue::Tuple(t) => {
                let mut vals = Vec::new();
                for sval in &t {
                    let v = sval.inner("translate_if_else_if_else")?;
                    self.builder
                        .append_block_param(merge_block, self.value_type(v));
                    vals.push(v);
                }
                vals
            }
            SValue::Void => vec![],
            sval => {
                let v = sval.inner("translate_if_else_if_else")?;
                self.builder
                    .append_block_param(merge_block, self.value_type(v));
                vec![v]
            }
        };

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &else_return);

        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);
        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);

        // Read the value of the if-else by reading the merge block
        // parameter.
        let phi = self.builder.block_params(merge_block);

        trace!("{:?} | {:?}", phi, first_branch_block_value);

        if phi.len() > 1 {
            //TODO the frontend doesn't have the syntax support for this yet
            if let SValue::Tuple(then_tuple) = first_branch_block_value.unwrap() {
                let mut ret_tuple = Vec::new();
                for (phi_val, sval) in phi.iter().zip(then_tuple.iter()) {
                    ret_tuple.push(sval.replace_value(*phi_val)?)
                }
                Ok(SValue::Tuple(ret_tuple))
            } else {
                anyhow::bail!("expected tuple")
            }
        } else if phi.len() == 1 {
            first_branch_block_value
                .unwrap()
                .replace_value(*phi.first().unwrap())
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

        let b_condition_value = self
            .translate_expr(condition)?
            .inner("translate_while_loop")?;

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
        code_ref: &CodeRef,
        fn_name: &str,
        args: &[Expr],
        impl_val: Option<SValue>,
        is_macro: bool,
    ) -> anyhow::Result<SValue> {
        let mut fn_name = fn_name.to_string();
        trace!(
            "{}: translate_call {} {:?} {:?}",
            code_ref,
            &fn_name,
            &args,
            &impl_val
        );
        let mut arg_values = Vec::new();
        if let Some(impl_sval) = impl_val {
            fn_name = format!("{}.{}", impl_sval.to_string(), fn_name);
            arg_values.push(impl_sval);
        }

        for expr in args.iter() {
            arg_values.push(self.translate_expr(expr)?)
        }

        if is_macro {
            todo!()
            // returns = (macros[fn_name])(code_ref, arg_values, self.env)
        }

        if !self.env.funcs.contains_key(&fn_name) {
            anyhow::bail!("{} function {} not found", code_ref, fn_name)
        }
        let returns = &self.env.funcs[&fn_name].returns;

        let mut stack_slot_return = None;

        if returns.len() > 0 {
            if let ExprType::Struct(_code_ref, name) = &returns[0].expr_type {
                if let Some(s) = self.env.struct_map.get(&name.to_string()) {
                    stack_slot_return = Some((name.to_string(), s.size));
                }
            }
        }

        self.call_with_svalues(&code_ref, &fn_name, arg_values, stack_slot_return)
    }

    fn translate_constant(
        &mut self,
        code_ref: &CodeRef,
        name: &str,
    ) -> anyhow::Result<Option<SValue>> {
        if let Some(const_var) = self.env.constant_vars.get(name) {
            trace!("{}: translate_constant {}", code_ref, name);
            let expr = const_var.expr_type(None);
            let data_addr =
                self.translate_global_data_addr(expr.cranelift_type(self.ptr_ty, true)?, name);
            Ok(Some(SValue::from(&mut self.builder, &expr, data_addr)?))
        } else {
            Ok(None)
        }
    }

    fn translate_global_data_addr(&mut self, data_type: Type, name: &str) -> Value {
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
            global_type: data_type,
            readonly: true,
        });

        //self.builder.ins().symbol_value(ptr_ty, local_id)
        self.builder.ins().global_value(data_type, global_val)
    }

    fn value_type(&self, val: Value) -> Type {
        self.builder.func.dfg.value_type(val)
    }

    fn translate_new_struct(
        &mut self,
        code_ref: &CodeRef,
        struct_name: &str,
        fields: &[StructAssignField],
    ) -> anyhow::Result<SValue> {
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            self.env.struct_map[struct_name].size as u32,
        ));

        for field in fields.iter() {
            let dst_field_def = &self.env.struct_map[struct_name].fields[&field.field_name].clone();
            let sval = self.translate_expr(&field.expr)?;

            let mem_copy = match &sval {
                SValue::Struct(src_name, src_start_ptr) => {
                    trace!(
                        "{}: copy struct {} into struct {}",
                        code_ref,
                        &src_name,
                        &struct_name
                    );
                    if *src_name != *dst_field_def.expr_type.to_string() {
                        anyhow::bail!(
                            "{} struct {} expected struct {} for field {} but got {} instead",
                            code_ref,
                            struct_name,
                            dst_field_def.expr_type.to_string(),
                            dst_field_def.name,
                            src_name
                        )
                    }

                    Some((*src_start_ptr, self.env.struct_map[src_name].size as u64))
                }
                SValue::Array(sval_item, size_type) => match size_type {
                    ArraySized::Unsized => None,
                    ArraySized::Sized => todo!(),
                    ArraySized::Fixed(_sval_len, array_len) => {
                        trace!(
                            "{}: copy array {} into struct {}",
                            code_ref,
                            &field.field_name,
                            &struct_name
                        );
                        let array_item_width = (sval_item
                            .expr_type(code_ref)?
                            .width(self.ptr_ty, &self.env.struct_map))
                        .unwrap();
                        Some((
                            sval_item.inner("translate_new_struct")?,
                            (*array_len * array_item_width) as u64,
                        ))
                    }
                },
                SValue::Void
                | SValue::Unknown(_)
                | SValue::Bool(_)
                | SValue::F32(_)
                | SValue::I64(_)
                | SValue::Address(_)
                | SValue::Tuple(_) => None,
            };
            if let Some((src_start_ptr, size)) = mem_copy {
                trace!(
                    "{}: mem copy {} {} bytes {} into struct {}",
                    code_ref,
                    &sval,
                    size,
                    &field.field_name,
                    &struct_name
                );
                let stack_slot_address = self.builder.ins().stack_addr(
                    self.ptr_ty,
                    stack_slot,
                    Offset32::new(dst_field_def.offset as i32),
                );
                self.builder.emit_small_memory_copy(
                    self.module.target_config(),
                    stack_slot_address,
                    src_start_ptr,
                    size,
                    1,
                    1,
                    true,
                    MemFlags::new(),
                );
            } else {
                trace!(
                    "{}: copy single value or address {} {} into struct {}",
                    code_ref,
                    &sval,
                    &field.field_name,
                    &struct_name
                );
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
                            "{} struct {} expected type {} for field {} but got {} instead",
                            code_ref,
                            struct_name,
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
        Ok(SValue::Struct(struct_name.to_string(), stack_slot_address))
    }

    fn get_struct_field_location(
        &mut self,
        parts: Vec<String>,
        lhs_val: Option<SValue>,
    ) -> anyhow::Result<(StructField, Value, usize)> {
        let mut struct_name: String;
        let start: usize;
        //println!("get_struct_field_location {:?}", &parts);
        let base_struct_var_ptr = if let Some(lhs_val) = lhs_val {
            if let SValue::Struct(vstruct_name, base_struct_var_ptr) = lhs_val {
                start = 0;
                struct_name = vstruct_name;
                base_struct_var_ptr
            } else {
                anyhow::bail!("struct type not found")
            }
        } else {
            let svar = self.get_variable(&CodeRef::z(), &parts[0])?.clone();
            if let SVariable::Struct(_var_name, vstruct_name, var, _return_struct) = svar {
                let base_struct_var_ptr = self.builder.use_var(var);
                start = 1;
                struct_name = vstruct_name.to_string();
                base_struct_var_ptr
            } else {
                anyhow::bail!("struct type not found")
            }
        };

        let mut parent_struct_field = &self.env.struct_map[&struct_name].fields[&parts[start]];
        let mut offset = parent_struct_field.offset;
        if parts.len() > 2 {
            offset = 0;
            for i in start..parts.len() {
                if let ExprType::Struct(_code_ref, _name) = &parent_struct_field.expr_type {
                    parent_struct_field = &self.env.struct_map[&struct_name].fields[&parts[i]];
                    offset += parent_struct_field.offset;
                    struct_name = parent_struct_field.expr_type.to_string();
                } else {
                    break;
                }
            }
        }
        Ok(((*parent_struct_field).clone(), base_struct_var_ptr, offset))
    }

    fn get_struct_field_address(
        &mut self,
        code_ref: &CodeRef,
        parts: Vec<String>,
        lhs_val: Option<SValue>,
    ) -> anyhow::Result<(SValue, StructField)> {
        let (parent_struct_field_def, base_struct_var_ptr, offset) =
            self.get_struct_field_location(parts, lhs_val)?;
        trace!(
            "{}: get_struct_field_address\n{:?} base_struct_var_ptr {} offset {}",
            code_ref,
            &parent_struct_field_def,
            base_struct_var_ptr,
            offset
        );

        let offset_v = self.builder.ins().iconst(self.ptr_ty, offset as i64);
        let address = self.builder.ins().iadd(base_struct_var_ptr, offset_v);
        if let ExprType::Struct(code_ref, name) = &parent_struct_field_def.expr_type {
            trace!("{}: ExprType::Struct {}", code_ref, name);
            //If the struct field is a struct, return address of sub struct
            Ok((
                SValue::Struct(name.to_string(), address),
                parent_struct_field_def,
            ))
        } else if let ExprType::Array(code_ref, item_type, size_type) =
            &parent_struct_field_def.expr_type
        {
            trace!(
                "{}: ExprType::Array {} {:?}",
                code_ref,
                item_type,
                size_type
            );
            match size_type {
                ArraySizedExpr::Unsized => Ok((SValue::Address(address), parent_struct_field_def)),
                ArraySizedExpr::Sized => todo!(),
                ArraySizedExpr::Fixed(_len) => {
                    Ok((SValue::Address(address), parent_struct_field_def))
                }
            }
        } else {
            trace!("SValue::Address");
            //If the struct field is not a struct, return address of value
            Ok((SValue::Address(address), parent_struct_field_def))
        }
    }

    fn get_struct_field(
        &mut self,
        code_ref: &CodeRef,
        field_address: SValue,
        parent_struct_field_def: &StructField,
    ) -> anyhow::Result<SValue> {
        trace!(
            "{}: get_struct_field\n{:?} address {}",
            code_ref,
            &parent_struct_field_def,
            field_address
        );

        match field_address {
            SValue::Address(_) => {
                if let ExprType::Array(coderef, expr_type, size_type) =
                    &parent_struct_field_def.expr_type
                {
                    match size_type {
                        ArraySizedExpr::Unsized => (),
                        ArraySizedExpr::Sized => todo!(),
                        ArraySizedExpr::Fixed(_len) => {
                            //TODO will this have the correct size relative to the index?
                            trace!("{}: array {} is fixed in length and is stored directly in struct, returning fixed array SValue with address of array field", coderef, expr_type);
                            return Ok(SValue::Array(
                                Box::new(SValue::from(
                                    &mut self.builder,
                                    expr_type,
                                    field_address.inner("get_struct_field")?,
                                )?),
                                ArraySized::from(&mut self.builder, size_type),
                            ));
                        }
                    }
                }
                let mut val = self.builder.ins().load(
                    parent_struct_field_def
                        .expr_type
                        .cranelift_type(self.ptr_ty, true)?,
                    MemFlags::new(),
                    field_address.inner("get_struct_field")?,
                    Offset32::new(0),
                );
                if let ExprType::Bool(_code_ref) = parent_struct_field_def.expr_type {
                    let t = self.builder.ins().iconst(types::I8, 1);
                    val = self.builder.ins().icmp(IntCC::Equal, t, val)
                }

                SValue::from(&mut self.builder, &parent_struct_field_def.expr_type, val)
            }
            //TODO Currently returning struct sub fields as reference.
            //Should we copy, or should there be syntax for copy?
            SValue::Struct(_, _) => Ok(field_address),
            _ => todo!(),
        }
    }

    fn set_struct_field_at_address(
        &mut self,
        dst_address: SValue,
        set_value: SValue,
        dst_field_def: StructField,
        array_field: bool,
    ) -> anyhow::Result<()> {
        let copy_size = match &dst_field_def.expr_type {
            ExprType::Void(_)
            | ExprType::Bool(_)
            | ExprType::F32(_)
            | ExprType::I64(_)
            | ExprType::Tuple(_, _)
            | ExprType::Address(_) => None,
            ExprType::Array(_code_ref, expr_type, size_type) => match size_type {
                ArraySizedExpr::Unsized => None,
                ArraySizedExpr::Sized => todo!(),
                ArraySizedExpr::Fixed(_len) => {
                    if array_field {
                        let width = expr_type.width(self.ptr_ty, &self.env.struct_map).unwrap();
                        Some(width as u64)
                    } else {
                        Some(dst_field_def.size as u64)
                    }
                }
            },
            ExprType::Struct(_code_ref, _struct_name) => Some(dst_field_def.size as u64),
        };
        let copy_size = match &set_value {
            SValue::Bool(_) => None, //TODO Refactor
            SValue::F32(_) => None,
            SValue::I64(_) => None,
            _ => copy_size,
        };
        if let Some(copy_size) = copy_size {
            trace!(
                "{}: set_struct_field_at_address {} emit_small_memory_copy {} of size {}",
                &dst_field_def.expr_type.get_code_ref(),
                &dst_field_def.expr_type,
                dst_field_def.name,
                copy_size,
            );
            self.builder.emit_small_memory_copy(
                self.module.target_config(),
                dst_address.inner("set_struct_field_at_address")?,
                set_value.inner("set_struct_field_at_address")?,
                copy_size,
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
            trace!(
                "copy single value or address {} {} into struct",
                &set_value,
                &dst_field_def.name,
            );
            //If the struct field is not a struct, set copy of value
            self.builder.ins().store(
                MemFlags::new(),
                val,
                dst_address.inner("set_struct_field_at_address")?,
                Offset32::new(0),
            );
            Ok(())
        }
    }

    fn exec_if_start(&mut self, b_condition_value: Value) -> Block {
        let then_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // Test the if condition and conditionally branch.
        self.builder.ins().brz(b_condition_value, merge_block, &[]);
        // Fall through to then block.
        self.builder.ins().jump(then_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);

        merge_block
    }

    fn exec_if_end(&mut self, merge_block: Block) {
        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[]);
        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);
        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);
    }

    fn call_panic(&mut self, code_ref: &CodeRef, message: &String) -> anyhow::Result<()> {
        let arg_values = vec![self.translate_string(message)?];
        self.call_with_svalues(&code_ref, "panic", arg_values, None)?;
        Ok(())
    }

    fn call_with_svalues(
        &mut self,
        code_ref: &CodeRef,
        fn_name: &str,
        arg_svalues: Vec<SValue>,
        stack_slot_return: Option<(String, usize)>,
    ) -> anyhow::Result<SValue> {
        let fn_name = &fn_name.to_string();

        if !self.env.funcs.contains_key(fn_name) {
            anyhow::bail!("{} function {} not found", code_ref, fn_name)
        }
        let func = self.env.funcs[fn_name].clone();

        let inline_function_requested = match func.inline {
            InlineKind::Default => false, //TODO make default still inline if it seems worth it
            InlineKind::Never => false,
            InlineKind::Always => true,
            InlineKind::Often => true, //TODO make often not inline if it's an issue
        };

        let inline_function = inline_function_requested && !func.extern_func;

        if func.params.len() != arg_svalues.len() {
            anyhow::bail!(
                "function call to {} has {} args, but function description has {}",
                fn_name,
                arg_svalues.len(),
                func.params.len()
            )
        }

        let mut arg_values = Vec::new();

        for arg_svalue in &arg_svalues {
            arg_values.push(arg_svalue.inner("call_with_svalues")?)
        }

        if func.extern_func {
            if let Some(v) = sarus_std_lib::translate_std(
                self.module.target_config().pointer_type(),
                &mut self.builder,
                code_ref,
                fn_name,
                &arg_values,
            )? {
                return Ok(v);
            }
        }

        let mut sig = self.module.make_signature();

        let ptr_ty = self.module.target_config().pointer_type();

        let mut arg_values = Vec::from(arg_values); //in case we need to insert a val for the StructReturnSlot

        for val in arg_values.iter() {
            sig.params
                .push(AbiParam::new(self.builder.func.dfg.value_type(*val)));
        }

        let stack_slot_address = if let Some((fn_name, size)) = &stack_slot_return {
            //setup StackSlotData to be used as a StructReturnSlot
            let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
                if inline_function {
                    StackSlotKind::ExplicitSlot
                } else {
                    StackSlotKind::StructReturnSlot
                },
                *size as u32,
            ));
            //get stack address of StackSlotData
            let stack_slot_address =
                self.builder
                    .ins()
                    .stack_addr(ptr_ty, stack_slot, Offset32::new(0));
            arg_values.insert(0, stack_slot_address);
            sig.params
                .insert(0, AbiParam::special(ptr_ty, ArgumentPurpose::StructReturn));
            Some((fn_name.clone(), stack_slot_address))
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

        let res = if inline_function && !func.extern_func {
            trace!("{} inlining function {}", code_ref, &func.name);

            //It seems like this could be done once and the block could be stored, then just jumped to.
            //But, is this really the same as inlining? It seems like it should at worst have the
            //performance of a if/then branch without the branch condition (just a jump),
            //also would it be faster if we were calling all Sarus internal functions this way?

            //let func_block = self.builder.create_block();
            //self.builder.ins().jump(func_block, &[]); //&arg_values
            //self.builder.switch_to_block(func_block);
            //self.builder.seal_block(func_block);
            //for expr in &func.body {
            //    trans.translate_expr(expr)?;
            //}
            //self.builder.block_params(func_block).to_vec()

            //------------
            /*
            TODO: create alias table for func parameter and return var names:
                (this will only work for explicit var names in the func call
                 we'll also need a way to actually get these identifiers)
                a = 5
                b = 6
                c = add_and_1(a, b)

                fn add(x, y) -> (z) {
                    and_1 = 1
                    c = x + y + and_1
                }

                alias_table = [
                    "add->x" : "a"
                    "add->y" : "b"
                    "add->z" : "c"
                ]

                the alias table needs to work like a stack, removing just the ones added here
                    because this could already be in an inlined section
                    also, if an alias already existed, don't add it again, and don't remove it after
                    just make a local list here that has what needs to be removed
                no need to declare new vars for things in the alias table

                any new vars will need to have a prefix added to them.
                    It could be something that isn't allowed in normal var names. currently it's using "->"
                    "func_name1->var_name"
                    then two layers deep it would be
                    "func_name1->func_name2->var_name"
                    this is kept in self.inline_prefix

                we also have to include the prefix in the alias table so we don't reference some var 2 layers up

            */

            self.inline_prefix.push(fn_name.to_string());

            let existing_vars = self
                .inline_variables
                .keys()
                .map(|v| v.to_string())
                .collect::<HashSet<_>>();

            let inline_data = InlineVarData {
                prefix: self.inline_prefix.join("->"),
                arg_values,
            };

            declare_variables(
                &mut self.var_index,
                &mut self.builder,
                &mut self.module,
                &func.params,
                &func.returns,
                func.body.as_slice(),
                self.entry_block,
                &mut self.env,
                &mut self.inline_variables,
                &mut Some(inline_data),
            )?;

            for expr in &func.body {
                self.translate_expr(expr)?;
            }

            let mut _return = Vec::new();
            for ret in &func.returns {
                let v = self.inline_variables
                    [&format!("{}->{}", self.inline_prefix.join("->"), &ret.name)]
                    .inner();
                _return.push(self.builder.use_var(v))
            }

            self.inline_prefix.pop();

            //Remove vars from self.inline_variables that are no longer needed
            self.inline_variables
                .retain(|k, _| existing_vars.contains(k));

            _return
        } else {
            let callee = self
                .module
                .declare_function(&fn_name, Linkage::Import, &sig)
                .expect("problem declaring function");
            let local_callee = self
                .module
                .declare_func_in_func(callee, &mut self.builder.func);
            let call = self.builder.ins().call(local_callee, &arg_values);
            self.builder.inst_results(call).to_vec()
        };

        if let Some((fn_name, stack_slot_address)) = stack_slot_address {
            Ok(SValue::Struct(fn_name, stack_slot_address))
        } else if res.len() > 1 {
            Ok(SValue::Tuple(
                res.iter()
                    .zip(func.returns.iter())
                    .map(move |(v, arg)| {
                        SValue::from(&mut self.builder, &arg.expr_type, *v).unwrap()
                    })
                    .collect::<Vec<SValue>>(),
            ))
        } else if res.len() == 1 {
            let res = *res.first().unwrap();
            Ok(SValue::from(
                &mut self.builder,
                &func.returns.first().unwrap().expr_type,
                res,
            )?)
        } else {
            Ok(SValue::Void)
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
    trace!("emit_small_memory_copy size {} offset {}", size, offset);
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
    fn inner(&self) -> Variable {
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
    fn expect_f32(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::F32(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Float {}", code_ref, v, ctx),
        }
    }
    fn expect_i64(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::I64(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Int {}", code_ref, v, ctx),
        }
    }
    fn expect_bool(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Bool(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Bool {}", code_ref, v, ctx),
        }
    }
    fn expect_array(
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
    fn expect_address(&self, code_ref: &CodeRef, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Address(_, v) => Ok(*v),
            v => anyhow::bail!("{} incorrect type {} expected Address {}", code_ref, v, ctx),
        }
    }
    fn expect_struct(&self, code_ref: &CodeRef, name: &str, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Struct(varname, sname, v, _return_struct) => {
                if sname == name {
                    return Ok(*v);
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

    fn from(
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

struct InlineVarData {
    prefix: String,
    arg_values: Vec<Value>,
}

impl InlineVarData {
    fn get_name(&self, name: &str) -> String {
        format!("{}->{}", self.prefix, name)
    }
}

#[instrument(
    level = "info",
    skip(
        builder,
        module,
        params,
        returns,
        stmts,
        entry_block,
        env,
        variables,
        inline_data
    )
)]
fn declare_variables(
    index: &mut usize,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    params: &[Arg],
    returns: &[Arg],
    stmts: &[Expr],
    entry_block: Block,
    env: &mut Env,
    variables: &mut HashMap<String, SVariable>,
    inline_data: &mut Option<InlineVarData>,
) -> anyhow::Result<()> {
    let entry_block_is_offset = if returns.len() > 0 {
        if let ExprType::Struct(code_ref, struct_name) = &returns[0].expr_type {
            trace!(
                "{}: fn is returning struct {} declaring var {}",
                code_ref,
                struct_name,
                &returns[0].name
            );
            // When calling a function that will return a struct, Rust (or possibly anything using the C ABI),
            // will allocate the stack space needed for the struct that will be returned. This is allocated in
            // the callers frame, then the stack address is passed as a special argument to the first parameter
            // of the callee.
            // https://docs.wasmtime.dev/api/cranelift/prelude/enum.StackSlotKind.html#variant.StructReturnSlot
            // https://docs.wasmtime.dev/api/cranelift_codegen/ir/enum.ArgumentPurpose.html#variant.StructReturn

            let return_struct_arg = &returns[0];
            let (name, val) = if let Some(inline_data) = inline_data {
                (
                    inline_data.get_name(&return_struct_arg.name),
                    inline_data.arg_values[0],
                )
            } else {
                let return_struct_param_val = builder.block_params(entry_block)[0];
                (return_struct_arg.name.clone(), return_struct_param_val)
            };
            let var = declare_variable(
                module,
                builder,
                variables,
                index,
                name,
                &return_struct_arg.expr_type,
                true,
            )?;
            if let Some(var) = var {
                builder.def_var(var.inner(), val);
            }
            true
        } else {
            for arg in returns {
                let name = if let Some(inline_data) = inline_data {
                    inline_data.get_name(&arg.name)
                } else {
                    arg.name.clone()
                };
                declare_variable(
                    module,
                    builder,
                    variables,
                    index,
                    name,
                    &arg.expr_type,
                    false,
                )?;
            }
            false
        }
    } else {
        false
    };

    for (i, arg) in params.iter().enumerate() {
        let val = if let Some(inline_data) = inline_data {
            if entry_block_is_offset {
                inline_data.arg_values[i + 1]
            } else {
                inline_data.arg_values[i]
            }
        } else {
            if entry_block_is_offset {
                builder.block_params(entry_block)[i + 1]
            } else {
                builder.block_params(entry_block)[i]
            }
        };
        let name = if let Some(inline_data) = inline_data {
            inline_data.get_name(&arg.name)
        } else {
            arg.name.clone()
        };
        let var = declare_variable(
            module,
            builder,
            variables,
            index,
            name,
            &arg.expr_type,
            false,
        )?;
        if let Some(var) = var {
            builder.def_var(var.inner(), val);
        }
    }

    for expr in stmts {
        declare_variables_in_stmt(
            module.target_config().pointer_type(),
            builder,
            index,
            expr,
            env,
            variables,
            inline_data,
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
    variables: &mut HashMap<String, SVariable>,
    inline_data: &mut Option<InlineVarData>,
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
                            &vec![name.to_string()],
                            env,
                            variables,
                            inline_data,
                        )?;
                    }
                }
            } else {
                let mut sto_exprs = Vec::new();
                for to_expr in to_exprs.iter() {
                    if let Expr::Identifier(_code_ref, name) = to_expr {
                        sto_exprs.push(name.to_string());
                    }
                }
                declare_variable_from_expr(
                    ptr_type,
                    expr,
                    builder,
                    index,
                    &sto_exprs,
                    env,
                    variables,
                    inline_data,
                )?;
            }
        }
        Expr::IfElse(_code_ref, ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(
                    ptr_type,
                    builder,
                    index,
                    &stmt,
                    env,
                    variables,
                    inline_data,
                )?;
            }
            for stmt in else_body {
                declare_variables_in_stmt(
                    ptr_type,
                    builder,
                    index,
                    &stmt,
                    env,
                    variables,
                    inline_data,
                )?;
            }
        }
        Expr::WhileLoop(_code_ref, ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(
                    ptr_type,
                    builder,
                    index,
                    &stmt,
                    env,
                    variables,
                    inline_data,
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
    index: &mut usize,
    names: &Vec<String>,
    env: &mut Env,
    variables: &mut HashMap<String, SVariable>,
    inline_data: &mut Option<InlineVarData>,
) -> anyhow::Result<()> {
    match expr {
        Expr::IfElse(_code_ref, _condition, then_body, _else_body) => {
            //TODO make sure then & else returns match
            declare_variable_from_expr(
                ptr_type,
                then_body.last().unwrap(),
                builder,
                index,
                names,
                env,
                variables,
                inline_data,
            )?;
        }
        Expr::Assign(code_ref, ref _to_exprs, ref from_exprs) => {
            for from_expr in from_exprs.iter() {
                trace!(
                    "{} declare_variable_from_expr Expr::Assign {}",
                    code_ref,
                    from_expr,
                );
                if let Some(inline_data) = inline_data {
                    let expr_type = ExprType::of(from_expr, env, &variables, &inline_data.prefix)?;
                    let mut inline_names = Vec::new();
                    for name in names {
                        inline_names.push(inline_data.get_name(name));
                    }
                    trace!("{:?}", inline_names);
                    declare_variable_from_type(
                        ptr_type,
                        &expr_type,
                        builder,
                        index,
                        &inline_names,
                        variables,
                    )?;
                } else {
                    let expr_type = ExprType::of(from_expr, env, &variables, "")?;
                    trace!("{:?}", names);
                    declare_variable_from_type(
                        ptr_type, &expr_type, builder, index, names, variables,
                    )?;
                }
            }
        }
        expr => {
            if let Some(inline_data) = inline_data {
                let expr_type = ExprType::of(expr, env, variables, &inline_data.prefix)?;
                let mut inline_names = Vec::new();
                for name in names {
                    inline_names.push(inline_data.get_name(name));
                }
                declare_variable_from_type(ptr_type, &expr_type, builder, index, names, variables)?;
            } else {
                let expr_type = ExprType::of(expr, env, &variables, "")?;
                declare_variable_from_type(ptr_type, &expr_type, builder, index, names, variables)?;
            }
        }
    };
    Ok(())
}

fn declare_variable_from_type(
    ptr_type: Type,
    expr_type: &ExprType,
    builder: &mut FunctionBuilder,
    index: &mut usize,
    names: &Vec<String>,
    variables: &mut HashMap<String, SVariable>,
) -> anyhow::Result<()> {
    let name = names.first().unwrap();
    if name.contains(".") && !name.contains("->") {
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
                        declare_variable_from_type(
                            ptr_type,
                            expr_type,
                            builder,
                            index,
                            &vec![sname.to_string()],
                            variables,
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
                    index,
                    &vec![sname.to_string()],
                    variables,
                )?
            }
        }
        ExprType::Struct(code_ref, structname) => {
            if !variables.contains_key(name) {
                trace!("{} {} {}", code_ref, expr_type, name);
                let var = Variable::new(*index);
                variables.insert(
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
    arg_name: String,
    expr_type: &ExprType,
    return_struct: bool,
) -> anyhow::Result<Option<SVariable>> {
    let ptr_ty = module.target_config().pointer_type();
    if !variables.contains_key(&arg_name) {
        trace!("declaring var {}", arg_name);
        let (var, ty) = match expr_type {
            ExprType::F32(_code_ref) => (
                SVariable::F32(arg_name.clone(), Variable::new(*index)),
                types::F32,
            ),
            ExprType::I64(_code_ref) => (
                SVariable::I64(arg_name.clone(), Variable::new(*index)),
                types::I64,
            ),
            ExprType::Array(_code_ref, ty, size_type) => (
                SVariable::Array(
                    Box::new(SVariable::from(
                        builder,
                        ty,
                        arg_name.clone(),
                        Variable::new(*index),
                    )?),
                    ArraySized::from(builder, size_type),
                ),
                ptr_ty,
            ),
            ExprType::Address(_code_ref) => (
                SVariable::Address(arg_name.clone(), Variable::new(*index)),
                ptr_ty,
            ),
            ExprType::Void(_code_ref) => return Ok(None),
            ExprType::Bool(_code_ref) => (
                SVariable::Bool(arg_name.clone(), Variable::new(*index)),
                types::B1,
            ),
            ExprType::Tuple(_code_ref, _) => return Ok(None), //anyhow::bail!("single variable tuple not supported"),
            ExprType::Struct(_code_ref, structname) => (
                SVariable::Struct(
                    arg_name.clone(),
                    structname.to_string(),
                    Variable::new(*index),
                    return_struct,
                ),
                ptr_ty,
            ),
        };
        variables.insert(arg_name.clone(), var.clone());
        builder.declare_var(var.inner(), ty);
        *index += 1;
        Ok(Some(var))
    } else {
        trace!(
            "variables already contains key {} no need to declare",
            arg_name
        );
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
        trace!("determine size of struct {}", struct_name);
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
            trace!(
                "struct {} field {} with size {} \t",
                struct_name,
                field.name,
                field_size
            );

            if i < fields_def.len() - 1 {
                //repr(C) alignment see memoffset crate

                // pad based on size of next non struct/array field,
                // or largest field if next item is struct/array of structs
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
                if padding > 0 {
                    trace!("padding added for next field: {}", padding);
                }
            }
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
            if padding > 0 {
                trace!("{} padding added at end of struct", padding);
            }
        }

        trace!("struct {} final size {}", struct_name, struct_size);
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
    struct_map: &HashMap<String, StructDef>,
    ptr_type: types::Type,
    max_base_field: bool,
) -> anyhow::Result<(usize, bool)> {
    Ok(match expr_type {
        ExprType::Struct(_code_ref, name) => {
            if max_base_field {
                (
                    get_largest_field_size(0, &expr_type, struct_map, ptr_type)?,
                    true,
                )
            } else {
                (struct_map[&name.to_string()].size, true)
            }
        }
        ExprType::Array(_code_ref, expr_type, size_type) => match size_type {
            ArraySizedExpr::Unsized => (ptr_type.bytes() as usize, false),
            ArraySizedExpr::Sized => todo!(),
            ArraySizedExpr::Fixed(len) => {
                if max_base_field {
                    get_field_size(expr_type, struct_map, ptr_type, max_base_field)?
                } else {
                    let (size, is_struct) =
                        get_field_size(expr_type, struct_map, ptr_type, max_base_field)?;
                    (size * len, is_struct)
                }
            }
        },
        _ => (
            (expr_type
                .width(ptr_type, struct_map)
                .unwrap_or(ptr_type.bytes() as usize) as usize),
            false,
        ),
    })
}

fn get_largest_field_size(
    largest: usize,
    expr_type: &ExprType,
    struct_map: &HashMap<String, StructDef>,
    ptr_type: types::Type,
) -> anyhow::Result<usize> {
    let mut largest = largest;
    match expr_type {
        ExprType::Struct(_code_ref, name) => {
            for (_name, field) in &struct_map[&name.to_string()].fields {
                let size = get_largest_field_size(largest, &field.expr_type, struct_map, ptr_type)?;
                if size > largest {
                    largest = size;
                }
            }
        }
        _ => {
            let size = expr_type
                .width(ptr_type, struct_map)
                .unwrap_or(ptr_type.bytes() as usize);
            if size > largest {
                largest = size;
            }
        }
    };
    Ok(largest)
}

fn can_insert_into_map(
    struct_name: &str,
    field_name: &str,
    expr_type: &ExprType,
    in_structs: &HashMap<String, &Struct>,
    structs_order: &Vec<String>,
    can_insert: bool,
) -> anyhow::Result<bool> {
    // if this expr's dependencies (if it has any) are already in the
    // structs_order, then we can safely add this one
    Ok(match expr_type {
        ExprType::Void(_code_ref)
        | ExprType::Bool(_code_ref)
        | ExprType::F32(_code_ref)
        | ExprType::I64(_code_ref)
        | ExprType::Address(_code_ref)
        | ExprType::Tuple(_code_ref, _) => can_insert,
        ExprType::Struct(code_ref, field_struct_name) => {
            if !in_structs.contains_key(&field_struct_name.to_string()) {
                anyhow::bail!(
                    "{} Can't find Struct {} referenced in Struct {} field {}",
                    code_ref,
                    field_struct_name,
                    struct_name,
                    field_name
                )
            }
            if structs_order.contains(&field_struct_name.to_string()) {
                can_insert
            } else {
                false
            }
        }
        ExprType::Array(_code_ref, expr_type, _size_type) => can_insert_into_map(
            struct_name,
            field_name,
            expr_type,
            in_structs,
            structs_order,
            can_insert,
        )?,
    })
}

fn order_structs(in_structs: &HashMap<String, &Struct>) -> anyhow::Result<Vec<String>> {
    // find order of structs based on dependency hierarchy
    let mut structs_order = Vec::new();
    let mut last_structs_len = 0usize;
    while structs_order.len() < in_structs.len() {
        for (name, struc) in in_structs {
            let mut can_insert = true;
            for field in &struc.fields {
                can_insert = can_insert_into_map(
                    &struc.name,
                    &field.name,
                    &field.expr_type,
                    in_structs,
                    &structs_order,
                    can_insert,
                )?
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
