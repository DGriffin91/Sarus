use crate::frontend::*;
use crate::function_translator::*;
use crate::sarus_std_lib;
use crate::sarus_std_lib::SConstant;
pub use crate::structs::*;
use crate::validator::ExprType;
pub use crate::variables::*;
use cranelift::codegen::ir::ArgumentPurpose;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
use std::path::PathBuf;
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

    // don't emit call to __cranelift_probestack. This is an issue for larger stack allocated arrays
    flag_builder.set("enable_probestack", "false").unwrap();

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
        }
    }

    /// Compile the ast into machine code.
    #[instrument(level = "info", skip(self, prog, file_index_table))]
    pub fn translate(
        &mut self,
        prog: Vec<Declaration>,
        file_index_table: Option<Vec<PathBuf>>,
    ) -> anyhow::Result<()> {
        info!("--------------- translate ---------------");

        let mut prog = prog;

        let struct_map = create_struct_map(&prog, self.module.target_config().pointer_type())?;
        let constant_vars = sarus_std_lib::get_constants(&struct_map);

        let mut funcs = HashMap::new();
        let mut inline_closures: HashMap<String, HashMap<String, Closure>> = HashMap::new();

        for decl in prog.iter_mut() {
            match decl {
                Declaration::Function(func) => {
                    funcs.insert(func.name.clone(), func.clone());
                    setup_inline_closures(&func.name, &func.body, &mut inline_closures);
                    if let InlineKind::Always = func.inline {
                    } else {
                        for param in &func.params {
                            if param.closure_arg.is_some() {
                                anyhow::bail!("function {} takes a closure in parameter {} but is not declared as inline_always", func.name, param.name)
                            }
                        }
                    }
                }
                _ => continue,
            }
        }

        for d in prog.clone() {
            match d {
                Declaration::Function(func) => {
                    if func.extern_func {
                        // Don't compile the contents of std func, it will be empty
                        trace!(
                            "Function {} is an external function, skipping codegen",
                            func.sig_string()?
                        );
                        continue;
                    }
                    if let InlineKind::Always = func.inline {
                        // Don't compile the contents of Inline::Always func
                        // it should always be inlined
                        trace!(
                            "Function {} is Inline::Always, skipping codegen",
                            func.sig_string()?
                        );
                        continue;
                    }

                    // Then, translate the AST nodes into Cranelift IR.
                    self.codegen(
                        &func,
                        funcs.to_owned(),
                        &struct_map,
                        &constant_vars,
                        &file_index_table,
                        &inline_closures,
                    )?;
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
                SConstant::I64(n) => self.create_data(name, (*n).to_ne_bytes().to_vec())?,
                SConstant::F32(n) => self.create_data(name, (*n).to_ne_bytes().to_vec())?,
                SConstant::Bool(n) => self.create_data(name, (*n as i8).to_ne_bytes().to_vec())?,
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
    #[instrument(
        level = "info",
        skip(
            self,
            func,
            funcs,
            struct_map,
            constant_vars,
            file_index_table,
            inline_closures
        )
    )]
    fn codegen(
        &mut self,
        func: &Function,
        funcs: HashMap<String, Function>,
        struct_map: &HashMap<String, StructDef>,
        constant_vars: &HashMap<String, SConstant>,
        file_index_table: &Option<Vec<PathBuf>>,
        inline_closures: &HashMap<String, HashMap<String, Closure>>,
    ) -> anyhow::Result<()> {
        info!("{}", func.sig_string()?);
        let ptr_ty = self.module.target_config().pointer_type();

        if !func.returns.is_empty() {
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

        let env = Env {
            constant_vars: constant_vars.clone(),
            struct_map: struct_map.clone(),
            ptr_ty,
            funcs,
            file_idx: file_index_table.clone(),
            inline_closures: inline_closures.clone(),
            temp_inline_closures: HashMap::new(),
        };

        //println!("declare_variables {}", func.name);

        let mut variables = HashMap::new();

        let mut var_index = 0;
        declare_param_and_return_variables(
            &mut var_index,
            &mut builder,
            &mut self.module,
            func,
            entry_block,
            &mut variables,
            &None,
        )?;

        //println!("FunctionTranslator {}", func.name);
        let ptr_ty = self.module.target_config().pointer_type();

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            module: &mut self.module,
            ptr_ty,
            env,
            func_stack: vec![func.clone()],
            entry_block,
            var_index,
            variables: vec![variables],
        };
        for expr in &func.body {
            trans.translate_expr(expr)?;
        }

        // Set up the return variable of the function. Above, we declared a
        // variable to hold the return value. Here, we just do a use of that
        // variable.
        let mut return_values = Vec::new();
        for ret in func.returns.iter() {
            let return_variable = trans.variables.last().unwrap().get(&ret.name).unwrap();
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
}

pub struct Env {
    pub constant_vars: HashMap<String, SConstant>,
    pub struct_map: HashMap<String, StructDef>,
    pub ptr_ty: types::Type,
    pub funcs: HashMap<String, Function>,
    pub file_idx: Option<Vec<PathBuf>>,

    // inline closures are only available in the scope of this function
    // are not stored in a variable
    // can only be called directly, or passed to other functions that are inlined
    // can't be in a branch, because it's all compile time
    // since it's all inlined can safely act like a closure
    // These are stored by inline_closures[containing func name][closure name]
    pub inline_closures: HashMap<String, HashMap<String, Closure>>,

    // temp inline closures are ones that are passed in as arguments
    // functions that are always::inline can take closures as parameters
    // the closure gets cloned into temp_inline_closures with an alias
    // to the parameter name in the function that the closure is being
    // passed to. These are initialized when the function is inlined
    // and cleaned up afterward.
    // TODO make recursively inlining an error
    // These are stored by temp_inline_closures[containing func name][closure name]
    pub temp_inline_closures: HashMap<String, HashMap<String, Closure>>,
}

impl Env {
    pub fn get_inline_closure(
        &self,
        callee_func_name: &str,
        fn_name: &str,
    ) -> Option<(Closure, bool)> {
        if let Some(closures) = self.inline_closures.get(callee_func_name) {
            if let Some(closure) = closures.get(fn_name) {
                return Some((closure.clone(), false));
            }
        };
        if let Some(closures) = self.temp_inline_closures.get(callee_func_name) {
            if let Some(closure) = closures.get(fn_name) {
                return Some((closure.clone(), true));
            }
        };
        None
    }
}
