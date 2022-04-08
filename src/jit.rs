use crate::frontend::*;
use crate::function_translator::*;
use crate::sarus_std_lib;
use crate::sarus_std_lib::SConstant;
pub use crate::structs::*;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
pub use crate::variables::*;
use cranelift::codegen::ir::ArgumentPurpose;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
use std::collections::HashSet;
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

    // An optional larger stack for larger allocations
    // Essentially a lifo arena
    // Size determined at compile time, then used globally at runtime
    // Must be accessed by only one thread at a time
    deep_stack: Option<Heap>,

    // pointer to address at the top of the stack in the heap
    // at run time, when something is added to the deep stack, this is incremented
    // when something is removed from the deep stack, this is decremented
    deep_stack_pointer: Option<*mut u64>,

    // pointer to start of deep stack, this is where we keep the last address we
    // will go back to when leaving a stack frame
    bottom_of_deep_stack_pointer: Option<*mut u64>,

    /*
    TODO
        to determine size of auto deep stack:
        for each function compute max size of deep stack used
        in each function also take into account sub calls
        the functions will have to be in order for this to work
        the deep stack should be the size of the largest function's
        maximum stack size
    CURRENTLY
        this is just the sum of all functions max stack size
    */
    total_max_deep_stack_size: usize,
    use_deep_stack: bool,
}

impl Default for JIT {
    fn default() -> Self {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
        let module = JITModule::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            clif: HashMap::new(),
            use_deep_stack: true,
            deep_stack: None,
            deep_stack_pointer: None,
            bottom_of_deep_stack_pointer: None,
            total_max_deep_stack_size: 0,
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
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    JITBuilder::with_isa(isa, cranelift_module::default_libcall_names())
}

impl JIT {
    pub fn from(jit_builder: JITBuilder, use_deep_stack: bool) -> Self {
        let module = JITModule::new(jit_builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            clif: HashMap::new(),
            use_deep_stack,
            deep_stack: None,
            deep_stack_pointer: None,
            bottom_of_deep_stack_pointer: None,
            total_max_deep_stack_size: 0,
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

        //let _ = order_funcs(&funcs);

        for func in funcs.values() {
            if func.extern_func {
                // Don't compile the contents of std func, it will be empty
                trace!(
                    "Function {} is an external function, skipping codegen.",
                    func.sig_string()?,
                );
                //TODO self.module.lookup_symbol(func.name);
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
                func,
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
                .map_err(|e| anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e))?;

            trace!("cranelift func id is {}", id);
            // Define the function to jit. This finishes compilation, although
            // there may be outstanding relocations to perform. Currently, jit
            // cannot finish relocations until all functions to be called are
            // defined. For this toy demo for now, we'll just finalize the
            // function below.
            self.module
                .define_function(id, &mut self.ctx)
                .map_err(|e| anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e))?;

            // Now that compilation is finished, we can clear out the context state.
            self.module.clear_context(&mut self.ctx);

            // Finalize the functions which we just defined, which resolves any
            // outstanding relocations (patching in addresses, now that they're
            // available).
            self.module.finalize_definitions();
        }

        if self.use_deep_stack {
            trace!(
                "total_max_deep_stack_size {}",
                self.total_max_deep_stack_size
            );
            let deep_stack = Heap::new(
                self.total_max_deep_stack_size
                    + 1024 * self.module.target_config().pointer_bytes() as usize, //extra space for deep stack checkpoints
            )
            .unwrap();
            let deep_stack_ptr = deep_stack.get_ptr() as u64;
            let deep_stack_pointer = Box::into_raw(Box::new(deep_stack_ptr));
            let bottom_of_deep_stack_pointer = Box::into_raw(Box::new(deep_stack_ptr));
            self.create_data(
                "__DEEP_STACK_CURSOR__",
                (deep_stack_pointer as u64).to_ne_bytes().to_vec(),
            )?;
            self.deep_stack = Some(deep_stack);
            self.deep_stack_pointer = Some(deep_stack_pointer);

            self.create_data(
                "__DEEP_STACK_BOTTOM__",
                (bottom_of_deep_stack_pointer as u64).to_ne_bytes().to_vec(),
            )?;
            self.bottom_of_deep_stack_pointer = Some(bottom_of_deep_stack_pointer);

            // write checkpoints to the deep stack itself
            //             &0              &1      &28
            // Deep Stack: latest func_top data... last(1) data... last(28) data...
            //                                    ^---------------v
            //             ^----------------------v

            // When entering a new stack write the current location to latest
            // write the location of the last latest to the current cursor location on the deep stack
            // if entering a function, write current cursor location also to func_top

            // something also needs to hold the cursor position for the start
            // of the current function to be able to deal with early returns form
            // inside a while loop or other stack frame
        }

        for (name, val) in constant_vars.iter() {
            match val {
                SConstant::Address(n) => self.create_data(name, (*n).to_ne_bytes().to_vec())?,
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
            match &func.returns[0].expr_type {
                ExprType::Struct(_, ..)
                | ExprType::Array(_, _, ArraySizedExpr::Fixed(..) | ArraySizedExpr::Slice) => {
                    if func.returns.len() > 1 {
                        anyhow::bail!(
                            "If returning a fixed length array, slice, or struct, only 1 return value is currently supported"
                        )
                    }
                    self.ctx
                        .func
                        .signature
                        .params
                        .push(AbiParam::special(ptr_ty, ArgumentPurpose::StructReturn));
                }
                _ => {
                    for ret_arg in &func.returns {
                        self.ctx.func.signature.returns.push(AbiParam::new(
                            ret_arg.expr_type.cranelift_type(
                                self.module.target_config().pointer_type(),
                                false,
                            )?,
                        ));
                    }
                }
            }
        }

        for p in &func.params {
            self.ctx.func.signature.params.push({
                match &p.expr_type {
                    ExprType::F32(_code_ref) => AbiParam::new(types::F32),
                    ExprType::I64(_code_ref) => AbiParam::new(types::I64),
                    ExprType::U8(_code_ref) => AbiParam::new(types::I8),
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
        let mut per_scope_vars = vec![vec![HashSet::new()]];

        let mut var_index = 0;
        let return_var_names = declare_param_and_return_variables(
            &mut var_index,
            &mut builder,
            &mut self.module,
            func,
            entry_block,
            &mut variables,
            per_scope_vars.last_mut().unwrap().last_mut().unwrap(),
            &None,
        )?;

        //println!("FunctionTranslator {}", func.name);
        let ptr_ty = self.module.target_config().pointer_type();
        let ptr_width = ptr_ty.bytes() as i64;

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            builder,
            module: &mut self.module,
            ptr_ty,
            ptr_width,
            env,
            func_stack: vec![func.clone()],
            unassigned_return_var_names: vec![return_var_names],
            entry_block,
            var_index,
            variables: vec![variables],
            per_scope_vars,
            expr_depth: 0,
            deep_stack_widths: Vec::new(),
            use_deep_stack: self.use_deep_stack,
            max_deep_stack_size: 0,
            while_exit_blocks: Vec::new(),
            while_continue_blocks: Vec::new(),
            deep_stack_debug: false,
        };
        if self.use_deep_stack {
            trans.deep_stack_init();
            trans.add_deep_stack_frame(true);
        }
        for expr in &func.body {
            trans.translate_expr(expr)?;
        }

        trans.return_(false)?;
        if self.use_deep_stack {
            self.total_max_deep_stack_size += trans.max_deep_stack_size;
        }

        //Keep clif around for later debug/print
        self.clif.insert(
            func.name.to_string(),
            trans.builder.func.display().to_string(),
        );
        trace!("{}", trans.builder.func.display().to_string());

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

fn find_calls(expr: &Expr, calls: &mut Vec<String>) {
    match expr {
        Expr::LiteralFloat { .. }
        | Expr::LiteralInt { .. }
        | Expr::LiteralU8 { .. }
        | Expr::LiteralBool { .. }
        | Expr::LiteralString { .. }
        | Expr::Break { .. }
        | Expr::Continue { .. }
        | Expr::Return { .. }
        | Expr::Declaration { .. }
        | Expr::Identifier { .. }
        | Expr::GlobalDataAddr { .. } => (),
        Expr::LiteralArray { exprs, .. } => {
            for expr in exprs {
                find_calls(expr, calls)
            }
        }
        Expr::Binop { lhs, rhs, .. } => {
            find_calls(lhs, calls);
            find_calls(rhs, calls)
        }
        Expr::Unaryop { expr, .. } => find_calls(expr, calls),
        Expr::Compare { lhs, rhs, .. } => {
            find_calls(lhs, calls);
            find_calls(rhs, calls)
        }
        Expr::IfThen {
            condition,
            then_body,
            ..
        } => {
            find_calls(condition, calls);
            for e in then_body {
                find_calls(e, calls)
            }
        }
        Expr::IfElse {
            condition,
            then_body,
            else_body,
            ..
        } => {
            find_calls(condition, calls);
            for e in then_body {
                find_calls(e, calls)
            }
            for e in else_body {
                find_calls(e, calls)
            }
        }
        Expr::IfThenElseIf { expr_bodies, .. } => {
            for (a, b) in expr_bodies {
                find_calls(a, calls);
                for e in b {
                    find_calls(e, calls)
                }
            }
        }
        Expr::IfThenElseIfElse {
            expr_bodies,
            else_body,
            ..
        } => {
            for (a, b) in expr_bodies {
                find_calls(a, calls);
                for e in b {
                    find_calls(e, calls)
                }
            }
            for e in else_body {
                find_calls(e, calls)
            }
        }
        Expr::Assign {
            to_exprs,
            from_exprs,
            ..
        } => {
            for e in to_exprs {
                find_calls(e, calls)
            }
            for e in from_exprs {
                find_calls(e, calls)
            }
        }
        Expr::NewStruct { fields, .. } => {
            for e in fields {
                find_calls(&e.expr, calls)
            }
        }
        Expr::Match {
            expr_arg, fields, ..
        } => {
            find_calls(&expr_arg, calls);
            for e in fields {
                find_calls(&e.expr, calls)
            }
        }
        Expr::WhileLoop {
            condition,
            iter_body,
            loop_body,
            ..
        } => {
            find_calls(condition, calls);
            if let Some(b) = iter_body {
                for e in b {
                    find_calls(e, calls)
                }
            }
            for e in loop_body {
                find_calls(e, calls)
            }
        }
        Expr::Block { block, .. } => {
            for e in block {
                find_calls(e, calls)
            }
        }
        Expr::Call { args, .. } => {
            for e in args {
                find_calls(e, calls)
            }
        }
        Expr::Parentheses { expr, .. } => find_calls(expr, calls),
        Expr::ArrayAccess { expr, idx_expr, .. } => {
            find_calls(expr, calls);
            find_calls(idx_expr, calls)
        }
    }
}

fn order_funcs(funcs: &HashMap<String, Function>) -> Vec<String> {
    let mut func_calls: HashMap<String, Vec<String>> = HashMap::new(); //func_name: its calls
    for (func_name, func) in funcs {
        let mut calls = Vec::new();
        for expr in &func.body {
            find_calls(expr, &mut calls)
        }
        func_calls.insert(func_name.to_string(), calls);
    }
    dbg!(func_calls);
    todo!()
}

//TODO move to its own file if we use this
use std::alloc::{alloc, dealloc, Layout};

#[derive(Clone)]
pub struct Heap {
    ptr: *mut u8,
    layout: Layout,
}

impl Drop for Heap {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

impl Heap {
    pub fn new(size: usize) -> anyhow::Result<Self> {
        let layout = Layout::from_size_align(size, 8)?;
        let ptr = unsafe { alloc(layout) };
        Ok(Heap { ptr, layout })
    }

    pub fn get_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}
