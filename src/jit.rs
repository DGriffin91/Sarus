use crate::frontend::*;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
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
        }
    }
}

impl JIT {
    /// Compile a string in the toy language into machine code.
    pub fn translate(&mut self, prog: Vec<Declaration>) -> anyhow::Result<()> {
        let mut return_counts = HashMap::new();
        for Declaration { name, returns, .. } in &prog {
            return_counts.insert(name.to_string(), returns.len());
        }

        // First, parse the string, producing AST nodes.
        for Declaration {
            name,
            params,
            returns,
            body: stmts,
        } in prog
        {
            ////println!(
            ////    "name {:?}, params {:?}, the_return {:?}",
            ////    &name, &params, &the_return
            ////);
            //// Then, translate the AST nodes into Cranelift IR.
            self.codegen(params, returns, stmts, return_counts.to_owned())?;
            // Next, declare the function to jit. Functions must be declared
            // before they can be called, or defined.
            //
            // TODO: This may be an area where the API should be streamlined; should
            // we have a version of `declare_function` that automatically declares
            // the function?
            let id = self
                .module
                .declare_function(&name, Linkage::Export, &self.ctx.func.signature)
                .map_err(|e| anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e))?;

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
                .map_err(|e| anyhow::anyhow!("{}:{}:{} {:?}", file!(), line!(), column!(), e))?;

            // Now that compilation is finished, we can clear out the context state.
            self.module.clear_context(&mut self.ctx);

            // Finalize the functions which we just defined, which resolves any
            // outstanding relocations (patching in addresses, now that they're
            // available).
            self.module.finalize_definitions();
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
        // TODO: Can we move the unsafe into cranelift?
        Ok(unsafe { slice::from_raw_parts(buffer.0, buffer.1) })
    }

    // Translate from toy-language AST nodes into Cranelift IR.
    fn codegen(
        &mut self,
        params: Vec<String>,
        returns: Vec<String>,
        stmts: Vec<Expr>,
        return_counts: HashMap<String, usize>,
    ) -> anyhow::Result<()> {
        // Our toy language currently only supports I64 values, though Cranelift
        // supports other types.
        let float = types::F64; //self.module.target_config().pointer_type();

        for _p in &params {
            self.ctx.func.signature.params.push(AbiParam::new(float));
        }

        for _p in &returns {
            self.ctx.func.signature.returns.push(AbiParam::new(float));
        }

        // Create the builder to build a function.
        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);

        // Create the entry block, to start emitting code in.
        let entry_block = builder.create_block();

        // Since this is the entry block, add block parameters corresponding to
        // the function's parameters.
        //
        // TODO: Streamline the API here.
        builder.append_block_params_for_function_params(entry_block);

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);

        // And, tell the builder that this block will have no further
        // predecessors. Since it's the entry block, it won't have any
        // predecessors.
        builder.seal_block(entry_block);

        // The toy language allows variables to be declared implicitly.
        // Walk the AST and declare all implicitly-declared variables.
        let variables =
            declare_variables(float, &mut builder, &params, &returns, &stmts, entry_block);

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            float,
            builder,
            variables,
            return_counts,
            module: &mut self.module,
        };
        for expr in &stmts {
            trans.translate_expr(expr);
        }

        // Set up the return variable of the function. Above, we declared a
        // variable to hold the return value. Here, we just do a use of that
        // variable.
        let return_values: Vec<Value> = returns
            .iter()
            .map(|ret| {
                let return_variable = trans.variables.get(ret).unwrap();
                trans.builder.use_var(*return_variable)
            })
            .collect();

        // Emit the return instruction.
        trans.builder.ins().return_(&return_values);

        // Tell the builder we're done with this function.
        trans.builder.finalize();
        Ok(())
    }
}

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionTranslator<'a> {
    float: types::Type,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, Variable>,
    return_counts: HashMap<String, usize>,
    module: &'a mut JITModule,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &Expr) -> Vec<Value> {
        match expr {
            Expr::Literal(literal) => {
                vec![self.builder.ins().f64const::<f64>(literal.parse().unwrap())]
            }
            Expr::Binop(op, lhs, rhs) => self.translate_binop(*op, lhs, rhs),
            Expr::Compare(cmp, lhs, rhs) => self.translate_cmp(*cmp, lhs, rhs),
            Expr::Call(name, args) => self.translate_call(name, args),
            Expr::GlobalDataAddr(name) => vec![self.translate_global_data_addr(name)],
            Expr::Identifier(name) => {
                // `use_var` is used to read the value of a variable.
                let variable = self
                    .variables
                    .get(name)
                    .copied()
                    .expect("variable not defined");
                vec![self.builder.use_var(variable)]
            }
            Expr::Assign(names, expr) => self.translate_assign(names, expr),
            Expr::AssignOp(op, lhs, rhs) => self.translate_math_assign(*op, lhs, rhs),
            Expr::IfThen(condition, then_body) => {
                vec![self.translate_if_then(condition, then_body)]
            }
            Expr::IfElse(condition, then_body, else_body) => {
                self.translate_if_else(condition, then_body, else_body)
            }
            Expr::WhileLoop(condition, loop_body) => {
                vec![self.translate_while_loop(condition, loop_body)]
            }
            Expr::Block(b) => {
                vec![b
                    .into_iter()
                    .map(|e| self.translate_expr(e))
                    .last()
                    .and_then(|v| v.first().cloned())
                    .unwrap()]
            }
            Expr::Bool(b) => vec![self.builder.ins().bconst(types::B1, *b)],
        }
    }

    fn translate_binop(&mut self, op: Binop, lhs: &Expr, rhs: &Expr) -> Vec<Value> {
        let lhs = *self.translate_expr(lhs).first().unwrap();
        let rhs = *self.translate_expr(rhs).first().unwrap();
        match op {
            Binop::Add => vec![self.builder.ins().fadd(lhs, rhs)],
            Binop::Sub => vec![self.builder.ins().fsub(lhs, rhs)],
            Binop::Mul => vec![self.builder.ins().fmul(lhs, rhs)],
            Binop::Div => vec![self.builder.ins().fdiv(lhs, rhs)],
        }
    }

    fn translate_cmp(&mut self, cmp: Cmp, lhs: &Expr, rhs: &Expr) -> Vec<Value> {
        let icmp = match cmp {
            Cmp::Eq => FloatCC::Equal,
            Cmp::Ne => FloatCC::NotEqual,
            Cmp::Lt => FloatCC::LessThan,
            Cmp::Le => FloatCC::LessThanOrEqual,
            Cmp::Gt => FloatCC::GreaterThan,
            Cmp::Ge => FloatCC::GreaterThanOrEqual,
        };
        vec![self.translate_icmp(icmp, lhs, rhs)]
    }

    fn translate_assign(&mut self, names: &[String], expr: &[Expr]) -> Vec<Value> {
        // `def_var` is used to write the value of a variable. Note that
        // variables can have multiple definitions. Cranelift will
        // convert them into SSA form for itself automatically.

        //if there are the same number of expressions as there are names
        //eg: `a, b = b, a` then use the first output of each expression
        //But if there is not, use the output of the first expression:
        //eg: `a, b = func_that_outputs_2_floats(1.0)`
        if names.len() == expr.len() {
            let mut values = Vec::new();
            for (i, name) in names.iter().enumerate() {
                values.push(*self.translate_expr(expr.get(i).unwrap()).first().unwrap());
                let variable = self.variables.get(name).unwrap();
                self.builder.def_var(*variable, *values.last().unwrap());
            }
            values
        } else {
            let new_value = self.translate_expr(expr.first().unwrap());
            for (i, name) in names.iter().enumerate() {
                let variable = self.variables.get(name).unwrap();
                self.builder.def_var(*variable, new_value[i]);
            }
            new_value
        }
    }

    fn translate_math_assign(&mut self, op: Binop, name: &str, expr: &Expr) -> Vec<Value> {
        let new_value = *self.translate_expr(expr).first().unwrap();
        let orig_variable = self.variables.get(&*name).unwrap();
        let orig_value = self.builder.use_var(*orig_variable);
        let added_val = match op {
            Binop::Add => self.builder.ins().fadd(orig_value, new_value),
            Binop::Sub => self.builder.ins().fsub(orig_value, new_value),
            Binop::Mul => self.builder.ins().fmul(orig_value, new_value),
            Binop::Div => self.builder.ins().fdiv(orig_value, new_value),
        };
        self.builder.def_var(*orig_variable, added_val);
        vec![added_val]
    }

    fn translate_icmp(&mut self, cmp: FloatCC, lhs: &Expr, rhs: &Expr) -> Value {
        let lhs = *self.translate_expr(lhs).first().unwrap();
        let rhs = *self.translate_expr(rhs).first().unwrap();
        self.builder.ins().fcmp(cmp, lhs, rhs)
    }

    fn translate_if_then(&mut self, condition: &Expr, then_body: &[Expr]) -> Value {
        let b_condition_value = *self.translate_expr(condition).first().unwrap();

        let then_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // Test the if condition and conditionally branch.
        self.builder.ins().brz(b_condition_value, merge_block, &[]);
        // Fall through to then block.
        self.builder.ins().jump(then_block, &[]);

        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        for expr in then_body {
            self.translate_expr(expr).first().unwrap();
        }

        // Jump to the merge block, passing it the block return value.
        self.builder.ins().jump(merge_block, &[]);
        // Switch to the merge block for subsequent statements.
        self.builder.switch_to_block(merge_block);
        // We've now seen all the predecessors of the merge block.
        self.builder.seal_block(merge_block);
        b_condition_value
    }

    fn translate_if_else(
        &mut self,
        condition: &Expr,
        then_body: &[Expr],
        else_body: &[Expr],
    ) -> Vec<Value> {
        let b_condition_value = *self.translate_expr(condition).first().unwrap();

        let then_block = self.builder.create_block();
        let else_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        let then_return = self.translate_expr(then_body.last().unwrap());
        let else_return = self.translate_expr(else_body.last().unwrap());

        for _ in 0..then_return.len() {
            // If-else constructs in the toy language have a return value.
            // In traditional SSA form, this would produce a PHI between
            // the then and else bodies. Cranelift uses block parameters,
            // so set up a parameter in the merge block, and we'll pass
            // the return values to it from the branches.
            self.builder.append_block_param(merge_block, self.float);
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

        phi.to_vec()
    }

    fn translate_while_loop(&mut self, condition: &Expr, loop_body: &[Expr]) -> Value {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);
        self.builder.switch_to_block(header_block);

        let condition_value = *self.translate_expr(condition).first().unwrap();
        //Convert condition from float to bool
        let zero = self.builder.ins().f64const(0.0);
        let b_condition_value = self
            .builder
            .ins()
            .fcmp(FloatCC::NotEqual, condition_value, zero);

        self.builder.ins().brz(b_condition_value, exit_block, &[]);
        self.builder.ins().jump(body_block, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        for expr in loop_body {
            self.translate_expr(expr);
        }
        self.builder.ins().jump(header_block, &[]);

        self.builder.switch_to_block(exit_block);

        // We've reached the bottom of the loop, so there will be no
        // more backedges to the header to exits to the bottom.
        self.builder.seal_block(header_block);
        self.builder.seal_block(exit_block);

        // Just return 0 for now.
        self.builder.ins().f64const(0.0)
    }

    fn translate_call(&mut self, name: &str, args: &[Expr]) -> Vec<Value> {
        let mut sig = self.module.make_signature();

        // Add a parameter for each argument.
        for _ in args {
            sig.params.push(AbiParam::new(self.float));
        }

        for _ in 0..self.return_counts[name] {
            sig.returns.push(AbiParam::new(self.float));
        }

        // TODO: Streamline the API here?
        let callee = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .expect("problem declaring function");
        let local_callee = self
            .module
            .declare_func_in_func(callee, &mut self.builder.func);

        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(*self.translate_expr(arg).first().unwrap())
        }
        let call = self.builder.ins().call(local_callee, &arg_values);
        self.builder.inst_results(call).to_vec()
    }

    fn translate_global_data_addr(&mut self, name: &str) -> Value {
        let sym = self
            .module
            .declare_data(&name, Linkage::Export, true, false)
            .expect("problem declaring data object");
        let local_id = self
            .module
            .declare_data_in_func(sym, &mut self.builder.func);

        let pointer = self.module.target_config().pointer_type();
        self.builder.ins().symbol_value(pointer, local_id)
    }
}

fn declare_variables(
    float: types::Type,
    builder: &mut FunctionBuilder,
    params: &[String],
    returns: &[String],
    stmts: &[Expr],
    entry_block: Block,
) -> HashMap<String, Variable> {
    let mut variables = HashMap::new();
    let mut index = 0;

    for (i, name) in params.iter().enumerate() {
        // TODO: cranelift_frontend should really have an API to make it easy to set
        // up param variables.
        let val = builder.block_params(entry_block)[i];
        let var = declare_variable(float, builder, &mut variables, &mut index, name);
        builder.def_var(var, val);
    }

    for name in returns {
        let zero = builder.ins().f64const(0.0);
        let var = declare_variable(float, builder, &mut variables, &mut index, name);
        //TODO: should we check if there is an input var with the same name and use that instead? (like with params)
        builder.def_var(var, zero);
    }

    //builder.def_var(return_variable, zero);
    for expr in stmts {
        declare_variables_in_stmt(float, builder, &mut variables, &mut index, expr);
    }

    variables
}

/// Recursively descend through the AST, translating all implicit
/// variable declarations.
fn declare_variables_in_stmt(
    int: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, Variable>,
    index: &mut usize,
    expr: &Expr,
) {
    match *expr {
        Expr::Assign(ref names, _) => {
            for name in names.iter() {
                declare_variable(int, builder, variables, index, name);
            }
        }
        Expr::IfElse(ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(int, builder, variables, index, &stmt);
            }
            for stmt in else_body {
                declare_variables_in_stmt(int, builder, variables, index, &stmt);
            }
        }
        Expr::WhileLoop(ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(int, builder, variables, index, &stmt);
            }
        }
        _ => (),
    }
}

/// Declare a single variable declaration.
fn declare_variable(
    float: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, Variable>,
    index: &mut usize,
    name: &str,
) -> Variable {
    let var = Variable::new(*index);
    if !variables.contains_key(name) {
        variables.insert(name.into(), var);
        builder.declare_var(var, float);
        *index += 1;
    }
    var
}
