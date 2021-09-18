use crate::frontend::*;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use std::collections::HashMap;
use std::fmt::{format, Display};
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
        for func in prog.iter().filter_map(|d| match d {
            Declaration::Function(func) => Some(func.clone()),
            _ => None,
        }) {
            return_counts.insert(func.name.to_string(), func.returns.len());
        }

        // First, parse the string, producing AST nodes.
        for d in prog {
            match d {
                Declaration::Function(func) => {
                    ////println!(
                    ////    "name {:?}, params {:?}, the_return {:?}",
                    ////    &name, &params, &the_return
                    ////);
                    //// Then, translate the AST nodes into Cranelift IR.
                    self.codegen(
                        func.params,
                        func.returns,
                        func.body,
                        return_counts.to_owned(),
                    )?;
                    // Next, declare the function to jit. Functions must be declared
                    // before they can be called, or defined.
                    //
                    // TODO: This may be an area where the API should be streamlined; should
                    // we have a version of `declare_function` that automatically declares
                    // the function?
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
        let float = types::F64; //self.module.target_config().pointer_type();

        for p in &params {
            if p.starts_with("&") {
                self.ctx
                    .func
                    .signature
                    .params
                    .push(AbiParam::new(self.module.target_config().pointer_type()));
            } else {
                self.ctx.func.signature.params.push(AbiParam::new(float));
            }
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
        let variables = declare_variables(
            float,
            &mut builder,
            &mut self.module,
            &params,
            &returns,
            &stmts,
            entry_block,
        );

        // Now translate the statements of the function body.
        let mut trans = FunctionTranslator {
            ty: float,
            builder,
            variables,
            return_counts,
            module: &mut self.module,
        };
        for expr in &stmts {
            trans.translate_expr(expr)?;
        }

        // Set up the return variable of the function. Above, we declared a
        // variable to hold the return value. Here, we just do a use of that
        // variable.
        let mut return_values = Vec::new();
        for ret in returns.iter() {
            let return_variable = trans.variables.get(ret).unwrap();
            return_values.push(
                trans
                    .builder
                    .use_var(return_variable.expect_float("return_variable")?),
            );
        }

        // Emit the return instruction.
        trans.builder.ins().return_(&return_values);

        // Tell the builder we're done with this function.
        trans.builder.finalize();

        //println!("{}", trans.builder.func.display(None));
        Ok(())
    }

    pub fn add_math_constants(&mut self) -> anyhow::Result<()> {
        let names = [
            ("E", std::f64::consts::E),
            ("FRAC_1_PI", std::f64::consts::FRAC_1_PI),
            ("FRAC_1_SQRT_2", std::f64::consts::FRAC_1_SQRT_2),
            ("FRAC_2_SQRT_PI", std::f64::consts::FRAC_2_SQRT_PI),
            ("FRAC_PI_2", std::f64::consts::FRAC_PI_2),
            ("FRAC_PI_3", std::f64::consts::FRAC_PI_3),
            ("FRAC_PI_4", std::f64::consts::FRAC_PI_4),
            ("FRAC_PI_6", std::f64::consts::FRAC_PI_6),
            ("FRAC_PI_8", std::f64::consts::FRAC_PI_8),
            ("LN_2", std::f64::consts::LN_2),
            ("LN_10", std::f64::consts::LN_10),
            ("LOG2_10", std::f64::consts::LOG2_10),
            ("LOG2_E", std::f64::consts::LOG2_E),
            ("LOG10_2", std::f64::consts::LOG10_2),
            ("LOG10_E", std::f64::consts::LOG10_E),
            ("PI", std::f64::consts::PI),
            ("SQRT_2", std::f64::consts::SQRT_2),
            ("TAU", std::f64::consts::TAU),
        ];
        for (name, val) in &names {
            self.create_data(name, val.to_ne_bytes().to_vec())?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SValue {
    Void,
    Unknown(Value),
    Bool(Value),
    Float(Value),
    Int(Value),
    Address(Value),
    Tuple(Vec<SValue>),
}

impl Display for SValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SValue::Unknown(_) => write!(f, "Unknown"),
            SValue::Bool(_) => write!(f, "Bool"),
            SValue::Float(_) => write!(f, "Float"),
            SValue::Int(_) => write!(f, "Int"),
            SValue::Address(_) => write!(f, "Address"),
            SValue::Void => write!(f, "Void"),
            SValue::Tuple(v) => write!(f, "Tuple ({})", v.len()),
        }
    }
}

impl SValue {
    fn inner(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Unknown(v) => Ok(*v),
            SValue::Bool(v) => Ok(*v),
            SValue::Float(v) => Ok(*v),
            SValue::Int(v) => Ok(*v),
            SValue::Address(v) => Ok(*v),
            SValue::Void => anyhow::bail!("void has no inner {}", ctx),
            SValue::Tuple(v) => anyhow::bail!("inner does not support tuple {:?} {}", v, ctx),
        }
    }
    fn expect_float(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Float(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Float {}", v, ctx),
        }
    }
    fn expect_int(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Int(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Int {}", v, ctx),
        }
    }
    fn expect_bool(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Bool(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Bool {}", v, ctx),
        }
    }
    fn expect_address(&self, ctx: &str) -> anyhow::Result<Value> {
        match self {
            SValue::Address(v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Address {}", v, ctx),
        }
    }
}

/// A collection of state used for translating from toy-language AST nodes
/// into Cranelift IR.
struct FunctionTranslator<'a> {
    ty: types::Type,
    builder: FunctionBuilder<'a>,
    variables: HashMap<String, SVariable>,
    return_counts: HashMap<String, usize>,
    module: &'a mut JITModule,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    fn translate_expr(&mut self, expr: &Expr) -> anyhow::Result<SValue> {
        match expr {
            Expr::LiteralFloat(literal) => Ok(SValue::Float(
                self.builder.ins().f64const::<f64>(literal.parse().unwrap()),
            )),
            Expr::LiteralInt(literal) => Ok(SValue::Int(
                self.builder
                    .ins()
                    .iconst::<i64>(types::I64, literal.parse().unwrap()),
            )),
            Expr::Binop(op, lhs, rhs) => self.translate_binop(*op, lhs, rhs),
            Expr::Compare(cmp, lhs, rhs) => self.translate_cmp(*cmp, lhs, rhs),
            Expr::Call(name, args) => self.translate_call(name, args),
            Expr::GlobalDataAddr(name) => Ok(SValue::Address(
                self.translate_global_data_addr(self.module.target_config().pointer_type(), name),
            )),
            Expr::Identifier(name) => {
                //TODO should this be moved into pattern matching frontend?
                if name.starts_with("&") {
                    let var = self
                        .variables
                        .get(name)
                        .expect(&format!("variable {} not found", name))
                        .expect_address("address Identifier")?;
                    Ok(SValue::Address(self.builder.use_var(var)))
                } else {
                    match self.variables.get(name) {
                        Some(var) => Ok(match var {
                            SVariable::Unknown(_, v) => SValue::Unknown(self.builder.use_var(*v)),
                            SVariable::Bool(_, v) => SValue::Bool(self.builder.use_var(*v)),
                            SVariable::Float(_, v) => SValue::Float(self.builder.use_var(*v)),
                            SVariable::Int(_, v) => SValue::Int(self.builder.use_var(*v)),
                            SVariable::Address(_, v) => SValue::Address(self.builder.use_var(*v)),
                        }),
                        None => Ok(SValue::Float(
                            //TODO Don't assume this is a float
                            self.translate_global_data_addr(self.ty, name),
                        )), //Try to load global
                    }
                }
            }
            Expr::Assign(names, expr) => self.translate_assign(names, expr),
            Expr::AssignOp(op, lhs, rhs) => self.translate_math_assign(*op, lhs, rhs),
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
            Expr::Bool(b) => Ok(SValue::Bool(self.builder.ins().bconst(types::B1, *b))),
            Expr::Parentheses(expr) => self.translate_expr(expr),
            Expr::ArrayGet(name, idx_expr) => self.translate_array_get(name.to_string(), idx_expr),
            Expr::ArraySet(name, idx_expr, expr) => {
                self.translate_array_set(name.to_string(), idx_expr, expr)
            }
        }
    }

    fn translate_binop(&mut self, op: Binop, lhs: &Expr, rhs: &Expr) -> anyhow::Result<SValue> {
        let lhs = self.translate_expr(lhs)?;
        let rhs = self.translate_expr(rhs)?;
        // if a or b is a float, convert to other to a float
        match lhs {
            SValue::Float(a) => match rhs {
                SValue::Float(b) => Ok(SValue::Float(self.binop_float(op, a, b))),
                SValue::Int(b) => {
                    let f_b = self.builder.ins().fcvt_from_sint(types::F64, b);
                    Ok(SValue::Float(self.binop_float(op, a, f_b)))
                }
                _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs),
            },
            SValue::Int(a) => match rhs {
                SValue::Float(b) => {
                    let f_a = self.builder.ins().fcvt_from_sint(types::F64, a);
                    Ok(SValue::Float(self.binop_float(op, f_a, b)))
                }
                SValue::Int(b) => Ok(SValue::Float(self.binop_int(op, a, b))),
                _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs),
            },
            _ => anyhow::bail!("operation not supported: {:?} {} {:?}", lhs, op, rhs),
        }
    }

    fn binop_float(&mut self, op: Binop, lhs: Value, rhs: Value) -> Value {
        match op {
            Binop::Add => self.builder.ins().fadd(lhs, rhs),
            Binop::Sub => self.builder.ins().fsub(lhs, rhs),
            Binop::Mul => self.builder.ins().fmul(lhs, rhs),
            Binop::Div => self.builder.ins().fdiv(lhs, rhs),
        }
    }

    fn binop_int(&mut self, op: Binop, lhs: Value, rhs: Value) -> Value {
        match op {
            Binop::Add => self.builder.ins().iadd(lhs, rhs),
            Binop::Sub => self.builder.ins().isub(lhs, rhs),
            Binop::Mul => self.builder.ins().imul(lhs, rhs),
            Binop::Div => self.builder.ins().sdiv(lhs, rhs),
        }
    }

    fn translate_cmp(&mut self, cmp: Cmp, lhs: &Expr, rhs: &Expr) -> anyhow::Result<SValue> {
        let lhs = self.translate_expr(lhs).unwrap();
        let rhs = self.translate_expr(rhs).unwrap();
        // if a or b is a float, convert to other to a float
        match lhs {
            SValue::Float(a) => match rhs {
                SValue::Float(b) => Ok(SValue::Bool(self.cmp_float(cmp, a, b))),
                SValue::Int(b) => {
                    let f_b = self.builder.ins().fcvt_from_sint(types::F64, b);
                    Ok(SValue::Bool(self.cmp_float(cmp, f_b, b)))
                }
                _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
            },
            SValue::Int(a) => match rhs {
                SValue::Float(b) => {
                    let f_a = self.builder.ins().fcvt_from_sint(types::F64, a);
                    Ok(SValue::Bool(self.cmp_float(cmp, f_a, b)))
                }
                SValue::Int(b) => Ok(SValue::Bool(self.cmp_int(cmp, a, b))),
                _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
            },
            _ => anyhow::bail!("compare not supported: {:?} {} {:?}", lhs, cmp, rhs),
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

    fn translate_assign(&mut self, names: &[String], expr: &[Expr]) -> anyhow::Result<SValue> {
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
                let expr = self.translate_expr(expr.get(i).unwrap())?;
                let var = match self.variables.get(name) {
                    Some(v) => v,
                    None => anyhow::bail!("variable {} not found", name),
                };
                match expr {
                    SValue::Tuple(_) => anyhow::bail!("operation not supported: assign Tuple"),
                    SValue::Void => anyhow::bail!("operation not supported: assign Void"),
                    SValue::Unknown(v) => {
                        values.push(SValue::Unknown(v));
                        self.builder.def_var(var.inner(), v);
                        v
                    }
                    SValue::Bool(v) => {
                        values.push(SValue::Bool(v));
                        self.builder.def_var(var.expect_bool("assign")?, v);
                        v
                    }
                    SValue::Float(v) => {
                        values.push(SValue::Float(v));
                        self.builder.def_var(var.expect_float("assign")?, v);
                        v
                    }
                    SValue::Int(v) => {
                        values.push(SValue::Int(v));
                        self.builder.def_var(var.expect_int("assign")?, v);
                        v
                    }
                    SValue::Address(v) => {
                        values.push(SValue::Address(v));
                        self.builder.def_var(var.expect_address("assign")?, v);
                        v
                    }
                };
            }
            if values.len() > 1 {
                Ok(SValue::Tuple(
                    values.iter().map(|v| v.clone()).collect::<Vec<SValue>>(),
                ))
            } else if values.len() == 1 {
                Ok(values.first().unwrap().clone())
            } else {
                Ok(SValue::Void)
            }
        } else {
            match self.translate_expr(expr.first().unwrap())? {
                SValue::Tuple(values) => {
                    for (i, name) in names.iter().enumerate() {
                        let variable = match self.variables.get(name) {
                            Some(v) => v,
                            None => anyhow::bail!("variable {} not found", name),
                        };
                        let sval = match values[i] {
                            SValue::Tuple(_) => {
                                anyhow::bail!("operation not supported {:?}", expr)
                            }
                            SValue::Void => anyhow::bail!("operation not supported {:?}", expr),
                            SValue::Unknown(v) => v,
                            SValue::Bool(v) => v,
                            SValue::Float(v) => v,
                            SValue::Int(v) => v,
                            SValue::Address(v) => v,
                        };
                        self.builder.def_var(variable.inner(), sval);
                    }
                    if values.len() > 1 {
                        Ok(SValue::Tuple(
                            values.iter().map(|v| v.clone()).collect::<Vec<SValue>>(),
                        ))
                    } else if values.len() == 1 {
                        Ok(values.first().unwrap().clone())
                    } else {
                        Ok(SValue::Void)
                    }
                }
                SValue::Void => anyhow::bail!("operation not supported {:?}", expr),
                SValue::Unknown(_) => anyhow::bail!("operation not supported {:?}", expr),
                SValue::Bool(_) => anyhow::bail!("operation not supported {:?}", expr),
                SValue::Float(_) => anyhow::bail!("operation not supported {:?}", expr),
                SValue::Int(_) => anyhow::bail!("operation not supported {:?}", expr),
                SValue::Address(_) => anyhow::bail!("operation not supported {:?}", expr),
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
            SValue::Float(v) => self.builder.ins().fcvt_to_uint(ptr_ty, v),
            SValue::Int(v) => v,
            _ => anyhow::bail!("only int and float supported for array access"),
        };
        let mult_n = self.builder.ins().iconst(ptr_ty, types::F64.bytes() as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        let val =
            self.builder
                .ins()
                .load(types::F64, MemFlags::trusted(), idx_ptr, Offset32::new(0));
        Ok(SValue::Float(val)) //todo, don't assume this is a float
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
            .expect_address("array_set")?;

        let array_ptr = self.builder.use_var(variable);

        let idx_val = self.translate_expr(idx_expr)?;
        let idx_val = match idx_val {
            SValue::Float(v) => self.builder.ins().fcvt_to_uint(ptr_ty, v),
            SValue::Int(v) => v,
            _ => anyhow::bail!("only int and float supported for array access"),
        };
        let mult_n = self.builder.ins().iconst(ptr_ty, types::F64.bytes() as i64);
        let idx_val = self.builder.ins().imul(mult_n, idx_val);
        let idx_ptr = self.builder.ins().iadd(idx_val, array_ptr);

        let new_val = match new_val {
            SValue::Float(v) => v,
            SValue::Int(v) => v,
            SValue::Void => anyhow::bail!("store Void not supported"),
            SValue::Unknown(v) => v,
            SValue::Bool(v) => v,
            SValue::Address(v) => v,
            SValue::Tuple(_) => anyhow::bail!("store tuple not supported"),
        };

        self.builder
            .ins()
            .store(MemFlags::trusted(), new_val, idx_ptr, Offset32::new(0));
        Ok(SValue::Void)
    }

    fn translate_math_assign(
        &mut self,
        op: Binop,
        name: &str,
        expr: &Expr,
    ) -> anyhow::Result<SValue> {
        //TODO Don't just assume that destination var is same type as set var
        match self.translate_expr(expr)? {
            SValue::Float(v) => {
                let orig_variable = self.variables.get(&*name).unwrap();
                let orig_value = self
                    .builder
                    .use_var(orig_variable.expect_float("math_assign")?);
                let added_val = match op {
                    Binop::Add => self.builder.ins().fadd(orig_value, v),
                    Binop::Sub => self.builder.ins().fsub(orig_value, v),
                    Binop::Mul => self.builder.ins().fmul(orig_value, v),
                    Binop::Div => self.builder.ins().fdiv(orig_value, v),
                };
                self.builder
                    .def_var(orig_variable.expect_float("math_assign")?, added_val);
                Ok(SValue::Float(added_val))
            }
            SValue::Int(v) => {
                let orig_variable = self.variables.get(&*name).unwrap();
                let orig_value = self
                    .builder
                    .use_var(orig_variable.expect_int("math_assign")?);
                let added_val = match op {
                    Binop::Add => self.builder.ins().iadd(orig_value, v),
                    Binop::Sub => self.builder.ins().isub(orig_value, v),
                    Binop::Mul => self.builder.ins().imul(orig_value, v),
                    Binop::Div => self.builder.ins().sdiv(orig_value, v),
                };
                self.builder
                    .def_var(orig_variable.expect_int("math_assign")?, added_val);
                Ok(SValue::Int(added_val))
            }
            SValue::Void => anyhow::bail!("math assign Void not supported"),
            SValue::Unknown(_) => anyhow::bail!("math assign unknown not supported"),
            SValue::Bool(_) => anyhow::bail!("math assign bool not supported"),
            SValue::Address(_) => anyhow::bail!("math assign address not supported"),
            SValue::Tuple(_) => anyhow::bail!("math assign tuple not supported"),
        }
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
            SValue::Float(v) => vec![v],
            SValue::Int(v) => vec![v],
            SValue::Address(v) => vec![v],
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

        // TODO stop assuming everything is a float
        // to be able to do this here, we'll need a way to find where
        // all these vars came from, and what their types would be

        if phi.len() > 1 {
            Ok(SValue::Tuple(
                phi.iter()
                    .map(|v| SValue::Float(*v))
                    .collect::<Vec<SValue>>(),
            ))
        } else if phi.len() == 1 {
            match then_value {
                SValue::Void => Ok(SValue::Void),
                SValue::Unknown(_) => Ok(SValue::Unknown(*phi.first().unwrap())),
                SValue::Bool(_) => Ok(SValue::Bool(*phi.first().unwrap())),
                SValue::Float(_) => Ok(SValue::Float(*phi.first().unwrap())),
                SValue::Int(_) => Ok(SValue::Int(*phi.first().unwrap())),
                SValue::Address(_) => Ok(SValue::Address(*phi.first().unwrap())),
                SValue::Tuple(_) => anyhow::bail!("not supported"),
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

    fn translate_call(&mut self, name: &str, args: &[Expr]) -> anyhow::Result<SValue> {
        let mut sig = self.module.make_signature();

        // Add a parameter for each argument.
        for _ in args {
            sig.params.push(AbiParam::new(self.ty));
        }

        if self.return_counts.contains_key(name) {
            for _ in 0..self.return_counts[name] {
                sig.returns.push(AbiParam::new(self.ty));
            }
        } else {
            match self.translate_std(name, args)? {
                Some(v) => return Ok(v),
                None => {
                    // If we can't find the function name, maybe it's a libc function.
                    // For now, assume it will return a float.
                    sig.returns.push(AbiParam::new(self.ty))
                }
            }
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
        for (i, arg) in args.iter().enumerate() {
            arg_values.push({
                //TODO support returning more than just float
                self.translate_expr(arg)?
                    .expect_float(&format!("{} arg {}", name, i))?
            })
        }
        let call = self.builder.ins().call(local_callee, &arg_values);
        let res = self.builder.inst_results(call);

        if res.len() > 1 {
            Ok(SValue::Tuple(
                res.iter()
                    .map(|v| SValue::Float(*v))
                    .collect::<Vec<SValue>>(),
            ))
        } else if res.len() == 1 {
            Ok(SValue::Float(*res.first().unwrap()))
        } else {
            Ok(SValue::Void)
        }
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

        //TODO see if this still works with strings like in original toy example

        //self.builder.ins().symbol_value(ptr_ty, local_id)
        self.builder.ins().global_value(ptr_ty, global_val)
    }

    fn translate_std(&mut self, name: &str, args: &[Expr]) -> anyhow::Result<Option<SValue>> {
        match args.len() {
            1 => match self.translate_expr(&args[0])? {
                SValue::Float(v) => match name {
                    "trunc" => Ok(Some(SValue::Float(self.builder.ins().trunc(v)))),
                    "floor" => Ok(Some(SValue::Float(self.builder.ins().floor(v)))),
                    "ceil" => Ok(Some(SValue::Float(self.builder.ins().ceil(v)))),
                    "fract" => {
                        let v_int = self.builder.ins().trunc(v);
                        let v = self.builder.ins().fsub(v, v_int);
                        Ok(Some(SValue::Float(v)))
                    }
                    "abs" => Ok(Some(SValue::Float(self.builder.ins().fabs(v)))),
                    "round" => Ok(Some(SValue::Float(self.builder.ins().nearest(v)))),
                    "int" => Ok(Some(SValue::Int(
                        self.builder.ins().fcvt_to_sint(types::I64, v),
                    ))),
                    _ => Ok(None),
                },
                SValue::Int(v) => match name {
                    "float" => Ok(Some(SValue::Float(
                        self.builder.ins().fcvt_from_sint(types::F64, v),
                    ))),
                    _ => Ok(None),
                },
                t => anyhow::bail!("type {:?} not supported", t),
            },
            2 => {
                let v0 = self
                    .translate_expr(&args[0])?
                    .expect_float("translate_std")?;
                let v1 = self
                    .translate_expr(&args[1])?
                    .expect_float("translate_std")?;
                match name {
                    //TODO int versions of this
                    "min" => Ok(Some(SValue::Float(self.builder.ins().fmin(v0, v1)))),
                    "max" => Ok(Some(SValue::Float(self.builder.ins().fmax(v0, v1)))),
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    fn value_type(&self, val: Value) -> Type {
        self.builder.func.dfg.value_type(val)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SVariable {
    Unknown(String, Variable),
    Bool(String, Variable),
    Float(String, Variable),
    Int(String, Variable),
    Address(String, Variable),
}

impl Display for SVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVariable::Unknown(name, _) => write!(f, "Unknown {}", name),
            SVariable::Bool(name, _) => write!(f, "Bool {}", name),
            SVariable::Float(name, _) => write!(f, "Float {}", name),
            SVariable::Int(name, _) => write!(f, "Int {}", name),
            SVariable::Address(name, _) => write!(f, "Address {}", name),
        }
    }
}

impl SVariable {
    fn inner(&self) -> Variable {
        match self {
            SVariable::Unknown(_, v) => *v,
            SVariable::Bool(_, v) => *v,
            SVariable::Float(_, v) => *v,
            SVariable::Int(_, v) => *v,
            SVariable::Address(_, v) => *v,
        }
    }
    fn expect_float(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Float(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Float {}", v, ctx),
        }
    }
    fn expect_int(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Int(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Int {}", v, ctx),
        }
    }
    fn expect_bool(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Bool(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Bool {}", v, ctx),
        }
    }
    fn expect_address(&self, ctx: &str) -> anyhow::Result<Variable> {
        match self {
            SVariable::Address(_, v) => Ok(*v),
            v => anyhow::bail!("incorrect type {} expected Address {}", v, ctx),
        }
    }
}

fn declare_variables(
    float: types::Type,
    builder: &mut FunctionBuilder,
    module: &mut dyn Module,
    params: &[String],
    returns: &[String],
    stmts: &[Expr],
    entry_block: Block,
) -> HashMap<String, SVariable> {
    //TODO we should create a list of the variables here with their expected type, so it can be referenced later
    let mut variables: HashMap<String, SVariable> = HashMap::new();
    let mut index = 0;

    for (i, name) in params.iter().enumerate() {
        let val = builder.block_params(entry_block)[i];

        // for now all function parameters are either float or array
        let var = if name.starts_with("&") {
            declare_variable(
                true,
                module.target_config().pointer_type(),
                builder,
                &mut variables,
                &mut index,
                name,
            )
        } else {
            declare_variable(false, float, builder, &mut variables, &mut index, name)
        };
        builder.def_var(var.inner(), val);
    }

    for name in returns {
        let zero = builder.ins().f64const(0.0);
        let var = declare_variable(false, float, builder, &mut variables, &mut index, name);
        //TODO: should we check if there is an input var with the same name and use that instead? (like with params)
        // for now all function returns are either float or array
        builder.def_var(var.inner(), zero);
    }

    //builder.def_var(return_variable, zero);
    for expr in stmts {
        declare_variables_in_stmt(
            module.target_config().pointer_type(),
            float,
            builder,
            &mut variables,
            &mut index,
            expr,
        );
    }

    variables
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
) {
    match *expr {
        Expr::Assign(ref names, ref exprs) => {
            if exprs.len() == names.len() {
                for (name, expr) in names.iter().zip(exprs.iter()) {
                    declare_variable_from_expr(ptr_type, expr, builder, variables, index, name);
                }
            } else {
                for name in names.iter() {
                    declare_variable(false, ty, builder, variables, index, name);
                }
            }
        }
        Expr::IfElse(ref _condition, ref then_body, ref else_body) => {
            for stmt in then_body {
                declare_variables_in_stmt(ptr_type, ty, builder, variables, index, &stmt);
            }
            for stmt in else_body {
                declare_variables_in_stmt(ptr_type, ty, builder, variables, index, &stmt);
            }
        }
        Expr::WhileLoop(ref _condition, ref loop_body) => {
            for stmt in loop_body {
                declare_variables_in_stmt(ptr_type, ty, builder, variables, index, &stmt);
            }
        }
        _ => (),
    }
}

/// Declare a single variable declaration.
fn declare_variable_from_expr(
    ptr_type: Type,
    expr: &Expr,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    name: &str,
) -> Variable {
    let var = Variable::new(*index);
    if !variables.contains_key(name) {
        match expr {
            Expr::LiteralFloat(_) => {
                variables.insert(name.into(), SVariable::Float(name.into(), var));
                builder.declare_var(var, types::F64);
            }
            Expr::LiteralInt(_) => {
                variables.insert(name.into(), SVariable::Int(name.into(), var));
                builder.declare_var(var, types::I64);
            }
            Expr::Bool(_) => {
                variables.insert(name.into(), SVariable::Bool(name.into(), var));
                builder.declare_var(var, types::B1);
            }
            Expr::Identifier(_) => {
                if name.starts_with("&") {
                    variables.insert(name.into(), SVariable::Address(name.into(), var));
                    builder.declare_var(var, ptr_type);
                } else {
                    //Don't assume this is a float. (maybe look at types of existing vars?)
                    variables.insert(name.into(), SVariable::Float(name.into(), var));
                    builder.declare_var(var, types::F64);
                }
            }
            Expr::Call(c, _) => {
                if c == "int" {
                    variables.insert(name.into(), SVariable::Int(name.into(), var));
                    builder.declare_var(var, types::I64);
                } else {
                    variables.insert(name.into(), SVariable::Float(name.into(), var));
                    builder.declare_var(var, types::F64);
                }
            }
            Expr::IfElse(_condition, then_body, else_body) => {
                //TODO make sure then & else returns match
                declare_variable_from_expr(
                    ptr_type,
                    then_body.last().unwrap(),
                    builder,
                    variables,
                    index,
                    name,
                );
            }
            _ => {
                variables.insert(name.into(), SVariable::Float(name.into(), var));
                builder.declare_var(var, types::F64);
            }
        };
        *index += 1;
    }
    var
}

fn declare_variable(
    is_pointer: bool,
    ty: types::Type,
    builder: &mut FunctionBuilder,
    variables: &mut HashMap<String, SVariable>,
    index: &mut usize,
    name: &str,
) -> SVariable {
    let mut var = SVariable::Float(name.into(), Variable::new(*index));
    if !variables.contains_key(name) {
        var = if is_pointer {
            SVariable::Address(name.into(), Variable::new(*index))
        } else {
            match ty {
                types::F64 => SVariable::Float(name.into(), Variable::new(*index)),
                types::B1 => SVariable::Bool(name.into(), Variable::new(*index)),
                types::I64 => SVariable::Int(name.into(), Variable::new(*index)),
                _ => SVariable::Float(name.into(), Variable::new(*index)),
            }
        };
        variables.insert(name.into(), var.clone());
        builder.declare_var(var.inner(), ty);
        *index += 1;
    }
    var
}
