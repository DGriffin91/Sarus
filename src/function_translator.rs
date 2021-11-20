use crate::frontend::*;
use crate::jit::Env;
use crate::sarus_std_lib;
pub use crate::structs::*;
use crate::validator::ArraySizedExpr;
use crate::validator::ExprType;
pub use crate::variables::*;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::ir::ArgumentPurpose;
use cranelift::prelude::*;
pub use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::ffi::CString;
use tracing::info;
use tracing::instrument;
use tracing::trace;
use tracing::warn;

/// A collection of state used for translating from Sarus AST nodes
/// into Cranelift IR.
pub struct FunctionTranslator<'a> {
    pub builder: FunctionBuilder<'a>,
    pub module: &'a mut JITModule,
    pub env: Env,
    pub ptr_ty: types::Type,

    // This is a vec where inline calls will push hashmaps of vars as they
    // are called, and pop off the end when they are done variables.last()
    // should have the variables for the "current" function (inline or not)
    pub variables: Vec<HashMap<String, SVariable>>,

    // Stack of inlined function names. Used as prefixes for inline
    // when entering an inline section the func name will be appended
    // when leaving the inline section the func name will be removed
    pub func_stack: Vec<Function>,

    // so that we can add inline vars
    pub entry_block: Block,
    pub var_index: usize,

    //incremented when starting to translate an expression, decremented when returning value
    pub expr_depth: usize,

    // Stack of widths of the deep stack used in each scope
    // when entering an inline section a new width starting at 0 will be pushed on
    // when leaving the inline section or function the width will be popped off
    // and the deep stack pointer we be decremented proportionally
    pub deep_stack_widths: Vec<usize>,
    pub use_deep_stack: bool,
    pub max_deep_stack_size: usize,
}

impl<'a> FunctionTranslator<'a> {
    /// When you write out instructions in Cranelift, you get back `Value`s. You
    /// can then use these references in other instructions.
    #[instrument(name = "expr", skip(self, expr))]
    pub fn translate_expr(&mut self, expr: &Expr) -> anyhow::Result<SValue> {
        self.expr_depth += 1;
        info!(
            "{}: {} | {} (expr_depth {})",
            expr.get_code_ref(),
            expr,
            expr.debug_get_name(),
            self.expr_depth,
        );
        //dbg!(&expr);
        let v = match expr {
            Expr::LiteralFloat(_code_ref, literal) => {
                Ok(SValue::F32(self.builder.ins().f32const::<f32>(*literal)))
            }
            Expr::LiteralInt(_code_ref, literal) => Ok(SValue::I64(
                self.builder.ins().iconst::<i64>(types::I64, *literal),
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
                } else if self
                    .env
                    .get_inline_closure(&self.func_stack.last().unwrap().name, name)
                    .is_some()
                {
                    //This is a closure identifier
                    //TODO if this was in an inline function it would not show up as &self.func.name
                    Ok(SValue::Void)
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
            Expr::Block(_code_ref, b) => b.iter().map(|e| self.translate_expr(e)).last().unwrap(),
            Expr::LiteralBool(_code_ref, b) => {
                Ok(SValue::Bool(self.builder.ins().bconst(types::B1, *b)))
            }
            Expr::Parentheses(_code_ref, expr) => self.translate_expr(expr),
            Expr::ArrayAccess(code_ref, expr, idx_expr) => {
                let idx_val = self.idx_expr_to_val(idx_expr)?;
                self.translate_array_get(code_ref, expr, idx_val, false)
            }
            Expr::Declaration(_code_ref, declaration) => match &declaration {
                Declaration::Function(_closure) => Ok(SValue::Void),
                Declaration::Metadata(_, _) => todo!(),
                Declaration::Struct(_) => todo!(),
                Declaration::StructMacro(_, _) => todo!(),
                Declaration::Include(_) => todo!(),
            },
        };
        self.expr_depth -= 1;
        v
    }

    fn get_variable(&mut self, code_ref: &CodeRef, name: &str) -> anyhow::Result<&SVariable> {
        match self.variables.last().unwrap().get(name) {
            Some(v) => Ok(v),
            None => anyhow::bail!(
                "{} variable {} not found",
                code_ref.s(&self.env.file_idx),
                name
            ),
        }
    }

    fn translate_string(&mut self, literal: &str) -> anyhow::Result<SValue> {
        let cstr = CString::new(literal.replace("\\n", "\n")).unwrap();
        let bytes = cstr.to_bytes_with_nul();
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            types::I8.bytes() * bytes.len() as u32,
        ));
        let stack_slot_address =
            self.builder
                .ins()
                .stack_addr(self.ptr_ty, stack_slot, Offset32::new(0));
        //TODO use self.alloc (returns_a_fixed_array_in_a_struct_basic is failing if the deep stack is always used)
        //let stack_slot_address = self.alloc((types::I8.bytes() * bytes.len() as u32) as i64);

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
            None => anyhow::bail!(
                "{} expression {} has no size",
                code_ref.s(&self.env.file_idx),
                item_expr
            ),
        };

        let stack_slot_address = self.alloc((item_width * len) as usize);

        let len_val = self.builder.ins().iconst(types::I64, len as i64);

        //if len > 16, fill using while loop
        if len > 16 {
            let header_block = self.builder.create_block();
            let body_block = self.builder.create_block();
            let exit_block = self.builder.create_block();

            let inc_val = self.builder.ins().iconst(types::I64, 1);

            let i_val = self.builder.ins().iconst(types::I64, 0);
            let i_var = Variable::new(self.var_index);
            self.var_index += 1;
            self.builder.declare_var(i_var, types::I64);
            self.builder.def_var(i_var, i_val);

            let len_val = self.builder.ins().iconst(types::I64, len as i64);
            let width_val = self.builder.ins().iconst(types::I64, item_width as i64);

            let set_val = item_value.inner("translate_array_create")?;

            self.builder.ins().jump(header_block, &[]);
            self.builder.switch_to_block(header_block);

            let i_val = self.builder.use_var(i_var);

            let b_condition_value = self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThan, i_val, len_val);

            self.builder.ins().brz(b_condition_value, exit_block, &[]);
            self.builder.ins().jump(body_block, &[]);

            self.builder.switch_to_block(body_block);
            self.builder.seal_block(body_block);

            let i_val = self.builder.use_var(i_var);

            let offset_val = self.builder.ins().imul(i_val, width_val);
            let stack_slot_address_abs_pos =
                self.builder.ins().iadd(stack_slot_address, offset_val);

            //Essentially same as below
            match item_expr_type {
                ExprType::Bool(_) | ExprType::F32(_) | ExprType::I64(_) | ExprType::Address(_) => {
                    self.builder.ins().store(
                        MemFlags::new(),
                        set_val,
                        stack_slot_address_abs_pos,
                        Offset32::new(0),
                    );
                }
                ExprType::Struct(_code_ref, _) | ExprType::Array(_code_ref, _, _) => {
                    self.builder.emit_small_memory_copy(
                        self.module.target_config(),
                        stack_slot_address_abs_pos,
                        set_val,
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

            let i_val = self.builder.ins().iadd(i_val, inc_val);
            self.builder.def_var(i_var, i_val);

            self.builder.ins().jump(header_block, &[]);

            self.builder.switch_to_block(exit_block);

            self.builder.seal_block(header_block);
            self.builder.seal_block(exit_block);
        } else {
            for i in 0..len {
                let val = item_value.inner("translate_array_create")?;
                match item_expr_type {
                    ExprType::Bool(_)
                    | ExprType::F32(_)
                    | ExprType::I64(_)
                    | ExprType::Address(_) => {
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
                        let stack_slot_offset =
                            self.builder.ins().iadd(stack_slot_address, offset_v);
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
        }

        let len_sval = Box::new(SValue::I64(len_val));
        let ret_val = SValue::Array(
            Box::new(SValue::from(
                &mut self.builder,
                &item_value.expr_type(code_ref)?,
                stack_slot_address,
            )?),
            ArraySized::Fixed(len_sval, len),
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

        while let Some(expr) = curr_expr {
            //println!("curr_expr {:?} next_expr {:?}", &curr_expr, &next_expr);
            //println!("path {:?}", &path);

            let debug_name = expr.debug_get_name();
            if debug_name != "Binop" {
                log_path.push(debug_name);
            }
            curr_expr = next_expr;
            next_expr = None;
            array_field = false;
            match expr {
                Expr::Call(code_ref, fn_name, args, is_macro) => {
                    if path.is_empty() {
                        lhs_val =
                            Some(self.translate_call(code_ref, fn_name, args, lhs_val, *is_macro)?);
                    } else {
                        let sval = if path.len() > 1 || lhs_val.is_some() {
                            let spath = path
                                .iter()
                                .map(|lhs_i: &Expr| lhs_i.to_string())
                                .collect::<Vec<String>>();
                            let (sval_address, struct_def) =
                                self.get_struct_field_address(code_ref, spath, lhs_val)?;
                            if let ExprType::Struct(_code_ref, _name) = struct_def.clone().expr_type
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
                Expr::LiteralString(_code_ref, _) => lhs_val = Some(self.translate_expr(expr)?),
                Expr::LiteralArray(_code_ref, _, _) => lhs_val = Some(self.translate_expr(expr)?),
                Expr::Identifier(_code_ref, _) => path.push(expr.clone()),
                Expr::Binop(_code_ref, op, lhs, rhs) => {
                    if let Binop::DotAccess = op {
                        curr_expr = Some(lhs);
                        next_expr = Some(rhs);
                    } else {
                        todo!();
                    }
                }

                Expr::Parentheses(_code_ref, e) => lhs_val = Some(self.translate_expr(e)?),
                Expr::ArrayAccess(code_ref, expr, idx_expr) => {
                    if !path.is_empty() {
                        let mut spath = path
                            .iter()
                            .map(|lhs_i: &Expr| lhs_i.to_string())
                            .collect::<Vec<String>>();
                        let name = if let Expr::Identifier(_code_ref, name) = *expr.clone() {
                            name
                        } else {
                            anyhow::bail!("{} array access on dot binop of non identifier not supported yet {}", code_ref, expr)
                        };
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
                        lhs_val =
                            Some(self.translate_array_get(code_ref, expr, idx_val, get_address)?);
                    }

                    path = Vec::new();
                }
                Expr::NewStruct(_code_ref, _, _) => lhs_val = Some(self.translate_expr(expr)?),
                Expr::Declaration(code_ref, ..)
                | Expr::Unaryop(code_ref, ..)
                | Expr::Compare(code_ref, ..)
                | Expr::IfThen(code_ref, ..)
                | Expr::IfElse(code_ref, ..)
                | Expr::IfThenElseIf(code_ref, ..)
                | Expr::IfThenElseIfElse(code_ref, ..)
                | Expr::Assign(code_ref, ..)
                | Expr::WhileLoop(code_ref, ..)
                | Expr::Block(code_ref, ..)
                | Expr::GlobalDataAddr(code_ref, ..)
                | Expr::LiteralFloat(code_ref, ..)
                | Expr::LiteralInt(code_ref, ..)
                | Expr::LiteralBool(code_ref, ..) => anyhow::bail!(
                    "{} dot binop not supported, try putting expression in parenthesis: ({})",
                    code_ref,
                    expr
                ),
            }
        }
        if !path.is_empty() {
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

        Ok(match op {
            Unaryop::Not => match lhs {
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
                    anyhow::bail!("operation not supported: {} {:?}", op, lhs)
                }
            },
            Unaryop::Negative => match lhs {
                SValue::F32(lhs) => SValue::F32(self.builder.ins().fneg(lhs)),
                SValue::I64(lhs) => SValue::I64(self.builder.ins().ineg(lhs)),
                SValue::Void
                | SValue::Bool(_)
                | SValue::Unknown(_)
                | SValue::Array(_, _)
                | SValue::Address(_)
                | SValue::Struct(_, _)
                | SValue::Tuple(_) => {
                    anyhow::bail!("operation not supported: {} {:?}", op, lhs)
                }
            },
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
                        declare_variable(
                            self.ptr_ty,
                            &src_sval.expr_type(code_ref)?,
                            &mut self.builder,
                            &mut self.var_index,
                            &[&name],
                            &mut self.variables.last_mut().unwrap(),
                            false,
                        )?;
                        //Can this be done without clone?
                        let dst_svar = self.get_variable(code_ref, name)?.clone();

                        let this_func = self.func_stack.last().unwrap();

                        for arg in this_func.params.iter().chain(this_func.returns.iter()) {
                            if *name == arg.name {
                                /*
                                Should this happen also if the var already exists and has already been initialized?
                                (this can't really be determined at compile time. One option would be to allocate
                                the stack space for all potential vars to be used in a given function. But that
                                could also be excessive. Also even if this copy happens, the stack data from the
                                src_var won't be freed until after the function returns. This could be another
                                reason for having scopes work more like they do in other languages. Then if a stack
                                allocation is being created in a loop, it will be freed on each loop if it hasn't been
                                stored to a var that is outside of the scope of the loop) (How it works also has
                                implications for how aliasing works)
                                */
                                if let ExprType::Struct(code_ref, struct_name) = &arg.expr_type {
                                    trace!(
                                        "{} struct {} is arg {} of this fn {} copying onto memory at {} on assignment",
                                        code_ref,
                                        struct_name,
                                        arg.name,
                                        &self.func_stack.last().unwrap().name,
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
                                            &format!(
                                                "{} translate_assign",
                                                code_ref.s(&self.env.file_idx)
                                            ),
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
                                                &self.func_stack.last().unwrap(),
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
                            declare_variable(
                                self.ptr_ty,
                                &values[i].expr_type(code_ref)?,
                                &mut self.builder,
                                &mut self.var_index,
                                &[&name],
                                &mut self.variables.last_mut().unwrap(),
                                false,
                            )?;
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

    fn translate_array_get(
        &mut self,
        code_ref: &CodeRef,
        expr: &Expr,
        idx_val: Value,
        get_address: bool,
    ) -> anyhow::Result<SValue> {
        let sval = self.translate_expr(expr)?;

        match &sval {
            SValue::Array(address, size_type) => {
                let mut bound_check_at_get = true;
                match size_type {
                    ArraySized::Unsized => (),
                    ArraySized::Sized => todo!(),
                    ArraySized::Fixed(sval, _len) => {
                        let b_condition_value = self.builder.ins().icmp(
                            IntCC::SignedGreaterThanOrEqual,
                            idx_val,
                            sval.inner("translate_array_get")?,
                        );
                        let merge_block = self.exec_if_start(b_condition_value);
                        self.call_panic(
                            code_ref,
                            &format!("{} index out of bounds", code_ref.s(&self.env.file_idx)),
                        )?;
                        self.exec_if_end(merge_block);
                        bound_check_at_get = false;
                    }
                }
                self.array_get(
                    address.inner("translate_array_get")?,
                    &sval.expr_type(code_ref)?,
                    idx_val,
                    get_address,
                    bound_check_at_get,
                )
            }
            _ => anyhow::bail!(
                "{} expression {} is not an array",
                code_ref.s(&self.env.file_idx),
                expr
            ),
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
                                &format!("{} index out of bounds", code_ref.s(&self.env.file_idx)),
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
        condition_bodies: &[(Expr, Vec<Expr>)],
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
        condition_bodies: &[(Expr, Vec<Expr>)],
        else_body: &[Expr],
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
        let else_return = match else_value {
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

        if self.use_deep_stack {
            self.deep_stack_widths.push(0);
        }

        for expr in loop_body {
            self.translate_expr(expr)?;
        }

        self.dealloc_deep_stack();

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

        if fn_name == "unsized" {
            return self.call_with_svalues(
                code_ref, &fn_name, None, None, false, false, arg_values, None,
            );
        }

        let callee_func_name = &self.func_stack.last().unwrap().name;
        let mut is_closure = false;
        let mut is_temp_closure = false;
        let mut closure_src_scope_name = None;
        let closure = self.env.get_inline_closure(callee_func_name, &fn_name);

        let func = if let Some((closure, temp_closure)) = closure {
            is_temp_closure = temp_closure;
            is_closure = true;
            closure_src_scope_name = Some(closure.src_scope);
            Some(closure.func)
        } else {
            self.env.funcs.get(&fn_name).cloned()
        };

        let func = if let Some(func) = func {
            func
        } else {
            anyhow::bail!(
                "{} function {} not found",
                code_ref.s(&self.env.file_idx),
                fn_name
            )
        };

        let mut closure_args = Vec::new();
        // Put arg closures into function being called with name from args
        // closures are currently only passed in as a declaration or identifier
        // which closures are used for args has to be known at compile time, can't
        // be behind an if/then, etc... Eventually there may be compile time
        // execution that would allow this.
        if let InlineKind::Always = func.inline {
            let func = func.clone();
            for (expr, arg) in args.iter().zip(func.params.iter()) {
                let closure_d = match expr {
                    Expr::Identifier(_, name) => {
                        if let Some((closure, temp_closure)) =
                            self.env.get_inline_closure(callee_func_name, name)
                        {
                            Some((closure, temp_closure))
                        } else {
                            None
                        }
                    }
                    Expr::Declaration(_, Declaration::Function(closure)) => Some((
                        Closure {
                            func: closure.clone(),
                            src_scope: self.func_stack.last().unwrap().name.to_string(),
                        },
                        false,
                    )),
                    _ => None,
                };
                if let Some((closure, temp_closure)) = closure_d.clone() {
                    trace!(
                        "{} func {} arg {} is a closure parameter. temp_closure: {}",
                        code_ref,
                        &func.name,
                        &arg.name,
                        temp_closure
                    );
                    if let Some(closure_arg) = &arg.closure_arg {
                        {
                            //TODO move to validator
                            if closure.func.params.len() != closure_arg.params.len() {
                                anyhow::bail!(
                                "{} func {} arg {} closure parameter count does not match signature",
                                code_ref,
                                &func.name,
                                &arg.name,
                            )
                            }
                            if closure.func.returns.len() != closure_arg.returns.len() {
                                anyhow::bail!(
                                "{} func {} arg {} closure return count does not match signature",
                                code_ref,
                                &func.name,
                                &arg.name,
                            )
                            }
                            for (param, arg_param) in
                                closure.func.params.iter().zip(closure_arg.params.iter())
                            {
                                if param.expr_type != arg_param.expr_type {
                                    anyhow::bail!(
                                    "{} func {} arg {} closure parameter types do not match. Expected {} but found {}",
                                    code_ref,
                                    &func.name,
                                    &arg.name,
                                    arg_param.expr_type,
                                    param.expr_type,
                                )
                                }
                            }
                            for (return_, arg_return) in
                                closure.func.returns.iter().zip(closure_arg.returns.iter())
                            {
                                if return_.expr_type != arg_return.expr_type {
                                    anyhow::bail!(
                                    "{} func {} arg {} closure returns types do not match. Expected {} but found {}",
                                    code_ref,
                                    &func.name,
                                    &arg.name,
                                    arg_return.expr_type,
                                    return_.expr_type,
                                )
                                }
                            }
                        }
                        //If func does not have any closures, it wont have a hashmap under its name
                        if !self.env.temp_inline_closures.contains_key(&func.name) {
                            self.env
                                .temp_inline_closures
                                .insert(func.name.to_string(), HashMap::new());
                        }
                        closure_args.push(arg.name.to_string());

                        //alias closure into function being called with name from arg
                        self.env
                            .temp_inline_closures
                            .get_mut(&func.name)
                            .unwrap()
                            .insert(arg.name.to_string(), closure);
                    } else {
                        anyhow::bail!(
                            "{} func {} arg {} is not a closure arg",
                            code_ref,
                            &func.name,
                            &arg.name
                        )
                    }
                }
            }
        }

        let returns = &func.returns;

        let mut stack_slot_return = None;

        if !returns.is_empty() {
            if let ExprType::Struct(_code_ref, name) = &returns[0].expr_type {
                if let Some(s) = self.env.struct_map.get(&name.to_string()) {
                    stack_slot_return = Some(s.size);
                }
            } else if let ExprType::Array(_code_ref, expr_type, size_type) = &returns[0].expr_type {
                match size_type {
                    ArraySizedExpr::Unsized => {}
                    ArraySizedExpr::Sized => todo!(),
                    ArraySizedExpr::Fixed(len) => {
                        stack_slot_return = Some(
                            expr_type
                                .width(self.ptr_ty, &self.env.struct_map)
                                .unwrap_or(0)
                                * len,
                        )
                    }
                }
            }
        }
        let ret = self.call_with_svalues(
            code_ref,
            &fn_name,
            Some(&func),
            closure_src_scope_name,
            is_closure,
            is_temp_closure,
            arg_values,
            stack_slot_return,
        );

        //Clean up temporary closure args
        for closure_arg in closure_args {
            self.env
                .temp_inline_closures
                .get_mut(&func.name)
                .unwrap()
                .remove(&closure_arg);
        }

        ret
    }

    fn translate_constant(
        &mut self,
        code_ref: &CodeRef,
        name: &str,
    ) -> anyhow::Result<Option<SValue>> {
        if let Some(const_var) = self.env.constant_vars.get(name) {
            trace!(
                "{}: translate_constant {}",
                code_ref.s(&self.env.file_idx),
                name
            );
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
            .declare_data(name, Linkage::Export, true, false)
            .expect("problem declaring data object");
        let local_id = self.module.declare_data_in_func(sym, self.builder.func);
        let global_val = self.builder.create_global_value(GlobalValueData::Load {
            base: local_id,
            offset: Offset32::new(0),
            global_type: data_type,
            readonly: false,
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
        // TODO avoid unnecessary allocation
        /*
            if this is the RHS of assignment
            and the LHS var is already allocated (like in a func param or return)
                then use that address, and don't create a stack slot here
            (later we could also look at vars declared higher up in the func
                and not reallocate for those either)

            additionally, if there are fields in the struct that would
            create a stack slot (like an array) and those are created when
            this struct is initialized, then use the space that is already
            allocated in this struct, instead of allocating it separately
            and copying. Note that this can only be done with fix sized
            fields. (like fixed arrays or sub structs)

            eventually, possibly look at expressions that are further down
            the ast like: if a {[1.0;1000]} else {[2.0;1000]}
        */

        let struct_address = self.alloc(self.env.struct_map[struct_name].size);

        for field in fields.iter() {
            let dst_field_def = &self.env.struct_map[struct_name].fields[&field.field_name].clone();

            let offset = self
                .builder
                .ins()
                .iconst(types::I64, dst_field_def.offset as i64);

            let field_address = self.builder.ins().iadd(struct_address, offset);

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
                self.builder.emit_small_memory_copy(
                    self.module.target_config(),
                    field_address,
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

                self.builder
                    .ins()
                    .store(MemFlags::new(), val, field_address, Offset32::new(0));
            }
        }

        Ok(SValue::Struct(struct_name.to_string(), struct_address))
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
                anyhow::bail!("variable {} is not a struct type {:?}", lhs_val, parts)
            }
        } else {
            let svar = self.get_variable(&CodeRef::z(), &parts[0])?.clone();
            if let SVariable::Struct(_var_name, vstruct_name, var, _return_struct) = svar {
                let base_struct_var_ptr = self.builder.use_var(var);
                start = 1;
                struct_name = vstruct_name;
                base_struct_var_ptr
            } else {
                anyhow::bail!("variable {} is not a struct type {:?}", svar, parts)
            }
        };

        let mut parent_struct_field = &self.env.struct_map[&struct_name].fields[&parts[start]];
        let mut offset = parent_struct_field.offset;
        if parts.len() > 2 {
            offset = 0;
            for part in parts.iter().skip(start) {
                if let ExprType::Struct(_code_ref, _name) = &parent_struct_field.expr_type {
                    parent_struct_field = &self.env.struct_map[&struct_name].fields[part];
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
            trace!(
                "{}: ExprType::Struct {}",
                code_ref.s(&self.env.file_idx),
                name
            );
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

    fn call_panic(&mut self, code_ref: &CodeRef, message: &str) -> anyhow::Result<()> {
        let arg_values = vec![Expr::LiteralString(*code_ref, message.to_string())];
        self.translate_call(code_ref, "panic", &arg_values, None, false)?;
        Ok(())
    }

    fn call_with_svalues(
        &mut self,
        code_ref: &CodeRef,
        fn_name: &str,
        func: Option<&Function>,
        closure_src_scope_name: Option<String>,
        is_closure: bool,
        is_temp_closure: bool,
        arg_svalues: Vec<SValue>,
        stack_slot_return: Option<usize>,
    ) -> anyhow::Result<SValue> {
        let fn_name = &fn_name.to_string();
        trace!(
            "{} call_with_svalues: {} is_closure: {} is_temp_closure: {} closure_src_scope_name: {:?} stack_slot_return: {}",
            code_ref.s(&self.env.file_idx),
            &fn_name,
            is_closure,
            is_temp_closure,
            closure_src_scope_name,
            stack_slot_return.is_some()
        );
        if fn_name == "unsized" {
            if let Some(v) = sarus_std_lib::translate_std(
                self.module.target_config().pointer_type(),
                &mut self.builder,
                code_ref,
                fn_name,
                &arg_svalues,
            )? {
                return Ok(v);
            }
        }
        let func = if let Some(func) = func {
            func
        } else {
            anyhow::bail!(
                "{} function {} not found",
                code_ref.s(&self.env.file_idx),
                fn_name
            )
        };

        let mut inline_function_requested = match func.inline {
            InlineKind::Default => false, //TODO make default still inline if it seems worth it
            InlineKind::Never => false,
            InlineKind::Always => true,
            InlineKind::Often => true, //TODO make often not inline if it's an issue
        };

        //let inline_function_requested = true; //make everything that not external inline

        if inline_function_requested && func.name != "~anon~" {
            //make sure we aren't recursively inlining this
            for f in &self.func_stack {
                if f.name == func.name {
                    warn!(
                        "{} function {} is called inline recursively, switching to non-inline call",
                        code_ref.s(&self.env.file_idx),
                        func.name
                    );
                    if let InlineKind::Always = func.inline {
                        anyhow::bail!("{} function {} is called always_inline recursively. always_inline recursion is not supported.",
                        code_ref.s(&self.env.file_idx),
                        func.name)
                    }
                    inline_function_requested = false;
                }
            }
        }

        let inline_function = (inline_function_requested && !func.extern_func) || is_closure;

        if func.params.len() != arg_svalues.len() {
            anyhow::bail!(
                "function call to {} has {} args, but function description has {}",
                fn_name,
                arg_svalues.len(),
                func.params.len()
            )
        }

        if func.extern_func {
            if let Some(v) = sarus_std_lib::translate_std(
                self.module.target_config().pointer_type(),
                &mut self.builder,
                code_ref,
                fn_name,
                &arg_svalues,
            )? {
                return Ok(v);
            }
        }

        let mut arg_values = Vec::new();

        for arg_svalue in &arg_svalues {
            let v = if let SValue::Void = arg_svalue {
                continue;
            } else {
                arg_svalue.inner("call_with_svalues")?
            };
            arg_values.push(v)
        }

        let mut sig = if !inline_function {
            Some(self.module.make_signature())
        } else {
            None
        };

        let ptr_ty = self.module.target_config().pointer_type();

        if let Some(sig) = &mut sig {
            for val in arg_values.iter() {
                sig.params
                    .push(AbiParam::new(self.builder.func.dfg.value_type(*val)));
            }
        }

        let stack_slot_address = if let Some(size) = &stack_slot_return {
            let stack_slot_address = self.alloc(*size);
            arg_values.insert(0, stack_slot_address);
            if let Some(sig) = &mut sig {
                sig.params
                    .insert(0, AbiParam::special(ptr_ty, ArgumentPurpose::StructReturn));
            }
            Some(stack_slot_address)
        } else {
            None
        };

        if stack_slot_return.is_none() {
            if let Some(sig) = &mut sig {
                for ret_arg in &func.returns {
                    sig.returns.push(AbiParam::new(
                        ret_arg.expr_type.cranelift_type(ptr_ty, false)?,
                    ));
                }
            }
        }

        let res = if inline_function {
            trace!(
                "{} inlining function {}",
                code_ref.s(&self.env.file_idx),
                &func.name
            );

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
            self.func_stack.push(func.clone()); //push inlined func onto func_stack
            self.variables.push(HashMap::new()); //new variable scope for inline func
            if self.use_deep_stack {
                self.deep_stack_widths.push(0);
            }

            if let Some(closure_src_scope_name) = closure_src_scope_name {
                //closures need the variables form the the scope they close over to be included
                let mut vars_to_expose = HashMap::new();

                let mut scope_closure_src_pos = None;

                //Find func in the stack that is the closure src
                for (i, func) in self.func_stack.iter().enumerate() {
                    if func.name == closure_src_scope_name {
                        scope_closure_src_pos = Some(i);
                        break;
                    }
                }

                let scope_closure_src_pos = if let Some(s) = scope_closure_src_pos {
                    s
                } else {
                    anyhow::bail!(
                        "{} could not find closure src scope {}",
                        code_ref.s(&self.env.file_idx),
                        closure_src_scope_name
                    )
                };

                // append the closed over scope to the closure's variables
                for (k, v) in &self.variables[scope_closure_src_pos] {
                    vars_to_expose.insert(k.to_string(), v.clone());
                    trace!("adding var {} to closure {}", k, func.name);
                }
                for (k, v) in vars_to_expose {
                    self.variables.last_mut().unwrap().insert(k, v);
                }

                let scope_src_func_name = &self.func_stack[scope_closure_src_pos].name;

                // Make any other closures in the callee available to the closure
                if let Some(scope_src_func_closures) =
                    self.env.inline_closures.get(scope_src_func_name)
                {
                    //If func does not have any closures, it wont have a hashmap under its name
                    if !self.env.temp_inline_closures.contains_key(&func.name) {
                        self.env
                            .temp_inline_closures
                            .insert(func.name.to_string(), HashMap::new());
                    }
                    for (k, closure) in scope_src_func_closures {
                        self.env
                            .temp_inline_closures
                            .get_mut(&func.name)
                            .unwrap()
                            .insert(k.clone(), closure.clone());
                    }
                }
            }

            // inlined functions will not have variables already declared
            declare_param_and_return_variables(
                &mut self.var_index,
                &mut self.builder,
                &mut self.module,
                func,
                self.entry_block,
                self.variables.last_mut().unwrap(),
                &Some(arg_values),
            )?;

            // translate inline func body
            for expr in &func.body {
                self.translate_expr(expr)?;
            }

            // get values from return variables
            let mut _return = Vec::new();
            for ret in &func.returns {
                let v = self.variables.last().unwrap()[&ret.name].inner();
                _return.push(self.builder.use_var(v))
            }

            //finished with inline scope
            self.func_stack.pop();
            self.variables.pop();
            self.dealloc_deep_stack();

            _return
        } else if let Some(sig) = sig {
            let callee = self
                .module
                .declare_function(fn_name, Linkage::Import, &sig)
                .expect("problem declaring function");
            let local_callee = self.module.declare_func_in_func(callee, self.builder.func);
            let call = self.builder.ins().call(local_callee, &arg_values);
            self.builder.inst_results(call).to_vec()
        } else {
            anyhow::bail!("Expected sig")
        };

        if let Some(stack_slot_address) = stack_slot_address {
            let expr_type = &func.returns.first().unwrap().expr_type;
            match expr_type {
                ExprType::Array(_, _, ArraySizedExpr::Fixed(..)) => Ok(SValue::from(
                    &mut self.builder,
                    expr_type,
                    stack_slot_address,
                )?),
                ExprType::Struct(_, struct_name) => {
                    Ok(SValue::Struct(struct_name.to_string(), stack_slot_address))
                }
                _ => {
                    anyhow::bail!("expected fixed array or struct return")
                }
            }
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

    fn alloc(&mut self, width: usize) -> Value {
        // if > 4KB //TODO make adjustable
        if width > 4096 && self.use_deep_stack {
            self.alloc_deep_stack(width)
        } else {
            self.alloc_stack(width)
        }
    }

    fn alloc_stack(&mut self, width: usize) -> Value {
        trace!("alloc_stack {}", width);
        let stack_slot = self.builder.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            width as u32,
        ));
        self.builder
            .ins()
            .stack_addr(self.ptr_ty, stack_slot, Offset32::new(0))
    }

    fn alloc_deep_stack(&mut self, width: usize) -> Value {
        //TODO consider bumping downwards (possible alignment issues?)
        //https://fitzgeraldnick.com/2019/11/01/always-bump-downwards.html

        trace!("alloc_deep_stack {}", width);
        *self.deep_stack_widths.last_mut().unwrap() += width;
        self.max_deep_stack_size = self
            .max_deep_stack_size
            .max(self.deep_stack_widths.iter().sum());
        trace!("max_deep_stack_size {}", self.max_deep_stack_size);

        let deep_stack_address = self.translate_global_data_addr(self.ptr_ty, "__DEEP_STACK__");

        let address = self.builder.ins().load(
            self.ptr_ty,
            MemFlags::new(),
            deep_stack_address,
            Offset32::new(0),
        );

        let alloc_size_val = self.builder.ins().iconst(types::I64, width as i64);

        let address_new_start = self.builder.ins().iadd(address, alloc_size_val);

        self.builder.ins().store(
            MemFlags::new(),
            address_new_start,
            deep_stack_address,
            Offset32::new(0),
        );
        address
    }

    pub fn dealloc_deep_stack(&mut self) {
        if self.use_deep_stack {
            let width_to_remove = self.deep_stack_widths.pop().unwrap();
            trace!("alloc_deep_stack width_to_remove {}", width_to_remove);
            let deep_stack_address = self.translate_global_data_addr(self.ptr_ty, "__DEEP_STACK__");

            let stack_slot_address = self.builder.ins().load(
                self.ptr_ty,
                MemFlags::new(),
                deep_stack_address,
                Offset32::new(0),
            );

            let alloc_size_val = self
                .builder
                .ins()
                .iconst(types::I64, width_to_remove as i64);

            let address_new_start = self.builder.ins().isub(stack_slot_address, alloc_size_val);

            self.builder.ins().store(
                MemFlags::new(),
                address_new_start,
                deep_stack_address,
                Offset32::new(0),
            );
        }
    }
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

#[derive(Debug, Clone)]
pub struct Closure {
    pub func: Function,
    pub src_scope: String,
}

pub fn setup_inline_closures(
    func_name: &str,
    stmts: &[Expr],
    inline_closures: &mut HashMap<String, HashMap<String, Closure>>,
) {
    for stmt in stmts {
        match stmt {
            Expr::Declaration(_coderef, Declaration::Function(closure_fn)) => {
                if !inline_closures.contains_key(func_name) {
                    inline_closures.insert(func_name.to_string(), HashMap::new());
                }
                inline_closures.get_mut(func_name).unwrap().insert(
                    closure_fn.name.clone(),
                    Closure {
                        func: closure_fn.clone(),
                        src_scope: func_name.to_string(),
                    },
                );
                setup_inline_closures(&closure_fn.name, &closure_fn.body, inline_closures);
            }
            _ => continue,
        }
    }
}
