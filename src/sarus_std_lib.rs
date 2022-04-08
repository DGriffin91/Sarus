use std::collections::{HashMap, HashSet};
use std::slice;

use crate::frontend::Expr;
use crate::function_translator::SVariable;
use crate::jit::{ArraySized, Env, SValue, StructDef};
use crate::validator::{bool_t, f32_t, i64_t, str_t, u8_t, ArraySizedExpr, ExprType, TypeError};
use crate::{
    decl,
    frontend::{Arg, CodeRef, Declaration, Function},
};
use crate::{hashmap, make_decl};
use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::{types, InstBuilder};
use cranelift_jit::JITBuilder;

#[repr(C)]
pub struct SarusSlice<T> {
    start_ptr: *const T,
    len: i64,
    cap: i64,
}

impl<T> SarusSlice<T> {
    #[inline]
    pub fn slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.start_ptr, self.len as usize) }
    }
    #[inline]
    pub fn len(&self) -> i64 {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len <= 0
    }
    #[inline]
    pub fn cap(&self) -> i64 {
        self.cap
    }
}

#[inline]
fn utf8<'a>(s: &'a SarusSlice<u8>) -> &'a str {
    std::str::from_utf8(s.slice()).unwrap()
}

extern "C" fn f32_print(x: f32) {
    print!("{}", x);
}

extern "C" fn i64_print(x: i64) {
    print!("{}", x);
}

extern "C" fn u8_print(x: u8) {
    print!("{}", x);
}

extern "C" fn bool_print(x: bool) {
    print!("{}", x);
}

extern "C" fn str_print(s: SarusSlice<u8>) {
    print!("{}", std::str::from_utf8(s.slice()).unwrap());
}

extern "C" fn f32_println(x: f32) {
    println!("{}", x);
}

extern "C" fn i64_println(x: i64) {
    println!("{}", x);
}

extern "C" fn u8_println(x: u8) {
    println!("{}", x);
}

extern "C" fn bool_println(x: bool) {
    println!("{}", x);
}

extern "C" fn str_println(s: SarusSlice<u8>) {
    println!("{}", std::str::from_utf8(s.slice()).unwrap());
}

extern "C" fn f32_assert_eq(x: f32, y: f32) {
    assert_eq!(x, y);
}

extern "C" fn i64_assert_eq(x: i64, y: i64) {
    assert_eq!(x, y);
}

extern "C" fn u8_assert_eq(x: u8, y: u8) {
    assert_eq!(x, y);
}

extern "C" fn bool_assert_eq(x: bool, y: bool) {
    assert_eq!(x, y);
}

extern "C" fn str_assert_eq(s1: SarusSlice<u8>, s2: SarusSlice<u8>) {
    let s1 = std::str::from_utf8(s1.slice()).unwrap();
    let s2 = std::str::from_utf8(s2.slice()).unwrap();
    assert_eq!(s1, s2);
}

extern "C" fn spanic(s: SarusSlice<u8>) {
    panic!("{}", std::str::from_utf8(s.slice()).unwrap())
}

#[rustfmt::skip]
pub fn append_std_math(
    prog: &mut Vec<Declaration>,
    jit_builder: &mut JITBuilder,
) {

    let jb = jit_builder;
    decl!(prog, jb, "f32.signum",           f32::signum,           (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.copysign",         f32::copysign,         (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.mul_add",          f32::mul_add,          (f32_t(), f32_t(), f32_t()), (f32_t()));
    decl!(prog, jb, "f32.div_euclid",       f32::div_euclid,       (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.rem_euclid",       f32::rem_euclid,       (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.powi",             f32::powi,             (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.powf",             f32::powf,             (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.sqrt",             f32::sqrt,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.exp",              f32::exp,              (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.exp2",             f32::exp2,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.ln",               f32::ln,               (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.log",              f32::log,              (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.log2",             f32::log2,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.log10",            f32::log10,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.cbrt",             f32::cbrt,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.hypot",            f32::hypot,            (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.sin",              f32::sin,              (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.tan",              f32::tan,              (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.cos",              f32::cos,              (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.asin",             f32::asin,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.acos",             f32::acos,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.atan",             f32::atan,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.atan2",            f32::atan2,            (f32_t(), f32_t()),          (f32_t()));
    decl!(prog, jb, "f32.sin_cos",          f32::sin_cos,          (f32_t()),                   (f32_t(), f32_t()));
    decl!(prog, jb, "f32.exp_m1",           f32::exp_m1,           (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.ln_1p",            f32::ln_1p,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.sinh",             f32::sinh,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.cosh",             f32::cosh,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.tanh",             f32::tanh,             (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.asinh",            f32::asinh,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.acosh",            f32::acosh,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.atanh",            f32::atanh,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.is_nan",           f32::is_nan,           (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_infinite",      f32::is_infinite,      (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_finite",        f32::is_finite,        (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_subnormal",     f32::is_subnormal,     (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_normal",        f32::is_normal,        (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_sign_positive", f32::is_sign_positive, (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.is_sign_negative", f32::is_sign_negative, (f32_t()),                   (bool_t()));
    decl!(prog, jb, "f32.recip",            f32::recip,            (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.to_degrees",       f32::to_degrees,       (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.to_radians",       f32::to_radians,       (f32_t()),                   (f32_t()));
    decl!(prog, jb, "f32.cosh",             f32::cosh,             (f32_t()),                   (f32_t()));
    /*TODO
    pub fn to_bits(self) -> u64
    pub fn from_bits(v: u64) -> f32
    pub fn to_be_bytes(self) -> [u8; 8]
    pub fn to_le_bytes(self) -> [u8; 8]
    pub fn to_ne_bytes(self) -> [u8; 8]
    pub fn from_be_bytes(bytes: [u8; 8]) -> f32
    pub fn from_le_bytes(bytes: [u8; 8]) -> f32
    pub fn from_ne_bytes(bytes: [u8; 8]) -> f32
    */
    /* Using Cranelift directly:
    "f32.trunc"
    "f32.floor"
    "f32.ceil"
    "f32.fract"
    "f32.abs"
    "f32.round"
    "f32.i64"
    "f32.min"
    "f32.max"
    */
}

#[rustfmt::skip]
pub fn append_std(prog: &mut Vec<Declaration>, jit_builder: &mut JITBuilder) {
    let jb = jit_builder;
    for n in ["f32.ceil", "f32.floor", "f32.trunc", "f32.fract", "f32.abs", "f32.round"] {
        prog.push(make_decl(n, vec![("x", f32_t())], vec![("y", f32_t())]));
    }

    prog.push(make_decl("f32.i64", vec![("x", f32_t())], vec![("y", i64_t())]));
    prog.push(make_decl("i64.f32", vec![("x", i64_t())], vec![("y", f32_t())]));
    prog.push(make_decl("u8.f32", vec![("x", u8_t())], vec![("y", f32_t())]));
    prog.push(make_decl("u8.i64", vec![("x", u8_t())], vec![("y", i64_t())]));
    prog.push(make_decl("f32.u8", vec![("x", f32_t())], vec![("y", u8_t())]));
    prog.push(make_decl("i64.u8", vec![("x", i64_t())], vec![("y", u8_t())]));

    for n in ["f32.min", "f32.max"] {
        prog.push(make_decl(
            n,
            vec![("x", f32_t()), ("y", f32_t())],
            vec![("z", f32_t())],
        ));
    }
    for n in ["i64.min", "i64.max"] {
        prog.push(make_decl(
            n,
            vec![("x", i64_t()), ("y", i64_t())],
            vec![("z", i64_t())],
        ));
    }

    for n in ["u8.min", "u8.max"] {
        prog.push(make_decl(
            n,
            vec![("x", u8_t()), ("y", u8_t())],
            vec![("z", u8_t())],
        ));
    }


    decl!(prog, jb, "f32.print",           f32_print,           (f32_t()),                 ());
    decl!(prog, jb, "i64.print",           i64_print,           (i64_t()),                 ());
    decl!(prog, jb, "u8.print",            u8_print,            (u8_t()),                  ());
    decl!(prog, jb, "bool.print",          bool_print,          (bool_t()),                ());
    decl!(prog, jb, "[u8].print",          str_print,           (str_t()),                 ());

    decl!(prog, jb, "f32.println",         f32_println,         (f32_t()),                 ());
    decl!(prog, jb, "i64.println",         i64_println,         (i64_t()),                 ());
    decl!(prog, jb, "u8.println",          u8_println,          (u8_t()),                  ());
    decl!(prog, jb, "bool.println",        bool_println,        (bool_t()),                ());
    decl!(prog, jb, "[u8].println",        str_println,         (str_t()),                 ());

    
    decl!(prog, jb, "f32.assert_eq",       f32_assert_eq,       (f32_t(), f32_t()),        ());
    decl!(prog, jb, "i64.assert_eq",       i64_assert_eq,       (i64_t(), i64_t()),        ());
    decl!(prog, jb, "u8.assert_eq",        u8_assert_eq,        (u8_t(), u8_t()),          ());
    decl!(prog, jb, "bool.assert_eq",      bool_assert_eq,      (bool_t(), bool_t()),      ());
    decl!(prog, jb, "[u8].assert_eq",      str_assert_eq,       (str_t(), str_t()),        ());
    
    decl!(prog, jb, "panic",               spanic,              (str_t()),                 ());
    
    prog.push(make_decl("src_line", vec![], vec![("line", i64_t())]));

}

extern "C" fn str_find(s: SarusSlice<u8>, pat: SarusSlice<u8>) -> i64 {
    if let Some(n) = utf8(&s).find(utf8(&pat)) {
        n as i64
    } else {
        -1
    }
}

extern "C" fn str_rfind(s: SarusSlice<u8>, pat: SarusSlice<u8>) -> i64 {
    if let Some(n) = utf8(&s).rfind(utf8(&pat)) {
        n as i64
    } else {
        -1
    }
}

extern "C" fn str_starts_with(s: SarusSlice<u8>, pat: SarusSlice<u8>) -> bool {
    utf8(&s).starts_with(utf8(&pat))
}

extern "C" fn str_ends_with(s: SarusSlice<u8>, pat: SarusSlice<u8>) -> bool {
    utf8(&s).ends_with(utf8(&pat))
}

pub fn append_std_strings(prog: &mut Vec<Declaration>, jit_builder: &mut JITBuilder) {
    let jb = jit_builder;

    prog.push(make_decl(
        "[u8].starts_with",
        vec![("src", str_t()), ("pattern", str_t())],
        vec![("result", bool_t())],
    ));
    jb.symbol("[u8].starts_with", str_starts_with as *const u8);

    prog.push(make_decl(
        "[u8].ends_with",
        vec![("src", str_t()), ("pattern", str_t())],
        vec![("result", bool_t())],
    ));
    jb.symbol("[u8].ends_with", str_ends_with as *const u8);

    prog.push(make_decl(
        "[u8].find",
        vec![("src", str_t()), ("pattern", str_t())],
        vec![("position", i64_t())],
    ));
    jb.symbol("[u8].find", str_find as *const u8);

    prog.push(make_decl(
        "[u8].rfind",
        vec![("src", str_t()), ("pattern", str_t())],
        vec![("position", i64_t())],
    ));
    jb.symbol("[u8].rfind", str_rfind as *const u8);
}

pub(crate) fn check_core_generics(fn_name: &str, impl_val: Option<SValue>) -> bool {
    if HashSet::from(["push", "pop", "len", "cap", "append"]).contains(fn_name) {
        if let Some(SValue::Array(_sval, ArraySized::Slice)) = &impl_val {
            return true;
        }
    }
    if fn_name == "len" {
        if let Some(SValue::Array(_sval, ArraySized::Fixed(..))) = &impl_val {
            return true;
        }
    }
    HashSet::from(["unsized"]).contains(fn_name)
}

pub(crate) fn validate_core_generics(
    fn_name: &str,
    args: &Vec<Expr>,
    code_ref: &CodeRef,
    lhs_val: &Option<ExprType>,
    env: &Env,
    variables: &HashMap<String, SVariable>,
) -> Result<Option<ExprType>, TypeError> {
    if fn_name == "push" {
        if let Some(ExprType::Array(code_ref, expr_type, ArraySizedExpr::Slice)) = lhs_val {
            if args.len() != 1 {
                return Err(TypeError::TupleLengthMismatch {
                    c: code_ref.s(&env.file_idx),
                    actual: args.len(),
                    expected: 1,
                });
            }
            let targ = ExprType::of(&args[0], env, fn_name, variables)?;
            if **expr_type != targ {
                return Err(TypeError::TypeMismatchSpecific {
                    c: code_ref.s(&env.file_idx),
                    s: format!(
                        "function {} expected parameter {} to be of type {} but type {} was found",
                        fn_name, 1, expr_type, targ
                    ),
                });
            }
            return Ok(Some(ExprType::Void(*code_ref)));
        }
    }
    if fn_name == "pop" {
        if let Some(ExprType::Array(code_ref, expr_type, ArraySizedExpr::Slice)) = lhs_val {
            if !args.is_empty() {
                return Err(TypeError::TupleLengthMismatch {
                    c: code_ref.s(&env.file_idx),
                    actual: args.len(),
                    expected: 0,
                });
            }
            return Ok(Some(*expr_type.clone()));
        }
    }
    if fn_name == "len" {
        if let Some(lhs_val) = lhs_val {
            if let ExprType::Array(
                code_ref,
                _expr_type,
                ArraySizedExpr::Slice | ArraySizedExpr::Fixed(..),
            ) = lhs_val
            {
                if !args.is_empty() {
                    return Err(TypeError::TupleLengthMismatch {
                        c: code_ref.s(&env.file_idx),
                        actual: args.len(),
                        expected: 0,
                    });
                }
                return Ok(Some(ExprType::I64(*code_ref)));
            }
        }
    }
    if fn_name == "cap" {
        if let Some(lhs_val) = lhs_val {
            if let ExprType::Array(code_ref, _expr_type, ArraySizedExpr::Slice) = lhs_val {
                if args.len() != 0 {
                    return Err(TypeError::TupleLengthMismatch {
                        c: code_ref.s(&env.file_idx),
                        actual: args.len(),
                        expected: 0,
                    });
                }
                return Ok(Some(ExprType::I64(*code_ref)));
            }
        }
    }
    if fn_name == "append" {
        if let Some(lhs_val) = lhs_val {
            if let ExprType::Array(code_ref, expr_type, ArraySizedExpr::Slice) = lhs_val {
                if args.len() != 1 {
                    return Err(TypeError::TupleLengthMismatch {
                        c: code_ref.s(&env.file_idx),
                        actual: args.len(),
                        expected: 1,
                    });
                }
                let targ = ExprType::of(&args[0], env, fn_name, variables)?;
                if let ExprType::Array(code_ref, ref arg_expr_type, ArraySizedExpr::Fixed(_len)) =
                    targ
                {
                    if *expr_type != *arg_expr_type {
                        return Err(TypeError::TypeMismatchSpecific {
                                        c: code_ref.s(&env.file_idx),
                                        s: format!("function {} expected parameter {} to be slice or fixed array of type {} but type {} was found", fn_name, 1, lhs_val, targ)
                                    });
                    }
                    return Ok(Some(ExprType::Void(code_ref)));
                }
                if *lhs_val != targ {
                    return Err(TypeError::TypeMismatchSpecific {
                                    c: code_ref.s(&env.file_idx),
                                    s: format!("function {} expected parameter {} to be slice or fixed array of type {} but type {} was found", fn_name, 1, lhs_val, targ)
                                });
                }
                return Ok(Some(ExprType::Void(*code_ref)));
            }
        }
    }
    if fn_name == "unsized" {
        if args.len() != 0 {
            return Err(TypeError::TupleLengthMismatch {
                c: code_ref.s(&env.file_idx),
                actual: args.len(),
                expected: 0,
            });
        }

        if let Some(lhs_val) = lhs_val {
            return match lhs_val {
                ExprType::Array(c, expr, _) => Ok(Some(ExprType::Array(
                    *c,
                    expr.clone(),
                    ArraySizedExpr::Unsized,
                ))),
                ExprType::Address(c) => Ok(Some(ExprType::Array(
                    *c,
                    Box::new(lhs_val.clone()),
                    ArraySizedExpr::Unsized,
                ))),
                sv => Err(TypeError::TypeMismatchSpecific {
                    c: code_ref.s(&env.file_idx),
                    s: format!("function unsized does not support {}", sv),
                }),
            };
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

pub(crate) fn translate_std(
    _ptr_ty: cranelift::prelude::Type,
    builder: &mut FunctionBuilder,
    code_ref: &CodeRef,
    name: &str,
    args: &[SValue],
) -> anyhow::Result<Option<SValue>> {
    macro_rules! v {
        ($e:expr) => {
            $e.inner("translate_std")?
        };
    }
    Ok(match name {
        "unsized" => match &args[0] {
            SValue::Array(sval, _) => Some(SValue::Array(sval.clone(), ArraySized::Unsized)),
            SValue::Address(_) => Some(SValue::Array(
                Box::new(args[0].clone()),
                ArraySized::Unsized,
            )),
            sv => anyhow::bail!("unsized does not support {}", sv),
        },
        "f32.trunc" => Some(SValue::F32(builder.ins().trunc(v!(args[0])))),
        "f32.floor" => Some(SValue::F32(builder.ins().floor(v!(args[0])))),
        "f32.ceil" => Some(SValue::F32(builder.ins().ceil(v!(args[0])))),
        "f32.fract" => {
            let v_int = builder.ins().trunc(v!(args[0]));
            let v = builder.ins().fsub(v!(args[0]), v_int);
            Some(SValue::F32(v))
        }
        "f32.abs" => Some(SValue::F32(builder.ins().fabs(v!(args[0])))),
        "f32.round" => Some(SValue::F32(builder.ins().nearest(v!(args[0])))),
        "f32.i64" => Some(SValue::I64(
            builder.ins().fcvt_to_sint(types::I64, v!(args[0])),
        )),
        "f32.u8" => {
            let i_val = builder.ins().fcvt_to_sint(types::I32, v!(args[0]));
            Some(SValue::U8(builder.ins().ireduce(types::I8, i_val)))
        }
        "i64.f32" => Some(SValue::F32(
            builder.ins().fcvt_from_sint(types::F32, v!(args[0])),
        )),
        "i64.u8" => Some(SValue::U8(builder.ins().ireduce(types::I8, v!(args[0])))),
        "u8.f32" => Some(SValue::F32(
            builder.ins().fcvt_from_uint(types::F32, v!(args[0])),
        )),
        "u8.i64" => Some(SValue::I64(builder.ins().uextend(types::I64, v!(args[0])))),
        "f32.min" => Some(SValue::F32(builder.ins().fmin(v!(args[0]), v!(args[1])))),
        "f32.max" => Some(SValue::F32(builder.ins().fmax(v!(args[0]), v!(args[1])))),
        "i64.min" => Some(SValue::I64(builder.ins().imin(v!(args[0]), v!(args[1])))),
        "i64.max" => Some(SValue::I64(builder.ins().imax(v!(args[0]), v!(args[1])))),
        "u8.min" => Some(SValue::U8(builder.ins().umax(v!(args[0]), v!(args[1])))),
        "u8.max" => Some(SValue::U8(builder.ins().umin(v!(args[0]), v!(args[1])))),
        "src_line" => {
            let line = code_ref.line.unwrap_or(0) as i64;
            Some(SValue::I64(builder.ins().iconst(types::I64, line)))
        }
        _ => None,
    })
}

#[derive(Debug, Clone, Copy)]
pub enum SConstant {
    Address(i64),
    F32(f32),
    I64(i64),
    Bool(bool),
}

impl SConstant {
    pub fn expr_type(&self, code_ref: Option<CodeRef>) -> ExprType {
        let code_ref = if let Some(code_ref) = code_ref {
            code_ref
        } else {
            CodeRef::default()
        };
        match self {
            SConstant::Address(_) => ExprType::Address(code_ref),
            SConstant::F32(_) => ExprType::F32(code_ref),
            SConstant::I64(_) => ExprType::I64(code_ref),
            SConstant::Bool(_) => ExprType::Bool(code_ref),
        }
    }
}

pub fn get_constants(struct_map: &HashMap<String, StructDef>) -> HashMap<String, SConstant> {
    use std::f32::consts::*;
    let mut constants = hashmap!(
        "E".into() => SConstant::F32(E),
        "FRAC_1_PI".into() => SConstant::F32(FRAC_1_PI),
        "FRAC_1_SQRT_2".into() => SConstant::F32(FRAC_1_SQRT_2),
        "FRAC_2_SQRT_PI".into() => SConstant::F32(FRAC_2_SQRT_PI),
        "FRAC_PI_2".into() => SConstant::F32(FRAC_PI_2),
        "FRAC_PI_3".into() => SConstant::F32(FRAC_PI_3),
        "FRAC_PI_4".into() => SConstant::F32(FRAC_PI_4),
        "FRAC_PI_6".into() => SConstant::F32(FRAC_PI_6),
        "FRAC_PI_8".into() => SConstant::F32(FRAC_PI_8),
        "LN_2".into() => SConstant::F32(LN_2),
        "LN_10".into() => SConstant::F32(LN_10),
        "LOG2_10".into() => SConstant::F32(LOG2_10),
        "LOG2_E".into() => SConstant::F32(LOG2_E),
        "LOG10_2".into() => SConstant::F32(LOG10_2),
        "LOG10_E".into() => SConstant::F32(LOG10_E),
        "PI".into() => SConstant::F32(PI),
        "SQRT_2".into() => SConstant::F32(SQRT_2),
        "TAU".into() => SConstant::F32(TAU),
        "f32::size".into() => SConstant::I64(types::F32.bytes() as i64),
        "i64::size".into() => SConstant::I64(types::I64.bytes() as i64),
        "bool::size".into() => SConstant::I64(types::I8.bytes() as i64) //for extern and structs we use I8 for bool

    );
    for (name, def) in struct_map {
        constants.insert(format!("{}::size", name), SConstant::I64(def.size as i64));
    }
    constants
}
