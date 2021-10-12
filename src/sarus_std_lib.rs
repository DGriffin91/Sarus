use std::collections::HashMap;
use std::ffi::CStr;

use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::{types, InstBuilder, Value};
use cranelift_jit::JITBuilder;

use crate::frontend::{Arg, CodeRef};
use crate::jit::{SValue, StructDef};
use crate::validator::{address_t, bool_t, f64_t, i64_t, ExprType};
use crate::{
    decl,
    frontend::{Declaration, Function},
};
use crate::{hashmap, make_decl};

//Reference: https://www.gnu.org/software/libc/manual/html_node/Mathematics.html
//https://docs.rs/libc/0.2.101/libc/
//should this include bessel functions? It seems like they would pollute the name space.

//couldn't get to work (STATUS_ACCESS_VIOLATION):
// "asinh", "acosh", "atanh", "erf", "erfc", "lgamma", "gamma", "tgamma", "exp2", "exp10", "log2"
const STD_1ARG_FF: [&str; 20] = [
    "sin", // libc
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "exp",
    "log",
    "log10",
    "sqrt",
    "sinh",
    "cosh",
    "exp10",
    "tanh",
    "f64.ceil", // built in std
    "f64.floor",
    "f64.trunc",
    "f64.fract",
    "f64.abs",
    "f64.round",
];
const STD_1ARG_IF: [&str; 1] = [
    "i64.f64", // built in std
];
const STD_1ARG_FI: [&str; 1] = [
    "f64.i64", // built in std
];

//couldn't get to work (STATUS_ACCESS_VIOLATION):
// "hypot", "expm1", "log1p"
const STD_2ARG_FF: [&str; 4] = [
    "atan2", "pow", // libc
    "f64.min", "f64.max", // built in std
];
const STD_2ARG_II: [&str; 2] = [
    "i64.min", "i64.max", // built in std
];

extern "C" fn f64_print(x: f64) {
    print!("{}", x);
}

extern "C" fn i64_print(x: i64) {
    print!("{}", x);
}

extern "C" fn bool_print(x: bool) {
    print!("{}", x);
}

extern "C" fn str_print(s: *const i8) {
    unsafe {
        print!("{}", CStr::from_ptr(s).to_str().unwrap());
    }
}

extern "C" fn f64_println(x: f64) {
    println!("{}", x);
}

extern "C" fn i64_println(x: i64) {
    println!("{}", x);
}

extern "C" fn bool_println(x: bool) {
    println!("{}", x);
}

extern "C" fn str_println(s: *const i8) {
    unsafe {
        println!("{}", CStr::from_ptr(s).to_str().unwrap());
    }
}

extern "C" fn f64_assert_eq(x: f64, y: f64) {
    assert_eq!(x, y);
}

extern "C" fn i64_assert_eq(x: i64, y: i64) {
    assert_eq!(x, y);
}

extern "C" fn bool_assert_eq(x: bool, y: bool) {
    assert_eq!(x, y);
}

extern "C" fn str_assert_eq(s1: *const i8, s2: *const i8) {
    unsafe {
        let s1 = CStr::from_ptr(s1).to_str().unwrap();
        let s2 = CStr::from_ptr(s2).to_str().unwrap();
        assert_eq!(s1, s2);
    }
}

extern "C" fn spanic(s: *const i8) {
    unsafe { panic!("{}", CStr::from_ptr(s).to_str().unwrap()) }
}

#[rustfmt::skip]
pub fn append_std_math(
    prog: &mut Vec<Declaration>,
    jit_builder: &mut JITBuilder,
) {
    let jb = jit_builder;
    decl!(prog, jb, "f64.signum",           f64::signum,           (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.copysign",         f64::copysign,         (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.mul_add",          f64::mul_add,          (f64_t(), f64_t(), f64_t()), (f64_t()));
    decl!(prog, jb, "f64.div_euclid",       f64::div_euclid,       (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.rem_euclid",       f64::rem_euclid,       (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.powi",             f64::powi,             (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.powf",             f64::powf,             (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.sqrt",             f64::sqrt,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.exp",              f64::exp,              (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.exp2",             f64::exp2,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.ln",               f64::ln,               (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.log",              f64::log,              (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.log2",             f64::log2,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.log10",            f64::log10,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.cbrt",             f64::cbrt,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.hypot",            f64::hypot,            (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.sin",              f64::sin,              (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.tan",              f64::tan,              (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.cos",              f64::cos,              (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.asin",             f64::asin,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.acos",             f64::acos,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.atan",             f64::atan,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.atan2",            f64::atan2,            (f64_t(), f64_t()),         (f64_t()));
    decl!(prog, jb, "f64.sin_cos",          f64::sin_cos,          (f64_t()),                 (f64_t(), f64_t()));
    decl!(prog, jb, "f64.exp_m1",           f64::exp_m1,           (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.ln_1p",            f64::ln_1p,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.sinh",             f64::sinh,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.cosh",             f64::cosh,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.tanh",             f64::tanh,             (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.asinh",            f64::asinh,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.acosh",            f64::acosh,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.atanh",            f64::atanh,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.is_nan",           f64::is_nan,           (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_infinite",      f64::is_infinite,      (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_finite",        f64::is_finite,        (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_subnormal",     f64::is_subnormal,     (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_normal",        f64::is_normal,        (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_sign_positive", f64::is_sign_positive, (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.is_sign_negative", f64::is_sign_negative, (f64_t()),                 (bool_t()));
    decl!(prog, jb, "f64.recip",            f64::recip,            (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.to_degrees",       f64::to_degrees,       (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.to_radians",       f64::to_radians,       (f64_t()),                 (f64_t()));
    decl!(prog, jb, "f64.cosh",             f64::cosh,             (f64_t()),                 (f64_t()));
    /*TODO
    pub fn to_bits(self) -> u64
    pub fn from_bits(v: u64) -> f64
    pub fn to_be_bytes(self) -> [u8; 8]
    pub fn to_le_bytes(self) -> [u8; 8]
    pub fn to_ne_bytes(self) -> [u8; 8]
    pub fn from_be_bytes(bytes: [u8; 8]) -> f64
    pub fn from_le_bytes(bytes: [u8; 8]) -> f64
    pub fn from_ne_bytes(bytes: [u8; 8]) -> f64
    */
    /* Using Cranelift directly:
    "f64.trunc"
    "f64.floor"
    "f64.ceil"
    "f64.fract"
    "f64.abs"
    "f64.round"
    "f64.i64"
    "f64.min"
    "f64.max"
    */
}

#[rustfmt::skip]
pub fn append_std(prog: &mut Vec<Declaration>, jit_builder: &mut JITBuilder) {
    let jb = jit_builder;
    for n in STD_1ARG_FF {
        prog.push(make_decl(n, vec![("x", f64_t())], vec![("y", f64_t())]));
    }
    for n in STD_1ARG_FI {
        prog.push(make_decl(n, vec![("x", f64_t())], vec![("y", i64_t())]));
    }
    for n in STD_1ARG_IF {
        prog.push(make_decl(n, vec![("x", i64_t())], vec![("y", f64_t())]));
    }
    for n in STD_2ARG_FF {
        prog.push(make_decl(
            n,
            vec![("x", f64_t()), ("y", f64_t())],
            vec![("z", f64_t())],
        ));
    }
    for n in STD_2ARG_II {
        prog.push(make_decl(
            n,
            vec![("x", i64_t()), ("y", i64_t())],
            vec![("z", i64_t())],
        ));
    }

    decl!(prog, jb, "f64.print",           f64_print,           (f64_t()),                 ());
    decl!(prog, jb, "i64.print",           i64_print,           (i64_t()),                 ());
    decl!(prog, jb, "bool.print",          bool_print,          (bool_t()),                ());
    decl!(prog, jb, "&.print",             str_print,           (address_t()),             ()); //TODO setup actual str type

    decl!(prog, jb, "f64.println",         f64_println,         (f64_t()),                 ());
    decl!(prog, jb, "i64.println",         i64_println,         (i64_t()),                 ());
    decl!(prog, jb, "bool.println",        bool_println,        (bool_t()),                ());
    decl!(prog, jb, "&.println",           str_println,         (address_t()),             ());

    
    decl!(prog, jb, "f64.assert_eq",       f64_assert_eq,       (f64_t(), f64_t()),        ());
    decl!(prog, jb, "i64.assert_eq",       i64_assert_eq,       (i64_t(), i64_t()),        ());
    decl!(prog, jb, "bool.assert_eq",      bool_assert_eq,      (bool_t(), bool_t()),      ());
    decl!(prog, jb, "&.assert_eq",         str_assert_eq,       (address_t(), address_t()),());
    
    decl!(prog, jb, "panic",               spanic,              (address_t()),             ());
    
    prog.push(make_decl("src_line", vec![], vec![("line", i64_t())]));

}

pub(crate) fn translate_std(
    _ptr_ty: cranelift::prelude::Type,
    builder: &mut FunctionBuilder,
    code_ref: &CodeRef,
    name: &str,
    args: &[Value],
) -> anyhow::Result<Option<SValue>> {
    match name {
        "f64.trunc" => Ok(Some(SValue::F64(builder.ins().trunc(args[0])))),
        "f64.floor" => Ok(Some(SValue::F64(builder.ins().floor(args[0])))),
        "f64.ceil" => Ok(Some(SValue::F64(builder.ins().ceil(args[0])))),
        "f64.fract" => {
            let v_int = builder.ins().trunc(args[0]);
            let v = builder.ins().fsub(args[0], v_int);
            Ok(Some(SValue::F64(v)))
        }
        "f64.abs" => Ok(Some(SValue::F64(builder.ins().fabs(args[0])))),
        "f64.round" => Ok(Some(SValue::F64(builder.ins().nearest(args[0])))),
        "f64.i64" => Ok(Some(SValue::I64(
            builder.ins().fcvt_to_sint(types::I64, args[0]),
        ))),
        "i64.f64" => Ok(Some(SValue::F64(
            builder.ins().fcvt_from_sint(types::F64, args[0]),
        ))),
        "f64.min" => Ok(Some(SValue::F64(builder.ins().fmin(args[0], args[1])))),
        "f64.max" => Ok(Some(SValue::F64(builder.ins().fmax(args[0], args[1])))),
        "i64.min" => Ok(Some(SValue::I64(builder.ins().imin(args[0], args[1])))),
        "i64.max" => Ok(Some(SValue::I64(builder.ins().imax(args[0], args[1])))),
        "src_line" => {
            let line = code_ref.line.unwrap_or(0) as i64;
            Ok(Some(SValue::I64(builder.ins().iconst(types::I64, line))))
        }
        _ => Ok(None),
    }
}

pub enum SConstant {
    F64(f64),
    I64(i64),
    Bool(bool),
}

impl SConstant {
    pub fn expr_type(&self, code_ref: Option<CodeRef>) -> ExprType {
        let code_ref = if let Some(code_ref) = code_ref {
            code_ref
        } else {
            CodeRef::z()
        };
        match self {
            SConstant::F64(_) => ExprType::F64(code_ref),
            SConstant::I64(_) => ExprType::I64(code_ref),
            SConstant::Bool(_) => ExprType::Bool(code_ref),
        }
    }
}

pub fn get_constants(struct_map: &HashMap<String, StructDef>) -> HashMap<String, SConstant> {
    use std::f64::consts::*;
    let mut constants = hashmap!(
        "E".into() => SConstant::F64(E),
        "FRAC_1_PI".into() => SConstant::F64(FRAC_1_PI),
        "FRAC_1_SQRT_2".into() => SConstant::F64(FRAC_1_SQRT_2),
        "FRAC_2_SQRT_PI".into() => SConstant::F64(FRAC_2_SQRT_PI),
        "FRAC_PI_2".into() => SConstant::F64(FRAC_PI_2),
        "FRAC_PI_3".into() => SConstant::F64(FRAC_PI_3),
        "FRAC_PI_4".into() => SConstant::F64(FRAC_PI_4),
        "FRAC_PI_6".into() => SConstant::F64(FRAC_PI_6),
        "FRAC_PI_8".into() => SConstant::F64(FRAC_PI_8),
        "LN_2".into() => SConstant::F64(LN_2),
        "LN_10".into() => SConstant::F64(LN_10),
        "LOG2_10".into() => SConstant::F64(LOG2_10),
        "LOG2_E".into() => SConstant::F64(LOG2_E),
        "LOG10_2".into() => SConstant::F64(LOG10_2),
        "LOG10_E".into() => SConstant::F64(LOG10_E),
        "PI".into() => SConstant::F64(PI),
        "SQRT_2".into() => SConstant::F64(SQRT_2),
        "TAU".into() => SConstant::F64(TAU),
        "f64::size".into() => SConstant::I64(types::F64.bytes() as i64),
        "i64::size".into() => SConstant::I64(types::I64.bytes() as i64),
        "bool::size".into() => SConstant::I64(types::I8.bytes() as i64) //for extern and structs we use I8 for bool

    );
    for (name, def) in struct_map {
        constants.insert(format!("{}::size", name), SConstant::I64(def.size as i64));
    }
    constants
}
