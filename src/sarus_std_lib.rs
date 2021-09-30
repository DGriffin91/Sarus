use std::collections::HashMap;
use std::ffi::CStr;

use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::{types, InstBuilder, Value};
use cranelift_jit::JITBuilder;

use crate::frontend::Arg;
use crate::hashmap;
use crate::jit::SValue;
use crate::{
    frontend::{Declaration, Function},
    validator::ExprType,
};

use crate::validator::ExprType as E;

fn decl(name: &str, params: Vec<(&str, ExprType)>, returns: Vec<(&str, ExprType)>) -> Declaration {
    Declaration::Function(Function {
        name: name.to_string(),
        params: params
            .into_iter()
            .map(|(name, expr)| Arg {
                name: name.to_string(),
                expr_type: Some(expr),
            })
            .collect(),
        returns: returns
            .into_iter()
            .map(|(name, expr)| Arg {
                name: name.to_string(),
                expr_type: Some(expr),
            })
            .collect(),
        body: vec![],
        extern_func: true,
    })
}

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

pub fn append_std_symbols(jit_builder: &mut JITBuilder) {
    jit_builder.symbols([
        ("f64.print", f64_print as *const u8),
        ("i64.print", i64_print as *const u8),
        ("bool.print", bool_print as *const u8),
        ("&.print", str_print as *const u8), //TODO setup actual str type
        ("f64.println", f64_println as *const u8),
        ("i64.println", i64_println as *const u8),
        ("bool.println", bool_println as *const u8),
        ("&.println", str_println as *const u8), //TODO setup actual str type
        ("f64.assert_eq", f64_assert_eq as *const u8),
        ("i64.assert_eq", i64_assert_eq as *const u8),
        ("bool.assert_eq", bool_assert_eq as *const u8),
        ("&.assert_eq", str_assert_eq as *const u8), //TODO setup actual str type
    ]);
}

#[macro_export]
macro_rules! decl {
    ( $prog:expr, $jit_builder:expr, $name:expr, $func:expr,  ($( $param:expr ),*), ($( $ret:expr ),*) ) => {
        {
            let mut params = Vec::new();
            $(
                params.push(Arg {
                    name: format!("in{}", params.len()),
                    expr_type: Some($param),
                });
            )*

            let mut returns = Vec::new();
            $(
                returns.push(Arg {
                    name: format!("out{}", returns.len()),
                    expr_type: Some($ret),
                });
            )*

            $jit_builder.symbol($name, $func as *const u8);

            $prog.push(Declaration::Function(Function {
                name: $name.to_string(),
                params,
                returns,
                body: vec![],
                extern_func: true,
            }))

        }
    };
}

#[rustfmt::skip]
pub fn append_std_math(
    prog: &mut Vec<Declaration>,
    jit_builder: &mut JITBuilder,
) {
    let jb = jit_builder;
    decl!(prog, jb, "f64.signum",           f64::signum,           (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.copysign",         f64::copysign,         (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.mul_add",          f64::mul_add,          (E::F64, E::F64, E::F64), (E::F64));
    decl!(prog, jb, "f64.div_euclid",       f64::div_euclid,       (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.rem_euclid",       f64::rem_euclid,       (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.powi",             f64::powi,             (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.powf",             f64::powf,             (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.sqrt",             f64::sqrt,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.exp",              f64::exp,              (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.exp2",             f64::exp2,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.ln",               f64::ln,               (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.log",              f64::log,              (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.log2",             f64::log2,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.log10",            f64::log10,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.cbrt",             f64::cbrt,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.hypot",            f64::hypot,            (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.sin",              f64::sin,              (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.tan",              f64::tan,              (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.cos",              f64::cos,              (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.asin",             f64::asin,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.acos",             f64::acos,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.atan",             f64::atan,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.atan2",            f64::atan2,            (E::F64, E::F64),         (E::F64));
    decl!(prog, jb, "f64.sin_cos",          f64::sin_cos,          (E::F64),                 (E::F64, E::F64));
    decl!(prog, jb, "f64.exp_m1",           f64::exp_m1,           (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.ln_1p",            f64::ln_1p,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.sinh",             f64::sinh,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.cosh",             f64::cosh,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.tanh",             f64::tanh,             (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.asinh",            f64::asinh,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.acosh",            f64::acosh,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.atanh",            f64::atanh,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.is_nan",           f64::is_nan,           (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_infinite",      f64::is_infinite,      (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_finite",        f64::is_finite,        (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_subnormal",     f64::is_subnormal,     (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_normal",        f64::is_normal,        (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_sign_positive", f64::is_sign_positive, (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.is_sign_negative", f64::is_sign_negative, (E::F64),                 (E::Bool));
    decl!(prog, jb, "f64.recip",            f64::recip,            (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.to_degrees",       f64::to_degrees,       (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.to_radians",       f64::to_radians,       (E::F64),                 (E::F64));
    decl!(prog, jb, "f64.cosh",             f64::cosh,             (E::F64),                 (E::F64));
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

pub fn append_std_funcs(prog: &mut Vec<Declaration>) {
    for n in STD_1ARG_FF {
        prog.push(decl(n, vec![("x", E::F64)], vec![("y", E::F64)]));
    }
    for n in STD_1ARG_FI {
        prog.push(decl(n, vec![("x", E::F64)], vec![("y", E::I64)]));
    }
    for n in STD_1ARG_IF {
        prog.push(decl(n, vec![("x", E::I64)], vec![("y", E::F64)]));
    }
    for n in STD_2ARG_FF {
        prog.push(decl(
            n,
            vec![("x", E::F64), ("y", E::F64)],
            vec![("z", E::F64)],
        ));
    }
    for n in STD_2ARG_II {
        prog.push(decl(
            n,
            vec![("x", E::I64), ("y", E::I64)],
            vec![("z", E::I64)],
        ));
    }
    prog.push(decl("f64.print", vec![("x", E::F64)], vec![]));
    prog.push(decl("i64.print", vec![("x", E::I64)], vec![]));
    prog.push(decl("bool.print", vec![("x", E::Bool)], vec![]));
    prog.push(decl("&.print", vec![("x", E::Address)], vec![]));
    prog.push(decl("f64.println", vec![("x", E::F64)], vec![]));
    prog.push(decl("i64.println", vec![("x", E::I64)], vec![]));
    prog.push(decl("bool.println", vec![("x", E::Bool)], vec![]));
    prog.push(decl("&.println", vec![("x", E::Address)], vec![]));
    prog.push(decl(
        "f64.assert_eq",
        vec![("x", E::F64), ("y", E::F64)],
        vec![],
    ));
    prog.push(decl(
        "i64.assert_eq",
        vec![("x", E::I64), ("y", E::I64)],
        vec![],
    ));
    prog.push(decl(
        "bool.assert_eq",
        vec![("x", E::Bool), ("y", E::Bool)],
        vec![],
    ));
    prog.push(decl(
        "&.assert_eq",
        vec![("x", E::Address), ("y", E::Address)],
        vec![],
    ));

    //prog.push(decl(
    //    "bytes",
    //    vec![("size", ExprType::I64)],
    //    vec![("mem", ExprType::Address)],
    //));
}

pub(crate) fn translate_std(
    _ptr_ty: cranelift::prelude::Type,
    builder: &mut FunctionBuilder,
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
        _ => Ok(None),
    }
}

pub fn get_constants() -> HashMap<String, f64> {
    hashmap!(
        "E".into() => std::f64::consts::E,
        "FRAC_1_PI".into() => std::f64::consts::FRAC_1_PI,
        "FRAC_1_SQRT_2".into() => std::f64::consts::FRAC_1_SQRT_2,
        "FRAC_2_SQRT_PI".into() => std::f64::consts::FRAC_2_SQRT_PI,
        "FRAC_PI_2".into() => std::f64::consts::FRAC_PI_2,
        "FRAC_PI_3".into() => std::f64::consts::FRAC_PI_3,
        "FRAC_PI_4".into() => std::f64::consts::FRAC_PI_4,
        "FRAC_PI_6".into() => std::f64::consts::FRAC_PI_6,
        "FRAC_PI_8".into() => std::f64::consts::FRAC_PI_8,
        "LN_2".into() => std::f64::consts::LN_2,
        "LN_10".into() => std::f64::consts::LN_10,
        "LOG2_10".into() => std::f64::consts::LOG2_10,
        "LOG2_E".into() => std::f64::consts::LOG2_E,
        "LOG10_2".into() => std::f64::consts::LOG10_2,
        "LOG10_E".into() => std::f64::consts::LOG10_E,
        "PI".into() => std::f64::consts::PI,
        "SQRT_2".into() => std::f64::consts::SQRT_2,
        "TAU".into() => std::f64::consts::TAU
    )
}
