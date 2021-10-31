use std::collections::HashMap;
use std::ffi::CStr;

use cranelift::frontend::FunctionBuilder;
use cranelift::prelude::{types, InstBuilder, Value};
use cranelift_jit::JITBuilder;
use tracing::trace;

use crate::jit::{SValue, StructDef};
use crate::validator::{address_t, bool_t, f32_t, i64_t, ArraySizedExpr, ExprType};
use crate::{
    decl,
    frontend::{Arg, CodeRef, Declaration, Function},
};
use crate::{hashmap, make_decl, parse};

const STD_1ARG_FF: [&str; 6] = [
    "f32.ceil", // built in std
    "f32.floor",
    "f32.trunc",
    "f32.fract",
    "f32.abs",
    "f32.round",
];
const STD_1ARG_IF: [&str; 1] = [
    "i64.f32", // built in std
];
const STD_1ARG_FI: [&str; 1] = [
    "f32.i64", // built in std
];

const STD_2ARG_FF: [&str; 2] = [
    "f32.min", "f32.max", // built in std
];
const STD_2ARG_II: [&str; 2] = [
    "i64.min", "i64.max", // built in std
];

extern "C" fn f32_print(x: f32) {
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

extern "C" fn f32_println(x: f32) {
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

extern "C" fn f32_assert_eq(x: f32, y: f32) {
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
    decl!(prog, jb, "f32.signum",           f32::signum,           (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.copysign",         f32::copysign,         (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.mul_add",          f32::mul_add,          (f32_t(), f32_t(), f32_t()), (f32_t()));
    decl!(prog, jb, "f32.div_euclid",       f32::div_euclid,       (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.rem_euclid",       f32::rem_euclid,       (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.powi",             f32::powi,             (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.powf",             f32::powf,             (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.sqrt",             f32::sqrt,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.exp",              f32::exp,              (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.exp2",             f32::exp2,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.ln",               f32::ln,               (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.log",              f32::log,              (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.log2",             f32::log2,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.log10",            f32::log10,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.cbrt",             f32::cbrt,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.hypot",            f32::hypot,            (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.sin",              f32::sin,              (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.tan",              f32::tan,              (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.cos",              f32::cos,              (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.asin",             f32::asin,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.acos",             f32::acos,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.atan",             f32::atan,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.atan2",            f32::atan2,            (f32_t(), f32_t()),         (f32_t()));
    decl!(prog, jb, "f32.sin_cos",          f32::sin_cos,          (f32_t()),                 (f32_t(), f32_t()));
    decl!(prog, jb, "f32.exp_m1",           f32::exp_m1,           (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.ln_1p",            f32::ln_1p,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.sinh",             f32::sinh,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.cosh",             f32::cosh,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.tanh",             f32::tanh,             (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.asinh",            f32::asinh,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.acosh",            f32::acosh,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.atanh",            f32::atanh,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.is_nan",           f32::is_nan,           (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_infinite",      f32::is_infinite,      (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_finite",        f32::is_finite,        (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_subnormal",     f32::is_subnormal,     (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_normal",        f32::is_normal,        (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_sign_positive", f32::is_sign_positive, (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.is_sign_negative", f32::is_sign_negative, (f32_t()),                 (bool_t()));
    decl!(prog, jb, "f32.recip",            f32::recip,            (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.to_degrees",       f32::to_degrees,       (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.to_radians",       f32::to_radians,       (f32_t()),                 (f32_t()));
    decl!(prog, jb, "f32.cosh",             f32::cosh,             (f32_t()),                 (f32_t()));
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
    for n in STD_1ARG_FF {
        prog.push(make_decl(n, vec![("x", f32_t())], vec![("y", f32_t())]));
    }
    for n in STD_1ARG_FI {
        prog.push(make_decl(n, vec![("x", f32_t())], vec![("y", i64_t())]));
    }
    for n in STD_1ARG_IF {
        prog.push(make_decl(n, vec![("x", i64_t())], vec![("y", f32_t())]));
    }
    for n in STD_2ARG_FF {
        prog.push(make_decl(
            n,
            vec![("x", f32_t()), ("y", f32_t())],
            vec![("z", f32_t())],
        ));
    }
    for n in STD_2ARG_II {
        prog.push(make_decl(
            n,
            vec![("x", i64_t()), ("y", i64_t())],
            vec![("z", i64_t())],
        ));
    }

    decl!(prog, jb, "f32.print",           f32_print,           (f32_t()),                 ());
    decl!(prog, jb, "i64.print",           i64_print,           (i64_t()),                 ());
    decl!(prog, jb, "bool.print",          bool_print,          (bool_t()),                ());
    decl!(prog, jb, "&.print",             str_print,           (address_t()),             ()); //TODO setup actual str type

    decl!(prog, jb, "f32.println",         f32_println,         (f32_t()),                 ());
    decl!(prog, jb, "i64.println",         i64_println,         (i64_t()),                 ());
    decl!(prog, jb, "bool.println",        bool_println,        (bool_t()),                ());
    decl!(prog, jb, "&.println",           str_println,         (address_t()),             ());

    
    decl!(prog, jb, "f32.assert_eq",       f32_assert_eq,       (f32_t(), f32_t()),        ());
    decl!(prog, jb, "i64.assert_eq",       i64_assert_eq,       (i64_t(), i64_t()),        ());
    decl!(prog, jb, "bool.assert_eq",      bool_assert_eq,      (bool_t(), bool_t()),      ());
    decl!(prog, jb, "&.assert_eq",         str_assert_eq,       (address_t(), address_t()),());
    
    decl!(prog, jb, "panic",               spanic,              (address_t()),             ());
    
    prog.push(make_decl("src_line", vec![], vec![("line", i64_t())]));

    append_struct_macros(prog);
}

pub(crate) fn translate_std(
    _ptr_ty: cranelift::prelude::Type,
    builder: &mut FunctionBuilder,
    code_ref: &CodeRef,
    name: &str,
    args: &[Value],
) -> anyhow::Result<Option<SValue>> {
    match name {
        "f32.trunc" => Ok(Some(SValue::F32(builder.ins().trunc(args[0])))),
        "f32.floor" => Ok(Some(SValue::F32(builder.ins().floor(args[0])))),
        "f32.ceil" => Ok(Some(SValue::F32(builder.ins().ceil(args[0])))),
        "f32.fract" => {
            let v_int = builder.ins().trunc(args[0]);
            let v = builder.ins().fsub(args[0], v_int);
            Ok(Some(SValue::F32(v)))
        }
        "f32.abs" => Ok(Some(SValue::F32(builder.ins().fabs(args[0])))),
        "f32.round" => Ok(Some(SValue::F32(builder.ins().nearest(args[0])))),
        "f32.i64" => Ok(Some(SValue::I64(
            builder.ins().fcvt_to_sint(types::I64, args[0]),
        ))),
        "i64.f32" => Ok(Some(SValue::F32(
            builder.ins().fcvt_from_sint(types::F32, args[0]),
        ))),
        "f32.min" => Ok(Some(SValue::F32(builder.ins().fmin(args[0], args[1])))),
        "f32.max" => Ok(Some(SValue::F32(builder.ins().fmax(args[0], args[1])))),
        "i64.min" => Ok(Some(SValue::I64(builder.ins().imin(args[0], args[1])))),
        "i64.max" => Ok(Some(SValue::I64(builder.ins().imax(args[0], args[1])))),
        "src_line" => {
            let line = code_ref.line.unwrap_or(0) as i64;
            Ok(Some(SValue::I64(builder.ins().iconst(types::I64, line))))
        }
        _ => Ok(None),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SConstant {
    F32(f32),
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

pub fn append_struct_macros(prog: &mut Vec<Declaration>) {
    //Will probably change significantly or be removed
    let mut new_decls = Vec::new();
    for decl in prog.iter() {
        if let Declaration::StructMacro(name, ty) = decl {
            if name == "Slice" {
                if let ExprType::Array(code_ref, expr_type, size_type) = &**ty {
                    let code = format!(
                        r#"
struct Slice::{1} {{
    arr: {0},
    len: i64,
}}
inline fn get(self: Slice::{1}, i: i64) -> (r: {1}) {{
    if i >= 0 && i < self.len {{
        r = self.arr[i]
    }} else {{
        panic("index out of bounds")
    }}
}}
inline fn set(self: Slice::{1}, i: i64, val: {1}) -> () {{
    if i >= 0 && i < self.len {{
        self.arr[i] = val
    }} else {{
        panic("index out of bounds")
    }}
}}
"#,
                        ty, expr_type
                    );
                    trace!("{}: {}", code_ref, &code);
                    new_decls.append(&mut parse(&code).unwrap()); //TODO handle errors

                    let code = match size_type {
                        ArraySizedExpr::Unsized => {
                            format!(
                                r#"
inline fn into_slice(self: {0}, len: i64) -> (r: Slice::{1}) {{
    r = Slice::{1} {{
        arr: self,
        len: len,
    }}
}}
"#,
                                ty, expr_type
                            )
                        }
                        ArraySizedExpr::Sized => todo!(),
                        ArraySizedExpr::Fixed(len) => {
                            format!(
                                r#"
inline fn into_slice(self: {0}) -> (r: Slice::{1}) {{
    r = Slice::{1} {{
        arr: self,
        len: {2},
    }}
}}
"#,
                                ty, expr_type, len
                            )
                        }
                    };
                    trace!("{}: {}", code_ref, &code);
                    new_decls.append(&mut parse(&code).unwrap());
                //TODO handle errors
                } else {
                    panic!("Slice for type {} is unsupported", ty)
                }
            }
        }
    }
    prog.append(&mut new_decls);
}
