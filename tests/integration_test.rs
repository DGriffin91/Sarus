use serde::Deserialize;
use std::{collections::HashMap, f64::consts::*, ffi::CStr, mem};

use sarus::*;

fn default_std_jit_from_code(
    code: &str,
    symbols: Option<Vec<(&str, *const u8)>>,
) -> anyhow::Result<jit::JIT> {
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std_symbols(&mut jit_builder);
    if let Some(symbols) = symbols {
        jit_builder.symbols(symbols);
    }
    let mut jit = jit::JIT::from(jit_builder);
    jit.add_math_constants()?;
    let ast = parser::program(&code)?;
    let ast = sarus_std_lib::append_std_funcs(ast);
    jit.translate(ast.clone())?;
    Ok(jit)
}

#[test]
fn parentheses() -> anyhow::Result<()> {
    let code = r#"
fn main(a, b) -> (c) {
    c = a * (a - b) * (a * (2.0 + b))
}
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(a * (a - b) * (a * (2.0 + b)), func(a, b));
    Ok(())
}

#[test]
fn libc_math() -> anyhow::Result<()> {
    let code = r#"
fn main(a, b) -> (c) {
    c = b
    c = sin(c)
    c = cos(c)
    c = tan(c)
    c = asin(c)
    c = acos(c)
    c = atan(c)
    c = exp(c)
    c = log(c)
    c = log10(c)
    c = sqrt(c + 10.0)
    c = sinh(c)
    c = cosh(c)
    c = tanh(c * 0.00001)
    c = atan2(c, a)
    c = pow(c, a * 0.001)
    c *= nums()
}
fn nums() -> (r) {
    r = E + FRAC_1_PI + FRAC_1_SQRT_2 + FRAC_2_SQRT_PI + FRAC_PI_2 + FRAC_PI_3 + FRAC_PI_4 + FRAC_PI_6 + FRAC_PI_8 + LN_2 + LN_10 + LOG2_10 + LOG2_E + LOG10_2 + LOG10_E + PI + SQRT_2 + TAU
}
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut c = b;
    c = c.sin();
    c = c.cos();
    c = c.tan();
    c = c.asin();
    c = c.acos();
    c = c.atan();
    c = c.exp();
    c = c.log(E);
    c = c.log10();
    c = (c + 10.0).sqrt();
    c = c.sinh();
    c = c.cosh();
    c = (c * 0.00001).tanh();
    c = c.atan2(a);
    c = c.powf(a * 0.001);
    c *= E
        + FRAC_1_PI
        + FRAC_1_SQRT_2
        + FRAC_2_SQRT_PI
        + FRAC_PI_2
        + FRAC_PI_3
        + FRAC_PI_4
        + FRAC_PI_6
        + FRAC_PI_8
        + LN_2
        + LN_10
        + LOG2_10
        + LOG2_E
        + LOG10_2
        + LOG10_E
        + PI
        + SQRT_2
        + TAU;

    let epsilon = 0.00000000000001;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    let result = func(a, b);
    assert!(result >= c - epsilon && result <= c + epsilon);
    Ok(())
}

#[test]
fn rounding() -> anyhow::Result<()> {
    let code = r#"
fn main(a, b) -> (c) {
    f = (1.5).floor()
    c = a.ceil() * b.floor() * a.trunc() * (a * b * -1.234).fract() * (1.5).round()
}
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(
        a.ceil() * b.floor() * a.trunc() * (a * b * -1.234).fract() * 1.5f64.round(),
        func(a, b)
    );
    Ok(())
}

#[test]
fn minmax() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (c) {
        c = a.min(b)
    }
    "#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(a, func(a, b));

    let code = r#"
    fn main(a, b) -> (c) {
        c = a.max(b)
    }
    "#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(b, func(a, b));

    Ok(())
}

#[test]
fn comments() -> anyhow::Result<()> {
    let code = r#"
//test
fn main(a, b) -> (c) {//test
//test
    //test
    d = foodd(a, b) + foodd(a, b) //test
//test


//test
    c = d + 1.0 //test
//test//test
}//test

//test
//test

fn maina(a, b) -> (c) {//test
    c = foodd(a, b) + 2.12312 + 1.1//test
    c = c + 10.0//test
}//test
//test
fn foodd(a, b) -> (c) {
    c = a + b//test
}//test

//fn foodd(a, b) -> (c) {
//    c = a + b//test
//}//test
    
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(601.0, func(a, b));
    Ok(())
}

#[test]
fn multiple_returns() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        c, d = stuff(a, b)
        c, d = d, c
        e, f = if a == b {
            stuff(b, a)
        } else {
            stuff(a, b)
        }
        if 1.0 == 1.0 {
            e = e * 100.0
        }
        e *= 2.0
        e /= 3.0
        e -= 1.0
        i = 0.0
        while i < 10.0 {
            e = e * 2.0
            i += 1.0
        }
    }
    
    fn stuff(a, b) -> (c, d) {
        c = a + 1.0
        d = c + b + 10.0
    }
    
    fn stuff2(a) -> (c) {
        c = a + 1.0
    }
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(6893909.333333333, func(a, b));
    Ok(())
}

#[test]
fn bools() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (c) {
        c = if true {
            a * b
        } else {
            0.0
        }
        if false {
            c = 999999999.0
        }
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(20000.0, func(a, b));
    Ok(())
}

#[test]
fn ifelse_assign() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (c) {
        c = if a < b {
            a * b
        } else {
            0.0
        }
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(20000.0, func(a, b));
    Ok(())
}

#[test]
fn order() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (c) {
        c = a
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(100.0, func(a, b));
    Ok(())
}

#[test]
fn array_read_write() -> anyhow::Result<()> {
    let code = r#"
fn main(arr: &[f64], b) -> () {
    arr[0.0] = arr[0.0] * b
    arr[1.0] = arr[1.0] * b
    arr[2.0] = arr[2.0] * b
    arr[3.0] = arr[3.0] * b
}
"#;

    let mut arr = [1.0, 2.0, 3.0, 4.0];
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(*mut f64, f64)>(func_ptr) };
    func(arr.as_mut_ptr(), b);
    assert_eq!([200.0, 400.0, 600.0, 800.0], arr);
    Ok(())
}

#[test]
fn negative() -> anyhow::Result<()> {
    let code = r#"
    fn main(a) -> (c) {
        c = -1.0 + a
    }
"#;
    let a = -100.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64) -> f64>(func_ptr) };
    assert_eq!(-101.0, func(a));
    Ok(())
}

#[test]
fn compiled_graph() -> anyhow::Result<()> {
    let code = r#"
    fn add_node (a, b) -> (c) {
        c = a + b
    }
        
    fn sub_node (a, b) -> (c) {
        c = a - b
    }
        
    fn sin_node (a) -> (c) {
        c = sin(a)
    }
        
    fn graph (audio: &[f64]) -> () {
        i = 0
        while i <= 7 {
            vINPUT_0 = audio[i]
            vadd1_0 = add_node(vINPUT_0, 2.0000000000)
            vsin1_0 = sin_node(vadd1_0)
            vadd2_0 = add_node(vsin1_0, 4.0000000000)
            vsub1_0 = sub_node(vadd2_0, vadd1_0)
            vOUTPUT_0 = vsub1_0
            audio[i] = vOUTPUT_0
            i += 1
        }
    }
"#;

    let mut audio = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("graph")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(&mut [f64; 8])>(func_ptr) };
    dbg!(func(&mut audio));
    Ok(())
}

#[derive(Deserialize, Debug)]
struct Metadata {
    description: Option<String>,
    inputs: HashMap<String, MetadataInput>,
}

#[derive(Deserialize, Debug)]
struct MetadataInput {
    default: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    description: Option<String>,
    label: Option<String>,
    unit: Option<String>,
    gradient: Option<String>,
}

#[test]
fn metadata() -> anyhow::Result<()> {
    let code = r#"    
    
    @ add_node node
        description = "add two numbers!"

        [inputs]
        a = {default = 0.0, description = "1st number"}
        b = {default = 0.0, description = "2nd number"}
    @
    fn add_node (a, b) -> (c) {
        c = a + b
    }
        
    fn sub_node (a, b) -> (c) {
        c = a - b
    }
        
    fn sin_node (a) -> (c) {
        c = sin(a)
    }
        
    fn graph (audio: &[f64]) -> () {
        i = 0
        while i <= 7 {
            vINPUT_0 = audio[i]
            vadd1_0 = add_node(vINPUT_0, 2.0000000000)
            vsin1_0 = sin_node(vadd1_0)
            vadd2_0 = add_node(vsin1_0, 4.0000000000)
            vsub1_0 = sub_node(vadd2_0, vadd1_0)
            vOUTPUT_0 = vsub1_0
            audio[i] = vOUTPUT_0
            i += 1
        }
    }
"#;
    let ast = parser::program(&code)?;
    let mut jit = default_std_jit_from_code(&code, None)?;

    let func_meta: Option<Metadata> = ast.iter().find_map(|d| match d {
        frontend::Declaration::Metadata(head, body) => {
            if let Some(head) = head.first() {
                if head == "add_node" {
                    Some(toml::from_str(&body).unwrap())
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    });

    dbg!(&func_meta);

    let mut audio = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let func_ptr = jit.get_func("graph")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(&mut [f64; 8])>(func_ptr) };
    dbg!(func(&mut audio));
    //assert_eq!([200.0, 400.0, 600.0, 800.0], arr);
    Ok(())
}

#[test]
fn int_while_loop() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        e = 2.0
        i = 0
        while i < 10 {
            e = e * 2.0
            i += 1
        }
    }
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(2048.0, func(a, b));
    Ok(())
}

#[test]
fn int_to_float() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        i = 2
        e = i.f64() * a * b * (2).f64() * 2.0 * (2).f64()
    }
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(320000.0, func(a, b));
    Ok(())
}

#[test]
fn float_conversion() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        i_a = a.i64()
        e = if i_a < b.i64() {
            i_a.f64().i64().f64() //TODO chaining not working
        } else {
            2.0
        }
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(100.0, func(a, b));
    Ok(())
}

#[test]
fn float_as_bool_error() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        i_a = a
        e_i = if true {
            1
        } else {
            2
        }
        e = e_i.f64()
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(1.0, func(a, b));
    Ok(())
}

#[test]
fn array_return_from_if() -> anyhow::Result<()> {
    let code = r#"
fn main(arr1: &[f64], arr2: &[f64], b) -> () {
    arr3 = if b < 100.0 {
        arr1
    } else {
        arr2
    }
    arr3[0] = arr3[0] * 20.0
}
"#;

    let mut arr1 = [1.0, 2.0, 3.0, 4.0];
    let mut arr2 = [10.0, 20.0, 30.0, 40.0];
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(*mut f64, *mut f64, f64)>(func_ptr) };
    func(arr1.as_mut_ptr(), arr2.as_mut_ptr(), b);
    assert_eq!(200.0, arr2[0]);
    Ok(())
}

#[test]
fn var_type_consistency() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        n = 1
        n1 = n
        n2 = n1
        n3 = n2
        e = n3.f64()
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(1.0, func(a, b));
    Ok(())
}

#[test]
fn three_inputs() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b, c) -> (e) {
        e = a + b + c
    }
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let c = 300.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64, f64) -> f64>(func_ptr) };
    assert_eq!(600.0, func(a, b, c));
    Ok(())
}

#[test]
fn manual_types() -> anyhow::Result<()> {
    let code = r#"
fn main(a: f64, b: f64) -> (c: f64) {
    c = a * (a - b) * (a * (2.0 + b))
}
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(a * (a - b) * (a * (2.0 + b)), func(a, b));
    Ok(())
}

#[test]
fn i64_params() -> anyhow::Result<()> {
    let code = r#"
fn main(a: f64, b: i64) -> (c: i64) {
    e = a * (a - b.f64()) * (a * (2.0 + b.f64()))
    c = e.i64()
}
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, i64) -> i64>(func_ptr) };
    assert_eq!((a * (a - b) * (a * (2.0 + b))) as i64, func(a, b as i64));
    Ok(())
}

#[test]
fn i64_params_multifunc() -> anyhow::Result<()> {
    //Not currently working, see BLOCKERs in jit.rs
    let code = r#"
fn main(a: f64, b: i64) -> (c: i64) {
    c = foo(a, b, 2)
}
fn foo(a: f64, b: i64, c: i64) -> (d: i64) {
    d = a.i64() + b + c
}
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, i64) -> i64>(func_ptr) };
    assert_eq!(302, func(a, b as i64));
    Ok(())
}

#[test]
fn bool_params() -> anyhow::Result<()> {
    let code = r#"
fn main(a: f64, b: bool) -> (c: f64) {
    c = if b {
        a
    } else {
        0.0-a
    }
}
"#;
    let a = 100.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, bool) -> f64>(func_ptr) };
    assert_eq!(a, func(a, true));
    assert_eq!(-a, func(a, false));
    Ok(())
}

#[test]
fn logical_operators() -> anyhow::Result<()> {
    let code = r#"
fn and(a: bool, b: bool) -> (c: bool) {
    c = a && b
}
fn or(a: bool, b: bool) -> (c: bool) {
    c = a || b
}
fn gt(a: bool, b: bool) -> (c: bool) {
    c = a > b
}
fn ge(a: bool, b: bool) -> (c: bool) {
    c = a >= b
}
fn lt(a: bool, b: bool) -> (c: bool) {
    c = a < b
}
fn le(a: bool, b: bool) -> (c: bool) {
    c = a <= b
}
fn eq(a: bool, b: bool) -> (c: bool) {
    c = a == b
}
fn ne(a: bool, b: bool) -> (c: bool) {
    c = a != b
}
fn ifthen() -> (c: bool) {
    c = false
    if 1.0 < 2.0 && 2.0 < 3.0 {
        c = true
    }
}
fn ifthen2() -> (c: bool) {
    c = false
    if 1.0 < 2.0 || 2.0 < 1.0 {
        c = true
    }
}
fn ifthenparen() -> (c: bool) {
    c = false
    if (1.0 < 2.0) && (2.0 < 3.0) {
        c = true
    }
}
fn ifthennestedparen() -> (c: bool) {
    c = false
    if ((1.0 < 2.0) && (2.0 < 3.0) && true) {
        c = true
    }
}
fn parenassign() -> (c: bool) {
    c = ((1.0 < 2.0) && (2.0 < 3.0) && true)
}
"#;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("and")?) };
    assert_eq!(true, f(true, true));
    assert_eq!(false, f(true, false));
    assert_eq!(false, f(false, true));
    assert_eq!(false, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("or")?) };
    assert_eq!(true, f(true, true));
    assert_eq!(true, f(true, false));
    assert_eq!(true, f(false, true));
    assert_eq!(false, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("gt")?) };
    assert_eq!(false, f(true, true));
    assert_eq!(true, f(true, false));
    assert_eq!(false, f(false, true));
    assert_eq!(false, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("ge")?) };
    assert_eq!(true, f(true, true));
    assert_eq!(true, f(true, false));
    assert_eq!(false, f(false, true));
    assert_eq!(true, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("lt")?) };
    assert_eq!(false, f(true, true));
    assert_eq!(false, f(true, false));
    assert_eq!(true, f(false, true));
    assert_eq!(false, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("le")?) };
    assert_eq!(true, f(true, true));
    assert_eq!(false, f(true, false));
    assert_eq!(true, f(false, true));
    assert_eq!(true, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("eq")?) };
    assert_eq!(true, f(true, true));
    assert_eq!(false, f(true, false));
    assert_eq!(false, f(false, true));
    assert_eq!(true, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool, bool) -> bool>(jit.get_func("ne")?) };
    assert_eq!(false, f(true, true));
    assert_eq!(true, f(true, false));
    assert_eq!(true, f(false, true));
    assert_eq!(false, f(false, false));
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthen")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthen2")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthenparen")?) };
    assert_eq!(true, f());
    let f =
        unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthennestedparen")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("parenassign")?) };
    assert_eq!(true, f());
    Ok(())
}

#[test]
fn unary_not() -> anyhow::Result<()> {
    let code = r#"
fn direct() -> (c: bool) {
    c = !true
}
fn direct2() -> (c: bool) {
    c = !false
}
fn direct3() -> (c: bool) {
    c = !(false)
}
fn not(a: bool) -> (c: bool) {
    c = !a
}
fn not2(a: bool) -> (c: bool) {
    c = !(a)
}
fn ifthen() -> (c: bool) {
    c = false
    if !(false) {
        c = true
    }
}
fn ifthen2() -> (c: bool) {
    c = false
    if !(!(false || !false)) {
        c = true
    }
}
fn ifthen3() -> (c: bool) {
    c = true
    if !(!(1.0 < 2.0) && !(2.0 < 3.0)) {
        c = false
    }
}
fn parenassign() -> (c: bool) {
    c = !((1.0 < 2.0) && (2.0 < 3.0) && true)
}
"#;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("direct")?) };
    assert_eq!(false, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("direct2")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("direct3")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn(bool) -> bool>(jit.get_func("not2")?) };
    assert_eq!(false, f(true));
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthen")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthen2")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("ifthen3")?) };
    assert_eq!(true, f());
    let f = unsafe { mem::transmute::<_, extern "C" fn() -> bool>(jit.get_func("parenassign")?) };
    assert_eq!(false, f());
    Ok(())
}

extern "C" fn mult(a: f64, b: f64) -> f64 {
    a * b
}

extern "C" fn dbg(a: f64) {
    dbg!(a);
}

#[test]
fn extern_func() -> anyhow::Result<()> {
    let code = r#"
extern fn mult(a: f64, b: f64) -> (c: f64) {}
extern fn dbg(a: f64) -> () {}

fn main(a: f64, b: f64) -> (c: f64) {
    c = mult(a, b)
    dbg(a)
}
"#;
    let a = 100.0f64;
    let b = 100.0f64;
    let mut jit = default_std_jit_from_code(
        &code,
        Some(vec![("mult", mult as *const u8), ("dbg", dbg as *const u8)]),
    )?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    assert_eq!(mult(a, b), func(a, b));
    Ok(())
}

extern "C" fn prt2(s: *const i8) {
    unsafe {
        print!("{}", CStr::from_ptr(s).to_str().unwrap());
    }
}

#[test]
fn create_string() -> anyhow::Result<()> {
    let code = r#"
fn main(a: f64, b: f64) -> (c: f64) {
    print("HELLO\n")
    print(["-"; 5])
    print("WORLD\n")
    c = a
}

extern fn print(s: &) -> () {}
"#;
    let a = 100.0f64;
    let b = 100.0f64;
    let mut jit = default_std_jit_from_code(&code, Some(vec![("print", prt2 as *const u8)]))?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
    func(a, b);

    Ok(())
}

#[test]
fn struct_access() -> anyhow::Result<()> {
    let code = r#"
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
fn main(a: f64) -> (c: f64) {
    p = Point {
        x: a,
        y: 200.0,
        z: 300.0,
    }
    c = p.x + p.y + p.z
}
"#;
    let a = 100.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64) -> f64>(func_ptr) };
    assert_eq!(600.0, func(a));
    Ok(())
}

#[test]
fn struct_impl() -> anyhow::Result<()> {
    let code = r#"
struct Point {
    x: f64,
    y: f64,
    z: f64,
}
fn length(self: Point) -> (r: f64) {
    r = sqrt(pow(self.x, 2.0) + pow(self.y, 2.0) + pow(self.z, 2.0))
}
fn main(a: f64) -> (c: f64) {
    p = Point {
        x: a,
        y: 200.0,
        z: 300.0,
    }
    c = p.length()
}
"#;
    let a = 100.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64) -> f64>(func_ptr) };
    assert_eq!(374.16573867739413, func(a));
    Ok(())
}

extern "C" fn dbgi(s: *const i8, a: i64) {
    unsafe {
        println!("{} {}", CStr::from_ptr(s).to_str().unwrap(), a);
    }
}

#[test]
fn struct_impl_nested() -> anyhow::Result<()> {
    let code = r#"
extern fn dbg(a: f64) -> () {}
extern fn dbgi(s: &, a: i64) -> () {}
struct Line {
    a: Point,
    b: Point,
}

fn print(self: Line) -> () {
    "Line {".println()
    "a: ".print() self.a.print() ",".println()
    "b: ".print() self.b.print() ",".println()
    "}".println()
}

//fn length(self: Line) -> (r: f64) {
//    r = sqrt(pow(self.a.x - self.b.x, 2.0) + 
//             pow(self.a.y - self.b.y, 2.0) + 
//             pow(self.a.z - self.b.z, 2.0))
//}
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

fn print(self: Point) -> () {
    "Point {".println()
    "x: ".print() self.x.print() ",".println()
    "y: ".print() self.y.print() ",".println()
    "z: ".print() self.z.print() ",".println()
    "}".println()
}

//fn length(self: Point) -> (r: f64) {
//    r = sqrt(pow(self.x, 2.0) + pow(self.y, 2.0) + pow(self.z, 2.0))
//}
fn main(n: f64) -> (c: f64) {
    p1 = Point {
        x: n,
        y: 200.0,
        z: 300.0,
    }
    p2 = Point {
        x: n * 4.0,
        y: 500.0,
        z: 600.0,
    }
    l1 = Line {
        a: p1,
        b: p2,
    }
    l1.print()
    //d = l1.c //struct is copied
    l1.b.x.println() //TODO these don't get initialized
    l1.b.y.println() //TODO these don't get initialized
    l1.b.z.println() //TODO these don't get initialized
    ////e = d.x + l1.a.x //f64's are copied
    ////l1.a.y = e * d.z
    ////c = l1.length()
}
"#;
    let a = 100.0f64;
    let mut jit = default_std_jit_from_code(
        &code,
        Some(vec![("dbg", dbg as *const u8), ("dbgi", dbgi as *const u8)]),
    )?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64) -> f64>(func_ptr) };
    dbg!(func(a));
    //assert_eq!(200.0, func(a));
    //jit.print_clif(true);
    Ok(())
}

#[test]
fn type_impl() -> anyhow::Result<()> {
    let code = r#"
fn square(self: f64) -> (r: f64) {
    r = self * self
}
fn square(self: i64) -> (r: i64) {
    r = self * self
}
fn main(a: f64, b: i64) -> (c: f64) {
    c = a.square() + b.square().f64()
}
"#;
    let a = 100.0f64;
    let b = 100i64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, i64) -> f64>(func_ptr) };
    assert_eq!(20000.0, func(a, b));
    Ok(())
}

#[test]
fn stacked_paren() -> anyhow::Result<()> {
    let code = r#"
fn main(a: f64) -> (c: bool) {
    d = a.i64().f64().i64().f64()
    e = ((((d).i64()).f64()).i64()).f64()
    c = d == e
}
"#;
    let a = 100.0f64;
    let mut jit = default_std_jit_from_code(&code, None)?;
    let func_ptr = jit.get_func("main")?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64) -> bool>(func_ptr) };
    assert_eq!(true, func(a));
    Ok(())
}

//#[test]
//fn int_min_max() -> anyhow::Result<()> {
//    //Not currently working: Unsupported type for imin instruction: i64
//    //https://github.com/bytecodealliance/wasmtime/issues/3370
//    let code = r#"
//    fn main() -> (e) {
//        c = imin(1, 2)
//        //d = imax(3, 4)
//        //f = c * d
//        e = float(c)
//    }
//"#;
//    let a = 100.0f64;
//    let b = 100.0f64;
//    let mut jit = jit::JIT::new(&[("print", prt2 as *const u8)]);
//    let ast = parser::program(&code)?;
//    let ast = sarus_std_lib::append_std_funcs( ast);
//    jit.translate(ast.clone())?;
//    let func_ptr = jit.get_func("main")?;
//    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };
//    func(a, b);
//    Ok(())
//}
