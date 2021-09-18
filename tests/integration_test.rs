use serde::Deserialize;
use std::{collections::HashMap, f64::consts::*};

use sarus::*;

#[test]
fn parentheses() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
    let code = r#"
fn main(a, b) -> (c) {
    c = a * (a - b) * (a * (2.0 + b))
}
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(a * (a - b) * (a * (2.0 + b)), result);
    Ok(())
}

#[test]
fn libc_math() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
    jit.add_math_constants()?;
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert!(result >= c - epsilon && result <= c + epsilon);
    Ok(())
}

#[test]
fn rounding() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();

    let code = r#"
fn main(a, b) -> (c) {
    c = ceil(a) * floor(b) * trunc(a) * fract(a * b * -1.234) * round(1.5)
}
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(
        a.ceil() * b.floor() * a.trunc() * (a * b * -1.234).fract() * 1.5f64.round(),
        result
    );
    Ok(())
}

#[test]
fn minmax() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();

    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe {
        run_string(
            &mut jit,
            r#"
            fn main(a, b) -> (c) {
                c = min(a, b)
            }
            "#,
            "main",
            (a, b),
        )?
    };
    assert_eq!(result, a);
    let mut jit = jit::JIT::default();
    let result: f64 = unsafe {
        run_string(
            &mut jit,
            r#"
            fn main(a, b) -> (c) {
                c = max(a, b)
            }
            "#,
            "main",
            (a, b),
        )?
    };
    assert_eq!(result, b);

    Ok(())
}

#[test]
fn comments() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(601.0, result);
    Ok(())
}

#[test]
fn multiple_returns() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(6893909.333333333, result);
    Ok(())
}

#[test]
fn bools() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(20000.0, result);
    Ok(())
}

#[test]
fn ifelse_assign() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(20000.0, result);
    Ok(())
}

#[test]
fn order() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
    let code = r#"
    fn main(a, b) -> (c) {
        c = a
    }
"#;
    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(100.0, result);
    Ok(())
}

#[test]
fn array_read_write() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();

    let code = r#"
fn main(&arr, b) -> () {
    &arr[0.0] = &arr[0.0] * b
    &arr[1.0] = &arr[1.0] * b
    &arr[2.0] = &arr[2.0] * b
    &arr[3.0] = &arr[3.0] * b
}
"#;

    let mut arr = [1.0, 2.0, 3.0, 4.0];
    let b = 200.0f64;
    unsafe { run_string(&mut jit, code, "main", (&mut arr, b))? };
    assert_eq!([200.0, 400.0, 600.0, 800.0], arr);
    Ok(())
}

#[test]
fn negative() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
    let code = r#"
    fn main(a) -> (c) {
        c = -1.0 + a
    }
"#;
    let a = -100.0f64;
    let result: f64 = unsafe { run_string(&mut jit, code, "main", a)? };
    assert_eq!(-101.0, result);
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
        
    fn graph (&audio) -> () {
        i = 0
        while i <= 7 {
            vINPUT_0 = &audio[i]
            vadd1_0 = add_node(vINPUT_0, 2.0000000000)
            vsin1_0 = sin_node(vadd1_0)
            vadd2_0 = add_node(vsin1_0, 4.0000000000)
            vsub1_0 = sub_node(vadd2_0, vadd1_0)
            vOUTPUT_0 = vsub1_0
            &audio[i] = vOUTPUT_0
            i += 1
        }
    }
"#;

    let mut jit = jit::JIT::default();
    let mut audio = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    unsafe { run_string(&mut jit, code, "graph", &mut audio)? };
    dbg!(audio);
    //assert_eq!([200.0, 400.0, 600.0, 800.0], arr);
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
        
    fn graph (&audio) -> () {
        i = 0
        while i <= 7 {
            vINPUT_0 = &audio[i]
            vadd1_0 = add_node(vINPUT_0, 2.0000000000)
            vsin1_0 = sin_node(vadd1_0)
            vadd2_0 = add_node(vsin1_0, 4.0000000000)
            vsub1_0 = sub_node(vadd2_0, vadd1_0)
            vOUTPUT_0 = vsub1_0
            &audio[i] = vOUTPUT_0
            i += 1
        }
    }
"#;

    let mut jit = jit::JIT::default();
    let ast = parser::program(&code)?;
    let ast = validate_program(ast)?;
    jit.translate(ast.clone())?;

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
    unsafe { run_fn(&mut jit, "graph", &mut audio)? };
    dbg!(audio);
    //assert_eq!([200.0, 400.0, 600.0, 800.0], arr);
    Ok(())
}

#[test]
fn int_while_loop() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
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
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(2048.0, result);
    Ok(())
}

#[test]
fn int_to_float() -> anyhow::Result<()> {
    let mut jit = jit::JIT::default();
    let code = r#"
    fn main(a, b) -> (e) {
        i = 2
        e = i * a * b * 2 * 1.0 * 1
    }
"#;

    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (a, b))? };
    assert_eq!(80000.0, result);
    Ok(())
}

#[test]
fn float_conversion() -> anyhow::Result<()> {
    let code = r#"
    fn main(a, b) -> (e) {
        i_a = int(a)
        e = if i_a > int(b) {
            float(int(float(i_a)))
        } else {
            2.0
        }
    }
"#;
    let mut jit = jit::JIT::default();
    let ast = parser::program(&code)?;
    //let ast = validate_program(ast)?;
    jit.translate(ast.clone())?;

    let a = 100.0f64;
    let b = 200.0f64;
    let result: f64 = unsafe { run_fn(&mut jit, "main", (a, b))? };
    assert_eq!(2.0, result);
    Ok(())
}
