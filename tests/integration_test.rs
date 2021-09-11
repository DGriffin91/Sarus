use std::f64::consts::*;

use cranelift_jit_demo::*;

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
