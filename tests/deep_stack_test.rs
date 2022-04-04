#![feature(core_intrinsics)]
#![feature(try_blocks)]

use std::mem;

#[allow(unused_imports)]
use sarus::logging::setup_logging;

use sarus::*;

fn only_run_func(code: &str) -> anyhow::Result<()> {
    let mut jit = default_std_jit_from_code(code, true)?;
    let func_ptr = jit.get_func("main")?;
    let funcc = unsafe { mem::transmute::<_, extern "C" fn()>(func_ptr) };
    funcc();
    Ok(())
}

#[test]
fn deep_stack_basic() -> anyhow::Result<()> {
    //setup_logging();
    let code = r#"

fn main() -> () { 
    a = [1; 1000000]
    b = [2; 1000000]
    a[1].assert_eq(1)
    b[1].assert_eq(2)
}
"#;
    only_run_func(code)
}

#[test]
fn deep_stack_while_loop() -> anyhow::Result<()> {
    //setup_logging();
    let code = r#"

fn main() -> () { 
    i = 0
    while i <= 10 {
        a = [i; 1000000]
        b = [1; 1000000]
        a[i].assert_eq(i)
        i += 1
    }
}
"#;
    only_run_func(code)
}

#[test]
fn deep_stack_takes_array() -> anyhow::Result<()> {
    //setup_logging();
    let code = r#"
fn takes_arr(n: [f32; 1000000]) -> () {
    n = [1.0; 1000000]
}

fn main() -> () { 
    a = [0.0; 1000000]
    a[1].assert_eq(0.0)
    takes_arr(a)
    a[1].assert_eq(1.0)
    
}
"#;
    only_run_func(code)
}
