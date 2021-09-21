use std::mem;

use sarus::{jit, parser};

fn main() -> anyhow::Result<()> {
    // Create the JIT instance, which manages all generated functions and data.
    let mut jit = jit::JIT::default();

    // Generate AST from string
    let ast = parser::program(
        r#"
fn add(a, b) -> (c) {
    c = a + b
}
"#,
    )?;
    // Pass the AST to the JIT to compile
    jit.translate(ast)?;

    //Get the function, returns a raw pointer to machine code.
    let func_ptr = jit.get_func("add")?;

    // Cast the raw pointer to a typed function pointer. This is unsafe, because
    // this is the critical point where you have to trust that the generated code
    // is safe to be called.
    let func = unsafe { mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(func_ptr) };

    println!("the answer is: {}", func(3.0f64, 5.0f64));

    // Generate AST from 2nd string
    let ast = parser::program(
        r#"
fn mult(a, b) -> (c) {
    c = a * b
}
"#,
    )?;

    jit.translate(ast)?;
    let func_ptr = jit.get_func("mult")?;
    let func = unsafe { mem::transmute::<_, fn(f64, f64) -> f64>(func_ptr) };
    println!("the answer is: {}", func(3.0f64, 5.0f64));

    // TODO allow validator to look at previously compiled strings to allow this:
    /*
    fn main(a, b) -> (c) {
        c = add(a, b) / mult(a, b)
    }
    */
    Ok(())
}
