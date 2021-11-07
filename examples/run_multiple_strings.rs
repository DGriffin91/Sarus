use std::mem;

use sarus::{jit, parse};

fn main() -> anyhow::Result<()> {
    // Create the JIT instance, which manages all generated functions and data.
    let mut jit = jit::JIT::default();

    let code = r#"
    fn add(a, b) -> (c) {
        c = a + b
    }
    "#;

    // Generate AST from string
    let ast = parse(code)?;
    // Pass the AST to the JIT to compile
    jit.translate(ast, None)?;

    //Get the function, returns a raw pointer to machine code.
    let func_ptr = jit.get_func("add")?;

    // Cast the raw pointer to a typed function pointer. This is unsafe, because
    // this is the critical point where you have to trust that the generated code
    // is safe to be called.
    let func = unsafe { mem::transmute::<_, extern "C" fn(f32, f32) -> f32>(func_ptr) };

    println!("the answer is: {}", func(3.0f32, 5.0f32));

    let code = r#"
    fn mult(a, b) -> (c) {
        c = a * b
    }
    "#;

    // Generate AST from 2nd string
    let ast = parse(code)?;

    //TODO fails here Error: Duplicate definition of identifier: LN_10
    //Need to only add constants once
    jit.translate(ast, None)?;
    let func_ptr = jit.get_func("mult")?;
    let func = unsafe { mem::transmute::<_, fn(f32, f32) -> f32>(func_ptr) };
    println!("the answer is: {}", func(3.0f32, 5.0f32));

    // TODO allow validator to look at previously compiled strings to allow this:
    /*
    fn main(a, b) -> (c) {
        c = add(a, b) / mult(a, b)
    }
    */
    Ok(())
}
