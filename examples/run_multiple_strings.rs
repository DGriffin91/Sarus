use sarus::{compile_string, jit, run_fn};

fn main() -> anyhow::Result<()> {
    // Create the JIT instance, which manages all generated functions and data.
    let mut jit = jit::JIT::default();

    //Compiles the code, add it to the current JIT instance
    compile_string(
        &mut jit,
        r#"
fn add(a, b) -> (c) {
    c = a + b
}
    "#,
    )?;

    let res: f64 = unsafe { run_fn(&mut jit, "add", (3.0f64, 5.0f64))? };
    println!("the answer is: {}", res);

    compile_string(
        &mut jit,
        r#"
fn mult(a, b) -> (c) {
    c = a * b
}
    "#,
    )?;

    let res: f64 = unsafe { run_fn(&mut jit, "mult", (3.0f64, 5.0f64))? };
    println!("the answer is: {}", res);

    // TODO allow validator to look at previously compiled strings to allow this:
    /*
    fn main(a, b) -> (c) {
        c = add(a, b) / mult(a, b)
    }
    */
    Ok(())
}
