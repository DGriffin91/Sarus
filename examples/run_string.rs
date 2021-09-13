use sarus::{jit, run_string};

fn main() -> anyhow::Result<()> {
    // Create the JIT instance, which manages all generated functions and data.
    let mut jit = jit::JIT::default();

    let code = r#"
fn main(a, b) -> (c) {
    c = if a < b {
        if b > 10.0 {
            30.0
        } else {
            40.0
        }
    } else {
        50.0
    }
    c = c + 2.0
}
"#;

    // Run string with jit instance.
    // This function is unsafe since it relies on the caller to provide it with the correct
    // input and output types. Using incorrect types at this point may corrupt the program's state.
    // Check out run_string() source if you need to separate out execution and parsing steps
    let result: f64 = unsafe { run_string(&mut jit, code, "main", (100.0f64, 200.0f64))? };

    println!("the answer is: {}", result);

    Ok(())
}
