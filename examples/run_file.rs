use sarus::{jit, run_string};
use std::{env, fs};

fn main() -> anyhow::Result<()> {
    if let Some(filename) = env::args().nth(1) {
        // Import file as string
        let code = fs::read_to_string(filename).expect("Something went wrong reading the file");

        // Create the JIT instance, which manages all generated functions and data.
        let mut jit = jit::JIT::default();
        jit.add_math_constants()?;

        // Run string with jit instance.
        // This function is unsafe since it relies on the caller to provide it with the correct
        // input and output types. Using incorrect types at this point may corrupt the program's state.
        // Check out run_string() source if you need to separate out execution and parsing steps
        let result: f64 = unsafe { run_string(&mut jit, &code, "main", (100.0f64, 200.0f64))? };

        println!("the answer is: {}", result);
    } else {
        anyhow::bail!("Couldn't load file");
    }
    Ok(())
}
