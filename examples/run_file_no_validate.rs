use sarus::{frontend::parser, jit, run_fn};
use std::{env, fs};

fn main() -> anyhow::Result<()> {
    if let Some(filename) = env::args().nth(1) {
        // Import file as string
        let code = fs::read_to_string(filename).expect("Something went wrong reading the file");

        // Create the JIT instance, which manages all generated functions and data.
        let mut jit = jit::JIT::default();
        jit.add_math_constants()?;

        // Generate AST from string
        let ast = parser::program(&code)?;

        // Pass the AST to the JIT to compile
        jit.translate(ast)?;

        // Run compiled code
        let result: f64 = unsafe { run_fn(&mut jit, "main", (100.0f64, 200.0f64))? };

        println!("the answer is: {}", result);
    } else {
        anyhow::bail!("Couldn't load file");
    }
    Ok(())
}
