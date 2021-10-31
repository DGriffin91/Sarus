use sarus::{frontend::parse, jit};
use std::{env, fs, mem};

fn main() -> anyhow::Result<()> {
    if let Some(filename) = env::args().nth(1) {
        // Import file as string
        let code = fs::read_to_string(filename).expect("Something went wrong reading the file");

        // Create the JIT instance, which manages all generated functions and data.
        let mut jit = jit::JIT::default();

        // Generate AST from string
        let ast = parse(&code)?;

        // Pass the AST to the JIT to compile
        jit.translate(ast, None)?;

        //Get the function, returns a raw pointer to machine code.
        let func_ptr = jit.get_func("main")?;

        // Cast the raw pointer to a typed function pointer. This is unsafe, because
        // this is the critical point where you have to trust that the generated code
        // is safe to be called.
        let func = unsafe { mem::transmute::<_, extern "C" fn(f32, f32) -> f32>(func_ptr) };

        println!("the answer is: {}", func(100.0f32, 200.0f32));
    } else {
        anyhow::bail!("Couldn't load file");
    }
    Ok(())
}
