use core::mem;
use cranelift_jit_demo::{frontend::parser, jit, validator::validate_program};
use std::{env, fs};

fn main() -> anyhow::Result<()> {
    // Create the JIT instance, which manages all generated functions and data.
    let mut jit = jit::JIT::default();
    println!("the answer is: {}", run_file(&mut jit)?);
    Ok(())
}

fn run_file(jit: &mut jit::JIT) -> anyhow::Result<f64> {
    if let Some(filename) = env::args().nth(1) {
        let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");
        //Inputs need to be explicitly f32
        unsafe { run_code(jit, &contents, (100.0f64, 200.0f64)) }
    } else {
        anyhow::bail!("Couldn't load file");
    }
}

/// Executes the given code using the cranelift JIT compiler.
///
/// Feeds the given input into the JIT compiled function and returns the resulting output.
///
/// # Safety
///
/// This function is unsafe since it relies on the caller to provide it with the correct
/// input and output types. Using incorrect types at this point may corrupt the program's state.
unsafe fn run_code<I, O>(jit: &mut jit::JIT, code: &str, input: I) -> anyhow::Result<O> {
    // Pass the string to the JIT, and it returns a raw pointer to machine code.
    let main_ptr = jit
        .translate(validate_program(parser::program(code)?)?)
        .map_err(|os| anyhow::anyhow!("{}", os))?;

    // Cast the raw pointer to a typed function pointer. This is unsafe, because
    // this is the critical point where you have to trust that the generated code
    // is safe to be called.
    let main_fn = mem::transmute::<_, fn(I) -> O>(main_ptr);
    // And now we can call it!
    Ok(main_fn(input))
}

#[allow(dead_code)]
fn run_hello(jit: &mut jit::JIT) -> anyhow::Result<isize> {
    jit.create_data("hello_string", "hello world!\0".as_bytes().to_vec())
        .map_err(|os| anyhow::anyhow!("{}", os))?;
    unsafe { run_code(jit, HELLO_CODE, ()) }
}

/// Let's say hello, by calling into libc. The puts function is resolved by
/// dlsym to the libc function, and the string &hello_string is defined below.
const HELLO_CODE: &str = r#"
fn hello() -> (r) {
    puts(&hello_string)
}
"#;
