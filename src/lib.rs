use std::mem;

use crate::validator::validate_program;

use crate::frontend::parser;

pub mod frontend;
pub mod graph;
pub mod jit;
pub mod validator;

/// Compiles the given code using the cranelift JIT compiler.
///
/// Adds the compiled code to the provided jit instance.
pub fn compile_string(jit: &mut jit::JIT, code: &str) -> anyhow::Result<()> {
    // Generate AST from string
    let ast = parser::program(code)?;

    // Validate type useage
    let ast = validate_program(ast)?;

    // Pass the AST to the JIT to compile
    jit.translate(ast)?;
    Ok(())
}

/// Executes an existing function that was already compiled in the JIT
///
/// Feeds the given input into the JIT compiled function and returns the resulting output.
///
/// # Safety
///
/// This function is unsafe since it relies on the caller to provide it with the correct
/// input and output types. Using incorrect types at this point may corrupt the program's state.
#[allow(unused_unsafe)]
pub unsafe fn run_fn<I, O>(jit: &mut jit::JIT, fn_name: &str, input: I) -> anyhow::Result<O> {
    //Get the function, returns a raw pointer to machine code.
    let func_ptr = jit.get_func(fn_name)?;

    // This is unsafe since it relies on the caller to provide it with the correct
    // input and output types. Using incorrect types at this point may corrupt the program's state.
    Ok(unsafe {
        // Cast the raw pointer to a typed function pointer. This is unsafe, because
        // this is the critical point where you have to trust that the generated code
        // is safe to be called.

        //unsafe extern "C" (seems to reverse input order or something?)
        let func = mem::transmute::<_, fn(I) -> O>(func_ptr);

        func(input)
    })
}

/// Executes the given code using the cranelift JIT compiler.
///
/// Feeds the given input into the JIT compiled function and returns the resulting output.
///
/// # Safety
///
/// This function is unsafe since it relies on the caller to provide it with the correct
/// input and output types. Using incorrect types at this point may corrupt the program's state.
#[allow(unused_unsafe)]
pub unsafe fn run_string<I, O>(
    jit: &mut jit::JIT,
    code: &str,
    fn_name: &str,
    input: I,
) -> anyhow::Result<O> {
    //Compile code with JIT
    compile_string(jit, code)?;

    Ok(unsafe { run_fn(jit, fn_name, input)? })
}
