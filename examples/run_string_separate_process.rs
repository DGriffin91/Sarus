use std::mem;

use sarus::{jit, parser, validate_program};

use mitosis::{self, JoinHandle};

// NOTE: this method won't work in a VST plugin.
// It will try to open the whole DAW in a second instance

fn subprocess_code<I, O>(input: (String, I)) -> Result<O, String> {
    let (code, values) = input;

    // Parse and validate (this could probably happen safely in the same process)
    let ast = parser::program(&code).map_err(|e| format!("parser failed: {}", e))?;
    let ast = validate_program(ast).map_err(|e| format!("validate ast failed: {}", e))?;

    let mut jit = jit::JIT::default();
    // Pass the AST to the JIT to compile
    jit.translate(ast)
        .map_err(|e| format!("jit translate failed: {}", e))?;

    // Run compiled code
    //unsafe { run_fn(&mut jit, "main", values).map_err(|e| format!("run_fn main failed: {}", e)) }

    let func_ptr = jit
        .get_func("main")
        .map_err(|e| format!("get_func failed: {}", e))?;
    let func = unsafe { mem::transmute::<_, extern "C" fn(I) -> O>(func_ptr) };
    Ok(func(values))
}

fn main() -> anyhow::Result<()> {
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

    mitosis::init();

    // Spawn separate process that will jit compile code, returning result
    let runner: JoinHandle<Result<f64, String>> =
        mitosis::spawn((code.to_string(), (100.0f64, 200.0f64)), subprocess_code);

    // Wait for the child process to return a result
    match runner.join() {
        Ok(res) => match res {
            Ok(res) => println!("the answer is: {:?}", res),
            Err(e) => println!("test failed: {:?}", e),
        },
        Err(e) => println!("sub process failed: {:?}", e),
    }

    Ok(())
}
