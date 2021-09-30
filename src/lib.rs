use cranelift_jit::JITBuilder;
use frontend::Declaration;

pub use crate::frontend::parser;

pub mod frontend;
pub mod graph;
pub mod jit;
pub mod sarus_std_lib;
pub mod validator;

#[macro_export]
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub fn default_std_jit_from_code(code: &str) -> anyhow::Result<jit::JIT> {
    let mut ast = parser::program(&code)?;
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std_symbols(&mut jit_builder);
    sarus_std_lib::append_std_funcs(&mut ast);
    sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

    let mut jit = jit::JIT::from(jit_builder);
    jit.add_math_constants()?;

    jit.translate(ast.clone())?;
    Ok(jit)
}

//TODO use builder pattern?
pub fn default_std_jit_from_code_with_importer(
    code: &str,
    importer: impl FnOnce(&mut Vec<Declaration>, &mut JITBuilder),
) -> anyhow::Result<jit::JIT> {
    let mut ast = parser::program(&code)?;
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std_symbols(&mut jit_builder);
    sarus_std_lib::append_std_funcs(&mut ast);
    sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

    importer(&mut ast, &mut jit_builder);

    let mut jit = jit::JIT::from(jit_builder);
    jit.add_math_constants()?;

    jit.translate(ast.clone())?;
    Ok(jit)
}
