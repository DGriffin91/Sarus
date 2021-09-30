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

pub fn default_std_jit_from_code(
    code: &str,
    symbols: Option<Vec<(&str, *const u8)>>,
) -> anyhow::Result<jit::JIT> {
    let ast = parser::program(&code)?;
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std_symbols(&mut jit_builder);
    let ast = sarus_std_lib::append_std_funcs(ast);
    let ast = sarus_std_lib::append_std_math(ast, &mut jit_builder);

    if let Some(symbols) = symbols {
        jit_builder.symbols(symbols);
    }
    let mut jit = jit::JIT::from(jit_builder);
    jit.add_math_constants()?;

    jit.translate(ast.clone())?;
    Ok(jit)
}
