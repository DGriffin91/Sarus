use std::path::PathBuf;

use cranelift_jit::JITBuilder;

use validator::ExprType;

pub use frontend::parse;
pub use frontend::parse_with_context;
pub use frontend::{Arg, Declaration, Function};

use crate::frontend::InlineKind;

pub mod frontend;
pub mod graph;
pub mod jit;
pub mod logging;
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
    let mut ast = parse(&code)?;
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std(&mut ast, &mut jit_builder);
    sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

    let mut jit = jit::JIT::from(jit_builder);

    jit.translate(ast, None)?;
    Ok(jit)
}

//TODO use builder pattern?
pub fn default_std_jit_from_code_with_importer(
    mut ast: Vec<Declaration>,
    file_index_table: Option<Vec<PathBuf>>,
    importer: impl FnOnce(&mut Vec<Declaration>, &mut JITBuilder),
) -> anyhow::Result<jit::JIT> {
    let mut jit_builder = jit::new_jit_builder();
    sarus_std_lib::append_std(&mut ast, &mut jit_builder);
    sarus_std_lib::append_std_math(&mut ast, &mut jit_builder);

    importer(&mut ast, &mut jit_builder);

    let mut jit = jit::JIT::from(jit_builder);

    jit.translate(ast, file_index_table)?;
    Ok(jit)
}

#[macro_export]
macro_rules! decl {
        ( $prog:expr, $jit_builder:expr, $name:expr, $func:expr,  ($( $param:expr ),*), ($( $ret:expr ),*) ) => {

            {

                #[allow(unused_mut)]
                let mut params = Vec::new();
                $(
                    params.push(Arg {
                        name: format!("in{}", params.len()),
                        expr_type: $param,
                        default_to_float: false,
                        closure_arg: None,
                    });
                )*

                #[allow(unused_mut)]
                let mut returns = Vec::new();
                $(
                    returns.push(Arg {
                        name: format!("out{}", returns.len()),
                        expr_type: $ret,
                        default_to_float: false,
                        closure_arg: None,
                    });
                )*

                $jit_builder.symbol($name, $func as *const u8);

                $prog.push(Declaration::Function(Function::external($name.to_string(), params, returns)))

            }
        };
    }

fn make_decl(
    name: &str,
    params: Vec<(&str, ExprType)>,
    returns: Vec<(&str, ExprType)>,
) -> Declaration {
    Declaration::Function(Function {
        name: name.to_string(),
        params: params
            .into_iter()
            .map(|(name, expr)| Arg {
                name: name.to_string(),
                expr_type: expr,
                default_to_float: false,
                closure_arg: None,
            })
            .collect(),
        returns: returns
            .into_iter()
            .map(|(name, expr)| Arg {
                name: name.to_string(),
                expr_type: expr,
                default_to_float: false,
                closure_arg: None,
            })
            .collect(),
        body: vec![],
        extern_func: true,
        inline: InlineKind::Default,
    })
}
