pub use crate::validator::validate_program;

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
