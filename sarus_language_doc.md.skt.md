```rust,skt-sarus_single_func
use sarus::*;
use std::mem;
fn main() {{
    let code = r#"fn main() -> () {{ {}
}}
"#;
    let mut jit = default_std_jit_from_code(code).unwrap();
    let func_ptr = jit.get_func("main").unwrap();
    let func = unsafe {{ mem::transmute::<_, extern "C" fn()>(func_ptr) }};
    func();
}}
```

```rust,skt-sarus_multi_func
use sarus::*;
use std::mem;
fn main() {{
    let code = r#"
{}
"#;
    let mut jit = default_std_jit_from_code(code).unwrap();
    let func_ptr = jit.get_func("main").unwrap();
    let func = unsafe {{ mem::transmute::<_, extern "C" fn()>(func_ptr) }};
    func();
}}
```