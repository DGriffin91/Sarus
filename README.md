# Sarus

## A jit engine

test with `cargo run --example run_file examples/test`

Derived from https://github.com/bytecodealliance/cranelift-jit-demo

DONE:
- Basic support of f64 type
- Functions with multiple return variables
- Basic branching (if/then, if/then/else)
- While loop

TODO:
- Calling functions defined in rust
- Array type
- Struct type
- Int type
- Statically initialized heap allocated array-like type