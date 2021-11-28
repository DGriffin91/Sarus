# Sarus

## A JIT compiled language designed for realtime contexts
### No hidden allocations or GC

Sarus is in very early stages of development.

[Sarus Language Documentation](https://github.com/DGriffin91/Sarus/blob/main/sarus_language_doc.md)

See `tests/integration_test.rs` for a breadth of code examples.
[sarus-editor-plugin](https://github.com/DGriffin91/sarus-editor-plugin) uses Sarus to JIT compile DSP/UI code in a VST plugin.

test with `cargo test`

In Progress: (more or less usable)
- Base types: bool, u8, f32, i64
- Composite types: Fixed length arrays [f32; 100], Slices [f32], Strings [u8]
- `repr(C)` Structs with method implementations
- Functions with multiple return variables, and optional inlineing 
- Basic branching (if/then, if/then/else, etc...)
- While loop   
- Call Sarus functions from Rust and vice versa with `extern "C"`
- Custom metadata associated functions/expressions


### Sarus Code Example:
```rust
struct Line {
    a: Point,
    b: Point,
}

fn length(self: Line) -> (r: f32) {
    r = ((self.a.x - self.b.x).powf(2.0) + 
         (self.a.y - self.b.y).powf(2.0) + 
         (self.a.z - self.b.z).powf(2.0)).sqrt()
}

struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn length(self: Point) -> (r: f32) {
    r = (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt()
}

fn main(n: f32) -> (c: f32) {
    p1 = Point {
        x: n,
        y: 200.0,
        z: 300.0,
    }
    p2 = Point {
        x: n * 4.0,
        y: 500.0,
        z: 600.0,
    }
    l1 = Line {
        a: p1,
        b: p2,
    }

    d = l1.a
    e = d.x + l1.a.x
    
    p1.y = e * d.z
    p1.y.assert_eq(e * d.z)

    c = l1.length()
}
```