# Sarus

## A jit engine

test with `cargo run --example run_file examples/test`

Derived from https://github.com/bytecodealliance/cranelift-jit-demo

In Progress: (more or less usable)
- Base types: bool, f64, i64, &[f64], &[i64] 
- Structs with impls
- Functions with multiple return variables
- Basic branching (if/then, if/then/else)
- While loop          
- Calling functions defined in rust
- Custom metadata associated functions/expressions


### Sarus Code Example:
```rust
struct Line {
    a: Point,
    b: Point,
}

fn length(self: Line) -> (r: f64) {
    r = sqrt(pow(self.a.x - self.b.x, 2.0) + 
             pow(self.a.y - self.b.y, 2.0) + 
             pow(self.a.z - self.b.z, 2.0))
}

struct Point {
    x: f64,
    y: f64,
    z: f64,
}

fn length(self: Point) -> (r: f64) {
    r = sqrt(pow(self.x, 2.0) + pow(self.y, 2.0) + pow(self.z, 2.0))
}

fn main(n: f64) -> (c: f64) {
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

    d = l1.a //struct is copied
    e = d.x + l1.a.x
    
    p1.y = e * d.z
    p1.y.assert_eq(e * d.z)

    c = l1.length()
}
```