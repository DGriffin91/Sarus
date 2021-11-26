# Sarus Language Documentation

- [Types](#types)
    - [Basic Types](#basic-types)
    - [Arrays](#arrays)
    - [Slices](#slices)
    - [Structs](#structs)

- [Control Flow](#control-flow)
    - [While Loop](#while-loop)

- [Functions](#functions)
    - [Methods](#methods)
    - [Closures](#closures)

- [Math](#math)
- [Rust Interop](#rust-interop)

-----

Most of the following examples are run using:
```rust
use sarus::*;
use std::mem;
fn main() {
    //Define Sarus Code
    let code = r#"
fn main() -> () {
    "Hello, World!".println()
}
"#;
    // Get jit default instance
    let mut jit = default_std_jit_from_code(code).unwrap();

    // Get pointer to Sarus main function
    let func_ptr = jit.get_func("main").unwrap();

    // Get function that can be called from rust code
    let func = unsafe { mem::transmute::<_, extern "C" fn()>(func_ptr) };

    // Call Sarus Function
    func();
}
```

Sarus is JIT compiled to executable machine code using [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) for code generation at run time. In many cases it should have similar performance to unoptimized, or lightly optimized C or Rust.

Sarus does not use a garbage collector, and does not make heap allocations at run time. Sarus can optionally allocate space for large data types at compile time. All memory used by Sarus is either on the stack, or in the compile time preallocated space.

Sarus is in very early stages of development. There will likely be many changes to the language before it stabilizes. Features for memory safety are planned, but few are currently implemented. 

# Variables

In Sarus variables are always mutable. In each function, a variable can only refer to a single type.

```rust , ignore
x = 5   // x is a 64 bit integer
x = 5.0 // will result in an error at JIT compile time
//cannot assign value of type f32 to variable x of type i64
```

# Types


## Basic Types

```rust , skt-sarus_single_func
a = 1      // i64
b = 1.0    // f32
c = true   // bool
//d = 'a'  // byte (TODO not yet implemented)
```

Expression types are inferred implicitly. Operations are only allowed between expressions of the same type.

```rust , ignore
a = 1 + 1.0 // will result in an error at JIT compile time
// Type mismatch; expected i64, found f32
```

Types can have associated methods. These are covered later in the guide.
There are included methods for converting between numeric types.
The `assert_eq` method will panic if the values are not equal.

```rust , skt-sarus_single_func
a = 1       // i64
b = a.f32() // f32
c = 1.0     // f32
d = c.i64() // i64

b.assert_eq(c)
a.assert_eq(d)
```

## Arrays

Arrays in sarus can contain any other type. Every element in the array must of be the same type. Arrays are fixed in length, and the length of the array is encoded in the type.

```rust , skt-sarus_single_func
a = [0, 1, 2, 3]             // [i64; 4]
b = [0, 1, 2, 3, 4]          // [i64; 5]
c = [0.0, 1.0, 2.0]          // [f32; 3]
d = [[0.0, 1.0], [2.0, 3.0]] // [[f32; 2]; 2]
```

Array elements are accessed via indexing. Indexing arrays can only be done using the i64 type.
```rust , skt-sarus_single_func
a = [0, 1, 2, 3] // [i64; 4]
b = a[2]         // i64

b.assert_eq(2)
```

Out of bounds access will result in an aborting panic at JIT runtime
```rust , ignore
a = [0, 1, 2, 3] // [i64; 4]
d = a[4] // index out of bounds
```

In the future there will be support for wrapping & clipping accessors. There will also be support for falling back to clipping and reporting an error at runtime instead of aborting.

Multidimensional array access:
```rust , skt-sarus_single_func
a = [[0.0, 1.0], [2.0, 3.0]] // [[f32; 2]; 2]
b = a[0]                     // [f32; 2]

b[1].assert_eq(1.0)
(a[1])[1].assert_eq(3.0) 
//TODO - don't require parenthesis for multidimensional array access
```

Initializing large arrays:
```rust , skt-sarus_single_func
a = [0.0; 10000] // [f32; 10000]

a[0].assert_eq(0.0)
a[9999].assert_eq(0.0)
```

Types of any size can be allocated (given there is sufficient memory avaliable)
```rust , ignore
a = [0.0; 800000000] // 3200MB of floats
```
Most data in Sarus is allocated on the stack. However, the amount of memory avaliable on the stack is limited. To enable larger allocations without relying on runtime heap allocations the memory for this operation is allocated at compile time using the *deep stack*. The deep stack operates similarly to the stack, but is allocated on the heap at compile time. Currently, anything over 4KB is allocated on the deep stack. The deep stack can also be optionally disabled.

When arrays are assigned to another variable, the new variable refers to the same array:
```rust , skt-sarus_single_func
a = [0.0; 2] // [f32; 2]
b = a        // [f32; 2]
b[0] = 1.0   // modifies array underlying both a and b

a[0].assert_eq(1.0)
```

## Slices

Slices in Sarus are more similar slices in Go then the ones in Rust. In Sarus, slices refer to a contiguous segment of an underlying array. Slices contain both a length and a capacity.

```rust , skt-sarus_single_func
a = [0, 1, 2, 3, 4, 5] // array [i64; 6] 
b = a[..] // [i64] Slice pointing to the start of a. Has a length and capacity of 6

b.len().assert_eq(6)
b.cap().assert_eq(6)
```

Get subslice of array:
```rust , skt-sarus_single_func
a = [0, 1, 2, 3, 4, 5] // array [i64; 6] 
b = a[1..4] // [i64] Slice of array a starting at 1 up until, but not including 4

b.len().assert_eq(3)
b.cap().assert_eq(5)
b[0].assert_eq(1)
b[1].assert_eq(2)
b[2].assert_eq(3)

c = b[1..3] // [i64] Slice of slice b starting at 1 up until, but not including 3

c.len().assert_eq(2)
c.cap().assert_eq(4)
c[0].assert_eq(2)
c[1].assert_eq(3)
```

Slices can be made of existing slices that expand the size. (up to the slice capacity):
```rust , skt-sarus_single_func
a = [0, 1, 2, 3, 4, 5]
b = a[0..2]

b.len().assert_eq(2)
b.cap().assert_eq(6)

c = b[..6]

c.len().assert_eq(6)
c.cap().assert_eq(6)
c[4].assert_eq(4)
c[5].assert_eq(5)
```

Assignments to slice indices change the underlying array:
```rust , skt-sarus_single_func
a = [0, 1, 2, 3, 4, 5]
b = a[..]
b[5] = 500

a[5].assert_eq(500)
```

Slices have methods `push` and `pop`
```rust , skt-sarus_single_func
a = [0; 100][0..0] // [i64] Slice of array with length of 0 and capacity of 100

a.len().assert_eq(0)
a.cap().assert_eq(100)

a.push(1)
a.push(2)
a.push(3)

a.len().assert_eq(3)
a[0].assert_eq(1)
a[1].assert_eq(2)
a[2].assert_eq(3)

a.pop().assert_eq(3)

a.len().assert_eq(2)
```

Slices can also be appended to other slices or arrays using the `append` method
```rust , skt-sarus_single_func
a = [0; 100][0..0] // [i64] Slice of array with length of 0 and capacity of 100
a.append([0, 1, 2])

a.len().assert_eq(3)
a[0].assert_eq(0)
a[1].assert_eq(1)
a[2].assert_eq(2)

b = [5, 6, 7, 8, 9, 10]
c = b[1..3]
a.append(c)

a.len().assert_eq(5)
a[3].assert_eq(6)
a[4].assert_eq(7)
```

## Structs

The memory layout of structs in Sarus are designed to be compatible with C FFI. (Like `#[repr(C)]` in Rust) This allows for more convenient interop between Sarus and Rust (or other C FFI compatible sources).

```rust , skt-sarus_multi_func
struct Point {
    x: f32,
    y: f32,
    z: f32,
}
fn main() -> () {
    p = Point {
        x: 100.0,
        y: 200.0,
        z: 300.0,
    }
    p.x.assert_eq(100.0)
    p.y.assert_eq(200.0)
    p.z.assert_eq(300.0)
}
```

The `f32` type is assumed is if no type is given:
```rust , skt-sarus_multi_func
struct Point { x, y, z, }
fn main() -> () {
    p = Point {
        x: 100.0,
        y: 200.0,
        z: 300.0,
    }
}
```

Structs can be stored in arrays:
```rust , skt-sarus_multi_func
struct Point { x, y, z, }
fn main() -> () {
    pts = [Point {
               x: 100.0,
               y: 200.0,
               z: 300.0,
           }; 10] // [Point; 10]

    pts[1].x.assert_eq(100.0)
    pts[1].y.assert_eq(200.0)
    pts[1].z.assert_eq(300.0)
}
```

Structs can contain other structs:
```rust , skt-sarus_multi_func
struct Point { x, y, z, }
struct Line { 
    p1: Point,
    p2: Point,
}
fn main() -> () {
    p1 = Point {
        x: 123.0,
        y: 234.0,
        z: 456.0,
    }
    p2 = Point {
        x: 1230.0,
        y: 2340.0,
        z: 4560.0,
    }
    line = Line {
        p1: p1,
        p2: p2,
    }

    line.p1.x.assert_eq(123.0)
    line.p1.y.assert_eq(234.0)
    line.p1.z.assert_eq(456.0)
    line.p2.x.assert_eq(1230.0)
    line.p2.y.assert_eq(2340.0)
    line.p2.z.assert_eq(4560.0)

    //when a struct is put into another it is copied
    line.p1.x = 0.0
    p1.x.assert_eq(123.0)
}
```

Structs can contain arrays:
```rust , skt-sarus_multi_func
struct SubStuff { a, }
struct Stuff { 
    numbers: [i64; 10],
    something: bool,
    sub_thing: SubStuff,
}
fn main() -> () {
    initial_numbers = [0; 10]
    a_stuff = Stuff { 
        numbers: initial_numbers, //array is copied into struct
        something: true,
        sub_thing: SubStuff {a:0.0,},
    }
    initial_numbers[0] = 500
    a_stuff.numbers[0].assert_eq(0)

    // When a field that is either an array or struct is assigned 
    // to a variable, that variable points to the array or struct 
    // allocated in the parent struct, this operation does not 
    // create a copy

    sub_thing = a_stuff.sub_thing
    a_stuff.sub_thing.a = 5.0
    sub_thing.a.assert_eq(5.0)

    numbers = a_stuff.numbers
    a_stuff.numbers[1] = 5
    numbers[1].assert_eq(5)
}
```

Structs can contain slices:
```rust , skt-sarus_multi_func
struct Stuff { 
    numbers: [i64],
}
fn main() -> () {
    a = [0, 1, 2, 3, 4, 5, 6][..]
    a_stuff = Stuff { 
        numbers: a, // Only slice is copied, not underlying array
    }
    a_stuff.numbers.a[1].assert_eq(1)
    a[0] = 123 // Writes 123 to underlying array
    a_stuff.numbers.a[0].assert_eq(123)
}
```

# Control Flow

`if` expressions conditionally branch based on the state of a boolean value

```rust , skt-sarus_single_func
a = 5
if true {
    a = 6
}
a.assert_eq(6)

if a == 6 {
    a = 7
}
a.assert_eq(7)

if a > 6 || false {
    a += 1
}
a.assert_eq(8)
```

`if/else` & `if / else if / else` expressions return their last expression as a value
```rust , skt-sarus_single_func
(if false {5} else {6}).assert_eq(6)
(if false {5} else if true {6} else {7}).assert_eq(6)
```

## While Loop

While loops conditionally
```rust , skt-sarus_single_func
a = [0; 100][0..0] // Slice with length of 0 and capacity of 100
i = 0
while i < a.cap() {
    a.push(i)
    i += 1
}

i.assert_eq(100)
a[99].assert_eq(99)
```

# Functions

In Sarus both the parameters and the returns are named, and have type definitions:
```rust , skt-sarus_multi_func
fn add(a: i64, b: i64) -> (c: i64) {
    c = a + b
}
fn main() -> () {
    c = add(1, 2)
    c.assert_eq(3)
}
```

If no type definition is given f32 is assumed:
```rust , skt-sarus_multi_func
fn add(a, b) -> (c) {
    c = a + b
}
fn main() -> () {
    c = add(1.0, 2.0)
    c.assert_eq(3.0)
}
```

Functions can be inlined using the inline keyword. This can result in a decent improvement in performance for small functions.
```rust , skt-sarus_multi_func
inline fn add(a, b) -> (c) {
    c = a + b
}
fn main() -> () {
    c = add(1.0, 2.0)
    c.assert_eq(3.0)
}
```

Functions can also have multiple return values:
```rust , skt-sarus_multi_func
fn a_bunch_of_stuff(a, b) -> (c, d, e, f, g, h, i, j) {
    c, d, e, f = a + b, a * b, a / b, b * b
    g, h, i, j = c * a, d * a, e * a, f * a
}
fn main() -> () {
    c, d, e, f, g, h, i, j = a_bunch_of_stuff(1.0, 2.0)
}
```

Functions can return arrays:
```rust , skt-sarus_multi_func
fn an_array(a) -> (b: [f32;100]) {
    b = [a; 100]
}
fn main() -> () {
    b = an_array(5.0)
    b[20].assert_eq(5.0)
}
```

Currently, if the return value is a struct or array only one return value is supported. In the future there will be support for returning multiple arrays, structs, etc...

If a function returns a slice, the function must be inlined. This is because the memory for the underlying array is reclamed when the caller's scope ends, if the array was also allocated in that function. 

When a C FFI function needs to return more data than will fit in registers, the caller will allocate the memory required for the return value and pass a pointer to a memory location where the return value can be written. But in Sarus when returning a slice, it is only returning a pointer to memory that is allocated in the callee. If the function is inlined, and the compiler sees that the function returns a slice, it will not reclaim the memory allocated in the callee.

```rust , skt-sarus_multi_func
inline fn a_slice(a) -> (b: [f32]) {
    b = [a; 100][..]
}
fn main() -> () {
    b = a_slice(5.0)
    b[20].assert_eq(5.0)
}
```

# Methods

All types can have methods. Methods operate on their associated types. Methods are declared by using the identifier `self` as the first parameter of a function.
```rust , skt-sarus_multi_func
fn times_2(self: i64) -> (y: i64) {
    y = self * 2
}
fn main() -> () {
    (5).times_2().assert_eq(10)
}
```

Using methods with structs:
```rust , skt-sarus_multi_func
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

# Closures

In Sarus, closures are inlined, and are not an actual values. Their callsite must be known at compile time. This means they are more limited in some ways than in other languages. But it also means they don't incur a runtime cost, and memory safety is much simpler because they are ultimately executed in the scope they close around.

Closures currently have very similar syntax to functions, with both named parameters and returns. In the future, the return names and possibly even types will not be required.

```rust , skt-sarus_single_func
c = 5.0 + 6.0
a_closure|e| -> () {
    c *= e
}
if c > 10.0 {
    a_closure(5.0)
} else {
    a_closure(4.0)
}
c.assert_eq(55.0)
```

Closures can be passed to functions that are marked with `always_inline`
```rust , skt-sarus_multi_func
always_inline fn map_f32_slice(a: [f32], some_closure: |x| -> (y)) -> () {
    i = 0
    while i < a.len() {
        a[i] = some_closure(a[i])
        i += 1
    }
}
fn main() -> () {
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..]
    c = 5.0
    map_f32_slice(a, |x| -> (y) {y = x * x + c})
    a[0].assert_eq(6.0)
    a[1].assert_eq(9.0)
    a[2].assert_eq(14.0)
}
```

TODO - map functions for using slices with closures: map, sort, filter, fold, sort, etc...

# Math

Sarus provides a portion of the Rust math library for f32 operations.

```rust , skt-sarus_multi_func
fn nums() -> (r) {
    r = E
    r += FRAC_1_PI + FRAC_1_SQRT_2 + FRAC_2_SQRT_PI
    r += FRAC_PI_2 + FRAC_PI_3 + FRAC_PI_4 + FRAC_PI_6 + FRAC_PI_8
    r += LN_2 + LN_10
    r += LOG2_10 + LOG2_E + LOG10_2 + LOG10_E
    r += PI + TAU
    r += SQRT_2 
}
fn main() -> () {
    c = 200.0
    c = c.sin()
    c = c.cos()
    c = c.tan()
    c = c.asin()
    c = c.acos()
    c = c.atan()
    c = c.exp()
    c = c.log(E)
    c = c.log10()
    c = (c + 10.0).sqrt()
    c = c.sinh()
    c = c.cosh()
    c = (c * 0.00001).tanh()
    c = c.atan2(100.0)
    c = c.powf(100.0 * 0.001)
    c *= nums()
}
```

# Rust Interop

To pass structs between Sarus and Rust they need to be declared with `#[repr(C)]`. Sarus functions are callable using `extern "C"`. In most cases the C FFI doesn't support multiple returns so Sarus functions that are going to be called from Rust should only have one return.

```rust 
use sarus::*;
use std::mem;

#[repr(C)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn main() {
    //Define Sarus Code
    let code = r#"
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn length(self: Point) -> (r: f32) {
    r = (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt()
}

fn main(p1: Point) -> (c: f32) {
    c = p1.length()
}
"#;
    // Get jit default instance
    let mut jit = default_std_jit_from_code(code).unwrap();

    // Get pointer to Sarus main function
    let func_ptr = jit.get_func("main").unwrap();

    // Get function that can be called from rust code
    let func = unsafe { mem::transmute::<_, extern "C" fn(Point) -> f32>(func_ptr) };

    let p2 = Point {
        x: 100.0,
        y: 200.0,
        z: 300.0,
    };

    // Call Sarus Function
    assert_eq!(func(p2), 374.16574)
}
```

Calling Rust function from Sarus:
```rust 
use sarus::*;
use std::mem;

extern "C" fn length(p: Point) -> f32 {
    (p.x.powf(2.0) + p.y.powf(2.0) + p.z.powf(2.0)).sqrt()
}

#[repr(C)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn main() {
    let code = r#"
extern fn length(self: Point) -> (l: f32) {}

struct Point {
    x: f32,
    y: f32,
    z: f32,
}

fn main(p1: Point) -> () {
    p1.length().assert_eq(374.16574)
}
"#;
    let ast = parse(code).unwrap();
    let mut jit = default_std_jit_from_code_with_importer(ast, None, |_ast, jit_builder| {
        // Give Rust function pointer to Sarus compiler
        // When a function is a method, we use the format struct_name.method_name
        jit_builder.symbols([("Point.length", length as *const u8)]);
    }).unwrap();
    let func_ptr = jit.get_func("main").unwrap();
    let func = unsafe { mem::transmute::<_, extern "C" fn(Point) >(func_ptr) };

    let p2 = Point {
        x: 100.0,
        y: 200.0,
        z: 300.0,
    };

    func(p2)
}
```