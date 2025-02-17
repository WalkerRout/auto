# auto: Automatic Differentiation with Rust

## Features

- **Reverse Mode Differentiation:**
  Ideal for functions with many inputs and one output (think neural nets).

- **Foolproof API:**
  Lifetimes and phantom types ensure correctness at compile time, at no cost to the API consumer.

- **Scoped Variables:**
  Keep your variables around and reuse them across computations. This allows for effective variable storage in model structures, bounded by the variable scope's lifetime.

## Getting Started

Here’s a quick example:

```rust
use auto::Tape;

fn main() {
  // Create a new tape (Wengert list) to store the nodes of our computation
  let mut tape = Tape::new();
  
  // Define a scope to play around in
  tape.scope(|guard| {
    let x = guard.var(2.0);
    let y = x.sin() + x.cos();
    // After locking a guard, we can only spawn more subcomputations, or collapse into gradients
    let grads = guard.lock().collapse().of(&y);
    println!("Value: {}, dy/dx: {}", *y, grads[&x]);
  });
}

```

In this snippet, we create a variable, compute a function, and automatically derive its gradient...
