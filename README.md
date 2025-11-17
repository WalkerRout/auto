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
use lib_auto::scalar::{Pullback, VarExt};
use lib_auto::{Gradient, Tape};

fn main() {
  // Create a new tape (Wengert List) to store the nodes of our computation
  let mut tape: Tape<f64, Pullback> = Tape::new();
  // Define a scope to play around in
  tape.scope(|guard| {
    let x = guard.var(1.0);
    let y = x.mul(&x);
    // After locking a guard, we can only spawn more subcomputations, or collapse into gradients
    let snap = guard.lock();
    let (_, grads) = snap.collapse();
    let wrt_y = y.deltas(&grads);
    println!("Value: {}, dy/dx: {}", y.value(), wrt_y[&x]);
  });
}
```

In this snippet, we create a variable, compute a function, and automatically derive its gradient...
