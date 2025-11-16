use std::ops::Mul;

use auto::Tape;

fn main() {
  // Create a new tape (Wengert List) to store the nodes of our computation
  let mut tape: Tape<f64> = Tape::default();
  // Define a scope to play around in
  tape.scope(|guard| {
    let x = guard.var(1.0);
    let y = x.mul(&x);
    // After locking a guard, we can only spawn more subcomputations, or collapse into gradients
    let grads = guard.lock().collapse().of(&y);
    println!("Value: {}, dy/dx: {}", y.value(), grads[&x]);
  });
}
