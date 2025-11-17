use lib_auto::scalar::{Pullback, VarExt};
use lib_auto::{Gradient, Tape};

fn main() {
  // Create a new tape (Wengert List) to store the nodes of our computation
  let mut tape: Tape<f64, Pullback> = Tape::new();
  // Define a scope to play around in; you can return the result of the computation
  let () = tape.scope(|guard| {
    let x = guard.var(1.0);
    let y = x.mul(&x);
    // After locking a guard, we can only spawn more subcomputations, or collapse into gradients
    let snap = guard.lock();
    let (_, grads) = snap.collapse();
    let wrt_y = y.deltas(&grads);
    println!("Value: {}, dy/dx: {}", y.value(), wrt_y[&x]);
  });
}