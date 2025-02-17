use auto::Tape;

fn main() {
  // Create a new tape (Wengert List) to store the nodes of our computation
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
