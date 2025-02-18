use std::fs::File;
use std::io::{BufWriter, Write};

use auto::{Guard, Tape};

fn linear_regression(guard: Guard<'_, '_>) {
  let x1_data = [1.0, 2.0, 3.0, 4.0, 5.0];
  let x2_data = [2.0, 1.0, 0.0, -1.0, 2.0];
  let mut y_data = Vec::with_capacity(x1_data.len());
  // y = 5.4*x1 - 2.3*x2 - 1.4 for each sample
  for i in 0..x1_data.len() {
    let y = 5.4 * x1_data[i] - 2.3 * x2_data[i] - 1.4;
    y_data.push(y);
  }

  let learning_rate = 0.02;
  let epochs = 100000;

  let file = File::create("training_loss.csv").unwrap();
  let mut buf = BufWriter::new(file);
  writeln!(buf, "epoch,loss").unwrap();

  let mut w1 = guard.var(0.0);
  let mut w2 = guard.var(0.0);
  let mut b = guard.var(0.0);

  let mut snapshot = guard.lock();

  for epoch in 0..epochs {
    snapshot.scope(|guard| {
      let n = x1_data.len() as f64;
      let mut mse = guard.var(0.0);
      for i in 0..x1_data.len() {
        // y_pred = w1*x1 + w2*x2 + b
        let y_pred = &w1 * x1_data[i] + &w2 * x2_data[i] + &b;
        let err = y_pred - y_data[i];
        let sq_err = &err * &err;
        mse = mse + sq_err;
      }
      mse = mse / n;

      let grads = guard.lock().collapse().of(&mse);
      *w1 = *w1 - learning_rate * grads[&w1];
      *w2 = *w2 - learning_rate * grads[&w2];
      *b = *b - learning_rate * grads[&b];

      #[cfg(debug_assertions)]
      if epoch % 30 == 0 {
        println!(
          "epoch {} | MSE = {:.4} | w1 = {:.4} | w2 = {:.4} | b = {:.4}",
          epoch, *mse, *w1, *w2, *b
        );
      }
      let _ = writeln!(buf, "{},{}", epoch, *mse);
    });
  }
  println!("trained parameters:");
  println!("w1 = {w1:?}");
  println!("w2 = {w2:?}");
  println!("b  = {b:?}");

  buf.flush().unwrap();
}

fn main() {
  let mut tape = Tape::new();
  tape.scope(linear_regression);
}
