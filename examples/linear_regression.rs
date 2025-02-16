use std::fs::File;
use std::io::{BufWriter, Write};

use auto::Tape;

fn main() {
  let x1_data = [1.0, 2.0, 3.0, 4.0, 5.0];
  let x2_data = [2.0, 1.0, 0.0, -1.0, 2.0];
  let mut y_data = Vec::with_capacity(x1_data.len());

  for i in 0..x1_data.len() {
    // y = 5.4*x1 - 2.3*x2 - 1.4
    let y = 5.4 * x1_data[i] - 2.3 * x2_data[i] - 1.4;
    y_data.push(y);
  }

  // parameter initialization
  let mut w1_val = 0.0;
  let mut w2_val = 0.0;
  let mut b_val = 0.0;
  let learning_rate = 0.02;
  let epochs = 500000;

  // loss file
  let file = File::create("training_loss.csv").unwrap();
  // we always write to the file in a hot loop -> we can buffer writes since we only
  // care about state at the end (flush after loop)...
  let mut buf = BufWriter::new(file);

  // training loop
  let _ = writeln!(buf, "epoch,loss");
  let mut tape = Tape::new();
  for epoch in 0..epochs {
    let guard = tape.guard();

    // todo; persist...
    let w1 = &guard.var(w1_val);
    let w2 = &guard.var(w2_val);
    let b = &guard.var(b_val);

    let n = x1_data.len() as f64;
    let mut mse = guard.var(0.0);

    for i in 0..x1_data.len() {
      let y_pred = w1 * x1_data[i] + w2 * x2_data[i] + b;
      let err = y_pred - y_data[i];
      // cant move out of err yet...
      let sq_err = &err * &err;
      mse = mse + sq_err;
    }
    mse = mse / n;

    let grads = guard.collapse().of(&mse);
    w1_val -= learning_rate * grads[w1];
    w2_val -= learning_rate * grads[w2];
    b_val -= learning_rate * grads[b];

    if epoch % 30 == 0 {
      println!(
        "epoch {} | MSE = {:.4} | w1 = {:.4} | w2 = {:.4} | b = {:.4}",
        epoch,
        mse.value(),
        w1_val,
        w2_val,
        b_val
      );
    }

    let _ = writeln!(buf, "{},{}", epoch, mse.value());
  }
  buf.flush().unwrap();

  println!("trained parameters:");
  println!("w1 = {w1_val}");
  println!("w2 = {w2_val}");
  println!("b  = {b_val}");
}
