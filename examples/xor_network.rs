use std::fs::File;
use std::io::{BufWriter, Write};

use auto::{Guard, Tape};

type Var<'a> = auto::Var<'a, f64>;

// a variable's scope cannot exceed its tape, so lets just use the same lifetime
// for brevity...
fn sigmoid<'a>(x: &Var<'a>) -> Var<'a> {
  // todo add more f64 <op> var operators...
  ((-x).exp() + 1.0) ^ -1.0
}

fn binary_cross_entropy<'a>(a: &Var<'a>, y: f64) -> Var<'a> {
  let term1 = a.ln() * y;
  let term2 = (-a + 1.0).ln() * (1.0 - y);
  -(term1 + term2)
}

/// Feedforward Neural Network for an XOR gate
///
/// Architecture:
/// - 2 inputs
/// - 1 hidden layer with 2 neurons
/// - 1 output neuron
///
/// Note: again, we take the same lifetime for tape and scope...
pub struct XorNet<'a> {
  // hidden params
  w11: Var<'a>,
  w12: Var<'a>,
  b1: Var<'a>,
  w21: Var<'a>,
  w22: Var<'a>,
  b2: Var<'a>,
  // output params
  v1: Var<'a>,
  v2: Var<'a>,
  b_out: Var<'a>,
  learning_rate: f64,
}

impl<'a> XorNet<'a> {
  pub fn new(guard: &Guard<'a>) -> Self {
    // fake random initial weights (i dont want to import rand...)
    XorNet {
      w11: guard.var(1.2),
      w12: guard.var(0.5),
      b1: guard.var(0.6),
      w21: guard.var(1.5),
      w22: guard.var(-0.4),
      b2: guard.var(0.4),
      v1: guard.var(-2.3),
      v2: guard.var(-0.3),
      b_out: guard.var(0.63),
      learning_rate: 0.2,
    }
  }

  pub fn forward(&self, x1: f64, x2: f64) -> Var<'a> {
    // hidden neuron 1
    let z1 = &self.w11 * x1 + &self.w12 * x2 + &self.b1;
    let a1 = sigmoid(&z1);
    // hidden neuron 2
    let z2 = &self.w21 * x1 + &self.w22 * x2 + &self.b2;
    let a2 = sigmoid(&z2);
    // output neuron
    let z_out = &self.v1 * a1 + &self.v2 * a2 + &self.b_out;
    sigmoid(&z_out)
  }

  pub fn train(&mut self, guard: Guard<'_>, epochs: usize) -> f64 {
    let x1_data = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let x2_data = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let y_data = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

    let file = File::create("training_loss.csv").unwrap();
    let mut buf = BufWriter::new(file);
    writeln!(buf, "epoch,loss").unwrap();

    let mut final_loss = 0.0;
    let mut snapshot = guard.lock();
    for epoch in 0..epochs {
      snapshot.scope(|guard| {
        let mut loss_sum = guard.var(0.0);
        let n = x1_data.len() as f64;
        for i in 0..x1_data.len() {
          let z1 = &self.w11 * x1_data[i] + &self.w12 * x2_data[i] + &self.b1;
          let a1 = sigmoid(&z1);
          let z2 = &self.w21 * x1_data[i] + &self.w22 * x2_data[i] + &self.b2;
          let a2 = sigmoid(&z2);
          let z_out = &self.v1 * a1 + &self.v2 * a2 + &self.b_out;
          let a_out = sigmoid(&z_out);
          let loss_sample = binary_cross_entropy(&a_out, y_data[i]);
          loss_sum = loss_sum + loss_sample;
        }
        let loss_avg = loss_sum / n;

        // we finished with this epoch's calculations, lets get some gradients...
        let grads = guard.lock().collapse().of(&loss_avg);

        // simple gradient descent for each weight/bias
        *self.w11 -= self.learning_rate * grads[&self.w11];
        *self.w12 -= self.learning_rate * grads[&self.w12];
        *self.b1 -= self.learning_rate * grads[&self.b1];

        *self.w21 -= self.learning_rate * grads[&self.w21];
        *self.w22 -= self.learning_rate * grads[&self.w22];
        *self.b2 -= self.learning_rate * grads[&self.b2];

        *self.v1 -= self.learning_rate * grads[&self.v1];
        *self.v2 -= self.learning_rate * grads[&self.v2];
        *self.b_out -= self.learning_rate * grads[&self.b_out];

        if epoch % 1000 == 0 {
          println!("Epoch {} | Loss = {:.6}", epoch, *loss_avg);
        }
        let _ = writeln!(buf, "{},{}", epoch, *loss_avg);
        // we track the loss of the network
        final_loss = *loss_avg;
      });
    }
    buf.flush().unwrap();
    final_loss
  }
}

fn main() {
  let mut tape = Tape::new();
  tape.scope(|guard| {
    let epochs = 100_000;
    let mut net = XorNet::new(&guard);
    let mut snapshot = guard.lock();
    let final_loss = snapshot.scope(|guard| net.train(guard, epochs));

    println!("trained parameters:");
    println!("w11 = {}", *net.w11);
    println!("w12 = {}", *net.w12);
    println!("b1  = {}", *net.b1);
    println!("w21 = {}", *net.w21);
    println!("w22 = {}", *net.w22);
    println!("b2  = {}", *net.b2);
    println!("v1  = {}", *net.v1);
    println!("v2  = {}", *net.v2);
    println!("b_out = {}", *net.b_out);
    println!("final loss = {:.6}%\n", final_loss * 100.0);

    let x1_data = [0.0, 0.0, 1.0, 1.0];
    let x2_data = [0.0, 1.0, 0.0, 1.0];
    println!("testing network predictions:");
    for i in 0..x1_data.len() {
      let output = net.forward(x1_data[i], x2_data[i]);
      println!(
        "input: ({}, {}), output: {:.6}",
        x1_data[i], x2_data[i], *output
      );
    }
  });
}
