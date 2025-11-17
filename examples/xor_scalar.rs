use std::fs::File;
use std::io::{BufWriter, Write};

use lib_auto::{Tape, Locked};
use lib_auto::scalar::{Guard, Pullback, Var, VarExt};

// a variable's scope cannot exceed its tape, so lets just use the same lifetime
// for brevity...
fn sigmoid<'a>(x: &Var<'a>) -> Var<'a> {
  // sigmoid(x) = 1 / (1 + exp(-x))
  x.neg().exp().add_f64(1.0).reciprocal()
}

fn binary_cross_entropy<'a>(a: &Var<'a>, y: f64) -> Var<'a> {
  let term1 = a.ln().mul_f64(y);
  let term2 = a.neg().add_f64(1.0).ln().mul_f64(1.0 - y);
  term1.add(&term2).neg()
}

/// Feedforward Neural Network for an XOR gate
///
/// Architecture:
/// - 2 inputs
/// - 1 hidden layer with 2 neurons
/// - 1 output neuron
///
/// Note: again, we take the same lifetime for tape and scope...
struct XorNet<'a> {
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
  fn new(guard: &Guard<'a>) -> Self {
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

  #[inline]
  fn forward(&self, x1: f64, x2: f64) -> Var<'a> {
    // hidden neuron 1
    let z1 = self
      .w11
      .mul_f64(x1)
      .add(&self.w12.mul_f64(x2))
      .add(&self.b1);
    let a1 = sigmoid(&z1);
    // hidden neuron 2
    let z2 = self
      .w21
      .mul_f64(x1)
      .add(&self.w22.mul_f64(x2))
      .add(&self.b2);
    let a2 = sigmoid(&z2);
    // output neuron
    let z_out = self.v1.mul(&a1).add(&self.v2.mul(&a2)).add(&self.b_out);
    sigmoid(&z_out)
  }
}

fn train<'a>(net: &mut XorNet<'a>, guard: Guard<'a, Locked>, epochs: usize) -> f64 {
  let x1_data = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
  let x2_data = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
  let y_data = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

  let file = File::create("training_loss.csv").unwrap();
  let mut buf = BufWriter::new(file);
  writeln!(buf, "epoch,loss").unwrap();

  let mut final_loss = 0.0;
  let mut snapshot = guard;
  for epoch in 0..epochs {
    snapshot.scope(|guard| {
      let mut loss_sum = guard.var(0.0);
      let n = x1_data.len() as f64;
      for i in 0..x1_data.len() {
        let a_out = net.forward(x1_data[i], x2_data[i]);
        let loss_sample = binary_cross_entropy(&a_out, y_data[i]);
        loss_sum = loss_sum.add(&loss_sample);
      }
      let loss_avg = loss_sum.div_f64(n);

      // we finished with this epoch's calculations, lets get some gradients...
      let grads = guard.lock().collapse();
      let dloss = loss_avg.deltas(&grads);

      // simple gradient descent for each weight/bias
      *net.w11.value_mut() -= net.learning_rate * dloss[&net.w11];
      *net.w12.value_mut() -= net.learning_rate * dloss[&net.w12];
      *net.b1.value_mut() -= net.learning_rate * dloss[&net.b1];

      *net.w21.value_mut() -= net.learning_rate * dloss[&net.w21];
      *net.w22.value_mut() -= net.learning_rate * dloss[&net.w22];
      *net.b2.value_mut() -= net.learning_rate * dloss[&net.b2];

      *net.v1.value_mut() -= net.learning_rate * dloss[&net.v1];
      *net.v2.value_mut() -= net.learning_rate * dloss[&net.v2];
      *net.b_out.value_mut() -= net.learning_rate * dloss[&net.b_out];

      if epoch % 1000 == 0 {
        println!("Epoch {} | Loss = {:.6}", epoch, *loss_avg.value());
      }
      let _ = writeln!(buf, "{},{}", epoch, *loss_avg.value());
      // we track the loss of the network
      final_loss = *loss_avg.value();
    });
  }
  buf.flush().unwrap();
  final_loss
}

fn main() {
  let mut tape: Tape<f64, Pullback> = Tape::new();
  tape.scope(|guard| {
    let epochs = 100_000;
    let mut net = XorNet::new(&guard);
    let snapshot = guard.lock();
    let final_loss = train(&mut net, snapshot, epochs);

    println!("trained parameters:");
    println!("w11 = {}", *net.w11.value());
    println!("w12 = {}", *net.w12.value());
    println!("b1  = {}", *net.b1.value());
    println!("w21 = {}", *net.w21.value());
    println!("w22 = {}", *net.w22.value());
    println!("b2  = {}", *net.b2.value());
    println!("v1  = {}", *net.v1.value());
    println!("v2  = {}", *net.v2.value());
    println!("b_out = {}", *net.b_out.value());
    println!("final loss = {:.6}%\n", final_loss * 100.0);

    let x1_data = [0.0, 0.0, 1.0, 1.0];
    let x2_data = [0.0, 1.0, 0.0, 1.0];
    println!("testing network predictions:");
    for i in 0..x1_data.len() {
      let output = net.forward(x1_data[i], x2_data[i]);
      println!(
        "input: ({}, {}), output: {:.6}",
        x1_data[i],
        x2_data[i],
        *output.value()
      );
    }
  });
}