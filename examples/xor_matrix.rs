use std::fs::File;
use std::io::{BufWriter, Write};

use nalgebra::{dmatrix, DMatrix};

use lib_auto_core::Tape;
use lib_auto_matrix::{Guard, Pullback, Var, VarExt};

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
struct XorNet<'a> {
  // Layer 1: 2 inputs -> 2 hidden neurons
  w1: Var<'a>,
  b1: Var<'a>,

  // Layer 2: 2 hidden -> 1 output neuron
  w2: Var<'a>,
  b2: Var<'a>,

  learning_rate: f64,
}

impl<'a> XorNet<'a> {
  fn new(guard: &Guard<'a>) -> Self {
    XorNet {
      w1: guard.var(dmatrix![1.2, 0.5; 1.5, -0.4]),
      b1: guard.var(dmatrix![0.6; 0.4]),
      w2: guard.var(dmatrix![-2.3; -0.3]),
      b2: guard.var(dmatrix![0.63]),
      learning_rate: 0.2,
    }
  }

  #[inline]
  fn forward(&self, guard: &Guard<'a>, x1: f64, x2: f64) -> Var<'a> {
    let x = guard.var(dmatrix![x1; x2]);
    let z1 = self.w1.t().matmul(&x).add(&self.b1);
    let a1 = sigmoid(&z1);
    let z_out = self.w2.t().matmul(&a1).add(&self.b2);
    sigmoid(&z_out)
  }
}

fn train<'a, 'b>(net: &mut XorNet<'a>, guard: Guard<'b>, epochs: usize) -> f64 {
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
      let mut loss_sum = guard.var(dmatrix![0.0]);
      let n = x1_data.len() as f64;
      for i in 0..x1_data.len() {
        let a_out = net.forward(&guard, x1_data[i], x2_data[i]);
        let loss_sample = binary_cross_entropy(&a_out, y_data[i]);
        loss_sum = loss_sum.add(&loss_sample);
      }
      let loss_avg = loss_sum.div_f64(n);

      // we finished with this epoch's calculations, lets get some gradients...
      let grads = guard.lock().collapse();
      let dloss = grads.of(&loss_avg, dmatrix![1.0]);

      // simple gradient descent for each weight/bias
      let grad_w1 = &dloss[&net.w1] * net.learning_rate;
      let grad_b1 = &dloss[&net.b1] * net.learning_rate;
      let grad_w2 = &dloss[&net.w2] * net.learning_rate;
      let grad_b2 = &dloss[&net.b2] * net.learning_rate;

      *net.w1.value_mut() -= grad_w1;
      *net.b1.value_mut() -= grad_b1;
      *net.w2.value_mut() -= grad_w2;
      *net.b2.value_mut() -= grad_b2;

      if epoch % 1000 == 0 {
        println!("Epoch {} | Loss = {:.6}", epoch, loss_avg.value()[(0, 0)]);
      }
      let _ = writeln!(buf, "{},{}", epoch, loss_avg.value()[(0, 0)]);
      // we track the loss of the network
      final_loss = loss_avg.value()[(0, 0)];
    });
  }
  buf.flush().unwrap();
  final_loss
}

fn main() {
  let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
  tape.scope(|guard| {
    let epochs = 100_000;
    let mut net = XorNet::new(&guard);
    let mut snapshot = guard.lock();
    let final_loss = snapshot.scope(|guard| train(&mut net, guard, epochs));

    println!("trained parameters:");
    println!("W1 =\n{}", net.w1.value());
    println!("b1 =\n{}", net.b1.value());
    println!("W2 =\n{}", net.w2.value());
    println!("b2 = {}", net.b2.value()[(0, 0)]);
    println!("final loss = {:.6}%\n", final_loss * 100.0);

    let x1_data = [0.0, 0.0, 1.0, 1.0];
    let x2_data = [0.0, 1.0, 0.0, 1.0];
    println!("testing network predictions:");
    for i in 0..x1_data.len() {
      snapshot.scope(|guard| {
        let output = net.forward(&guard, x1_data[i], x2_data[i]);
        println!(
          "input: ({}, {}), output: {:.6}",
          x1_data[i],
          x2_data[i],
          output.value()[(0, 0)]
        );
      });
    }
  });
}
