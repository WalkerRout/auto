use std::fs::File;
use std::io::{BufWriter, Write};

use nalgebra::{dmatrix, DMatrix};

use lib_auto::matrix::{Guard, Mutator, Pullback, Var, VarExt};
use lib_auto::{Deltas, Gradient, Locked, Tape};

fn sigmoid<'a>(x: &Var<'a>) -> Var<'a> {
  x.neg().exp().add_f64(1.0).reciprocal()
}

fn binary_cross_entropy<'a>(prediction: &Var<'a>, target: f64) -> Var<'a> {
  let term1 = prediction.ln().mul_f64(target);
  let term2 = prediction.neg().add_f64(1.0).ln().mul_f64(1.0 - target);
  term1.add(&term2).neg()
}

struct Layer<'a> {
  w: Var<'a>,
  b: Var<'a>,
}

impl<'a> Layer<'a> {
  fn new(guard: &Guard<'a>, input_size: usize, output_size: usize) -> Self {
    // xavier/glorot initialization for better example gradient flow
    let scale = (2.0 / (input_size + output_size) as f64).sqrt();
    Layer {
      // output_size x input_size
      w: guard.var(DMatrix::from_fn(output_size, input_size, |i, j| {
        // pseudorandom garbage
        ((i * 7 + j * 13 + 3) as f64 * 0.1).sin() * scale
      })),
      // output_size x 1
      b: guard.var(DMatrix::zeros(output_size, 1)),
    }
  }
}

struct FeedForward<'a> {
  layers: Vec<Layer<'a>>,
  learning_rate: f64,
}

impl<'a> FeedForward<'a> {
  fn new(guard: &Guard<'a>, layer_sizes: &[usize], learning_rate: f64) -> Self {
    assert!(
      layer_sizes.len() >= 2,
      "Need at least input and output layer"
    );

    let mut layers = Vec::new();
    for i in 0..layer_sizes.len() - 1 {
      layers.push(Layer::new(guard, layer_sizes[i], layer_sizes[i + 1]));
    }

    FeedForward {
      layers,
      learning_rate,
    }
  }

  fn forward(&self, x: DMatrix<f64>) -> Var<'a> {
    let mut activation = self.layers[0].w.matmul_const(&x).add(&self.layers[0].b);
    activation = sigmoid(&activation);

    for i in 1..self.layers.len() - 1 {
      activation = self.layers[i].w.matmul(&activation).add(&self.layers[i].b);
      activation = sigmoid(&activation);
    }

    let last = self.layers.len() - 1;
    activation = self.layers[last]
      .w
      .matmul(&activation)
      .add(&self.layers[last].b);
    // todo softmax, need to implement a sum function for matrices...
    sigmoid(&activation)
  }

  fn update_parameters<'scope>(
    &mut self,
    muter: &mut Mutator<'scope>,
    deltas: &Deltas<'scope, DMatrix<f64>>,
  ) where
    'a: 'scope,
  {
    for layer in &mut self.layers {
      let grad_w = &deltas[&layer.w] * self.learning_rate;
      let grad_b = &deltas[&layer.b] * self.learning_rate;
      muter.update(&mut layer.w, |w| w - grad_w);
      muter.update(&mut layer.b, |b| b - grad_b);
    }
  }
}

fn train<'a, 'b>(
  net: &mut FeedForward<'a>,
  mut snapshot: Guard<'b, Locked>,
  x_data: &[(f64, f64)],
  y_data: &[f64],
  epochs: usize,
) -> f64 {
  let file = File::create("training_loss.csv").unwrap();
  let mut buf = BufWriter::new(file);
  writeln!(buf, "epoch,loss").unwrap();

  let mut final_loss = 0.0;

  for epoch in 0..epochs {
    snapshot.scope(|guard| {
      let mut loss_sum = guard.var(dmatrix![0.0]);
      let n = x_data.len() as f64;

      // accumulate loss...
      for i in 0..x_data.len() {
        let (x1, x2) = x_data[i];
        let input = dmatrix![x1; x2];
        let prediction = net.forward(input);
        let loss = binary_cross_entropy(&prediction, y_data[i]);
        loss_sum = loss_sum.add(&loss);
      }
      let loss_avg = loss_sum.div_f64(n);

      let (mut muter, grads) = guard.lock().collapse();
      let deltas = loss_avg.deltas(&grads);
      net.update_parameters(&mut muter, &deltas);

      if epoch % 1000 == 0 {
        println!("Epoch {} | Loss = {:.6}", epoch, loss_avg.value()[(0, 0)]);
      }
      writeln!(buf, "{},{}", epoch, loss_avg.value()[(0, 0)]).unwrap();
      final_loss = loss_avg.value()[(0, 0)];
    });
  }

  buf.flush().unwrap();
  final_loss
}

fn train_network<'a>(guard: Guard<'a>) {
  let layer_sizes = [2, 6, 1];
  let learning_rate = 0.1;
  let epochs = 50_000;
  let mut net = FeedForward::new(&guard, &layer_sizes, learning_rate);

  // training data...
  let x_train = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
  ];
  let y_train = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

  let snapshot = guard.lock();
  let final_loss = train(&mut net, snapshot, &x_train, &y_train, epochs);

  println!("\ntraining complete");
  println!("final loss: {:.6}%\n", final_loss * 100.0);

  let test_data = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
  println!("testing network predictions:");
  println!("┌─────────┬─────────┬────────────┬──────────┐");
  println!("│  input  │  input  │ prediction │ expected │");
  println!("│   x1    │   x2    │            │          │");
  println!("├─────────┼─────────┼────────────┼──────────┤");

  for &(x1, x2) in &test_data {
    let input = dmatrix![x1; x2];
    let output = net.forward(input);
    let prediction = output.value()[(0, 0)];
    // calculate actual on the fly, but this is y_test...
    let expected = if (x1 as i32) ^ (x2 as i32) == 1 {
      1.0
    } else {
      0.0
    };
    let correct = if (prediction > 0.5 && expected == 1.0) || (prediction <= 0.5 && expected == 0.0)
    {
      "OK"
    } else {
      "FAIL"
    };

    println!(
      "│  {:.1}    │  {:.1}    │   {:.4}   │  {:.1}     │ {}",
      x1, x2, prediction, expected, correct
    );
  }
  println!("└─────────┴─────────┴────────────┴──────────┘");

  let total_params: usize = layer_sizes
    .windows(2)
    .map(|w| w[0] * w[1] + w[1]) // weights + biases
    .sum();

  println!("\nnetwork statistics:");
  println!("  total layers: {}", layer_sizes.len());
  println!("  total parameters: {}", total_params);
  println!("  parameters per layer:");
  for i in 0..layer_sizes.len() - 1 {
    let layer_params = layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1];
    println!(
      "    layer {}: {} -> {} ({} params)",
      i + 1,
      layer_sizes[i],
      layer_sizes[i + 1],
      layer_params
    );
  }
}

fn main() {
  let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
  let () = tape.scope(|guard| train_network(guard));
}
