// THIS IS GENERATED FOR QUICK PROFILING

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::{dmatrix, DMatrix};

use lib_auto::matrix::{Pullback as MatrixPullback, VarExt as MatrixVarExt};
use lib_auto::scalar::{Pullback as ScalarPullback, VarExt as ScalarVarExt};
use lib_auto::{Gradient, Tape};

// =============================================================================
// SCALAR BENCHMARKS
// =============================================================================

fn scalar_forward_chain(c: &mut Criterion) {
  let mut group = c.benchmark_group("scalar/forward_chain");

  for chain_len in [10, 50, 100, 500, 1000] {
    group.throughput(Throughput::Elements(chain_len as u64));
    group.bench_with_input(
      BenchmarkId::from_parameter(chain_len),
      &chain_len,
      |b, &len| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();
        b.iter(|| {
          tape.scope(|guard| {
            let mut x = guard.var(black_box(2.0));
            for _ in 0..len {
              x = x.mul(&x).add_f64(1.0).sin();
            }
            black_box(*x.value())
          })
        });
      },
    );
  }
  group.finish();
}

fn scalar_backward_chain(c: &mut Criterion) {
  let mut group = c.benchmark_group("scalar/backward_chain");

  for chain_len in [10, 50, 100, 500, 1000] {
    group.throughput(Throughput::Elements(chain_len as u64));
    group.bench_with_input(
      BenchmarkId::from_parameter(chain_len),
      &chain_len,
      |b, &len| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();
        b.iter(|| {
          tape.scope(|guard| {
            let x = guard.var(black_box(2.0));
            // Build chain starting from x
            let mut result = x.mul_f64(1.0); // identity via mul by 1
            for _ in 0..len {
              result = result.mul(&result).add_f64(1.0).sin();
            }
            let (_, grads) = guard.lock().collapse();
            let deltas = result.deltas(&grads);
            black_box(deltas[&x])
          })
        });
      },
    );
  }
  group.finish();
}

fn scalar_xor_forward(c: &mut Criterion) {
  let mut group = c.benchmark_group("scalar/xor_forward");

  group.bench_function("single_pass", |b| {
    let mut tape: Tape<f64, ScalarPullback> = Tape::new();
    b.iter(|| {
      tape.scope(|guard| {
        // XOR network: 2 inputs, 2 hidden, 1 output
        let w11 = guard.var(1.2);
        let w12 = guard.var(0.5);
        let b1 = guard.var(0.6);
        let w21 = guard.var(1.5);
        let w22 = guard.var(-0.4);
        let b2 = guard.var(0.4);
        let v1 = guard.var(-2.3);
        let v2 = guard.var(-0.3);
        let b_out = guard.var(0.63);

        let x1 = black_box(1.0);
        let x2 = black_box(0.0);

        // Hidden layer
        let z1 = w11.mul_f64(x1).add(&w12.mul_f64(x2)).add(&b1);
        let a1 = z1.neg().exp().add_f64(1.0).reciprocal(); // sigmoid
        let z2 = w21.mul_f64(x1).add(&w22.mul_f64(x2)).add(&b2);
        let a2 = z2.neg().exp().add_f64(1.0).reciprocal();

        // Output
        let z_out = v1.mul(&a1).add(&v2.mul(&a2)).add(&b_out);
        let output = z_out.neg().exp().add_f64(1.0).reciprocal();

        black_box(*output.value())
      })
    });
  });

  group.finish();
}

fn scalar_xor_backward(c: &mut Criterion) {
  let mut group = c.benchmark_group("scalar/xor_backward");

  group.bench_function("single_pass", |b| {
    let mut tape: Tape<f64, ScalarPullback> = Tape::new();
    b.iter(|| {
      tape.scope(|guard| {
        let w11 = guard.var(1.2);
        let w12 = guard.var(0.5);
        let b1 = guard.var(0.6);
        let w21 = guard.var(1.5);
        let w22 = guard.var(-0.4);
        let b2 = guard.var(0.4);
        let v1 = guard.var(-2.3);
        let v2 = guard.var(-0.3);
        let b_out = guard.var(0.63);

        let x1 = black_box(1.0);
        let x2 = black_box(0.0);
        let y = black_box(1.0); // target

        // Forward
        let z1 = w11.mul_f64(x1).add(&w12.mul_f64(x2)).add(&b1);
        let a1 = z1.neg().exp().add_f64(1.0).reciprocal();
        let z2 = w21.mul_f64(x1).add(&w22.mul_f64(x2)).add(&b2);
        let a2 = z2.neg().exp().add_f64(1.0).reciprocal();
        let z_out = v1.mul(&a1).add(&v2.mul(&a2)).add(&b_out);
        let output = z_out.neg().exp().add_f64(1.0).reciprocal();

        // BCE loss
        let term1 = output.ln().mul_f64(y);
        let term2 = output.neg().add_f64(1.0).ln().mul_f64(1.0 - y);
        let loss = term1.add(&term2).neg();

        // Backward
        let (_, grads) = guard.lock().collapse();
        let deltas = loss.deltas(&grads);

        black_box(deltas[&w11])
      })
    });
  });

  group.finish();
}

// =============================================================================
// MATRIX BENCHMARKS
// =============================================================================

fn matrix_forward_chain(c: &mut Criterion) {
  let mut group = c.benchmark_group("matrix/forward_chain");

  for size in [2, 4, 8, 16, 32] {
    for chain_len in [10, 50, 100] {
      group.throughput(Throughput::Elements((size * size * chain_len) as u64));
      group.bench_with_input(
        BenchmarkId::new(format!("{}x{}", size, size), chain_len),
        &(size, chain_len),
        |b, &(sz, len)| {
          let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
          let init_matrix = DMatrix::from_fn(sz, sz, |i, j| ((i + j) as f64) * 0.1);
          b.iter(|| {
            tape.scope(|guard| {
              let mut x = guard.var(black_box(init_matrix.clone()));
              for _ in 0..len {
                x = x.hadamard(&x).add_f64(1.0).exp();
              }
              black_box(x.value().clone())
            })
          });
        },
      );
    }
  }
  group.finish();
}

fn matrix_backward_chain(c: &mut Criterion) {
  let mut group = c.benchmark_group("matrix/backward_chain");

  for size in [2, 4, 8, 16] {
    for chain_len in [10, 50, 100] {
      group.throughput(Throughput::Elements((size * size * chain_len) as u64));
      group.bench_with_input(
        BenchmarkId::new(format!("{}x{}", size, size), chain_len),
        &(size, chain_len),
        |b, &(sz, len)| {
          let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
          let init_matrix = DMatrix::from_fn(sz, sz, |i, j| ((i + j) as f64) * 0.1);
          b.iter(|| {
            tape.scope(|guard| {
              let x = guard.var(black_box(init_matrix.clone()));
              // Build chain with identity start
              let mut result = x.mul_f64(1.0);
              for _ in 0..len {
                result = result.hadamard(&result).add_f64(1.0).exp();
              }
              let (_, grads) = guard.lock().collapse();
              let deltas = result.deltas(&grads);
              black_box(deltas[&x].clone())
            })
          });
        },
      );
    }
  }
  group.finish();
}

fn matrix_matmul(c: &mut Criterion) {
  let mut group = c.benchmark_group("matrix/matmul");

  for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
    group.throughput(Throughput::Elements((size * size) as u64));
    group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &sz| {
      let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
      let mat_a = DMatrix::from_fn(sz, sz, |i, j| ((i * 7 + j * 13) as f64) * 0.01);
      let mat_b = DMatrix::from_fn(sz, sz, |i, j| ((i * 11 + j * 3) as f64) * 0.01);
      b.iter(|| {
        tape.scope(|guard| {
          let a = guard.var(black_box(mat_a.clone()));
          let b_var = guard.var(black_box(mat_b.clone()));
          let result = a.matmul(&b_var);
          let (_, grads) = guard.lock().collapse();
          let deltas = result.deltas(&grads);
          black_box((deltas[&a].clone(), deltas[&b_var].clone()))
        })
      });
    });
  }
  group.finish();
}

fn matrix_xor_forward(c: &mut Criterion) {
  let mut group = c.benchmark_group("matrix/xor_forward");

  for hidden_size in [2, 4, 8, 16] {
    group.bench_with_input(
      BenchmarkId::from_parameter(hidden_size),
      &hidden_size,
      |b, &hidden| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
        let w1 = DMatrix::from_fn(hidden, 2, |i, j| ((i * 7 + j * 13 + 3) as f64 * 0.1).sin());
        let b1 = DMatrix::zeros(hidden, 1);
        let w2 = DMatrix::from_fn(1, hidden, |i, j| ((i * 11 + j * 3 + 7) as f64 * 0.1).sin());
        let b2 = DMatrix::zeros(1, 1);
        let input = dmatrix![1.0; 0.0];

        b.iter(|| {
          tape.scope(|guard| {
            let w1_var = guard.var(black_box(w1.clone()));
            let b1_var = guard.var(black_box(b1.clone()));
            let w2_var = guard.var(black_box(w2.clone()));
            let b2_var = guard.var(black_box(b2.clone()));

            // Forward
            let z1 = w1_var.matmul_const(&input).add(&b1_var);
            let a1 = z1.neg().exp().add_f64(1.0).reciprocal();
            let z2 = w2_var.matmul(&a1).add(&b2_var);
            let output = z2.neg().exp().add_f64(1.0).reciprocal();

            black_box(output.value().clone())
          })
        });
      },
    );
  }
  group.finish();
}

fn matrix_xor_backward(c: &mut Criterion) {
  let mut group = c.benchmark_group("matrix/xor_backward");

  for hidden_size in [2, 4, 8, 16] {
    group.bench_with_input(
      BenchmarkId::from_parameter(hidden_size),
      &hidden_size,
      |b, &hidden| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
        let w1 = DMatrix::from_fn(hidden, 2, |i, j| ((i * 7 + j * 13 + 3) as f64 * 0.1).sin());
        let b1 = DMatrix::zeros(hidden, 1);
        let w2 = DMatrix::from_fn(1, hidden, |i, j| ((i * 11 + j * 3 + 7) as f64 * 0.1).sin());
        let b2 = DMatrix::zeros(1, 1);
        let input = dmatrix![1.0; 0.0];
        let target = 1.0;

        b.iter(|| {
          tape.scope(|guard| {
            let w1_var = guard.var(black_box(w1.clone()));
            let b1_var = guard.var(black_box(b1.clone()));
            let w2_var = guard.var(black_box(w2.clone()));
            let b2_var = guard.var(black_box(b2.clone()));

            // Forward
            let z1 = w1_var.matmul_const(&input).add(&b1_var);
            let a1 = z1.neg().exp().add_f64(1.0).reciprocal();
            let z2 = w2_var.matmul(&a1).add(&b2_var);
            let output = z2.neg().exp().add_f64(1.0).reciprocal();

            // BCE loss
            let term1 = output.ln().mul_f64(target);
            let term2 = output.neg().add_f64(1.0).ln().mul_f64(1.0 - target);
            let loss = term1.add(&term2).neg();

            // Backward
            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);

            black_box(deltas[&w1_var].clone())
          })
        });
      },
    );
  }
  group.finish();
}

// =============================================================================
// COMPARATIVE BENCHMARKS: SCALAR vs MATRIX (equivalent operations)
// =============================================================================

fn compare_scalar_vs_1x1_matrix(c: &mut Criterion) {
  let mut group = c.benchmark_group("comparison/scalar_vs_1x1");

  // Scalar version
  group.bench_function("scalar_chain_100", |b| {
    let mut tape: Tape<f64, ScalarPullback> = Tape::new();
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(2.0));
        let mut result = x.mul_f64(1.0); // identity
        for _ in 0..100 {
          result = result.mul(&result).add_f64(1.0);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x])
      })
    });
  });

  // 1x1 Matrix version (should be comparable if overhead is minimal)
  group.bench_function("matrix_1x1_chain_100", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = dmatrix![2.0];
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let mut result = x.mul_f64(1.0); // identity
        for _ in 0..100 {
          result = result.hadamard(&result).add_f64(1.0);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  group.finish();
}

// =============================================================================
// ALLOCATION HOTSPOT BENCHMARKS
// =============================================================================

fn bench_topological_sort(c: &mut Criterion) {
  let mut group = c.benchmark_group("internals/topo_sort");

  for graph_size in [50, 100, 500, 1000] {
    group.bench_with_input(
      BenchmarkId::from_parameter(graph_size),
      &graph_size,
      |b, &size| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();
        b.iter(|| {
          tape.scope(|guard| {
            // Build a graph of given size
            let x = guard.var(black_box(1.0));
            let mut result = x.mul_f64(1.0); // identity
            for _ in 0..size {
              result = result.mul(&result).add_f64(0.001);
            }
            // This triggers topological sort
            let (_, grads) = guard.lock().collapse();
            let deltas = result.deltas(&grads);
            black_box(deltas[&x])
          })
        });
      },
    );
  }
  group.finish();
}

fn bench_gradient_accumulation(c: &mut Criterion) {
  let mut group = c.benchmark_group("internals/grad_accumulation");

  // Diamond pattern: many paths converge to same variable
  group.bench_function("diamond_scalar", |b| {
    let mut tape: Tape<f64, ScalarPullback> = Tape::new();
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(1.0));
        // Create diamond: x -> (a, b, c, d) -> result
        let a = x.mul_f64(2.0);
        let b = x.mul_f64(3.0);
        let c = x.mul_f64(4.0);
        let d = x.mul_f64(5.0);
        let result = a.add(&b).add(&c).add(&d);

        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x])
      })
    });
  });

  group.bench_function("diamond_matrix_4x4", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = DMatrix::from_element(4, 4, 1.0);
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let a = x.mul_f64(2.0);
        let b = x.mul_f64(3.0);
        let c = x.mul_f64(4.0);
        let d = x.mul_f64(5.0);
        let result = a.add(&b).add(&c).add(&d);

        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  group.finish();
}

// =============================================================================
// THROUGHPUT BENCHMARKS: Elements processed per second
// =============================================================================

/// Compare throughput: how many elements can we differentiate per second?
/// This shows where matrix batching wins over scalar operations.
fn throughput_comparison(c: &mut Criterion) {
  let mut group = c.benchmark_group("throughput/elements_per_sec");

  // Process ~1024 elements total, compare different batch sizes
  const TOTAL_ELEMENTS: usize = 1024;

  // Scalar: 1024 individual operations
  group.throughput(Throughput::Elements(TOTAL_ELEMENTS as u64));
  group.bench_function("scalar_1x1_x1024", |b| {
    let mut tape: Tape<f64, ScalarPullback> = Tape::new();
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(2.0));
        let mut result = x.mul_f64(1.0);
        // 1024 scalar multiplications
        for _ in 0..TOTAL_ELEMENTS {
          result = result.mul(&result);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x])
      })
    });
  });

  // Matrix 4x4: 64 operations (16 elements each = 1024 total)
  group.bench_function("matrix_4x4_x64", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = DMatrix::from_element(4, 4, 2.0);
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let mut result = x.mul_f64(1.0);
        // 64 matrix hadamard ops (16 elements each)
        for _ in 0..(TOTAL_ELEMENTS / 16) {
          result = result.hadamard(&result);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  // Matrix 8x8: 16 operations (64 elements each = 1024 total)
  group.bench_function("matrix_8x8_x16", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = DMatrix::from_element(8, 8, 2.0);
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let mut result = x.mul_f64(1.0);
        // 16 matrix hadamard ops (64 elements each)
        for _ in 0..(TOTAL_ELEMENTS / 64) {
          result = result.hadamard(&result);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  // Matrix 16x16: 4 operations (256 elements each = 1024 total)
  group.bench_function("matrix_16x16_x4", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = DMatrix::from_element(16, 16, 2.0);
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let mut result = x.mul_f64(1.0);
        // 4 matrix hadamard ops (256 elements each)
        for _ in 0..(TOTAL_ELEMENTS / 256) {
          result = result.hadamard(&result);
        }
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  // Matrix 32x32: 1 operation (1024 elements)
  group.bench_function("matrix_32x32_x1", |b| {
    let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
    let init = DMatrix::from_element(32, 32, 2.0);
    b.iter(|| {
      tape.scope(|guard| {
        let x = guard.var(black_box(init.clone()));
        let mut result = x.mul_f64(1.0);
        // 1 matrix hadamard op (1024 elements)
        result = result.hadamard(&result);
        let (_, grads) = guard.lock().collapse();
        let deltas = result.deltas(&grads);
        black_box(deltas[&x].clone())
      })
    });
  });

  group.finish();
}

/// Throughput scaling: how does throughput change with matrix size?
fn throughput_scaling(c: &mut Criterion) {
  let mut group = c.benchmark_group("throughput/scaling");

  // Same number of operations, different matrix sizes
  const NUM_OPS: usize = 10;

  for size in [1, 2, 4, 8, 16, 32, 64] {
    let elements_per_iter = size * size * NUM_OPS;
    group.throughput(Throughput::Elements(elements_per_iter as u64));

    if size == 1 {
      // Special case: scalar
      group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();
        b.iter(|| {
          tape.scope(|guard| {
            let x = guard.var(black_box(2.0));
            let mut result = x.mul_f64(1.0);
            for _ in 0..NUM_OPS {
              result = result.mul(&result);
            }
            let (_, grads) = guard.lock().collapse();
            let deltas = result.deltas(&grads);
            black_box(deltas[&x])
          })
        });
      });
    } else {
      group.bench_with_input(BenchmarkId::new("matrix", size), &size, |b, &sz| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
        let init = DMatrix::from_element(sz, sz, 2.0);
        b.iter(|| {
          tape.scope(|guard| {
            let x = guard.var(black_box(init.clone()));
            let mut result = x.mul_f64(1.0);
            for _ in 0..NUM_OPS {
              result = result.hadamard(&result);
            }
            let (_, grads) = guard.lock().collapse();
            let deltas = result.deltas(&grads);
            black_box(deltas[&x].clone())
          })
        });
      });
    }
  }

  group.finish();
}

/// Real-world throughput: batched forward+backward for neural network layers
fn throughput_neural_layer(c: &mut Criterion) {
  let mut group = c.benchmark_group("throughput/neural_layer");

  // Simulate a single dense layer: output = sigmoid(W @ input + b)
  // Compare: N separate scalar neurons vs 1 matrix layer with N neurons

  for num_neurons in [4, 8, 16, 32, 64] {
    let input_size = 4;
    // Total parameters: num_neurons * input_size (weights) + num_neurons (biases)
    let total_params = num_neurons * input_size + num_neurons;
    group.throughput(Throughput::Elements(total_params as u64));

    // Matrix version: single batched operation
    group.bench_with_input(
      BenchmarkId::new("matrix_layer", num_neurons),
      &num_neurons,
      |b, &n| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
        let w = DMatrix::from_fn(n, input_size, |i, j| ((i * 7 + j * 13) as f64 * 0.1).sin());
        let bias = DMatrix::zeros(n, 1);
        let input = DMatrix::from_fn(input_size, 1, |i, _| (i as f64) * 0.5);

        b.iter(|| {
          tape.scope(|guard| {
            let w_var = guard.var(black_box(w.clone()));
            let b_var = guard.var(black_box(bias.clone()));

            // Forward: sigmoid(W @ x + b)
            let z = w_var.matmul_const(&input).add(&b_var);
            let output = z.neg().exp().add_f64(1.0).reciprocal();

            // Simple loss: sum of outputs
            let loss = output.mul_f64(1.0); // identity for now

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            black_box(deltas[&w_var].clone())
          })
        });
      },
    );

    // Scalar version: N separate neurons (simulated)
    group.bench_with_input(
      BenchmarkId::new("scalar_neurons", num_neurons),
      &num_neurons,
      |b, &n| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();

        b.iter(|| {
          tape.scope(|guard| {
            // Create N neurons, each with input_size weights + 1 bias
            let mut outputs = Vec::with_capacity(n);

            for neuron_idx in 0..n {
              // Weights for this neuron
              let mut z = guard.var(0.0); // accumulator
              for input_idx in 0..input_size {
                let w = guard.var(((neuron_idx * 7 + input_idx * 13) as f64 * 0.1).sin());
                let x_val = (input_idx as f64) * 0.5;
                z = z.add(&w.mul_f64(x_val));
              }
              let b = guard.var(0.0);
              z = z.add(&b);

              // Sigmoid
              let output = z.neg().exp().add_f64(1.0).reciprocal();
              outputs.push(output);
            }

            // Sum outputs as loss
            let mut loss = outputs[0].mul_f64(1.0);
            for out in outputs.iter().skip(1) {
              loss = loss.add(out);
            }

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            // Get gradient of first neuron's first weight
            black_box(deltas.get(&outputs[0]).unwrap_or(0.0))
          })
        });
      },
    );
  }

  group.finish();
}

/// EXTREME neural layer sizes - where matrix batching absolutely dominates
fn throughput_extreme_layers(c: &mut Criterion) {
  let mut group = c.benchmark_group("throughput/extreme_layers");
  // Increase sample size time for slow benchmarks
  group.sample_size(20);
  group.measurement_time(std::time::Duration::from_secs(10));

  let input_size = 16; // Larger input for more realistic workload

  for num_neurons in [128, 256, 512, 1024] {
    let total_params = num_neurons * input_size + num_neurons;
    group.throughput(Throughput::Elements(total_params as u64));

    // Matrix version: single batched operation
    group.bench_with_input(
      BenchmarkId::new("matrix", num_neurons),
      &num_neurons,
      |b, &n| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
        let w = DMatrix::from_fn(n, input_size, |i, j| ((i * 7 + j * 13) as f64 * 0.1).sin());
        let bias = DMatrix::zeros(n, 1);
        let input = DMatrix::from_fn(input_size, 1, |i, _| (i as f64) * 0.1);

        b.iter(|| {
          tape.scope(|guard| {
            let w_var = guard.var(black_box(w.clone()));
            let b_var = guard.var(black_box(bias.clone()));

            // Forward: sigmoid(W @ x + b)
            let z = w_var.matmul_const(&input).add(&b_var);
            let output = z.neg().exp().add_f64(1.0).reciprocal();

            // Loss
            let loss = output.mul_f64(1.0);

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            black_box(deltas[&w_var].clone())
          })
        });
      },
    );

    // Scalar version: N separate neurons - THIS WILL BE SLOW
    group.bench_with_input(
      BenchmarkId::new("scalar", num_neurons),
      &num_neurons,
      |b, &n| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();

        b.iter(|| {
          tape.scope(|guard| {
            let mut outputs = Vec::with_capacity(n);

            for neuron_idx in 0..n {
              let mut z = guard.var(0.0);
              for input_idx in 0..input_size {
                let w = guard.var(((neuron_idx * 7 + input_idx * 13) as f64 * 0.1).sin());
                let x_val = (input_idx as f64) * 0.1;
                z = z.add(&w.mul_f64(x_val));
              }
              let b = guard.var(0.0);
              z = z.add(&b);
              let output = z.neg().exp().add_f64(1.0).reciprocal();
              outputs.push(output);
            }

            let mut loss = outputs[0].mul_f64(1.0);
            for out in outputs.iter().skip(1) {
              loss = loss.add(out);
            }

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            black_box(deltas.get(&outputs[0]).unwrap_or(0.0))
          })
        });
      },
    );
  }

  group.finish();
}

/// Multi-layer network comparison - stacked layers
fn throughput_mlp(c: &mut Criterion) {
  let mut group = c.benchmark_group("throughput/mlp");
  group.sample_size(20);
  group.measurement_time(std::time::Duration::from_secs(8));

  // MLP: input -> hidden1 -> hidden2 -> output
  // Compare matrix batched vs scalar for different hidden sizes

  for hidden_size in [32, 64, 128, 256] {
    let input_size = 16;
    let output_size = 8;
    // Total params: (input->h1) + (h1->h2) + (h2->out) + biases
    let total_params = input_size * hidden_size
      + hidden_size * hidden_size
      + hidden_size * output_size
      + hidden_size
      + hidden_size
      + output_size;
    group.throughput(Throughput::Elements(total_params as u64));

    // Matrix MLP
    group.bench_with_input(
      BenchmarkId::new("matrix_mlp", hidden_size),
      &hidden_size,
      |b, &h| {
        let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();

        let w1 = DMatrix::from_fn(h, input_size, |i, j| ((i * 7 + j * 13) as f64 * 0.01).sin());
        let b1 = DMatrix::zeros(h, 1);
        let w2 = DMatrix::from_fn(h, h, |i, j| ((i * 11 + j * 3) as f64 * 0.01).sin());
        let b2 = DMatrix::zeros(h, 1);
        let w3 = DMatrix::from_fn(output_size, h, |i, j| {
          ((i * 5 + j * 17) as f64 * 0.01).sin()
        });
        let b3 = DMatrix::zeros(output_size, 1);
        let input = DMatrix::from_fn(input_size, 1, |i, _| (i as f64) * 0.1);

        b.iter(|| {
          tape.scope(|guard| {
            let w1_var = guard.var(black_box(w1.clone()));
            let b1_var = guard.var(black_box(b1.clone()));
            let w2_var = guard.var(black_box(w2.clone()));
            let b2_var = guard.var(black_box(b2.clone()));
            let w3_var = guard.var(black_box(w3.clone()));
            let b3_var = guard.var(black_box(b3.clone()));

            // Layer 1
            let z1 = w1_var.matmul_const(&input).add(&b1_var);
            let a1 = z1.neg().exp().add_f64(1.0).reciprocal();

            // Layer 2
            let z2 = w2_var.matmul(&a1).add(&b2_var);
            let a2 = z2.neg().exp().add_f64(1.0).reciprocal();

            // Layer 3 (output)
            let z3 = w3_var.matmul(&a2).add(&b3_var);
            let output = z3.neg().exp().add_f64(1.0).reciprocal();

            let loss = output.mul_f64(1.0);

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            black_box(deltas[&w1_var].clone())
          })
        });
      },
    );

    // Scalar MLP - simulating all neurons individually
    group.bench_with_input(
      BenchmarkId::new("scalar_mlp", hidden_size),
      &hidden_size,
      |b, &h| {
        let mut tape: Tape<f64, ScalarPullback> = Tape::new();

        b.iter(|| {
          tape.scope(|guard| {
            // Layer 1: input_size -> h neurons
            let mut layer1_out = Vec::with_capacity(h);
            for neuron in 0..h {
              let mut z = guard.var(0.0);
              for inp in 0..input_size {
                let w = guard.var(((neuron * 7 + inp * 13) as f64 * 0.01).sin());
                z = z.add(&w.mul_f64((inp as f64) * 0.1));
              }
              let bias = guard.var(0.0);
              z = z.add(&bias);
              layer1_out.push(z.neg().exp().add_f64(1.0).reciprocal());
            }

            // Layer 2: h -> h neurons
            let mut layer2_out = Vec::with_capacity(h);
            for neuron in 0..h {
              let mut z = guard.var(0.0);
              for (inp, prev) in layer1_out.iter().enumerate() {
                let w = guard.var(((neuron * 11 + inp * 3) as f64 * 0.01).sin());
                z = z.add(&w.mul(prev));
              }
              let bias = guard.var(0.0);
              z = z.add(&bias);
              layer2_out.push(z.neg().exp().add_f64(1.0).reciprocal());
            }

            // Layer 3: h -> output_size neurons
            let mut layer3_out = Vec::with_capacity(output_size);
            for neuron in 0..output_size {
              let mut z = guard.var(0.0);
              for (inp, prev) in layer2_out.iter().enumerate() {
                let w = guard.var(((neuron * 5 + inp * 17) as f64 * 0.01).sin());
                z = z.add(&w.mul(prev));
              }
              let bias = guard.var(0.0);
              z = z.add(&bias);
              layer3_out.push(z.neg().exp().add_f64(1.0).reciprocal());
            }

            // Sum for loss
            let mut loss = layer3_out[0].mul_f64(1.0);
            for out in layer3_out.iter().skip(1) {
              loss = loss.add(out);
            }

            let (_, grads) = guard.lock().collapse();
            let deltas = loss.deltas(&grads);
            black_box(deltas.get(&layer3_out[0]).unwrap_or(0.0))
          })
        });
      },
    );
  }

  group.finish();
}

// =============================================================================
// CLONE-HEAVY OPERATION BENCHMARKS
// =============================================================================

fn bench_matrix_clone_intensive(c: &mut Criterion) {
  let mut group = c.benchmark_group("clone_intensive");

  for size in [4, 8, 16, 32] {
    group.bench_with_input(BenchmarkId::new("matmul_chain", size), &size, |b, &sz| {
      let mut tape: Tape<DMatrix<f64>, MatrixPullback> = Tape::new();
      let mat = DMatrix::from_fn(sz, sz, |i, j| ((i + j) as f64) * 0.1);
      b.iter(|| {
        tape.scope(|guard| {
          let a = guard.var(black_box(mat.clone()));
          let b_var = guard.var(black_box(mat.clone()));
          // Chain of matmuls (each captures both matrices)
          let c = a.matmul(&b_var);
          let d = c.matmul(&a);
          let e = d.matmul(&b_var);
          let result = e.matmul(&a);

          let (_, grads) = guard.lock().collapse();
          let deltas = result.deltas(&grads);
          black_box(deltas[&a].clone())
        })
      });
    });
  }
  group.finish();
}
criterion_group!(
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(100));
  targets =
    // Scalar benchmarks
    scalar_forward_chain,
    scalar_backward_chain,
    scalar_xor_forward,
    scalar_xor_backward,
    // Matrix benchmarks
    matrix_forward_chain,
    matrix_backward_chain,
    matrix_matmul,
    matrix_xor_forward,
    matrix_xor_backward,
    // Comparison benchmarks
    compare_scalar_vs_1x1_matrix,
    // Throughput benchmarks
    throughput_comparison,
    throughput_scaling,
    throughput_neural_layer,
    throughput_extreme_layers,
    throughput_mlp,
    // Internal benchmarks
    bench_topological_sort,
    bench_gradient_accumulation,
    bench_matrix_clone_intensive,
);

criterion_main!(benches);
