//!
//! # auto-matrix
//!
//! This crate provides matrix operation implementations for the automatic
//! differentiation library using nalgebra for efficient matrix computations.
//!

use nalgebra::DMatrix;

use smallvec::smallvec;

use lib_auto_core::{self as core, OpId, Operation, Deltas, Gradients, PullbackFamily, PullbackSpec, Unlocked};

// Public API type aliases matching auto-scalar pattern
pub type Guard<'a, L = Unlocked> = core::Guard<'a, DMatrix<f64>, Pullback, L>;

pub type Var<'a> = core::Var<'a, DMatrix<f64>, MatrixOp>;

/// Matrix operation types for automatic differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixOp {
  Add,
  Sub,
  MatMul,
  Hadamard,
  Div,
  Neg,
  Exp,
  Ln,
  Reciprocal,
  Transpose,
  MulScalar,
  DivScalar,
  // Constant matrix operations (matrix constant has no gradient)
  AddConst,
  SubConst,
  MulConst,    // Element-wise multiply (Hadamard)
  DivConst,    // Element-wise divide
  MatMulConst, // Matrix multiplication
}

/// Pullback family for matrix operations
pub struct Pullback;

impl PullbackFamily<DMatrix<f64>> for Pullback {
  type Operand = MatrixOp;

  fn apply_a(
    op: Self::Operand,
    captures: &[DMatrix<f64>],
    upstream: &DMatrix<f64>,
  ) -> DMatrix<f64> {
    match op {
      MatrixOp::Add => upstream.clone(),
      MatrixOp::Sub => upstream.clone(),
      MatrixOp::MatMul => {
        let b_t = captures[1].transpose();
        upstream * b_t
      }
      MatrixOp::Hadamard => upstream.component_mul(&captures[1]),
      MatrixOp::Div => upstream.component_div(&captures[1]),
      MatrixOp::Neg => -upstream,
      MatrixOp::Exp => {
        let a = &captures[0];
        a.map(|x| x.exp()).component_mul(upstream)
      }
      MatrixOp::Ln => {
        let a = &captures[0];
        upstream.component_div(a)
      }
      MatrixOp::Reciprocal => {
        let a = &captures[0];
        let a_sq = a.component_mul(a);
        -upstream.component_div(&a_sq)
      }
      MatrixOp::Transpose => upstream.transpose(),
      MatrixOp::MulScalar => {
        let scalar = captures[0][(0, 0)];
        upstream * scalar
      }
      MatrixOp::DivScalar => {
        let scalar = captures[0][(0, 0)];
        upstream / scalar
      }
      MatrixOp::AddConst => {
        // C = A + B_const, dL/dA = dL/dC (identity)
        upstream.clone()
      }
      MatrixOp::SubConst => {
        // C = A - B_const, dL/dA = dL/dC (identity)
        upstream.clone()
      }
      MatrixOp::MulConst => {
        // C = A ⊙ B_const, dL/dA = dL/dC ⊙ B_const
        upstream.component_mul(&captures[0])
      }
      MatrixOp::DivConst => {
        // C = A ⊘ B_const, dL/dA = dL/dC ⊘ B_const
        upstream.component_div(&captures[0])
      }
      MatrixOp::MatMulConst => {
        // C = A * B_const, dL/dA = dL/dC * B_const^T
        let b_const_t = captures[0].transpose();
        upstream * b_const_t
      }
    }
  }

  fn apply_b(
    op: Self::Operand,
    captures: &[DMatrix<f64>],
    upstream: &DMatrix<f64>,
  ) -> DMatrix<f64> {
    match op {
      MatrixOp::Add => upstream.clone(),
      MatrixOp::Sub => -upstream,
      MatrixOp::MatMul => {
        let a_t = captures[0].transpose();
        a_t * upstream
      }
      MatrixOp::Hadamard => upstream.component_mul(&captures[0]),
      MatrixOp::Div => {
        let a = &captures[0];
        let b = &captures[1];
        let b_sq = b.component_mul(b);
        -a.component_div(&b_sq).component_mul(upstream)
      }
      MatrixOp::Neg => DMatrix::zeros(0, 0),
      MatrixOp::Exp => DMatrix::zeros(0, 0),
      MatrixOp::Ln => DMatrix::zeros(0, 0),
      MatrixOp::Reciprocal => DMatrix::zeros(0, 0),
      MatrixOp::Transpose => DMatrix::zeros(0, 0),
      MatrixOp::MulScalar => DMatrix::zeros(0, 0),
      MatrixOp::DivScalar => DMatrix::zeros(0, 0),
      MatrixOp::AddConst => DMatrix::zeros(0, 0),    // Const has no gradient
      MatrixOp::SubConst => DMatrix::zeros(0, 0),    // Const has no gradient
      MatrixOp::MulConst => DMatrix::zeros(0, 0),    // Const has no gradient
      MatrixOp::DivConst => DMatrix::zeros(0, 0),    // Const has no gradient
      MatrixOp::MatMulConst => DMatrix::zeros(0, 0), // Const has no gradient
    }
  }
}

pub struct AddOp;

impl Operation<DMatrix<f64>, Pullback> for AddOp {
  fn forward(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a + b
  }

  fn pullback_spec(
    &self,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::Add),
      op_id_b: OpId::User(MatrixOp::Add),
      captures: smallvec![a.clone(), b.clone()],
    }
  }
}

pub struct SubOp;

impl Operation<DMatrix<f64>, Pullback> for SubOp {
  fn forward(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a - b
  }

  fn pullback_spec(
    &self,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::Sub),
      op_id_b: OpId::User(MatrixOp::Sub),
      captures: smallvec![a.clone(), b.clone()],
    }
  }
}

pub struct MatMulOp;

impl Operation<DMatrix<f64>, Pullback> for MatMulOp {
  fn forward(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a * b
  }

  fn pullback_spec(
    &self,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::MatMul),
      op_id_b: OpId::User(MatrixOp::MatMul),
      captures: smallvec![a.clone(), b.clone()],
    }
  }
}

pub struct HadamardOp;

impl Operation<DMatrix<f64>, Pullback> for HadamardOp {
  fn forward(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.component_mul(b)
  }

  fn pullback_spec(
    &self,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::Hadamard),
      op_id_b: OpId::User(MatrixOp::Hadamard),
      captures: smallvec![a.clone(), b.clone()],
    }
  }
}

pub struct DivOp;

impl Operation<DMatrix<f64>, Pullback> for DivOp {
  fn forward(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a.component_div(b)
  }

  fn pullback_spec(
    &self,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::Div),
      op_id_b: OpId::User(MatrixOp::Div),
      captures: smallvec![a.clone(), b.clone()],
    }
  }
}

// Unary operations - we create a dummy second input that points to self
macro_rules! unary_op {
  ($name:ident, $variant:ident, $forward:expr, $capture:expr) => {
    pub struct $name;

    impl Operation<DMatrix<f64>, Pullback> for $name {
      fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
        $forward(a)
      }

      fn pullback_spec(
        &self,
        a: &DMatrix<f64>,
        _b: &DMatrix<f64>,
      ) -> PullbackSpec<DMatrix<f64>, Pullback> {
        PullbackSpec {
          op_id_a: OpId::User(MatrixOp::$variant),
          op_id_b: OpId::Ignore,
          captures: $capture(a),
        }
      }
    }
  };
}

unary_op!(NegOp, Neg, |a: &DMatrix<f64>| -a, |_a: &DMatrix<f64>| {
  smallvec![]
});
unary_op!(
  ExpOp,
  Exp,
  |a: &DMatrix<f64>| a.map(|x| x.exp()),
  |a: &DMatrix<f64>| smallvec![a.clone()]
);
unary_op!(
  LnOp,
  Ln,
  |a: &DMatrix<f64>| a.map(|x| x.ln()),
  |a: &DMatrix<f64>| smallvec![a.clone()]
);
unary_op!(
  ReciprocalOp,
  Reciprocal,
  |a: &DMatrix<f64>| a.map(|x| 1.0 / x),
  |a: &DMatrix<f64>| smallvec![a.clone()]
);
unary_op!(
  TransposeOp,
  Transpose,
  |a: &DMatrix<f64>| a.transpose(),
  |_a: &DMatrix<f64>| smallvec![]
);

pub struct AddF64Op(pub f64);
pub struct MulF64Op(pub f64);
pub struct DivF64Op(pub f64);

impl Operation<DMatrix<f64>, Pullback> for AddF64Op {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a.map(|x| x + self.0)
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::Identity,
      op_id_b: OpId::Ignore,
      captures: smallvec![],
    }
  }
}

impl Operation<DMatrix<f64>, Pullback> for MulF64Op {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a * self.0
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::MulScalar),
      op_id_b: OpId::Ignore,
      captures: smallvec![DMatrix::from_element(1, 1, self.0)],
    }
  }
}

impl Operation<DMatrix<f64>, Pullback> for DivF64Op {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a / self.0
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::DivScalar),
      op_id_b: OpId::Ignore,
      captures: smallvec![DMatrix::from_element(1, 1, self.0)],
    }
  }
}

pub struct AddConstOp(pub DMatrix<f64>);

impl Operation<DMatrix<f64>, Pullback> for AddConstOp {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a + &self.0
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::AddConst),
      op_id_b: OpId::Ignore,
      captures: smallvec![self.0.clone()],
    }
  }
}

pub struct SubConstOp(pub DMatrix<f64>);

impl Operation<DMatrix<f64>, Pullback> for SubConstOp {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a - &self.0
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::SubConst),
      op_id_b: OpId::Ignore,
      captures: smallvec![self.0.clone()],
    }
  }
}

pub struct MulConstOp(pub DMatrix<f64>);

impl Operation<DMatrix<f64>, Pullback> for MulConstOp {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a.component_mul(&self.0)
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::MulConst),
      op_id_b: OpId::Ignore,
      captures: smallvec![self.0.clone()],
    }
  }
}

pub struct DivConstOp(pub DMatrix<f64>);

impl Operation<DMatrix<f64>, Pullback> for DivConstOp {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a.component_div(&self.0)
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::DivConst),
      op_id_b: OpId::Ignore,
      captures: smallvec![self.0.clone()],
    }
  }
}

pub struct MatMulConstOp(pub DMatrix<f64>);

impl Operation<DMatrix<f64>, Pullback> for MatMulConstOp {
  fn forward(&self, a: &DMatrix<f64>, _b: &DMatrix<f64>) -> DMatrix<f64> {
    a * &self.0
  }

  fn pullback_spec(
    &self,
    _a: &DMatrix<f64>,
    _b: &DMatrix<f64>,
  ) -> PullbackSpec<DMatrix<f64>, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(MatrixOp::MatMulConst),
      op_id_b: OpId::Ignore,
      captures: smallvec![self.0.clone()],
    }
  }
}

/// Extension trait providing matrix operations on Var<DMatrix<f64>>
pub trait VarExt<'scope> {
  fn add(&self, other: &Self) -> Self;

  fn sub(&self, other: &Self) -> Self;

  fn matmul(&self, other: &Self) -> Self;

  fn hadamard(&self, other: &Self) -> Self;

  fn div(&self, other: &Self) -> Self;

  fn neg(&self) -> Self;

  fn exp(&self) -> Self;

  fn ln(&self) -> Self;

  fn reciprocal(&self) -> Self;

  fn t(&self) -> Self;

  fn add_f64(&self, other: f64) -> Self;

  fn mul_f64(&self, other: f64) -> Self;

  fn div_f64(&self, other: f64) -> Self;

  fn add_const(&self, other: &DMatrix<f64>) -> Self;

  fn sub_const(&self, other: &DMatrix<f64>) -> Self;

  fn mul_const(&self, other: &DMatrix<f64>) -> Self;

  fn div_const(&self, other: &DMatrix<f64>) -> Self;

  fn matmul_const(&self, other: &DMatrix<f64>) -> Self;

  fn deltas<F>(&self, gradients: &Gradients<'scope, DMatrix<f64>, F>) -> Deltas<'scope, DMatrix<f64>>
  where
    F: PullbackFamily<DMatrix<f64>, Operand = MatrixOp>;
}

impl<'scope> VarExt<'scope> for core::Var<'scope, DMatrix<f64>, MatrixOp> {
  fn add(&self, other: &Self) -> Self {
    self.binary_op(other, AddOp)
  }

  fn sub(&self, other: &Self) -> Self {
    self.binary_op(other, SubOp)
  }

  fn matmul(&self, other: &Self) -> Self {
    self.binary_op(other, MatMulOp)
  }

  fn hadamard(&self, other: &Self) -> Self {
    self.binary_op(other, HadamardOp)
  }

  fn div(&self, other: &Self) -> Self {
    self.binary_op(other, DivOp)
  }

  fn neg(&self) -> Self {
    self.binary_op(self, NegOp)
  }

  fn exp(&self) -> Self {
    self.binary_op(self, ExpOp)
  }

  fn ln(&self) -> Self {
    self.binary_op(self, LnOp)
  }

  fn reciprocal(&self) -> Self {
    self.binary_op(self, ReciprocalOp)
  }

  fn t(&self) -> Self {
    self.binary_op(self, TransposeOp)
  }

  fn add_f64(&self, other: f64) -> Self {
    self.binary_op(self, AddF64Op(other))
  }

  fn mul_f64(&self, other: f64) -> Self {
    self.binary_op(self, MulF64Op(other))
  }

  fn div_f64(&self, other: f64) -> Self {
    self.binary_op(self, DivF64Op(other))
  }

  fn add_const(&self, other: &DMatrix<f64>) -> Self {
    self.binary_op(self, AddConstOp(other.clone()))
  }

  fn sub_const(&self, other: &DMatrix<f64>) -> Self {
    self.binary_op(self, SubConstOp(other.clone()))
  }

  fn mul_const(&self, other: &DMatrix<f64>) -> Self {
    self.binary_op(self, MulConstOp(other.clone()))
  }

  fn div_const(&self, other: &DMatrix<f64>) -> Self {
    self.binary_op(self, DivConstOp(other.clone()))
  }

  fn matmul_const(&self, other: &DMatrix<f64>) -> Self {
    self.binary_op(self, MatMulConstOp(other.clone()))
  }

  fn deltas<F>(&self, gradients: &Gradients<'scope, DMatrix<f64>, F>) -> Deltas<'scope, DMatrix<f64>>
  where
    F: PullbackFamily<DMatrix<f64>, Operand = MatrixOp>,
  {
    let (rows, cols) = self.value().shape();
    gradients.of(self, DMatrix::from_element(rows, cols, 1.0))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use lib_auto_core::Tape;
  use nalgebra::dmatrix;

  #[test]
  fn test_add() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let b = guard.var(dmatrix![5.0, 6.0; 7.0, 8.0]);
      let c = a.add(&b);

      assert_eq!(*c.value(), dmatrix![6.0, 8.0; 10.0, 12.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));

      assert_eq!(dc[&a], DMatrix::from_element(2, 2, 1.0));
      assert_eq!(dc[&b], DMatrix::from_element(2, 2, 1.0));
    });
  }

  #[test]
  fn test_sub() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![5.0, 6.0; 7.0, 8.0]);
      let b = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let c = a.sub(&b);

      assert_eq!(*c.value(), dmatrix![4.0, 4.0; 4.0, 4.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));

      assert_eq!(dc[&a], DMatrix::from_element(2, 2, 1.0));
      assert_eq!(dc[&b], DMatrix::from_element(2, 2, -1.0));
    });
  }

  #[test]
  fn test_matmul() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let b = guard.var(dmatrix![5.0, 6.0; 7.0, 8.0]);
      let c = a.matmul(&b);
      assert_eq!(*c.value(), dmatrix![19.0, 22.0; 43.0, 50.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      let expected_da = DMatrix::from_element(2, 2, 1.0) * dmatrix![5.0, 7.0; 6.0, 8.0];
      assert_eq!(dc[&a], expected_da);

      let expected_db = dmatrix![1.0, 3.0; 2.0, 4.0] * DMatrix::from_element(2, 2, 1.0);
      assert_eq!(dc[&b], expected_db);
    });
  }

  #[test]
  fn test_hadamard() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let b = guard.var(dmatrix![5.0, 6.0; 7.0, 8.0]);
      let c = a.hadamard(&b);
      assert_eq!(*c.value(), dmatrix![5.0, 12.0; 21.0, 32.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      assert_eq!(dc[&a], dmatrix![5.0, 6.0; 7.0, 8.0]);
      assert_eq!(dc[&b], dmatrix![1.0, 2.0; 3.0, 4.0]);
    });
  }

  #[test]
  fn test_div() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![6.0, 8.0; 10.0, 12.0]);
      let b = guard.var(dmatrix![2.0, 4.0; 5.0, 6.0]);
      let c = a.div(&b);
      assert_eq!(*c.value(), dmatrix![3.0, 2.0; 2.0, 2.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      assert_eq!(dc[&a], dmatrix![0.5, 0.25; 0.2, 1.0/6.0]);

      let expected_db = dmatrix![
        -6.0 / 4.0, -8.0 / 16.0;
        -10.0 / 25.0, -12.0 / 36.0
      ];
      assert_eq!(dc[&b], expected_db);
    });
  }

  #[test]
  fn test_composite() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let b = guard.var(dmatrix![2.0, 0.0; 1.0, 2.0]);

      let matmul = a.matmul(&b);
      let hadamard = a.hadamard(&b);
      let c = matmul.add(&hadamard);

      let expected_matmul = dmatrix![4.0, 4.0; 10.0, 8.0];
      let expected_hadamard = dmatrix![2.0, 0.0; 3.0, 8.0];
      let expected = expected_matmul + expected_hadamard;
      assert_eq!(*c.value(), expected);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      // verify they exists and are finite... todo make better...
      assert!(dc[&a].iter().all(|x| x.is_finite()));
      assert!(dc[&b].iter().all(|x| x.is_finite()));
    });
  }

  #[test]
  fn test_matmul_const() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let w = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let x = dmatrix![5.0; 6.0];
      let y = w.matmul_const(&x);
      assert_eq!(*y.value(), dmatrix![17.0; 39.0]);

      let grads = guard.lock().collapse();
      let dy = grads.of(&y, dmatrix![1.0; 1.0]);
      assert_eq!(dy[&w], dmatrix![5.0, 6.0; 5.0, 6.0]);
    });
  }

  #[test]
  fn test_add_const() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![1.0, 2.0; 3.0, 4.0]);
      let b_const = dmatrix![10.0, 20.0; 30.0, 40.0];
      let c = a.add_const(&b_const);
      assert_eq!(*c.value(), dmatrix![11.0, 22.0; 33.0, 44.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      // gradient is identity...
      assert_eq!(dc[&a], DMatrix::from_element(2, 2, 1.0));
    });
  }

  #[test]
  fn test_mul_const() {
    let mut tape: Tape<DMatrix<f64>, Pullback> = Tape::new();
    tape.scope(|guard| {
      let a = guard.var(dmatrix![2.0, 3.0; 4.0, 5.0]);
      let b_const = dmatrix![1.0, 2.0; 3.0, 4.0];
      let c = a.mul_const(&b_const);
      assert_eq!(*c.value(), dmatrix![2.0, 6.0; 12.0, 20.0]);

      let grads = guard.lock().collapse();
      let dc = grads.of(&c, DMatrix::from_element(2, 2, 1.0));
      assert_eq!(dc[&a], b_const);
    });
  }
}