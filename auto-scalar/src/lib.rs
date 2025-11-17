//!
//! # auto-scalar
//!
//! This crate provides f64 (scalar) operation implementations for the library
//!

use smallvec::SmallVec;

use lib_auto_core::{self as core,
  Deltas, Gradients, OpId, Operation, PullbackFamily, PullbackSpec, Unlocked,
};

pub type Guard<'a, L = Unlocked> = core::Guard<'a, f64, Pullback, L>;

pub type Var<'a> = core::Var<'a, f64, <Pullback as PullbackFamily<f64>>::Operand>;

/// Scalar operation types for f64 automatic differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarOp {
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Powf,
  Sin,
  Cos,
  Tan,
  Ln,
  Log,
  Asin,
  Acos,
  Atan,
  Sinh,
  Cosh,
  Tanh,
  Asinh,
  Acosh,
  Atanh,
  Exp,
  Exp2,
  Reciprocal,
  Neg,
  Abs,
}

/// Pullback family for scalar f64 operations
pub struct Pullback;

impl PullbackFamily<f64> for Pullback {
  type Operand = ScalarOp;

  fn apply_a(op: Self::Operand, captures: &[f64], upstream: &f64) -> f64 {
    match op {
      ScalarOp::Add => *upstream,
      ScalarOp::Sub => *upstream,
      ScalarOp::Mul => captures[0] * upstream,
      ScalarOp::Div => upstream / captures[1],
      ScalarOp::Pow => {
        let a = captures[0];
        let b = captures[1];
        b * a.powf(b - 1.0) * upstream
      }
      ScalarOp::Powf => {
        let a = captures[0];
        let exp = captures[1];
        exp * a.powf(exp - 1.0) * upstream
      }
      ScalarOp::Sin => captures[0].cos() * upstream,
      ScalarOp::Cos => -captures[0].sin() * upstream,
      ScalarOp::Tan => {
        let a = captures[0];
        upstream / (a.cos() * a.cos())
      }
      ScalarOp::Ln => upstream / captures[0],
      ScalarOp::Log => {
        let a = captures[0];
        let base = captures[1];
        upstream / (a * base.ln())
      }
      ScalarOp::Asin => {
        let a = captures[0];
        upstream / (1.0 - a * a).sqrt()
      }
      ScalarOp::Acos => {
        let a = captures[0];
        -upstream / (1.0 - a * a).sqrt()
      }
      ScalarOp::Atan => {
        let a = captures[0];
        upstream / (1.0 + a * a)
      }
      ScalarOp::Sinh => captures[0].cosh() * upstream,
      ScalarOp::Cosh => captures[0].sinh() * upstream,
      ScalarOp::Tanh => {
        let a = captures[0];
        upstream / (a.cosh() * a.cosh())
      }
      ScalarOp::Asinh => {
        let a = captures[0];
        upstream / (1.0 + a * a).sqrt()
      }
      ScalarOp::Acosh => {
        let a = captures[0];
        upstream / (a * a - 1.0).sqrt()
      }
      ScalarOp::Atanh => {
        let a = captures[0];
        upstream / (1.0 - a * a).sqrt()
      }
      ScalarOp::Exp => captures[0].exp() * upstream,
      ScalarOp::Exp2 => captures[0].exp2() * 2.0_f64.ln() * upstream,
      ScalarOp::Reciprocal => {
        let a = captures[0];
        -upstream / (a * a)
      }
      ScalarOp::Neg => -upstream,
      ScalarOp::Abs => {
        let a = captures[0];
        (a / a.abs()) * upstream
      }
    }
  }

  fn apply_b(op: Self::Operand, captures: &[f64], upstream: &f64) -> f64 {
    match op {
      ScalarOp::Add => *upstream,
      ScalarOp::Sub => -*upstream,
      ScalarOp::Mul => captures[1] * upstream, // multiply by a
      ScalarOp::Div => {
        let a = captures[0];
        let b = captures[1];
        -a / (b * b) * upstream
      }
      ScalarOp::Pow => {
        let a = captures[0];
        let b = captures[1];
        a.powf(b) * a.ln() * upstream
      }
      _ => 0.0, // unary operations don't use b branch
    }
  }
}

pub struct AddOp;

impl Operation<f64, Pullback> for AddOp {
  fn forward(&self, a: &f64, b: &f64) -> f64 {
    a + b
  }

  fn pullback_spec(&self, _a: &f64, b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Add),
      op_id_b: OpId::User(ScalarOp::Add),
      captures: SmallVec::from_slice(&[*b]),
    }
  }
}

pub struct SubOp;

impl Operation<f64, Pullback> for SubOp {
  fn forward(&self, a: &f64, b: &f64) -> f64 {
    a - b
  }

  fn pullback_spec(&self, _a: &f64, b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Sub),
      op_id_b: OpId::User(ScalarOp::Sub),
      captures: SmallVec::from_slice(&[*b]),
    }
  }
}

pub struct MulOp;

impl Operation<f64, Pullback> for MulOp {
  fn forward(&self, a: &f64, b: &f64) -> f64 {
    a * b
  }

  fn pullback_spec(&self, a: &f64, b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Mul),
      op_id_b: OpId::User(ScalarOp::Mul),
      captures: SmallVec::from_slice(&[*b, *a]),
    }
  }
}

pub struct DivOp;

impl Operation<f64, Pullback> for DivOp {
  fn forward(&self, a: &f64, b: &f64) -> f64 {
    a / b
  }

  fn pullback_spec(&self, a: &f64, b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Div),
      op_id_b: OpId::User(ScalarOp::Div),
      captures: SmallVec::from_slice(&[*a, *b]),
    }
  }
}

pub struct PowOp;

impl Operation<f64, Pullback> for PowOp {
  fn forward(&self, a: &f64, b: &f64) -> f64 {
    a.powf(*b)
  }

  fn pullback_spec(&self, a: &f64, b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Pow),
      op_id_b: OpId::User(ScalarOp::Pow),
      captures: SmallVec::from_slice(&[*a, *b]),
    }
  }
}

pub struct PowfOp(pub f64);

impl Operation<f64, Pullback> for PowfOp {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a.powf(self.0)
  }

  fn pullback_spec(&self, a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Powf),
      op_id_b: OpId::Ignore,
      captures: SmallVec::from_slice(&[*a, self.0]),
    }
  }
}

// Unary operations - we create a dummy second input that points to self
macro_rules! unary_op {
  ($name:ident, $variant:ident, $forward:expr, $capture:expr) => {
    pub struct $name;

    impl Operation<f64, Pullback> for $name {
      fn forward(&self, a: &f64, _b: &f64) -> f64 {
        $forward(*a)
      }

      fn pullback_spec(&self, a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
        PullbackSpec {
          op_id_a: OpId::User(ScalarOp::$variant),
          op_id_b: OpId::Ignore,
          captures: $capture(*a),
        }
      }
    }
  };
}

unary_op!(SinOp, Sin, |a: f64| a.sin(), |a: f64| SmallVec::from_slice(
  &[a]
));
unary_op!(CosOp, Cos, |a: f64| a.cos(), |a: f64| SmallVec::from_slice(
  &[a]
));
unary_op!(TanOp, Tan, |a: f64| a.tan(), |a: f64| SmallVec::from_slice(
  &[a]
));
unary_op!(LnOp, Ln, |a: f64| a.ln(), |a: f64| SmallVec::from_slice(&[
  a
]));
unary_op!(AsinOp, Asin, |a: f64| a.asin(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(AcosOp, Acos, |a: f64| a.acos(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(AtanOp, Atan, |a: f64| a.atan(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(SinhOp, Sinh, |a: f64| a.sinh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(CoshOp, Cosh, |a: f64| a.cosh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(TanhOp, Tanh, |a: f64| a.tanh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(AsinhOp, Asinh, |a: f64| a.asinh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(AcoshOp, Acosh, |a: f64| a.acosh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(AtanhOp, Atanh, |a: f64| a.atanh(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(ExpOp, Exp, |a: f64| a.exp(), |a: f64| SmallVec::from_slice(
  &[a]
));
unary_op!(Exp2Op, Exp2, |a: f64| a.exp2(), |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(ReciprocalOp, Reciprocal, |a: f64| 1.0 / a, |a: f64| {
  SmallVec::from_slice(&[a])
});
unary_op!(NegOp, Neg, |a: f64| -a, |_a: f64| SmallVec::new());
unary_op!(AbsOp, Abs, |a: f64| a.abs(), |a: f64| SmallVec::from_slice(
  &[a]
));

pub struct LogOp(pub f64);

impl Operation<f64, Pullback> for LogOp {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a.log(self.0)
  }

  fn pullback_spec(&self, a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Log),
      op_id_b: OpId::Ignore,
      captures: SmallVec::from_slice(&[*a, self.0]),
    }
  }
}

pub struct AddF64Op(pub f64);

impl Operation<f64, Pullback> for AddF64Op {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a + self.0
  }

  fn pullback_spec(&self, _a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::Identity,
      op_id_b: OpId::Ignore,
      captures: SmallVec::new(),
    }
  }
}

pub struct SubF64Op(pub f64);

impl Operation<f64, Pullback> for SubF64Op {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a - self.0
  }

  fn pullback_spec(&self, _a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::Identity,
      op_id_b: OpId::Ignore,
      captures: SmallVec::new(),
    }
  }
}

pub struct MulF64Op(pub f64);

impl Operation<f64, Pullback> for MulF64Op {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a * self.0
  }

  fn pullback_spec(&self, _a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Mul),
      op_id_b: OpId::Ignore,
      captures: SmallVec::from_slice(&[self.0, 0.0]),
    }
  }
}

pub struct DivF64Op(pub f64);

impl Operation<f64, Pullback> for DivF64Op {
  fn forward(&self, a: &f64, _b: &f64) -> f64 {
    a / self.0
  }

  fn pullback_spec(&self, _a: &f64, _b: &f64) -> PullbackSpec<f64, Pullback> {
    PullbackSpec {
      op_id_a: OpId::User(ScalarOp::Mul),
      op_id_b: OpId::Ignore,
      captures: SmallVec::from_slice(&[1.0 / self.0, 0.0]),
    }
  }
}

/// Extension trait providing scalar operations on Var<f64>
pub trait VarExt<'scope> {
  fn add(&self, other: &Self) -> Self;

  fn add_f64(&self, other: f64) -> Self;

  fn sub(&self, other: &Self) -> Self;

  fn sub_f64(&self, other: f64) -> Self;

  fn mul(&self, other: &Self) -> Self;

  fn mul_f64(&self, other: f64) -> Self;

  fn div(&self, other: &Self) -> Self;

  fn div_f64(&self, other: f64) -> Self;

  fn pow(&self, other: &Self) -> Self;

  fn powf(&self, exp: f64) -> Self;

  fn sin(&self) -> Self;

  fn cos(&self) -> Self;

  fn tan(&self) -> Self;

  fn ln(&self) -> Self;

  fn log(&self, base: f64) -> Self;

  fn log10(&self) -> Self;

  fn log2(&self) -> Self;

  fn asin(&self) -> Self;

  fn acos(&self) -> Self;

  fn atan(&self) -> Self;

  fn sinh(&self) -> Self;

  fn cosh(&self) -> Self;

  fn tanh(&self) -> Self;

  fn asinh(&self) -> Self;

  fn acosh(&self) -> Self;

  fn atanh(&self) -> Self;

  fn exp(&self) -> Self;

  fn exp2(&self) -> Self;

  fn reciprocal(&self) -> Self;

  fn sqrt(&self) -> Self;

  fn cbrt(&self) -> Self;

  fn abs(&self) -> Self;

  fn neg(&self) -> Self;

  fn grads<F>(&self, gradients: &Gradients<'scope, f64, F>) -> Deltas<'scope, f64>
  where
    F: PullbackFamily<f64, Operand = ScalarOp>;
}

impl<'scope> VarExt<'scope> for core::Var<'scope, f64, ScalarOp> {
  fn add(&self, other: &Self) -> Self {
    self.binary_op(other, AddOp)
  }

  fn add_f64(&self, other: f64) -> Self {
    self.binary_op(self, AddF64Op(other))
  }

  fn sub(&self, other: &Self) -> Self {
    self.binary_op(other, SubOp)
  }

  fn sub_f64(&self, other: f64) -> Self {
    self.binary_op(self, SubF64Op(other))
  }

  fn mul(&self, other: &Self) -> Self {
    self.binary_op(other, MulOp)
  }

  fn mul_f64(&self, other: f64) -> Self {
    self.binary_op(self, MulF64Op(other))
  }

  fn div(&self, other: &Self) -> Self {
    self.binary_op(other, DivOp)
  }

  fn div_f64(&self, other: f64) -> Self {
    self.binary_op(self, DivF64Op(other))
  }

  fn pow(&self, other: &Self) -> Self {
    self.binary_op(other, PowOp)
  }

  fn powf(&self, exp: f64) -> Self {
    self.binary_op(self, PowfOp(exp))
  }

  fn sin(&self) -> Self {
    self.binary_op(self, SinOp)
  }

  fn cos(&self) -> Self {
    self.binary_op(self, CosOp)
  }

  fn tan(&self) -> Self {
    self.binary_op(self, TanOp)
  }

  fn ln(&self) -> Self {
    self.binary_op(self, LnOp)
  }

  fn log(&self, base: f64) -> Self {
    self.binary_op(self, LogOp(base))
  }

  fn log10(&self) -> Self {
    self.log(10.0)
  }

  fn log2(&self) -> Self {
    self.log(2.0)
  }

  fn asin(&self) -> Self {
    self.binary_op(self, AsinOp)
  }

  fn acos(&self) -> Self {
    self.binary_op(self, AcosOp)
  }

  fn atan(&self) -> Self {
    self.binary_op(self, AtanOp)
  }

  fn sinh(&self) -> Self {
    self.binary_op(self, SinhOp)
  }

  fn cosh(&self) -> Self {
    self.binary_op(self, CoshOp)
  }

  fn tanh(&self) -> Self {
    self.binary_op(self, TanhOp)
  }

  fn asinh(&self) -> Self {
    self.binary_op(self, AsinhOp)
  }

  fn acosh(&self) -> Self {
    self.binary_op(self, AcoshOp)
  }

  fn atanh(&self) -> Self {
    self.binary_op(self, AtanhOp)
  }

  fn exp(&self) -> Self {
    self.binary_op(self, ExpOp)
  }

  fn exp2(&self) -> Self {
    self.binary_op(self, Exp2Op)
  }

  fn reciprocal(&self) -> Self {
    self.binary_op(self, ReciprocalOp)
  }

  fn sqrt(&self) -> Self {
    self.powf(0.5)
  }

  fn cbrt(&self) -> Self {
    self.powf(1.0 / 3.0)
  }

  fn abs(&self) -> Self {
    self.binary_op(self, AbsOp)
  }

  fn neg(&self) -> Self {
    self.binary_op(self, NegOp)
  }

  fn grads<F>(&self, gradients: &Gradients<'scope, f64, F>) -> Deltas<'scope, f64>
  where
    F: PullbackFamily<f64, Operand = ScalarOp>,
  {
    // scalar seed is 1.0
    gradients.of(self, 1.0)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use lib_auto_core::Tape;

  mod var {
    use super::*;

    #[test]
    fn value() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        assert_eq!(*a.value(), 1.3);
      });
    }

    #[test]
    fn add() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let b = guard.var(4.0);
        let c = a.add(&b);
        assert_eq!(*c.value(), 7.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1, df/db = 1
        assert_eq!(dc[&a], 1.0);
        assert_eq!(dc[&b], 1.0);
      });
    }

    #[test]
    fn add_f64() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let c = a.add_f64(5.0);
        assert_eq!(*c.value(), 8.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1
        assert_eq!(dc[&a], 1.0);
      });
    }

    #[test]
    fn sub() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(7.0);
        let b = guard.var(4.0);
        let c = a.sub(&b);
        assert_eq!(*c.value(), 3.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1, df/db = -1
        assert_eq!(dc[&a], 1.0);
        assert_eq!(dc[&b], -1.0);
      });
    }

    #[test]
    fn sub_f64() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(7.0);
        let c = a.sub_f64(3.0);
        assert_eq!(*c.value(), 4.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1
        assert_eq!(dc[&a], 1.0);
      });
    }

    #[test]
    fn mul() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let b = guard.var(4.0);
        let c = a.mul(&b);
        assert_eq!(*c.value(), 12.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = b, df/db = a
        assert_eq!(dc[&a], 4.0);
        assert_eq!(dc[&b], 3.0);
      });
    }

    #[test]
    fn mul_f64() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let c = a.mul_f64(5.0);
        assert_eq!(*c.value(), 15.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 5
        assert_eq!(dc[&a], 5.0);
      });
    }

    #[test]
    fn div() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(6.0);
        let b = guard.var(3.0);
        let c = a.div(&b);
        assert_eq!(*c.value(), 2.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1/b, df/db = -a/b^2
        assert_eq!(dc[&a], 1.0 / 3.0);
        assert_eq!(dc[&b], -6.0 / 9.0);
      });
    }

    #[test]
    fn div_f64() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(6.0);
        let c = a.div_f64(2.0);
        assert_eq!(*c.value(), 3.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 1/2
        assert_eq!(dc[&a], 0.5);
      });
    }

    #[test]
    fn pow() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let b = guard.var(3.0);
        let c = a.pow(&b);
        assert_eq!(*c.value(), 8.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = b * a^(b-1)
        // df/db = a^b * ln(a)
        assert_eq!(dc[&a], 3.0 * 4.0);
        assert_eq!(dc[&b], 8.0 * 2.0f64.ln());
      });
    }

    #[test]
    fn powf() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let c = a.powf(3.0);
        assert_eq!(*c.value(), 8.0);
        let grads = guard.lock().collapse();
        let dc = grads.of(&c, 1.0);
        // df/da = 3 * a^(3-1)
        assert_eq!(dc[&a], 12.0);
      });
    }

    #[test]
    fn neg() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let neg_a = a.neg();
        assert_eq!(*neg_a.value(), -2.0);
        let grads = guard.lock().collapse();
        let dneg_a = grads.of(&neg_a, 1.0);
        // df/da = -1
        assert_eq!(dneg_a[&a], -1.0);
      });
    }

    #[test]
    fn reciprocal() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.reciprocal();
        assert_eq!(*b.value(), 1.0 / 1.3);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = -1/a^2
        assert_eq!(db[&a], -1.0 / (1.3 * 1.3));
      });
    }

    #[test]
    fn sin() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.sin();
        assert_eq!(*b.value(), 1.3f64.sin());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = cos(a)
        assert_eq!(db[&a], 1.3f64.cos());
      });
    }

    #[test]
    fn cos() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.1);
        let b = a.cos();
        assert_eq!(*b.value(), 3.1f64.cos());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = -sin(a)
        assert_eq!(db[&a], -3.1f64.sin());
      });
    }

    #[test]
    fn tan() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(5.6);
        let b = a.tan();
        assert_eq!(*b.value(), 5.6f64.tan());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = sec^2(a)
        assert_eq!(db[&a], 1.0 / (5.6f64.cos().powi(2)));
      });
    }

    #[test]
    fn ln() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(5.6);
        let b = a.ln();
        assert_eq!(*b.value(), 5.6f64.ln());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/a
        assert_eq!(db[&a], 1.0 / 5.6);
      });
    }

    #[test]
    fn log() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(5.6);
        let base = 3.0;
        let b = a.log(base);
        assert_eq!(*b.value(), 5.6f64.log(base));
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(a ln(b))
        assert_eq!(db[&a], 1.0 / (5.6 * base.ln()));
      });
    }

    #[test]
    fn log10() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.0);
        let b = a.log10();
        assert_eq!(*b.value(), 0.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(a ln(10))
        assert_eq!(db[&a], 1.0 / 10.0f64.ln());
      });
    }

    #[test]
    fn log2() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let b = a.log2();
        assert_eq!(*b.value(), 1.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(a ln(2))
        assert_eq!(db[&a], 1.0 / (2.0 * 2.0f64.ln()));
      });
    }

    #[test]
    fn asin() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(0.5);
        let b = a.asin();
        assert_eq!(*b.value(), 0.5f64.asin());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/sqrt(1-a^2)
        assert_eq!(db[&a], 1.0 / f64::sqrt(1.0 - 0.25));
      });
    }

    #[test]
    fn acos() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(0.5);
        let b = a.acos();
        assert_eq!(*b.value(), 0.5f64.acos());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = -1/sqrt(1-a^2)
        assert_eq!(db[&a], -1.0 / f64::sqrt(1.0 - 0.25));
      });
    }

    #[test]
    fn atan() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.2);
        let b = a.atan();
        assert_eq!(*b.value(), 1.2f64.atan());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(1+a^2)
        assert_eq!(db[&a], 1.0 / (1.0 + 1.44));
      });
    }

    #[test]
    fn sinh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.sinh();
        assert_eq!(*b.value(), 1.3f64.sinh());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = cosh(a)
        assert_eq!(db[&a], 1.3f64.cosh());
      });
    }

    #[test]
    fn cosh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.cosh();
        assert_eq!(*b.value(), 1.3f64.cosh());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = sinh(a)
        assert_eq!(db[&a], 1.3f64.sinh());
      });
    }

    #[test]
    fn tanh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(0.8);
        let b = a.tanh();
        assert_eq!(*b.value(), 0.8f64.tanh());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/cosh^2(a)
        assert_eq!(db[&a], 1.0 / (0.8f64.cosh() * 0.8f64.cosh()));
      });
    }

    #[test]
    fn asinh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.5);
        let b = a.asinh();
        assert_eq!(*b.value(), 1.5f64.asinh());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/sqrt(1+a^2)
        assert_eq!(db[&a], 1.0 / f64::sqrt(1.0 + 2.25));
      });
    }

    #[test]
    fn acosh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let b = a.acosh();
        assert_eq!(*b.value(), 2.0f64.acosh());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/sqrt(a^2-1)
        assert_eq!(db[&a], 1.0 / f64::sqrt(4.0 - 1.0));
      });
    }

    #[test]
    fn atanh() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(0.0);
        let b = a.atanh();
        assert_eq!(*b.value(), 0.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(1-a^2)
        assert_eq!(db[&a], 1.0);
      });
    }

    #[test]
    fn exp() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.exp();
        assert_eq!(*b.value(), 1.3f64.exp());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = exp(a)
        assert_eq!(db[&a], 1.3f64.exp());
      });
    }

    #[test]
    fn exp2() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        let b = a.exp2();
        assert_eq!(*b.value(), 1.3f64.exp2());
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = exp2(a) * ln(2)
        assert_eq!(db[&a], 1.3f64.exp2() * 2.0f64.ln());
      });
    }

    #[test]
    fn sqrt() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(4.0);
        let b = a.sqrt();
        assert_eq!(*b.value(), 2.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(2*sqrt(a))
        assert_eq!(db[&a], 0.25);
      });
    }

    #[test]
    fn cbrt() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.0);
        let b = a.cbrt();
        assert_eq!(*b.value(), 1.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = 1/(3*cbrt(a^2))
        assert_eq!(db[&a], 1.0 / 3.0);
      });
    }

    #[test]
    fn abs() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(-3.0);
        let b = a.abs();
        assert_eq!(*b.value(), 3.0);
        let grads = guard.lock().collapse();
        let db = grads.of(&b, 1.0);
        // df/da = a/|a| = -1 for a < 0
        assert_eq!(db[&a], -1.0);
      });
    }
  }

  mod gradients {
    use super::*;

    #[test]
    fn of() {
      let mut tape: Tape<f64, Pullback> = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(5.0);
        let b = guard.var(2.0);
        let c = guard.var(1.0);
        let res = a.pow(&b).sub(&c.asinh().div_f64(2.0)).add_f64(1.0f64.sin());
        let expected = 5.0f64.powf(2.0) - 1.0f64.asinh() / 2.0 + 1.0f64.sin();
        assert_eq!(*res.value(), expected);
        let grads = guard.lock().collapse();
        let dres = grads.of(&res, 1.0);
        let ga = dres[&a]; // df/da
        let gb = dres[&b]; // df/db
        let gc = dres[&c]; // df/dc
        assert_eq!(ga, 2.0 * 5.0);
        assert_eq!(gb, 25.0 * 5.0f64.ln());
        assert_eq!(gc, -1.0 / (2.0 * 2.0f64.sqrt()));
      });
    }
  }
}
