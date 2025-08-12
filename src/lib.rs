//! # auto: Reverse-Mode Automatic Differentiation with Tensor Abstraction
//!
//! This version abstracts tensor operations behind a `TensorOps` trait, allowing
//! different backend implementations while maintaining a consistent API.

use std::cell::{RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Deref, DerefMut, Div, Index, Mul, Neg, Sub};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

/// Trait defining all tensor operations needed by the autodiff system.
/// Implementations can be provided for scalars (f64), vectors, matrices, or n-d tensors.
pub trait TensorOps: Clone + fmt::Debug + 'static {
  /// Create a zero tensor with the same shape as self
  fn zeros_like(&self) -> Self;

  /// Create a ones tensor with the same shape as self
  fn ones_like(&self) -> Self;

  /// Element-wise addition
  fn add(&self, other: &Self) -> Self;

  /// Add a scalar to all elements
  fn add_scalar(&self, scalar: f64) -> Self;

  /// Element-wise subtraction
  fn sub(&self, other: &Self) -> Self;

  /// Subtract a scalar from all elements
  fn sub_scalar(&self, scalar: f64) -> Self;

  /// Matrix multiplication (or dot product for vectors, multiplication for scalars)
  fn matmul(&self, other: &Self) -> Self;

  /// Element-wise multiplication (Hadamard product)
  fn hadamard(&self, other: &Self) -> Self;

  /// Multiply all elements by a scalar
  fn mul_scalar(&self, scalar: f64) -> Self;

  /// Element-wise division
  fn div(&self, other: &Self) -> Self;

  /// Divide all elements by a scalar
  fn div_scalar(&self, scalar: f64) -> Self;

  /// Element-wise reciprocal
  fn reciprocal(&self) -> Self;

  /// Element-wise negation
  fn neg(&self) -> Self;

  /// Element-wise sine
  fn sin(&self) -> Self;

  /// Element-wise cosine
  fn cos(&self) -> Self;

  /// Element-wise tangent
  fn tan(&self) -> Self;

  /// Element-wise natural logarithm
  fn ln(&self) -> Self;

  /// Element-wise logarithm with specified base
  fn log(&self, base: f64) -> Self;

  /// Element-wise exponential
  fn exp(&self) -> Self;

  /// Element-wise power
  fn pow(&self, exponent: &Self) -> Self;

  /// Element-wise power with scalar exponent
  fn powf(&self, exponent: f64) -> Self;

  /// Element-wise square root
  fn sqrt(&self) -> Self;

  /// Element-wise absolute value
  fn abs(&self) -> Self;

  /// Element-wise hyperbolic sine
  fn sinh(&self) -> Self;

  /// Element-wise hyperbolic cosine
  fn cosh(&self) -> Self;

  /// Element-wise hyperbolic tangent
  fn tanh(&self) -> Self;

  /// Element-wise arcsine
  fn asin(&self) -> Self;

  /// Element-wise arccosine
  fn acos(&self) -> Self;

  /// Element-wise arctangent
  fn atan(&self) -> Self;

  /// Element-wise inverse hyperbolic sine
  fn asinh(&self) -> Self;

  /// Element-wise inverse hyperbolic cosine
  fn acosh(&self) -> Self;

  /// Element-wise inverse hyperbolic tangent
  fn atanh(&self) -> Self;

  /// Transpose (for matrices) or identity (for scalars/vectors)
  fn transpose(&self) -> Self;
}

/// Implementation of TensorOps for f64 (scalar operations)
impl TensorOps for f64 {
  fn zeros_like(&self) -> Self {
    0.0
  }
  fn ones_like(&self) -> Self {
    1.0
  }
  fn add(&self, other: &Self) -> Self {
    self + other
  }
  fn add_scalar(&self, scalar: f64) -> Self {
    self + scalar
  }
  fn sub(&self, other: &Self) -> Self {
    self - other
  }
  fn sub_scalar(&self, scalar: f64) -> Self {
    self - scalar
  }
  fn matmul(&self, other: &Self) -> Self {
    self * other
  }
  fn hadamard(&self, other: &Self) -> Self {
    self * other
  }
  fn mul_scalar(&self, scalar: f64) -> Self {
    self * scalar
  }
  fn div(&self, other: &Self) -> Self {
    self / other
  }
  fn div_scalar(&self, scalar: f64) -> Self {
    self / scalar
  }
  fn reciprocal(&self) -> Self {
    1.0 / self
  }
  fn neg(&self) -> Self {
    -self
  }
  fn sin(&self) -> Self {
    f64::sin(*self)
  }
  fn cos(&self) -> Self {
    f64::cos(*self)
  }
  fn tan(&self) -> Self {
    f64::tan(*self)
  }
  fn ln(&self) -> Self {
    f64::ln(*self)
  }
  fn log(&self, base: f64) -> Self {
    f64::log(*self, base)
  }
  fn exp(&self) -> Self {
    f64::exp(*self)
  }
  fn pow(&self, exponent: &Self) -> Self {
    f64::powf(*self, *exponent)
  }
  fn powf(&self, exponent: f64) -> Self {
    f64::powf(*self, exponent)
  }
  fn sqrt(&self) -> Self {
    f64::sqrt(*self)
  }
  fn abs(&self) -> Self {
    f64::abs(*self)
  }
  fn sinh(&self) -> Self {
    f64::sinh(*self)
  }
  fn cosh(&self) -> Self {
    f64::cosh(*self)
  }
  fn tanh(&self) -> Self {
    f64::tanh(*self)
  }
  fn asin(&self) -> Self {
    f64::asin(*self)
  }
  fn acos(&self) -> Self {
    f64::acos(*self)
  }
  fn atan(&self) -> Self {
    f64::atan(*self)
  }
  fn asinh(&self) -> Self {
    f64::asinh(*self)
  }
  fn acosh(&self) -> Self {
    f64::acosh(*self)
  }
  fn atanh(&self) -> Self {
    f64::atanh(*self)
  }
  fn transpose(&self) -> Self {
    *self
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
struct NodeIndex(u64);

impl NodeIndex {
  #[inline(always)]
  fn new(level: u8, index: u64) -> Self {
    let level = level as u64;
    Self(index << 8 | level)
  }

  #[inline(always)]
  fn level(&self) -> u8 {
    (self.0 & 0xFF) as u8
  }

  #[inline(always)]
  fn index(&self) -> u64 {
    self.0 >> 8
  }
}

/// Type alias for gradient computation function
type GradFn<T> = Rc<dyn Fn(&T) -> T>;

/// A predecessor stores a gradient computation function
struct Predecessor<T> {
  grad_fn: GradFn<T>,
  index: NodeIndex,
}

impl<T> Clone for Predecessor<T> {
  fn clone(&self) -> Self {
    Self {
      grad_fn: Rc::clone(&self.grad_fn),
      index: self.index,
    }
  }
}

#[derive(Clone)]
struct Node<T> {
  pred_a: Predecessor<T>,
  pred_b: Predecessor<T>,
}

#[derive(Clone)]
pub struct Var<'scope, T> {
  value: T,
  tape: &'scope TapeInner<T>,
  index: NodeIndex,
}

impl<'scope, T: TensorOps> Var<'scope, T> {
  #[inline(always)]
  pub fn value(&self) -> &T {
    &self.value
  }

  /// Element-wise multiplication (Hadamard product)
  #[inline]
  pub fn hadamard(&self, other: &Self) -> Self {
    let v = self.value.clone();
    let ov = other.value.clone();
    let result = v.hadamard(&ov);

    // d/dx[x ⊙ y] = y (element-wise)
    let grad_fn_a = Rc::new(move |upstream: &T| upstream.hadamard(&ov));
    // d/dy[x ⊙ y] = x (element-wise)
    let grad_fn_b = Rc::new(move |upstream: &T| upstream.hadamard(&v));

    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }

  #[inline]
  pub fn reciprocal(&self) -> Self {
    let v = self.value.clone();
    let v_recip = v.reciprocal();
    let v_sq_recip = v_recip.hadamard(&v_recip);
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&v_sq_recip).neg());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.reciprocal(), pred_a, pred_b)
  }

  #[inline]
  pub fn sin(&self) -> Self {
    let v = self.value.clone();
    let cos_v = v.cos();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&cos_v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.sin(), pred_a, pred_b)
  }

  #[inline]
  pub fn cos(&self) -> Self {
    let v = self.value.clone();
    let neg_sin_v = v.sin().neg();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&neg_sin_v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.cos(), pred_a, pred_b)
  }

  #[inline]
  pub fn tan(&self) -> Self {
    let v = self.value.clone();
    let sec_sq = v.cos().reciprocal().powf(2.0);
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&sec_sq));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.tan(), pred_a, pred_b)
  }

  #[inline]
  pub fn ln(&self) -> Self {
    let v = self.value.clone();
    let v_recip = v.reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&v_recip));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.ln(), pred_a, pred_b)
  }

  #[inline]
  pub fn log(&self, base: f64) -> Self {
    let v = self.value.clone();
    let scale = v.reciprocal().div_scalar(base.ln());
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&scale));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.log(base), pred_a, pred_b)
  }

  #[inline]
  pub fn log10(&self) -> Self {
    self.log(10.0)
  }

  #[inline]
  pub fn log2(&self) -> Self {
    self.log(2.0)
  }

  #[inline]
  pub fn exp(&self) -> Self {
    let v = self.value.clone();
    let exp_v = v.exp();
    let exp_v_clone = exp_v.clone();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&exp_v_clone));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(exp_v, pred_a, pred_b)
  }

  #[inline]
  pub fn pow(&self, other: &Self) -> Self {
    let v = self.value.clone();
    let ov = other.value.clone();
    let result = v.pow(&ov);

    // d/dx[x^y] = y * x^(y-1)
    let v_clone = v.clone();
    let ov_clone = ov.clone();
    let grad_fn_a = Rc::new(move |upstream: &T| {
      let grad = ov_clone.hadamard(&v_clone.pow(&ov_clone.sub_scalar(1.0)));
      upstream.hadamard(&grad)
    });

    // d/dy[x^y] = x^y * ln(x)
    let result_clone = result.clone();
    let ln_v = v.ln();
    let grad_fn_b = Rc::new(move |upstream: &T| upstream.hadamard(&result_clone.hadamard(&ln_v)));

    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }

  #[inline]
  pub fn powf(&self, exponent: f64) -> Self {
    let v = self.value.clone();
    let result = v.powf(exponent);
    let grad = v.powf(exponent - 1.0).mul_scalar(exponent);
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(result, pred_a, pred_b)
  }

  #[inline]
  pub fn sqrt(&self) -> Self {
    self.powf(0.5)
  }

  #[inline]
  pub fn cbrt(&self) -> Self {
    self.powf(1.0 / 3.0)
  }

  #[inline]
  pub fn abs(&self) -> Self {
    let v = self.value.clone();
    let sign = v.div(&v.abs());
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&sign));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.abs(), pred_a, pred_b)
  }

  #[inline]
  pub fn sinh(&self) -> Self {
    let v = self.value.clone();
    let cosh_v = v.cosh();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&cosh_v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.sinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn cosh(&self) -> Self {
    let v = self.value.clone();
    let sinh_v = v.sinh();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&sinh_v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.cosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn tanh(&self) -> Self {
    let v = self.value.clone();
    let sech_sq = v.cosh().reciprocal().powf(2.0);
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&sech_sq));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.tanh(), pred_a, pred_b)
  }

  #[inline]
  pub fn asin(&self) -> Self {
    let v = self.value.clone();
    let grad = v.hadamard(&v).neg().add_scalar(1.0).sqrt().reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.asin(), pred_a, pred_b)
  }

  #[inline]
  pub fn acos(&self) -> Self {
    let v = self.value.clone();
    let grad = v
      .hadamard(&v)
      .neg()
      .add_scalar(1.0)
      .sqrt()
      .reciprocal()
      .neg();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.acos(), pred_a, pred_b)
  }

  #[inline]
  pub fn atan(&self) -> Self {
    let v = self.value.clone();
    let grad = v.hadamard(&v).add_scalar(1.0).reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.atan(), pred_a, pred_b)
  }

  #[inline]
  pub fn asinh(&self) -> Self {
    let v = self.value.clone();
    let grad = v.hadamard(&v).add_scalar(1.0).sqrt().reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.asinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn acosh(&self) -> Self {
    let v = self.value.clone();
    let grad = v.hadamard(&v).sub_scalar(1.0).sqrt().reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.acosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn atanh(&self) -> Self {
    let v = self.value.clone();
    let grad = v.hadamard(&v).neg().add_scalar(1.0).reciprocal();
    let grad_fn = Rc::new(move |upstream: &T| upstream.hadamard(&grad));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.atanh(), pred_a, pred_b)
  }

  #[inline(always)]
  fn produce(&self, value: T, pred_a: Predecessor<T>, pred_b: Predecessor<T>) -> Self {
    Var {
      value,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
    }
  }

  #[inline(always)]
  fn to_predecessor(&self, grad_fn: GradFn<T>) -> Predecessor<T> {
    Predecessor {
      grad_fn,
      index: self.index,
    }
  }

  #[inline(always)]
  fn to_predecessor_zero(&self) -> Predecessor<T> {
    Predecessor {
      grad_fn: Rc::new(|upstream: &T| upstream.zeros_like()),
      index: self.index,
    }
  }

  #[inline]
  fn add_node(&self, pred_a: Predecessor<T>, pred_b: Predecessor<T>) -> NodeIndex {
    self.tape.current_frame_mut().add_node(pred_a, pred_b)
  }
}

// Operator implementations - now Mul is matrix multiplication!

impl<'scope, T: TensorOps> Add for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn add(self, other: Self) -> Self::Output {
    let result = self.value.add(&other.value);
    let grad_fn_a = Rc::new(|upstream: &T| upstream.clone());
    let grad_fn_b = Rc::new(|upstream: &T| upstream.clone());
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }
}

impl<'scope, T: TensorOps> Add<f64> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn add(self, other: f64) -> Self::Output {
    let result = self.value.add_scalar(other);
    let grad_fn = Rc::new(|upstream: &T| upstream.clone());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(result, pred_a, pred_b)
  }
}

impl<'scope, T: TensorOps> Sub for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn sub(self, other: Self) -> Self::Output {
    let result = self.value.sub(&other.value);
    let grad_fn_a = Rc::new(|upstream: &T| upstream.clone());
    let grad_fn_b = Rc::new(|upstream: &T| upstream.neg());
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }
}

impl<'scope, T: TensorOps> Sub<f64> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn sub(self, other: f64) -> Self::Output {
    let result = self.value.sub_scalar(other);
    let grad_fn = Rc::new(|upstream: &T| upstream.clone());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(result, pred_a, pred_b)
  }
}

// Mul is now matrix multiplication!
impl<'scope, T: TensorOps> Mul for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    let v = self.value.clone();
    let ov = other.value.clone();
    let result = v.matmul(&ov);

    // For matrix multiplication: d/dA[A @ B] = upstream @ B^T
    let ov_t = ov.transpose();
    let grad_fn_a = Rc::new(move |upstream: &T| upstream.matmul(&ov_t));

    // d/dB[A @ B] = A^T @ upstream
    let v_t = v.transpose();
    let grad_fn_b = Rc::new(move |upstream: &T| v_t.matmul(upstream));

    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }
}

// Scalar multiplication
impl<'scope, T: TensorOps> Mul<f64> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn mul(self, other: f64) -> Self::Output {
    let result = self.value.mul_scalar(other);
    let grad_fn = Rc::new(move |upstream: &T| upstream.mul_scalar(other));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(result, pred_a, pred_b)
  }
}

impl<'scope, T: TensorOps> Div for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn div(self, other: Self) -> Self::Output {
    let result = self.value.div(&other.value);
    let ov_recip = other.value.reciprocal();
    let grad_fn_a = Rc::new(move |upstream: &T| upstream.hadamard(&ov_recip));

    let v = self.value.clone();
    let ov_sq = other.value.hadamard(&other.value);
    let grad_fn_b = Rc::new(move |upstream: &T| upstream.hadamard(&v).div(&ov_sq).neg());

    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }
}

impl<'scope, T: TensorOps> Div<f64> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn div(self, other: f64) -> Self::Output {
    self.mul(1.0 / other)
  }
}

impl<'scope, T: TensorOps> BitXor for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    self.pow(other)
  }
}

impl<'scope, T: TensorOps> BitXor<f64> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    self.powf(other)
  }
}

impl<'scope, T: TensorOps> Neg for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline]
  fn neg(self) -> Self::Output {
    let result = self.value.neg();
    let grad_fn = Rc::new(|upstream: &T| upstream.neg());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(result, pred_a, pred_b)
  }
}

// Forwarding implementations for owned types
impl<'scope, T: TensorOps> Add<Var<'scope, T>> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn add(self, other: Var<'scope, T>) -> Self::Output {
    self.add(&other)
  }
}

impl<'scope, T: TensorOps> Add for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn add(self, other: Self) -> Self::Output {
    (&self).add(&other)
  }
}

impl<'scope, T: TensorOps> Add<f64> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn add(self, other: f64) -> Self::Output {
    (&self).add(other)
  }
}

impl<'scope, T: TensorOps> Add<&Var<'scope, T>> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn add(self, other: &Var<'scope, T>) -> Self::Output {
    (&self).add(other)
  }
}

impl<'scope, T: TensorOps> Sub<Var<'scope, T>> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn sub(self, other: Var<'scope, T>) -> Self::Output {
    self.sub(&other)
  }
}

impl<'scope, T: TensorOps> Sub for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn sub(self, other: Self) -> Self::Output {
    (&self).sub(&other)
  }
}

impl<'scope, T: TensorOps> Sub<f64> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn sub(self, other: f64) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'scope, T: TensorOps> Sub<&Var<'scope, T>> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn sub(self, other: &Var<'scope, T>) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'scope, T: TensorOps> Mul<Var<'scope, T>> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn mul(self, other: Var<'scope, T>) -> Self::Output {
    self.mul(&other)
  }
}

impl<'scope, T: TensorOps> Mul for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn mul(self, other: Self) -> Self::Output {
    (&self).mul(&other)
  }
}

impl<'scope, T: TensorOps> Mul<f64> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn mul(self, other: f64) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'scope, T: TensorOps> Mul<&Var<'scope, T>> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn mul(self, other: &Var<'scope, T>) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'scope, T: TensorOps> Div<Var<'scope, T>> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn div(self, other: Var<'scope, T>) -> Self::Output {
    self.div(&other)
  }
}

impl<'scope, T: TensorOps> Div for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    (&self).div(&other)
  }
}

impl<'scope, T: TensorOps> Div<f64> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    (&self).div(other)
  }
}

impl<'scope, T: TensorOps> Div<&Var<'scope, T>> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn div(self, other: &Var<'scope, T>) -> Self::Output {
    (&self).div(other)
  }
}

impl<'scope, T: TensorOps> BitXor<Var<'scope, T>> for &Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: Var<'scope, T>) -> Self::Output {
    self.bitxor(&other)
  }
}

impl<'scope, T: TensorOps> BitXor for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    (&self).bitxor(&other)
  }
}

impl<'scope, T: TensorOps> BitXor<f64> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'scope, T: TensorOps> BitXor<&Var<'scope, T>> for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn bitxor(self, other: &Var<'scope, T>) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'scope, T: TensorOps> Neg for Var<'scope, T> {
  type Output = Var<'scope, T>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    (&self).neg()
  }
}

impl<T: TensorOps> Deref for Var<'_, T> {
  type Target = T;

  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    &self.value
  }
}

impl<T: TensorOps> DerefMut for Var<'_, T> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.value
  }
}

impl<T: TensorOps> fmt::Debug for Var<'_, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Var")
      .field("value", &self.value)
      .field("level", &self.index.level())
      .field("index", &self.index.index())
      .finish()
  }
}

#[derive(Default, Clone)]
struct Frame<T> {
  level: u8,
  nodes: Vec<Node<T>>,
}

impl<T> Frame<T> {
  fn new(level: u8) -> Self {
    Self {
      level,
      nodes: Vec::new(),
    }
  }

  #[inline(always)]
  fn add_node(&mut self, pred_a: Predecessor<T>, pred_b: Predecessor<T>) -> NodeIndex {
    let node = self.nodes.len();
    self.nodes.push(Node { pred_a, pred_b });
    NodeIndex::new(self.level, node as u64)
  }
}

#[derive(Default)]
struct Frames<T> {
  stack: Vec<Frame<T>>,
}

impl<T> Frames<T> {
  #[inline]
  fn get_node(&self, index: NodeIndex) -> Option<&Node<T>> {
    let level = index.level() as usize;
    let idx = index.index() as usize;
    self.stack.get(level).and_then(|frame| frame.nodes.get(idx))
  }
}

struct FrameGuard<'tape, T> {
  tape: &'tape TapeInner<T>,
}

impl<'tape, T> FrameGuard<'tape, T> {
  fn new(tape: &'tape TapeInner<T>, frame: Frame<T>) -> Self {
    let guard = Self { tape };
    guard.tape.frames.borrow_mut().stack.push(frame);
    guard
  }
}

impl<T> Drop for FrameGuard<'_, T> {
  fn drop(&mut self) {
    self.tape.frames.borrow_mut().stack.pop();
  }
}

#[derive(Default)]
struct TapeInner<T> {
  frames: RefCell<Frames<T>>,
}

impl<T: TensorOps> TapeInner<T> {
  #[inline]
  pub fn current_frame_mut(&self) -> RefMut<Frame<T>> {
    RefMut::map(self.frames.borrow_mut(), |frames| {
      frames
        .stack
        .last_mut()
        .expect("there should always be a current frame")
    })
  }

  fn with_scope<F, R>(&self, level: u8, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, T, Unlocked>) -> R,
  {
    let _tape_frame = FrameGuard::new(self, Frame::new(level));
    f(Guard {
      level,
      tape: self as *const TapeInner<T>,
      phantom: PhantomData,
    })
  }
}

#[derive(Default)]
pub struct Tape<T> {
  inner: TapeInner<T>,
}

impl<T: TensorOps> Tape<T> {
  pub fn new() -> Self
  where
    T: Default,
  {
    Self::default()
  }

  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, T, Unlocked>) -> R,
  {
    self.inner.with_scope(0, f)
  }
}

pub struct Locked;
pub struct Unlocked;

pub struct Guard<'scope, T: TensorOps = f64, S = Unlocked> {
  level: u8,
  tape: *const TapeInner<T>,
  phantom: PhantomData<&'scope S>,
}

impl<'scope, T: TensorOps> Guard<'scope, T, Unlocked> {
  #[inline]
  pub fn var(&self, value: T) -> Var<'scope, T> {
    let tape = unsafe { &*self.tape };
    let mut current_frame = tape.current_frame_mut();

    let index = NodeIndex::new(self.level, current_frame.nodes.len() as u64);
    let identity = Rc::new(|upstream: &T| upstream.zeros_like());
    let index = current_frame.add_node(
      Predecessor {
        index,
        grad_fn: identity.clone(),
      },
      Predecessor {
        index,
        grad_fn: identity,
      },
    );

    Var { value, index, tape }
  }

  #[inline]
  pub fn lock(self) -> Guard<'scope, T, Locked> {
    Guard {
      level: self.level,
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'scope, T: TensorOps> Guard<'scope, T, Locked> {
  #[inline]
  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, T, Unlocked>) -> R,
  {
    let tape: &TapeInner<T> = unsafe { &*self.tape };
    assert!(self.level < u8::MAX);
    tape.with_scope(self.level + 1, f)
  }

  #[inline]
  pub fn collapse(self) -> Gradients<'scope, T> {
    Gradients {
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

trait HashMapExt {
  fn with_capacity(x: usize) -> Self;
}

impl<K, V, S> HashMapExt for HashMap<K, V, S>
where
  K: Hash + Eq,
  S: BuildHasher + Default,
{
  fn with_capacity(capacity: usize) -> Self {
    HashMap::with_capacity_and_hasher(capacity, S::default())
  }
}

trait HashSetExt {
  fn with_capacity(x: usize) -> Self;
}

impl<K, S> HashSetExt for HashSet<K, S>
where
  K: Hash + Eq,
  S: BuildHasher + Default,
{
  fn with_capacity(capacity: usize) -> Self {
    HashSet::with_capacity_and_hasher(capacity, S::default())
  }
}

type IndexNode<T> = (NodeIndex, Node<T>);

pub struct Gradients<'scope, T: TensorOps = f64> {
  tape: *const TapeInner<T>,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope, T: TensorOps> Gradients<'scope, T> {
  pub fn of(&self, var: &Var<'scope, T>) -> Deltas<'scope, T> {
    let tape = unsafe { &*self.tape };
    let subgraph = topological_subgraph(tape, var);

    let mut deltas = FxHashMap::with_capacity(subgraph.len());
    deltas.insert(var.index, var.value.ones_like());

    for (index, node) in subgraph.into_iter().rev() {
      let Node { pred_a, pred_b, .. } = node;

      let upstream = deltas.get(&index).unwrap();

      let grad_a = (pred_a.grad_fn)(upstream);
      let grad_b = (pred_b.grad_fn)(upstream);

      deltas
        .entry(pred_a.index)
        .and_modify(|e| *e = e.add(&grad_a))
        .or_insert(grad_a);

      deltas
        .entry(pred_b.index)
        .and_modify(|e| *e = e.add(&grad_b))
        .or_insert(grad_b);
    }

    Deltas {
      deltas,
      phantom: PhantomData,
    }
  }
}

fn topological_subgraph<T>(tape: &TapeInner<T>, var: &Var<'_, T>) -> Vec<IndexNode<T>>
where
  T: TensorOps + Clone,
{
  let nodes = tape.frames.borrow();

  let mut stack = Vec::with_capacity(512);
  let mut result = Vec::with_capacity(512);
  let mut visited = FxHashSet::with_capacity(512);

  stack.push((var.index, false));

  while let Some((node_index, children_processed)) = stack.pop() {
    if children_processed {
      if let Some(node) = nodes.get_node(node_index) {
        result.push((node_index, node.clone()));
      }
    } else if visited.insert(node_index) {
      stack.push((node_index, true));
      if let Some(node) = nodes.get_node(node_index) {
        if !visited.contains(&node.pred_b.index) {
          stack.push((node.pred_b.index, false));
        }
        if !visited.contains(&node.pred_a.index) {
          stack.push((node.pred_a.index, false));
        }
      }
    }
  }

  result
}

#[derive(Debug)]
pub struct Deltas<'scope, T: TensorOps = f64> {
  deltas: FxHashMap<NodeIndex, T>,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope, T: TensorOps> Index<&Var<'scope, T>> for Deltas<'scope, T> {
  type Output = T;

  fn index(&self, var: &Var<'scope, T>) -> &Self::Output {
    static ZERO_F64: f64 = 0.0;
    self.deltas.get(&var.index).unwrap_or(
      if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // Safety: We just checked that T is f64
        unsafe { &*(&ZERO_F64 as *const f64 as *const T) }
      } else {
        panic!("No gradient found for variable")
      },
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  mod scalar_ops {
    use super::*;

    #[test]
    fn test_add() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let b = guard.var(4.0);
        let c = &a + &b;
        assert_eq!(*c.value(), 7.0);

        let grads = guard.lock().collapse().of(&c);
        assert_eq!(grads[&a], 1.0);
        assert_eq!(grads[&b], 1.0);
      });
    }

    #[test]
    fn test_matmul_scalar() {
      // For scalars, matmul is just multiplication
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let b = guard.var(4.0);
        let c = &a * &b; // This is now matmul
        assert_eq!(*c.value(), 12.0);

        let grads = guard.lock().collapse().of(&c);
        assert_eq!(grads[&a], 4.0);
        assert_eq!(grads[&b], 3.0);
      });
    }

    #[test]
    fn test_hadamard() {
      // For scalars, hadamard is also multiplication
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(3.0);
        let b = guard.var(4.0);
        let c = a.hadamard(&b);
        assert_eq!(*c.value(), 12.0);

        let grads = guard.lock().collapse().of(&c);
        assert_eq!(grads[&a], 4.0);
        assert_eq!(grads[&b], 3.0);
      });
    }

    #[test]
    fn test_complex_expression() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let x = guard.var(2.0);
        let y = guard.var(3.0);

        // z = x^2 * y + sin(x)
        let z = x.powf(2.0).hadamard(&y) + x.sin();

        let expected = 4.0 * 3.0 + 2.0_f64.sin();
        assert!((z.value() - expected).abs() < 1e-10);

        let grads = guard.lock().collapse().of(&z);
        // dz/dx = 2xy + cos(x) = 2*2*3 + cos(2)
        let expected_dx = 12.0 + 2.0_f64.cos();
        assert!((grads[&x] - expected_dx).abs() < 1e-10);

        // dz/dy = x^2 = 4
        assert!((grads[&y] - 4.0).abs() < 1e-10);
      });
    }
  }
}
