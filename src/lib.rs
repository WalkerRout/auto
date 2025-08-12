//! # auto: Reverse-Mode Automatic Differentiation with Gradient Functions
//!
//! This refactored version stores gradient computation functions instead of 
//! precomputed gradient values, allowing for operations that need access to
//! upstream gradients during backpropagation.

use std::cell::{RefCell, RefMut};
use std::collections::{HashSet, HashMap};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Deref, DerefMut, Div, Index, Mul, Neg, Sub};
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

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
/// Takes the upstream gradient and returns the contribution to this node's gradient
type GradFn = Rc<dyn Fn(f64) -> f64>;

/// A predecessor now stores a gradient computation function instead of a fixed gradient
struct Predecessor {
  grad_fn: GradFn,
  index: NodeIndex,
}

impl Clone for Predecessor {
  fn clone(&self) -> Self {
    Self {
      grad_fn: Rc::clone(&self.grad_fn),
      index: self.index,
    }
  }
}

#[derive(Clone)]
struct Node {
  pred_a: Predecessor,
  pred_b: Predecessor,
  value: f64,  // Store the forward pass value in the node
}

#[derive(Clone)]
pub struct Var<'scope> {
  value: f64,
  tape: &'scope TapeInner,
  index: NodeIndex,
}

impl Var<'_> {
  #[inline(always)]
  pub fn value(&self) -> f64 {
    self.value
  }

  #[inline]
  pub fn reciprocal(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| -upstream / (v * v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(1.0 / v, pred_a, pred_b)
  }

  #[inline]
  pub fn sin(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream * v.cos());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.sin(), pred_a, pred_b)
  }

  #[inline]
  pub fn cos(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| -upstream * v.sin());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.cos(), pred_a, pred_b)
  }

  #[inline]
  pub fn tan(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (v.cos() * v.cos()));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.tan(), pred_a, pred_b)
  }

  #[inline]
  pub fn ln(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / v);
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.ln(), pred_a, pred_b)
  }

  #[inline]
  pub fn log(&self, base: f64) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (v * base.ln()));
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
  pub fn asin(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (1.0 - v * v).sqrt());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.asin(), pred_a, pred_b)
  }

  #[inline]
  pub fn acos(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| -upstream / (1.0 - v * v).sqrt());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.acos(), pred_a, pred_b)
  }

  #[inline]
  pub fn atan(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (1.0 + v * v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.atan(), pred_a, pred_b)
  }

  #[inline]
  pub fn sinh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream * v.cosh());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.sinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn cosh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream * v.sinh());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.cosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn tanh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (v.cosh() * v.cosh()));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.tanh(), pred_a, pred_b)
  }

  #[inline]
  pub fn asinh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (1.0 + v * v).sqrt());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.asinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn acosh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (v * v - 1.0).sqrt());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.acosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn atanh(&self) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream / (1.0 - v * v));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.atanh(), pred_a, pred_b)
  }

  #[inline]
  pub fn exp(&self) -> Self {
    let v = self.value;
    let exp_v = v.exp();
    let grad_fn = Rc::new(move |upstream: f64| upstream * exp_v);
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(exp_v, pred_a, pred_b)
  }

  #[inline]
  pub fn exp2(&self) -> Self {
    let v = self.value;
    let exp2_v = v.exp2();
    let grad_fn = Rc::new(move |upstream: f64| upstream * exp2_v * 2.0_f64.ln());
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(exp2_v, pred_a, pred_b)
  }

  #[inline]
  pub fn pow(&self, other: &Self) -> Self {
    let v = self.value;
    let ov = other.value;
    let result = v.powf(ov);
    
    // Gradient w.r.t. base: d/dx[x^y] = y * x^(y-1)
    let grad_fn_a = Rc::new(move |upstream: f64| upstream * ov * v.powf(ov - 1.0));
    
    // Gradient w.r.t. exponent: d/dy[x^y] = x^y * ln(x)
    let grad_fn_b = Rc::new(move |upstream: f64| upstream * result * v.ln());
    
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(result, pred_a, pred_b)
  }

  #[inline]
  pub fn powf(&self, other: f64) -> Self {
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream * other * v.powf(other - 1.0));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.powf(other), pred_a, pred_b)
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
    let v = self.value;
    let grad_fn = Rc::new(move |upstream: f64| upstream * (v / v.abs()));
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(v.abs(), pred_a, pred_b)
  }

  #[inline(always)]
  fn produce(&self, value: f64, pred_a: Predecessor, pred_b: Predecessor) -> Self {
    Var {
      value,
      index: self.add_node(value, pred_a, pred_b),
      tape: self.tape,
    }
  }

  #[inline(always)]
  fn to_predecessor(&self, grad_fn: GradFn) -> Predecessor {
    Predecessor {
      grad_fn,
      index: self.index,
    }
  }

  #[inline(always)]
  fn to_predecessor_zero(&self) -> Predecessor {
    Predecessor {
      grad_fn: Rc::new(|_| 0.0),
      index: self.index,
    }
  }

  #[inline]
  fn add_node(&self, value: f64, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    self.tape.current_frame_mut().add_node(value, pred_a, pred_b)
  }
}

// Binary operations
impl<'scope> Add for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn add(self, other: Self) -> Self::Output {
    // Addition: both gradients pass through unchanged
    let grad_fn_a = Rc::new(|upstream: f64| upstream);
    let grad_fn_b = Rc::new(|upstream: f64| upstream);
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(self.value + other.value, pred_a, pred_b)
  }
}

impl<'scope> Add<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn add(self, other: f64) -> Self::Output {
    let grad_fn = Rc::new(|upstream: f64| upstream);
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(self.value + other, pred_a, pred_b)
  }
}

impl<'scope> Sub for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn sub(self, other: Self) -> Self::Output {
    let grad_fn_a = Rc::new(|upstream: f64| upstream);
    let grad_fn_b = Rc::new(|upstream: f64| -upstream);
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(self.value - other.value, pred_a, pred_b)
  }
}

impl<'scope> Sub<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn sub(self, other: f64) -> Self::Output {
    let grad_fn = Rc::new(|upstream: f64| upstream);
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(self.value - other, pred_a, pred_b)
  }
}

impl<'scope> Mul for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    let v = self.value;
    let ov = other.value;
    // d/dx[x*y] = y, d/dy[x*y] = x
    let grad_fn_a = Rc::new(move |upstream: f64| upstream * ov);
    let grad_fn_b = Rc::new(move |upstream: f64| upstream * v);
    let pred_a = self.to_predecessor(grad_fn_a);
    let pred_b = other.to_predecessor(grad_fn_b);
    self.produce(v * ov, pred_a, pred_b)
  }
}

impl<'scope> Mul<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn mul(self, other: f64) -> Self::Output {
    let grad_fn = Rc::new(move |upstream: f64| upstream * other);
    let pred_a = self.to_predecessor(grad_fn);
    let pred_b = self.to_predecessor_zero();
    self.produce(self.value * other, pred_a, pred_b)
  }
}

impl<'scope> Div for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    self.mul(&other.reciprocal())
  }
}

impl<'scope> Div<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    self.mul(1.0 / other)
  }
}

impl<'scope> BitXor for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    self.pow(other)
  }
}

impl<'scope> BitXor<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    self.powf(other)
  }
}

impl<'scope> Neg for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    self * -1.0
  }
}

// Forwarding implementations for owned types
impl<'scope> Add<Var<'scope>> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn add(self, other: Var<'scope>) -> Self::Output {
    self.add(&other)
  }
}

impl<'scope> Add for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn add(self, other: Self) -> Self::Output {
    (&self).add(&other)
  }
}

impl<'scope> Add<f64> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn add(self, other: f64) -> Self::Output {
    (&self).add(other)
  }
}

impl<'scope> Add<&Var<'scope>> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn add(self, other: &Var<'scope>) -> Self::Output {
    (&self).add(other)
  }
}

impl<'scope> Sub<Var<'scope>> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn sub(self, other: Var<'scope>) -> Self::Output {
    self.sub(&other)
  }
}

impl<'scope> Sub for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn sub(self, other: Self) -> Self::Output {
    (&self).sub(&other)
  }
}

impl<'scope> Sub<f64> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn sub(self, other: f64) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'scope> Sub<&Var<'scope>> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn sub(self, other: &Var<'scope>) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'scope> Mul<Var<'scope>> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn mul(self, other: Var<'scope>) -> Self::Output {
    self.mul(&other)
  }
}

impl<'scope> Mul for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn mul(self, other: Self) -> Self::Output {
    (&self).mul(&other)
  }
}

impl<'scope> Mul<f64> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn mul(self, other: f64) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'scope> Mul<&Var<'scope>> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn mul(self, other: &Var<'scope>) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'scope> Div<Var<'scope>> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: Var<'scope>) -> Self::Output {
    self.div(&other)
  }
}

impl<'scope> Div for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    (&self).div(&other)
  }
}

impl<'scope> Div<f64> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    (&self).div(other)
  }
}

impl<'scope> Div<&Var<'scope>> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn div(self, other: &Var<'scope>) -> Self::Output {
    (&self).div(other)
  }
}

impl<'scope> BitXor<Var<'scope>> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: Var<'scope>) -> Self::Output {
    self.bitxor(&other)
  }
}

impl<'scope> BitXor for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    (&self).bitxor(&other)
  }
}

impl<'scope> BitXor<f64> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'scope> BitXor<&Var<'scope>> for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn bitxor(self, other: &Var<'scope>) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'scope> Neg for Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    (&self).neg()
  }
}

impl Deref for Var<'_> {
  type Target = f64;

  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    &self.value
  }
}

impl DerefMut for Var<'_> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.value
  }
}

impl fmt::Debug for Var<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Var")
      .field("value", &self.value())
      .field("level", &self.index.level())
      .field("index", &self.index.index())
      .finish()
  }
}

#[derive(Default, Clone)]
struct Frame {
  level: u8,
  nodes: Vec<Node>,
}

impl Frame {
  fn new(level: u8) -> Self {
    Self {
      level,
      nodes: Vec::new(),
    }
  }

  #[inline(always)]
  fn add_node(&mut self, value: f64, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    let node = self.nodes.len();
    self.nodes.push(Node { pred_a, pred_b, value });
    NodeIndex::new(self.level, node as u64)
  }
}

#[derive(Default)]
struct Frames {
  stack: Vec<Frame>,
}

impl Frames {
  #[inline]
  fn get_node(&self, index: NodeIndex) -> Option<Node> {
    let level = index.level() as usize;
    let idx = index.index() as usize;
    self
      .stack
      .get(level)
      .and_then(|frame| frame.nodes.get(idx))
      .cloned()
  }
}

struct FrameGuard<'tape> {
  tape: &'tape TapeInner,
}

impl<'tape> FrameGuard<'tape> {
  fn new(tape: &'tape TapeInner, frame: Frame) -> Self {
    let guard = Self { tape };
    guard.tape.frames.borrow_mut().stack.push(frame);
    guard
  }
}

impl Drop for FrameGuard<'_> {
  fn drop(&mut self) {
    self.tape.frames.borrow_mut().stack.pop();
  }
}

#[derive(Default)]
struct TapeInner {
  frames: RefCell<Frames>,
}

impl TapeInner {
  #[inline]
  pub fn current_frame_mut(&self) -> RefMut<Frame> {
    RefMut::map(self.frames.borrow_mut(), |frames| {
      frames
        .stack
        .last_mut()
        .expect("there should always be a current frame")
    })
  }

  fn with_scope<F, R>(&self, level: u8, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, Unlocked>) -> R,
  {
    let _tape_frame = FrameGuard::new(self, Frame::new(level));
    f(Guard {
      level,
      tape: self as *const TapeInner,
      phantom: PhantomData,
    })
  }
}

#[derive(Default)]
pub struct Tape {
  inner: TapeInner,
}

impl Tape {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, Unlocked>) -> R,
  {
    self.inner.with_scope(0, f)
  }
}

pub struct Locked;
pub struct Unlocked;

pub struct Guard<'scope, S = Unlocked> {
  level: u8,
  tape: *const TapeInner,
  phantom: PhantomData<&'scope S>,
}

impl<'scope> Guard<'scope, Unlocked> {
  #[inline]
  pub fn var(&self, value: f64) -> Var<'scope> {
    let tape = unsafe { &*self.tape };
    let mut current_frame = tape.current_frame_mut();
    
    // For input variables, create self-referential node with identity gradient
    let index = NodeIndex::new(self.level, current_frame.nodes.len() as u64);
    let identity = Rc::new(|_: f64| 0.0);
    let index = current_frame.add_node(
      value,
      Predecessor { index, grad_fn: identity.clone() },
      Predecessor { index, grad_fn: identity },
    );
    
    Var {
      value,
      index,
      tape,
    }
  }

  #[inline]
  pub fn lock(self) -> Guard<'scope, Locked> {
    Guard {
      level: self.level,
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'scope> Guard<'scope, Locked> {
  #[inline]
  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, Unlocked>) -> R,
  {
    let tape: &TapeInner = unsafe { &*self.tape };
    assert!(self.level < u8::MAX);
    tape.with_scope(self.level + 1, f)
  }

  #[inline]
  pub fn collapse(self) -> Gradients<'scope> {
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

type IndexNode = (NodeIndex, Node);

pub struct Gradients<'scope> {
  tape: *const TapeInner,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope> Gradients<'scope> {
  /// Now gradient computation happens here with the gradient functions
  pub fn of(&self, var: &Var<'scope>) -> Deltas<'scope> {
    let tape = unsafe { &*self.tape };
    let subgraph = topological_subgraph(tape, var);

    let mut deltas = FxHashMap::with_capacity(subgraph.len());
    deltas.insert(var.index, 1.0);

    // Process nodes in reverse topological order
    for (index, node) in subgraph.into_iter().rev() {
      let Node { pred_a, pred_b, .. } = node;
      
      // Get the upstream gradient for this node
      let upstream = deltas.get(&index).copied().unwrap_or(0.0);
      
      // Apply gradient functions to compute contributions to predecessors
      let grad_a = (pred_a.grad_fn)(upstream);
      let grad_b = (pred_b.grad_fn)(upstream);
      
      *deltas.entry(pred_a.index).or_insert(0.0) += grad_a;
      *deltas.entry(pred_b.index).or_insert(0.0) += grad_b;
    }
    
    Deltas {
      deltas,
      phantom: PhantomData,
    }
  }
}

fn topological_subgraph(tape: &TapeInner, var: &Var<'_>) -> Vec<IndexNode> {
  let nodes = tape.frames.borrow();

  let mut stack = Vec::with_capacity(512);
  let mut result = Vec::with_capacity(512);
  let mut visited = FxHashSet::with_capacity(512);

  stack.push((var.index, false));

  while let Some((node_index, children_processed)) = stack.pop() {
    if children_processed {
      if let Some(node) = nodes.get_node(node_index) {
        result.push((node_index, node));
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
pub struct Deltas<'scope> {
  deltas: FxHashMap<NodeIndex, f64>,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope> Index<&Var<'scope>> for Deltas<'scope> {
  type Output = f64;

  fn index(&self, var: &Var<'scope>) -> &Self::Output {
    self.deltas.get(&var.index).unwrap_or(&0.0)
  }
}