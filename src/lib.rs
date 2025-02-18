use std::cell::{RefCell, RefMut};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Deref, DerefMut, Div, Index, Mul, Neg, Sub};

use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
struct NodeIndex(u64);

impl NodeIndex {
  const LEVEL_MASK: u64 = (1 << 48) - 1;

  #[inline(always)]
  fn new(level: u8, index: u64) -> Self {
    let level = level as u64;
    Self((level << 48) | (index & Self::LEVEL_MASK))
  }

  #[inline(always)]
  fn level(&self) -> u8 {
    (self.0 >> 48) as u8
  }

  #[inline(always)]
  fn index(&self) -> u64 {
    self.0 & Self::LEVEL_MASK
  }
}

#[derive(Clone)]
struct Predecessor {
  grad: f64,
  index: NodeIndex,
}

#[derive(Clone)]
struct Node {
  pred_a: Predecessor,
  pred_b: Predecessor,
}

#[derive(Clone)]
pub struct Var<'tape, 'scope> {
  value: f64,
  tape: &'tape Tape,
  index: NodeIndex,
  phantom: PhantomData<&'scope ()>,
}

impl<'tape, 'scope> Var<'tape, 'scope> {
  #[inline(always)]
  pub fn value(&self) -> f64 {
    self.value
  }

  #[inline]
  pub fn reciprocal(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(-1.0 / (v * v));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: 1.0 / v,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn sin(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cos());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.sin(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn cos(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(-v.sin());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.cos(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn tan(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cos() * v.cos()));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.tan(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn ln(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / v);
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.ln(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn log(&self, base: f64) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * base.ln()));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.log(base),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn log10(&self) -> Var<'tape, 'scope> {
    self.log(10.0)
  }

  #[inline]
  pub fn log2(&self) -> Var<'tape, 'scope> {
    self.log(2.0)
  }

  #[inline]
  pub fn asin(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.asin(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn acos(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(-1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.acos(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn atan(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.atan(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn sinh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cosh());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.sinh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn cosh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v.sinh());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.cosh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn tanh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cosh() * v.cosh()));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.tanh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn asinh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.asinh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn acosh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * v - 1.0).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.acosh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn atanh(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.atanh(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn exp(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.exp(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn exp2(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp2() * 2.0_f64.ln());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.exp2(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  /// We explicitly declare a function named `pow` on Var since `bitxor` is
  /// not as descriptive as `add` and `sub`...
  #[inline]
  pub fn pow(&self, other: &Var<'tape, 'scope>) -> Var<'tape, 'scope> {
    let v = self.value;
    let ov = other.value;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = other.to_predecessor(v.powf(ov) * v.ln());
    Var {
      value: v.powf(ov),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  /// Same as sister above, but takes a float exponent
  #[inline]
  pub fn powf(&self, other: f64) -> Var<'tape, 'scope> {
    let v = self.value;
    let ov = other;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.powf(ov),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn sqrt(&self) -> Var<'tape, 'scope> {
    self.powf(0.5)
  }

  #[inline]
  pub fn cbrt(&self) -> Var<'tape, 'scope> {
    self.powf(1.0 / 3.0)
  }

  #[inline]
  pub fn abs(&self) -> Var<'tape, 'scope> {
    let v = self.value;
    let pred_a = self.to_predecessor(v / v.abs());
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: v.abs(),
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  fn to_predecessor(&self, grad: f64) -> Predecessor {
    Predecessor {
      grad,
      index: self.index,
    }
  }

  #[inline]
  fn add_node(&self, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    self.tape.inner.current_frame_mut().add_node(pred_a, pred_b)
  }
}

// we implement 2 variants for each binary operation; &a <op> &b and &a <op> f64
// - owned implementations defer to reference implementations

impl<'tape, 'scope> Add for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn add(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(1.0);
    Var {
      value: self.value + other.value,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Add<f64> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn add(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: self.value + other,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Add<Var<'tape, 'scope>> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn add(self, other: Var<'tape, 'scope>) -> Self::Output {
    self.add(&other)
  }
}

impl<'tape, 'scope> Add for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn add(self, other: Self) -> Self::Output {
    (&self).add(&other)
  }
}

impl<'tape, 'scope> Add<f64> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn add(self, other: f64) -> Self::Output {
    (&self).add(other)
  }
}

impl<'tape, 'scope> Add<&Var<'tape, 'scope>> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn add(self, other: &Var<'tape, 'scope>) -> Self::Output {
    (&self).add(other)
  }
}

impl<'tape, 'scope> Sub for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn sub(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(-1.0);
    Var {
      value: self.value - other.value,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Sub<f64> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn sub(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: self.value - other,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Sub<Var<'tape, 'scope>> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn sub(self, other: Var<'tape, 'scope>) -> Self::Output {
    self.sub(&other)
  }
}

impl<'tape, 'scope> Sub for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn sub(self, other: Self) -> Self::Output {
    (&self).sub(&other)
  }
}

impl<'tape, 'scope> Sub<f64> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn sub(self, other: f64) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'tape, 'scope> Sub<&Var<'tape, 'scope>> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn sub(self, other: &Var<'tape, 'scope>) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'tape, 'scope> Mul for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(other.value);
    let pred_b = other.to_predecessor(self.value);
    Var {
      value: self.value * other.value,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Mul<f64> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline]
  fn mul(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(other);
    let pred_b = self.to_predecessor(0.0);
    Var {
      value: self.value * other,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Mul<Var<'tape, 'scope>> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn mul(self, other: Var<'tape, 'scope>) -> Self::Output {
    self.mul(&other)
  }
}

impl<'tape, 'scope> Mul for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn mul(self, other: Self) -> Self::Output {
    (&self).mul(&other)
  }
}

impl<'tape, 'scope> Mul<f64> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn mul(self, other: f64) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'tape, 'scope> Mul<&Var<'tape, 'scope>> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn mul(self, other: &Var<'tape, 'scope>) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'tape, 'scope> Div for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    self.mul(&other.reciprocal())
  }
}

impl<'tape, 'scope> Div<f64> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    // same thing...
    self.mul(1.0 / other)
  }
}

impl<'tape, 'scope> Div<Var<'tape, 'scope>> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: Var<'tape, 'scope>) -> Self::Output {
    self.div(&other)
  }
}

impl<'tape, 'scope> Div for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    (&self).div(&other)
  }
}

impl<'tape, 'scope> Div<f64> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    (&self).div(other)
  }
}

impl<'tape, 'scope> Div<&Var<'tape, 'scope>> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn div(self, other: &Var<'tape, 'scope>) -> Self::Output {
    (&self).div(other)
  }
}

impl<'tape, 'scope> BitXor for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    self.pow(other)
  }
}

impl<'tape, 'scope> BitXor<f64> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    self.powf(other)
  }
}

impl<'tape, 'scope> BitXor<Var<'tape, 'scope>> for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: Var<'tape, 'scope>) -> Self::Output {
    self.bitxor(&other)
  }
}

impl<'tape, 'scope> BitXor for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    (&self).bitxor(&other)
  }
}

impl<'tape, 'scope> BitXor<f64> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'tape, 'scope> BitXor<&Var<'tape, 'scope>> for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn bitxor(self, other: &Var<'tape, 'scope>) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'tape, 'scope> Neg for &Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    self * -1.0
  }
}

impl<'tape, 'scope> Neg for Var<'tape, 'scope> {
  type Output = Var<'tape, 'scope>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    (&self).neg()
  }
}

impl Deref for Var<'_, '_> {
  type Target = f64;

  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    &self.value
  }
}

impl DerefMut for Var<'_, '_> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.value
  }
}

impl fmt::Debug for Var<'_, '_> {
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
  fn add_node(&mut self, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    let node = self.nodes.len();
    self.nodes.push(Node { pred_a, pred_b });
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
}

#[derive(Default)]
pub struct Tape {
  inner: TapeInner,
}

impl Tape {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn scope<'tape, F, R>(&'tape mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>) -> R,
  {
    self.with_scope(0, f)
  }

  fn with_scope<'tape, F, R>(&'tape self, level: u8, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>) -> R,
  {
    let _tape_frame = FrameGuard::new(self, Frame::new(level));
    f(Guard {
      level,
      tape: self,
      phantom: PhantomData,
    })
  }
}

/// Manage/guard the creation/deletion of the tape's stack frame...
struct FrameGuard<'tape> {
  tape: &'tape Tape,
}

impl<'tape> FrameGuard<'tape> {
  fn new(tape: &'tape Tape, frame: Frame) -> Self {
    let guard = Self { tape };
    guard.tape.inner.frames.borrow_mut().stack.push(frame);
    guard
  }
}

impl Drop for FrameGuard<'_> {
  fn drop(&mut self) {
    self.tape.inner.frames.borrow_mut().stack.pop();
  }
}

pub struct Locked;

pub struct Unlocked;

pub struct Guard<'tape, 'scope, S> {
  level: u8,
  tape: &'tape Tape,
  phantom: PhantomData<&'scope S>,
}

impl<'tape, 'scope> Guard<'tape, 'scope, Unlocked>
where
  'scope: 'tape,
{
  #[inline]
  pub fn var(&self, value: f64) -> Var<'tape, 'scope> {
    let mut current_frame = self.tape.inner.current_frame_mut();
    let index = NodeIndex::new(self.level, current_frame.nodes.len() as u64);
    let index = current_frame.add_node(
      Predecessor { index, grad: 0.0 },
      Predecessor { index, grad: 0.0 },
    );
    Var {
      value,
      index,
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  #[inline]
  pub fn lock(self) -> Guard<'tape, 'scope, Locked> {
    Guard {
      level: self.level,
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'tape, 'scope> Guard<'tape, 'scope, Locked>
where
  'scope: 'tape,
{
  #[inline]
  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>) -> R,
  {
    self.tape.with_scope(self.level + 1, f)
  }

  #[inline]
  pub fn collapse(self) -> Gradients<'tape, 'scope> {
    Gradients {
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

type IndexNode = (Node, NodeIndex);

pub struct Gradients<'tape, 'scope> {
  tape: &'tape Tape,
  phantom: PhantomData<&'scope ()>,
}

impl<'tape, 'scope> Gradients<'tape, 'scope>
where
  'scope: 'tape,
{
  pub fn of(&self, var: &Var<'tape, 'scope>) -> Deltas<'scope> {
    let subgraph = topological_subgraph(self.tape, var);
    let mut deltas = FxHashMap::default();
    deltas.insert(var.index, 1.0);
    for (node, index) in subgraph.into_iter().rev() {
      let Node { pred_a, pred_b } = node;
      let single = deltas.get(&index).copied().unwrap_or(0.0);
      *deltas.entry(pred_a.index).or_insert(0.0) += pred_a.grad * single;
      *deltas.entry(pred_b.index).or_insert(0.0) += pred_b.grad * single;
    }
    Deltas {
      deltas,
      phantom: PhantomData,
    }
  }
}

fn topological_subgraph(tape: &Tape, var: &Var<'_, '_>) -> Vec<IndexNode> {
  // linear dfs was fuckin ugly and also slower than this one...
  fn dfs(
    frames: &Frames,
    root: NodeIndex,
    visited: &mut FxHashSet<NodeIndex>,
    rsf: &mut Vec<IndexNode>,
  ) {
    // a NodeIndex is just a u64, so we COULD use a bitset...
    if visited.contains(&root) {
      return;
    }
    visited.insert(root);
    let node = frames.get_node(root).unwrap();
    dfs(frames, node.pred_a.index, visited, rsf);
    dfs(frames, node.pred_b.index, visited, rsf);
    rsf.push((node, root));
  }
  let nodes = tape.inner.frames.borrow();
  let mut result = Vec::new();
  let mut visited = FxHashSet::default();
  dfs(&nodes, var.index, &mut visited, &mut result);
  result
}

#[derive(Debug)]
pub struct Deltas<'scope> {
  deltas: FxHashMap<NodeIndex, f64>,
  phantom: PhantomData<&'scope ()>,
}

impl<'tape, 'scope> Index<&Var<'tape, 'scope>> for Deltas<'scope>
where
  'scope: 'tape,
{
  type Output = f64;
  fn index(&self, var: &Var<'tape, 'scope>) -> &Self::Output {
    self.deltas.get(&var.index).unwrap_or(&0.0)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  mod var {
    use super::*;

    #[test]
    fn value() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(1.3);
        assert_eq!(a.value(), 1.3);
      });
    }

    #[test]
    fn add() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(3.0);
        let b = &guard.var(4.0);
        let c = a + b;
        assert_eq!(c.value(), 3.0 + 4.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1, df/db = 1
        assert_eq!(grads[a], 1.0);
        assert_eq!(grads[b], 1.0);
      });
    }

    #[test]
    fn add_f64() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(3.0);
        let c = a + 5.0;
        assert_eq!(c.value(), 3.0 + 5.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1
        assert_eq!(grads[a], 1.0);
      });
    }

    #[test]
    fn sub() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(7.0);
        let b = &guard.var(4.0);
        let c = a - b;
        assert_eq!(c.value(), 7.0 - 4.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1, df/db = -1
        assert_eq!(grads[a], 1.0);
        assert_eq!(grads[b], -1.0);
      });
    }

    #[test]
    fn sub_f64() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(7.0);
        let c = a - 3.0;
        assert_eq!(c.value(), 7.0 - 3.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1
        assert_eq!(grads[a], 1.0);
      });
    }

    #[test]
    fn mul() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(3.0);
        let b = &guard.var(4.0);
        let c = a * b;
        assert_eq!(c.value(), 3.0 * 4.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = b, df/db = a
        assert_eq!(grads[a], 4.0);
        assert_eq!(grads[b], 3.0);
      });
    }

    #[test]
    fn mul_f64() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(3.0);
        let c = a * 5.0;
        assert_eq!(c.value(), 3.0 * 5.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 5
        assert_eq!(grads[a], 5.0);
      });
    }

    #[test]
    fn div() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(6.0);
        let b = &guard.var(3.0);
        let c = a / b;
        assert_eq!(c.value(), 6.0 / 3.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1/b, df/db = -a/b^2
        assert_eq!(grads[a], 1.0 / 3.0);
        assert_eq!(grads[b], -6.0 / (3.0 * 3.0));
      });
    }

    #[test]
    fn div_f64() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(6.0);
        let c = a / 2.0;
        assert_eq!(c.value(), 6.0 / 2.0);
        let grads = guard.lock().collapse().of(&c);
        // df/da = 1/2
        assert_eq!(grads[a], 1.0 / 2.0);
      });
    }

    #[test]
    fn bitxor() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = &guard.var(3.0);
        let c = a ^ b;
        assert_eq!(c.value(), f64::powf(2.0, 3.0));
        let grads = guard.lock().collapse().of(&c);
        // df/da = b * a^(b-1)
        // df/db = a^b * ln(a)
        assert_eq!(grads[a], 3.0 * f64::powf(2.0, 2.0));
        assert_eq!(grads[b], f64::powf(2.0, 3.0) * f64::ln(2.0));
      });
    }

    #[test]
    fn bitxor_f64() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let c = a ^ 3.0;
        assert_eq!(c.value(), f64::powf(2.0, 3.0));
        let grads = guard.lock().collapse().of(&c);
        // df/da = 3 * a^(3-1)
        assert_eq!(grads[a], 3.0 * f64::powf(2.0, 2.0));
      });
    }

    #[test]
    fn neg() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = guard.var(2.0);
        let neg_a = -&a; // using the reference negation operator
                         // Verify that the value is correctly negated.
        assert_eq!(neg_a.value(), -2.0);
        let grads = guard.lock().collapse().of(&neg_a);
        // For f(x) = -x, the derivative with respect to x should be -1.
        assert_eq!(grads[&a], -1.0);
      });
    }

    #[test]
    fn reciprocal() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.reciprocal();
        assert_eq!(b.value(), 1.0 / 1.3);
        let grads = guard.lock().collapse().of(&b);
        // df/da = -1/a^2
        assert_eq!(grads[a], -1.0 / (1.3 * 1.3));
      });
    }

    #[test]
    fn sin() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.sin();
        assert_eq!(b.value(), f64::sin(1.3));
        let grads = guard.lock().collapse().of(&b);
        // df/da = cos(a)
        assert_eq!(grads[a], f64::cos(1.3));
      });
    }

    #[test]
    fn cos() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(3.1);
        let b = a.cos();
        assert_eq!(b.value(), f64::cos(3.1));
        let grads = guard.lock().collapse().of(&b);
        // df/da = -sin(a)
        assert_eq!(grads[a], -f64::sin(3.1));
      });
    }

    #[test]
    fn tan() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(5.6);
        let b = a.tan();
        assert_eq!(b.value(), f64::tan(5.6));
        let grads = guard.lock().collapse().of(&b);
        // df/da = sec^2(a)
        assert_eq!(grads[a], 1.0 / (f64::cos(5.6).powi(2)));
      });
    }

    #[test]
    fn ln() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(5.6);
        let b = a.ln();
        assert_eq!(b.value(), f64::ln(5.6));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/a
        assert_eq!(grads[a], 1.0 / 5.6);
      });
    }

    #[test]
    fn log() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(5.6);
        let base = 3.0;
        let b = a.log(base);
        assert_eq!(b.value(), f64::log(5.6, base));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(a ln(b))
        assert_eq!(grads[a], 1.0 / (5.6 * base.ln()));
      });
    }

    #[test]
    fn log10() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.0);
        let b = a.log10();
        assert_eq!(b.value(), f64::log10(1.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(a ln(10))
        assert_eq!(grads[a], 1.0 / (1.0 * f64::ln(10.0)));
      });
    }

    #[test]
    fn log2() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = a.log2();
        assert_eq!(b.value(), f64::log2(2.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(a ln(2))
        assert_eq!(grads[a], 1.0 / (2.0 * f64::ln(2.0)));
      });
    }

    #[test]
    fn asin() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(0.5);
        let b = a.asin();
        assert_eq!(b.value(), f64::asin(0.5));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/sqrt(1-a^2)
        assert_eq!(grads[a], 1.0 / f64::sqrt(1.0 - 0.5 * 0.5));
      });
    }

    #[test]
    fn acos() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(0.5);
        let b = a.acos();
        assert_eq!(b.value(), f64::acos(0.5));
        let grads = guard.lock().collapse().of(&b);
        // df/da = -1/sqrt(1-a^2)
        assert_eq!(grads[a], -1.0 / f64::sqrt(1.0 - 0.5 * 0.5));
      });
    }

    #[test]
    fn atan() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.2);
        let b = a.atan();
        assert_eq!(b.value(), f64::atan(1.2));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(1+a^2)
        assert_eq!(grads[a], 1.0 / (1.0 + 1.2 * 1.2));
      });
    }

    #[test]
    fn sinh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.sinh();
        assert_eq!(b.value(), f64::sinh(1.3));
        let grads = guard.lock().collapse().of(&b);
        // df/da = cosh(a)
        assert_eq!(grads[a], f64::cosh(1.3));
      });
    }

    #[test]
    fn cosh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.cosh();
        assert_eq!(b.value(), f64::cosh(1.3));
        let grads = guard.lock().collapse().of(&b);
        // df/da = sinh(a)
        assert_eq!(grads[a], f64::sinh(1.3));
      });
    }

    #[test]
    fn tanh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(0.8);
        let b = a.tanh();
        assert_eq!(b.value(), f64::tanh(0.8));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/cosh^2(a)
        assert_eq!(grads[a], 1.0 / (f64::cosh(0.8) * f64::cosh(0.8)));
      });
    }

    #[test]
    fn asinh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.5);
        let b = a.asinh();
        assert_eq!(b.value(), f64::asinh(1.5));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/sqrt(1+a^2)
        assert_eq!(grads[a], 1.0 / f64::sqrt(1.0 + 1.5 * 1.5));
      });
    }

    #[test]
    fn acosh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = a.acosh();
        assert_eq!(b.value(), f64::acosh(2.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/sqrt(a^2-1)
        assert_eq!(grads[a], 1.0 / f64::sqrt(2.0 * 2.0 - 1.0));
      });
    }

    #[test]
    fn atanh() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(0.0);
        let b = a.atanh();
        assert_eq!(b.value(), f64::atanh(0.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(1-a^2)
        assert_eq!(grads[a], 1.0 / (1.0 - 0.0 * 0.0));
      });
    }

    #[test]
    fn exp() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.exp();
        assert_eq!(b.value(), f64::exp(1.3));
        let grads = guard.lock().collapse().of(&b);
        // df/da = exp(a)
        assert_eq!(grads[a], f64::exp(1.3));
      });
    }

    #[test]
    fn exp2() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.3);
        let b = a.exp2();
        assert_eq!(b.value(), f64::exp2(1.3));
        let grads = guard.lock().collapse().of(&b);
        // df/da = exp2(a) * ln(2)
        assert_eq!(grads[a], f64::exp2(1.3) * f64::ln(2.0));
      });
    }

    #[test]
    fn pow() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = &guard.var(3.0);
        let c = a.pow(b);
        assert_eq!(c.value(), f64::powf(2.0, 3.0));
        let grads = guard.lock().collapse().of(&c);
        // df/da = b * a^(b-1)
        assert_eq!(grads[a], 3.0 * f64::powf(2.0, 3.0 - 1.0));
        // df/db = a^b * ln(a)
        assert_eq!(grads[b], f64::powf(2.0, 3.0) * f64::ln(2.0));
      });
    }

    #[test]
    fn powf() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = a.powf(3.0);
        assert_eq!(b.value(), f64::powf(2.0, 3.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 3 * a^(3-1)
        assert_eq!(grads[a], 3.0 * f64::powf(2.0, 3.0 - 1.0));
      });
    }

    #[test]
    fn sqrt() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(4.0);
        let b = a.sqrt();
        assert_eq!(b.value(), f64::sqrt(4.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(2*sqrt(a))
        assert_eq!(grads[a], 1.0 / (2.0 * f64::sqrt(4.0)));
      });
    }

    #[test]
    fn cbrt() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(1.0);
        let b = a.cbrt();
        assert_eq!(b.value(), f64::cbrt(1.0));
        let grads = guard.lock().collapse().of(&b);
        // df/da = 1/(3*cbrt(a^2))
        assert_eq!(grads[a], 1.0 / (3.0 * f64::cbrt(1.0 * 1.0)));
      });
    }
  }

  mod tape {
    use super::*;

    #[test]
    fn guard() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = &guard.var(3.0);
        assert_eq!(a.value(), 2.0);
        assert_eq!(b.value(), 3.0);
        let a_ptr = a.tape as *const Tape;
        let b_ptr = b.tape as *const Tape;
        assert_eq!(a_ptr, b_ptr);
      });
    }

    #[test]
    fn grads() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(2.0);
        let b = &guard.var(3.0);
        let grads = guard.lock().collapse();
        assert_eq!(a.value(), 2.0);
        assert_eq!(b.value(), 3.0);
        let da = grads.of(a);
        assert_eq!(da[a], 1.0);
        assert_eq!(da[b], 0.0);
      });
    }

    #[test]
    fn reset() {
      let mut tape = Tape::new();
      {
        tape.scope(|guard| {
          let a = &guard.var(1.0);
          let _da = guard.lock().collapse().of(a);
        });
      }
      // after scope ends frame stack should be empty
      assert!(tape.inner.frames.borrow().stack.is_empty());
    }
  }

  mod gradients {
    use super::*;

    #[test]
    fn of() {
      let mut tape = Tape::new();
      tape.scope(|guard| {
        let a = &guard.var(5.0);
        let b = &guard.var(2.0);
        let c = &guard.var(1.0);
        let res = a.pow(b).sub(c.asinh().div(2.0)).add(f64::sin(1.0));
        assert_eq!(
          res.value(),
          f64::powf(5.0, 2.0) - f64::asinh(1.0) / 2.0 + f64::sin(1.0)
        );
        let grads = guard.lock().collapse();
        let dres = grads.of(&res);
        let ga = dres[a]; // df/da
        let gb = dres[b]; // df/db
        let gc = dres[c]; // df/dc
        assert_eq!(ga, 2.0 * 5.0);
        assert_eq!(gb, 25.0 * f64::ln(5.0));
        assert_eq!(gc, -1.0 / (2.0 * 2.0f64.sqrt()));
      });
    }
  }
}
