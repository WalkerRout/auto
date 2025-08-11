//! # auto: Reverse-Mode Automatic Differentiation
//!
//! This library provides a safe, efficient implementation of reverse-mode automatic 
//! differentiation using Rust's type system to enforce memory safety and prevent API
//! misuse.
//!
//! ## Core API
//!
//! The main entry point is [`Tape::scope`], which creates a computational scope where
//! variables can be created and operations performed:
//!
//! ```rust
//! use auto::Tape;
//!
//! let mut tape = Tape::new();
//! tape.scope(|guard| {
//!   let x = guard.var(2.0);
//!   let y = guard.var(3.0);
//!   // z = x*y + 1
//!   let z = &x * &y + 1.0;
//!     
//!   let gradients = guard.lock().collapse().of(&z);
//!   println!("dz/dx = {}", gradients[&x]); // 3.0
//!   println!("dz/dy = {}", gradients[&y]); // 2.0
//! });
//! ```
//!
//! ## Safety Invariants
//!
//! The unsafe code in this library maintains these invariants:
//!
//! 1. The tape remains valid for the entire duration of any scope created from it
//! 2. Raw pointers are only dereferenced under the scope where they were created
//! 3. The computational graph structure is immutable once gradient computation begins
//!
//! These invariants are enforced by:
//! - Taking a mutable borrow of the tape at scope creation
//! - Typestate-based guards to manage variable lifetimes
//! - Frame guards that automatically clean up computational state

use std::cell::{RefCell, RefMut};
use std::collections::{HashSet, HashMap};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Deref, DerefMut, Div, Index, Mul, Neg, Sub};

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
    let pred_a = self.to_predecessor(-1.0 / (v * v));
    let pred_b = self.to_predecessor(0.0);
    self.produce(1.0 / v, pred_a, pred_b)
  }

  #[inline]
  pub fn sin(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cos());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.sin(), pred_a, pred_b)
  }

  #[inline]
  pub fn cos(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(-v.sin());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.cos(), pred_a, pred_b)
  }

  #[inline]
  pub fn tan(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cos() * v.cos()));
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.tan(), pred_a, pred_b)
  }

  #[inline]
  pub fn ln(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / v);
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.ln(), pred_a, pred_b)
  }

  #[inline]
  pub fn log(&self, base: f64) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * base.ln()));
    let pred_b = self.to_predecessor(0.0);
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
    let pred_a = self.to_predecessor(1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.asin(), pred_a, pred_b)
  }

  #[inline]
  pub fn acos(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(-1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.acos(), pred_a, pred_b)
  }

  #[inline]
  pub fn atan(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v));
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.atan(), pred_a, pred_b)
  }

  #[inline]
  pub fn sinh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cosh());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.sinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn cosh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.sinh());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.cosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn tanh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cosh() * v.cosh()));
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.tanh(), pred_a, pred_b)
  }

  #[inline]
  pub fn asinh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.asinh(), pred_a, pred_b)
  }

  #[inline]
  pub fn acosh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * v - 1.0).sqrt());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.acosh(), pred_a, pred_b)
  }

  #[inline]
  pub fn atanh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.atanh(), pred_a, pred_b)
  }

  #[inline]
  pub fn exp(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.exp(), pred_a, pred_b)
  }

  #[inline]
  pub fn exp2(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp2() * 2.0_f64.ln());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.exp2(), pred_a, pred_b)
  }

  /// We explicitly declare a function named `pow` on Var since `bitxor` is
  /// not as descriptive as `add` and `sub`...
  #[inline]
  pub fn pow(&self, other: &Self) -> Self {
    let v = self.value;
    let ov = other.value;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = other.to_predecessor(v.powf(ov) * v.ln());
    self.produce(v.powf(ov), pred_a, pred_b)
  }

  /// Same as sister above, but takes a float exponent
  #[inline]
  pub fn powf(&self, other: f64) -> Self {
    let v = self.value;
    let ov = other;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.powf(ov), pred_a, pred_b)
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
    let pred_a = self.to_predecessor(v / v.abs());
    let pred_b = self.to_predecessor(0.0);
    self.produce(v.abs(), pred_a, pred_b)
  }

  #[inline(always)]
  fn produce(&self, value: f64, pred_a: Predecessor, pred_b: Predecessor) -> Self {
    Var {
      value,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
    }
  }

  #[inline(always)]
  fn to_predecessor(&self, grad: f64) -> Predecessor {
    Predecessor {
      grad,
      index: self.index,
    }
  }

  #[inline]
  fn add_node(&self, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    self.tape.current_frame_mut().add_node(pred_a, pred_b)
  }
}

// we implement 2 variants for each binary operation; &a <op> &b and &a <op> f64
// - owned implementations defer to reference implementations

impl<'scope> Add for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn add(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(1.0);
    self.produce(self.value + other.value, pred_a, pred_b)
  }
}

impl<'scope> Add<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn add(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    self.produce(self.value + other, pred_a, pred_b)
  }
}

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

impl<'scope> Sub for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn sub(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(-1.0);
    self.produce(self.value - other.value, pred_a, pred_b)
  }
}

impl<'scope> Sub<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn sub(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    self.produce(self.value - other, pred_a, pred_b)
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

impl<'scope> Mul for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(other.value);
    let pred_b = other.to_predecessor(self.value);
    self.produce(self.value * other.value, pred_a, pred_b)
  }
}

impl<'scope> Mul<f64> for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline]
  fn mul(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(other);
    let pred_b = self.to_predecessor(0.0);
    self.produce(self.value * other, pred_a, pred_b)
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
    // same thing...
    self.mul(1.0 / other)
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

impl<'scope> Neg for &Var<'scope> {
  type Output = Var<'scope>;
  #[inline(always)]
  fn neg(self) -> Self::Output {
    self * -1.0
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
    // to get a node, we need to know which frame it is on (its level) along with
    // its index in that level...
    let level = index.level() as usize;
    let idx = index.index() as usize;
    self
      .stack
      .get(level)
      .and_then(|frame| frame.nodes.get(idx))
      .cloned()
  }
}

/// Manage/guard the creation/deletion of the tape's stack frame...
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

/// A `Tape` is almost like a 'tiered' Wengert list, in that it kind of holds a
/// stack of frames, with each frame representing a guarded scope, instead of just
/// a plain old list of nodes... this struct is just a public wrapper around
/// `TapeInner`...
///
/// Cool things to note:
/// - We can and will only ever modify what is the top frame
/// - When we calculate gradients, they are usually of the same shape for a given
///   variable... we should store a fingerprint (a graphed-hash)...
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
    // taking a mutable reference to the tape means we have exclusive access in
    // subscopes...
    self.inner.with_scope(0, f)
  }
}

/// Phantom type for a locked guard; a locked guard cannot create variables
/// in that it is not allowed to modify it's scope
///
/// Honestly we could probably add the ability to transition back to unlocked
/// after locking...
pub struct Locked;

/// Phantom type for an unlocked guard, something we CAN create variables on...
pub struct Unlocked;

/// An unlocked `Guard` provides an API to construct variables in a specific scope,
/// once a guard is locked it can create subscopes or produce gradients for the
/// variables constructed while unlocked...
pub struct Guard<'scope, S = Unlocked> {
  level: u8,
  /// Pointer is guaranteed to outlast 'scope...
  tape: *const TapeInner,
  phantom: PhantomData<&'scope S>,
}

impl<'scope> Guard<'scope, Unlocked> {
  /// Construct a new variable of a specific value; the created variable cannot
  /// migrate out of the enclosing scope, but it can propagate mutations made in
  /// lower scopes...
  #[inline]
  pub fn var(&self, value: f64) -> Var<'scope> {
    // Safety: tape is valid for entire computation because it was borrowed
    // mutably at the top scope...
    let tape = unsafe { &*self.tape };
    let mut current_frame = tape.current_frame_mut();
    let index = NodeIndex::new(self.level, current_frame.nodes.len() as u64);
    let index = current_frame.add_node(
      Predecessor { index, grad: 0.0 },
      Predecessor { index, grad: 0.0 },
    );
    
    Var {
      value,
      index,
      tape,
    }
  }

  /// Lock a guard...
  ///
  /// Consume an unlocked guard and produce a locked guard with same lifetimes
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
  /// A locked guard can spawn additional scopes for computation
  ///
  /// Subscopes will themselves provide a new guard for their own scopes...
  #[inline]
  pub fn scope<F, R>(&mut self, f: F) -> R
  where
    F: for<'inner> FnOnce(Guard<'inner, Unlocked>) -> R,
  {
    // Safety: tape is valid for entire computation
    let tape: &TapeInner = unsafe { &*self.tape };
    assert!(self.level < u8::MAX);
    tape.with_scope(self.level + 1, f)
  }

  /// A locked guard can collapse into gradients for the variables that were
  /// created on it while unlocked...
  #[inline]
  pub fn collapse(self) -> Gradients<'scope> {
    Gradients {
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

/// We know how big of a hashmap we want to store our deltas in, but we cant
/// create a map with a given capacity since we need to supply the hasher too...
trait HashMapExt {
  fn with_capacity(x: usize) -> Self;
}

/// Just specialize with_capacity for those maps who have a default hasher...
impl<K, V, S> HashMapExt for HashMap<K, V, S>
where
  K: Hash + Eq,
  S: BuildHasher + Default,
{
  fn with_capacity(capacity: usize) -> Self {
    HashMap::with_capacity_and_hasher(capacity, S::default())
  }
}

/// Unfortunately we need to duplicate the above to work for hashsets too...
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

/// An intermediate way to represent an enumerated node...
type IndexNode = (NodeIndex, Node);

/// Possible derivatives for a specified scope...
pub struct Gradients<'scope> {
  /// Pointer is guaranteed to outlast 'scope...
  tape: *const TapeInner,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope> Gradients<'scope> {
  /// Get the deltas of some variable in some scope
  ///
  /// This function is the hottest part of the program, occupying on average ~65%
  /// of the run time...
  pub fn of(&self, var: &Var<'scope>) -> Deltas<'scope> {
    // Safety: tape is valid for entire computation
    let tape = unsafe { &*self.tape };
    let subgraph = topological_subgraph(tape, var);

    let mut deltas = FxHashMap::with_capacity(subgraph.len());
    deltas.insert(var.index, 1.0);

    for (index, node) in subgraph.into_iter().rev() {
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

/// Topologically sort our graph moving backwards along predecessors of a given
/// variable...
///
/// TODO: store in/out degrees and use kahns algorithm
fn topological_subgraph(tape: &TapeInner, var: &Var<'_>) -> Vec<IndexNode> {
  let nodes = tape.frames.borrow();

  // preallocating a little bit of extra room provides ~20% speedup...
  let mut stack = Vec::with_capacity(512);
  let mut result = Vec::with_capacity(512);
  let mut visited = FxHashSet::with_capacity(512); // extension used here...

  stack.push((var.index, false));

  // linear dfs for easier tracing... can always revert to prettier recursive...
  while let Some((node_index, children_processed)) = stack.pop() {
    if children_processed {
      // add to result in postorder
      if let Some(node) = nodes.get_node(node_index) {
        result.push((node_index, node));
      }
    } else if visited.insert(node_index) {
      // marker to add node after children
      stack.push((node_index, true));
      if let Some(node) = nodes.get_node(node_index) {
        // process pred_a before pred_b, order matters...
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

/// The deltas of some variable in a specified scope; deltas can be with respect
/// to variables declared in outer scopes...
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
        let a_ptr = a.tape as *const TapeInner;
        let b_ptr = b.tape as *const TapeInner;
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