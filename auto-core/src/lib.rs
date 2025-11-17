//!
//! # auto-core
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
//!

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem;
use std::ops::Index;

use noether::operations::ClosedAddAssign;

use rustc_hash::{FxHashMap, FxHashSet};

use smallvec::SmallVec;

/// Operation identifier to compose inner and outer operation sets...
///
/// The core handles identity (grad passthrough) by default
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpId<U> {
  /// Ignore pullback, ignore gradient...
  Ignore,
  /// Identity pullback, passes gradient through unchanged
  Identity,
  /// User-defined operation, used in tandem with PullbackFamily<T>::Operand
  User(U),
}

/// Produce mempty to seed a gradient computation (multiplicative identity)
pub trait Seedable<T> {
  fn seed(value: &T) -> T;
}

/// A family of pullback operations
///
/// This trait allows users to define their own operation families with custom
/// dispatch mechanisms.
///
/// Lemma: all n-ary tuples can be constructed through a composition of pairs
/// Proof: take some tuple (a, b, c, d), then construct, using cantor pairs, (a, (b, (c, d)))
///
/// As a result, we only provide an interface for binary functions, as they are
/// expressive enough to represent unary function symbols, and by the lemma, all
/// n-ary function symbols (through multiple implementations composed, of course)
pub trait PullbackFamily<T>: Seedable<T> {
  /// User defined operation identifier, usually an enum...
  type Operand: Clone;

  /// Apply pullback for user-defined operation on binary function arg 1
  fn apply_a(op: Self::Operand, captures: &[T], upstream: &T) -> T;

  /// Apply pullback for user-defined operation on binary function arg 2
  fn apply_b(op: Self::Operand, captures: &[T], upstream: &T) -> T;
}

/// Operation trait for defining forward and pullback computations.
///
/// Each concrete operation (Add, Mul, Sin, etc.) implements this trait.
/// This keeps operations modular - no giant trait with every possible operation.
pub trait Operation<T, F>
where
  F: PullbackFamily<T>,
{
  /// Forward computation: binary function T^2 -> T
  fn forward(&self, a: &T, b: &T) -> T;

  /// Pullback specification: how to compute downstream gradients...
  fn pullback_spec(&self, a: &T, b: &T) -> PullbackSpec<T, F>;
}

/// Public spec on how to compute pullbacks for an operation
pub struct PullbackSpec<T, F: PullbackFamily<T>> {
  pub op_id_a: OpId<F::Operand>,
  pub op_id_b: OpId<F::Operand>,
  pub captures: SmallVec<[T; 2]>,
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

#[derive(Clone)]
struct Predecessor<U> {
  op_id: OpId<U>,
  node: NodeIndex,
}

#[derive(Clone)]
struct Node<T, U> {
  pred_a: Predecessor<U>,
  pred_b: Predecessor<U>,
  captures: SmallVec<[T; 2]>,
}

struct VarInner<T, U> {
  index: NodeIndex,
  tape: *const TapeInner<T, U>,
}

/// A variable in the computational graph.
///
/// Variables are created via `Guard::var()`
pub struct Var<'scope, T, U> {
  value: T,
  inner: VarInner<T, U>,
  phantom: PhantomData<&'scope ()>,
}

impl<T, U> Var<'_, T, U> {
  /// Get a readonly reference to the value stored in this variable
  #[inline(always)]
  pub fn value(&self) -> &T {
    &self.value
  }

  /// Core primitive for constructing new variables from binary operations.
  ///
  /// Var provides a generic way of constructing binary operations, but nothing
  /// else... separate implementations are needed to use operations...
  ///
  /// # Example
  ///
  /// ```ignore
  /// impl<'scope, F> Var<'scope, f64, F::Operand>
  /// where
  ///   F: PullbackFamily<f64>,
  /// {
  ///   pub fn add(&self, other: &Self) -> Self {
  ///     self.binary_op(other, AddOp)
  ///   }
  /// }
  /// ```
  #[inline]
  pub fn binary_op<O, F>(&self, other: &Self, op: O) -> Self
  where
    O: Operation<T, F>,
    F: PullbackFamily<T, Operand = U>,
  {
    let value = op.forward(&self.value, &other.value);
    let spec = op.pullback_spec(&self.value, &other.value);

    // safety: a var can only be created under a guard's scope, which is always
    // outlived by the tape... so this is always valid...
    let index = unsafe { &*self.inner.tape }.map_current_frame_mut(move |frame| {
      frame.add_node(
        self.inner.index,
        spec.op_id_a,
        other.inner.index,
        spec.op_id_b,
        spec.captures,
      )
    });

    Var {
      value,
      inner: VarInner {
        index,
        tape: self.inner.tape,
      },
      phantom: PhantomData,
    }
  }
}

impl<T: fmt::Debug, U> fmt::Debug for Var<'_, T, U> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Var")
      .field("value", &self.value)
      .field("level", &self.inner.index.level())
      .field("index", &self.inner.index.index())
      .finish()
  }
}

struct Frame<T, U> {
  level: u8,
  nodes: Vec<Node<T, U>>,
}

impl<T, U> Frame<T, U> {
  fn new(level: u8) -> Self {
    Self {
      level,
      nodes: Vec::new(),
    }
  }

  #[inline]
  fn add_node(
    &mut self,
    pred_a_idx: NodeIndex,
    op_id_a: OpId<U>,
    pred_b_idx: NodeIndex,
    op_id_b: OpId<U>,
    captures: SmallVec<[T; 2]>,
  ) -> NodeIndex {
    let node_idx = self.nodes.len();
    self.nodes.push(Node {
      pred_a: Predecessor {
        op_id: op_id_a,
        node: pred_a_idx,
      },
      pred_b: Predecessor {
        op_id: op_id_b,
        node: pred_b_idx,
      },
      captures,
    });
    NodeIndex::new(self.level, node_idx as u64)
  }
}

impl<T, U> Default for Frame<T, U> {
  fn default() -> Self {
    Self {
      level: 0,
      nodes: Vec::new(),
    }
  }
}

struct Frames<T, U> {
  current: Frame<T, U>,
  rest: Vec<Frame<T, U>>,
}

impl<T, U> Frames<T, U> {
  #[inline]
  fn get_node(&self, index: NodeIndex) -> Option<&Node<T, U>> {
    // to get a node, we need to know which frame it is on (its level) along with
    // its index in that level...
    let level = index.level() as usize;
    let index = index.index() as usize;
    if level == self.current.level as usize {
      self.current.nodes.get(index)
    } else {
      self
        .rest
        .get(level)
        .and_then(|frame| frame.nodes.get(index))
    }
  }

  #[inline]
  fn push(&mut self, frame: Frame<T, U>) {
    let old_current = mem::replace(&mut self.current, frame);
    self.rest.push(old_current);
  }

  #[inline]
  fn pop(&mut self) {
    if let Some(frame) = self.rest.pop() {
      self.current = frame;
    }
  }
}

impl<T, U> Default for Frames<T, U> {
  fn default() -> Self {
    Self {
      current: Frame::default(),
      rest: Vec::new(),
    }
  }
}

struct FrameGuard<'tape, T, U> {
  tape: *const TapeInner<T, U>,
  phantom: PhantomData<&'tape ()>,
}

impl<'tape, T, U> FrameGuard<'tape, T, U> {
  fn new(tape: &'tape TapeInner<T, U>, frame: Frame<T, U>) -> Self {
    let guard = Self {
      tape: tape as *const TapeInner<T, U>,
      phantom: PhantomData,
    };
    // safety: we just got a reference, it was valid on input...
    unsafe { &*guard.tape }.frames.borrow_mut().push(frame);
    guard
  }
}

impl<T, U> Drop for FrameGuard<'_, T, U> {
  fn drop(&mut self) {
    // safety: self is bound by 'tape, we guarantee that tape outlives self
    unsafe { &*self.tape }.frames.borrow_mut().pop();
  }
}

struct TapeInner<T, U> {
  frames: RefCell<Frames<T, U>>,
}

impl<T, U> TapeInner<T, U> {
  #[inline]
  fn map_current_frame_mut<R, G>(&self, f: G) -> R
  where
    G: FnOnce(&mut Frame<T, U>) -> R,
  {
    let mut frames = self.frames.borrow_mut();
    f(&mut frames.current)
  }

  fn with_scope<G, R, F>(&self, level: u8, f: G) -> R
  where
    F: PullbackFamily<T, Operand = U>,
    G: for<'inner> FnOnce(Guard<'inner, T, F, Unlocked>) -> R,
  {
    let _tape_frame = FrameGuard::new(self, Frame::new(level));
    f(Guard {
      level,
      tape: self as *const TapeInner<T, U>,
      phantom: PhantomData,
    })
  }
}

impl<T, U> Default for TapeInner<T, U> {
  fn default() -> Self {
    Self {
      frames: RefCell::new(Frames::default()),
    }
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
pub struct Tape<T, F>
where
  F: PullbackFamily<T>,
{
  inner: TapeInner<T, F::Operand>,
}

impl<T, F> Tape<T, F>
where
  F: PullbackFamily<T>,
{
  pub fn new() -> Self {
    Self {
      inner: TapeInner::default(),
    }
  }

  pub fn scope<G, R>(&mut self, f: G) -> R
  where
    G: for<'inner> FnOnce(Guard<'inner, T, F, Unlocked>) -> R,
  {
    self.inner.with_scope(1, f)
  }
}

impl<T, F> Default for Tape<T, F>
where
  F: PullbackFamily<T>,
{
  fn default() -> Self {
    Self::new()
  }
}

/// Phantom type for a locked guard; a locked guard cannot create variables
/// in that it is not allowed to modify it's scope
pub struct Locked;

/// Phantom type for an unlocked guard, something we CAN create variables on...
pub struct Unlocked;

/// An unlocked `Guard` provides an API to construct variables in a specific scope,
/// once a guard is locked it can create subscopes or produce gradients for the
/// variables constructed while unlocked...
pub struct Guard<'scope, T, F, S = Unlocked>
where
  F: PullbackFamily<T>,
{
  level: u8,
  /// Pointer is guaranteed to outlast 'scope...
  tape: *const TapeInner<T, F::Operand>,
  phantom: PhantomData<(&'scope S, F)>,
}

impl<'scope, T, F> Guard<'scope, T, F, Unlocked>
where
  F: PullbackFamily<T>,
{
  /// Construct a new variable of a specific value; the created variable cannot
  /// migrate out of the enclosing scope, but it can propagate mutations made in
  /// lower scopes...
  #[inline]
  pub fn var(&self, value: T) -> Var<'scope, T, F::Operand> {
    let tape = self.tape;
    let level = self.level;

    // safety: guards are always created under a tapes 'scope...
    let index = unsafe { &*tape }.map_current_frame_mut(|frame| {
      let self_idx = NodeIndex::new(level, frame.nodes.len() as u64);
      // leaves: both predecessors point to self with identity...
      // the loop check in Gradients::of ensures we only apply pred_a (identity)
      frame.add_node(
        self_idx,
        OpId::Identity,
        self_idx,
        OpId::Identity, // never called due to loop check
        SmallVec::new(),
      )
    });

    Var {
      value,
      inner: VarInner {
        index,
        tape: tape as *const _,
      },
      phantom: PhantomData,
    }
  }

  /// Lock a guard...
  ///
  /// Consume an unlocked guard and produce a locked guard with same lifetimes
  #[inline]
  pub fn lock(self) -> Guard<'scope, T, F, Locked> {
    Guard {
      level: self.level,
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

impl<'scope, T, F> Guard<'scope, T, F, Locked>
where
  F: PullbackFamily<T>,
{
  /// A locked guard can spawn additional scopes for computation
  ///
  /// Subscopes will themselves provide a new guard for their own scopes...
  #[inline]
  pub fn scope<G, R>(&mut self, f: G) -> R
  where
    G: for<'inner> FnOnce(Guard<'inner, T, F, Unlocked>) -> R,
  {
    // safety: we are under 'scope, so tape is valid...
    let tape: &TapeInner<T, F::Operand> = unsafe { &*self.tape };
    assert!(self.level < u8::MAX);
    tape.with_scope(self.level + 1, f)
  }

  /// A locked guard can collapse into gradients for the variables that were
  /// created on it while unlocked...
  #[inline]
  pub fn collapse(self) -> (Mutator<'scope, F>, Gradients<'scope, T, F>) {
    (
      Mutator {
        phantom: PhantomData,
      },
      Gradients {
        tape: self.tape,
        phantom: PhantomData,
      },
    )
  }
}

pub struct Mutator<'scope, F> {
  phantom: PhantomData<&'scope F>,
}

impl<'scope, F> Mutator<'scope, F> {
  // jesus christ
  pub fn update<'a, T, U, G>(&mut self, var: &mut Var<'a, T, U>, f: G)
  where
    'a: 'scope,
    G: FnOnce(&T) -> T,
    F: PullbackFamily<T, Operand = U>,
  {
    var.value = f(&var.value);
  }

  pub fn set<'a, T, U>(&mut self, var: &mut Var<'a, T, U>, value: T)
  where
    'a: 'scope,
  {
    var.value = value;
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

/// Possible derivatives for a specified scope...
pub struct Gradients<'scope, T, F>
where
  F: PullbackFamily<T>,
{
  /// Pointer is guaranteed to outlast 'scope...
  tape: *const TapeInner<T, F::Operand>,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope, T, F> Gradients<'scope, T, F>
where
  F: PullbackFamily<T>,
{
  /// Get the deltas of some variable in some scope
  ///
  /// This function is the hottest part of the program, occupying on average ~65%
  /// of the run time...
  ///
  /// Note: we require a seed gradient to start the computation. For scalar types,
  /// this is typically 1.0. For shaped types (matrices, tensors), this must match
  /// the shape of the output variable...
  pub fn of(&self, var: &Var<'scope, T, F::Operand>, seed: T) -> Deltas<'scope, T>
  where
    T: ClosedAddAssign + Clone,
    F::Operand: Clone,
  {
    // safety: we are under the tapes 'scope, so tape is valid...
    let tape = unsafe { &*self.tape };
    let subgraph = topological_subgraph(tape, var);
    let mut deltas = FxHashMap::with_capacity(subgraph.len());
    // seed dv/dv with provided gradient (e.g., 1.0 for scalars, ones(shape) for matrices)
    deltas.insert(var.inner.index, seed);

    for (index, node) in subgraph.into_iter().rev() {
      let Node {
        pred_a,
        pred_b,
        captures,
      } = node;

      // read phase, get upstream gradient, skip if n/a
      let upstream = match deltas.get(&index) {
        Some(v) => v,
        None => continue,
      };

      // compute phase, calculate gradients without borrowing...
      let mut grads: SmallVec<[(NodeIndex, T); 2]> = SmallVec::new();

      // a-branch pullback
      if pred_a.node != index {
        match pred_a.op_id {
          OpId::Ignore => {}
          OpId::Identity => {
            grads.push((pred_a.node, upstream.clone()));
          }
          OpId::User(ref op) => {
            let grad_a = F::apply_a(op.clone(), &captures, upstream);
            grads.push((pred_a.node, grad_a));
          }
        }
      }

      // b-branch pullback
      if pred_b.node != index {
        match pred_b.op_id {
          OpId::Ignore => {}
          OpId::Identity => {
            grads.push((pred_b.node, upstream.clone()));
          }
          OpId::User(ref op) => {
            let grad_b = F::apply_b(op.clone(), &captures, upstream);
            grads.push((pred_b.node, grad_b));
          }
        }
      }

      // write phase, emit updated deltas...
      for (node_idx, grad) in grads {
        deltas
          .entry(node_idx)
          .and_modify(|g| {
            *g += grad.clone();
          })
          .or_insert(grad);
      }
    }

    Deltas {
      deltas,
      phantom: PhantomData,
    }
  }
}

/// Topologically sort our graph moving backwards along predecessors of a given
/// variable...
fn topological_subgraph<T, U>(
  tape: &TapeInner<T, U>,
  var: &Var<'_, T, U>,
) -> Vec<(NodeIndex, Node<T, U>)>
where
  T: Clone,
  U: Clone,
{
  let frames = tape.frames.borrow();

  // preallocating a little bit of extra room provides ~20% speedup...
  let mut stack = Vec::with_capacity(512);
  let mut result = Vec::with_capacity(512);
  let mut visited = FxHashSet::with_capacity(512);

  stack.push((var.inner.index, false));

  // linear dfs for easier tracing... can always revert to prettier recursive...
  while let Some((node_index, children_processed)) = stack.pop() {
    if children_processed {
      // add to result in postorder
      if let Some(node) = frames.get_node(node_index).cloned() {
        result.push((node_index, node));
      }
    } else if visited.insert(node_index) {
      // marker to add node after children
      stack.push((node_index, true));
      if let Some(node) = frames.get_node(node_index) {
        // process pred_a before pred_b, order matters...
        if !visited.contains(&node.pred_b.node) {
          stack.push((node.pred_b.node, false));
        }
        if !visited.contains(&node.pred_a.node) {
          stack.push((node.pred_a.node, false));
        }
      }
    }
  }

  result
}

/// Extension trait for computing gradients of a variable
///
/// This trait is implemented by type-specific crates to provide the appropriate
/// seed/unit gradient for differentiation
pub trait Gradient<'scope, T, U> {
  /// Compute the deltas/gradient set for self
  fn deltas<F>(&self, gradients: &Gradients<'scope, T, F>) -> Deltas<'scope, T>
  where
    F: PullbackFamily<T, Operand = U>;
}

impl<'scope, T, U> Gradient<'scope, T, U> for Var<'scope, T, U>
where
  U: Clone,
  T: ClosedAddAssign + Clone,
{
  fn deltas<F>(&self, gradients: &Gradients<'scope, T, F>) -> Deltas<'scope, T>
  where
    F: PullbackFamily<T, Operand = U>,
  {
    let seed = F::seed(self.value());
    gradients.of(self, seed)
  }
}

/// The deltas of some variable in a specified scope; deltas can be with respect
/// to variables declared in outer scopes...
pub struct Deltas<'scope, T> {
  deltas: FxHashMap<NodeIndex, T>,
  phantom: PhantomData<&'scope ()>,
}

impl<'scope, T> Deltas<'scope, T> {
  pub fn get<U>(&self, var: &Var<'scope, T, U>) -> Option<T>
  where
    T: Clone,
  {
    self.deltas.get(&var.inner.index).cloned()
  }
}

impl<'scope, T, U> Index<&Var<'scope, T, U>> for Deltas<'scope, T> {
  type Output = T;

  fn index(&self, var: &Var<'scope, T, U>) -> &Self::Output {
    self.deltas.get(&var.inner.index).unwrap()
  }
}

impl<T> fmt::Debug for Deltas<'_, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Deltas")
      .field("count", &self.deltas.len())
      .finish()
  }
}
