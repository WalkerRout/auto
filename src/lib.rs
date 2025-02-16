use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Add, BitXor, Div, Index, Mul, Sub};

use bit_set::BitSet;

type NodeIndex = usize;

/// We have code that relies on `NodeIndex` being equal in size to a usize...
const _: () = {
  use std::mem::size_of;
  assert!(size_of::<NodeIndex>() <= size_of::<usize>())
};

#[derive(Debug, PartialEq)]
struct Predecessor {
  loc: NodeIndex,
  grad: f64,
}

#[derive(Debug, PartialEq)]
struct Node {
  pred_a: Predecessor,
  pred_b: Predecessor,
}

pub struct Var<'snap> {
  value: f64,
  node: NodeIndex,
  snap: &'snap Snapshot,
}

impl<'snap> Var<'snap> {
  #[inline]
  pub fn value(&self) -> f64 {
    self.value
  }

  #[inline]
  pub fn reciprocal(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(-1.0 / (v * v));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: 1.0 / v,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn sin(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cos());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.sin(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn cos(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(-v.sin());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.cos(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn tan(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cos() * v.cos()));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.tan(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn ln(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / v);
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.ln(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn log(&self, base: f64) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * base.ln()));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.log(base),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
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
    Self {
      value: v.asin(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn acos(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(-1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.acos(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn atan(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.atan(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn sinh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.cosh());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.sinh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn cosh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.sinh());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.cosh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn tanh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v.cosh() * v.cosh()));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.tanh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn asinh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 + v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.asinh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn acosh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (v * v - 1.0).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.acosh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn atanh(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(1.0 / (1.0 - v * v).sqrt());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.atanh(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn exp(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.exp(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn exp2(&self) -> Self {
    let v = self.value;
    let pred_a = self.to_predecessor(v.exp2() * 2.0f64.ln());
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.exp2(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  /// We explicitly declare a function named `pow` on var,
  /// since `bitxor` is unclear as to what this does...
  #[inline]
  pub fn pow(&self, other: &Self) -> Self {
    let v = self.value;
    let ov = other.value;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = other.to_predecessor(v.powf(ov) * v.ln());
    Self {
      value: v.powf(ov),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  /// Same as its brother
  #[inline]
  pub fn powf(&self, other: f64) -> Self {
    let v = self.value;
    let ov = other;
    let pred_a = self.to_predecessor(ov * v.powf(ov - 1.0));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: v.powf(ov),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  pub fn sqrt(&self) -> Self {
    self.powf(1.0 / 2.0)
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
    Self {
      value: v.abs(),
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }

  #[inline]
  fn to_predecessor(&self, grad: f64) -> Predecessor {
    Predecessor {
      loc: self.node,
      grad,
    }
  }
}

impl<'snap> Add for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn add(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(1.0);
    Self::Output {
      value: self.value + other.value,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Add<f64> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn add(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    Self::Output {
      value: self.value + other,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Add<Var<'snap>> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn add(self, other: Var<'snap>) -> Self::Output {
    self.add(&other)
  }
}

impl<'snap> Add for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn add(self, other: Self) -> Self::Output {
    (&self).add(&other)
  }
}

impl<'snap> Add<f64> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn add(self, other: f64) -> Self::Output {
    (&self).add(other)
  }
}

impl<'snap> Add<&Var<'snap>> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn add(self, other: &Var<'snap>) -> Self::Output {
    (&self).add(other)
  }
}

impl<'snap> Sub for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn sub(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(-1.0);
    Self::Output {
      value: self.value - other.value,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Sub<f64> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn sub(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(1.0);
    let pred_b = self.to_predecessor(0.0);
    Self::Output {
      value: self.value - other,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Sub<Var<'snap>> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn sub(self, other: Var<'snap>) -> Self::Output {
    self.sub(&other)
  }
}

impl<'snap> Sub for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn sub(self, other: Self) -> Self::Output {
    (&self).sub(&other)
  }
}

impl<'snap> Sub<f64> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn sub(self, other: f64) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'snap> Sub<&Var<'snap>> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn sub(self, other: &Var<'snap>) -> Self::Output {
    (&self).sub(other)
  }
}

impl<'snap> Mul for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn mul(self, other: Self) -> Self::Output {
    let pred_a = self.to_predecessor(other.value);
    let pred_b = other.to_predecessor(self.value);
    Self::Output {
      value: self.value * other.value,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Mul<f64> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline]
  fn mul(self, other: f64) -> Self::Output {
    let pred_a = self.to_predecessor(other);
    let pred_b = self.to_predecessor(0.0);
    Self::Output {
      value: self.value * other,
      node: self.snap.add_node(pred_a, pred_b),
      snap: self.snap,
    }
  }
}

impl<'snap> Mul<Var<'snap>> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn mul(self, other: Var<'snap>) -> Self::Output {
    self.mul(&other)
  }
}

impl<'snap> Mul for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn mul(self, other: Self) -> Self::Output {
    (&self).mul(&other)
  }
}

impl<'snap> Mul<f64> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn mul(self, other: f64) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'snap> Mul<&Var<'snap>> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn mul(self, other: &Var<'snap>) -> Self::Output {
    (&self).mul(other)
  }
}

impl<'snap> Div for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    self.mul(&other.reciprocal())
  }
}

impl<'snap> Div<f64> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    self.mul(other.recip())
  }
}

impl<'snap> Div<Var<'snap>> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: Var<'snap>) -> Self::Output {
    self.div(&other)
  }
}

impl<'snap> Div for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: Self) -> Self::Output {
    (&self).div(&other)
  }
}

impl<'snap> Div<f64> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: f64) -> Self::Output {
    (&self).div(other)
  }
}

impl<'snap> Div<&Var<'snap>> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn div(self, other: &Var<'snap>) -> Self::Output {
    (&self).div(other)
  }
}

impl<'snap> BitXor for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    self.pow(other)
  }
}

impl<'snap> BitXor<f64> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    self.powf(other)
  }
}

impl<'snap> BitXor<Var<'snap>> for &Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: Var<'snap>) -> Self::Output {
    self.bitxor(&other)
  }
}

impl<'snap> BitXor for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: Self) -> Self::Output {
    (&self).bitxor(&other)
  }
}

impl<'snap> BitXor<f64> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: f64) -> Self::Output {
    (&self).bitxor(other)
  }
}

impl<'snap> BitXor<&Var<'snap>> for Var<'snap> {
  type Output = Var<'snap>;

  #[inline(always)]
  fn bitxor(self, other: &Var<'snap>) -> Self::Output {
    (&self).bitxor(other)
  }
}

#[derive(Debug, Default)]
struct Snapshot {
  nodes: RefCell<Vec<Node>>,
}

impl Snapshot {
  #[inline]
  fn var(&self, value: f64) -> Var {
    let me = self.nodes.borrow().len();
    Var {
      value,
      node: self.add_node(
        Predecessor { loc: me, grad: 0.0 },
        Predecessor { loc: me, grad: 0.0 },
      ),
      snap: self,
    }
  }

  #[inline]
  fn add_node(&self, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    let mut nodes = self.nodes.borrow_mut();
    let node = nodes.len();
    nodes.push(Node { pred_a, pred_b });
    node
  }
}

#[derive(Debug, Default)]
pub struct Tape {
  snap: Snapshot,
}

impl Tape {
  pub fn new() -> Self {
    Self {
      snap: Snapshot::default(),
    }
  }

  pub fn guard(&mut self) -> TapeGuard {
    TapeGuard { snap: &self.snap }
  }
}

pub struct TapeGuard<'snap> {
  snap: &'snap Snapshot,
}

impl<'snap> TapeGuard<'snap> {
  #[inline(always)]
  pub fn var(&self, value: f64) -> Var<'snap> {
    self.snap.var(value)
  }

  /// Collapse a guard into gradients, forfeiting exclusive access
  /// - Gradients carry on the borrow to outstanding Var's
  pub fn collapse(self) -> Gradients<'snap> {
    Gradients { snap: self.snap }
  }
}

/// Gradients exist for a specific snapshot of the tape
/// -> we can get gradients by consuming that snapshot!
#[derive(Debug)]
pub struct Gradients<'snap> {
  snap: &'snap Snapshot,
}

impl<'snap> Gradients<'snap> {
  pub fn of(&self, var: &Var<'snap>) -> Deltas<'snap> {
    let nodes = self.snap.nodes.borrow();

    let mut deltas = HashMap::new();
    deltas.insert(var.node, 1.0);

    let subgraph = self.topological_subgraph_of(var);

    for i in subgraph.into_iter().rev() {
      let Node { pred_a, pred_b } = &nodes[i];
      let Predecessor {
        loc: loc_a,
        grad: grad_a,
      } = *pred_a;
      let Predecessor {
        loc: loc_b,
        grad: grad_b,
      } = *pred_b;
      // a single derivative to apply to our jacobian based accumulation
      let single = deltas.get(&i).copied().unwrap_or(0.0);
      *deltas.entry(loc_a).or_insert(0.0) += grad_a * single;
      *deltas.entry(loc_b).or_insert(0.0) += grad_b * single;
    }

    Deltas {
      deltas,
      _phantom: PhantomData,
    }
  }

  fn topological_subgraph_of(&self, var: &Var<'snap>) -> Vec<NodeIndex> {
    // TODO: refactor to linear dfs w/ stack
    fn dfs(nodes: &[Node], root: NodeIndex, visited: &mut BitSet, rsf: &mut Vec<NodeIndex>) {
      // a NodeIndex is just a usize, so we can use a bitset...
      if visited.contains(root) {
        return;
      }
      visited.insert(root);
      let Node { pred_a, pred_b } = &nodes[root];
      dfs(nodes, pred_a.loc, visited, rsf);
      dfs(nodes, pred_b.loc, visited, rsf);
      rsf.push(root);
    }
    let nodes = self.snap.nodes.borrow();
    let mut result = Vec::new();
    let mut visited = BitSet::new();
    dfs(&nodes, var.node, &mut visited, &mut result);
    result
  }
}

impl<'snap> Drop for Gradients<'snap> {
  fn drop(&mut self) {
    self.snap.nodes.borrow_mut().clear();
  }
}

#[derive(Debug)]
pub struct Deltas<'snap> {
  deltas: HashMap<NodeIndex, f64>,
  _phantom: PhantomData<&'snap ()>,
}

impl<'snap> Index<&Var<'snap>> for Deltas<'snap> {
  type Output = f64;

  #[inline]
  fn index(&self, var: &Var<'snap>) -> &Self::Output {
    self.deltas.get(&var.node).unwrap_or(&0.0)
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
      let guard = tape.guard();
      let a = guard.var(1.3);
      assert_eq!(a.value(), 1.3);
    }

    #[test]
    fn add_var() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(3.0);
      let b = &guard.var(4.0);
      let c = a + b;
      assert_eq!(c.value(), 3.0 + 4.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1
      // df/db = 1
      assert_eq!(dc[a], 1.0);
      assert_eq!(dc[b], 1.0);
    }

    #[test]
    fn add_f64() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(3.0);
      let c = a + 5.0;
      assert_eq!(c.value(), 3.0 + 5.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1
      assert_eq!(dc[a], 1.0);
    }

    #[test]
    fn sub_var() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(7.0);
      let b = &guard.var(4.0);
      let c = a - b;
      assert_eq!(c.value(), 7.0 - 4.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1
      // df/db = -1
      assert_eq!(dc[a], 1.0);
      assert_eq!(dc[b], -1.0);
    }

    #[test]
    fn sub_f64() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(7.0);
      let c = a - 3.0;
      assert_eq!(c.value(), 7.0 - 3.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1
      assert_eq!(dc[a], 1.0);
    }

    #[test]
    fn mul_var() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(3.0);
      let b = &guard.var(4.0);
      let c = a * b;
      assert_eq!(c.value(), 3.0 * 4.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = b
      // df/db = a
      assert_eq!(dc[a], 4.0);
      assert_eq!(dc[b], 3.0);
    }

    #[test]
    fn mul_f64() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(3.0);
      let c = a * 5.0;
      assert_eq!(c.value(), 3.0 * 5.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 5
      assert_eq!(dc[a], 5.0);
    }

    #[test]
    fn div_var() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(6.0);
      let b = &guard.var(3.0);
      let c = a / b;
      assert_eq!(c.value(), 6.0 / 3.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1/b
      // df/db = -a/b^2
      assert_eq!(dc[a], 1.0 / 3.0);
      assert_eq!(dc[b], -6.0 / (3.0 * 3.0));
    }

    #[test]
    fn div_f64() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(6.0);
      let c = a / 2.0;
      assert_eq!(c.value(), 6.0 / 2.0);
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 1/2
      assert_eq!(dc[&a], 1.0 / 2.0);
    }

    #[test]
    fn bitxor_var() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(2.0);
      let b = &guard.var(3.0);
      let c = a ^ b;
      assert_eq!(c.value(), f64::powf(2.0, 3.0));
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = b * a^(b-1)
      // df/db = a^b * ln(a)
      assert_eq!(dc[a], 3.0 * f64::powf(2.0, 2.0));
      assert_eq!(dc[b], f64::powf(2.0, 3.0) * f64::ln(2.0));
    }

    #[test]
    fn bitxor_f64() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(2.0);
      let c = &a ^ 3.0;
      assert_eq!(c.value(), f64::powf(2.0, 3.0));
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = 3 * a^(3-1)
      assert_eq!(dc[&a], 3.0 * f64::powf(2.0, 2.0));
    }

    #[test]
    fn reciprocal() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.3);
      let b = a.reciprocal();
      assert_eq!(b.value(), 1.0 / 1.3);
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = -1/a^2
      assert_eq!(db[&a], -1.0 / (1.3 * 1.3));
    }

    #[test]
    fn sin() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.3);
      let b = a.sin();
      assert_eq!(b.value(), f64::sin(1.3));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = cos(a)
      assert_eq!(db[&a], f64::cos(1.3));
    }

    #[test]
    fn cos() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(3.1);
      let b = a.cos();
      assert_eq!(b.value(), f64::cos(3.1));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = -sin(a)
      assert_eq!(db[&a], -f64::sin(3.1));
    }

    #[test]
    fn tan() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(5.6);
      let b = a.tan();
      assert_eq!(b.value(), f64::tan(5.6));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = sec^2(a)
      assert_eq!(db[&a], 1.0 / (f64::cos(5.6).powi(2)));
    }

    #[test]
    fn ln() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(5.6);
      let b = a.ln();
      assert_eq!(b.value(), f64::ln(5.6));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/a
      assert_eq!(db[&a], 1.0 / 5.6);
    }

    #[test]
    fn log() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(5.6);
      let base = 3.0;
      let b = a.log(base);
      assert_eq!(b.value(), f64::log(5.6, base));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(aln(b))
      assert_eq!(db[&a], 1.0 / (5.6 * base.ln()));
    }

    #[test]
    fn log10() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.0);
      let b = a.log10();
      assert_eq!(b.value(), f64::log10(1.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(aln(10))
      assert_eq!(db[&a], 1.0 / (1.0 * f64::ln(10.0)));
    }

    #[test]
    fn log2() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(2.0);
      let b = a.log2();
      assert_eq!(b.value(), f64::log2(2.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(aln(2))
      assert_eq!(db[&a], 1.0 / (2.0 * f64::ln(2.0)));
    }

    #[test]
    fn asin() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(0.5);
      let b = a.asin();
      assert_eq!(b.value(), f64::asin(0.5));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/sqrt(1-a^2)
      assert_eq!(db[&a], 1.0 / f64::sqrt(1.0 - 0.5 * 0.5));
    }

    #[test]
    fn acos() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(0.5);
      let b = a.acos();
      assert_eq!(b.value(), f64::acos(0.5));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = -1/sqrt(1-a^2)
      assert_eq!(db[&a], -1.0 / f64::sqrt(1.0 - 0.5 * 0.5));
    }

    #[test]
    fn atan() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.2);
      let b = a.atan();
      assert_eq!(b.value(), f64::atan(1.2));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(1+a^2)
      assert_eq!(db[&a], 1.0 / (1.0 + 1.2 * 1.2));
    }

    #[test]
    fn sinh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.3);
      let b = a.sinh();
      assert_eq!(b.value(), f64::sinh(1.3));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = cosh(a)
      assert_eq!(db[&a], f64::cosh(1.3));
    }

    #[test]
    fn cosh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.3);
      let b = a.cosh();
      assert_eq!(b.value(), f64::cosh(1.3));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = sinh(a)
      assert_eq!(db[&a], f64::sinh(1.3));
    }

    #[test]
    fn tanh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(0.8);
      let b = a.tanh();
      assert_eq!(b.value(), f64::tanh(0.8));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/cosh^2(a)
      assert_eq!(db[&a], 1.0 / (f64::cosh(0.8) * f64::cosh(0.8)));
    }

    #[test]
    fn asinh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = guard.var(1.5);
      let b = a.asinh();
      assert_eq!(b.value(), f64::asinh(1.5));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/sqrt(1+a^2)
      assert_eq!(db[&a], 1.0 / f64::sqrt(1.0 + 1.5 * 1.5));
    }

    #[test]
    fn acosh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(2.0);
      let b = a.acosh();
      assert_eq!(b.value(), f64::acosh(2.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/sqrt(a^2-1)
      assert_eq!(db[a], 1.0 / f64::sqrt(2.0 * 2.0 - 1.0));
    }

    #[test]
    fn atanh() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(0.0);
      let b = a.atanh();
      assert_eq!(b.value(), f64::atanh(0.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(1-a^2)
      assert_eq!(db[a], 1.0 / (1.0 - 0.0 * 0.0));
    }

    #[test]
    fn exp() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(1.3);
      let b = a.exp();
      assert_eq!(b.value(), f64::exp(1.3));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = exp(a)
      assert_eq!(db[a], f64::exp(1.3));
    }

    #[test]
    fn exp2() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(1.3);
      let b = a.exp2();
      assert_eq!(b.value(), f64::exp2(1.3));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = exp2(a)ln(2)
      assert_eq!(db[a], f64::exp2(1.3) * f64::ln(2.0));
    }

    #[test]
    fn pow() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(2.0);
      let b = &guard.var(3.0);
      let c = a.pow(b);
      assert_eq!(c.value(), f64::powf(2.0, 3.0));
      let grads = guard.collapse();
      let dc = grads.of(&c);
      // df/da = ba^(b-1)
      assert_eq!(dc[a], 3.0 * f64::powf(2.0, 3.0 - 1.0));
      // df/db = a^bln(a)
      assert_eq!(dc[b], f64::powf(2.0, 3.0) * f64::ln(2.0));
    }

    #[test]
    fn powf() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(2.0);
      let b = a.powf(3.0);
      assert_eq!(b.value(), f64::powf(2.0, 3.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 3a^(3-1)
      assert_eq!(db[a], 3.0 * f64::powf(2.0, 3.0 - 1.0));
    }

    #[test]
    fn sqrt() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(4.0);
      let b = a.sqrt();
      assert_eq!(b.value(), f64::sqrt(4.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(2sqrt(a))
      assert_eq!(db[a], 1.0 / (2.0 * f64::sqrt(4.0)));
    }

    #[test]
    fn cbrt() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(1.0);
      let b = a.cbrt();
      assert_eq!(b.value(), f64::cbrt(1.0));
      let grads = guard.collapse();
      let db = grads.of(&b);
      // df/da = 1/(3cbrt(a^2))
      assert_eq!(db[a], 1.0 / (3.0 * f64::cbrt(1.0 * 1.0)));
    }
  }

  mod tape {
    use super::*;

    #[test]
    fn guard() {
      let mut tape = Tape::new();
      let snap_addr = &tape.snap as *const _;
      let guard = tape.guard();
      let a = guard.var(2.0);
      let b = guard.var(3.0);
      assert_eq!(a.value, 2.0);
      assert_eq!(b.value, 3.0);
      assert_eq!(a.snap as *const _, snap_addr);
      assert_eq!(b.snap as *const _, snap_addr);
    }

    #[test]
    fn grads() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      // take reference right away since we dont really need to store the actual
      // var, and we only use references...
      let a = &guard.var(2.0);
      let b = &guard.var(3.0);
      let grads = guard.collapse();
      assert_eq!(a.value, 2.0);
      assert_eq!(b.value, 3.0);
      let da = grads.of(a);
      assert_eq!(da[a], 1.0);
      assert_eq!(da[b], 0.0);
    }

    #[test]
    fn reset() {
      let mut tape = Tape::new();
      { 
        let guard = tape.guard();
        let a = &guard.var(1.0);
        let grads = guard.collapse();
        let _da = grads.of(a);
      }
      assert_eq!(tape.snap.nodes, RefCell::new(Vec::new()));
    }
  }

  mod gradients {
    use super::*;

    #[test]
    fn of() {
      let mut tape = Tape::new();
      let guard = tape.guard();
      let a = &guard.var(5.0);
      let b = &guard.var(2.0);
      let c = &guard.var(1.0);
      let res = a.pow(b).sub(c.asinh().div(2.0)).add(f64::sin(1.0));
      assert_eq!(
        res.value(),
        f64::powf(5.0, 2.0) - f64::asinh(1.0) / 2.0 + f64::sin(1.0)
      );
      let grads = guard.collapse();
      let dres = grads.of(&res);
      let ga = dres[a]; // df/da
      let gb = dres[b]; // df/db
      let gc = dres[c]; // df/dc
      assert_eq!(ga, 2.0 * 5.0);
      assert_eq!(gb, 25.0 * 5.0f64.ln());
      assert_eq!(gc, -1.0 / (2.0 * 2.0f64.sqrt()));
    }
  }
}
