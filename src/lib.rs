use std::cell::{RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Index;

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
struct NodeIndex(u64);

impl NodeIndex {
  const LEVEL_MASK: u64 = (1 << 48) - 1;

  fn new(level: u8, index: u64) -> Self {
    let level = level as u64;
    Self((level << 48) | (index & Self::LEVEL_MASK))
  }

  fn level(&self) -> u8 {
    (self.0 >> 48) as u8
  }

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

pub struct Var<'tape, 'scope> {
  value: f64,
  tape: &'tape Tape,
  index: NodeIndex,
  phantom: PhantomData<&'scope ()>,
}

impl<'tape, 'scope> Var<'tape, 'scope> {
  pub fn value(&self) -> f64 {
    self.value
  }

  fn to_predecessor(&self, grad: f64) -> Predecessor {
    Predecessor {
      grad,
      index: self.index,
    }
  }

  // Instead of using with_frames_mut, we simply grab the current frame.
  fn add_node(&self, pred_a: Predecessor, pred_b: Predecessor) -> NodeIndex {
    self.tape.inner.current_frame_mut().add_node(pred_a, pred_b)
  }

  pub fn reciprocal(&self) -> Var<'tape, 'scope> {
    let v = self.value();
    let pred_a = self.to_predecessor(-1.0 / (v * v));
    let pred_b = self.to_predecessor(0.0);
    Self {
      value: 1.0 / v,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  pub fn add<'other>(&self, other: &Var<'tape, 'other>) -> Var<'tape, '_> {
    let v = self.value();
    let ov = other.value();
    let pred_a = self.to_predecessor(1.0);
    let pred_b = other.to_predecessor(1.0);
    Var {
      value: v + ov,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }

  pub fn mul<'other>(&self, other: &Var<'tape, 'other>) -> Var<'tape, '_> {
    let v = self.value();
    let ov = other.value();
    let pred_a = self.to_predecessor(ov);
    let pred_b = other.to_predecessor(v);
    Var {
      value: v * ov,
      index: self.add_node(pred_a, pred_b),
      tape: self.tape,
      phantom: PhantomData,
    }
  }
}

#[derive(Default, Clone)]
struct Frame {
  level: u8,
  nodes: Vec<Node>,
}

impl Frame {
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

struct TapeInner {
  frames: RefCell<Frames>,
}

impl Default for TapeInner {
  fn default() -> Self {
    Self {
      frames: RefCell::new(Frames::default()),
    }
  }
}

impl TapeInner {
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
  fn push_frame(&self, frame: Frame) {
    self.inner.frames.borrow_mut().stack.push(frame);
  }

  fn pop_frame(&self) {
    self.inner.frames.borrow_mut().stack.pop();
  }

  pub fn with_scope<'tape, F>(&'tape self, level: u8, f: F)
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>),
  {
    self.push_frame(Frame {
      level,
      nodes: Vec::new(),
    });

    f(Guard {
      level,
      tape: self,
      phantom: PhantomData,
    });

    self.pop_frame();
  }

  pub fn scope<'tape, F>(&'tape mut self, f: F)
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>),
  {
    self.with_scope(0, f);
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
  pub fn scope<F>(&mut self, f: F)
  where
    F: for<'inner> FnOnce(Guard<'tape, 'inner, Unlocked>),
  {
    self.tape.with_scope(self.level + 1, f);
  }

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

impl<'tape, 'scope> Gradients<'tape, 'scope> {
  pub fn of(&self, var: &Var<'tape, 'scope>) -> Deltas<'scope> {
    let subgraph = self.topological_subgraph_of(var);
    let mut deltas = HashMap::new();
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

  fn topological_subgraph_of(&self, var: &Var<'tape, 'scope>) -> Vec<IndexNode> {
    let nodes = self.tape.inner.frames.borrow();
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![(var.index, false)];

    while let Some((node_index, children_processed)) = stack.pop() {
      if children_processed {
        if let Some(node) = nodes.get_node(node_index) {
          result.push((node, node_index));
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
}

#[derive(Debug)]
pub struct Deltas<'scope> {
  deltas: HashMap<NodeIndex, f64>,
  phantom: PhantomData<&'scope ()>,
}

impl<'tape, 'scope> Index<&Var<'tape, 'scope>> for Deltas<'scope> {
  type Output = f64;
  fn index(&self, var: &Var<'tape, 'scope>) -> &Self::Output {
    self.deltas.get(&var.index).unwrap_or(&0.0)
  }
}

