//!
//! # auto
//!
//! ## Core API
//!
//! This library provides a safe, efficient implementation of reverse-mode automatic
//! differentiation using Rust's type system to enforce memory safety and prevent API
//! misuse.
//!
//! The main entry point is [`Tape::scope`], which creates a computational scope where
//! variables can be created and operations performed through guards
//!

pub use lib_auto_core::*;

#[cfg(feature = "scalar")]
pub use lib_auto_scalar as scalar;

#[cfg(feature = "matrix")]
pub use lib_auto_matrix as matrix;
