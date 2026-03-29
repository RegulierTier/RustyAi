//! Core tensor types and CPU numerical operations for RustyAi.
//!
//! This crate provides:
//! - [`Tensor`]: contiguous `f32` storage in row-major (C) order.
//! - NumPy-style [`broadcast_shapes`] for elementwise binary ops.
//! - Linear algebra via [`matmul`] (2D and batched 3D) using the `matrixmultiply` crate.
//! - Activations and reductions used by autograd, neural nets, and LLM code.
//!
//! TODO: optional BLIS/OpenBLAS linkage behind a feature for very large GEMMs.

mod dtype;
mod error;
mod ops;
mod shape;
mod tensor;

pub use dtype::DType;
pub use error::{ShapeError, TensorError};

pub use ops::{
    add, div, log_softmax, matmul, mse, mul, relu, softmax, sqrt, sub, sum_axis_0, transpose_2d,
    transpose_batched_last2,
};
pub use shape::broadcast_shapes;
pub use tensor::Tensor;

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _sparse_tensor_stub() {
        unimplemented!("TODO: sparse / structured tensors (not implemented)");
    }
}
