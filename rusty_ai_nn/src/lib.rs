//! Neural network building blocks: layers and activation functions on [`rusty_ai_core::Tensor`]
//! and [`rusty_ai_autograd::Variable`].
//!
//! TODO: Conv1d/Conv2d stubs for non-LLM experiments.

mod activation;
mod init;
mod linear;

pub use activation::{gelu, layer_norm, layer_norm_affine};
pub use init::{glorot_uniform, ones_scale, uniform, zeros_bias};
pub use linear::Linear;

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _dropout_stub() {
        unimplemented!("TODO: dropout / stochastic depth for regularization");
    }
}
