//! Reverse-mode automatic differentiation for [`rusty_ai_core::Tensor`].
//!
//! [`Variable`] wraps a tensor and records how it was produced (`Op`). Calling [`backward`]
//! on a scalar loss node propagates gradients backward through the graph and accumulates
//! them in each leaf's `grad` field. Optimizers then read those gradients and update weights.
//!
//! Includes ops for small MLPs and for a trainable Mini-GPT (GELU, layer norm, softmax,
//! embedding gather, causal attention, next-token cross-entropy).
//!
//! Use [`no_grad`] when you only need forward evaluation (e.g. pure inference) to avoid
//! building a graph.
//!
//! FIXME: graph retention until `backward` — no implicit freeing of intermediate nodes (watch memory on huge graphs).

mod context;
mod grad_utils;
mod structural;
mod variable;

pub use context::{grad_enabled, no_grad, set_grad_enabled};
pub use variable::{backward, Variable};

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _second_order_stub() {
        unimplemented!("TODO: higher-order / Hessian-vector products (not on roadmap)");
    }
}
