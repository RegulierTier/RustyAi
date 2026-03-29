//! Fully connected (affine) layer for autograd training.

use std::rc::Rc;

use rusty_ai_autograd::Variable;
use rusty_ai_core::TensorError;

use crate::init::{glorot_uniform, zeros_bias};

/// Dense layer: `y = x @ W + b` with `x` shaped `(batch, in_features)`, `W` `(in, out)`, `b` `(1, out)`.
///
/// Weights and biases are shared [`Rc<Variable>`] nodes so the same parameters are updated
/// across forward passes when optimizers call [`Variable::set_data`].
pub struct Linear {
    pub weight: Rc<Variable>,
    pub bias: Rc<Variable>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Xavier-uniform weights and zero bias. `seed` advances deterministically.
    pub fn new(
        in_features: usize,
        out_features: usize,
        seed: &mut u32,
    ) -> Result<Self, TensorError> {
        let w = glorot_uniform(in_features, out_features, seed)?;
        let b = zeros_bias(out_features)?;
        Ok(Self {
            weight: Variable::leaf(w),
            bias: Variable::leaf(b),
            in_features,
            out_features,
        })
    }

    /// Forward: matmul then broadcast bias add.
    pub fn forward(&self, x: &Rc<Variable>) -> Result<Rc<Variable>, TensorError> {
        let h = Variable::matmul(x, &self.weight)?;
        Variable::bias_add(&h, &self.bias)
    }
}
