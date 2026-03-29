//! Tensor-only activations and normalization (no autograd hooks here).

use rusty_ai_core::{Tensor, TensorError};

/// Gaussian Error Linear Unit — tanh approximation from Hendrycks & Gimpel.
///
/// Used in Transformer FFNs; operates elementwise on any shape.
pub fn gelu(t: &Tensor) -> Tensor {
    let mut v = t.data().to_vec();
    for x in &mut v {
        let x3 = *x * *x * *x;
        *x = 0.5 * *x * (1.0 + ((0.797_884_6 * (*x + 0.044715 * x3)).tanh()));
    }
    Tensor::from_vec(v, t.shape().to_vec()).expect("shape")
}

/// Layer normalization over the **last** dimension only: normalizes each last-axis slice to
/// mean 0 and variance 1 (with `eps`), **without** learnable scale/shift (γ/β).
///
/// Input rank ≥ 1; shape `[..., H]` is preserved.
pub fn layer_norm(t: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    let shape = t.shape();
    if shape.is_empty() {
        return Ok(t.clone());
    }
    let h = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product();
    let mut out = vec![0.0f32; t.numel()];
    let d = t.data();

    for o in 0..outer {
        let base = o * h;
        let mut sum = 0.0f32;
        for i in 0..h {
            sum += d[base + i];
        }
        let mean = sum / h as f32;
        let mut var = 0.0f32;
        for i in 0..h {
            let z = d[base + i] - mean;
            var += z * z;
        }
        var /= h as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..h {
            out[base + i] = (d[base + i] - mean) * inv_std;
        }
    }

    Tensor::from_vec(out, shape.to_vec())
}
