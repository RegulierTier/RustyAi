//! Tensor-only activations and normalization (no autograd hooks here).

use rusty_ai_core::{add, mul, Tensor, TensorError};

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

/// LayerNorm over the last axis with learnable affine parameters (γ, β), matching
/// `torch.nn.LayerNorm(..., elementwise_affine=True)` when `gamma`/`beta` are `(1, H)`.
///
/// `y = gamma * layer_norm(x) + beta` with broadcasting.
pub fn layer_norm_affine(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<Tensor, TensorError> {
    let n = layer_norm(x, eps)?;
    let scaled = mul(&n, gamma)?;
    add(&scaled, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::{ones_scale, zeros_bias};

    #[test]
    fn layer_norm_affine_identity_matches_layer_norm() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 1.0], vec![2, 3]).unwrap();
        let g = ones_scale(3).unwrap();
        let b = zeros_bias(3).unwrap();
        let y = layer_norm_affine(&x, &g, &b, 1e-5).unwrap();
        let ln = layer_norm(&x, 1e-5).unwrap();
        assert_eq!(y.data(), ln.data());
    }
}
