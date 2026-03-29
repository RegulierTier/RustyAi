//! Random initialization helpers (deterministic given a `u32` seed).
//!
//! Uses a simple linear congruential generator (LCG) — not cryptographically secure,
//! but sufficient for weight initialization and reproducible tests.

use rusty_ai_core::{DType, Tensor, TensorError};

/// One LCG step; returns a value in approximately `[0, 1)`.
fn lcg_next(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7fff) as f32 / 32768.0
}

/// Uniform random tensor in `[-limit, limit]` with the given shape.
pub fn uniform(shape: &[usize], limit: f32, seed: &mut u32) -> Result<Tensor, TensorError> {
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push((lcg_next(seed) * 2.0 - 1.0) * limit);
    }
    Tensor::from_vec(v, shape.to_vec())
}

/// Glorot/Xavier uniform bounds for a weight matrix of shape `(in_f, out_f)`.
///
/// See Glorot & Bengio (2010): variance scales with `2 / (fan_in + fan_out)`.
pub fn glorot_uniform(in_f: usize, out_f: usize, seed: &mut u32) -> Result<Tensor, TensorError> {
    let limit = (6.0f32 / (in_f + out_f) as f32).sqrt();
    uniform(&[in_f, out_f], limit, seed)
}

/// Row bias shaped `(1, out)` filled with zeros (common default before training).
pub fn zeros_bias(out: usize) -> Result<Tensor, TensorError> {
    Tensor::zeros(&[1, out], DType::F32)
}
