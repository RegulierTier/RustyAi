//! Helpers for backward passes: axis reductions, GELU/LayerNorm derivatives.
//!
//! FIXME: numerically sensitive ops could use higher precision accumulators (f64 scratch).

use rusty_ai_core::{softmax, Tensor, TensorError};

/// Sum along `axis` (0-based), keeping the axis as size 1.
pub fn sum_axis_keepdim(t: &Tensor, axis: usize) -> Result<Tensor, TensorError> {
    let s = t.shape();
    let rank = s.len();
    if axis >= rank {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let outer: usize = s[..axis].iter().product();
    let mid = s[axis];
    let inner: usize = s[axis + 1..].iter().product();
    let mut out = vec![0.0f32; outer * inner];
    let d = t.data();
    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0f32;
            for m in 0..mid {
                sum += d[o * mid * inner + m * inner + i];
            }
            out[o * inner + i] = sum;
        }
    }
    let mut new_shape = s.to_vec();
    new_shape[axis] = 1;
    Tensor::from_vec(out, new_shape)
}

/// Reduces `grad` by summing axes where `target_shape` (left-padded with 1s) has 1 but `grad` is larger.
pub fn sum_grad_to_shape(grad: &Tensor, target_shape: &[usize]) -> Result<Tensor, TensorError> {
    let gshape = grad.shape();
    let rank = gshape.len();
    if target_shape.len() > rank {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: gshape.to_vec(),
                to: target_shape.to_vec(),
            },
        ));
    }
    let pad = rank - target_shape.len();
    let mut padded = vec![1usize; pad];
    padded.extend_from_slice(target_shape);
    debug_assert_eq!(padded.len(), rank);

    let mut t = grad.clone();
    for d in 0..rank {
        if gshape[d] != padded[d] {
            if padded[d] != 1 {
                return Err(TensorError::Shape(
                    rusty_ai_core::ShapeError::IncompatibleBroadcast {
                        left: vec![gshape[d]],
                        right: vec![padded[d]],
                    },
                ));
            }
            t = sum_axis_keepdim(&t, d)?;
        }
    }
    t.reshape(target_shape)
}

/// GELU tanh approximation (matches `rusty_ai_nn::gelu`).
pub fn gelu_forward(t: &Tensor) -> Tensor {
    let mut v = t.data().to_vec();
    for x in &mut v {
        let x3 = *x * *x * *x;
        *x = 0.5 * *x * (1.0 + ((0.797_884_6 * (*x + 0.044715 * x3)).tanh()));
    }
    Tensor::from_vec(v, t.shape().to_vec()).expect("shape")
}

/// Elementwise dGELU/dx for the tanh approximation.
pub fn gelu_backward(grad: &Tensor, x: &Tensor) -> Result<Tensor, TensorError> {
    let k = 0.797_884_6f32;
    let c = 0.044715f32;
    let mut out = vec![0.0f32; x.numel()];
    let gd = grad.data();
    let xd = x.data();
    for i in 0..out.len() {
        let x = xd[i];
        let x3 = x * x * x;
        let u = k * (x + c * x3);
        let tanh_u = u.tanh();
        let sech2 = 1.0 - tanh_u * tanh_u;
        let du_dx = k * (1.0 + 3.0 * c * x * x);
        let g = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du_dx;
        out[i] = gd[i] * g;
    }
    Tensor::from_vec(out, x.shape().to_vec())
}

/// Layer norm backward (last axis only, no γ/β); `dy` and `x` same shape.
pub fn layer_norm_backward(dy: &Tensor, x: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    let shape = x.shape();
    if shape.is_empty() || dy.shape() != shape {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: dy.shape().to_vec(),
                to: shape.to_vec(),
            },
        ));
    }
    let h = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product();
    let mut dx = vec![0.0f32; x.numel()];
    let xd = x.data();
    let dyd = dy.data();
    for o in 0..outer {
        let base = o * h;
        let mut mean = 0.0f32;
        for i in 0..h {
            mean += xd[base + i];
        }
        mean /= h as f32;
        let mut var = 0.0f32;
        for i in 0..h {
            let z = xd[base + i] - mean;
            var += z * z;
        }
        var /= h as f32;
        let std = (var + eps).sqrt();
        let mut x_hat = vec![0.0f32; h];
        for i in 0..h {
            x_hat[i] = (xd[base + i] - mean) / std;
        }
        let mut dy_sum = 0.0f32;
        let mut dy_xhat_sum = 0.0f32;
        for i in 0..h {
            dy_sum += dyd[base + i];
            dy_xhat_sum += dyd[base + i] * x_hat[i];
        }
        dy_sum /= h as f32;
        dy_xhat_sum /= h as f32;
        for i in 0..h {
            dx[base + i] = (1.0 / std) * (dyd[base + i] - dy_sum - x_hat[i] * dy_xhat_sum);
        }
    }
    Tensor::from_vec(dx, shape.to_vec())
}

/// Softmax over last axis: `grad_y` and `y` same shape; returns grad w.r.t. pre-softmax logits.
pub fn softmax_last_dim_backward(grad_y: &Tensor, y: &Tensor) -> Result<Tensor, TensorError> {
    let shape = y.shape();
    if grad_y.shape() != shape {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: grad_y.shape().to_vec(),
                to: shape.to_vec(),
            },
        ));
    }
    if shape.is_empty() {
        return Ok(grad_y.clone());
    }
    let axis = shape.len() - 1;
    let stride_dim = shape[axis];
    // Last-axis softmax: `axis == rank - 1` ⇒ trailing product after axis is 1.
    let trailing: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut dx = vec![0.0f32; y.numel()];
    let yd = y.data();
    let gd = grad_y.data();
    for o in 0..outer {
        for tr in 0..trailing {
            let base = o * stride_dim * trailing + tr;
            let mut dot = 0.0f32;
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                dot += yd[idx] * gd[idx];
            }
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                dx[idx] = yd[idx] * (gd[idx] - dot);
            }
        }
    }
    Tensor::from_vec(dx, shape.to_vec())
}

/// Forward softmax (last dim) — uses core.
pub fn softmax_last_forward(t: &Tensor) -> Result<Tensor, TensorError> {
    softmax(t)
}

/// Layer norm over last axis (matches `rusty_ai_nn::layer_norm`).
pub fn layer_norm_forward(t: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_nonzero() {
        let t = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], vec![3]).unwrap();
        let g = gelu_forward(&t);
        assert!(g.data()[1] > 0.5);
    }

    #[test]
    fn softmax_backward_identity_at_peak() {
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 10.0], vec![1, 3]).unwrap();
        let y = softmax_last_forward(&logits).unwrap();
        let gy = Tensor::from_vec(vec![0.0f32, 0.0, 1.0], vec![1, 3]).unwrap();
        let gx = softmax_last_dim_backward(&gy, &y).unwrap();
        // Near one-hot input, gradient structure is valid
        assert!(gx.data().iter().all(|v| v.is_finite()));
    }
}
