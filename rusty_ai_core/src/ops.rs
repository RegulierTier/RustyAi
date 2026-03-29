//! Elementwise and linear-algebra operations on [`Tensor`].
//!
//! Binary ops delegate to [`Tensor::broadcast_binary`]. Matrix multiply uses the
//! `matrixmultiply` crate (`sgemm`) for CPU performance. Softmax applies along a single axis
//! with subtract-max stabilization to reduce overflow in exponentials.
//!
//! FIXME: no SIMD-specific fast paths beyond `matrixmultiply` / scalar loops.

use crate::error::{ShapeError, TensorError};
use crate::tensor::Tensor;

// --- Elementwise (broadcasting) -------------------------------------------------

/// Elementwise addition with broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    a.broadcast_binary(b, |x, y| x + y)
}

/// Elementwise subtraction with broadcasting.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    a.broadcast_binary(b, |x, y| x - y)
}

/// Elementwise multiplication with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    a.broadcast_binary(b, |x, y| x * y)
}

/// Elementwise division with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    a.broadcast_binary(b, |x, y| x / y)
}

// --- Transpose and reductions ---------------------------------------------------

/// Sums a 2-D tensor along axis 0: `(batch, n) -> (1, n)`.
///
/// Used in autograd when backpropagating through a broadcast bias add.
pub fn sum_axis_0(t: &Tensor) -> Result<Tensor, TensorError> {
    let s = t.shape();
    if s.len() != 2 {
        return Err(TensorError::Shape(ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![1, 0],
        }));
    }
    let batch = s[0];
    let n = s[1];
    let mut v = vec![0.0f32; n];
    let d = t.data();
    for i in 0..batch {
        for j in 0..n {
            v[j] += d[i * n + j];
        }
    }
    Tensor::from_vec(v, vec![1, n])
}

/// Swaps the two dimensions of a rank-2 tensor: `(m, n) -> (n, m)`.
pub fn transpose_2d(t: &Tensor) -> Result<Tensor, TensorError> {
    let s = t.shape();
    if s.len() != 2 {
        return Err(TensorError::Shape(ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let m = s[0];
    let n = s[1];
    let mut out = vec![0.0f32; m * n];
    let d = t.data();
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = d[i * n + j];
        }
    }
    Tensor::from_vec(out, vec![n, m])
}

/// Swaps the last two dimensions for each batch slice: `(B, M, N) -> (B, N, M)`.
///
/// Used to form `K^T` per batch in attention: `(B, L, d_h) @ (B, d_h, L)`.
pub fn transpose_batched_last2(t: &Tensor) -> Result<Tensor, TensorError> {
    let s = t.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(ShapeError::InvalidReshape {
            from: s.to_vec(),
            to: vec![],
        }));
    }
    let b = s[0];
    let m = s[1];
    let n = s[2];
    let mut out = vec![0.0f32; b * m * n];
    let d = t.data();
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                out[bi * n * m + j * m + i] = d[bi * m * n + i * n + j];
            }
        }
    }
    Tensor::from_vec(out, vec![b, n, m])
}

// --- Matrix multiplication ------------------------------------------------------

/// Matrix multiply: **2-D** `(m,k) @ (k,n)` or **batched 3-D** `(B,M,K) @ (B,K,N)`.
///
/// For batched mode, `B` must match and the contraction is over `K`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    match (a.shape().len(), b.shape().len()) {
        (2, 2) => matmul_2d(a, b),
        (3, 3) => matmul_batched(a, b),
        _ => Err(TensorError::Shape(ShapeError::MatmulIncompatible {
            left: a.shape().to_vec(),
            right: b.shape().to_vec(),
        })),
    }
}

fn matmul_2d(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let sa = a.shape();
    let sb = b.shape();
    if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
        return Err(TensorError::Shape(ShapeError::MatmulIncompatible {
            left: sa.to_vec(),
            right: sb.to_vec(),
        }));
    }
    let m = sa[0];
    let k = sa[1];
    let n = sb[1];
    let mut c = vec![0.0f32; m * n];
    // Row-major A (m×k): row stride k, col stride 1 — matches sgemm expectations.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.data().as_ptr(),
            k as isize,
            1,
            b.data().as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    Tensor::from_vec(c, vec![m, n])
}

fn matmul_batched(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let sa = a.shape();
    let sb = b.shape();
    if sa[0] != sb[0] || sa[2] != sb[1] {
        return Err(TensorError::Shape(ShapeError::MatmulIncompatible {
            left: sa.to_vec(),
            right: sb.to_vec(),
        }));
    }
    let batch = sa[0];
    let m = sa[1];
    let k = sa[2];
    let n = sb[2];
    let mut out = Vec::with_capacity(batch * m * n);
    let ad = a.data();
    let bd = b.data();
    for bi in 0..batch {
        let a_off = bi * (m * k);
        let b_off = bi * (k * n);
        let mut c = vec![0.0f32; m * n];
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                ad[a_off..].as_ptr(),
                k as isize,
                1,
                bd[b_off..].as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        out.extend(c);
    }
    Tensor::from_vec(out, vec![batch, m, n])
}

// --- Softmax --------------------------------------------------------------------

/// Softmax over the **last** dimension (standard for logits shaped `[..., classes]`).
///
/// Stabilized by subtracting the per-slice maximum before `exp`.
pub fn softmax(t: &Tensor) -> Result<Tensor, TensorError> {
    softmax_axis(t, t.shape().len().saturating_sub(1))
}

fn softmax_axis(t: &Tensor, axis: usize) -> Result<Tensor, TensorError> {
    let shape = t.shape();
    if shape.is_empty() {
        return Ok(t.clone());
    }
    let rank = shape.len();
    if axis >= rank {
        return Err(TensorError::Shape(ShapeError::InvalidReshape {
            from: shape.to_vec(),
            to: shape.to_vec(),
        }));
    }

    let mut out = t.data().to_vec();
    // For axis `a`, iterate "outer" blocks, then "trailing" after axis, then along axis.
    let trailing: usize = shape[axis + 1..].iter().product();
    let stride_dim = shape[axis];
    let outer: usize = shape[..axis].iter().product();

    for o in 0..outer {
        for tr in 0..trailing {
            let base = o * stride_dim * trailing + tr;
            let mut max = f32::NEG_INFINITY;
            for i in 0..stride_dim {
                let v = out[base + i * trailing];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                let e = (out[idx] - max).exp();
                out[idx] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                out[idx] *= inv;
            }
        }
    }

    Tensor::from_vec(out, shape.to_vec())
}

/// Natural logarithm of softmax over the last axis: `log(softmax(x))`.
///
/// More stable than `log(softmax(x))` computed separately.
pub fn log_softmax(t: &Tensor) -> Result<Tensor, TensorError> {
    let shape = t.shape();
    if shape.is_empty() {
        return Ok(t.clone());
    }
    let axis = shape.len() - 1;
    let mut out = t.data().to_vec();
    let trailing: usize = if axis + 1 < shape.len() {
        shape[axis + 1..].iter().product()
    } else {
        1
    };
    let stride_dim = shape[axis];
    let outer: usize = shape[..axis].iter().product();

    for o in 0..outer {
        for tr in 0..trailing {
            let base = o * stride_dim * trailing + tr;
            let mut max = f32::NEG_INFINITY;
            for i in 0..stride_dim {
                let v = out[base + i * trailing];
                if v > max {
                    max = v;
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                let e = (out[idx] - max).exp();
                sum_exp += e;
            }
            let log_sum = sum_exp.ln();
            for i in 0..stride_dim {
                let idx = base + i * trailing;
                out[idx] = out[idx] - max - log_sum;
            }
        }
    }

    Tensor::from_vec(out, shape.to_vec())
}

/// Elementwise square root (inputs should be non-negative for real results).
pub fn sqrt(t: &Tensor) -> Tensor {
    let mut v = t.data().to_vec();
    for x in &mut v {
        *x = x.sqrt();
    }
    Tensor::from_vec(v, t.shape().to_vec()).expect("shape")
}

/// Rectified linear unit applied elementwise.
pub fn relu(t: &Tensor) -> Tensor {
    let mut out = t.data().to_vec();
    for x in &mut out {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    Tensor::from_vec(out, t.shape().to_vec()).expect("same shape")
}

/// Mean squared error: `mean((a - b)^2)` as a **scalar** 0-D tensor.
///
/// Requires identical shapes (no broadcasting).
pub fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.shape() != b.shape() {
        return Err(TensorError::Shape(ShapeError::IncompatibleBroadcast {
            left: a.shape().to_vec(),
            right: b.shape().to_vec(),
        }));
    }
    let mut s = 0.0f32;
    for (x, y) in a.data().iter().zip(b.data().iter()) {
        let d = x - y;
        s += d * d;
    }
    let n = a.numel().max(1) as f32;
    Ok(Tensor::scalar(s / n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn softmax_last_dim() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let s = softmax(&t).unwrap();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
