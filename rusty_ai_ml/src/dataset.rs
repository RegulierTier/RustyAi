//! Batching and simple column-wise normalization.
//!
//! TODO: streaming / mmap datasets for large corpora (e.g. code files on disk).

use rusty_ai_core::{ShapeError, Tensor, TensorError};

/// Iterates row-aligned mini-batches from two tensors sharing the same leading dimension `n`.
///
/// `x` and `y` must have shapes `[n, …]`; each batch preserves trailing dimensions.
pub struct BatchIter {
    n: usize,
    batch: usize,
    offset: usize,
    x: Tensor,
    y: Tensor,
}

impl BatchIter {
    /// `batch_size` is clamped to at least 1.
    pub fn new(x: Tensor, y: Tensor, batch_size: usize) -> Result<Self, TensorError> {
        if x.shape().first() != y.shape().first() {
            return Err(TensorError::Shape(ShapeError::IncompatibleBroadcast {
                left: x.shape().to_vec(),
                right: y.shape().to_vec(),
            }));
        }
        let n = *x.shape().first().unwrap();
        Ok(Self {
            n,
            batch: batch_size.max(1),
            offset: 0,
            x,
            y,
        })
    }
}

impl Iterator for BatchIter {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.n {
            return None;
        }
        let end = (self.offset + self.batch).min(self.n);
        let rows = end - self.offset;
        let d_in: usize = self.x.shape()[1..].iter().product();
        let d_out: usize = self.y.shape()[1..].iter().product();

        let mut xb = Vec::with_capacity(rows * d_in);
        let mut yb = Vec::with_capacity(rows * d_out);
        let xd = self.x.data();
        let yd = self.y.data();
        let stride_x = d_in;
        let stride_y = d_out;

        for r in self.offset..end {
            let ox = r * stride_x;
            xb.extend_from_slice(&xd[ox..ox + stride_x]);
            let oy = r * stride_y;
            yb.extend_from_slice(&yd[oy..oy + stride_y]);
        }

        let mut x_shape = vec![rows];
        x_shape.extend_from_slice(&self.x.shape()[1..]);
        let mut y_shape = vec![rows];
        y_shape.extend_from_slice(&self.y.shape()[1..]);

        self.offset = end;
        let xb_t = Tensor::from_vec(xb, x_shape).expect("batch x");
        let yb_t = Tensor::from_vec(yb, y_shape).expect("batch y");
        Some((xb_t, yb_t))
    }
}

/// Per-column z-score using training-set mean and standard deviation (population std).
///
/// Returns the normalized matrix and `(mean, std)` per column for applying the same transform
/// to validation data later (not implemented here).
pub fn normalize_columns(train: &Tensor) -> Result<(Tensor, Vec<f32>, Vec<f32>), TensorError> {
    let shape = train.shape();
    if shape.len() != 2 {
        return Err(TensorError::Shape(ShapeError::InvalidReshape {
            from: shape.to_vec(),
            to: vec![0, 0],
        }));
    }
    let n = shape[0];
    let d = shape[1];
    let data = train.data();
    let mut mean = vec![0.0f32; d];
    let mut std_dev = vec![0.0f32; d];
    for j in 0..d {
        let mut s = 0.0f32;
        for i in 0..n {
            s += data[i * d + j];
        }
        mean[j] = s / n as f32;
    }
    for j in 0..d {
        let mut s2 = 0.0f32;
        for i in 0..n {
            let z = data[i * d + j] - mean[j];
            s2 += z * z;
        }
        std_dev[j] = (s2 / n as f32).sqrt().max(1e-8);
    }
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            out[i * d + j] = (data[i * d + j] - mean[j]) / std_dev[j];
        }
    }
    Ok((Tensor::from_vec(out, shape.to_vec())?, mean, std_dev))
}
