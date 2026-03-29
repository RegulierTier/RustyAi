//! Batched affine transform for 3-D activations `(batch, seq, *)` — no autograd.
//!
//! FIXME: no fused bias+GEMM dispatch (relies on generic `matmul`).

use rusty_ai_core::{add, matmul, Tensor, TensorError};

/// Applies `y = x @ W + b` with `x` shaped `(batch, seq, in)`, `W` `(in, out)`, `b` broadcast to rows `(1, out)`.
///
/// Implemented by reshaping to `(batch * seq, in)`, one matrix multiply, bias add, reshape back.
pub fn linear_3d(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![],
            },
        ));
    }
    let batch = s[0];
    let seq = s[1];
    let in_f = s[2];
    let out_f = w.shape()[1];
    let flat = batch * seq;
    let x2 = x.reshape(&[flat, in_f])?;
    let y = matmul(&x2, w)?;
    let y = add(&y, b)?;
    y.reshape(&[batch, seq, out_f])
}
