//! Batched 2D matrix multiply `C = A @ B` with shapes `(m, k)` and `(k, n)`.
//!
//! TODO: autotune batch vs `rusty_ai_core::matmul` for hybrid CPU/Candle pipelines.

use candle_core::{Device, Tensor};

/// Error from Candle matmul.
#[derive(Debug)]
pub struct MatmulError(pub candle_core::Error);

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MatmulError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl From<candle_core::Error> for MatmulError {
    fn from(e: candle_core::Error) -> Self {
        MatmulError(e)
    }
}

/// `a` is `(m, k)`, `b` is `(k, n)`; returns flattened row-major `f32` of shape `(m, n)`.
pub fn matmul_f32(
    device: &Device,
    a: &[f32],
    a_shape: [usize; 2],
    b: &[f32],
    b_shape: [usize; 2],
) -> Result<Vec<f32>, MatmulError> {
    let (m, k) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k != k2 {
        return Err(candle_core::Error::Msg(format!("inner dims mismatch: {k} vs {k2}")).into());
    }
    let ta = Tensor::from_vec(a.to_vec(), &[m, k], device)?;
    let tb = Tensor::from_vec(b.to_vec(), &[k2, n], device)?;
    let out = ta.matmul(&tb)?;
    let v = out.flatten_all()?.to_vec1::<f32>()?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn matmul_matches_small_example() {
        let dev = Device::Cpu;
        let a = vec![1f32, 2.0, 3.0, 4.0];
        let b = vec![10f32, 20.0, 30.0, 40.0];
        let c = matmul_f32(&dev, &a, [2, 2], &b, [2, 2]).unwrap();
        assert_eq!(c, vec![70.0, 100.0, 150.0, 220.0]);
    }
}
