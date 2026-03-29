//! FP8 E4M3 via Candle’s [`DType::F8E4M3`](candle_core::DType) — meaningful on CUDA when built with `--features cuda`.

use candle_core::{DType, Device, Tensor};

/// FP8 conversion error.
#[derive(Debug)]
pub struct Fp8Error(pub candle_core::Error);

impl std::fmt::Display for Fp8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Fp8Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl From<candle_core::Error> for Fp8Error {
    fn from(e: candle_core::Error) -> Self {
        Fp8Error(e)
    }
}

/// Cast `f32` buffer to FP8 E4M3 on `device`, then back to `f32` (round-trip for quantization tests).
pub fn f32_tensor_to_f8e4m3(
    device: &Device,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, Fp8Error> {
    let t = Tensor::from_vec(data.to_vec(), shape, device)?;
    Ok(t.to_dtype(DType::F8E4M3)?)
}

/// Cast FP8 tensor to `f32` flat buffer (row-major).
pub fn f8e4m3_tensor_to_f32(t: &Tensor) -> Result<Vec<f32>, Fp8Error> {
    let f = t.to_dtype(DType::F32)?;
    Ok(f.flatten_all()?.to_vec1::<f32>()?)
}
