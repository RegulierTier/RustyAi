//! Data-parallel gradient sync: **mean** of identical-length buffers across replicas.
//!
//! - [`all_reduce_mean_cpu`] is a pure reference implementation (single-process, many buffers).
//! - Multi-GPU NCCL is available in Candle when built with `features = ["nccl"]`; production training
//!   should use a process group that calls into NCCL `AllReduce` — this crate documents the **math**
//!   (`sum / world_size`) that must match distributed steps.
//!
//! TODO: thin async wrapper around real NCCL process groups when integrating with a trainer.

/// Error if replica buffers disagree on length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllReduceError {
    EmptyReplicas,
    MismatchedLen { expected: usize, got: usize },
}

impl std::fmt::Display for AllReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllReduceError::EmptyReplicas => write!(f, "no replica buffers"),
            AllReduceError::MismatchedLen { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for AllReduceError {}

/// Elementwise mean across `replicas.len()` buffers (same layout). Single-process reference for tests.
pub fn all_reduce_mean_cpu(replicas: &[Vec<f32>]) -> Result<Vec<f32>, AllReduceError> {
    let n = replicas.len();
    if n == 0 {
        return Err(AllReduceError::EmptyReplicas);
    }
    let len = replicas[0].len();
    for r in replicas.iter().skip(1) {
        if r.len() != len {
            return Err(AllReduceError::MismatchedLen {
                expected: len,
                got: r.len(),
            });
        }
    }
    let mut out = vec![0f32; len];
    for r in replicas {
        for (i, x) in r.iter().enumerate() {
            out[i] += x;
        }
    }
    let inv = 1.0f32 / n as f32;
    for x in &mut out {
        *x *= inv;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_two_vectors() {
        let a = vec![1f32, 2.0, 3.0];
        let b = vec![3f32, 2.0, 1.0];
        let m = all_reduce_mean_cpu(&[a, b]).unwrap();
        assert_eq!(m, vec![2.0, 2.0, 2.0]);
    }
}

#[cfg(feature = "nccl")]
mod nccl_note {
    //! When `nccl` is enabled, Candle links `cudarc` with NCCL. Use the upstream Candle examples
    //! for multi-process launch (`torchrun`-style env `RANK`, `WORLD_SIZE`) and wire all-reduce
    //! on gradient tensors there; [`super::all_reduce_mean_cpu`] remains the semantic reference.
}
