//! Optional **Candle** backend for RustyAi: device selection, `f32` matmul on CPU or CUDA,
//! FP8 (E4M3) conversion helpers when built with `--features cuda`, and gradient **all-reduce**
//! primitives for data-parallel training (reference CPU implementation plus NCCL hook when enabled).
//!
//! Enable GPU: `rusty_ai_backend_candle = { path = "...", features = ["cuda"] }`.
//!
//! FIXME: not wired end-to-end into `TrainableMiniGpt` — matmul/FP8 are standalone helpers.

mod device;
mod distributed;
mod fp8;
mod matmul;

pub use device::{default_device, BackendDevice};
pub use distributed::{all_reduce_mean_cpu, AllReduceError};
pub use fp8::{f32_tensor_to_f8e4m3, f8e4m3_tensor_to_f32, Fp8Error};
pub use matmul::{matmul_f32, MatmulError};

#[cfg(test)]
mod planned_unimplemented_markers {
    #[allow(dead_code)]
    fn _end_to_end_candle_train_stub() {
        unimplemented!("TODO: Candle-backed training loop for MiniGpt");
    }
}
