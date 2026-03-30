//! Optional **Candle** backend for RustyAi: device selection, `f32` matmul on CPU or CUDA,
//! FP8 (E4M3) conversion helpers when built with `--features cuda`, and gradient **all-reduce**
//! primitives for data-parallel training (reference CPU implementation plus NCCL hook when enabled).
//!
//! Enable GPU: `rusty_ai_backend_candle = { path = "...", features = ["cuda"] }`.
//!
//! **Training:** There is **no** shared autograd graph with [`TrainableMiniGpt`](https://docs.rs/rusty_ai_llm/latest/rusty_ai_llm/struct.TrainableMiniGpt.html)
//! (CPU `Variable` + `rusty_ai_core::Tensor`). This crate exposes **standalone** ops (matmul, FP8, all-reduce helpers).
//! For LM training, use the CPU path in `rusty_ai_llm` / `rusty_ai`; use Candle for experiments or forward-only weight checks.
//!
//! **Tier 1 (E2E light):** optional `#[ignore]`-Tests vergleichen Teil-Forwards (z. B. `lm_head`-Linear) mit
//! `rusty_ai_core` / `MiniGpt`-Gewichten — kein gemeinsamer Backward.
//! **Tier 2:** ein vollständiges Training nur in Candle wäre ein eigenes Milestone (eigenes Modell + Optimizer).

mod device;
mod distributed;
mod fp8;
mod matmul;

pub use device::{default_device, BackendDevice};
pub use distributed::{all_reduce_mean_cpu, AllReduceError};
pub use fp8::{f32_tensor_to_f8e4m3, f8e4m3_tensor_to_f32, Fp8Error};
pub use matmul::{matmul_f32, MatmulError};

#[cfg(test)]
mod minigpt_weight_bridge {
    //! Optional: compare one `MiniGpt` weight matrix matmul with Candle (forward-only).

    use candle_core::Device;

    use rusty_ai_core::{matmul, Tensor};
    use rusty_ai_llm::{MiniGpt, MiniGptConfig};

    use crate::matmul_f32;

    #[test]
    #[ignore = "optional bridge check: run with cargo test -p rusty_ai_backend_candle -- --ignored"]
    fn minigpt_w_q_matmul_matches_candle_cpu() {
        let mut seed = 11u32;
        let cfg = MiniGptConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 32,
            max_seq: 32,
            attention_window: None,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let w = &m.blocks[0].w_q;
        let d = cfg.d_model;
        assert_eq!(w.shape(), &[d, d]);
        let w_data = w.data();

        let x = Tensor::from_vec((0..d).map(|i| i as f32 * 0.1).collect(), vec![1, d]).unwrap();
        let w_t = Tensor::from_vec(w_data.to_vec(), vec![d, d]).unwrap();
        let ref_out = matmul(&x, &w_t).unwrap();
        let ref_flat = ref_out.data();

        let dev = Device::Cpu;
        let got = matmul_f32(&dev, x.data(), [1, d], w_data, [d, d]).unwrap();

        let diff: f32 = ref_flat
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "max abs diff {}", diff);
    }
}

#[cfg(test)]
mod minigpt_lm_head_candle {
    use candle_core::Device;

    use rusty_ai_core::Tensor;
    use rusty_ai_llm::{linear_3d, MiniGpt, MiniGptConfig};

    use crate::matmul_f32;

    #[test]
    #[ignore = "optional Tier-1: LM-Head linear vs Candle CPU matmul"]
    fn minigpt_lm_head_matches_candle_cpu() {
        let mut seed = 3u32;
        let cfg = MiniGptConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 32,
            max_seq: 16,
            attention_window: None,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let seq = 5usize;
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let mut hid = vec![0.0f32; seq * d];
        for (i, slot) in hid.iter_mut().enumerate() {
            *slot = (i as f32) * 0.02 - 0.3;
        }
        let h = Tensor::from_vec(hid, vec![1, seq, d]).unwrap();
        let logits_ref = linear_3d(&h, &m.lm_head_w, &m.lm_head_b).unwrap();

        let dev = Device::Cpu;
        let rows = seq;
        let got = matmul_f32(
            &dev,
            h.data(),
            [rows, d],
            m.lm_head_w.data(),
            [d, v],
        )
        .unwrap();
        let b = m.lm_head_b.data();
        let mut diff_max = 0.0f32;
        for row in 0..rows {
            for j in 0..v {
                let a = logits_ref.data()[row * v + j];
                let c = got[row * v + j] + b[j];
                diff_max = diff_max.max((a - c).abs());
            }
        }
        assert!(diff_max < 1e-4, "max abs diff {}", diff_max);
    }
}
