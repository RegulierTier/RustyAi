//! Small GPT-style decoder: token + learned positional embeddings, transformer blocks, LM head.

use rusty_ai_core::{add, Tensor, TensorError};
use rusty_ai_nn::{gelu, glorot_uniform, layer_norm, zeros_bias};

use crate::attention::causal_attention;
use crate::heads::{merge_heads, split_heads};
use crate::linear_tensor::linear_3d;

/// Hyperparameters for [`MiniGpt`].
#[derive(Clone, Debug)]
pub struct MiniGptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub max_seq: usize,
}

impl Default for MiniGptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 128,
            max_seq: 128,
        }
    }
}

/// One transformer decoder block: pre-norm attention sub-layer + pre-norm FFN sub-layer, residuals.
pub struct DecoderBlock {
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
    pub b_q: Tensor,
    pub b_k: Tensor,
    pub b_v: Tensor,
    pub b_o: Tensor,
    pub w_ff1: Tensor,
    pub b_ff1: Tensor,
    pub w_ff2: Tensor,
    pub b_ff2: Tensor,
    pub n_heads: usize,
    pub d_head: usize,
}

impl DecoderBlock {
    /// Random weights (Glorot) and zero biases. `d_model` must divide `n_heads`.
    pub fn random(cfg: &MiniGptConfig, seed: &mut u32) -> Result<Self, TensorError> {
        let d = cfg.d_model;
        let f = cfg.ffn_dim;
        let h = cfg.n_heads;
        if !d.is_multiple_of(h) {
            return Err(TensorError::Shape(rusty_ai_core::ShapeError::InvalidReshape {
                from: vec![d],
                to: vec![h, d / h],
            }));
        }
        let dh = d / h;
        let bz = |cols: usize| zeros_bias(cols);
        Ok(Self {
            w_q: glorot_uniform(d, d, seed)?,
            w_k: glorot_uniform(d, d, seed)?,
            w_v: glorot_uniform(d, d, seed)?,
            w_o: glorot_uniform(d, d, seed)?,
            b_q: bz(d)?,
            b_k: bz(d)?,
            b_v: bz(d)?,
            b_o: bz(d)?,
            w_ff1: glorot_uniform(d, f, seed)?,
            b_ff1: bz(f)?,
            w_ff2: glorot_uniform(f, d, seed)?,
            b_ff2: bz(d)?,
            n_heads: h,
            d_head: dh,
        })
    }

    /// Forward on `(batch, seq, d_model)`; returns same rank tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        let s = x.shape();
        let batch = s[0];
        let d = s[2];
        let h = self.n_heads;
        let dh = self.d_head;
        debug_assert_eq!(d, h * dh);

        let x_ln = layer_norm(x, 1e-5)?;
        let q = linear_3d(&x_ln, &self.w_q, &self.b_q)?;
        let k = linear_3d(&x_ln, &self.w_k, &self.b_k)?;
        let v = linear_3d(&x_ln, &self.w_v, &self.b_v)?;

        let qh = split_heads(&q, h, dh)?;
        let kh = split_heads(&k, h, dh)?;
        let vh = split_heads(&v, h, dh)?;

        let attn = causal_attention(&qh, &kh, &vh, dh)?;
        let merged = merge_heads(&attn, batch, h)?;
        let proj = linear_3d(&merged, &self.w_o, &self.b_o)?;

        let x1 = add(x, &proj)?;
        let x_ln2 = layer_norm(&x1, 1e-5)?;
        let h1 = linear_3d(&x_ln2, &self.w_ff1, &self.b_ff1)?;
        let h2 = gelu(&h1);
        let h3 = linear_3d(&h2, &self.w_ff2, &self.b_ff2)?;
        add(&x1, &h3)
    }
}

/// Full decoder-only language model (random init — training not included).
pub struct MiniGpt {
    pub cfg: MiniGptConfig,
    pub tok_embed: Tensor,
    pub pos_embed: Tensor,
    pub blocks: Vec<DecoderBlock>,
    pub lm_head_w: Tensor,
    pub lm_head_b: Tensor,
}

impl MiniGpt {
    /// Allocates embeddings, `n_layers` blocks, and the vocabulary projection head.
    pub fn random(cfg: MiniGptConfig, seed: &mut u32) -> Result<Self, TensorError> {
        let v = cfg.vocab_size;
        let d = cfg.d_model;
        let m = cfg.max_seq;
        let layers = cfg.n_layers;
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(DecoderBlock::random(&cfg, seed)?);
        }
        Ok(Self {
            tok_embed: glorot_uniform(v, d, seed)?,
            pos_embed: glorot_uniform(m, d, seed)?,
            blocks,
            lm_head_w: glorot_uniform(d, v, seed)?,
            lm_head_b: zeros_bias(v)?,
            cfg,
        })
    }

    fn embed_tokens(&self, ids: &[usize]) -> Result<Tensor, TensorError> {
        let seq = ids.len();
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let mut out = vec![0.0f32; seq * d];
        let w = self.tok_embed.data();
        for (t, &id) in ids.iter().enumerate() {
            let row = id.min(v - 1);
            for j in 0..d {
                out[t * d + j] = w[row * d + j];
            }
        }
        Tensor::from_vec(out, vec![1, seq, d])
    }

    fn embed_positions(&self, seq: usize) -> Result<Tensor, TensorError> {
        let d = self.cfg.d_model;
        let max_pos = self.cfg.max_seq.saturating_sub(1);
        let mut out = vec![0.0f32; seq * d];
        let w = self.pos_embed.data();
        for t in 0..seq {
            let pt = t.min(max_pos);
            for j in 0..d {
                out[t * d + j] = w[pt * d + j];
            }
        }
        Tensor::from_vec(out, vec![1, seq, d])
    }

    /// Returns logits `(1, seq_len, vocab_size)` for the token-id sequence (batch size 1).
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor, TensorError> {
        let tok = self.embed_tokens(token_ids)?;
        let pos = self.embed_positions(token_ids.len())?;
        let mut h = add(&tok, &pos)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = layer_norm(&h, 1e-5)?;
        linear_3d(&h, &self.lm_head_w, &self.lm_head_b)
    }

    /// Last time step only: logits `(1, vocab_size)` (for autoregressive sampling).
    pub fn forward_last(&self, token_ids: &[usize]) -> Result<Tensor, TensorError> {
        let logits = self.forward(token_ids)?;
        let s = logits.shape();
        let seq = s[1];
        let v = s[2];
        let data = logits.data();
        let start = (seq - 1) * v;
        let slice: Vec<f32> = data[start..start + v].to_vec();
        Tensor::from_vec(slice, vec![1, v])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_gpt_forward_shape() {
        let mut seed = 1u32;
        let cfg = MiniGptConfig {
            vocab_size: 256,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 64,
            max_seq: 32,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let ids = vec![1usize, 2, 3];
        let logits = m.forward(&ids).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 256]);
    }
}
