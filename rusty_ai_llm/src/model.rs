//! Small GPT-style decoder: token + learned positional embeddings, transformer blocks, LM head.
//!
//! FIXME: learned absolute positions cap extrapolation beyond `max_seq` (consider RoPE / ALiBi for long code contexts).

use rusty_ai_core::{add, Tensor, TensorError};
use rusty_ai_nn::{gelu, glorot_uniform, layer_norm_affine, ones_scale, zeros_bias};

use crate::attention::{attention_single_query, attention_with_additive_mask, causal_attention};
use crate::fim::fim_additive_mask;
use crate::heads::{merge_heads, split_heads};
use crate::kv_cache::{concat_along_seq, KvCache, LayerKv};
use crate::linear_tensor::linear_3d;

/// Last timestep of logits shaped `(1, seq, vocab)` (also works for `seq == 1`).
fn logits_last_timestep_1batch(logits: &Tensor) -> Result<Tensor, TensorError> {
    let s = logits.shape();
    if s.len() != 3 || s[0] != 1 {
        return Err(TensorError::Shape(
            rusty_ai_core::ShapeError::InvalidReshape {
                from: s.to_vec(),
                to: vec![1, 0, 0],
            },
        ));
    }
    let seq = s[1];
    let v = s[2];
    if seq == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let data = logits.data();
    let start = (seq - 1) * v;
    Tensor::from_vec(data[start..start + v].to_vec(), vec![1, v])
}

/// Hyperparameters for [`MiniGpt`].
#[derive(Clone, Copy, Debug)]
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

impl MiniGptConfig {
    /// Kleines Profil für lokales Byte-Level-LM mit [`crate::ByteTokenizer`] (256 Vokabeln), ohne Hub.
    ///
    /// Ungefähre FP32-Größe: [`Self::approx_weight_bytes`] (typisch **weit unter 5 MiB**).
    pub fn micro_local() -> Self {
        Self {
            vocab_size: 256,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 64,
            max_seq: 64,
        }
    }

    /// Geschätzte Größe aller Parameter in Bytes (FP32), ohne das Modell zu allokieren.
    ///
    /// Formel folgt der Tensorstruktur in [`MiniGpt::random`] / [`DecoderBlock::random`].
    pub fn approx_weight_bytes(&self) -> usize {
        let v = self.vocab_size;
        let d = self.d_model;
        let f = self.ffn_dim;
        let m = self.max_seq;
        let l = self.n_layers;

        let tok_pos_lm = v * d + m * d + d * v + v + 2 * d;
        let per_block = 4 * d * d + 4 * d + d * f + f + f * d + d + 4 * d;
        let floats = tok_pos_lm + l * per_block;
        floats * std::mem::size_of::<f32>()
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
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
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
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![d],
                    to: vec![h, d / h],
                },
            ));
        }
        let dh = d / h;
        let bz = |cols: usize| zeros_bias(cols);
        let oz = |cols: usize| ones_scale(cols);
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
            ln1_gamma: oz(d)?,
            ln1_beta: bz(d)?,
            ln2_gamma: oz(d)?,
            ln2_beta: bz(d)?,
            n_heads: h,
            d_head: dh,
        })
    }

    /// Forward on `(batch, seq, d_model)`; returns same rank tensor.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, TensorError> {
        self.forward_prefill_optional(x, None)
    }

    /// Like [`Self::forward`], but stores keys/values after head split for KV-cache decoding.
    pub fn forward_prefill(&self, x: &Tensor, kv: &mut LayerKv) -> Result<Tensor, TensorError> {
        self.forward_prefill_optional(x, Some(kv))
    }

    fn forward_prefill_optional(
        &self,
        x: &Tensor,
        kv: Option<&mut LayerKv>,
    ) -> Result<Tensor, TensorError> {
        let s = x.shape();
        let batch = s[0];
        let d = s[2];
        let h = self.n_heads;
        let dh = self.d_head;
        debug_assert_eq!(d, h * dh);

        let x_ln = layer_norm_affine(x, &self.ln1_gamma, &self.ln1_beta, 1e-5)?;
        let q = linear_3d(&x_ln, &self.w_q, &self.b_q)?;
        let k = linear_3d(&x_ln, &self.w_k, &self.b_k)?;
        let v = linear_3d(&x_ln, &self.w_v, &self.b_v)?;

        let qh = split_heads(&q, h, dh)?;
        let kh = split_heads(&k, h, dh)?;
        let vh = split_heads(&v, h, dh)?;

        if let Some(kv) = kv {
            kv.k = Some(kh.clone());
            kv.v = Some(vh.clone());
        }

        let attn = causal_attention(&qh, &kh, &vh, dh)?;
        let merged = merge_heads(&attn, batch, h)?;
        let proj = linear_3d(&merged, &self.w_o, &self.b_o)?;

        let x1 = add(x, &proj)?;
        let x_ln2 = layer_norm_affine(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5)?;
        let h1 = linear_3d(&x_ln2, &self.w_ff1, &self.b_ff1)?;
        let h2 = gelu(&h1);
        let h3 = linear_3d(&h2, &self.w_ff2, &self.b_ff2)?;
        add(&x1, &h3)
    }

    /// Wie [`Self::forward`], aber mit additiver Attention-Maske `(seq, seq)` (z. B. FIM).
    pub fn forward_with_additive_mask(
        &self,
        x: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, TensorError> {
        let s = x.shape();
        let batch = s[0];
        let d = s[2];
        let h = self.n_heads;
        let dh = self.d_head;
        debug_assert_eq!(d, h * dh);

        let x_ln = layer_norm_affine(x, &self.ln1_gamma, &self.ln1_beta, 1e-5)?;
        let q = linear_3d(&x_ln, &self.w_q, &self.b_q)?;
        let k = linear_3d(&x_ln, &self.w_k, &self.b_k)?;
        let v = linear_3d(&x_ln, &self.w_v, &self.b_v)?;

        let qh = split_heads(&q, h, dh)?;
        let kh = split_heads(&k, h, dh)?;
        let vh = split_heads(&v, h, dh)?;

        let attn = attention_with_additive_mask(&qh, &kh, &vh, dh, mask)?;
        let merged = merge_heads(&attn, batch, h)?;
        let proj = linear_3d(&merged, &self.w_o, &self.b_o)?;

        let x1 = add(x, &proj)?;
        let x_ln2 = layer_norm_affine(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5)?;
        let h1 = linear_3d(&x_ln2, &self.w_ff1, &self.b_ff1)?;
        let h2 = gelu(&h1);
        let h3 = linear_3d(&h2, &self.w_ff2, &self.b_ff2)?;
        add(&x1, &h3)
    }

    /// Single new token `(1, 1, d_model)` with existing KV; updates `kv` in place.
    pub fn forward_step(&self, x: &Tensor, kv: &mut LayerKv) -> Result<Tensor, TensorError> {
        let s = x.shape();
        if s != [1, 1, self.n_heads * self.d_head] {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: s.to_vec(),
                    to: vec![1, 1, self.n_heads * self.d_head],
                },
            ));
        }
        let batch = 1usize;
        let h = self.n_heads;
        let dh = self.d_head;

        let x_ln = layer_norm_affine(x, &self.ln1_gamma, &self.ln1_beta, 1e-5)?;
        let q = linear_3d(&x_ln, &self.w_q, &self.b_q)?;
        let k_new = linear_3d(&x_ln, &self.w_k, &self.b_k)?;
        let v_new = linear_3d(&x_ln, &self.w_v, &self.b_v)?;

        let qh = split_heads(&q, h, dh)?;
        let kh_new = split_heads(&k_new, h, dh)?;
        let vh_new = split_heads(&v_new, h, dh)?;

        let (k_full, v_full) = match (&kv.k, &kv.v) {
            (Some(kp), Some(vp)) => {
                let kc = concat_along_seq(kp, &kh_new)?;
                let vc = concat_along_seq(vp, &vh_new)?;
                (kc, vc)
            }
            (None, None) => (kh_new, vh_new),
            _ => {
                return Err(TensorError::Shape(
                    rusty_ai_core::ShapeError::IncompatibleBroadcast {
                        left: vec![kv.k.is_some() as usize],
                        right: vec![kv.v.is_some() as usize],
                    },
                ));
            }
        };

        let attn = attention_single_query(&qh, &k_full, &v_full, dh)?;
        kv.k = Some(k_full);
        kv.v = Some(v_full);
        let merged = merge_heads(&attn, batch, h)?;
        let proj = linear_3d(&merged, &self.w_o, &self.b_o)?;

        let x1 = add(x, &proj)?;
        let x_ln2 = layer_norm_affine(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5)?;
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
    pub ln_f_gamma: Tensor,
    pub ln_f_beta: Tensor,
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
            ln_f_gamma: ones_scale(d)?,
            ln_f_beta: zeros_bias(d)?,
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

    /// Token plus positional embedding for one timestep: shape `(1, 1, d_model)`.
    pub fn embed_token_at(&self, id: usize, pos: usize) -> Result<Tensor, TensorError> {
        let d = self.cfg.d_model;
        let v = self.cfg.vocab_size;
        let max_pos = self.cfg.max_seq.saturating_sub(1);
        let mut out = vec![0.0f32; d];
        let w = self.tok_embed.data();
        let row = id.min(v - 1);
        for j in 0..d {
            out[j] = w[row * d + j];
        }
        let pt = pos.min(max_pos);
        let wp = self.pos_embed.data();
        for j in 0..d {
            out[j] += wp[pt * d + j];
        }
        Tensor::from_vec(out, vec![1, 1, d])
    }

    /// Prefill prompt tokens and fill `cache`; returns logits for the **last** prompt position `(1, vocab_size)`.
    pub fn forward_prefill(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
    ) -> Result<Tensor, TensorError> {
        if token_ids.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        if cache.layers.len() != self.cfg.n_layers {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![cache.layers.len()],
                    to: vec![self.cfg.n_layers],
                },
            ));
        }
        cache.clear();
        let tok = self.embed_tokens(token_ids)?;
        let pos = self.embed_positions(token_ids.len())?;
        let mut h = add(&tok, &pos)?;
        for (block, kv) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            h = block.forward_prefill(&h, kv)?;
        }
        let h = layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        let logits = linear_3d(&h, &self.lm_head_w, &self.lm_head_b)?;
        logits_last_timestep_1batch(&logits)
    }

    /// One autoregressive step: embed `token_id` at absolute `position`, extend KV cache, return logits `(1, vocab_size)`.
    pub fn forward_decode_step(
        &self,
        token_id: usize,
        position: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor, TensorError> {
        if cache.layers.len() != self.cfg.n_layers {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![cache.layers.len()],
                    to: vec![self.cfg.n_layers],
                },
            ));
        }
        let mut h = self.embed_token_at(token_id, position)?;
        for (block, kv) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            h = block.forward_step(&h, kv)?;
        }
        let h = layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        let logits = linear_3d(&h, &self.lm_head_w, &self.lm_head_b)?;
        logits_last_timestep_1batch(&logits)
    }

    /// Returns logits `(1, seq_len, vocab_size)` for the token-id sequence (batch size 1).
    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor, TensorError> {
        if token_ids.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let tok = self.embed_tokens(token_ids)?;
        let pos = self.embed_positions(token_ids.len())?;
        let mut h = add(&tok, &pos)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        linear_3d(&h, &self.lm_head_w, &self.lm_head_b)
    }

    /// Fill-in-the-middle: Sequenz `[prefix][middle][suffix]`; `middle_len` Token in der Mitte, Rest Suffix.
    ///
    /// Kein KV-Cache — für Inferenz mit FIM später eigene Pfadplanung nötig.
    pub fn forward_fim(
        &self,
        token_ids: &[usize],
        prefix_len: usize,
        middle_len: usize,
    ) -> Result<Tensor, TensorError> {
        if token_ids.is_empty() {
            return Err(TensorError::EmptyTensor);
        }
        let seq = token_ids.len();
        if prefix_len + middle_len > seq {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![prefix_len, middle_len],
                    to: vec![seq],
                },
            ));
        }
        let mask = fim_additive_mask(seq, prefix_len, middle_len)?;
        let tok = self.embed_tokens(token_ids)?;
        let pos = self.embed_positions(seq)?;
        let mut h = add(&tok, &pos)?;
        for block in &self.blocks {
            h = block.forward_with_additive_mask(&h, &mask)?;
        }
        let h = layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        linear_3d(&h, &self.lm_head_w, &self.lm_head_b)
    }

    /// Last time step only: logits `(1, vocab_size)` (for autoregressive sampling).
    pub fn forward_last(&self, token_ids: &[usize]) -> Result<Tensor, TensorError> {
        let logits = self.forward(token_ids)?;
        logits_last_timestep_1batch(&logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvCache;

    fn assert_vec_close(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < eps, "diff at {i}: {x} vs {y} (>{eps})");
        }
    }

    #[test]
    fn kv_prefill_logits_match_forward_last() {
        let mut seed = 42u32;
        let cfg = MiniGptConfig {
            vocab_size: 64,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 64,
            max_seq: 64,
        };
        let n_layers = cfg.n_layers;
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let ids = vec![3usize, 7, 11, 2];
        let mut cache = KvCache::new(n_layers);
        let pre = m.forward_prefill(&ids, &mut cache).unwrap();
        let last = m.forward_last(&ids).unwrap();
        assert_vec_close(pre.data(), last.data(), 1e-4);
    }

    #[test]
    fn kv_incremental_matches_forward_last() {
        let mut seed = 7u32;
        let cfg = MiniGptConfig {
            vocab_size: 64,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 64,
            max_seq: 64,
        };
        let n_layers = cfg.n_layers;
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let prefix = vec![1usize, 5, 9];
        let t = 19usize;
        let full: Vec<usize> = prefix.iter().copied().chain(std::iter::once(t)).collect();

        let mut cache = KvCache::new(n_layers);
        m.forward_prefill(&prefix, &mut cache).unwrap();
        let step = m.forward_decode_step(t, prefix.len(), &mut cache).unwrap();
        let last = m.forward_last(&full).unwrap();
        assert_vec_close(step.data(), last.data(), 1e-4);
    }

    #[test]
    fn empty_sequence_returns_empty_tensor_error() {
        let mut seed = 1u32;
        let cfg = MiniGptConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 32,
            max_seq: 32,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        assert_eq!(m.forward(&[]).unwrap_err(), TensorError::EmptyTensor);
        let mut cache = KvCache::new(1);
        assert_eq!(
            m.forward_prefill(&[], &mut cache).unwrap_err(),
            TensorError::EmptyTensor
        );
    }

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

    #[test]
    fn approx_weight_bytes_matches_state_dict() {
        use crate::state_dict;
        let cfg = MiniGptConfig::micro_local();
        let mut seed = 1u32;
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let dict = state_dict(&m);
        let floats: usize = dict.values().map(|t| t.data().len()).sum();
        assert_eq!(
            floats * std::mem::size_of::<f32>(),
            cfg.approx_weight_bytes()
        );
        assert!(cfg.approx_weight_bytes() < 5 * 1024 * 1024);
    }
}
