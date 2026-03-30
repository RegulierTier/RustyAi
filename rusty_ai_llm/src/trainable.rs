//! Trainable [`MiniGpt`](crate::MiniGpt) using [`rusty_ai_autograd::Variable`] (same forward as tensor model).
//!
//! TODO: optional GPU / mixed-precision training path (e.g. via Candle) for larger runs.

use std::rc::Rc;

use rusty_ai_autograd::Variable;
use rusty_ai_core::TensorError;

use crate::attention_var::{causal_attention_var, fim_attention_var};
use crate::model::{DecoderBlock, MiniGpt, MiniGptConfig};

fn linear_3d_var(
    x: &Rc<Variable>,
    w: &Rc<Variable>,
    b: &Rc<Variable>,
) -> Result<Rc<Variable>, TensorError> {
    let xd = x.data();
    let s = xd.shape();
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
    let wd = w.data();
    let out_f = wd.shape()[1];
    let x2 = Variable::reshape(x, &[batch * seq, in_f])?;
    let y = Variable::matmul(&x2, w)?;
    let y = Variable::bias_add(&y, b)?;
    Variable::reshape(&y, &[batch, seq, out_f])
}

/// One decoder block with `Variable` weights.
pub struct DecoderBlockTrainable {
    pub w_q: Rc<Variable>,
    pub w_k: Rc<Variable>,
    pub w_v: Rc<Variable>,
    pub w_o: Rc<Variable>,
    pub b_q: Rc<Variable>,
    pub b_k: Rc<Variable>,
    pub b_v: Rc<Variable>,
    pub b_o: Rc<Variable>,
    pub w_ff1: Rc<Variable>,
    pub b_ff1: Rc<Variable>,
    pub w_ff2: Rc<Variable>,
    pub b_ff2: Rc<Variable>,
    pub ln1_gamma: Rc<Variable>,
    pub ln1_beta: Rc<Variable>,
    pub ln2_gamma: Rc<Variable>,
    pub ln2_beta: Rc<Variable>,
    pub n_heads: usize,
    pub d_head: usize,
}

impl DecoderBlockTrainable {
    pub fn from_block(b: &DecoderBlock) -> Self {
        Self {
            w_q: Variable::leaf(b.w_q.clone()),
            w_k: Variable::leaf(b.w_k.clone()),
            w_v: Variable::leaf(b.w_v.clone()),
            w_o: Variable::leaf(b.w_o.clone()),
            b_q: Variable::leaf(b.b_q.clone()),
            b_k: Variable::leaf(b.b_k.clone()),
            b_v: Variable::leaf(b.b_v.clone()),
            b_o: Variable::leaf(b.b_o.clone()),
            w_ff1: Variable::leaf(b.w_ff1.clone()),
            b_ff1: Variable::leaf(b.b_ff1.clone()),
            w_ff2: Variable::leaf(b.w_ff2.clone()),
            b_ff2: Variable::leaf(b.b_ff2.clone()),
            ln1_gamma: Variable::leaf(b.ln1_gamma.clone()),
            ln1_beta: Variable::leaf(b.ln1_beta.clone()),
            ln2_gamma: Variable::leaf(b.ln2_gamma.clone()),
            ln2_beta: Variable::leaf(b.ln2_beta.clone()),
            n_heads: b.n_heads,
            d_head: b.d_head,
        }
    }

    pub fn forward(&self, x: &Rc<Variable>) -> Result<Rc<Variable>, TensorError> {
        let xd = x.data();
        let s = xd.shape();
        let batch = s[0];
        let h = self.n_heads;
        let dh = self.d_head;

        let x_ln = Variable::layer_norm_affine(x, &self.ln1_gamma, &self.ln1_beta, 1e-5)?;
        let q = linear_3d_var(&x_ln, &self.w_q, &self.b_q)?;
        let k = linear_3d_var(&x_ln, &self.w_k, &self.b_k)?;
        let v = linear_3d_var(&x_ln, &self.w_v, &self.b_v)?;

        let qh = Variable::split_heads(&q, batch, h, dh)?;
        let kh = Variable::split_heads(&k, batch, h, dh)?;
        let vh = Variable::split_heads(&v, batch, h, dh)?;

        let attn = causal_attention_var(&qh, &kh, &vh, dh)?;
        let merged = Variable::merge_heads(&attn, batch, h)?;
        let proj = linear_3d_var(&merged, &self.w_o, &self.b_o)?;

        let x1 = Variable::add(x, &proj)?;
        let x_ln2 = Variable::layer_norm_affine(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5)?;
        let h1 = linear_3d_var(&x_ln2, &self.w_ff1, &self.b_ff1)?;
        let h2 = Variable::gelu(&h1);
        let h3 = linear_3d_var(&h2, &self.w_ff2, &self.b_ff2)?;
        Variable::add(&x1, &h3)
    }

    /// FIM-Forward mit [`fim_attention_var`] (Prefix/Mitte/Suffix-Längen).
    pub fn forward_fim(
        &self,
        x: &Rc<Variable>,
        prefix_len: usize,
        middle_len: usize,
    ) -> Result<Rc<Variable>, TensorError> {
        let xd = x.data();
        let s = xd.shape();
        let batch = s[0];
        let h = self.n_heads;
        let dh = self.d_head;

        let x_ln = Variable::layer_norm_affine(x, &self.ln1_gamma, &self.ln1_beta, 1e-5)?;
        let q = linear_3d_var(&x_ln, &self.w_q, &self.b_q)?;
        let k = linear_3d_var(&x_ln, &self.w_k, &self.b_k)?;
        let v = linear_3d_var(&x_ln, &self.w_v, &self.b_v)?;

        let qh = Variable::split_heads(&q, batch, h, dh)?;
        let kh = Variable::split_heads(&k, batch, h, dh)?;
        let vh = Variable::split_heads(&v, batch, h, dh)?;

        let attn = fim_attention_var(&qh, &kh, &vh, dh, prefix_len, middle_len)?;
        let merged = Variable::merge_heads(&attn, batch, h)?;
        let proj = linear_3d_var(&merged, &self.w_o, &self.b_o)?;

        let x1 = Variable::add(x, &proj)?;
        let x_ln2 = Variable::layer_norm_affine(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5)?;
        let h1 = linear_3d_var(&x_ln2, &self.w_ff1, &self.b_ff1)?;
        let h2 = Variable::gelu(&h1);
        let h3 = linear_3d_var(&h2, &self.w_ff2, &self.b_ff2)?;
        Variable::add(&x1, &h3)
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        vec![
            Rc::clone(&self.w_q),
            Rc::clone(&self.w_k),
            Rc::clone(&self.w_v),
            Rc::clone(&self.w_o),
            Rc::clone(&self.b_q),
            Rc::clone(&self.b_k),
            Rc::clone(&self.b_v),
            Rc::clone(&self.b_o),
            Rc::clone(&self.w_ff1),
            Rc::clone(&self.b_ff1),
            Rc::clone(&self.w_ff2),
            Rc::clone(&self.b_ff2),
            Rc::clone(&self.ln1_gamma),
            Rc::clone(&self.ln1_beta),
            Rc::clone(&self.ln2_gamma),
            Rc::clone(&self.ln2_beta),
        ]
    }
}

/// Full decoder LM with autograd; forward matches [`MiniGpt::forward`].
pub struct TrainableMiniGpt {
    pub cfg: MiniGptConfig,
    pub tok_embed: Rc<Variable>,
    pub pos_embed: Rc<Variable>,
    pub blocks: Vec<DecoderBlockTrainable>,
    pub ln_f_gamma: Rc<Variable>,
    pub ln_f_beta: Rc<Variable>,
    pub lm_head_w: Rc<Variable>,
    pub lm_head_b: Rc<Variable>,
}

impl TrainableMiniGpt {
    /// Copies weights from a tensor [`MiniGpt`] into trainable leaves.
    pub fn from_mini_gpt(m: &MiniGpt) -> Result<Self, TensorError> {
        let mut blocks = Vec::with_capacity(m.blocks.len());
        for b in &m.blocks {
            blocks.push(DecoderBlockTrainable::from_block(b));
        }
        Ok(Self {
            cfg: m.cfg,
            tok_embed: Variable::leaf(m.tok_embed.clone()),
            pos_embed: Variable::leaf(m.pos_embed.clone()),
            blocks,
            ln_f_gamma: Variable::leaf(m.ln_f_gamma.clone()),
            ln_f_beta: Variable::leaf(m.ln_f_beta.clone()),
            lm_head_w: Variable::leaf(m.lm_head_w.clone()),
            lm_head_b: Variable::leaf(m.lm_head_b.clone()),
        })
    }

    pub fn forward(&self, token_ids: &[usize]) -> Result<Rc<Variable>, TensorError> {
        let seq = token_ids.len();
        let max_pos = self.cfg.max_seq.saturating_sub(1);
        let tok = Variable::embedding_gather(&self.tok_embed, token_ids)?;
        let pos_ids: Vec<usize> = (0..seq).map(|t| t.min(max_pos)).collect();
        let pos = Variable::embedding_gather(&self.pos_embed, &pos_ids)?;
        let mut h = Variable::add(&tok, &pos)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = Variable::layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        linear_3d_var(&h, &self.lm_head_w, &self.lm_head_b)
    }

    /// FIM: gleiche Einbettungen wie [`Self::forward`], aber Attention mit FIM-Maske (`prefix` / `middle` / Rest = Suffix).
    pub fn forward_fim(
        &self,
        token_ids: &[usize],
        prefix_len: usize,
        middle_len: usize,
    ) -> Result<Rc<Variable>, TensorError> {
        let seq = token_ids.len();
        if prefix_len + middle_len > seq {
            return Err(TensorError::Shape(
                rusty_ai_core::ShapeError::InvalidReshape {
                    from: vec![prefix_len, middle_len],
                    to: vec![seq],
                },
            ));
        }
        let max_pos = self.cfg.max_seq.saturating_sub(1);
        let tok = Variable::embedding_gather(&self.tok_embed, token_ids)?;
        let pos_ids: Vec<usize> = (0..seq).map(|t| t.min(max_pos)).collect();
        let pos = Variable::embedding_gather(&self.pos_embed, &pos_ids)?;
        let mut h = Variable::add(&tok, &pos)?;
        for block in &self.blocks {
            h = block.forward_fim(&h, prefix_len, middle_len)?;
        }
        let h = Variable::layer_norm_affine(&h, &self.ln_f_gamma, &self.ln_f_beta, 1e-5)?;
        linear_3d_var(&h, &self.lm_head_w, &self.lm_head_b)
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        let mut p = vec![Rc::clone(&self.tok_embed), Rc::clone(&self.pos_embed)];
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.push(Rc::clone(&self.ln_f_gamma));
        p.push(Rc::clone(&self.ln_f_beta));
        p.push(Rc::clone(&self.lm_head_w));
        p.push(Rc::clone(&self.lm_head_b));
        p
    }

    /// Kopiert aktuelle Gewichte zurück in ein [`MiniGpt`] (z. B. für Checkpoints nach Training).
    pub fn to_mini_gpt(&self) -> MiniGpt {
        let blocks = self
            .blocks
            .iter()
            .map(|b| DecoderBlock {
                w_q: b.w_q.data().clone(),
                w_k: b.w_k.data().clone(),
                w_v: b.w_v.data().clone(),
                w_o: b.w_o.data().clone(),
                b_q: b.b_q.data().clone(),
                b_k: b.b_k.data().clone(),
                b_v: b.b_v.data().clone(),
                b_o: b.b_o.data().clone(),
                w_ff1: b.w_ff1.data().clone(),
                b_ff1: b.b_ff1.data().clone(),
                w_ff2: b.w_ff2.data().clone(),
                b_ff2: b.b_ff2.data().clone(),
                ln1_gamma: b.ln1_gamma.data().clone(),
                ln1_beta: b.ln1_beta.data().clone(),
                ln2_gamma: b.ln2_gamma.data().clone(),
                ln2_beta: b.ln2_beta.data().clone(),
                n_heads: b.n_heads,
                d_head: b.d_head,
            })
            .collect();
        MiniGpt {
            cfg: self.cfg,
            tok_embed: self.tok_embed.data().clone(),
            pos_embed: self.pos_embed.data().clone(),
            blocks,
            ln_f_gamma: self.ln_f_gamma.data().clone(),
            ln_f_beta: self.ln_f_beta.data().clone(),
            lm_head_w: self.lm_head_w.data().clone(),
            lm_head_b: self.lm_head_b.data().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fim::fim_middle_prediction_positions;
    use rusty_ai_autograd::{backward, Variable};

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn trainable_forward_matches_mini_gpt() {
        let mut seed = 42u32;
        let cfg = MiniGptConfig {
            vocab_size: 64,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 64,
            max_seq: 64,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let tv = TrainableMiniGpt::from_mini_gpt(&m).unwrap();
        let ids = vec![3usize, 7, 11, 2, 5];
        let ref_t = m.forward(&ids).unwrap();
        let logits_v = tv.forward(&ids).unwrap();
        let d = max_abs_diff(ref_t.data(), logits_v.data().data());
        assert!(d < 5e-3, "max abs diff {}", d);
    }

    #[test]
    fn trainable_forward_fim_matches_mini_gpt() {
        let mut seed = 99u32;
        let cfg = MiniGptConfig {
            vocab_size: 64,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 64,
            max_seq: 64,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let tv = TrainableMiniGpt::from_mini_gpt(&m).unwrap();
        let ids: Vec<usize> = (0..10).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();
        let prefix_len = 3usize;
        let middle_len = 4usize;
        let ref_t = m.forward_fim(&ids, prefix_len, middle_len).unwrap();
        let logits_v = tv.forward_fim(&ids, prefix_len, middle_len).unwrap();
        let d = max_abs_diff(ref_t.data(), logits_v.data().data());
        assert!(d < 5e-3, "max abs diff {}", d);
    }

    #[test]
    fn fim_subset_loss_backward_updates_head() {
        let mut seed = 3u32;
        let cfg = MiniGptConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 32,
            max_seq: 32,
        };
        let m = MiniGpt::random(cfg, &mut seed).unwrap();
        let tv = TrainableMiniGpt::from_mini_gpt(&m).unwrap();
        let token_ids: Vec<usize> = (0..8).map(|i| (i * 2 + 1) % cfg.vocab_size).collect();
        let prefix_len = 2usize;
        let middle_len = 3usize;
        let positions = fim_middle_prediction_positions(prefix_len, middle_len, token_ids.len());
        assert!(!positions.is_empty());

        for p in tv.parameters() {
            p.zero_grad();
        }
        let logits = tv.forward_fim(&token_ids, prefix_len, middle_len).unwrap();
        let loss =
            Variable::cross_entropy_next_token_subset(&logits, &token_ids, &positions).unwrap();
        backward(&loss).unwrap();
        let gw = tv.lm_head_w.grad().unwrap();
        assert!(gw.data().iter().any(|&x| x.abs() > 1e-8));
    }
}
