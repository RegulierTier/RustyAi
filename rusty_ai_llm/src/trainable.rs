//! Trainable [`MiniGpt`](crate::MiniGpt) using [`rusty_ai_autograd::Variable`] (same forward as tensor model).

use std::rc::Rc;

use rusty_ai_autograd::Variable;
use rusty_ai_core::TensorError;

use crate::attention_var::causal_attention_var;
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

        let x_ln = Variable::layer_norm(x, 1e-5)?;
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
        let x_ln2 = Variable::layer_norm(&x1, 1e-5)?;
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
        ]
    }
}

/// Full decoder LM with autograd; forward matches [`MiniGpt::forward`].
pub struct TrainableMiniGpt {
    pub cfg: MiniGptConfig,
    pub tok_embed: Rc<Variable>,
    pub pos_embed: Rc<Variable>,
    pub blocks: Vec<DecoderBlockTrainable>,
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
            cfg: m.cfg.clone(),
            tok_embed: Variable::leaf(m.tok_embed.clone()),
            pos_embed: Variable::leaf(m.pos_embed.clone()),
            blocks,
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
        let h = Variable::layer_norm(&h, 1e-5)?;
        linear_3d_var(&h, &self.lm_head_w, &self.lm_head_b)
    }

    pub fn parameters(&self) -> Vec<Rc<Variable>> {
        let mut p = vec![Rc::clone(&self.tok_embed), Rc::clone(&self.pos_embed)];
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.push(Rc::clone(&self.lm_head_w));
        p.push(Rc::clone(&self.lm_head_b));
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
