//! Causal attention built from [`rusty_ai_autograd::Variable`] (matches tensor [`causal_attention`] numerically).

use std::rc::Rc;

use rusty_ai_autograd::Variable;
use rusty_ai_core::Tensor;
use rusty_ai_core::TensorError;

/// Scaled dot-product causal attention on variable tensors `(B*H, L, d_h)`.
pub fn causal_attention_var(
    q: &Rc<Variable>,
    k: &Rc<Variable>,
    v: &Rc<Variable>,
    d_head: usize,
) -> Result<Rc<Variable>, TensorError> {
    let kt = Variable::transpose_batched_last2(k)?;
    let scores = Variable::matmul(q, &kt)?;
    let scale = 1.0f32 / (d_head as f32).sqrt();
    let scale_t = Variable::leaf(Tensor::scalar(scale));
    let scaled = Variable::mul(&scores, &scale_t)?;
    let masked = Variable::causal_mask_scores(&scaled)?;
    let attn = Variable::softmax_last_dim(&masked)?;
    Variable::matmul(&attn, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::causal_attention;
    use rusty_ai_core::Tensor;

    #[test]
    fn causal_attention_var_matches_tensor() {
        let bh = 2usize;
        let seq = 5usize;
        let dh = 4usize;
        let mut qd = vec![0.0f32; bh * seq * dh];
        let mut kd = vec![0.0f32; bh * seq * dh];
        let mut vd = vec![0.0f32; bh * seq * dh];
        for i in 0..qd.len() {
            qd[i] = (i as f32) * 0.01 - 0.2;
            kd[i] = (i as f32) * 0.02 + 0.05;
            vd[i] = (i as f32) * -0.015 + 0.1;
        }
        let qt = Tensor::from_vec(qd, vec![bh, seq, dh]).unwrap();
        let kt = Tensor::from_vec(kd, vec![bh, seq, dh]).unwrap();
        let vt = Tensor::from_vec(vd, vec![bh, seq, dh]).unwrap();
        let out_t = causal_attention(&qt, &kt, &vt, dh).unwrap();

        let qv = Variable::leaf(qt.clone());
        let kv = Variable::leaf(kt.clone());
        let vv = Variable::leaf(vt.clone());
        let out_v = causal_attention_var(&qv, &kv, &vv, dh).unwrap();
        let diff: f32 = out_t
            .data()
            .iter()
            .zip(out_v.data().data().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "max diff {}", diff);
    }
}
