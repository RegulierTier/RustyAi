//! Autoregressive text generation and token sampling from logits.

use rusty_ai_core::{softmax, Tensor, TensorError};

use crate::model::MiniGpt;
use crate::tokenizer::ByteTokenizer;

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Simple LCG for deterministic sampling from a seed (not crypto-safe).
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7fff) as f32 / 32768.0
}

/// Samples one vocabulary index from logits shaped `(1, vocab)` (flat buffer).
///
/// - **`temperature`:** divides logits before softmax; `≈ 0` with top-p disabled → greedy argmax on raw logits.
/// - **`top_p`:** if in `(0, 1)`, nucleus sampling (Holtzman et al.): keep smallest token set whose
///   cumulative probability ≥ `top_p`, then sample within it. If `≥ 1` or `≤ 0`, uses full distribution
///   (multinomial) or greedy when cold.
pub fn sample_token(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
) -> Result<usize, TensorError> {
    let data = logits.data();
    let v = data.len();
    if v == 0 {
        return Err(TensorError::EmptyTensor);
    }
    let mut scaled = data.to_vec();
    if temperature > 1e-6 {
        for x in &mut scaled {
            *x /= temperature;
        }
    }
    let t = Tensor::from_vec(scaled, vec![1, v])?;
    let probs = softmax(&t)?;
    let p = probs.data();

    if top_p >= 1.0 || top_p <= 0.0 {
        if temperature <= 1e-6 {
            return Ok(argmax(data));
        }
        let r = lcg(seed).clamp(0.0, 1.0 - 1e-7);
        let mut c = 0.0f32;
        for (i, &pi) in p.iter().enumerate() {
            c += pi;
            if r <= c {
                return Ok(i);
            }
        }
        return Ok(v - 1);
    }

    // Nucleus (top-p): sort tokens by probability, take minimal prefix with cumulative mass ≥ top_p.
    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));

    let mut keep = vec![false; v];
    let mut cum = 0.0f32;
    for &idx in &order {
        cum += p[idx];
        keep[idx] = true;
        if cum >= top_p {
            break;
        }
    }

    let mut sub_mass = 0.0f32;
    for i in 0..v {
        if keep[i] {
            sub_mass += p[i];
        }
    }
    if sub_mass <= 1e-12 {
        return Ok(argmax(data));
    }
    let r = lcg(seed).clamp(0.0, 1.0 - 1e-7) * sub_mass;
    let mut c = 0.0f32;
    for (i, &pi) in p.iter().enumerate() {
        if keep[i] {
            c += pi;
            if r <= c {
                return Ok(i);
            }
        }
    }
    Ok(argmax(data))
}

/// Runs autoregressive generation: encodes `prompt`, then appends `max_tokens` sampled ids, decodes.
pub fn generate(
    model: &MiniGpt,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
) -> Result<String, TensorError> {
    let mut ids = ByteTokenizer::encode(prompt);
    for _ in 0..max_tokens {
        let logits = model.forward_last(&ids)?;
        let next = sample_token(&logits, temperature, top_p, seed)?;
        ids.push(next);
    }
    Ok(ByteTokenizer::decode(&ids))
}
