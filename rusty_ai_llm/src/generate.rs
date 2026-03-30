//! Autoregressive text generation and token sampling from logits.
//!
//! - **Inkrementell / IDE:** [`generate_from_ids_with_callback`] liefert jeden neuen Token an einen Callback (Abbruch mit `false`).
//! - Greedy `argmax` nutzt [`f32::total_cmp`] — deterministische Reihenfolge bei Gleichstand und NaN.

use rusty_ai_core::{softmax, Tensor, TensorError};

use crate::kv_cache::KvCache;
use crate::model::MiniGpt;
use crate::tokenizer::ByteTokenizer;

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
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

/// Autoregressive generation from **token ids** (any tokenizer with matching `vocab_size`).
///
/// Appends up to `max_tokens` new ids to the prompt sequence. If `max_tokens == 0`, returns the
/// prompt ids unchanged without running the model.
///
/// **Contract:** Each id must be `< model.cfg.vocab_size`. An **empty** `prompt_ids` with
/// `max_tokens > 0` yields `TensorError::EmptyTensor` (prefill requires at least one token).
///
/// Siehe auch [`generate_from_ids_with_callback`] für tokenweises Streaming an einen Callback.
pub fn generate_from_ids(
    model: &MiniGpt,
    prompt_ids: &[usize],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
) -> Result<Vec<usize>, TensorError> {
    generate_from_ids_with_callback(
        model,
        prompt_ids,
        max_tokens,
        temperature,
        top_p,
        seed,
        |_| true,
    )
}

/// Wie [`generate_from_ids`], ruft aber nach **jedem** neu gesampelten Token `on_token(id)` auf.
/// Liefert `false`, um die Generierung vorzeitig zu beenden (z. B. UI-Abbruch, Stoppkriterium).
/// Der zuletzt erzeugte Token ist bereits in der Rückgabe-`Vec` enthalten.
pub fn generate_from_ids_with_callback<F>(
    model: &MiniGpt,
    prompt_ids: &[usize],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
    mut on_token: F,
) -> Result<Vec<usize>, TensorError>
where
    F: FnMut(usize) -> bool,
{
    let mut ids = prompt_ids.to_vec();
    if max_tokens == 0 {
        return Ok(ids);
    }
    let mut cache = KvCache::new(model.cfg.n_layers);
    let mut logits = model.forward_prefill(&ids, &mut cache)?;
    for i in 0..max_tokens {
        let next = sample_token(&logits, temperature, top_p, seed)?;
        ids.push(next);
        if !on_token(next) {
            break;
        }
        if i + 1 < max_tokens {
            logits = model.forward_decode_step(next, ids.len() - 1, &mut cache)?;
        }
    }
    Ok(ids)
}

/// Runs autoregressive generation: encodes `prompt` with [`ByteTokenizer`], samples, decodes.
pub fn generate(
    model: &MiniGpt,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: &mut u32,
) -> Result<String, TensorError> {
    let ids = ByteTokenizer::encode(prompt);
    if max_tokens == 0 {
        return Ok(ByteTokenizer::decode(&ids));
    }
    let out = generate_from_ids(model, &ids, max_tokens, temperature, top_p, seed)?;
    Ok(ByteTokenizer::decode(&out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MiniGptConfig;
    use crate::tokenizer::ByteTokenizer;

    #[test]
    fn generate_from_ids_matches_byte_generate() {
        let mut seed = 42u32;
        let cfg = MiniGptConfig {
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 128,
            max_seq: 128,
        };
        let model = MiniGpt::random(cfg, &mut seed).unwrap();
        let prompt = "hi";
        let mut s1 = 7u32;
        let s1_out = generate(&model, prompt, 5, 1.0, 1.0, &mut s1).unwrap();
        let mut s2 = 7u32;
        let ids = ByteTokenizer::encode(prompt);
        let s2_out = generate_from_ids(&model, &ids, 5, 1.0, 1.0, &mut s2).unwrap();
        assert_eq!(s1_out, ByteTokenizer::decode(&s2_out));
    }

    #[test]
    fn generate_from_ids_max_tokens_zero_returns_prompt_only() {
        let mut seed = 1u32;
        let cfg = MiniGptConfig {
            vocab_size: 256,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 64,
            max_seq: 64,
        };
        let model = MiniGpt::random(cfg, &mut seed).unwrap();
        let ids = ByteTokenizer::encode("abc");
        let out = generate_from_ids(&model, &ids, 0, 1.0, 1.0, &mut 0u32).unwrap();
        assert_eq!(out, ids);
    }

    #[test]
    fn generate_from_ids_with_callback_matches_non_callback() {
        let mut seed = 99u32;
        let cfg = MiniGptConfig {
            vocab_size: 256,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 64,
            max_seq: 64,
        };
        let model = MiniGpt::random(cfg, &mut seed).unwrap();
        let prompt = ByteTokenizer::encode("x");
        let mut s1 = 5u32;
        let mut s2 = 5u32;
        let full = generate_from_ids(&model, &prompt, 4, 0.9, 0.95, &mut s1).unwrap();
        let mut seen = Vec::new();
        let with_cb =
            generate_from_ids_with_callback(&model, &prompt, 4, 0.9, 0.95, &mut s2, |t| {
                seen.push(t);
                true
            })
            .unwrap();
        assert_eq!(full, with_cb);
        assert_eq!(seen, full[prompt.len()..]);
    }

    #[test]
    fn generate_from_ids_with_callback_stops_early() {
        let mut seed = 3u32;
        let cfg = MiniGptConfig {
            vocab_size: 256,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            ffn_dim: 64,
            max_seq: 64,
        };
        let model = MiniGpt::random(cfg, &mut seed).unwrap();
        let prompt = ByteTokenizer::encode("a");
        let mut count = 0usize;
        let out =
            generate_from_ids_with_callback(&model, &prompt, 10, 1.0, 1.0, &mut 11u32, |_| {
                count += 1;
                count < 2
            })
            .unwrap();
        assert_eq!(out.len(), prompt.len() + 2);
    }
}
