//! Map Hugging Face **GPT-2** `safetensors` / `state_dict` keys into [`MiniGpt`](crate::MiniGpt) parameters.
//!
//! - `c_attn` is **fused** QKV (Hugging Face `Conv1D`): weight shape **`[n_embd, 3 * n_embd]`** (rows × cols).
//!   Some exports use **`[3 * n_embd, n_embd]`**; those are transposed before splitting.
//! - If `lm_head.weight` is absent (weight tying), `lm_head_w` is copied from token embeddings (`wte`).
//! - **Tokenizer:** [`ByteTokenizer`](crate::ByteTokenizer) is byte-level only. For OpenAI/HF GPT-2 BPE, enable feature **`gpt2-bpe`**
//!   and use `Gpt2Tokenizer` with `tokenizer.json` from the same checkpoint directory.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors};

use rusty_ai_core::{transpose_2d, DType, Tensor};

use crate::checkpoint::{mini_gpt_from_state_dict, tensor_from_safetensors_view, CheckpointError};
use crate::model::MiniGptConfig;

/// Error when mapping GPT-2 tensors into [`MiniGpt`](crate::MiniGpt).
#[derive(Debug)]
pub enum Gpt2MappingError {
    Checkpoint(CheckpointError),
    Missing(String),
    BadShape {
        key: String,
        expected: String,
        got: Vec<usize>,
    },
    WrongDtype {
        key: String,
    },
}

impl std::fmt::Display for Gpt2MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Gpt2MappingError::Checkpoint(e) => write!(f, "{e}"),
            Gpt2MappingError::Missing(k) => write!(f, "missing tensor: {k}"),
            Gpt2MappingError::BadShape { key, expected, got } => {
                write!(f, "bad shape for {key}: expected {expected}, got {got:?}")
            }
            Gpt2MappingError::WrongDtype { key } => write!(f, "expected F32 for {key}"),
        }
    }
}

impl std::error::Error for Gpt2MappingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Gpt2MappingError::Checkpoint(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CheckpointError> for Gpt2MappingError {
    fn from(e: CheckpointError) -> Self {
        Gpt2MappingError::Checkpoint(e)
    }
}

impl From<SafeTensorError> for Gpt2MappingError {
    fn from(e: SafeTensorError) -> Self {
        Gpt2MappingError::Checkpoint(e.into())
    }
}

fn need_f32(tv: &safetensors::tensor::TensorView<'_>, key: &str) -> Result<(), Gpt2MappingError> {
    if tv.dtype() != Dtype::F32 {
        return Err(Gpt2MappingError::WrongDtype {
            key: key.to_string(),
        });
    }
    Ok(())
}

/// Split fused QKV weight into three `[d, d]` matrices (column blocks of `[d, 3*d]`).
///
/// Accepts HF layout **`[n_embd, 3 * n_embd]`** or the transposed **`[3 * n_embd, n_embd]`**.
fn split_qkv(w: &Tensor, key: &str) -> Result<(Tensor, Tensor, Tensor), Gpt2MappingError> {
    let s = w.shape();
    if s.len() != 2 {
        return Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: "rank-2 weight".into(),
            got: s.to_vec(),
        });
    }
    let (r, c) = (s[0], s[1]);
    if c == 3 * r {
        split_qkv_row_major(w, r, key)
    } else if r == 3 * c {
        let t = transpose_2d(w).map_err(|e| Gpt2MappingError::Checkpoint(e.into()))?;
        split_qkv_row_major(&t, c, key)
    } else {
        Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: "[n_embd, 3*n_embd] or [3*n_embd, n_embd]".into(),
            got: s.to_vec(),
        })
    }
}

fn split_qkv_row_major(
    w: &Tensor,
    d: usize,
    key: &str,
) -> Result<(Tensor, Tensor, Tensor), Gpt2MappingError> {
    if w.shape() != [d, 3 * d] {
        return Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: format!("[{d}, {}]", 3 * d),
            got: w.shape().to_vec(),
        });
    }
    let data = w.data();
    let mut q = Vec::with_capacity(d * d);
    let mut k = Vec::with_capacity(d * d);
    let mut v = Vec::with_capacity(d * d);
    for row in 0..d {
        let base = row * (3 * d);
        for col in 0..d {
            q.push(data[base + col]);
        }
        for col in 0..d {
            k.push(data[base + d + col]);
        }
        for col in 0..d {
            v.push(data[base + 2 * d + col]);
        }
    }
    Ok((
        Tensor::from_vec(q, vec![d, d]).map_err(CheckpointError::from)?,
        Tensor::from_vec(k, vec![d, d]).map_err(CheckpointError::from)?,
        Tensor::from_vec(v, vec![d, d]).map_err(CheckpointError::from)?,
    ))
}

/// Split fused QKV bias `[3*d]` (or `[1, 3*d]` flattened) into three `[1, d]` biases.
fn split_qkv_bias(b: &Tensor, key: &str) -> Result<(Tensor, Tensor, Tensor), Gpt2MappingError> {
    let n = b.data().len();
    if !n.is_multiple_of(3) {
        return Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: "length divisible by 3 (QKV)".into(),
            got: b.shape().to_vec(),
        });
    }
    let flat = b.data().to_vec();
    let d = n / 3;
    let shape_ok = matches!(b.shape(), [s] if *s == n) || matches!(b.shape(), [1, s] if *s == n);
    if !shape_ok {
        return Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: format!("[{n}] or [1, {n}]"),
            got: b.shape().to_vec(),
        });
    }
    let b_q = Tensor::from_vec(flat[0..d].to_vec(), vec![1, d]).map_err(CheckpointError::from)?;
    let b_k =
        Tensor::from_vec(flat[d..2 * d].to_vec(), vec![1, d]).map_err(CheckpointError::from)?;
    let b_v =
        Tensor::from_vec(flat[2 * d..3 * d].to_vec(), vec![1, d]).map_err(CheckpointError::from)?;
    Ok((b_q, b_k, b_v))
}

/// Build a RustyAi `state_dict` from Hugging Face GPT-2 tensor names.
///
/// Expected keys (prefix `transformer.` is optional depending on export):
/// `wte`, `wpe`, `h.{i}.attn.c_attn.weight`, … — see implementation for the full list.
pub fn gpt2_state_dict_to_minigpt(
    cfg: &MiniGptConfig,
    hf: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, Gpt2MappingError> {
    let d = cfg.d_model;
    let f = cfg.ffn_dim;
    let v = cfg.vocab_size;
    let m = cfg.max_seq;
    let n = cfg.n_layers;

    let mut out = HashMap::new();

    let wte = hf
        .get("transformer.wte.weight")
        .or_else(|| hf.get("wte.weight"))
        .ok_or_else(|| Gpt2MappingError::Missing("transformer.wte.weight".into()))?;
    if wte.shape() != [v, d] {
        return Err(Gpt2MappingError::BadShape {
            key: "wte".into(),
            expected: format!("[{v}, {d}]"),
            got: wte.shape().to_vec(),
        });
    }
    out.insert("tok_embed".into(), wte.clone());

    let wpe = hf
        .get("transformer.wpe.weight")
        .or_else(|| hf.get("wpe.weight"))
        .ok_or_else(|| Gpt2MappingError::Missing("transformer.wpe.weight".into()))?;
    if wpe.shape() != [m, d] {
        return Err(Gpt2MappingError::BadShape {
            key: "wpe".into(),
            expected: format!("[{m}, {d}]"),
            got: wpe.shape().to_vec(),
        });
    }
    out.insert("pos_embed".into(), wpe.clone());

    for i in 0..n {
        let pfx = format!("transformer.h.{i}");
        let c_attn_w = hf
            .get(&format!("{pfx}.attn.c_attn.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.attn.c_attn.weight")))?;
        let (w_q, w_k, w_v) = split_qkv(c_attn_w, &format!("{pfx}.attn.c_attn.weight"))?;
        out.insert(format!("blocks.{i}.w_q"), w_q);
        out.insert(format!("blocks.{i}.w_k"), w_k);
        out.insert(format!("blocks.{i}.w_v"), w_v);

        let c_attn_b = hf
            .get(&format!("{pfx}.attn.c_attn.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.attn.c_attn.bias")))?;
        let (b_q, b_k, b_v) = split_qkv_bias(c_attn_b, &format!("{pfx}.attn.c_attn.bias"))?;
        out.insert(format!("blocks.{i}.b_q"), b_q);
        out.insert(format!("blocks.{i}.b_k"), b_k);
        out.insert(format!("blocks.{i}.b_v"), b_v);

        let c_proj_w = hf
            .get(&format!("{pfx}.attn.c_proj.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.attn.c_proj.weight")))?;
        if c_proj_w.shape() != [d, d] {
            return Err(Gpt2MappingError::BadShape {
                key: format!("{pfx}.attn.c_proj.weight"),
                expected: format!("[{d}, {d}]"),
                got: c_proj_w.shape().to_vec(),
            });
        }
        out.insert(format!("blocks.{i}.w_o"), c_proj_w.clone());

        let c_proj_b = hf
            .get(&format!("{pfx}.attn.c_proj.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.attn.c_proj.bias")))?;
        if c_proj_b.shape() != [d] && c_proj_b.shape() != [1, d] {
            return Err(Gpt2MappingError::BadShape {
                key: format!("{pfx}.attn.c_proj.bias"),
                expected: format!("[{d}] or [1, {d}]"),
                got: c_proj_b.shape().to_vec(),
            });
        }
        let b_o = if c_proj_b.shape() == [d] {
            Tensor::from_vec(c_proj_b.data().to_vec(), vec![1, d]).map_err(CheckpointError::from)?
        } else {
            c_proj_b.clone()
        };
        out.insert(format!("blocks.{i}.b_o"), b_o);

        let c_fc_w = hf
            .get(&format!("{pfx}.mlp.c_fc.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.mlp.c_fc.weight")))?;
        if c_fc_w.shape() != [d, f] {
            return Err(Gpt2MappingError::BadShape {
                key: format!("{pfx}.mlp.c_fc.weight"),
                expected: format!("[{d}, {f}]"),
                got: c_fc_w.shape().to_vec(),
            });
        }
        out.insert(format!("blocks.{i}.w_ff1"), c_fc_w.clone());

        let c_fc_b = hf
            .get(&format!("{pfx}.mlp.c_fc.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.mlp.c_fc.bias")))?;
        let b_ff1 = match c_fc_b.shape() {
            [f_sz] if *f_sz == f => Tensor::from_vec(c_fc_b.data().to_vec(), vec![1, f])
                .map_err(CheckpointError::from)?,
            [1, f_sz] if *f_sz == f => c_fc_b.clone(),
            _ => {
                return Err(Gpt2MappingError::BadShape {
                    key: format!("{pfx}.mlp.c_fc.bias"),
                    expected: format!("[{f}] or [1, {f}]"),
                    got: c_fc_b.shape().to_vec(),
                });
            }
        };
        out.insert(format!("blocks.{i}.b_ff1"), b_ff1);

        let c_proj_mlp_w = hf
            .get(&format!("{pfx}.mlp.c_proj.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.mlp.c_proj.weight")))?;
        if c_proj_mlp_w.shape() != [f, d] {
            return Err(Gpt2MappingError::BadShape {
                key: format!("{pfx}.mlp.c_proj.weight"),
                expected: format!("[{f}, {d}]"),
                got: c_proj_mlp_w.shape().to_vec(),
            });
        }
        out.insert(format!("blocks.{i}.w_ff2"), c_proj_mlp_w.clone());

        let c_proj_mlp_b = hf
            .get(&format!("{pfx}.mlp.c_proj.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.mlp.c_proj.bias")))?;
        let b_ff2 = match c_proj_mlp_b.shape() {
            [d_sz] if *d_sz == d => Tensor::from_vec(c_proj_mlp_b.data().to_vec(), vec![1, d])
                .map_err(CheckpointError::from)?,
            [1, d_sz] if *d_sz == d => c_proj_mlp_b.clone(),
            _ => {
                return Err(Gpt2MappingError::BadShape {
                    key: format!("{pfx}.mlp.c_proj.bias"),
                    expected: format!("[{d}] or [1, {d}]"),
                    got: c_proj_mlp_b.shape().to_vec(),
                });
            }
        };
        out.insert(format!("blocks.{i}.b_ff2"), b_ff2);

        let ln1_w = hf
            .get(&format!("{pfx}.ln_1.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.ln_1.weight")))?;
        let ln1_b = hf
            .get(&format!("{pfx}.ln_1.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.ln_1.bias")))?;
        out.insert(
            format!("blocks.{i}.ln1_gamma"),
            ln1_to_affine(ln1_w, d, &format!("{pfx}.ln_1.weight"))?,
        );
        out.insert(
            format!("blocks.{i}.ln1_beta"),
            ln1_to_affine(ln1_b, d, &format!("{pfx}.ln_1.bias"))?,
        );

        let ln2_w = hf
            .get(&format!("{pfx}.ln_2.weight"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.ln_2.weight")))?;
        let ln2_b = hf
            .get(&format!("{pfx}.ln_2.bias"))
            .ok_or_else(|| Gpt2MappingError::Missing(format!("{pfx}.ln_2.bias")))?;
        out.insert(
            format!("blocks.{i}.ln2_gamma"),
            ln1_to_affine(ln2_w, d, &format!("{pfx}.ln_2.weight"))?,
        );
        out.insert(
            format!("blocks.{i}.ln2_beta"),
            ln1_to_affine(ln2_b, d, &format!("{pfx}.ln_2.bias"))?,
        );
    }

    let ln_f_w = hf
        .get("transformer.ln_f.weight")
        .or_else(|| hf.get("ln_f.weight"))
        .ok_or_else(|| Gpt2MappingError::Missing("transformer.ln_f.weight".into()))?;
    let ln_f_b = hf
        .get("transformer.ln_f.bias")
        .or_else(|| hf.get("ln_f.bias"))
        .ok_or_else(|| Gpt2MappingError::Missing("transformer.ln_f.bias".into()))?;
    out.insert(
        "ln_f_gamma".into(),
        ln1_to_affine(ln_f_w, d, "ln_f.weight")?,
    );
    out.insert("ln_f_beta".into(), ln1_to_affine(ln_f_b, d, "ln_f.bias")?);

    let lm = hf
        .get("lm_head.weight")
        .or_else(|| hf.get("transformer.wte.weight"))
        .ok_or_else(|| Gpt2MappingError::Missing("lm_head.weight (or wte for tying)".into()))?;
    // PyTorch / HF: `Linear` weight is `[out, in]` = `[vocab, n_embd]`; RustyAi stores `[d_model, vocab]`.
    let lm_head_w = match lm.shape() {
        [v_sz, d_sz] if *v_sz == v && *d_sz == d => {
            transpose_2d(lm).map_err(|e| Gpt2MappingError::Checkpoint(e.into()))?
        }
        [d_sz, v_sz] if *d_sz == d && *v_sz == v => lm.clone(),
        _ => {
            return Err(Gpt2MappingError::BadShape {
                key: "lm_head.weight".into(),
                expected: format!("[{v}, {d}] (HF) or [{d}, {v}] (RustyAi)"),
                got: lm.shape().to_vec(),
            });
        }
    };
    out.insert("lm_head_w".into(), lm_head_w);

    let lm_b = hf
        .get("lm_head.bias")
        .cloned()
        .unwrap_or_else(|| Tensor::zeros(&[1, v], DType::F32).expect("zeros"));
    if lm_b.shape() != [1, v] && lm_b.shape() != [v] {
        return Err(Gpt2MappingError::BadShape {
            key: "lm_head.bias".into(),
            expected: format!("[{v}] or [1, {v}]"),
            got: lm_b.shape().to_vec(),
        });
    }
    let lm_b = if lm_b.shape() == [v] {
        Tensor::from_vec(lm_b.data().to_vec(), vec![1, v]).map_err(CheckpointError::from)?
    } else {
        lm_b
    };
    out.insert("lm_head_b".into(), lm_b);

    Ok(out)
}

fn ln1_to_affine(t: &Tensor, d: usize, key: &str) -> Result<Tensor, Gpt2MappingError> {
    match t.shape() {
        [dd] if *dd == d => {
            Ok(Tensor::from_vec(t.data().to_vec(), vec![1, d]).map_err(CheckpointError::from)?)
        }
        [1, dd] if *dd == d => Ok(t.clone()),
        _ => Err(Gpt2MappingError::BadShape {
            key: key.to_string(),
            expected: format!("[{d}] or [1, {d}]"),
            got: t.shape().to_vec(),
        }),
    }
}

/// Load a GPT-2-format `model.safetensors` file and build a [`MiniGpt`](crate::MiniGpt).
///
/// `cfg` must match the checkpoint dimensions (`vocab_size`, `d_model`, `n_layers`, `ffn_dim`, `max_seq`, …).
pub fn load_minigpt_from_gpt2_safetensors(
    path: impl AsRef<Path>,
    cfg: MiniGptConfig,
) -> Result<crate::MiniGpt, Gpt2MappingError> {
    let buf = fs::read(path.as_ref())
        .map_err(|e| Gpt2MappingError::Checkpoint(CheckpointError::Io(e)))?;
    let st = SafeTensors::deserialize(&buf)?;
    let mut hf = HashMap::new();
    for (name, tv) in st.tensors() {
        need_f32(&tv, &name)?;
        let t = tensor_from_safetensors_view(&name, tv).map_err(Gpt2MappingError::from)?;
        hf.insert(name, t);
    }
    let mapped = gpt2_state_dict_to_minigpt(&cfg, &hf)?;
    let model = mini_gpt_from_state_dict(cfg, mapped).map_err(Gpt2MappingError::from)?;
    Ok(model)
}

#[cfg(test)]
mod gpt2_tests {
    use std::collections::HashMap;

    use rusty_ai_core::{transpose_2d, Tensor};

    use super::*;
    use crate::checkpoint::state_dict;
    use crate::model::{MiniGpt, MiniGptConfig};

    /// Fuse `w_q`, `w_k`, `w_v` (each `[d,d]`) into HF `c_attn.weight` `[d, 3d]`.
    fn fuse_qkv(w_q: &Tensor, w_k: &Tensor, w_v: &Tensor) -> Tensor {
        let d = w_q.shape()[0];
        assert_eq!(w_q.shape(), &[d, d]);
        let mut data = vec![0f32; d * 3 * d];
        let a = w_q.data();
        let b = w_k.data();
        let c = w_v.data();
        for r in 0..d {
            let row_off = r * (3 * d);
            for col in 0..d {
                data[row_off + col] = a[r * d + col];
                data[row_off + d + col] = b[r * d + col];
                data[row_off + 2 * d + col] = c[r * d + col];
            }
        }
        Tensor::from_vec(data, vec![d, 3 * d]).unwrap()
    }

    fn fuse_qkv_bias(b_q: &Tensor, b_k: &Tensor, b_v: &Tensor) -> Tensor {
        let d = b_q.shape()[1];
        let mut v = Vec::with_capacity(3 * d);
        v.extend_from_slice(&b_q.data()[0..d]);
        v.extend_from_slice(&b_k.data()[0..d]);
        v.extend_from_slice(&b_v.data()[0..d]);
        Tensor::from_vec(v, vec![3 * d]).unwrap()
    }

    #[test]
    fn gpt2_mapping_roundtrips_minigpt() {
        let mut seed = 3u32;
        let cfg = MiniGptConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 32,
            max_seq: 16,
        };
        let m = MiniGpt::random(cfg.clone(), &mut seed).unwrap();
        let d = state_dict(&m);

        let mut hf = HashMap::new();
        hf.insert("transformer.wte.weight".into(), d["tok_embed"].clone());
        hf.insert("transformer.wpe.weight".into(), d["pos_embed"].clone());
        for i in 0..cfg.n_layers {
            let p = format!("blocks.{i}");
            let wq = &d[&format!("{p}.w_q")];
            let wk = &d[&format!("{p}.w_k")];
            let wv = &d[&format!("{p}.w_v")];
            hf.insert(
                format!("transformer.h.{i}.attn.c_attn.weight"),
                fuse_qkv(wq, wk, wv),
            );
            let bq = &d[&format!("{p}.b_q")];
            let bk = &d[&format!("{p}.b_k")];
            let bv = &d[&format!("{p}.b_v")];
            hf.insert(
                format!("transformer.h.{i}.attn.c_attn.bias"),
                fuse_qkv_bias(bq, bk, bv),
            );
            hf.insert(
                format!("transformer.h.{i}.attn.c_proj.weight"),
                d[&format!("{p}.w_o")].clone(),
            );
            hf.insert(
                format!("transformer.h.{i}.attn.c_proj.bias"),
                Tensor::from_vec(d[&format!("{p}.b_o")].data().to_vec(), vec![cfg.d_model])
                    .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.mlp.c_fc.weight"),
                d[&format!("{p}.w_ff1")].clone(),
            );
            hf.insert(
                format!("transformer.h.{i}.mlp.c_fc.bias"),
                Tensor::from_vec(d[&format!("{p}.b_ff1")].data().to_vec(), vec![cfg.ffn_dim])
                    .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.mlp.c_proj.weight"),
                d[&format!("{p}.w_ff2")].clone(),
            );
            hf.insert(
                format!("transformer.h.{i}.mlp.c_proj.bias"),
                Tensor::from_vec(d[&format!("{p}.b_ff2")].data().to_vec(), vec![cfg.d_model])
                    .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.ln_1.weight"),
                Tensor::from_vec(
                    d[&format!("{p}.ln1_gamma")].data().to_vec(),
                    vec![cfg.d_model],
                )
                .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.ln_1.bias"),
                Tensor::from_vec(
                    d[&format!("{p}.ln1_beta")].data().to_vec(),
                    vec![cfg.d_model],
                )
                .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.ln_2.weight"),
                Tensor::from_vec(
                    d[&format!("{p}.ln2_gamma")].data().to_vec(),
                    vec![cfg.d_model],
                )
                .unwrap(),
            );
            hf.insert(
                format!("transformer.h.{i}.ln_2.bias"),
                Tensor::from_vec(
                    d[&format!("{p}.ln2_beta")].data().to_vec(),
                    vec![cfg.d_model],
                )
                .unwrap(),
            );
        }
        hf.insert(
            "transformer.ln_f.weight".into(),
            Tensor::from_vec(d["ln_f_gamma"].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        hf.insert(
            "transformer.ln_f.bias".into(),
            Tensor::from_vec(d["ln_f_beta"].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        hf.insert(
            "lm_head.weight".into(),
            transpose_2d(&d["lm_head_w"]).unwrap(),
        );
        hf.insert(
            "lm_head.bias".into(),
            Tensor::from_vec(d["lm_head_b"].data().to_vec(), vec![cfg.vocab_size]).unwrap(),
        );

        let mapped = gpt2_state_dict_to_minigpt(&cfg, &hf).unwrap();
        let m2 = crate::checkpoint::mini_gpt_from_state_dict(cfg.clone(), mapped).unwrap();

        let ids = vec![1usize, 2, 3, 4];
        let o1 = m.forward(&ids).unwrap();
        let o2 = m2.forward(&ids).unwrap();
        for (a, b) in o1.data().iter().zip(o2.data().iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// Some checkpoints store `c_attn.weight` as `[3*d, d]` instead of `[d, 3*d]`.
    #[test]
    fn gpt2_mapping_accepts_transposed_c_attn() {
        let mut seed = 11u32;
        let cfg = MiniGptConfig {
            vocab_size: 24,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ffn_dim: 16,
            max_seq: 8,
        };
        let m = MiniGpt::random(cfg.clone(), &mut seed).unwrap();
        let d = state_dict(&m);

        let mut hf = HashMap::new();
        hf.insert("transformer.wte.weight".into(), d["tok_embed"].clone());
        hf.insert("transformer.wpe.weight".into(), d["pos_embed"].clone());
        let p = "blocks.0";
        let fused = fuse_qkv(
            &d[&format!("{p}.w_q")],
            &d[&format!("{p}.w_k")],
            &d[&format!("{p}.w_v")],
        );
        hf.insert(
            "transformer.h.0.attn.c_attn.weight".into(),
            transpose_2d(&fused).unwrap(),
        );
        hf.insert(
            "transformer.h.0.attn.c_attn.bias".into(),
            fuse_qkv_bias(
                &d[&format!("{p}.b_q")],
                &d[&format!("{p}.b_k")],
                &d[&format!("{p}.b_v")],
            ),
        );
        hf.insert(
            "transformer.h.0.attn.c_proj.weight".into(),
            d[&format!("{p}.w_o")].clone(),
        );
        hf.insert(
            "transformer.h.0.attn.c_proj.bias".into(),
            Tensor::from_vec(d[&format!("{p}.b_o")].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        hf.insert(
            "transformer.h.0.mlp.c_fc.weight".into(),
            d[&format!("{p}.w_ff1")].clone(),
        );
        hf.insert(
            "transformer.h.0.mlp.c_fc.bias".into(),
            Tensor::from_vec(d[&format!("{p}.b_ff1")].data().to_vec(), vec![cfg.ffn_dim]).unwrap(),
        );
        hf.insert(
            "transformer.h.0.mlp.c_proj.weight".into(),
            d[&format!("{p}.w_ff2")].clone(),
        );
        hf.insert(
            "transformer.h.0.mlp.c_proj.bias".into(),
            Tensor::from_vec(d[&format!("{p}.b_ff2")].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        for (suffix, gamma) in [("ln_1", "ln1_gamma"), ("ln_2", "ln2_gamma")] {
            hf.insert(
                format!("transformer.h.0.{suffix}.weight"),
                Tensor::from_vec(
                    d[&format!("{p}.{gamma}")].data().to_vec(),
                    vec![cfg.d_model],
                )
                .unwrap(),
            );
        }
        for (suffix, beta) in [("ln_1", "ln1_beta"), ("ln_2", "ln2_beta")] {
            hf.insert(
                format!("transformer.h.0.{suffix}.bias"),
                Tensor::from_vec(d[&format!("{p}.{beta}")].data().to_vec(), vec![cfg.d_model])
                    .unwrap(),
            );
        }
        hf.insert(
            "transformer.ln_f.weight".into(),
            Tensor::from_vec(d["ln_f_gamma"].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        hf.insert(
            "transformer.ln_f.bias".into(),
            Tensor::from_vec(d["ln_f_beta"].data().to_vec(), vec![cfg.d_model]).unwrap(),
        );
        hf.insert(
            "lm_head.weight".into(),
            transpose_2d(&d["lm_head_w"]).unwrap(),
        );
        hf.insert(
            "lm_head.bias".into(),
            Tensor::from_vec(d["lm_head_b"].data().to_vec(), vec![cfg.vocab_size]).unwrap(),
        );

        let mapped = gpt2_state_dict_to_minigpt(&cfg, &hf).unwrap();
        let m2 = crate::checkpoint::mini_gpt_from_state_dict(cfg.clone(), mapped).unwrap();
        let ids = vec![1usize, 2];
        let o1 = m.forward(&ids).unwrap();
        let o2 = m2.forward(&ids).unwrap();
        for (a, b) in o1.data().iter().zip(o2.data().iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
