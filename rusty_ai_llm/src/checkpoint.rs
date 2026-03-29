//! Save and load [`MiniGpt`](crate::MiniGpt) weights as Hugging-Face-style `safetensors` + `config.json`.
//!
//! Tensor names follow a stable RustyAi scheme (`blocks.{i}.*`). Loading GPT-2 checkpoints from the Hub
//! is implemented in [`gpt2_hf`](crate::gpt2_hf).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors, View};
use safetensors::serialize;
use serde::{Deserialize, Serialize};

use rusty_ai_core::{Tensor, TensorError};

use crate::model::{DecoderBlock, MiniGpt, MiniGptConfig};

/// JSON companion file for a [`MiniGpt`] checkpoint (HF-style field names).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MiniGptConfigFile {
    /// Discriminator for tooling (`rusty_ai_minigpt`).
    #[serde(rename = "model_type")]
    pub model_type: String,
    pub vocab_size: usize,
    #[serde(rename = "n_embd")]
    pub n_embd: usize,
    #[serde(rename = "n_layer")]
    pub n_layer: usize,
    #[serde(rename = "n_head")]
    pub n_head: usize,
    #[serde(rename = "n_inner")]
    pub n_inner: usize,
    #[serde(rename = "n_positions")]
    pub n_positions: usize,
}

impl From<&MiniGptConfig> for MiniGptConfigFile {
    fn from(c: &MiniGptConfig) -> Self {
        Self {
            model_type: "rusty_ai_minigpt".to_string(),
            vocab_size: c.vocab_size,
            n_embd: c.d_model,
            n_layer: c.n_layers,
            n_head: c.n_heads,
            n_inner: c.ffn_dim,
            n_positions: c.max_seq,
        }
    }
}

impl TryFrom<MiniGptConfigFile> for MiniGptConfig {
    type Error = CheckpointError;

    fn try_from(f: MiniGptConfigFile) -> Result<Self, Self::Error> {
        if f.model_type != "rusty_ai_minigpt" {
            return Err(CheckpointError::UnsupportedModelType(f.model_type));
        }
        if f.vocab_size == 0 || f.n_layer == 0 || f.n_inner == 0 || f.n_positions == 0 {
            return Err(CheckpointError::InvalidConfig(
                "vocab_size, n_layer, n_inner, n_positions must be positive".into(),
            ));
        }
        if f.n_embd == 0 || f.n_head == 0 || !f.n_embd.is_multiple_of(f.n_head) {
            return Err(CheckpointError::InvalidConfig(
                "n_embd must be positive and divisible by n_head".into(),
            ));
        }
        Ok(Self {
            vocab_size: f.vocab_size,
            d_model: f.n_embd,
            n_heads: f.n_head,
            n_layers: f.n_layer,
            ffn_dim: f.n_inner,
            max_seq: f.n_positions,
        })
    }
}

/// Errors when reading/writing checkpoints.
#[derive(Debug)]
pub enum CheckpointError {
    Io(std::io::Error),
    SafeTensor(SafeTensorError),
    Json(serde_json::Error),
    Tensor(TensorError),
    MissingTensor(String),
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidConfig(String),
    UnsupportedModelType(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Io(e) => write!(f, "I/O error: {e}"),
            CheckpointError::SafeTensor(e) => write!(f, "safetensors: {e}"),
            CheckpointError::Json(e) => write!(f, "JSON: {e}"),
            CheckpointError::Tensor(e) => write!(f, "tensor: {e}"),
            CheckpointError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            CheckpointError::ShapeMismatch { name, expected, got } => {
                write!(f, "shape mismatch for {name}: expected {expected:?}, got {got:?}")
            }
            CheckpointError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            CheckpointError::UnsupportedModelType(s) => write!(f, "unsupported model_type: {s}"),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CheckpointError::Io(e) => Some(e),
            CheckpointError::SafeTensor(e) => Some(e),
            CheckpointError::Json(e) => Some(e),
            CheckpointError::Tensor(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        CheckpointError::Io(e)
    }
}

impl From<SafeTensorError> for CheckpointError {
    fn from(e: SafeTensorError) -> Self {
        CheckpointError::SafeTensor(e)
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(e: serde_json::Error) -> Self {
        CheckpointError::Json(e)
    }
}

impl From<TensorError> for CheckpointError {
    fn from(e: TensorError) -> Self {
        CheckpointError::Tensor(e)
    }
}

/// Wraps a [`Tensor`] for safetensors serialization (F32, row-major).
struct TensorAsView<'a>(&'a Tensor);

impl View for TensorAsView<'_> {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        let s = self.0.data();
        let bytes = std::mem::size_of_val(s);
        let mut v = vec![0u8; bytes];
        for (i, x) in s.iter().enumerate() {
            v[i * 4..(i + 1) * 4].copy_from_slice(&x.to_le_bytes());
        }
        std::borrow::Cow::Owned(v)
    }

    fn data_len(&self) -> usize {
        std::mem::size_of_val(self.0.data())
    }
}

/// Convert a safetensors [`TensorView`](safetensors::tensor::TensorView) to [`Tensor`] (F32).
pub fn tensor_from_safetensors_view(
    name: &str,
    tv: safetensors::tensor::TensorView<'_>,
) -> Result<Tensor, CheckpointError> {
    if tv.dtype() != Dtype::F32 {
        return Err(CheckpointError::InvalidConfig(format!(
            "tensor {name}: expected F32, got {:?}",
            tv.dtype()
        )));
    }
    let shape = tv.shape().to_vec();
    let data = tv.data();
    let n = shape.iter().product::<usize>();
    if data.len() != n * 4 {
        return Err(CheckpointError::InvalidConfig(format!(
            "tensor {name}: byte length {} does not match shape {:?}",
            data.len(),
            shape
        )));
    }
    let mut v = Vec::with_capacity(n);
    for chunk in data.chunks_exact(4) {
        v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Tensor::from_vec(v, shape)?)
}

/// Export all parameters as owned tensors keyed by name.
pub fn state_dict(model: &MiniGpt) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    m.insert("tok_embed".into(), model.tok_embed.clone());
    m.insert("pos_embed".into(), model.pos_embed.clone());
    for (i, b) in model.blocks.iter().enumerate() {
        let p = format!("blocks.{i}");
        m.insert(format!("{p}.w_q"), b.w_q.clone());
        m.insert(format!("{p}.w_k"), b.w_k.clone());
        m.insert(format!("{p}.w_v"), b.w_v.clone());
        m.insert(format!("{p}.w_o"), b.w_o.clone());
        m.insert(format!("{p}.b_q"), b.b_q.clone());
        m.insert(format!("{p}.b_k"), b.b_k.clone());
        m.insert(format!("{p}.b_v"), b.b_v.clone());
        m.insert(format!("{p}.b_o"), b.b_o.clone());
        m.insert(format!("{p}.w_ff1"), b.w_ff1.clone());
        m.insert(format!("{p}.b_ff1"), b.b_ff1.clone());
        m.insert(format!("{p}.w_ff2"), b.w_ff2.clone());
        m.insert(format!("{p}.b_ff2"), b.b_ff2.clone());
        m.insert(format!("{p}.ln1_gamma"), b.ln1_gamma.clone());
        m.insert(format!("{p}.ln1_beta"), b.ln1_beta.clone());
        m.insert(format!("{p}.ln2_gamma"), b.ln2_gamma.clone());
        m.insert(format!("{p}.ln2_beta"), b.ln2_beta.clone());
    }
    m.insert("ln_f_gamma".into(), model.ln_f_gamma.clone());
    m.insert("ln_f_beta".into(), model.ln_f_beta.clone());
    m.insert("lm_head_w".into(), model.lm_head_w.clone());
    m.insert("lm_head_b".into(), model.lm_head_b.clone());
    m
}

fn take_tensor(map: &mut HashMap<String, Tensor>, key: &str) -> Result<Tensor, CheckpointError> {
    map.remove(key)
        .ok_or_else(|| CheckpointError::MissingTensor(key.to_string()))
}

fn expect_shape(t: &Tensor, name: &str, shape: &[usize]) -> Result<(), CheckpointError> {
    if t.shape() != shape {
        return Err(CheckpointError::ShapeMismatch {
            name: name.to_string(),
            expected: shape.to_vec(),
            got: t.shape().to_vec(),
        });
    }
    Ok(())
}

/// Rebuild [`MiniGpt`] from a config and a name→tensor map (e.g. after loading safetensors).
pub fn mini_gpt_from_state_dict(
    cfg: MiniGptConfig,
    mut tensors: HashMap<String, Tensor>,
) -> Result<MiniGpt, CheckpointError> {
    let v = cfg.vocab_size;
    let d = cfg.d_model;
    let f = cfg.ffn_dim;
    let h = cfg.n_heads;
    let layers = cfg.n_layers;
    let m = cfg.max_seq;
    if !d.is_multiple_of(h) {
        return Err(CheckpointError::InvalidConfig(
            "d_model must be divisible by n_heads".into(),
        ));
    }
    let dh = d / h;

    let tok_embed = take_tensor(&mut tensors, "tok_embed")?;
    expect_shape(&tok_embed, "tok_embed", &[v, d])?;
    let pos_embed = take_tensor(&mut tensors, "pos_embed")?;
    expect_shape(&pos_embed, "pos_embed", &[m, d])?;

    let mut blocks = Vec::with_capacity(layers);
    for i in 0..layers {
        let p = format!("blocks.{i}");
        let w_q = take_tensor(&mut tensors, &format!("{p}.w_q"))?;
        expect_shape(&w_q, &format!("{p}.w_q"), &[d, d])?;
        let w_k = take_tensor(&mut tensors, &format!("{p}.w_k"))?;
        expect_shape(&w_k, &format!("{p}.w_k"), &[d, d])?;
        let w_v = take_tensor(&mut tensors, &format!("{p}.w_v"))?;
        expect_shape(&w_v, &format!("{p}.w_v"), &[d, d])?;
        let w_o = take_tensor(&mut tensors, &format!("{p}.w_o"))?;
        expect_shape(&w_o, &format!("{p}.w_o"), &[d, d])?;
        let b_q = take_tensor(&mut tensors, &format!("{p}.b_q"))?;
        expect_shape(&b_q, &format!("{p}.b_q"), &[1, d])?;
        let b_k = take_tensor(&mut tensors, &format!("{p}.b_k"))?;
        expect_shape(&b_k, &format!("{p}.b_k"), &[1, d])?;
        let b_v = take_tensor(&mut tensors, &format!("{p}.b_v"))?;
        expect_shape(&b_v, &format!("{p}.b_v"), &[1, d])?;
        let b_o = take_tensor(&mut tensors, &format!("{p}.b_o"))?;
        expect_shape(&b_o, &format!("{p}.b_o"), &[1, d])?;

        let w_ff1 = take_tensor(&mut tensors, &format!("{p}.w_ff1"))?;
        expect_shape(&w_ff1, &format!("{p}.w_ff1"), &[d, f])?;
        let b_ff1 = take_tensor(&mut tensors, &format!("{p}.b_ff1"))?;
        expect_shape(&b_ff1, &format!("{p}.b_ff1"), &[1, f])?;
        let w_ff2 = take_tensor(&mut tensors, &format!("{p}.w_ff2"))?;
        expect_shape(&w_ff2, &format!("{p}.w_ff2"), &[f, d])?;
        let b_ff2 = take_tensor(&mut tensors, &format!("{p}.b_ff2"))?;
        expect_shape(&b_ff2, &format!("{p}.b_ff2"), &[1, d])?;

        let ln1_gamma = take_tensor(&mut tensors, &format!("{p}.ln1_gamma"))?;
        expect_shape(&ln1_gamma, &format!("{p}.ln1_gamma"), &[1, d])?;
        let ln1_beta = take_tensor(&mut tensors, &format!("{p}.ln1_beta"))?;
        expect_shape(&ln1_beta, &format!("{p}.ln1_beta"), &[1, d])?;
        let ln2_gamma = take_tensor(&mut tensors, &format!("{p}.ln2_gamma"))?;
        expect_shape(&ln2_gamma, &format!("{p}.ln2_gamma"), &[1, d])?;
        let ln2_beta = take_tensor(&mut tensors, &format!("{p}.ln2_beta"))?;
        expect_shape(&ln2_beta, &format!("{p}.ln2_beta"), &[1, d])?;

        blocks.push(DecoderBlock {
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
            w_ff1,
            b_ff1,
            w_ff2,
            b_ff2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            n_heads: h,
            d_head: dh,
        });
    }

    let ln_f_gamma = take_tensor(&mut tensors, "ln_f_gamma")?;
    expect_shape(&ln_f_gamma, "ln_f_gamma", &[1, d])?;
    let ln_f_beta = take_tensor(&mut tensors, "ln_f_beta")?;
    expect_shape(&ln_f_beta, "ln_f_beta", &[1, d])?;
    let lm_head_w = take_tensor(&mut tensors, "lm_head_w")?;
    expect_shape(&lm_head_w, "lm_head_w", &[d, v])?;
    let lm_head_b = take_tensor(&mut tensors, "lm_head_b")?;
    expect_shape(&lm_head_b, "lm_head_b", &[1, v])?;

    if !tensors.is_empty() {
        let keys: Vec<_> = tensors.keys().cloned().collect();
        return Err(CheckpointError::InvalidConfig(format!(
            "unexpected tensors in checkpoint: {keys:?}"
        )));
    }

    Ok(MiniGpt {
        cfg,
        tok_embed,
        pos_embed,
        blocks,
        ln_f_gamma,
        ln_f_beta,
        lm_head_w,
        lm_head_b,
    })
}

/// Serialize weights to a safetensors byte buffer.
pub fn minigpt_to_safetensors_bytes(model: &MiniGpt) -> Result<Vec<u8>, CheckpointError> {
    let dict = state_dict(model);
    let pairs: Vec<_> = dict
        .iter()
        .map(|(k, t)| (k.as_str(), TensorAsView(t)))
        .collect();
    Ok(serialize(pairs, None)?)
}

/// Write `config.json` and `model.safetensors` into `dir`.
pub fn save_minigpt_checkpoint(dir: impl AsRef<Path>, model: &MiniGpt) -> Result<(), CheckpointError> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let cfg_path = dir.join("config.json");
    let file_cfg: MiniGptConfigFile = (&model.cfg).into();
    let json = serde_json::to_string_pretty(&file_cfg)?;
    fs::write(&cfg_path, json)?;

    let weights_path = dir.join("model.safetensors");
    let bytes = minigpt_to_safetensors_bytes(model)?;
    fs::write(&weights_path, bytes)?;
    Ok(())
}

/// Load [`MiniGpt`] from `dir/config.json` and `dir/model.safetensors`.
pub fn load_minigpt_checkpoint(dir: impl AsRef<Path>) -> Result<MiniGpt, CheckpointError> {
    let dir = dir.as_ref();
    let cfg: MiniGptConfigFile = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
    let cfg = MiniGptConfig::try_from(cfg)?;
    let buf = fs::read(dir.join("model.safetensors"))?;
    let st = SafeTensors::deserialize(&buf)?;
    let mut map = HashMap::new();
    for (name, tv) in st.tensors() {
        map.insert(name.clone(), tensor_from_safetensors_view(&name, tv)?);
    }
    mini_gpt_from_state_dict(cfg, map)
}

#[cfg(feature = "hf-hub")]
mod hf_hub_download {
    use std::fs;
    use std::path::Path;

    use super::{load_minigpt_checkpoint, CheckpointError, MiniGpt};

    /// Download `config.json` and `model.safetensors` from a Hub model repo into `dest_dir`, then load.
    ///
    /// **Erwartung:** Das Repo enthält einen **RustyAi**-Checkpoint (`model_type` = `rusty_ai_minigpt` in `config.json`),
    /// nicht ein rohes GPT-2-Gewichtsarchiv. Für GPT-2-`safetensors` siehe [`crate::load_minigpt_from_gpt2_safetensors`].
    ///
    /// Dateien werden über den hf-hub-Cache geholt (siehe [`hf_hub::api::sync::Api`]).
    pub fn load_minigpt_from_hf(repo_id: &str, dest_dir: impl AsRef<Path>) -> Result<MiniGpt, CheckpointError> {
        let dest_dir = dest_dir.as_ref();
        fs::create_dir_all(dest_dir)?;
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            CheckpointError::InvalidConfig(format!("hf_hub Api::new: {e}"))
        })?;
        for file in ["config.json", "model.safetensors"] {
            let path = api
                .model(repo_id.to_string())
                .get(file)
                .map_err(|e| CheckpointError::InvalidConfig(format!("hf_hub get {file}: {e}")))?;
            let dest = dest_dir.join(file);
            if path != dest {
                fs::copy(&path, &dest)?;
            }
        }
        load_minigpt_checkpoint(dest_dir)
    }
}

#[cfg(feature = "hf-hub")]
pub use hf_hub_download::load_minigpt_from_hf;

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::model::MiniGpt;

    fn assert_tensors_close(a: &Tensor, b: &Tensor, eps: f32) {
        assert_eq!(a.shape(), b.shape());
        for (x, y) in a.data().iter().zip(b.data().iter()) {
            assert!((x - y).abs() < eps, "{x} vs {y}");
        }
    }

    #[test]
    fn roundtrip_safetensors_and_disk() {
        let mut seed = 99u32;
        let cfg = MiniGptConfig {
            vocab_size: 48,
            d_model: 16,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 32,
            max_seq: 16,
        };
        let m1 = MiniGpt::random(cfg.clone(), &mut seed).unwrap();
        let bytes = minigpt_to_safetensors_bytes(&m1).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut map = HashMap::new();
        for (name, tv) in st.tensors() {
            map.insert(name.clone(), tensor_from_safetensors_view(&name, tv).unwrap());
        }
        let m2 = mini_gpt_from_state_dict(cfg.clone(), map).unwrap();
        let d1 = state_dict(&m1);
        let d2 = state_dict(&m2);
        assert_eq!(d1.len(), d2.len());
        for (k, t1) in &d1 {
            assert_tensors_close(t1, d2.get(k).unwrap(), 1e-6);
        }

        let dir = std::env::temp_dir().join(format!("rusty_ai_ckpt_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        save_minigpt_checkpoint(&dir, &m1).unwrap();
        let m3 = load_minigpt_checkpoint(&dir).unwrap();
        let d3 = state_dict(&m3);
        for (k, t1) in &d1 {
            assert_tensors_close(t1, d3.get(k).unwrap(), 1e-6);
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
