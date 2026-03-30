//! Zeilenbasierter Workspace-Index für Retrieval vor LLM-Aufrufen (Phase 2).
//!
//! - [`WorkspaceIndex::build`] — Dateien unter einem Root, Chunks mit Überlappung
//! - [`WorkspaceIndex::search_substring`] — einfache Textsuche
//! - Feature **`embeddings`**: HTTP-Embeddings (OpenAI-kompatibel) + Cosinus-Top-k
//! - Phase 3: [`WorkspaceIndex::build_cached`] — optionaler Disk-Cache (Manifest + max. mtime)

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

fn path_has_ignored_segment(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        matches!(s.as_ref(), "target" | ".git" | "node_modules")
    })
}

/// Konfiguration für [`WorkspaceIndex::build`].
#[derive(Clone, Debug, Hash)]
pub struct IndexConfig {
    /// Wurzelverzeichnis (rekursiv).
    pub root: PathBuf,
    /// Zeilen pro Chunk (mindestens 1).
    pub max_chunk_lines: usize,
    /// Überlappung zwischen aufeinanderfolgenden Chunks (Zeilen).
    pub overlap_lines: usize,
    /// Nur diese Endungen, z. B. `["rs", "md"]` — `None` = alle Dateien (binär wird UTF-8 lossy gelesen).
    pub extensions: Option<Vec<String>>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            max_chunk_lines: 48,
            overlap_lines: 8,
            extensions: Some(vec!["rs".into(), "md".into(), "toml".into()]),
        }
    }
}

/// Ein Textausschnitt mit Positionsangabe (Zeilen **1-basiert** für Anzeige).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextChunk {
    pub path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

/// Alle Chunks eines Workspaces.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkspaceIndex {
    chunks: Vec<TextChunk>,
}

impl WorkspaceIndex {
    pub fn chunks(&self) -> &[TextChunk] {
        &self.chunks
    }

    /// Indexiert lesbare Dateien (UTF-8 mit `lossy` bei Fehlern).
    pub fn build(config: &IndexConfig) -> Result<Self, std::io::Error> {
        let mut chunks = Vec::new();

        for entry in WalkDir::new(&config.root)
            .into_iter()
            .filter_entry(|e| !path_has_ignored_segment(e.path()))
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_dir() {
                continue;
            }
            if let Some(exts) = &config.extensions {
                let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("");
                if !exts.iter().any(|e| e == ext) {
                    continue;
                }
            }
            let rel = path.to_path_buf();
            let data = fs::read_to_string(path)?;
            push_chunks(
                &mut chunks,
                &rel,
                &data,
                config.max_chunk_lines,
                config.overlap_lines,
            )?;
        }

        Ok(Self { chunks })
    }

    /// Wie [`Self::build`], aber lädt/speichert unter `cache_dir`, wenn Manifest + `max_mtime` zur aktuellen Dateiwelt passen.
    ///
    /// Bei `force_rebuild == true` oder veraltetem Cache: neu bauen und schreiben.
    pub fn build_cached(
        config: &IndexConfig,
        cache_dir: &Path,
        force_rebuild: bool,
    ) -> Result<Self, std::io::Error> {
        fs::create_dir_all(cache_dir)?;
        let current_max = scan_max_mtime_secs(config)?;
        let fp = fingerprint_index_config(config);
        let manifest_path = cache_dir.join("index_manifest.json");
        let chunks_path = cache_dir.join("index_chunks.json");

        if !force_rebuild && manifest_path.exists() && chunks_path.exists() {
            let m: IndexManifest = serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
            let root_s = config.root.to_string_lossy();
            if m.version == 1
                && m.root == root_s.as_ref()
                && m.config_fingerprint == fp
                && m.max_mtime_secs == current_max
            {
                let data = fs::read_to_string(&chunks_path)?;
                let index: WorkspaceIndex = serde_json::from_str(&data)?;
                return Ok(index);
            }
        }

        let index = Self::build(config)?;
        let manifest = IndexManifest {
            version: 1,
            root: config.root.to_string_lossy().into_owned(),
            config_fingerprint: fp,
            max_mtime_secs: current_max,
        };
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
        )?;
        fs::write(
            &chunks_path,
            serde_json::to_string_pretty(&index)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?,
        )?;
        Ok(index)
    }

    /// Substring-Suche über alle Chunks (`query` leer → leeres Ergebnis).
    pub fn search_substring<'a>(
        &'a self,
        query: &str,
        case_insensitive: bool,
    ) -> Vec<&'a TextChunk> {
        if query.is_empty() {
            return Vec::new();
        }
        let needle = if case_insensitive {
            query.to_lowercase()
        } else {
            query.to_string()
        };
        self.chunks
            .iter()
            .filter(|c| {
                if case_insensitive {
                    c.text.to_lowercase().contains(&needle)
                } else {
                    c.text.contains(&needle)
                }
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
struct IndexManifest {
    version: u32,
    root: String,
    config_fingerprint: u64,
    max_mtime_secs: u64,
}

fn fingerprint_index_config(config: &IndexConfig) -> u64 {
    let mut h = DefaultHasher::new();
    config.hash(&mut h);
    h.finish()
}

fn scan_max_mtime_secs(config: &IndexConfig) -> Result<u64, std::io::Error> {
    let mut max = 0u64;
    for entry in WalkDir::new(&config.root)
        .into_iter()
        .filter_entry(|e| !path_has_ignored_segment(e.path()))
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        if let Some(exts) = &config.extensions {
            let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("");
            if !exts.iter().any(|e| e == ext) {
                continue;
            }
        }
        let meta = fs::metadata(path)?;
        if let Ok(t) = meta.modified() {
            let secs = t
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            max = max.max(secs);
        }
    }
    Ok(max)
}

fn push_chunks(
    out: &mut Vec<TextChunk>,
    path: &Path,
    data: &str,
    max_chunk_lines: usize,
    overlap_lines: usize,
) -> Result<(), std::io::Error> {
    let max = max_chunk_lines.max(1);
    let overlap = overlap_lines.min(max.saturating_sub(1));
    let lines: Vec<&str> = data.lines().collect();
    if lines.is_empty() {
        return Ok(());
    }
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + max).min(lines.len());
        let slice = &lines[start..end];
        let text = slice.join("\n");
        out.push(TextChunk {
            path: path.to_path_buf(),
            start_line: start + 1,
            end_line: end,
            text,
        });
        if end >= lines.len() {
            break;
        }
        let next = end.saturating_sub(overlap);
        if next <= start {
            break;
        }
        start = next;
    }
    Ok(())
}

/// Cosinus-Ähnlichkeit zweier gleich langer Vektoren.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(feature = "embeddings")]
pub mod embeddings {
    //! OpenAI-kompatibles HTTP-Embeddings (`POST …/embeddings`).

    use std::collections::hash_map::DefaultHasher;

    use super::{cosine_similarity, TextChunk, WorkspaceIndex};
    use serde::Deserialize;

    /// Fehler bei Embedding-HTTP.
    #[derive(Debug)]
    pub struct EmbeddingError(pub String);

    impl std::fmt::Display for EmbeddingError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for EmbeddingError {}

    /// Trait für austauschbare Embedding-Backends.
    pub trait EmbeddingClient {
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    }

    /// Konfiguration für OpenAI-kompatibles Embedding-API.
    #[derive(Clone, Debug)]
    pub struct OpenAiEmbeddingConfig {
        pub base_url: String,
        pub api_key: Option<String>,
        pub model: String,
    }

    impl Default for OpenAiEmbeddingConfig {
        fn default() -> Self {
            Self {
                base_url: "https://api.openai.com/v1".into(),
                api_key: std::env::var("OPENAI_API_KEY").ok(),
                model: "text-embedding-3-small".into(),
            }
        }
    }

    #[derive(Deserialize)]
    struct EmbeddingResponse {
        data: Vec<EmbeddingData>,
    }

    #[derive(Deserialize)]
    struct EmbeddingData {
        embedding: Vec<f32>,
    }

    /// HTTP-Client (`reqwest` blocking).
    pub struct HttpEmbeddingClient {
        pub config: OpenAiEmbeddingConfig,
    }

    /// In-Memory-Cache pro Text-Hash (std-only); reduziert API-Aufrufe bei wiederholten Chunks.
    pub struct CachingEmbeddingClient<C> {
        pub inner: C,
        cache: std::sync::Mutex<std::collections::HashMap<u64, Vec<f32>>>,
    }

    impl<C> CachingEmbeddingClient<C> {
        pub fn new(inner: C) -> Self {
            Self {
                inner,
                cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }
    }

    fn hash_embed_text(s: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }

    impl<C: EmbeddingClient> EmbeddingClient for CachingEmbeddingClient<C> {
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                let key = hash_embed_text(t);
                let mut cache = self
                    .cache
                    .lock()
                    .map_err(|e| EmbeddingError(e.to_string()))?;
                if let Some(v) = cache.get(&key) {
                    out.push(v.clone());
                } else {
                    drop(cache);
                    let v = self
                        .inner
                        .embed_batch(&[*t])?
                        .into_iter()
                        .next()
                        .ok_or_else(|| EmbeddingError("empty embedding batch".into()))?;
                    cache = self
                        .cache
                        .lock()
                        .map_err(|e| EmbeddingError(e.to_string()))?;
                    cache.insert(key, v.clone());
                    out.push(v);
                }
            }
            Ok(out)
        }
    }

    impl EmbeddingClient for HttpEmbeddingClient {
        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let url = format!("{}/embeddings", self.config.base_url.trim_end_matches('/'));
            let client = reqwest::blocking::Client::builder()
                .build()
                .map_err(|e| EmbeddingError(e.to_string()))?;
            let mut req = client.post(&url).json(&serde_json::json!({
                "model": self.config.model,
                "input": texts,
            }));
            if let Some(ref k) = self.config.api_key {
                req = req.bearer_auth(k);
            }
            let resp = req.send().map_err(|e| EmbeddingError(e.to_string()))?;
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            if !status.is_success() {
                return Err(EmbeddingError(format!("HTTP {}: {}", status, body)));
            }
            let r: EmbeddingResponse =
                serde_json::from_str(&body).map_err(|e| EmbeddingError(e.to_string()))?;
            let mut out = Vec::with_capacity(r.data.len());
            for d in r.data {
                out.push(d.embedding);
            }
            Ok(out)
        }
    }

    /// Index + Vektoren pro Chunk (in-memory).
    #[derive(Debug)]
    pub struct EmbeddingIndex {
        pub chunks: Vec<TextChunk>,
        pub vectors: Vec<Vec<f32>>,
    }

    impl EmbeddingIndex {
        /// Erzeugt Embeddings für alle Chunks eines [`WorkspaceIndex`].
        pub fn from_workspace(
            index: &WorkspaceIndex,
            client: &impl EmbeddingClient,
        ) -> Result<Self, EmbeddingError> {
            let texts: Vec<&str> = index.chunks().iter().map(|c| c.text.as_str()).collect();
            if texts.is_empty() {
                return Ok(Self {
                    chunks: Vec::new(),
                    vectors: Vec::new(),
                });
            }
            let vectors = client.embed_batch(&texts)?;
            if vectors.len() != index.chunks().len() {
                return Err(EmbeddingError(format!(
                    "expected {} vectors, got {}",
                    index.chunks().len(),
                    vectors.len()
                )));
            }
            Ok(Self {
                chunks: index.chunks().to_vec(),
                vectors,
            })
        }

        /// Top-k ähnlichste Chunks zu einem Abfragevektor (höhere Werte = ähnlicher).
        pub fn search_top_k(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
            let mut scored: Vec<(usize, f32)> = self
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(query, v)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k.min(scored.len()));
            scored
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_overlap() {
        let mut out = Vec::new();
        let data = "a\nb\nc\nd\ne\nf\n";
        push_chunks(&mut out, Path::new("x.rs"), data, 5, 2).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }
}
