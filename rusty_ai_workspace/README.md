# `rusty_ai_workspace`

**Workspace-Index** (Pfad B, Phase 2): Zeilen-Chunks aus Dateien unter einem Root, **Substring-Suche**, optional **HTTP-Embeddings** (OpenAI-kompatibel) mit Cosinus-Top-k.

| Feature | Wirkung |
| ------- | ------- |
| *(default)* | `walkdir`, Chunking, `search_substring` |
| **`embeddings`** | `reqwest` + [`embeddings::HttpEmbeddingClient`](src/lib.rs) |

```bash
cargo test -p rusty_ai_workspace
cargo test -p rusty_ai_workspace --features embeddings
cargo run -p rusty_ai_workspace --example workspace_index_demo
```

Verzeichnisse **`target/`**, **`.git/`**, **`node_modules/`** werden beim Walk übersprungen.

### Index-Cache (Phase 3)

[`WorkspaceIndex::build_cached`](src/lib.rs) speichert unter `cache_dir`:

| Datei | Inhalt |
| ----- | ------ |
| `index_manifest.json` | Version, Root-Pfad (String), `config_fingerprint`, `max_mtime_secs` der indexierten Dateien |
| `index_chunks.json` | Serialisierter [`WorkspaceIndex`](src/lib.rs) (Chunks als JSON) |

**Invalidierung:** Cache wird verworfen, wenn Manifest fehlt, Root/Fingerprint/`max_mtime` nicht zur aktuellen Dateiwelt passen, oder wenn `force_rebuild == true` — dann erfolgt voller Neuaufbau wie bei [`WorkspaceIndex::build`](src/lib.rs) und erneutes Schreiben.

```rust
// Pseudocode — siehe API in lib.rs
let index = WorkspaceIndex::build_cached(&config, Path::new("target/my_index_cache"), false)?;
```

Mit Feature **`embeddings`**: [`CachingEmbeddingClient`](src/lib.rs) umschließt einen [`HttpEmbeddingClient`](src/lib.rs) und puffert Embedding-Vektoren pro Text-Hash (In-Memory; reduziert wiederholte HTTP-Calls).

## Einordnung

- **Handbuch:** [§2.9 `rusty_ai_workspace`](../docs/HANDBUCH.md) — typische Nutzung neben `rusty_ai_agent`.
- **Roadmap:** [Phase 2 / 3](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md).
- **Konfiguration:** [`IndexConfig`](src/lib.rs) — `root`, `max_chunk_lines`, `overlap_lines`, optionale Dateiendungen (`extensions`).
