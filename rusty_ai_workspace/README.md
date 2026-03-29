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

## Einordnung

- **Handbuch:** [§2.9 `rusty_ai_workspace`](../docs/HANDBUCH.md) — typische Nutzung neben `rusty_ai_agent`.
- **Roadmap:** [Phase 2 / 3](../docs/ARCHITEKTUR_IDE_ROADMAP_B.md).
- **Konfiguration:** [`IndexConfig`](src/lib.rs) — `root`, `max_chunk_lines`, `overlap_lines`, optionale Dateiendungen (`extensions`).
