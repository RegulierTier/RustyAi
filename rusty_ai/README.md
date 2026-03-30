# `rusty_ai` (Meta-Crate)

**Sammel-Crate** für den ML-/LLM-Kern des Workspaces: Re-Exports aus `rusty_ai_core`, `rusty_ai_autograd`, `rusty_ai_nn`, `rusty_ai_ml` und `rusty_ai_llm` unter den Modulnamen `core`, `autograd`, `nn`, `ml`, `llm`. Häufig genutzte Typen und Funktionen stehen auch auf der Crate-Root (`Tensor`, `Variable`, `MiniGpt`, `generate`, …).

| Weiterlesen | Inhalt |
| ----------- | ------ |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** | Architektur, alle Crates, typische Abläufe (MLP, Mini-GPT, GPT-2), Glossar |
| **[`README.md`](../README.md)** (Repo-Root) | Schnellstart, Workspace-Übersicht, Roadmap, Dokumentations-Index |
| **`rusty_ai_llm`** | [`../rusty_ai_llm/README.md`](../rusty_ai_llm/README.md) — Features `hf-hub`, `gpt2-bpe`, Generierung |
| **Agent / IDE** | [`../rusty_ai_agent/README.md`](../rusty_ai_agent/README.md) — `LlmBackend`, Tools, optional `http` |

## Features (`Cargo.toml`)

| Feature | Wirkung |
| ------- | ------- |
| *(default)* | Kern-API ohne Candle, ohne Hub, ohne BPE-Tokenizer. |
| **`candle`** | Bindet `rusty_ai_backend_candle`; Zugriff über `rusty_ai::candle`. |
| **`candle-cuda`** | Wie `candle`, Candle mit CUDA. |
| **`hf-hub`** | Download von RustyAi-Checkpoints vom Hugging Face Hub (`load_minigpt_from_hf`). |
| **`gpt2-bpe`** | `Gpt2Tokenizer`, `generate_gpt2_text`, `Gpt2PipelineError` (über `rusty_ai_llm`). |

Details und Feature-Matrix: [Handbuch — Abschnitt 8](../docs/HANDBUCH.md) (*Feature-Matrix Meta-Crate `rusty_ai`*).

## Beispiele

```bash
cargo run -p rusty_ai --example train_mlp
cargo run -p rusty_ai --example train_mini_gpt
```

GPT-2-Gewichte und BPE: `cargo build -p rusty_ai --features gpt2-bpe` — siehe [Handbuch — GPT-2 mit HF-Gewichten](../docs/HANDBUCH.md) (Abschnitt *3.2.1*).

## `rustdoc`

```bash
cargo doc -p rusty_ai --no-deps --open
```

Mit allen optionalen Features: `--features candle,hf-hub,gpt2-bpe` (nach Bedarf).
