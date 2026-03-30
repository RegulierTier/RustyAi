# `rusty_ai` (Meta-Crate)

**Sammel-Crate** für den ML-/LLM-Kern des Workspaces: Re-Exports aus `rusty_ai_core`, `rusty_ai_autograd`, `rusty_ai_nn`, `rusty_ai_ml` und `rusty_ai_llm` unter den Modulnamen `core`, `autograd`, `nn`, `ml`, `llm`. Häufig genutzte Typen und Funktionen stehen auch auf der Crate-Root (`Tensor`, `Variable`, `MiniGpt`, `generate`, `load_minigpt_checkpoint`, `load_minigpt_checkpoint_bytes`, …).

| Weiterlesen | Inhalt |
| ----------- | ------ |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md)** | Architektur, alle Crates, typische Abläufe (MLP, Mini-GPT, **FIM**, GPT-2, lokales Mini-Bundle, Agent) |
| **[`README.md`](../README.md)** (Repo-Root) | Schnellstart, Workspace-Tabelle, Dokumentations-Index |
| **`rusty_ai_llm`** | [`../rusty_ai_llm/README.md`](../rusty_ai_llm/README.md) — Features `hf-hub`, `gpt2-bpe`, Generierung, Mini-Bundle |
| **Agent / IDE** | [`../rusty_ai_agent/README.md`](../rusty_ai_agent/README.md) — `LlmBackend`, Tools, optional `http` |

## Features (`Cargo.toml`)

| Feature | Wirkung |
| ------- | ------- |
| *(default)* | Kern-API ohne Candle, ohne Hub, ohne BPE-Tokenizer. |
| **`candle`** | Bindet `rusty_ai_backend_candle`; Zugriff über `rusty_ai::candle`. Kein gemeinsames Training mit `TrainableMiniGpt` — siehe [Handbuch §2.6](../docs/HANDBUCH.md). |
| **`candle-cuda`** | Wie `candle`, Candle mit CUDA. |
| **`hf-hub`** | Download von RustyAi-Checkpoints vom Hugging Face Hub (`load_minigpt_from_hf`). |
| **`gpt2-bpe`** | `Gpt2Tokenizer`, `generate_gpt2_text`, `Gpt2PipelineError` (über `rusty_ai_llm`). |

Details: [Handbuch — Abschnitt 8](../docs/HANDBUCH.md) (*Feature-Matrix Meta-Crate `rusty_ai`*).

## Beispiele (`examples/`)

| Datei / Name | Beschreibung |
| ------------ | ------------- |
| `train_mlp` | Regression / kleines MLP mit Autograd |
| `train_mini_gpt` | Next-Token-Training auf Byte-Sequenz (`TrainableMiniGpt`) |
| `train_micro_checkpoint` | Profil `MiniGptConfig::micro_local()`, kurzes Training, schreibt `assets/mini_local/` |
| `mini_local_inference` | Lädt `config.json` + `model.safetensors` per `include_str!` / `include_bytes!`, **ohne Netz** |

```bash
cargo run -p rusty_ai --example train_mlp
cargo run -p rusty_ai --example train_mini_gpt
cargo run -p rusty_ai --example train_micro_checkpoint
cargo run -p rusty_ai --example mini_local_inference
```

**Mini-Assets erneuern (Maintainer):** Zufallsinitialisierung einchecken: `cargo test -p rusty_ai_llm bootstrap_rusty_ai_mini_local_assets -- --ignored`. Oder trainierte Gewichte: `train_micro_checkpoint` ausführen und Diff prüfen.

GPT-2-Gewichte und BPE: `cargo build -p rusty_ai --features gpt2-bpe` — siehe [Handbuch — GPT-2 mit HF-Gewichten](../docs/HANDBUCH.md) (Abschnitt *3.2.1*).

## `rustdoc`

```bash
cargo doc -p rusty_ai --no-deps --open
```

Mit allen optionalen Features: `--features candle,hf-hub,gpt2-bpe` (nach Bedarf).
