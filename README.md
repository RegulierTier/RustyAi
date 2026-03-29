# RustyAi

Ein **Rust-Workspace** für maschinelles Lernen und kleine Sprachmodelle: CPU-Tensoren, einfache automatische Differentiation, Schichten, Optimierer, ein Mini-Transformer (Decoder-only) und Textgenerierung. Ziel ist **verständlicher Eigencode** auf der CPU, nicht ein Wrapper um große Frameworks.

**Lizenz:** MIT oder Apache-2.0 (siehe Workspace-`Cargo.toml`).

---

## Funktionen

| Bereich | Inhalt |
| ------- | ------ |
| **Tensor & Ops** | `f32`-Tensoren (row-major), Broadcasting, `matmul` (2D + Batch-3D), Aktivierungen, `softmax` / `log_softmax`, `mse`, … |
| **Autograd** | `Variable`, dynamischer Graph, `backward`, u. a. `MatMul`, `Mul`, `Add` (Broadcast-Gradienten), GELU, `layer_norm` / `layer_norm_affine`, Softmax, Embedding-Gather, Split/Merge-Heads, Cross-Entropy (Next-Token), `no_grad` für Inferenz |
| **NN** | `Linear`, GELU, `layer_norm` / `layer_norm_affine` (γ/β wie PyTorch), `ones_scale` / `zeros_bias`, Xavier-Initialisierung |
| **ML** | `Sgd`, `Adam`, Batch-Iterator, einfache Spalten-Normalisierung |
| **LLM** | Byte-Tokenizer; optional **`gpt2-bpe`**: `tokenizer.json` / `Gpt2Tokenizer` + `generate_from_ids` / `generate_gpt2_text` für GPT-2-BPE-Parität (Rust-Crate `tokenizers`). Causal-Attention, `MiniGpt`, `TrainableMiniGpt`, `generate` (Temperatur, top-p, **KV-Cache** nach Prefill) |
| **Checkpoints** | `save_minigpt_checkpoint` / `load_minigpt_checkpoint` (`config.json` + `model.safetensors`); optional Feature **`hf-hub`** zum Herunterladen aus dem Hugging Face Hub |
| **GPT-2-Import** | `load_minigpt_from_gpt2_safetensors` — HF-GPT-2-Gewichte (fused QKV → getrennte Matrizen). **BPE:** Feature `gpt2-bpe` auf `rusty_ai` / `rusty_ai_llm` + `Gpt2Tokenizer::from_model_dir` (Ordner mit `model.safetensors` und `tokenizer.json`) und `generate_gpt2_text` bzw. `generate_from_ids` |
| **Candle-Backend** | Crate `rusty_ai_backend_candle`: CPU- oder CUDA-**Matmul**, **FP8 E4M3**-Hilfen, Referenz-**All-Reduce-Mittelwert** für Datenparallelität; optional `rusty_ai` mit `--features candle` bzw. `candle-cuda` |

---

## LLM: Textgenerierung (Kurzüberblick)

`MiniGpt` ist ein kleiner **Decoder-only**-Transformer mit zufälliger Initialisierung. Zum **Trainieren** (Next-Token-Cross-Entropy, gleicher Forward wie `MiniGpt::forward`) dient **`TrainableMiniGpt`** plus `Variable::cross_entropy_next_token`; Beispiel: `cargo run -p rusty_ai --example train_mini_gpt`. Die Funktion **`generate`** arbeitet für Inferenz in zwei Phasen:

1. **Prefill:** Der Prompt wird einmal vollständig durch das Modell geschoben; pro Schicht werden Keys und Values im **`KvCache`** gespeichert.
2. **Decode:** Jedes neu generierte Token läuft nur mit Sequenzlänge 1 durch die Blöcke; K/V werden an den Cache angehängt — deutlich weniger Arbeit pro Schritt als bei wiederholter Vorwärtsrechnung über den ganzen Kontext.

```rust
use rusty_ai::{generate, MiniGpt, MiniGptConfig};

let mut seed = 1u32;
let model = MiniGpt::random(MiniGptConfig::default(), &mut seed).unwrap();
let text = generate(&model, "Hallo ", 32, 0.8, 0.95, &mut seed).unwrap();
```

**GPT-2-Gewichte + BPE** (`cargo build -p rusty_ai --features gpt2-bpe`): `MiniGptConfig` zur Checkpoint-Größe (z. B. `vocab_size: 50257`), `load_minigpt_from_gpt2_safetensors("…/model.safetensors", cfg)`, `Gpt2Tokenizer::from_model_dir("…/")` (enthält `tokenizer.json`), dann `generate_gpt2_text(&model, &tok, prompt, max_new, temp, top_p, &mut seed)`.

Manuelle Inferenz (z. B. eigenes Sampling) über **`forward_prefill`** / **`forward_decode_step`** und **`KvCache::new(model.cfg.n_layers)`** ist im **[Handbuch](docs/HANDBUCH.md)** unter „LLM: Vorwärtsrechnung / Sampling“ beschrieben.

---

## Voraussetzungen

- [Rust](https://www.rust-lang.org/tools/install) (stabiler Toolchain; Edition 2021)

---

## Schnellstart

Repository klonen, im Workspace-Verzeichnis:

```bash
cargo build --workspace
cargo test --workspace
```

**MLP-Beispiel** (synthetische Regression):

```bash
cargo run -p rusty_ai --example train_mlp
```

**Mini-GPT-Beispiel** (kurze Byte-Sequenz, Adam, Next-Token-Loss):

```bash
cargo run -p rusty_ai --example train_mini_gpt
```

---

## Workspace-Struktur

| Crate | Rolle |
| ----- | ----- |
| `rusty_ai` | Sammel-Crate: Re-Exports der übrigen Pakete |
| `rusty_ai_agent` | Phase-0-Agent-Protokoll: `LlmBackend`, Chat-/Tool-JSON (IDE-Roadmap) |
| `rusty_ai_core` | Tensoren, Formen, numerische Operationen |
| `rusty_ai_autograd` | `Variable`, Rückwärtsrechnung |
| `rusty_ai_nn` | Schichten & Aktivierungen |
| `rusty_ai_ml` | Optimierer & Daten-Helfer |
| `rusty_ai_llm` | Transformer, Tokenizer, Generierung, safetensors-Checkpoints, GPT-2-Mapping — siehe [`rusty_ai_llm/README.md`](rusty_ai_llm/README.md) |
| `rusty_ai_backend_candle` | Optional: Candle (CPU/CUDA), FP8, verteilte Referenz-Ops |

Abhängigkeit der Kernbibliothek: u. a. [`matrixmultiply`](https://crates.io/crates/matrixmultiply) für schnelles CPU-Matmul.

---

## Dokumentation

| Ressource | Inhalt |
| --------- | ------ |
| **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](docs/ARCHITEKTUR_IDE_ROADMAP_B.md)** | Pfad B (IDE-nah): Zielarchitektur, Tool-Loops, Roadmap in Phasen |
| **[`rusty_ai_agent/README.md`](rusty_ai_agent/README.md)** | `LlmBackend`-Trait, Tool-Protokoll, JSON Schema (`schemas/`) |
| **[`docs/HANDBUCH.md`](docs/HANDBUCH.md)** | Architektur, Crate-Referenz, Abläufe (MLP, Mini-GPT-Training, LLM-Inferenz mit KV-Cache), Checkpoints/GPT-2/Candle, Grenzen, Glossar |
| **[`docs/README.md`](docs/README.md)** | Kurzüberblick und Verweise auf weiterführende Dateien |
| **[`docs/BERICHT_PRÜFUNG.md`](docs/BERICHT_PRÜFUNG.md)** | Prüfbericht zur Scope-Erweiterung (Tests, Korrekturen) |
| **[`rusty_ai_backend_candle/README.md`](rusty_ai_backend_candle/README.md)** | Optionales Candle-Backend (Features, API-Kurzüberblick) |

Rust-API-Dokumentation lokal erzeugen:

```bash
cargo doc --workspace --no-deps --open
```

---

## Erweiterter Scope (experimentell)

- **Training** bleibt in den Kern-Crates **CPU-Autograd** (`TrainableMiniGpt`). GPU/FP8 und größere Matmuls laufen über das **optionale** Candle-Crate; produktives Multi-GPU-Training nutzt typischerweise NCCL (Candle-Feature `nccl`) oder externe Orchestrierung.
- **Checkpoints:** Eigenes Format `config.json` + `model.safetensors` (RustyAi-`model_type`: `rusty_ai_minigpt`).
- **Hugging Face:** GPT-2-`safetensors` können mit `load_minigpt_from_gpt2_safetensors` geladen werden, wenn die Hyperparameter zur gewählten `MiniGptConfig` passen (typisch `vocab_size` 50257). Mit **`gpt2-bpe`** lädt `Gpt2Tokenizer` dieselbe `tokenizer.json` wie die übliche HF-Pipeline (Implementierung: Rust-Crate **`tokenizers`**, ohne Python). **`load_minigpt_from_hf`** (Feature `hf-hub`) lädt **RustyAi**-Checkpoints aus einem Repo, nicht beliebige GPT-2-Archivformate.

---

## Mitwirken

Issues und Pull Requests sind willkommen. `cargo clippy --workspace --all-targets` und `cargo test --workspace` sollten vor einem Merge grün sein. Öffentliche API- oder Architekturänderungen idealerweise in [`docs/HANDBUCH.md`](docs/HANDBUCH.md) (und bei Bedarf hier im README) nachziehen.
