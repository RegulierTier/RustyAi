# RustyAi

Ein **Rust-Workspace** für maschinelles Lernen und kleine Sprachmodelle: CPU-Tensoren, einfache automatische Differentiation, Schichten, Optimierer, ein Mini-Transformer (Decoder-only) und Textgenerierung. Ziel ist **verständlicher Eigencode** auf der CPU, nicht ein Wrapper um große Frameworks.

**Lizenz:** MIT oder Apache-2.0 (siehe Workspace-`Cargo.toml`).

---

## Funktionen

| Bereich | Inhalt |
| ------- | ------ |
| **Tensor & Ops** | `f32`-Tensoren (row-major), Broadcasting, `matmul` (2D + Batch-3D), Aktivierungen, `softmax` / `log_softmax`, `mse`, … |
| **Autograd** | `Variable`, dynamischer Graph, `backward`, `BiasAdd` für Zeilen-Bias, `no_grad` für Inferenz |
| **NN** | `Linear`, GELU, LayerNorm (ohne lernbare γ/β), Xavier-Initialisierung |
| **ML** | `Sgd`, `Adam`, Batch-Iterator, einfache Spalten-Normalisierung |
| **LLM** | Byte-Tokenizer, Multi-Head-Causal-Attention, `MiniGpt`, `generate` (Temperatur, top-p), optional KV-Struktur als Erweiterungspunkt |

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

---

## Workspace-Struktur

| Crate | Rolle |
| ----- | ----- |
| `rusty_ai` | Sammel-Crate: Re-Exports der übrigen Pakete |
| `rusty_ai_core` | Tensoren, Formen, numerische Operationen |
| `rusty_ai_autograd` | `Variable`, Rückwärtsrechnung |
| `rusty_ai_nn` | Schichten & Aktivierungen |
| `rusty_ai_ml` | Optimierer & Daten-Helfer |
| `rusty_ai_llm` | Transformer, Tokenizer, Generierung |

Abhängigkeit der Bibliothek: u. a. [`matrixmultiply`](https://crates.io/crates/matrixmultiply) für schnelles CPU-Matmul.

---

## Dokumentation

Ausführliches **Handbuch** (Architektur, Module, typische Abläufe, Grenzen):

→ **[`docs/HANDBUCH.md`](docs/HANDBUCH.md)**

Rust-API-Dokumentation lokal erzeugen:

```bash
cargo doc --workspace --no-deps --open
```

---

## Nicht im Scope (v1)

Verteiltes Training, GPU-Kernels, FP8, Laden beliebiger Hugging-Face-Checkpoints. Die Basis ist **CPU-first** und für Lernzwecke sowie Prototypen gedacht.

---

## Mitwirken

Issues und Pull Requests sind willkommen. `cargo clippy --workspace --all-targets` und `cargo test --workspace` sollten vor einem Merge grün sein.
