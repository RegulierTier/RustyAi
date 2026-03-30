# RustyAi

**RustyAi** ist ein **Rust-Workspace**, in dem du kleine neuronale Netze und **Decoder-only-Sprachmodelle** (Mini-Transformer) auf der **CPU** verstehen, trainieren und ausführen kannst — mit nachvollziehbarem Eigencode statt Black-Box-Framework. Ergänzend gibt es ein **Agent-Protokoll** (`rusty_ai_agent`) für IDE-nahe Orchestrierung: Tools, Policies und optionale Anbindung an OpenAI-kompatible APIs.

**Lizenz:** MIT oder Apache-2.0 (siehe Workspace-`Cargo.toml`).

---

## Was ist das?

| Du bekommst … | Kurz |
| ------------- | ---- |
| **Kern-ML** | `f32`-Tensoren, Broadcasting, `matmul`, Autograd (`Variable`, `backward`), Schichten (`Linear`, GELU, LayerNorm), Optimierer (SGD, Adam) |
| **Mini-GPT / LLM** | `MiniGpt`, Training mit Next-Token-Loss, Textgenerierung mit **KV-Cache** (Prefill + Decode), optional **GPT-2-Gewichte + BPE** (`tokenizers`) |
| **Checkpoints** | Eigenes Format (`config.json` + `model.safetensors`), optional Hugging Face Hub |
| **Optional: Candle** | Separates Crate für CPU/CUDA-Matmul, FP8-Hilfen, Referenz-All-Reduce |
| **Optional: Agent** | Trait `LlmBackend`, JSON-Tools (`read_file`, `run_cmd`, …), Allowlist, HTTP-Backend, Retry/Telemetrie — für **eigenes** IDE-/CLI-Produkt, nicht als vollständiger Chat-Client |

Es ist **kein** Ersatz für PyTorch/JAX und **kein** fertiger Chatbot — sondern eine **lern- und erweiterbare Codebasis** für Experimente, Demos und die Schnittstelle zu größeren LLMs (Pfad B in der Architektur-Doku).

---

## Warum existiert dieses Projekt?

- **Transparenz:** Der Trainings- und Inferenzpfad soll im Code lesbar bleiben (CPU-Autograd, klarer `generate`-Ablauf mit KV-Cache).
- **Rust-only-Stack** für viele Workflows: kein Python-Zwang für Training/Demos; optional `tokenizers` (Rust) für GPT-2-BPE.
- **Klare Grenzen:** Großes LLM-Training auf GPU gehört nicht hierher; stattdessen **Schnittstellen** (`rusty_ai_agent`) für Orchestrierung und externe Modelle — siehe [Architektur Roadmap B](docs/ARCHITEKTUR_IDE_ROADMAP_B.md).

---

## Features

| Bereich | Inhalt |
| ------- | ------ |
| **Tensor & Ops** | `f32`-Tensoren (row-major), Broadcasting, `matmul` (2D + Batch-3D), Aktivierungen, `softmax` / `log_softmax`, `mse`, … |
| **Autograd** | `Variable`, dynamischer Graph, `backward`, u. a. `MatMul`, `Mul`, `Add` (Broadcast-Gradienten), GELU, `layer_norm` / `layer_norm_affine`, Softmax, Embedding-Gather, Split/Merge-Heads, Cross-Entropy (Next-Token), `no_grad` für Inferenz |
| **NN** | `Linear`, GELU, `layer_norm` / `layer_norm_affine` (γ/β wie PyTorch), `ones_scale` / `zeros_bias`, Xavier-Initialisierung |
| **ML** | `Sgd`, `Adam`, Batch-Iterator, einfache Spalten-Normalisierung |
| **LLM** | Byte-Tokenizer; optional **`gpt2-bpe`**: `tokenizer.json` / `Gpt2Tokenizer` + `generate_from_ids` / `generate_gpt2_text`. Causal-Attention, `MiniGpt`, `TrainableMiniGpt`, `generate` (Temperatur, top-p, **KV-Cache** nach Prefill) |
| **Checkpoints** | `save_minigpt_checkpoint` / `load_minigpt_checkpoint`; optional Feature **`hf-hub`** |
| **GPT-2-Import** | `load_minigpt_from_gpt2_safetensors` — HF-GPT-2-Gewichte; BPE mit Feature `gpt2-bpe` |
| **Candle-Backend** | `rusty_ai_backend_candle`: CPU/CUDA-**Matmul**, **FP8 E4M3**, Referenz-**All-Reduce**; optional `rusty_ai` mit `--features candle` / `candle-cuda` |
| **Agent (`rusty_ai_agent`)** | `LlmBackend`, Tool-JSON, Policy, optional **`http`** (OpenAI-kompatibel) und **`real-exec`** — Details: [`rusty_ai_agent/README.md`](rusty_ai_agent/README.md) |
| **Agent: Betrieb (Phase 3)** | Benannte Policies (`RUSTY_AI_AGENT_POLICY`), [`BatchReport`](rusty_ai_agent/src/batch/batch_report.rs) für CI-Artefakte, [`BudgetLlmBackend`](rusty_ai_agent/src/batch/budget.rs) für Token-/Aufruf-Limits — siehe [`rusty_ai_agent/README.md`](rusty_ai_agent/README.md#phase-3-betrieb-und-ci), [`SECURITY.md`](rusty_ai_agent/SECURITY.md) |

---

## Beispiel

Es gibt **kein eingebettetes Screenshot** (Bibliothek & CLI, kein separates UI). Stattdessen: **echtes Terminal-Output** aus den mitgelieferten Beispielen — so siehst du in wenigen Sekunden, ob der Build bei dir läuft.

**Kein GUI** — alles über die Konsole. Unten: ein **kurzes** Laufbeispiel (Regression) und ein **Agent-Demo**-Output.

### Mini-MLP trainieren (Regression)

```bash
cargo run -p rusty_ai --example train_mlp
```

Beispiel-Output (Auszug — Loss sinkt über Epochen):

```text
epoch   0  loss = 0.292513
epoch  20  loss = 0.013322
epoch  40  loss = 0.002707
epoch  60  loss = 0.000606
epoch  79  loss = 0.000300
```

### Agent: Fallback-Backend (ohne Netzwerk)

```bash
cargo run -p rusty_ai_agent --example dual_backend_demo
```

Beispiel-Output:

```text
from fallback
```

(Das Beispiel simuliert einen fehlschlagenden primären LLM-Call und nutzt dann den Fallback — gut sichtbar in einer Zeile.)

Weitere Demos: `train_mini_gpt`, `agent_demo`, `agent_retry_demo`, `telemetry_demo`; mit Features `http` / `real-exec` siehe [`rusty_ai_agent/README.md`](rusty_ai_agent/README.md).

---

## So funktioniert’s (kurz)

```text
  Eingabedaten
       │
       ▼
  ┌─────────┐     ┌──────────────┐     ┌─────────┐
  │ Tensor  │ ──► │  Variable /  │ ──► │  Loss   │
  │         │     │  Schichten   │     │         │
  └─────────┘     └──────────────┘     └────┬────┘
                                            │
                                            ▼ backward
                                     Optimizer-Update
```

**LLM-Inferenz** (`generate`): zuerst **Prefill** über den ganzen Prompt (KV-Cache füllen), dann pro neuem Token nur noch **Decode** mit Sequenzlänge 1 — weniger Arbeit als der volle Forward jedes Mal.

**Agent (optional):** Modell liefert Text und/oder **Tool-Calls** → euer Code parst JSON → **Policy** (Pfade, erlaubte Binaries) → optional **Executor** (`real-exec`).

Ausführlicher: **[Handbuch](docs/HANDBUCH.md)** (Architektur, KV-Cache, Agent-Abschnitte 2.8 / 3.4).

---

## Roadmap

Die **fachliche** Roadmap (IDE/Orchestrierung, Phasen 0–4) steht in **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](docs/ARCHITEKTUR_IDE_ROADMAP_B.md)**. Kurzfassung:

| Phase | Richtung |
| ----- | -------- |
| **0–1** | `LlmBackend`, Tool-Protokoll, HTTP/SSE, Retry, Telemetrie, Fallback — im Workspace weitgehend umgesetzt (`rusty_ai_agent`) |
| **2–3** | Index, Diagnosen, Prompts, Policies, Batch-Reports, Budgets, Cache — siehe Roadmap |
| **4 (optional)** | Doku/Fine-Tuning/DPO extern, `generate_from_ids_with_callback`, Kontextlimits — siehe Roadmap |

Im **ML-/LLM-Kern** bleiben bewusst **TODOs** für Dinge wie vollständiges GPU-Training, FIM, Candle-Anbindung an `TrainableMiniGpt` — siehe Kommentare in den Crates.

---

## Voraussetzungen

- [Rust](https://www.rust-lang.org/tools/install) (stabiler Toolchain; Edition 2021)

---

## Schnellstart

```bash
git clone <REPO-URL>
cd RustyAi
cargo build --workspace
cargo test --workspace
```

```bash
cargo run -p rusty_ai --example train_mlp
cargo run -p rusty_ai --example train_mini_gpt
cargo run -p rusty_ai_agent --example agent_demo
```

LLM-Kurzüberblick (Prefill/Decode, Code-Snippet) und GPT-2-Hinweise: siehe Abschnitte **„LLM: Textgenerierung“** und **Dokumentation** unten bzw. [Handbuch](docs/HANDBUCH.md).

```rust
use rusty_ai::{generate, MiniGpt, MiniGptConfig};

let mut seed = 1u32;
let model = MiniGpt::random(MiniGptConfig::default(), &mut seed).unwrap();
let text = generate(&model, "Hallo ", 32, 0.8, 0.95, &mut seed).unwrap();
```

**Agent / HTTP / `real-exec`:** [`rusty_ai_agent/README.md`](rusty_ai_agent/README.md) und [Handbuch §2.8 / §3.4](docs/HANDBUCH.md).

---

## Workspace-Struktur

| Crate | Rolle |
| ----- | ----- |
| `rusty_ai` | Sammel-Crate: Re-Exports — [`rusty_ai/README.md`](rusty_ai/README.md) |
| `rusty_ai_agent` | Agent-Protokoll, optional `http` / `real-exec`; Diagnosen, Prompt-Vorlagen, `CargoTestInvocation` |
| `rusty_ai_workspace` | Datei-Index (Chunks, Suche), optional Embeddings (`embeddings`) |
| `rusty_ai_core` | Tensoren, Formen, numerische Operationen |
| `rusty_ai_autograd` | `Variable`, Rückwärtsrechnung |
| `rusty_ai_nn` | Schichten & Aktivierungen |
| `rusty_ai_ml` | Optimierer & Daten-Helfer |
| `rusty_ai_llm` | Transformer, Tokenizer, Generierung, Checkpoints — [`rusty_ai_llm/README.md`](rusty_ai_llm/README.md) |
| `rusty_ai_backend_candle` | Optional: Candle (CPU/CUDA), FP8 |

Abhängigkeit der Kernbibliothek: u. a. [`matrixmultiply`](https://crates.io/crates/matrixmultiply) für CPU-Matmul.

---

## LLM: Textgenerierung (Kurzüberblick)

`MiniGpt` ist ein kleiner **Decoder-only**-Transformer. Zum **Trainieren** dient **`TrainableMiniGpt`** mit `Variable::cross_entropy_next_token` (`cargo run -p rusty_ai --example train_mini_gpt`).

**GPT-2-Gewichte + BPE** (`cargo build -p rusty_ai --features gpt2-bpe`): `MiniGptConfig` zur Checkpoint-Größe, `load_minigpt_from_gpt2_safetensors`, `Gpt2Tokenizer::from_model_dir`, dann `generate_gpt2_text` / `generate_from_ids`.

Manuelle Inferenz mit **`forward_prefill`** / **`forward_decode_step`** und **`KvCache`**: [Handbuch — LLM](docs/HANDBUCH.md).

---

## Dokumentation

| Ressource | Inhalt |
| --------- | ------ |
| **[`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`](docs/ARCHITEKTUR_IDE_ROADMAP_B.md)** | Pfad B: Architektur, Roadmap |
| **[`rusty_ai_agent/README.md`](rusty_ai_agent/README.md)** | Agent: Features, Beispiele, API |
| **[`rusty_ai_agent/SECURITY.md`](rusty_ai_agent/SECURITY.md)** | Sicherheit (Allowlist, `run_cmd`) |
| **[`rusty_ai_workspace/README.md`](rusty_ai_workspace/README.md)** | Workspace-Index, Suche, optional `build_cached`, Embeddings (`CachingEmbeddingClient`) |
| **[`docs/HANDBUCH.md`](docs/HANDBUCH.md)** | Zentrale Referenz: Crates, Abläufe, Glossar |
| **[`docs/README.md`](docs/README.md)** | Index der `docs/`-Dateien |
| **[`rusty_ai/README.md`](rusty_ai/README.md)** | Meta-Crate `rusty_ai`: Features, Beispiele, Links zum Handbuch |
| **[`docs/BERICHT_PRÜFUNG.md`](docs/BERICHT_PRÜFUNG.md)** | Prüfbericht Scope-Erweiterung |
| **[`rusty_ai_backend_candle/README.md`](rusty_ai_backend_candle/README.md)** | Candle-Backend |

```bash
cargo doc --workspace --no-deps --open
```

---

## Erweiterter Scope (experimentell)

- **Training:** Kern ist **CPU-Autograd**; Candle optional für Matmul/Experimente.
- **Checkpoints:** Eigenes Format; `load_minigpt_from_hf` (Feature `hf-hub`) für **RustyAi**-Checkpoints im Hub.
- **Hugging Face GPT-2:** `load_minigpt_from_gpt2_safetensors`; BPE mit **`gpt2-bpe`**.

---

## Mitwirken

Issues und Pull Requests sind willkommen. Vor einem Merge: `cargo clippy --workspace --all-targets` und `cargo test --workspace`. Änderungen an öffentlicher API oder Architektur idealerweise im [Handbuch](docs/HANDBUCH.md) spiegeln.
