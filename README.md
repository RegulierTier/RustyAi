# RustyAi

Experimenteller **Rust-Workspace** für nachvollziehbare ML-/LLM-Bausteine: eigene Tensor- und Autograd-Schichten, ein kleiner **Decoder-only-Transformer** (`MiniGpt`), optional GPT-2-Import und ein separates **Agent-Crate** (Tools, Policies, optional HTTP) für IDE-nahe Workflows.

**Lizenz:** MIT **oder** Apache-2.0 (siehe Workspace-`Cargo.toml`).

---

## Schnellstart

Vom Repository-Root:

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

**LLM lokal ohne Netz** (eingebettete Mini-Gewichte, Byte-Tokenizer):

```bash
cargo run -p rusty_ai --example mini_local_inference
```

Weitere Beispiele: siehe Abschnitt [Beispiele](#beispiele) und [`rusty_ai/README.md`](rusty_ai/README.md).

---

## Workspace-Übersicht

| Crate | Rolle (Kurz) |
| ----- | ------------- |
| [`rusty_ai_core`](rusty_ai_core) | `Tensor`, Shapes, Basissops |
| [`rusty_ai_autograd`](rusty_ai_autograd) | `Variable`, Backprop |
| [`rusty_ai_nn`](rusty_ai_nn) | Linear, GELU, LayerNorm, Init |
| [`rusty_ai_ml`](rusty_ai_ml) | SGD, Adam, Batches |
| [`rusty_ai_llm`](rusty_ai_llm) | `MiniGpt`, Generierung, Checkpoints, optional BPE/Hub |
| [`rusty_ai`](rusty_ai) | Meta-Crate: Re-Exports des Kerns |
| [`rusty_ai_backend_candle`](rusty_ai_backend_candle) | Optional: Candle (CPU/CUDA), FP8-Hilfen |
| [`rusty_ai_agent`](rusty_ai_agent) | `LlmBackend`, Tool-Protokoll, Policies, optional `http` |
| [`rusty_ai_workspace`](rusty_ai_workspace) | Workspace-Index, Suche, optional Embeddings |

Abhängigkeiten und Features pro Crate: jeweils `README.md` im Ordner sowie [**Handbuch §2**](docs/HANDBUCH.md).

---

## Dokumentation

| Ressource | Inhalt |
| --------- | ------ |
| [**`docs/HANDBUCH.md`**](docs/HANDBUCH.md) | Architektur, alle Crates, typische Abläufe (MLP, Mini-GPT, FIM, Agent, Checkpoints) |
| [**`docs/README.md`**](docs/README.md) | Index der Dokumentationsdateien und Crate-READMEs |
| [**`docs/ARCHITEKTUR_IDE_ROADMAP_B.md`**](docs/ARCHITEKTUR_IDE_ROADMAP_B.md) | IDE-nah: Ziele, Roadmap (Pfad B) |
| [**`rusty_ai_llm/README.md`**](rusty_ai_llm/README.md) | LLM-API, FIM, Features `gpt2-bpe` / `hf-hub`, **lokales Mini-Bundle** |
| [**`rusty_ai_backend_candle/README.md`**](rusty_ai_backend_candle/README.md) | Optionales Candle-Backend, Abgrenzung zu CPU-`TrainableMiniGpt`, optionaler Forward-Test |
| [**`rusty_ai_agent/README.md`**](rusty_ai_agent/README.md) | Agent, HTTP, Policies, Beispiele |

API-Details: `cargo doc -p rusty_ai --no-deps --open` (optional mit `--features …`).

---

## Beispiele (Auswahl)

| Befehl | Thema |
| ------ | ----- |
| `cargo run -p rusty_ai --example train_mlp` | Kleines MLP trainieren |
| `cargo run -p rusty_ai --example train_mini_gpt` | Mini-GPT Next-Token (CPU-Autograd) |
| `cargo run -p rusty_ai --example train_micro_checkpoint` | Mini-Profil trainieren → `assets/mini_local/` schreiben |
| `cargo run -p rusty_ai --example mini_local_inference` | Checkpoint aus eingebetteten Bytes, **ohne Netz** |

Agent- und Workspace-Beispiele: [`docs/HANDBUCH.md` §2.8–2.9](docs/HANDBUCH.md) bzw. [`rusty_ai_agent/README.md`](rusty_ai_agent/README.md).

---

## Status und Philosophie

Das Projekt ist **experimentell**: Fokus auf Verständnis und erweiterbare Bausteine, nicht auf einen produktionsreifen Chatbot. Der Kernpfad ist **CPU-Autograd**; GPU/Candle ist optional und **nicht** als Drop-in fürs gleiche `TrainableMiniGpt`-Training angebunden (siehe [Handbuch §2.6](docs/HANDBUCH.md) und [`rusty_ai_backend_candle/README.md`](rusty_ai_backend_candle/README.md)). **FIM** (Fill-in-the-Middle) ist im Kern-LLM-Crate dokumentiert ([`rusty_ai_llm/README.md`](rusty_ai_llm/README.md), Handbuch §2.5).

---

## Mitwirken

Issues und Beiträge sind willkommen. Vor größeren Änderungen die bestehenden Tests und `clippy` im Workspace ausführen.
