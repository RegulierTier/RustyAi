# Dokumentation (RustyAi)

Hier liegen **menschenlesbare** Beschreibungen des Projekts. Die Rust-API bleibt maßgeblich in den Quelltext-Kommentaren und in `cargo doc`.

| Datei | Zweck |
| ----- | ----- |
| **[HANDBUCH.md](HANDBUCH.md)** | Zentrale Referenz: Architektur, alle Workspace-Crates, typische Abläufe (Regression mit MLP, **Mini-GPT-Training** mit Next-Token-Cross-Entropy, LLM-Inferenz mit KV-Cache), **Checkpoints / GPT-2-Import / Candle-Backend**, Grenzen und Glossar. |
| **[BERICHT_PRÜFUNG.md](BERICHT_PRÜFUNG.md)** | Prüfbericht zur Scope-Erweiterung (Korrekturen, Tests, Grenzen). |
| **[../rusty_ai_backend_candle/README.md](../rusty_ai_backend_candle/README.md)** | Kurzbeschreibung des optionalen Candle-Crates (CPU/CUDA, FP8, All-Reduce-Referenz). |

**Einstieg im Repo:** das [README im Projektroot](../README.md) mit Schnellstart, Workspace-Übersicht, Beispielen (`train_mlp`, `train_mini_gpt`) und Kurzbeschreibung der LLM-Pipeline.

**Checkpoints (Kurz):** Eigenes Format: Verzeichnis mit `config.json` (`model_type`: `rusty_ai_minigpt`) und `model.safetensors`. API: `rusty_ai_llm::save_minigpt_checkpoint` / `load_minigpt_checkpoint`. GPT-2-HF: `load_minigpt_from_gpt2_safetensors`. Vom Hub nur mit passendem Feature: `rusty_ai` mit `--features hf-hub` und `load_minigpt_from_hf` — erwartet **RustyAi**-Checkpoints im Repo, keine beliebigen GPT-2-Dateien.

**LayerNorm (Kurz):** Reine letzte-Achsen-Normierung: `rusty_ai_nn::layer_norm` / `Variable::layer_norm`. Affine Parameter wie in PyTorch (`γ`, `β`): `layer_norm_affine` bzw. `Variable::layer_norm_affine`; `MiniGpt` bindet eigene γ/β pro Pre-Norm und vor dem LM-Head ein. Ausführlicher: [HANDBUCH.md](HANDBUCH.md) (Abschnitte 2.2–2.3 und `MiniGpt` unter 2.5).
