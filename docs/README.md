# Dokumentation (RustyAi)

Hier liegen **menschenlesbare** Beschreibungen des Projekts. Die Rust-API bleibt maßgeblich in den Quelltext-Kommentaren und in `cargo doc`.

| Datei | Zweck |
| ----- | ----- |
| **[HANDBUCH.md](HANDBUCH.md)** | Zentrale Referenz: Architektur, alle Workspace-Crates, typische Abläufe (Regression mit MLP, **Mini-GPT-Training** mit Next-Token-Cross-Entropy, LLM-Inferenz mit KV-Cache), Grenzen und Glossar. |

**Einstieg im Repo:** das [README im Projektroot](../README.md) mit Schnellstart, Workspace-Übersicht, Beispielen (`train_mlp`, `train_mini_gpt`) und Kurzbeschreibung der LLM-Pipeline.

**LayerNorm (Kurz):** Reine letzte-Achsen-Normierung: `rusty_ai_nn::layer_norm` / `Variable::layer_norm`. Affine Parameter wie in PyTorch (`γ`, `β`): `layer_norm_affine` bzw. `Variable::layer_norm_affine`; `MiniGpt` bindet eigene γ/β pro Pre-Norm und vor dem LM-Head ein. Ausführlicher: [HANDBUCH.md](HANDBUCH.md) (Abschnitte 2.2–2.3 und `MiniGpt` unter 2.5).
