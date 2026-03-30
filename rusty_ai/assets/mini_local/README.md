# Mini-Checkpoint (lokal)

Dieses Verzeichnis enthält ein **RustyAi**-Checkpoint im üblichen Format:

- `config.json` — `model_type`: `rusty_ai_minigpt`, Felder wie `n_embd`, `n_layer`, …
- `model.safetensors` — FP32-Gewichte (typisch **weit unter 5 MiB** für `MiniGptConfig::micro_local()`).

**Erneuern**

- Deterministische Zufallsinitialisierung (Seed 42):  
  `cargo test -p rusty_ai_llm bootstrap_rusty_ai_mini_local_assets -- --ignored`
- Kurzes Training und Überschreiben:  
  `cargo run -p rusty_ai --example train_micro_checkpoint`

**Verwendung:** `load_minigpt_checkpoint` (Pfad), `load_minigpt_checkpoint_bytes` (für `include_str!` / `include_bytes!`). Siehe [`docs/HANDBUCH.md`](../../../docs/HANDBUCH.md) (§2.5, §3.3.1) und [`rusty_ai/README.md`](../../README.md).
