# rusty_ai_backend_candle

Optionales **Candle**-Backend für den RustyAi-Workspace: Matmul auf CPU oder (mit Feature **`cuda`**) auf NVIDIA-GPUs, **FP8 E4M3**-Konvertierung und eine **Referenz-Implementierung** für den Mittelwert über Gradienten-Kopien (`all_reduce_mean_cpu`), wie er nach einem All-Reduce in Daten-Parallel-Training anfällt.

| Weiterlesen | Inhalt |
| ----------- | ------ |
| **[`docs/HANDBUCH.md`](../docs/HANDBUCH.md) §**2.6** | Einordnung im Workspace, Abgrenzung zu `TrainableMiniGpt` |
| **[`rusty_ai_llm/README.md`](../rusty_ai_llm/README.md)** | CPU-LLM-Training und FIM |
| **[`README.md`](../README.md)** (Root) | Workspace-Schnellstart |

## Features

| Feature | Bedeutung |
| ------- | --------- |
| *(keine)* | Nur CPU-Candle, keine CUDA-Bindung. |
| `cuda` | CUDA-Gerät über `candle-core` (Build benötigt passende CUDA-Toolchain). |
| `nccl` | NCCL für Multi-GPU (impliziert `cuda`); siehe Candle-Dokumentation für Prozessgruppen. |

## API (Kurz)

- `default_device` / `BackendDevice` — CPU oder CUDA ordinal 0, falls verfügbar.
- `matmul_f32` — `(m,k)×(k,n)` als flache `f32`-Puffer.
- `f32_tensor_to_f8e4m3` / `f8e4m3_tensor_to_f32` — FP8-Roundtrip.
- `all_reduce_mean_cpu` — elementweiser Mittelwert über mehrere gleich lange Puffer (ohne Netzwerk).

## Abgrenzung zu `TrainableMiniGpt`

[`TrainableMiniGpt`](https://docs.rs/rusty_ai_llm/latest/rusty_ai_llm/struct.TrainableMiniGpt.html) (Crate **`rusty_ai_llm`**) trainiert auf der CPU mit **`rusty_ai_autograd::Variable`** und **`rusty_ai_core::Tensor`**. Candle-Tensoren und deren Ops bilden **keinen** gemeinsamen Graphen mit diesem Pfad — ein **Drop-in**-GPU-Training für dasselbe Modell ist im Workspace **nicht** vorgesehen (hoher Portierungsaufwand oder ständige Gewichtskopien).

**Praktisch:** LM-Training wie gehabt auf der CPU; dieses Crate für Matmul auf CPU/CUDA, FP8-Hilfen und All-Reduce-Referenzen.

### Tier 1 vs. Tier 2 (E2E)

| Stufe | Bedeutung |
| ----- | --------- |
| **Tier 1** | „E2E light“: `MiniGpt`-Gewichte oder Zwischenaktivierung als `f32`-Slices → Candle-`matmul_f32` (CPU, optional CUDA-Gerät) → numerischer Vergleich mit `rusty_ai_core` / `linear_3d`. **Kein** `backward` über Candle mit dem `TrainableMiniGpt`-Graphen. |
| **Tier 2** | Vollständiges Modell + Optimizer in Candle — separates Projekt-Mandat, nicht dieses Crate. |

### Optionale Tests (`#[ignore]`)

Im Crate-`lib.rs` (Modul `minigpt_weight_bridge` und `minigpt_lm_head_candle`):

- **`minigpt_w_q_matmul_matches_candle_cpu`** — eine `w_q`-Matrix aus [`MiniGpt::random`](https://docs.rs/rusty_ai_llm/latest/rusty_ai_llm/struct.MiniGpt.html) vs. Candle-CPU-Matmul.
- **`minigpt_lm_head_matches_candle_cpu`** — LM-Head (`linear_3d`) vs. zusammengesetztes Matmul + Bias in Candle.

```bash
cargo test -p rusty_ai_backend_candle -- --ignored
```

CUDA-Toolchain für GPU-Matmul ist **nicht** Voraussetzung für diese CPU-Tests.
