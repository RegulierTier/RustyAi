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

**Praktisch:** LM-Training wie gehabt auf der CPU; dieses Crate für Matmul auf CPU/CUDA, FP8-Hilfen und All-Reduce-Referenzen. Optional: einzelne Gewichtsmatrizen aus einem [`MiniGpt`](https://docs.rs/rusty_ai_llm/latest/rusty_ai_llm/struct.MiniGpt.html) als `f32`-Slices nach Candle laden und **nur Forward** vergleichen (siehe ignorierter Test im `lib.rs` des Crates).

```bash
cargo test -p rusty_ai_backend_candle -- --ignored
```
