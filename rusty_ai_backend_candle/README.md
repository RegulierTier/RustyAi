# rusty_ai_backend_candle

Optionales **Candle**-Backend für den RustyAi-Workspace: Matmul auf CPU oder (mit Feature **`cuda`**) auf NVIDIA-GPUs, **FP8 E4M3**-Konvertierung und eine **Referenz-Implementierung** für den Mittelwert über Gradienten-Kopien (`all_reduce_mean_cpu`), wie er nach einem All-Reduce in Daten-Parallel-Training anfällt.

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

Die Kern-Trainingspfade (`TrainableMiniGpt`) bleiben CPU-Autograd; dieses Crate ist für Experimente, Portierung und Beschleunigung gedacht.
