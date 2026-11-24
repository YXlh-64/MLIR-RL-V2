# Track A: MLIR-RL (v5_model_1200) vs MLIR baseline on full models

Same node for both baseline runs and the RL run. Times in seconds.

## 1 thread

| Model | Baseline (run 1 / run 2) | Noise | MLIR-RL | Speedup | Failed RL actions | RL eval wall (min) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| gin_linalg | 0.0139 / 0.0139 | 0.074% | 0.00392 | **3.55×** | 0 | 0.58 | ok |
| gat_linalg | 0.044 / 0.0454 | 3.2% | 0.038 | **1.18×** | 1 | 0.77 | ok |
| mobilenet_v3_small_linalg | 0.081 / 0.0786 | 3.1% | 0.0879 | **0.907×** | 10 | 11 | ok |
| model_mobile_net_v2 | 0.386 / 0.395 | 2.2% | 0.34 | **1.15×** | 0 | 34 | ok |
| efficientnet_b0_linalg | 0.449 / 0.453 | 0.68% | 0.311 | **1.45×** | 17 | 6.9 | ok |
| t5_linalg | 1.61 / 1.55 | 3.8% |  |  | 19 | 2.9e+02 | rl failed |
| model_res_net | 5.26 / 5.26 | 0.0015% |  |  | 0 | 1.5e+02 | rl failed |
| resnet50_linalg | 5.21 / 5.22 | 0.11% | 2.41 | **2.16×** | 0 | 16 | ok |
| resnext50_linalg | 5.35 / 5.39 | 0.85% |  |  | 0 | 70 | rl failed |
| convnext_tiny_linalg | 10.9 / 10.9 | 0.5% |  |  | 44 | 16 | rl failed |
| model_vgg | 24.1 / 23.8 | 1.2% | 17.5 | **1.37×** | 0 | 38 | ok |
| vgg16_linalg | 22.3 / 22.2 | 0.36% | 18.5 | **1.2×** | 0 | 1.8 | ok |

Geomean speedup over 8 models: **1.48×**; faster on 7, within 5% on 0, slower on 1.

## 16 threads

| Model | Baseline (run 1 / run 2) | Noise | MLIR-RL | Speedup | Failed RL actions | RL eval wall (min) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| gin_linalg | 0.0142 / 0.0139 | 2.1% | 0.0178 | **0.789×** | 0 | 0.62 | ok |
| gat_linalg | 0.0387 / 0.0396 | 2.3% | 0.054 | **0.725×** | 1 | 0.52 | ok |
| mobilenet_v3_small_linalg | 0.0781 / 0.0806 | 3.2% | 0.107 | **0.743×** | 10 | 8 | ok |
| model_mobile_net_v2 | 0.335 / 0.334 | 0.4% | 0.267 | **1.25×** | 0 | 26 | ok |
| efficientnet_b0_linalg | 0.447 / 0.446 | 0.32% | 0.296 | **1.51×** | 17 | 6.5 | ok |
| t5_linalg | 1.6 / 1.56 | 2.5% |  |  | 19 | 91 | rl failed |
| model_res_net | 5.22 / 5.22 | 0.064% |  |  | 0 | 72 | rl failed |
| resnet50_linalg | 5.26 / 5.23 | 0.62% | 0.536 | **9.79×** | 0 | 15 | ok |
| resnext50_linalg | 5.43 / 5.41 | 0.48% |  |  | 0 | 66 | rl failed |
| convnext_tiny_linalg | 10.6 / 10.6 | 0.027% |  |  | 44 | 16 | rl failed |
| model_vgg | 23.2 / 23.3 | 0.23% | 2.19 | **10.6×** | 0 | 36 | ok |
| vgg16_linalg | 22.1 / 22 | 0.12% | 1.68 | **13.1×** | 0 | 1.4 | ok |

Geomean speedup over 8 models: **2.4×**; faster on 5, within 5% on 0, slower on 3.
