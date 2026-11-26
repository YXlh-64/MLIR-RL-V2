| model | baseline (ms) | v5_model_1200 (ms) | speedup |
|---|---|---|---|
| bart | 2130.2 | FAILED | - |
| bert | 4260.2 | FAILED | - |
| convnext_tiny | 10121.6 | FAILED | - |
| efficientnet_b0 | 442.4 | 111.5 | 3.97x |
| gat | 44.9 | 53.0 | 0.85x |
| gin | 13.9 | 24.6 | 0.57x |
| llama3_2_1b | 572734.4 | still running | - |
| mobilenet_v3_small | 76.9 | 61.2 | 1.26x |
| resnet50 | 5178.0 | 438.3 | 11.81x |
| resnext50 | 5308.5 | FAILED | - |
| t5 | 1596.1 | FAILED | - |
| vgg16 | 21743.3 | 2972.9 | 7.31x |
| vit_b_16 | 56097.6 | FAILED | - |

Geomean over 6 models: 2.43x, median 2.62x
