# Candidate model survey

## 1. Suitability per model

| model | suitable | linalg ops | GFLOPs | max tensor elems | % ops ≥2 data producers | top categories | failed rules |
|---|---|---|---|---|---|---|---|
| resnet18 | yes | 200 | 3.6449 | 2.4e+06 | 0.04 | generic_elementwise:169, conv:20, fill:8, pooling:1 | - |
| resnet34 | yes | 352 | 7.3512 | 2.4e+06 | 0.045 | generic_elementwise:305, conv:36, fill:8, pooling:1 | - |
| resnet26 | yes | 287 | 4.7304 | 2.4e+06 | 0.028 | generic_elementwise:240, conv:29, fill:15, pooling:1 | - |
| resnext26ts | yes | 338 | 4.8848 | 2.1e+06 | 0.104 | generic_elementwise:283, conv:31, fill:21, pooling:1 | - |
| regnety_008 | yes | 583 | 1.6271 | 8.3e+05 | 0.048 | generic_elementwise:474, conv:75, fill:18, generic_reduction:15 | - |
| densenet121 | yes | 1117 | 5.7526 | 1e+06 | 0.0 | generic_elementwise:975, conv:120, fill:16, pooling:4 | - |
| vgg11 | **no** | 48 | 15.2448 | 1e+08 | 0.0 | generic_elementwise:24, conv:10, fill:7, pooling:5 | max_tensor_elems=1.03e+08 > 6.71e+07 |
| repvgg_a0 | yes | 545 | 3.0497 | 2.2e+06 | 0.072 | generic_elementwise:492, conv:44, fill:7, generic_reduction:1 | - |
| mobilenetv2_100 | yes | 525 | 0.6454 | 1.3e+06 | 0.152 | generic_elementwise:450, conv:35, fill:21, depthwise_conv:17 | - |
| mobilenetv3_large | yes | 626 | 0.466 | 1.3e+06 | 0.121 | generic_elementwise:528, conv:48, fill:25, depthwise_conv:15 | - |
| efficientnet_b1 | yes | 927 | 1.4511 | 1.4e+06 | 0.117 | generic_elementwise:756, conv:92, fill:31, generic_reduction:24 | - |
| ghostnet_100 | yes | 819 | 0.3035 | 1.3e+06 | 0.02 | generic_elementwise:681, conv:54, depthwise_conv:41, fill:34 | - |
| levit_128s | yes | 820 | 0.6316 | 3.9e+05 | 0.126 | generic_elementwise:680, contraction:70, fill:43, generic_reduction:23 | - |
| mobilevit_s | yes | 959 | 4.1268 | 2.2e+06 | 0.122 | generic_elementwise:736, generic_reduction:82, contraction:55, fill:51 | - |
| poolformer_s12 | yes | 568 | 3.6711 | 1.5e+06 | 0.197 | generic_elementwise:443, generic_reduction:76, conv:28, pooling:12 | - |
| resmlp_12 | yes | 256 | 6.0534 | 5.9e+05 | 0.098 | generic_elementwise:212, contraction:37, fill:5, conv:1 | - |
| deit_tiny | yes | 731 | 2.5394 | 1.9e+05 | 0.153 | generic_elementwise:547, generic_reduction:99, contraction:73, fill:11 | - |
| segformer_b0 | yes | 863 | 0.9356 | 4.3e+05 | 0.148 | generic_elementwise:636, generic_reduction:106, contraction:64, fill:39 | - |
| bert_L4_H512 | yes | 301 | 0.4189 | 1.6e+07 | 0.153 | generic_elementwise:225, generic_reduction:35, contraction:32, fill:9 | - |
| bert_L4_H768 | yes | 301 | 0.9397 | 2.3e+07 | 0.153 | generic_elementwise:225, generic_reduction:35, contraction:32, fill:9 | - |
| electra_small | yes | 840 | 0.3178 | 3.9e+06 | 0.154 | generic_elementwise:635, generic_reduction:99, contraction:97, fill:9 | - |
| minilm_l6 | yes | 435 | 0.3545 | 1.2e+07 | 0.152 | generic_elementwise:327, generic_reduction:51, contraction:48, fill:9 | - |
| mobilebert | **no** | 2060 | 0.6757 | 3.9e+06 | 0.024 | generic_elementwise:1596, contraction:409, generic_reduction:48, fill:7 | linalg_ops=2060 outside [20,2000] |
| deberta_v3_xsmall | yes | 1102 | 4.5001 | 1.2e+07 | 0.197 | generic_elementwise:843, contraction:144, generic_reduction:102, fill:13 | - |
| distilgpt2 | yes | 416 | 1.3691 | 2.5e+07 | 0.171 | generic_elementwise:319, generic_reduction:51, contraction:36, fill:10 | - |
| pythia_70m | yes | 380 | 0.6279 | 1.6e+07 | 0.158 | generic_elementwise:299, contraction:37, generic_reduction:33, fill:11 | - |
| smollm_135m_L4 | yes | 262 | 0.4707 | 1.8e+07 | 0.088 | generic_elementwise:198, contraction:37, generic_reduction:17, fill:10 | - |
| qwen2_5_0_5b_L2 | yes | 154 | 0.9872 | 2.9e+07 | 0.084 | generic_elementwise:116, contraction:19, fill:10, generic_reduction:9 | - |
| t5_efficient_tiny | yes | 523 | 0.2463 | 8.2e+06 | 0.076 | generic_elementwise:380, contraction:88, generic_reduction:46, fill:9 | - |
| marian_en_de | yes | 377 | 0.4896 | 1.6e+07 | 0.13 | generic_elementwise:282, contraction:44, generic_reduction:42, fill:9 | - |
| whisper_tiny_enc | yes | 293 | 37.3344 | 1.4e+07 | 0.137 | generic_elementwise:215, generic_reduction:35, contraction:32, fill:9 | - |

## 2. Coverage of evaluation op names (suitable candidates only)

Eval op names: 11; covered: 9; missing: broadcast, transpose


## 3. Category share (suitable candidates vs eval)

Candidate rows weighted by model weight (--weights).

| category | candidates | eval |
|---|---|---|
| broadcast | 0.000 | 0.029 |
| contraction | 0.056 | 0.081 |
| conv | 0.045 | 0.040 |
| depthwise_conv | 0.007 | 0.008 |
| fill | 0.031 | 0.022 |
| generic_elementwise | 0.798 | 0.684 |
| generic_reduction | 0.061 | 0.061 |
| pooling | 0.001 | 0.002 |
| transpose | 0.000 | 0.074 |

## 4. Shape ranges (median, p10–p90 of log2 value)

| category | field | set | n | median log2 | p10–p90 log2 |
|---|---|---|---|---|---|
| contraction | M | cand | 957 | 4.0 | 4.0–8.0 |
| contraction | M | eval | 468 | 4.0 | 3.8–7.6 |
| contraction | N | cand | 957 | 8.6 | 5.0–10.0 |
| contraction | N | eval | 468 | 9.6 | 6.0–11.6 |
| contraction | K | cand | 957 | 8.0 | 4.0–10.0 |
| contraction | K | eval | 468 | 9.6 | 6.0–11.6 |
| conv | C | cand | 634 | 7.3 | 4.6–9.6 |
| conv | C | eval | 214 | 7.9 | 4.6–10.0 |
| conv | F | cand | 634 | 7.0 | 5.0–9.6 |
| conv | F | eval | 214 | 8.0 | 4.8–10.0 |
| conv | H | cand | 634 | 3.8 | 0.0–5.9 |
| conv | H | eval | 214 | 3.8 | 0.0–5.8 |
| depthwise_conv | C | cand | 111 | 7.9 | 5.0–10.0 |
| depthwise_conv | C | eval | 45 | 8.6 | 6.6–9.6 |
| depthwise_conv | H | cand | 111 | 4.2 | 3.2–6.8 |
| depthwise_conv | H | eval | 45 | 4.3 | 3.5–6.0 |
| generic_elementwise | out_elems | cand | 12654 | 13.6 | 5.3–17.6 |
| generic_elementwise | out_elems | eval | 3959 | 15.0 | 6.0–19.2 |
| generic_reduction | out_elems | cand | 1005 | 5.6 | 4.0–9.6 |
| generic_reduction | out_elems | eval | 352 | 7.6 | 4.0–10.3 |

## 4b. Key widths, log2 (target: candidates reach the eval median and p90)

| shape | cand median | eval median | cand p90 | eval p90 |
|---|---|---|---|---|
| contraction N | 8.6 | 9.6 | 10.0 | 11.6 |
| contraction K | 8.0 | 9.6 | 10.0 | 11.6 |
| conv C | 7.3 | 7.9 | 9.6 | 10.0 |
| conv F | 7.0 | 8.0 | 9.6 | 10.0 |

## 5. Plots

![shapes_contraction.png](plots/shapes_contraction.png)
![shapes_conv.png](plots/shapes_conv.png)
![shapes_depthwise_conv.png](plots/shapes_depthwise_conv.png)
![shapes_generic_elementwise.png](plots/shapes_generic_elementwise.png)
![shapes_generic_reduction.png](plots/shapes_generic_reduction.png)
![categories.png](plots/categories.png)
