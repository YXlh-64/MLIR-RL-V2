# Experiment 02: results on multi-operation blocks (checkpoint `v5_model_1200`)

**Run:** 2026-11-08, Jubail `compute` partition (AMD Bergamo), 1 pinned thread per method, warm cache.
**Status:** complete. All 6,693 blocks of `ops_and_blocks` have results for all four methods.

## Summary

- **MLIR-RL is 0.885× the plain MLIR pipeline** (geomean over 1,405 blocks, 95% CI 0.83–0.94):
  slightly slower overall. It is faster on 680 blocks, within 5% on 156 and slower on 569. The
  overall difference is not significant (Holm-adjusted Wilcoxon p = 0.89): wins and losses roughly cancel.
- **It depends on the block type:**
  - **matmul blocks:** RL is clearly better, 1.73× (57% of blocks faster).
  - **conv blocks:** RL is worse, 0.76×.
  - **reduction blocks:** RL is worse, 0.59×.
- **PyTorch is far faster than both MLIR methods** at 1 thread: eager 10.1×, Inductor 12.9× vs plain MLIR.
  MLIR-RL reaches 7% of Inductor's speed overall and 2.5% on matmul blocks.
- **RL's slowdowns are real, not noise:**
  - **Size:** the median losing block is 2.8× slower (speedup 0.36).
  - **Repeat runs:** re-measuring the same block changed its time by 1–4% (median).
- **Correctness:**
  - **No method produced a wrong result.** The 11–12 "failed" checks per method are float32
    rounding differences, analysed in the correctness section.
  - **42 RL schedules did not compile or crashed.**

## 1. What was run

| Item | Value |
|---|---|
| Data | `ops_and_blocks`: **all 6,693 multi-op blocks** from 17 models, no sampling. The 1,400 single-op files in the same folder were not part of this experiment. |
| Unique work | The 6,693 blocks are **1,474 distinct computations** (same MLIR apart from op tags). Each was measured once; `all_blocks_results.csv` gives every block the result of its identical twin (`measured_as` column). |
| Block shape | Every block has 5 linalg ops (3–5 compute ops plus `fill`s), 2–5 different op types. |
| Training overlap | None. v5 was trained on `mlir_rl_v1_paper`, which contains no blocks. |
| Checkpoint | `/scratch/mh8826/checkpoints/v5_model_1200.pt`, config `config/v5/v5_mlir_rl_v1_paper.json` (verified by strict `load_state_dict`) |
| Harness | `experiments/01_performance_diagnosis/experiment.py`, **unchanged**: one worker process per method, 100 timed runs, median latency, compile time, correctness |
| Correctness rule | Each method's output is compared with `mlir_baseline` and with `torch_eager` on 3 random input sets, `np.allclose(rtol=1e-4, atol=1e-5)` |
| Time limit | 1,200 s per method. The 74 blocks that hit it were re-run with 3,600 s (all 4 methods). |
| Resources | 2 CPUs and 24 GB per Slurm task, `--cpu-bind=cores`, 1 compute thread. 48 tasks for the main run, 30 for the retries. |

**Methods:**

| Method | What it is |
|---|---|
| `mlir_baseline` | V5 lowering pipeline, LLVM -O3, no schedule |
| `mlir_rl` | the same pipeline after the checkpoint's greedy schedule |
| `torch_eager` | an equivalent PyTorch program generated from the block (`block_to_torch.py`) |
| `torch_inductor` | the same program through `torch.compile` |

**How the RL schedule is applied ("lenient" mode).** Experiment 01 counts the whole method as an
error if any single scheduling action fails to apply. Here a failed action is skipped together
with the remaining actions of that op, which is the rule MLIR-RL's own training environment uses
(`Env.__apply_sequence`). Every failure is recorded in `methods/mlir_rl/rl_application.json`.
147 of the 1,405 compared blocks had at least one failed action. Under strict application,
about half of all blocks would have counted as RL errors (TiledFusion transform assertion).

**Torch translations were verified beforehand.** All 6,693 translations were checked against an
independent float64 NumPy interpreter of the MLIR, and all matched. The 3 VGG blocks match only
at float32 scale. Details: `results/preprocess/`, README §4b.

## 2. Headline results

Speedup = `mlir_baseline` median latency ÷ method median latency, per block. It uses only blocks
where both methods passed correctness. Win/tie/loss counts a block as a tie when it is within 5%
and its 95% CI includes 1.

| Comparison | Blocks | Geomean | 95% CI | Win / tie / loss |
|---|---:|---:|---|---|
| MLIR-RL vs MLIR baseline | 1405 | **0.885×** | 0.83–0.94 | 680 / 156 / 569 |
| Torch eager vs MLIR baseline | 1452 | 10.1× | 9.2–11.1 | 1238 / 14 / 200 |
| Torch Inductor vs MLIR baseline | 1452 | 12.9× | 11.8–13.9 | 1312 / 28 / 112 |
| MLIR-RL vs Torch Inductor | 1405 | 0.072× | 0.065–0.079 | — |

**By block class** (class = the dominant op of the block):

| Class | Blocks | MLIR-RL | Torch eager | Torch Inductor |
|---|---:|---:|---:|---:|
| matmul | 307 / 343 | **1.73×** (175/82/50) | 67.3× | 69.5× |
| conv | 959 / 966 | **0.76×** (475/42/442) | 7.5× | 9.4× |
| reduction | 139 / 143 | **0.59×** (30/32/77) | 0.85× | 1.95× |

**By model** (MLIR-RL vs baseline, geomean):

| Model | Geomean | Model | Geomean |
|---|---:|---|---:|
| vgg16 | 8.29× (n=5) | convnext_tiny | 0.93× |
| gin | 7.68× | resnet50 | 0.91× |
| bart | 1.32× | yolov8m | 0.84× |
| bert | 1.31× | gpt2 | 0.83× |
| albert | 1.29× | resnext50 | 0.81× |
| t5 | 1.28× | whisper_base | 0.76× |
| llama3_2_1b | 1.27× | mobilenet_v3_small | 0.75× |
| | | gat | 0.74× (n=5) |
| | | efficientnet_b0 | 0.71× (n=431) |
| | | vit_b_16 | 0.56× |

The transformer and graph models gain, and the CNNs lose. The CNN losses are the conv result
above, and efficientnet and mobilenet contribute half of all blocks.

**By op type** (median speedup over all 6,693 blocks; a block counts in every op-type group it contains):

| Op type in block | Blocks | MLIR-RL median | Torch Inductor median |
|---|---:|---:|---:|
| add | 5457 | 1.01× | 29.8× |
| mul | 4538 | 1.05× | 28.5× |
| matmul / batch_matmul | 3194 | 1.05× | 81.4× |
| sum reduction | 2544 | 0.99× | 1.56× |
| div | 2004 | 0.99× | 1.60× |
| conv2d | 1441 | 1.05× | 20.6× |
| relu / max | 679 | 0.97× | 27.7× |
| depthwise conv | 196 | **2.58×** | 12.1× |
| pooling | 33 | **1.86×** | 1.72× |
| erf (gelu) | 192 | 1.06× (only 42 RL results) | 116× |
| tanh | 36 | no RL result (all failed) | 54.3× |

The full table, including the means, is `analysis/speedup_by_op_type.csv`.

**Figures** (`results/ckpt_v5_model_1200/figures/`):

| Figure | Shows |
|---|---|
| `speedup_by_op_type.png` | Mean and median per op type, all four methods |
| `speedup_by_op_type_unique.png` | The same over the 1,474 distinct computations |
| `win_tie_loss.png` | Win/tie/loss counts |
| `speedup_distribution.png` | Distribution of per-block speedups |
| `per_model.png` | Results by model |
| `speedup_vs_size.png` | Speedup against block size |
| `roofline.png` | Roofline view |
| `rl_vs_inductor.png` | MLIR-RL against Inductor |

**Which average to quote.** Use the geometric mean (plus the median). The arithmetic mean of
speedups is biased upward by a few large wins: on blocks containing add, the RL mean is 1.30×
but the median is 1.005×.

## 3. Correctness check

| Method | Measured | Passed | "Failed" | Not checked | Error | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| mlir_baseline | 1463 | 1452 | 11 | 0 | 0 | 11 |
| mlir_rl | 1422 | 1405 | 12 | 5 | 42 | 10 |
| torch_eager | 1474 | 1452 | 11 | 0 | 0 | 0 |
| torch_inductor | 1474 | 1452 | 11 | 0 | 0 | 0 |

**The "failed" results are rounding differences, not wrong code.**
- **The same 11 blocks fail for every method**, including the plain MLIR baseline:
  - llama3_2_1b blocks 165 and 1053
  - vgg16 blocks 1–3
  - vit_b_16 blocks 119, 120 and 158
  - whisper_base blocks 125, 177 and 178
- **Where they fail:** in each one, the compiled MLIR and Torch eager disagree by 1e-5 to 2e-3 in
  absolute terms on outputs of long float32 reductions (layernorm statistics, 4096-wide matmuls).
  That is just above the harness's `atol=1e-5`.
- **Against float64:** the independent float64 reference agrees with Torch at a relative
  error of 3e-7 to 3e-6 on all 11 blocks.
- **MLIR-RL against MLIR baseline:** on these blocks RL matches the baseline exactly (error 0)
  or to about 3e-5.
- **RL's 12th failure, vgg16_block_6:** RL passes against the baseline and misses Torch by
  2.3e-5. It is the same rounding effect.

So the tolerance is slightly too tight for fp32 reductions of this length; no method is wrong.
These blocks are left out of the speedup numbers above, as the harness's rule requires.
- **5 RL results "not checked":** whisper blocks where the baseline timed out even at 3,600 s,
  so there was no baseline to compare with. RL also passed against Torch eager on these.
- **The 42 RL errors are real failures of the schedules:**
  - **15 worker crashes:** the schedule leaves `omp.wsloop` with an `unrealized_conversion_cast`,
    and the JIT segfaults.
  - **~23 `llvm.intr.powi` lowering errors:** these are the blocks with `pow`, `erf` and `tanh`.
  - **4 ExecutionEngine failures.**

## 4. Are RL's losses real?

| Check | Result |
|---|---|
| Size of the losses | 561 blocks below 0.95×. Median loss 0.36×. 348 blocks below 0.5×, 59 below 0.1× |
| Timing noise within a run | Coefficient of variation of the 100 samples: median 1.1%, p90 5%, p99 17% |
| Re-measurement | Some blocks were measured in two separate runs (first runs and retries): baseline n=38, median change 1.1%; RL n=26, median change 3.5%; worst cases 20–55%, all on long-running blocks |
| Failed RL actions | Not the cause. Blocks with a failed action: geomean 1.16×, 25% losses. Blocks without any: 0.86×, 42% losses. The losses come from schedules that applied fully. |
| Where losses concentrate | reduction 55% of blocks, conv 46%, matmul 16% |

The losses are 10–100× larger than the measurement noise, so they reflect the schedules the
policy picks, not the measurement. A few borderline wins and losses within ±5–20% may change
side on a rerun.

## 5. Caveats

- **1 thread only.** Inductor and eager use MKL/oneDNN kernels; MLIR has no such library. Multi-thread results may differ.
- **Remaining timeouts:**
  - **mlir_baseline:** 11 whisper blocks did not finish within 3,600 s.
  - **mlir_rl:** 10 blocks did not finish (7 efficientnet, 3 whisper).
  - Blocks without a baseline result are not in the speedups.
- **Lenient RL application** differs from experiment 01's strict rule (see §1). The per-action
  record lets anyone recount under the strict rule.
- **Block classes** come from the dominant op. Op-type groups overlap because each block has 2–5 op types.
- **Node sharing:** several Slurm tasks shared each node, each on its own 2 pinned cores, so
  memory-bound blocks may see some interference. Baseline and RL of one block ran back to back
  in the same task.

## 6. Where everything is

All paths are relative to `experiments/02_block_diagnosis/`.

| What | Path |
|---|---|
| Short report | `overnight/MORNING_REPORT.md` |
| Statistics (CIs, Wilcoxon, Holm) | `results/ckpt_v5_model_1200/analysis/headline_table.md`, `per_model.csv`, `per_size.csv`, `per_block.csv`, `stats.json` |
| Op-type table | `results/ckpt_v5_model_1200/analysis/speedup_by_op_type.csv` (and `_unique`) |
| One row per distinct block × method | `results/ckpt_v5_model_1200/all_results.csv` |
| One row per block (all 6,693) × method | `results/ckpt_v5_model_1200/all_blocks_results.csv` |
| Raw per-run data | `results/ckpt_v5_model_1200/t1_warm/r0/<class>/artifacts/<block>/<UTC>/`: `summary.json`, `manifest.json`, `methods/<m>/result.json` (100 latency samples), `rl_application.json`, scheduled MLIR. Older runs of a block are kept; reports use the latest. |
| Translations and their verification | `results/preprocess/`, `results/ckpt_v5_model_1200/translations/` |
| Run history | `overnight/PROGRESS.md`, `overnight/state/jobs.log`, `overnight/logs/` |

**Disk.** `results/ckpt_v5_model_1200/` holds 313 GB of file data in 162,674 files. Lustre
allocates and bills about 626 GB for it (`du -sh`).

| What | Size | Needed? |
|---|---|---|
| Inductor precompiled-header caches (`methods/torch_inductor/tmp/**/*.h.gch`) | 247 GB | No: 1,570 copies of the same compiler cache |
| Random input arrays (`inputs/case_*.npz`) | 85 GB | Only to replay a run's exact inputs |
| Everything else (results, logs, MLIR, assembly) | ~3 GB | Yes |

Check the `/scratch` quota (`lfs quota -u $USER /scratch`) before another full run. The file
count (soft limit 500,000) matters as much as the size.

## 7. Reproduce

```bash
cd /scratch/mh8826/MLIR-RL
bash experiments/02_block_diagnosis/overnight/overnight.sh       # setup (skipped if done) -> 48-task run -> report
bash experiments/02_block_diagnosis/overnight/retry_launch.sh    # re-run blocks that timed out, with 3600 s
sbatch experiments/02_block_diagnosis/overnight/report.sbatch    # rebuild all tables, figures and the short report
python experiments/02_block_diagnosis/plot_op_types.py [--unique]   # op-type figure
```

All of these resume from where they stopped:
- blocks that already have a result in the current mode are skipped;
- a block is retried only if its latest run timed out with the shorter limit.

Settings (checkpoint list, methods, limits) are in `overnight/settings.sh`.
