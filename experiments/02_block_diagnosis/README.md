# Experiment 02: multi-operation blocks

Experiment 01 compared four methods on **single operations**. This experiment applies the
same comparison to the **6,693 multi-operation blocks** of `ops_and_blocks`, which experiment 01
skipped:

| Method | What it is |
|---|---|
| MLIR baseline | V5 lowering pipeline, LLVM -O3, no learned schedule |
| MLIR RL | the same pipeline after `v5_model_1200`'s greedy schedule |
| PyTorch eager | an equivalent Torch program, generated from the block |
| TorchInductor | the same Torch program through `torch.compile` |

It also asks a question experiment 01 could not: **do the schedules of ops in a block, and blocks
themselves, depend on each other?** These are experiments D1–D4 below.

Experiment 01's code is **reused, not modified**. Its harness (`experiment.py`) still does the
inputs, the per-method worker processes, the timing, the correctness checks and `summary.json`.
Its report writer still produces `comparison.csv` / `REPORT.md`, and its plot script still works
on the results.

> Status: **complete for checkpoint `v5_model_1200`** (2026-11-08). All 6693 blocks translate and
> match an independent NumPy interpreter (§4b), and all 6693 were run with the four methods.
> Results, correctness analysis and caveats: **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)**.
> How the unattended run works and resumes: `overnight/PROGRESS.md`.

---

## 1. How the files work together

```
          data/ops_and_blocks/*_block_*.mlir
                        │
   (1) block_catalog.py │  reads text only
                        ▼
   results/catalog/ blocks_catalog.json   what each block contains, class, size, cost, pre-checks
                    blocks_index.json     overlaps / nesting / "feeds" / compositions between blocks
                        │
   (2) run_blocks.py ───┤  selection (stratified, seeded, saved once per --tag)
        │               │
        │   uses (3) block_to_torch.py   MLIR block → Torch graph + readable torch_block.py
        │               │
        │   uses (4) block_experiment.py experiment 01's harness + block ops (+ cold-cache mode)
        │               │        └── experiment.py (exp. 01, unchanged): inputs, workers, timing,
        │               │                                                correctness, summary.json
        │               ▼
        │   results/<tag>/<config>/r<k>/<class>/artifacts/<block>/<UTC>/   (exp. 01 layout)
        │   results/<tag>/<config>/r<k>/<class>/comparison.csv, REPORT.md   (exp. 01 report writer)
        │   results/<tag>/all_results.csv          one long table over all runs
        │
   (5) ablation.py prepare  ──► results/<tag>/variants/<block>/*.mlir   (D1 variants, D2 schedules)
        └─ run_blocks.py --ablation ──► results/<tag>/t1_warm_ablation/...
                        │
   (6) stats.py         ▼  CIs, wins/ties/losses, Wilcoxon + Holm   → results/<tag>/analysis/
   (5) ablation.py context | interaction | composition | cache       → results/<tag>/analysis/
   (7) plot_blocks.py      figures F1–F11                            → results/<tag>/figures/
   (8) tests/test_translator.py   checks the translator against hand-written references
   (9) verify_blocks.py   ALL blocks: translate → run Torch eager → run generated .py →
                          compare with reference_eval.py (independent float64 NumPy
                          interpreter of the MLIR loops) → results/preprocess/
```

| File | Role | Needs |
|---|---|---|
| `block_catalog.py` | Parses every block; class (conv / matmul / reduction / elementwise), size bucket, FLOPs, ABI bytes, pre-checks for the translator and the 12-loop limit; relations between blocks | Python only |
| `block_to_torch.py` | The translator (rules below); `torch_operations()` is the op table the runner uses | Python (torch only to execute) |
| `block_experiment.py` | Runs experiment 01's `run_experiment()` with: the larger Torch op table, optional cold caches, and workers started through this file | MLIR env |
| `run_blocks.py` | Selection → translation → manifests → runs → `all_results.csv` | MLIR env to run; Python only for `--list`/`--prepare-only` |
| `ablation.py` | `prepare` (policy schedules + variants), then the D1–D4 analyses | MLIR env for `prepare` only |
| `stats.py` | Statistics and the headline table | numpy (scipy optional) |
| `plot_blocks.py` | Figures | matplotlib |
| `tests/test_translator.py` | 11 translator tests | torch + numpy |
| `reference_eval.py` | Independent float64 interpreter of the block's MLIR (literal loop semantics, no shared translation logic) | numpy (+ torch for `erf` only) |
| `verify_blocks.py` | Pre-processing pass over all blocks: translation, Torch run, correctness, timings → `results/preprocess/` | torch + numpy, no MLIR build |
| `run_blocks.sbatch` | Slurm launcher, same as experiment 01's | — |

## 2. Translation rules (why the Torch program is equivalent)

- **Named ops add into `outs`** (MLIR semantics, often missed):
  - `batch_matmul` → `outs + a @ b`
  - `conv_2d` → `outs + conv2d(...)`
  - depthwise conv → `conv2d(groups=C)`
  - max pooling → `max(outs, pool)`
  - `fill`, `transpose` and `broadcast` overwrite `outs`.
- **`linalg.generic`** is evaluated over its whole loop space at once:
  - Each operand is placed into that space using its indexing map: `permute` and `reshape`, with size-1 dimensions for broadcasting; a constant index becomes `narrow`.
  - The scalar body becomes elementwise Torch ops (`addf`→`+`, `exp`, `rsqrt`, `cmpf`+`select`→`where`, …).
  - **Reductions** are recognised from the op that reads `%out`: `addf` → `sum`, `mulf` → `prod`, max/min forms → `amax`/`amin`. The result is combined with the initial `outs`, exactly like the MLIR loop.
- **Anything else is reported as `unsupported` with a reason, never guessed.** Examples: maps like `d0+d1`, unknown body patterns, non-splat tensor constants. The catalog's `coverage.json` counts these before anything runs.
- **Inputs:** seeded normal values with std 0.1, as in experiment 01. The exception is an argument feeding `rsqrt`/`sqrt`/`log`/a `powf` base/a division denominator, which is drawn from uniform(0.5, 1.5); otherwise MLIR and Torch would both produce NaN and the check would mean nothing. This is recorded in the manifest.
- **The translator is checked three ways:**
  1. the unit tests;
  2. experiment 01's own rule on every run: **MLIR baseline output must match Torch eager**, otherwise no speedup is counted. A translator bug shows up as a correctness failure, never as a fake speedup;
  3. the readable `torch_block.py` is generated from the same graph and tested to give the same output.

## 3. What is measured, and what is better than experiment 01

| Experiment 01 (single ops) | Experiment 02 (blocks) |
|---|---|
| 33 workloads, 3 per category | Stratified sample: 1 per (model × class × size) stratum, about 100–200 blocks, duplicates removed, seeded; plus D3 triples |
| 1 thread | `--threads 1` **and** e.g. `--threads 8` (the RL parallelisation action can only help with >1 thread) |
| One trial | `--repeat 0,1,2`: independent runs, reported as run-to-run variation |
| Median only | 95% bootstrap CI per block; win/tie/loss only if the CI excludes 1 **and** the gain is ≥ 5% |
| No tests across workloads | Geometric mean with CI, Wilcoxon signed-rank (paired), Holm correction over the whole table, rank-biserial effect size |
| Strict schedule only | Strict (`mlir_rl`, as in 01) **and** lenient (`full_lenient`, like v5 Env) |
| — | GFLOP/s and arithmetic intensity (roofline view), compile/search cost |
| Possible train-set overlap | All blocks are unseen by the checkpoint (trained on `mlir_rl_v1_paper`) |

### Dependency experiments

| | Question | How | Output |
|---|---|---|---|
| **D1 interaction** | Is a block's full schedule worth more (synergy) or less (interference) than its per-op parts? | Time the full schedule and each op's schedule alone. **I** = log(T_base/T_full) − Σ log(T_base/T_only_op), with a bootstrap CI. Plus a "no fusion" variant: is the synergy due to fusion? | `interaction.csv`, F8 |
| **D2 context** | Does the policy schedule the **same model op** the same way when it appears in different (overlapping) blocks? | Policy decisions only, no timing; pairwise agreement per op | `context.csv`, F9 |
| **D3 composition** | Block X = Y + Z, with Y feeding Z: does X's time and speedup follow from Y's and Z's? | Uses the normal runs. Composition gain = (T_Y+T_Z)/T_X per method; predicted vs measured speedup | `composition.csv`, F10 |
| **D4 cache** | Does a method's advantage depend on data left in cache by the previous block? | Same blocks with `--cache cold`, where >L3 is rewritten before each timed call, outside the timer | `cache.csv`, F11 |

## 4. Commands (in order)

One-time links, so the experiment 01 defaults find your data and checkpoint without editing code:
```bash
cd /scratch/mh8826/MLIR-RL
ln -s /scratch/mh8826/data/ops_and_blocks data/ops_and_blocks
ln -s /scratch/mh8826/checkpoints checkpoints
mkdir -p experiments/02_block_diagnosis/results
E=experiments/02_block_diagnosis
```

| Step | Command | Where |
|---|---|---|
| 1 Catalog | `python $E/block_catalog.py` | login node OK |
| 2 Translator tests | `python $E/tests/test_translator.py` | needs torch |
| 3 Selection | `python $E/run_blocks.py --tag main --list` | login node OK |
| 4 Translate + manifests | `python $E/run_blocks.py --tag main --prepare-only` | login node OK |
| 5 Main runs | `sbatch $E/run_blocks.sbatch --tag main --threads 1 --repeat 0` (then `--repeat 1`, `2`) | Bergamo |
| 6 Threads | `sbatch --cpus-per-task=8 $E/run_blocks.sbatch --tag main --threads 8 --repeat 0` | Bergamo |
| 7 D4 cold | `sbatch $E/run_blocks.sbatch --tag main --threads 1 --cache cold --repeat 0` | Bergamo |
| 8 D1 variants | `python $E/ablation.py prepare --selection $E/results/main/selection.json --ablation-only --out $E/results/main/variants` | MLIR env |
| 9 D1 runs | `sbatch $E/run_blocks.sbatch --tag main --ablation --variants $E/results/main/variants` | Bergamo |
| 10 D2 schedules | `python $E/ablation.py prepare --context-ops 60 --schedules-only --out $E/results/main/context` | MLIR env |
| 11 Collect | `python $E/run_blocks.py --tag main --collect-only` | anywhere |
| 12 Stats | `python $E/stats.py --all-results $E/results/main/all_results.csv` | anywhere |
| 13 Analyses | `A=$E/results/main/analysis`<br>`python $E/ablation.py interaction --runs $E/results/main/t1_warm_ablation --out $A`<br>`python $E/ablation.py context --variants $E/results/main/context --out $A`<br>`python $E/ablation.py composition --all-results $E/results/main/all_results.csv --out $A`<br>`python $E/ablation.py cache --all-results $E/results/main/all_results.csv --out $A` | anywhere |
| 14 Figures | `python $E/plot_blocks.py --tag main` (add `--peak-gflops/--peak-gbs` for the roofline) | needs matplotlib |
| 15 Exp. 01 style plot | `python experiments/01_performance_diagnosis/plot_speedups.py --results-dir $E/results/main/t1_warm/r0` | needs matplotlib |

Steps 8 and 10 need the same environment as the runs (`.env` sourced, LLVM Python bindings,
AstDumper, torch), e.g. inside `salloc` after `set -a; source .env; set +a`.

## 4b. Pre-processing and correctness check (done, no MLIR build needed)

```bash
P=/home/mh8826/.conda/envs/MLIR-ENV/bin/python      # torch 2.x + numpy
$P $E/block_catalog.py --data-dir /scratch/mh8826/data/ops_and_blocks
$P $E/tests/test_translator.py
$P $E/verify_blocks.py --jobs 3 --max-volume 1e10              # ~8 min on 4 CPUs
$P $E/verify_blocks.py --jobs 1 --max-volume 1e10 --max-parallel 2e8 \
    --rerun reference_skipped_too_large                        # vgg16_block_2 needs ~6 GB alone
```
The reference runs once per unique computation (1474 signatures). Blocks with the same signature
inherit the verdict (`verified_via`); `--verify-all` checks every block. The script can be resumed
(`records.jsonl`), `--rerun <verdict>...` re-checks a group, and if a worker is killed (for example
by the job's memory limit) the missing blocks are recorded as `lost` instead of hanging.

How to read a verdict:
* `correct`: matches the float64 reference with the harness tolerance (rtol 1e-4, atol 1e-5).
* `correct_fp32_tolerance`: fails that elementwise test only on near-zero outputs; the maximum
  error divided by the output's magnitude is at most 1e-4. This happens only in the three
  VGG16 classifier blocks (sums over K = 25088 in float32, error about 3e-6 of scale).
  **Note:** experiment 01's MLIR-vs-eager check uses the same strict tolerance, so it may
  reject these three blocks; read their correctness separately.
* `mismatch` / `mismatch_nonfinite` / `unsupported` / `lost`: listed in `failures.json`.

Input domains (recorded per input in each `.torch.json`): values that feed rsqrt, sqrt, log,
powf bases or divisors are traced back to the block inputs and drawn from uniform(0.5, 1.5).
Scalar powf exponents passed as inputs are drawn as the integer 2 or 3. Without this, blocks
such as LayerNorm tails or GELU produced NaN in **both** Torch and the reference, so their
correctness could not be checked.

## 5. Output files

| File | Content |
|---|---|
| `results/catalog/coverage.json` | How many blocks the translator and RL can handle, and why not |
| `results/preprocess/preprocess.json` | Pre-processing record: environment, settings, verdict counts (overall / by class / unique), timing statistics per stage (count, total, median, p95, max) overall and per class, every invocation, and one record per block |
| `results/preprocess/preprocess.csv` | The same, one row per block (times, errors, verdict) |
| `results/preprocess/translations/` | `<block>.torch.json` + `<block>.py` for all 6693 blocks |
| `results/preprocess/failures.json` | Blocks that are not `correct` / `correct_fp32_tolerance` |
| `results/<tag>/selection.json` | The exact blocks, strata, roles (sample / composition / ablation) and translation status |
| `results/<tag>/translations/<block>.py` | The generated PyTorch code for each block |
| `results/<tag>/<config>/r<k>/<class>/REPORT.md`, `comparison.csv` | Experiment 01's report, per class |
| `.../artifacts/<block>/<UTC>/` | Experiment 01's per-workload folder: `summary.json`, `methods/<m>/result.json` (all 100 latency samples), `schedule.json`, scheduled MLIR, … |
| `results/<tag>/all_results.csv` | One row per config × repeat × block × method |
| `results/<tag>/analysis/headline_table.md` | **The main result table** |
| `results/<tag>/analysis/per_block.csv`, `per_model.csv`, `per_size.csv` | Detail tables |
| `results/<tag>/analysis/{interaction,context,composition,cache}.csv` | D1–D4 |
| `results/<tag>/figures/*.png` | F1–F11 (questions listed in `plot_blocks.py`) |
| `results/<tag>/all_blocks_results.csv` | All 6693 blocks (duplicates carry their twin's result, column `measured_as`) |
| `results/<tag>/analysis/speedup_by_op_type.csv`, `figures/speedup_by_op_type.png` | Mean / median speedup per op type (`plot_op_types.py`) |
| `.../methods/mlir_rl/rl_application.json` | Lenient RL application: applied / failed / skipped actions |

## 6. Smoke test before spending compute

1. `block_catalog.py --models albert resnet50 --limit 50`, then look at `coverage.json`.
2. `tests/test_translator.py`: all tests must pass.
3. `run_blocks.py --tag smoke --per-stratum 1 --compositions 2 --ablation-per-class 1 --prepare-only`.
   Read a few `translations/*.py` files against their `.mlir` files.
4. Run **one** manifest interactively:
   `python $E/block_experiment.py run $E/results/smoke/t1_warm/r0/manifests/<block>.json`
   Check that `summary.json` shows `correctness: passed` for `mlir_baseline` and `torch_eager`.
5. Only then submit the full runs.

## 7. Known limits

- **Ablation stages not run.** The main comparison ran in full (EXPERIMENT_REPORT.md). The D1–D4
  dependency experiments (`ablation.py`) have not been run: every block has 5 ops, so there are no
  composition triples for D3.
- **Correctness tolerance.** The harness's `atol=1e-5` flags 11 blocks with long fp32 reductions for
  every method, baseline included; these are rounding differences (EXPERIMENT_REPORT.md §3).
- **The correctness reference is NumPy, not compiled MLIR.** It interprets the MLIR literally in
  float64. The harness will also compare compiled MLIR with eager Torch during the runs.
- **Eager times in `preprocess.json` are indicative only**: 1 thread, 5 calls, no pinning. The
  benchmark numbers come from the harness runs.
- **D1 variants depend on v5 internals.** `only_<op>` applies one op's action list (with the producers it fused). `no_fusion` is best-effort: TPF actions are replaced by TP with the same parameters. A variant with transform failures is reported, never silently counted.
- **Block relations are partial.** They are reconstructed from op tags inside the blocks (the extractor's metadata isn't in the dataset), so `feeds` can miss connections. `tag_conflicts` in `blocks_index.json` must be 0 for D2/D3 to be trusted.
- **FLOPs and intensity are static estimates** from loop sizes; bytes are the function ABI traffic, not measured DRAM traffic.
- **Cold-cache mode** rewrites a 1 GiB buffer by default (`--flush-mib`). Check this exceeds the L3 of the allocated Bergamo cores.
