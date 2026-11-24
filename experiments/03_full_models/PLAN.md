# Experiment 03 (plan): MLIR-RL `v5_model_1200` on full programs

**Goal.** For each full neural network, compare the same four methods as experiment 02, on the
same harness:

| Method | What it is |
|---|---|
| MLIR baseline | the whole model through the V5 pipeline, no schedule |
| MLIR-RL | the whole model after `v5_model_1200`'s greedy schedule, op by op |
| Torch eager | an equivalent PyTorch program of the same MLIR |
| Torch Inductor | that program through `torch.compile` |

**Measures.** Per model: latency (median), compile time, time to produce the RL schedule,
correctness, and how many RL actions applied, failed or were skipped.

**Optional extra column:** the original model run natively in PyTorch (torchvision /
HuggingFace), eager and Inductor.

Nothing below has been run yet. Code would live in `experiments/03_full_models/`. Experiments 01
and 02 and the repo's own code stay untouched.

---

## 1. What already exists

### The models: 21 full programs in 3 groups

All baseline times below are from the repo's eval lists; their thread count is unknown.

| Group | Models (MLIR baseline time) | Where |
|---|---|---|
| Paper | mobilenet_v2 0.39 s, resnet 5.4 s, vgg 21.8 s | `data/mlir_rl_v1_paper/model_*.mlir` (you have these) |
| Linalg | gin 0.02 s, gat 0.05 s, mobilenet_v3_small 0.08 s, efficientnet_b0 0.47 s, t5 2.0 s, resnet50 5.5 s, resnext50 5.7 s, convnext_tiny 9.6 s, vgg16 24.7 s | partner: `/scratch/mb10856/MLIR-RL/data/full_models/*_linalg.mlir` (readable) |
| Relowered | bart 3.4 s, distilbert 3.4 s, albert 6.8 s, bert 6.8 s, gpt2 6.7 s, yolov8m 33 s, vit_b_16 58 s, whisper_base 197 s, llama3_2_1b 948 s | partner: `data/full_models/relowered/`. **Not readable** (`drwx------`). |

### Existing evaluation path: `scripts/eval/eval_fullmodel.sh` (repo design doc V5.2)

- **What it does:** a greedy RL rollout over every op of the whole module, then it measures
  MLIR-RL against the stored MLIR baseline time.
- **Configs:**
  - `config/v5/v5_mlir_rl_v1_paper_fullmodel.json`
  - `config/v5/v5_fullmodel_linalg.json`
  - `config/v5/v5_fullmodel_relowered.json`
- **Checkpoint compatibility:** these configs differ from `v5_mlir_rl_v1_paper.json` only in
  their paths, so `v5_model_1200.pt` loads with them. The architecture is identical.
- **Limits for a fair comparison:**
  - **Timing:** only 2 timed runs (`execution.py`).
  - **Baselines:** stored from earlier runs, on other nodes.
  - **Threads:** thread count not controlled.
  - **No Torch:** no Torch methods at all.
  - **No correctness check:** none against an independent program.

### What the earlier work found

- The design doc reports large wins for an older checkpoint (6400): resnet50 12×, vgg16 19×,
  bert 15×, gpt2 18×.
- Its own notes warn that small models varied up to 5× between nodes.
- It also notes that LLVM -O3 compile of gpt2 took about 90 min.

---

## 2. Two tracks

### Track A (quick, about 1 day): the repo's own full-model eval with `v5_model_1200`

This gives MLIR-RL vs MLIR baseline numbers and the RL schedules quickly, with the caveats above.
Use it as a first look and a cross-check for Track B.

```bash
cd /scratch/mh8826/MLIR-RL
mkdir -p /scratch/mh8826/ckpt_dirs/v5_1200 && ln -sf /scratch/mh8826/checkpoints/v5_model_1200.pt /scratch/mh8826/ckpt_dirs/v5_1200/
# eval.py picks up any file matching model_<N>.pt, so v5_model_1200.pt is found as step 1200.
sbatch scripts/eval/eval_fullmodel.sh --checkpoint 1200 \
       --checkpoints-dir /scratch/mh8826/ckpt_dirs/v5_1200 --jobs 3 \
       config/v5/v5_mlir_rl_v1_paper_fullmodel.json        # 3 paper models
sbatch scripts/eval/eval_fullmodel.sh --checkpoint 1200 \
       --checkpoints-dir /scratch/mh8826/ckpt_dirs/v5_1200 --jobs 4 \
       config/v5/v5_fullmodel_linalg.json                  # 9 linalg models
```

**Fixes needed before submitting `eval_fullmodel.sh`:**
- **Hard-coded paths:**
  - the `--output`/`--error` paths point to `/scratch/mb10856/...`; override with
    `sbatch -o/-e`;
  - it sources `$HOME/envs/mlir`; set `CONDA_ENV` to your env's activate script, or adapt a
    copy inside `experiments/03_full_models/`.
- **Where results go:** they are written under the config's `results_dir`, so point that at
  your own folder.
- **Relowered models:** they need the partner's files, see §3 step 0.

### Track B (rigorous): experiment 03, four methods on the experiment-01 harness

This is the experiment-02 pipeline applied to whole models. The steps are below.

---

## 3. Track B step by step

### Step 0. Prerequisites (about half a day)

1. **Data access.** Ask your partner to run:
   ```bash
   chmod -R a+rX /scratch/mb10856/MLIR-RL/data/full_models/relowered
   ```
   That folder holds the weight-stripped, `tm_tensor`-free versions of the 9 transformer and large
   models. Rebuilding them yourself needs torch-mlir, which is not installed here.
2. **Disk.** You are at 470,334 files of a 500,000 soft limit.
   - Delete the 247 GB of Inductor `.gch` caches in experiment 02 first; they hold no results.
   - Set one shared Inductor cache for all runs (`TORCHINDUCTOR_CACHE_DIR=/scratch/mh8826/inductor_cache`)
     so each run stops writing its own 158 MB header.
   - Keep inputs (`.npz`) off by default for big models. vit/llama inputs are GBs each.
3. **Environment.** Same as experiment 02: `overnight/env.sh` with the LLVM 19 build, gcc 12,
   binutils 2.46, MLIR-ENV. No new build is needed.

### Step 1. Prepare each model as a harness workload (`prepare_models.py`, about 1 day)

For each model, write a new file `experiments/03_full_models/data/<model>.mlir`. The originals
are never edited.

1. **Load.** Read the original MLIR.
2. **Expand elided weights.** Materialise `dense_resource<__elided__>` weights
   (`rl_autoschedular_v5/full_model.materialize_elided_resources`).
3. **Lift weights to arguments.** Turn every large weight constant into an argument of `@main`.
   - **Why:** the stripped files hold weights as zero constants. Zero weights time the same, but
     they make every output nearly constant, so a correctness check would pass trivially.
   - **What it gives:** the harness then feeds random weights like any other input, and the
     translator's domain rules keep `sqrt`/`rsqrt`/`log`/`pow` inputs valid (as in exp 02).
   - Small and scalar constants stay inline.
4. **Timed ABI.** Rewrite the entry to the timed `@main -> (..., i64)` form
   (`full_model.ensure_timed_main_abi`, same as `scripts/full_model/add_timing_wrapper.py`).
5. **Op tags.** Tag every linalg op with `{tag = "operation_N"}` (`scripts/full_model/preprocess_model.py`
   runs the AST dumper).
6. **Record per model:** op count, op kinds, total FLOPs, input/weight bytes, and whether the
   file parses and runs once through the MLIR baseline. Write `results/catalog/models.json`.

**Check:** each prepared file must run once through `mlir_baseline` in the harness and finish
close to the repo's stored baseline time.

### Step 2. Torch side

**2a. Translation of the same MLIR (main comparison, about 2 days).**

Extend a copy of `02_block_diagnosis/block_to_torch.py`. It already covers every linalg kind in
these models: generic bodies, matmul/batch_matmul, conv, depthwise conv, pooling, fill,
transpose and broadcast. It still needs these ops:

| MLIR op | PyTorch |
|---|---|
| `tensor.pad` | `F.pad` |
| `tensor.expand_shape` / `collapse_shape` | `reshape` |
| `tensor.extract_slice` / `insert_slice` | slicing / `index_copy` |
| `tensor.concat` | `torch.cat` |
| `tensor.extract`, `linalg.index`, `arith.cmpi`, `arith.index_cast` | indexing / `arange` / comparisons |
| `tm_tensor.attention` | `F.scaled_dot_product_attention`. Not needed if the relowered files are used, since those have it lowered to loops. |

Then build the same kind of op graph as for blocks, with a whole-model output instead of one
block output. Test it in two ways:
- **Small models:** compare with `reference_eval.py` (float64) on gin, gat and
  mobilenet_v3_small. Raise `MAX_PARALLEL`, or check layer by layer.
- **Every model:** compare with compiled MLIR in the harness's own correctness step.

**2b. Native PyTorch model (optional extra column, about 1 day).**

- **Loaders:** `scripts/baseline/get_pytorch_baselines.py` already loads the torchvision and
  HuggingFace models.
- **Missing pieces:**
  - its registry `results/full_model_1/pytorch_models.json` is missing, so recreate it;
  - it times eager and `torch.jit`, so add `torch.compile`;
  - set the thread count and input shapes to match the MLIR `@main` (batch 1, 224×224, the
    sequence lengths in the MLIR).
- **Weights:** download them on the login node; compute nodes may have no internet.
- **Caveat:** this is not the same computation. The ONNX export folds or reorders some ops, so
  report it as a reference column, not in the main speedup.

### Step 3. RL schedule on the whole model (reuse, about half a day)

- **Scheduler:** the harness's `policy_schedule` already extracts features for every op of the
  module and runs the greedy policy.
- **Application mode:** use experiment 02's **lenient** mode (`block_experiment.install`), which
  records failed and skipped actions in `rl_application.json`.
- **Step limit:** raise `rl.max_steps` from 1000 to about 30× the op count. gpt2 has 984
  generics plus about 100 named ops.
- **Timing:** record the scheduling time separately. For big models it can be minutes.

### Step 4. Harness settings by model size

| Tier | Models | warmup / repeats | Correctness cases | `perf` | Time limit per method | Memory |
|---|---|---|---|---|---|---|
| Small (< 1 s) | gin, gat, mobilenets, efficientnet | 5 / 30 | 3 | on | 2 h | 32 GB |
| Medium (1–60 s) | t5, bart, distilbert, albert, bert, gpt2, resnets, resnext, convnext, vgg, yolov8m, vit | 1 / 10 | 1 | off | 8 h (gpt2 compile ~90 min) | 64 GB |
| Large (> 60 s) | whisper (197 s), llama (948 s) | 1 / 3 | 1 | off | 24 h | 128–256 GB |

**Threads**
- **Primary configuration, 1 thread:** directly comparable with experiment 02's block results.
  At 1 thread llama may take hours per run, so run llama last, or only in the multi-thread
  configuration.
- **Second configuration, 16 threads for all methods:** whole models are normally run
  multi-threaded, and RL schedules parallelise loops with OpenMP, which a 1-thread run cannot
  reward. Set it through `OMP_NUM_THREADS` and `torch.set_num_threads`, with pinned cores.

**Node sharing**
- **Request whole nodes or sockets:** use `--exclusive`, or at least `--cpus-per-task` equal to a
  full CCD. Long whole-model timings are more sensitive to neighbours than blocks were.
- **Same node for all four methods:** run them in one job, back to back. The repo's earlier
  numbers mixed nodes, and that caused the 5× swings.

### Step 5. Run it (reuse experiment 02's overnight pattern)

- **Jobs:** one Slurm array task per model, each running all four methods on its own node.
- **Pipeline:**
  - reuse `overnight/` (settings, env, run, retry, report) with a new `settings.sh`;
  - `summary.json` per model marks it done;
  - resubmitting skips finished models;
  - a retry stage re-runs timed-out methods with a longer limit.
- **Order of work:**
  1. **Smoke test:** gin, gat, mobilenet_v3_small (minutes).
  2. **The 3 paper models,** then compare with Track A numbers.
  3. **The remaining linalg models.**
  4. **The relowered group,** smallest first, then whisper and llama last.

### Step 6. Analysis and report

**Main table (per model):**
- median latency of the 4 methods;
- speedups vs MLIR baseline and RL vs Inductor;
- compile time, RL scheduling time;
- correctness; failed and skipped RL actions.

**Summary across models:** geomean and median across the 21 models (not the arithmetic mean),
with win/tie/loss counts.

**Figures:**
- grouped bars per model with a log axis, one bar per method;
- RL speedup on the full model vs the geomean RL speedup of that model's blocks from
  experiment 02. This tests whether block results predict whole-model results.

**Cross-checks:**
- Track A numbers vs Track B: same schedule, different timing method.
- The repo's earlier checkpoint-6400 full-model numbers.

**Write-up:** `EXPERIMENT_REPORT.md` in the same format as experiment 02.

---

## 4. Time and compute estimate

| Phase | Your time | Compute |
|---|---|---|
| Step 0 + Track A | 1 day | ~1 node-day (12 models) |
| Steps 1–3 (code + tests) | 4–5 days | small |
| Step 5, 1 thread, 19 models without llama/whisper | — | ~2–4 node-days, mostly compile and the 20–60 s models |
| Whisper + llama, 1 thread | — | 2–4 node-days, or skip at 1 thread |
| 16-thread configuration | — | ~1–2 node-days |
| Step 6 report | 1 day | — |

**Total:** about 1.5–2 weeks of work, mostly the translator extension and the weight-lifting
preparation. Compute fits within a few days of the partition.

## 5. Risks and what to do about them

| Risk | Mitigation |
|---|---|
| Relowered files unreadable | Partner `chmod`. Otherwise run 12 models only and note it. |
| Translator gaps in some model (new op or body) | Log `RefUnsupported`/translation errors per model. Report Torch "n/a" for that model, never drop it silently. |
| LLVM -O3 compile takes hours (gpt2) | Long limits for that tier. Compile time is recorded as a result. |
| RL schedule fails on many ops (TiledFusion assertion, omp `unrealized_conversion_cast`) | Lenient mode plus per-action record. Report the strict-mode count too. |
| Random weights produce NaN/Inf (norms, `pow`, `exp`) | Same domain rules as exp 02. Check outputs are finite before comparing. |
| fp32 tolerance too tight for long reductions (seen in exp 02) | Also record the error scaled to the output magnitude. Judge borderline cases against float64 where feasible. |
| Quota (files and size) | Clear `.gch` caches, use a shared Inductor cache, don't keep big inputs. |
| Node interference / variance | Exclusive nodes, all methods in one job, ≥10 repeats; re-run a few models to measure variance. |

## 6. Decisions needed from you

1. **Thread configuration:** 1 thread only (matches exp 02), or 1 thread plus 16 threads?
2. **The native PyTorch column (Step 2b):** include it or not?
3. **llama3_2_1b (~16 min per run at baseline):** include it, or leave it out at 1 thread?
4. **Track A:** run it first as a quick look (recommended), or go straight to Track B?
