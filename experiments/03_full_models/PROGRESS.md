# Experiment 03 (full programs): progress log

Plan: `PLAN.md`. Track A = the repo's own full-model RL eval with `v5_model_1200`, 12 models, own same-node baselines.

## Track A: how it works / resume
- **Files:**
  - `track_a/prepare.py` writes `track_a/models.tsv` and `track_a/configs/`. These are copies of the repo configs with absolute paths; repo files are untouched.
  - `track_a/launch.sh` submits 2 arrays (1 thread with 2 CPUs, 16 threads with 16 CPUs), one task per model, at most 6 at once.
- **Per task, on one node:**
  1. MLIR baseline (`scripts/baseline/baseline_fullmodels.py`);
  2. MLIR-RL (`scripts/eval/eval.py`, checkpoint `/scratch/mh8826/ckpt_dirs/v5_1200/model_1200.pt` -> `v5_model_1200.pt`);
  3. MLIR baseline again.
- **Outputs:** `track_a/results/t<threads>/<model>/{baseline_1,rl,baseline_2}/`; done markers in `track_a/state/`.
- **Resume:** `bash track_a/launch.sh`. Finished models exit immediately, and finished steps are skipped.

## Log
- 2026-11-10 00:17: smoke test (gin) jobs 19296490 (1t), 19296491 (16t): COMPLETED.
  - 1 thread: baseline 13.9 ms both runs, RL 3.9 ms (3.5×).
  - 16 threads: baseline 14.1 / 13.9 ms, RL 17.8 ms.
- 2026-11-11 02:22: launched all 12 models: 1t job 19340442, 16t job 19340443 (arrays 0-11, at most 6 at once).
- 2026-11-11 03:09: first pass: 16/24 tasks OK.
  - 8 tasks OOM-killed at 48 GB in the RL step: t5, model_res_net, resnext50, model_vgg, at both thread settings. Peak 45-50 GB; TiledFusion assertion retries grow memory.
  - convnext_tiny: RL schedule does not compile (omp.wsloop + unrealized_conversion_cast, same bug as exp 02). Recorded as an RL failure, not rerun.
- 2026-11-11 03:12: reran the 4 OOM models with 128 GB: 1t job 19340965, 16t job 19340966 (tasks 5,6,8,10). Baseline_1 is reused; RL and baseline_2 are rerun.
  - launch.sh now takes MEM / MAXPAR.
  - collect.py writes track_a/results/track_a.{csv,md}.
- Interim (7 models): RL geomean 1.49× at 1 thread, 1.94× at 16 threads.
  - At 16 threads: resnet50 9.8×, vgg16 13.1×.
  - On small models the RL schedule is slower at 16 threads (gin, gat, mobilenet_v3_small).
- 2026-11-11 (final, recorded 2026-11-14): Track A reruns at 128 GB finished.
  - model_vgg OK: 1.37× at 1 thread, 10.6× at 16 threads.
  - t5, model_res_net, resnext50: RL failed. Out of memory even at 128 GB, or no execution time.
  - Final geomean: 1.48× at 1 thread (8 models), 2.4× at 16 threads (8 models).
  - Full table: track_a/results/track_a.md.
- 2026-11-14: user asked to delete all experiment results, keeping only the translated programs, the block catalog + selection, and the written reports. Raw Track A results, logs and state were deleted; track_a.md is kept.
