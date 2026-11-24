# Overnight run: progress log

## Current state (2026-11-09): COMPLETE
| Stage | Jobs | Result |
|---|---|---|
| Setup (LLVM 19 + tools + env + checkpoint match) | 19168191 | done, `state/setup.done` |
| Main run, 48 shards (1200 s per method) | 19168867_[0-47] + shard 31 rerun 19191342 | 1474/1474 distinct blocks, all 4 methods |
| Interim report | 19191343 | superseded by the final report |
| Retry planning | 19191747 | 74 blocks had a timed-out method |
| Retry, 30 tasks (3600 s per method) | 19191748_[0-29] | 74/74 re-run, 13:20-16:51 on 2026-11-08 |
| Final report | 19191749 | `MORNING_REPORT.md`, `analysis/`, `figures/` |
| Op-type figure | interactive, 2026-11-09 | `figures/speedup_by_op_type{,_unique}.png` |
| Correctness and loss audit | interactive, 2026-11-09 | `../EXPERIMENT_REPORT.md` §3-4 |
| Documentation | 2026-11-09 | `../EXPERIMENT_REPORT.md`, README status/outputs/limits updated |

**Final numbers** (geomean vs mlir_baseline, both passed):
- mlir_rl 0.885× over 1405 blocks: matmul 1.73, conv 0.76, reduction 0.59; win/tie/loss 680/156/569.
- torch_eager 10.1×.
- torch_inductor 12.9×.

**Remaining failures:**
- mlir_rl: 42 errors (15 JIT crashes, ~23 powi lowering, 4 ExecutionEngine) and 10 timeouts at 3600 s.
- mlir_baseline: 11 timeouts (whisper).
- 11 blocks fail the harness tolerance for every method: fp32 rounding, not wrong results.

**Not done, by design:**
- The 1400 single-op files of ops_and_blocks; this experiment covers blocks only.
- The D1-D4 ablations (`ablation.py`).

Goal: evaluate checkpoint `v5_model_1200.pt` on all blocks of ops_and_blocks through experiment 01's
harness (unchanged) with all four methods: mlir_baseline, mlir_rl, torch_eager, torch_inductor.

All blocks = 6693 blocks = 1474 unique computations (same signature = same MLIR modulo op tags).
Each unique computation is measured once; `map_duplicates.py` expands the results to all 6693 blocks
(`results/ckpt_v5_model_1200/all_blocks_results.csv`, column `measured_as`).

## How to resume after anything stops
```bash
bash /scratch/mh8826/MLIR-RL/experiments/02_block_diagnosis/overnight/overnight.sh
```
It skips setup if `state/setup.done` exists and skips every block that already has a `summary.json`.
- Setup steps are also individually idempotent: LLVM is not rebuilt once `mlir-opt` and the bindings exist.
- Job ids of every launch are in `state/jobs.log`. Logs are in `logs/`.

## Saved milestones (where state lives)
| Milestone | Proof on disk |
|---|---|
| LLVM 19 built | `/scratch/mh8826/llvm-project/build/bin/mlir-opt` |
| Tools built | `tools/{ast_dumper,pre_vec,vectorizer}/build/bin/*` |
| `.env` written | `/scratch/mh8826/MLIR-RL/.env` |
| Checkpoint ↔ config | `state/checkpoints.tsv` |
| Setup finished | `state/setup.done` |
| Block selection frozen | `results/ckpt_v5_model_1200/selection.json` (1474 blocks) |
| Each block done | `results/ckpt_v5_model_1200/t1_warm/r0/<class>/artifacts/<block>/<UTC>/summary.json` |
| Shard done | `state/run_ckpt_v5_model_1200_shard<k>.done` |
| Final tables | `results/ckpt_v5_model_1200/all_results.csv`, `all_blocks_results.csv`, `analysis/`, `figures/`, `MORNING_REPORT.md` |

## Log
- 2026-11-08: translation verified for all 6693 blocks (`results/preprocess/`).
  Overnight pipeline written; nothing in experiment 01 modified.
- 2026-11-08 05:41:01: launched setup=19164487, run array=19164488 (48 shards, max 24 at once), report=19164489.
- 2026-11-08 05:54:48: first setup failed at cmake (in-tree openmp tests need the clang target). Added clang (project recipe), verified configure, relaunched: setup=19165695 runs=19165697 report=19165706
- 2026-11-08 06:16:36: build 2 reached 5549/6418, failed linking libomp.so: system ld 2.30 cannot read gcc-12 libgcc compressed debug sections. Added module binutils/2.46.0 (tested) to setup and env.sh. Incremental rebuild.
- 2026-11-08 06:17:24: Inductor verified with gcc12+binutils. Relaunched: setup=19167485 runs=19167508 report=19167540
- 2026-11-08 06:26:27: setup 3 completed but checkpoint match failed: rl_autoschedular_v5 imports pytz (not installed). Installed pytz, verified strict load with v5_mlir_rl_v1_paper.json, relaunched: setup=19168191 runs=19168192 report=19168193
- 2026-11-08 06:30:13: first real runs: Torch translations pass vs compiled MLIR; mlir_rl failed on ~half the blocks (TiledFusion transform error -> harness is strict). Cancelled array 19168192 to add a recorded lenient application mode (same rule as the RL Env).
- 2026-11-08 06:32:49: lenient mode tested on albert_block_1 (all 4 methods measured+passed; 1 TPF failed, 1 skipped, 11 applied). Run jobs now use --rl-apply lenient, 24G each. Strict runs of the first 19 blocks are kept in their older timestamp folders and rerun. Relaunched: setup=done runs=19168867 report=19168870
- 2026-11-08 06:53:28: 20-min check: 344 blocks done in lenient mode; baseline/eager/inductor 344/344 measured+passed; mlir_rl 339 measured+passed, 5 errors (RL schedules with omp.wsloop leave unrealized_conversion_cast -> LLVM translation fails / segfault in JIT; genuine result, recorded). ~17 blocks/min -> ETA ~1.5-2 h.
- 2026-11-08 06:54:21: report stage dry-run OK on partial data (345 blocks): collect, stats, 6 figures, exp-01 plot, all_blocks mapping, MORNING_REPORT. Partial geomean vs baseline: mlir_rl 0.965, eager 7.32, inductor 9.02 (1 thread).
- 2026-11-08 11:05:17: 42/48 shards done, 6 on their last block. Shard 31 never ran (srun launch timeout on bn002) but was marked done (bug). Fixed run.sbatch (retry srun x3; .done only when all blocks have results). Resubmitted shard 31 as 19191342; report moved to 19191343 (after all shards).
- 2026-11-08 11:05:45: interim (1437/1474 unique, 6532/6693 blocks): geomean vs baseline: mlir_rl 0.90x (matmul 1.77, conv 0.79, reduction 0.58), eager 9.3x, inductor 11.9x. Timeouts (1200 s): baseline 49, rl 54. RL lenient: 140/1342 had a failed action.
- 2026-11-08 11:12:56: added retry stage (retry.sbatch, retry_launch.sh): after all shards, plan blocks whose latest run had a timed-out method (68 so far: whisper 27, llama 23, efficientnet 16, yolov8 2), re-run them (all 4 methods) with a 3600 s limit in 30 tasks, then final report. retry: plan->arrays=19191748 report=19191749 (after 19168867:19191342)
- 2026-11-08 12:59:23: main run COMPLETE: all 48 shards + shard 31 rerun finished; 1474/1474 unique blocks have lenient results (74 with a timed-out method). Planner 19191747 + interim report 19191343 pending in queue (Priority). Retry array 19191748 and final report 19191749 wait on them.
- 2026-11-08 13:16:47: interim report 19191343 COMPLETED (all 1474 blocks): geomean vs baseline mlir_rl 0.904, eager 9.31, inductor 11.91; timeouts baseline 55, rl 58. Planner 19191747 still pending (Priority).
- 2026-11-08 13:19:10: planner 19191747 COMPLETED: 74 blocks listed for retry (state/retry_list_ckpt_v5_model_1200.txt). Retry array 19191748 pending (Priority).
- 2026-11-08 13:31:05: retry array 19191748: all 30 tasks RUNNING since 13:20 (bn001-bn010), first blocks progressing (baseline done -> mlir_rl). Final report 19191749 waits on it.
- 2026-11-09 22:09:59: (checked) retry array 19191748 COMPLETED 13:20-16:51 on 2026-11-08 (74/74 blocks retried at 3600 s); final report 19191749 COMPLETED 16:56. FINAL: geomean vs baseline mlir_rl 0.885x, eager 10.1x, inductor 12.9x; remaining timeouts baseline 11, rl 10. PIPELINE DONE.
- 2026-11-09: session check: nothing lost. Retry array 19191748 COMPLETED (all 30 tasks, 74/74 blocks), final report 19191749 COMPLETED 2026-11-08 16:56.
- 2026-11-09: added `plot_op_types.py`: mean and median speedup per op type, all 4 methods. A block counts in each op type it contains; every block has 5 ops with 2-5 op types.
- 2026-11-09: correctness audit:
  - The 11 blocks "failed" by baseline/eager/inductor are the same 11 for all methods: MLIR vs Torch fp32 reduction rounding just above atol=1e-5. The Torch translations match the fp64 reference to 3e-7..3e-6 relative error.
  - RL's 12th failure (vgg16_block_6) is the same effect.
  - The 5 "unverified" RL results lack a baseline (timeout) but pass against Torch eager.
- 2026-11-09: loss audit:
  - Median RL loss is 0.36×; 348 blocks are below 0.5×.
  - Repeat measurements change by 1.1% (baseline) and 3.5% (RL) median.
  - Losses are not caused by failed actions: blocks without failures lose more (42% vs 25%).
- 2026-11-09: wrote `EXPERIMENT_REPORT.md` and updated README. results/ckpt_v5_model_1200 is 313 GB of data (626 GB allocated) on /scratch.
- 2026-11-09: disk check: results/ckpt_v5_model_1200 = 313 GB apparent (626 GB allocated by Lustre), 162,674 files; 247 GB are 1,570 identical Inductor .h.gch caches, 85 GB input .npz. User quota: 653 GB of 4.9 TB, 470,334 files of a 500,000 soft limit. Nothing deleted.
