# scripts/ — MLIR-RL Orchestration

No files at root. Everything organized into subfolders by purpose.

## Directory Layout

```
scripts/
├── train/           Training entry points + Slurm wrappers
├── eval/            Evaluation + ablation eval entry points + Slurm wrappers
├── baseline/        MLIR & PyTorch baseline timing scripts
├── full_model/      Full-model end-to-end optimization
├── checkpoint/      Checkpoint scan + results merger
├── data/            Dataset processing (split, analysis)
└── utils/           Sanity checks + workflow helpers
```

## Subfolder Details

### `train/`
| File | Purpose |
|------|---------|
| `train.py` | PPO training loop (core entry point) |
| `train.sh` | Slurm training wrapper (also handles array jobs) |

### `eval/`
| File | Purpose |
|------|---------|
| `eval.py` | Evaluation loop (core entry point) |
| `ablation_eval.py` | Ablation study evaluation (requires `EVAL_DIR`) |
| `eval.sh` | Generic Slurm eval (resolves impl from config) |
| `eval_fullmodel.sh` | V5.2 full-model eval (greedy rollout, one model per process, `--jobs N`) |
| `submit_fullmodel_matrix.sh` | Checkpoints × models matrix driver over `eval_fullmodel.sh` |
| `merge_fullmodel_matrix.py` | Merge matrix cells into per-checkpoint summary |
| `submit_checkpoint_evals.sh` | Batch submit checkpoint evaluations |

### `baseline/`
| File | Purpose |
|------|---------|
| `get_base.py` | MLIR baseline execution timing |
| `get_pytorch_times.py` | Block-level PyTorch timing (eager + JIT per op) |
| `get_pytorch_baselines.py` | **Canonical** PyTorch eager + JIT timing for 22 models |
| `get_base.sh` | Slurm wrapper for `get_base.py` (chunked) |
| `get_pytorch_times.sh` | Slurm wrapper for `get_pytorch_times.py` (chunked) |

### `full_model/`
| File | Purpose |
|------|---------|
| `preprocess_model.py` | Runs C++ AST dumper to tag linalg ops |
| `add_timing_wrapper.py` | Wraps @main with @nanoTime() |
| `merge_full_model_results.sh` | Merges chunk files into unified JSON + CSV |
| `get_pytorch_full_times.sh` | Full PyTorch baseline timing |
| `retired/` | Pre-V5.2 v4_5 orchestrators (`optimize_full_model.py/.sh`, `optimize_model_via_blocks.py`) — see `../checkpoint/retired/README.md` |

### `checkpoint/`
| File | Purpose |
|------|---------|
| `retired/` | Pre-V5.2 v4_5 scan driver (`ckpt_scan_all.sh`, `submit_ckpt_scan.sh`, `merge_ckpt_scan.py`) — replaced by `eval/submit_fullmodel_matrix.sh` + `eval/merge_fullmodel_matrix.py` |

### `data/`
| File | Purpose |
|------|---------|
| `split_json.py` | Stratified train/eval split of benchmark JSON |
| `build_checkpoint_comparison.py` | Merges full-model + block results into comparison CSV/JSON |

### `utils/`
| File | Purpose |
|------|---------|
| `test_torch_mlir_compile.py` | MLIR binding sanity check |
| `test_torch_mlir.sh` | Shell wrapper for the above |
| `submit_and_monitor.sh` | Submit Slurm job + auto-tail output |


## Standard Workflow

### Old Dataset

```bash
# 1. MLIR baseline
sbatch scripts/baseline/get_base.sh config/old_dataset/train/v4_5.json

# 2. PyTorch baseline
sbatch scripts/baseline/get_pytorch_times.sh config/old_dataset/train/v4_5.json

# 3. Split train/eval
python scripts/data/split_json.py config/old_dataset/train/v4_5.json

# 4. Train
sbatch scripts/train/train.sh config/old_dataset/train/v4_5.json

# 5. Evaluate
sbatch scripts/eval/eval.sh config/old_dataset/train/v4_5.json
```

### New Dataset

```bash
# 1. MLIR baselines (3 directories)
python scripts/baseline/get_base.py --config config/new_dataset/train/v4_5.json
# Repeat for eval and eval_full

# 2. Train V4.5 on Bergamo
sbatch scripts/train/train.sh config/new_dataset/train/v4_5.json

# 3. Eval V4.5 on Bergamo (5-run median)
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_5_eval.json

# 4. Full model eval (all ops)
sbatch scripts/eval/eval.sh config/new_dataset/full_model/v4_5_eval_full.json
```
