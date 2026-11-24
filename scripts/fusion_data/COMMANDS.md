# From the current state to training every ablation row (A, A′, B, C, D, D+)

All commands run from the repo root `/scratch/mh8826/MLIR-RL`. Lines marked **login** are light
(seconds, or network-only). Everything heavy is an `sbatch` job on `--partition=compute --constraint=bergamo`,
the node type that `train.sh`/`eval.sh` already use (AMD EPYC 9754).
Nothing below has been run by Claude.

## 0. Checks and one-time setup (login)
```bash
cd /scratch/mh8826/MLIR-RL
source scripts/analysis/gap1/env.sh
# pytest is not in MLIR-ENV: install it to a private folder (the conda env stays unchanged)
python -m pip install --quiet --target $HOME/.local/pytest-lib pytest
for f in $(find rl_autoschedular/rl_autoschedular_v6 scripts/fusion_data -name "*.py" -not -path "*__pycache__*"); do python -m py_compile "$f" || echo "FAIL $f"; done
PYTHONPATH=$HOME/.local/pytest-lib:$PYTHONPATH python -m pytest rl_autoschedular/rl_autoschedular_v6/fusion/tests -q
python scripts/fusion_data/check_control.py                     # control file valid, prints the 37 enabled models
ln -s /scratch/mh8826/data/full_models data/full_models          # eval configs use data/full_models
mkdir -p logs
```

## 1. Export environment (after you approve option B; see CHANGES.md Part 2)
```bash
bash scripts/fusion_data/setup_export_env.sh find-commit         # login (network): note the torch-mlir sha for llvm cd708029
export TORCH_MLIR_COMMIT=<sha from torch_mlir_llvm_bumps.txt>
bash scripts/fusion_data/setup_export_env.sh create-env          # login (network): conda env mlir-export
sbatch --export=ALL scripts/fusion_data/sbatch/build_torch_mlir.sbatch
bash scripts/fusion_data/setup_export_env.sh check               # login, after the build job
```

## 2. Model configs (login, network, a few KB each)
```bash
(cd /scratch/mh8826/model_survey && $HOME/.conda/envs/mlir-export/bin/python download_models.py \
    /scratch/mh8826/MLIR-RL/data/fusion_units/models.json --cache hf_cache)
# fixes needed if it reports a missing repo (e.g. resnet26): edit data/fusion_units/models.json, re-run check_control.py
```

## 3. Compatibility test: resnet18, distilgpt2, deit_tiny (go/no-go for the export env)
```bash
sbatch scripts/fusion_data/sbatch/compat_test.sbatch
# read logs/fd_compat_<id>.out and data/fusion_units/compat/validation.json: all 3 ok, tm_tensor lines = 0
```

## 4. Data pipeline (chained jobs)
```bash
J1=$(sbatch --parsable scripts/fusion_data/sbatch/export.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/fusion_data/sbatch/validate.sbatch)
J3=$(sbatch --parsable --dependency=afterok:$J2 scripts/fusion_data/sbatch/units.sbatch)
sbatch --dependency=afterok:$J2 scripts/fusion_data/sbatch/profile.sbatch      # report only, not blocking
J5=$(sbatch --parsable --dependency=afterok:$J3 scripts/fusion_data/sbatch/baselines_units.sbatch)
sbatch scripts/fusion_data/sbatch/baselines_eval.sbatch                         # independent; long (full models)
```
After J3 (login, seconds):
```bash
python scripts/fusion_data/split.py split                       # by model, stratified by family, seed 0 (written once)
```
After J5 (login, seconds):
```bash
python scripts/fusion_data/split.py bench-json --times results/fusion_units_results/baselines/mlir/all_units.json
# -> results/fusion_units_results/baselines/mlir/base_train.json and base_val.json
```
Check before training: `data/fusion_units/profile/report/report.md` (section 4b) and
`data/fusion_units/units/units_manifest.json` (skipped units, missing op names).

## 5. Register the runs (experiments.json, as AGENTS.md requires)
One entry per run, e.g.
`{"name": "fusion_D", "config": "config/fusion/D.json", "results_dir": "/scratch/mh8826/fusion_runs/fusion_D_agent", "mode": "train", "seed": 42, "state": "pending", "description": "fusion layer row D: units + prefuse + producer_select"}`.
For more seeds: copy the config, change `seed` and `results_dir` (and the eval twin's `results_dir`).

## CPU usage (applies to every job below)
Every job that executes MLIR code uses one explicit split: **8 programs at a time × 8 OpenMP threads = 64 cores**
(`scripts/fusion_data/sbatch/_threads.sh`).
- The `scripts/fusion_data/sbatch/*` jobs set it themselves.
- Training and evaluation must be submitted through `scripts/fusion_data/submit.sh`, which passes the split with
  `--export` (overriding an `OMP_NUM_THREADS=1` your shell may carry from `scripts/analysis/gap1/env.sh`) and checks
  that it matches `--cpus-per-task`.
- To verify after a job: `seff <jobid>` (CPU efficiency well above 1/64), and the first line of the sbatch logs prints
  `exec_workers=8 omp_threads=8`.

## 6. Training, one job per row (all from scratch; never --resume from a v5 run)
The fusion configs are single-node, so give the job the single-node size (see train.sh's header):
```bash
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/A.json        # A : blocks, layer off
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/A_prime.json  # A′: units, layer off
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/C.json        # C : units + prefuse
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/D.json        # D : + producer_select
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/D_plus.json   # D+: + tile guard
sbatch scripts/fusion_data/sbatch/config_b.sbatch                                                                  # B : no training (layer alone)
```
**Row A needs its own baselines first** (none exist on this machine for `data/ops_and_blocks`):
```bash
sbatch scripts/fusion_data/sbatch/baselines_blocks.sbatch          # 8,093 files, resumable (re-submit to continue)
# after it finishes (login):
python scripts/data/split_json.py --input results/ops_and_blocks_results/baselines/mlir/base.json
# -> base_train.json / base_eval.json, the paths config/fusion/A.json reads; then submit A's training
```
It does not depend on the export env or the unit pipeline, so it can run right away, in parallel with everything else.
Known caveat: all 6,693 blocks in `ops_and_blocks` come from the 18 eval models, so A has seen eval-model structures.

**Checkpoints** are written outside the git repo, under `/scratch/mh8826/fusion_runs/fusion_<row>_agent/models/`:
`model_<step>.pt` every 50 iterations (model + optimizer + step; `scripts/train/train.py:206`) plus a final one
(`train.py:224`). Logs, exec cache and `train_results.json` are in the same folder. To continue a stopped run:
`bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/D.json --resume /scratch/mh8826/fusion_runs/fusion_D_agent`.

## 7. Evaluation (after training; N = a saved checkpoint, e.g. 500)
Full models (`eval_fullmodel.json`, written by `baselines_eval.sbatch`):
```bash
bash scripts/fusion_data/submit.sh --cpus-per-task=64 scripts/eval/eval_fullmodel.sh --checkpoints-dir /scratch/mh8826/fusion_runs/fusion_A_agent/models      --checkpoint N --jobs 4 config/fusion/A_eval_fullmodel.json
bash scripts/fusion_data/submit.sh --cpus-per-task=64 scripts/eval/eval_fullmodel.sh --checkpoints-dir /scratch/mh8826/fusion_runs/fusion_A_prime_agent/models --checkpoint N --jobs 4 config/fusion/A_prime_eval_fullmodel.json
bash scripts/fusion_data/submit.sh --cpus-per-task=64 scripts/eval/eval_fullmodel.sh --checkpoints-dir /scratch/mh8826/fusion_runs/fusion_C_agent/models       --checkpoint N --jobs 4 config/fusion/C_eval_fullmodel.json
bash scripts/fusion_data/submit.sh --cpus-per-task=64 scripts/eval/eval_fullmodel.sh --checkpoints-dir /scratch/mh8826/fusion_runs/fusion_D_agent/models       --checkpoint N --jobs 4 config/fusion/D_eval_fullmodel.json
bash scripts/fusion_data/submit.sh --cpus-per-task=64 scripts/eval/eval_fullmodel.sh --checkpoints-dir /scratch/mh8826/fusion_runs/fusion_D_plus_agent/models  --checkpoint N --jobs 4 config/fusion/D_plus_eval_fullmodel.json
```
(Each full model is one program: `--jobs 4` runs 4 models × 8 threads = 32 of the 64 cores.)
Held-out unit validation (`base_val.json`), optional:
```bash
bash scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/eval/eval.sh config/fusion/D.json --checkpoint N   # same for A_prime, C, D_plus
```
