# Running both checkpoints through the harness

Goal: for each checkpoint in `/scratch/mh8826/checkpoints/`, measure on the same blocks:

| Method | What it is |
|---|---|
| `mlir_baseline` | the block compiled by the plain MLIR pipeline, with no schedule |
| `mlir_rl` | the same pipeline after applying the checkpoint's schedule |
| `torch_eager` | our verified Torch translation, run eagerly |
| `torch_inductor` | the same graph after `torch.compile(backend="inductor")` |

Everything goes through experiment 01's `experiment.py`, which is not modified: same timing, same
correctness checks, same `summary.json`. `run_blocks.py` picks the blocks, writes the manifests and
starts the runs.

Paths used below:
```bash
REPO=/scratch/mh8826/MLIR-RL
E=experiments/02_block_diagnosis
CK1=/scratch/mh8826/checkpoints/v5_model_1200.pt
CK2=/scratch/mh8826/checkpoints/v5_3_set1_base_p_model_3450.pt
```

---

## Step 0: One-time setup (if not done yet)

Follow `BLOCKS_RUNBOOK.md` **Steps 0–4**: compute node, LLVM 19 with Python bindings, the C++ tools,
and `.env`. Two things differ from that runbook:

```bash
conda activate MLIR-ENV
pip install -r <(grep -v '^torch' $REPO/requirements.txt)   # torch 2.x is already installed
module unload gcc; module load gcc/12.2.0                    # TorchInductor needs a C++ compiler (gcc/13.2.0 here is NOT a compiler)
```
**Why:**
- `MLIR-ENV` already has torch 2.15, and reinstalling it pulls the multi-GB CUDA wheels again.
- `torch_inductor` compiles generated C++ at run time, so a recent `g++` must be on `PATH`. Put the
  same `module load` line in your shell before `sbatch`, or add it to `.env` / `run_blocks.sbatch`.

**Check:** both lines must print without an error.
```bash
cd $REPO && set -a && source .env && set +a
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular:$PYTHONPATH"
python -c "from mlir.ir import Context; import torch; print('MLIR + torch', torch.__version__)"
python -c "import torch; f=torch.compile(lambda x: x*2+1); print('inductor OK', f(torch.ones(3)))"
```

## Step 1: Link data and checkpoints into the repo

```bash
cd $REPO
ln -s /scratch/mh8826/data/ops_and_blocks data/ops_and_blocks
ln -s /scratch/mh8826/checkpoints checkpoints
mkdir -p $E/results
```
**Why:** the default paths in the scripts are repo-relative. The links keep the data and the
`.pt` files outside git.

## Step 2: Find the config that matches each checkpoint

The harness loads weights with `strict=True`, so each `.pt` needs the exact architecture config it
was trained with.
- `v5_model_1200` uses the 12-loop transformer, `config/v5/v5_mlir_rl_v1_paper.json`.
- `v5_3_set1_base_p_model_3450` is **unknown**: nothing in the repo records it.

Try the candidates. Each one runs in its own process, because the config is read at import time:
```bash
cd $REPO
for CK in $CK1 $CK2; do for CFG in config/v5/v5_mlir_rl_v1_paper.json \
    config/v5/v5_no_transformer_mlir_rl_v1_paper.json config/v5/v5_distributed.json \
    config/v5/v5_no_transformer.json; do
  python - "$CK" "$CFG" <<'PY' 2>/dev/null || echo "no   $(basename $1) $CFG"
import json, os, sys, tempfile, torch
ck, cfg = sys.argv[1], sys.argv[2]
c = json.load(open(cfg)); c.update(dask_node_count=0, results_dir=tempfile.mkdtemp())
p = tempfile.mktemp(suffix='.json'); json.dump(c, open(p, 'w'))
os.environ.update(CONFIG_FILE_PATH=p, DASK_NODES='0', DASK_TRAINING_ONLY='1')
from rl_autoschedular_v5.model import HiearchyModel
w = torch.load(ck, map_location='cpu', weights_only=True); w = w.get('model', w)
HiearchyModel().load_state_dict(w, strict=True)
print('YES ', os.path.basename(ck), cfg)
PY
done; done
```
Use the config that prints `YES` for each checkpoint below. Call them `CFG1` and `CFG2`.

If none prints `YES` for `CK2`, compare its keys and shapes with `CK1`'s:
```bash
python -c "import torch,sys; w=torch.load(sys.argv[1],map_location='cpu',weights_only=True); w=w.get('model',w); [print(k,tuple(v.shape)) for k,v in w.items()]" $CK2 | head -40
```
`summary_proj (64, 3491)` means 12 loops. Keys with `transformer` mean a transformer config.
Different shapes mean you need the training config of that run, which you should ask its author for.

## Step 3: Smoke test (one block, interactively, about 10–30 min)

```bash
salloc --partition=compute --constraint=bergamo --cpus-per-task=4 --mem=32G --time=02:00:00
cd $REPO && set -a && source .env && set +a
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular:$PYTHONPATH"
S="--tag smoke --per-stratum 1 --compositions 0 --ablation-per-class 0 --checkpoint $CK1 --config $CFG1"
python $E/run_blocks.py $S --list
python $E/run_blocks.py $S --prepare-only
ls $E/results/smoke/t1_warm/r0/manifests/ | head
python $E/block_experiment.py run $E/results/smoke/t1_warm/r0/manifests/<one_block>.json
```
**Check** in `results/manual_runs/<run>/summary.json`:
- all four methods have `status: ok`;
- `mlir_baseline` and `torch_eager` both show `correctness: passed`. This is the real MLIR-vs-Torch check of
  the translation;
- `mlir_rl` has a `schedule.json` that isn't empty.

Fix any problem here before spending a long job on it.

## Step 4: Choose and freeze the block set (shared by both checkpoints)

```bash
python $E/run_blocks.py --tag ckpt1200 --checkpoint $CK1 --config $CFG1 --list          # how many blocks?
python $E/run_blocks.py --tag ckpt1200 --checkpoint $CK1 --config $CFG1 --prepare-only
mkdir -p $E/results/ckpt3450
cp $E/results/ckpt1200/selection.json $E/results/ckpt3450/selection.json                # identical blocks
python $E/run_blocks.py --tag ckpt3450 --checkpoint $CK2 --config $CFG2 --prepare-only
```
**Why:**
- The selection is seeded (one block per model × class × size, plus composition triples) and saved to
  `selection.json`.
- Copying that file guarantees both checkpoints are measured on exactly the same blocks.
- Defaults match experiment 01: ≤ 32 MiB of inputs, ≤ 1e8 FLOPs, 10 warmup + 100 timed calls, 3
  correctness cases.

## Step 5: Submit the runs (Bergamo, one job per checkpoint)

```bash
cd $REPO
sbatch $E/run_blocks.sbatch --tag ckpt1200 --checkpoint $CK1 --config $CFG1 --threads 1 --repeat 0
sbatch $E/run_blocks.sbatch --tag ckpt3450 --checkpoint $CK2 --config $CFG2 --threads 1 --repeat 0
squeue -u $USER
tail -f $E/results/slurm_<JOBID>.log
```
**Why:**
- With no `--methods`, all four methods run: `mlir_baseline mlir_rl torch_eager torch_inductor`.
- Each method runs in its own fresh process pinned to 1 CPU, like experiment 01.
- Both jobs ask for the same node type (`bergamo`), so the absolute times are comparable across
  checkpoints.

Optional, for statistics with confidence intervals: add independent repeats (`--repeat 1`, `--repeat 2`)
and a multi-thread run:
```bash
sbatch --cpus-per-task=8 $E/run_blocks.sbatch --tag ckpt1200 --checkpoint $CK1 --config $CFG1 --threads 8 --repeat 0
```

**If a job stops** (time limit, node failure), submit the same command again. Blocks that already have a
`summary.json` are skipped; `--rerun` forces them to run again.

**To save compute on the second checkpoint:** only `mlir_rl` depends on it. You can run
`--methods mlir_baseline mlir_rl torch_eager` for `ckpt3450` and skip `torch_inductor`.
Keep `mlir_baseline` and `torch_eager` in the run, though: the harness validates a speedup only
against the baseline and eager results from the same run.

## Step 6: Collect, analyse, plot

```bash
for T in ckpt1200 ckpt3450; do
  python $E/run_blocks.py --tag $T --collect-only
  python $E/stats.py --all-results $E/results/$T/all_results.csv
  python $E/plot_blocks.py --tag $T --config t1_warm
  python experiments/01_performance_diagnosis/plot_speedups.py --results-dir $E/results/$T/t1_warm/r0
done
```
Per checkpoint, the main things to read are:
- `results/<tag>/analysis/headline_table.md`: geomean speedup with CI, win/tie/loss counts and Wilcoxon
  tests for RL vs baseline, RL vs eager and RL vs inductor;
- `results/<tag>/t1_warm/r0/<class>/REPORT.md`: experiment 01's own report, per block class;
- `results/<tag>/figures/`: the figures.

**Checkpoint A vs B:** both tags measure the same blocks. Compare their `per_block.csv` on `block`,
using the `mlir_rl` time or the speedup vs `mlir_baseline`.

---

## Before you trust the numbers

- **Only one comparison is new**: the RL schedule vs baseline / eager / inductor on **blocks**. Both
  checkpoints were trained on single ops, so these results measure generalisation.
- **3 VGG16 classifier blocks** may fail the harness's strict correctness check: these are float32 sums
  over K = 25088, which are correct within float32 (see README §4b). They then show no validated speedup.
- **Experiment 01's single-op suite (`run_suite.py`) hard-codes `checkpoints/v5_model_1200.pt`.** It
  can run checkpoint 1200 via the link from Step 1, but not checkpoint 3450 without editing
  experiment 01's code. The block experiment above takes `--checkpoint` / `--config` and has no such limit.
- **A run with no RL result** (`mlir_rl` error, for example a block with more than 12 loops or an
  `AstDumper` failure) is reported in `summary.json`. It is never counted as a speedup.
