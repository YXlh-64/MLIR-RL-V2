# Runbook: evaluating `v5_model_1200` on blocks (Jubail)

Goal: measure how the v5 checkpoint (trained on single ops, `mlir_rl_v1_paper`) performs on the **multi-op blocks** of `ops_and_blocks`.
Your partner's experiment (`experiments/01_performance_diagnosis`) **excludes blocks**, so this uses the standard `eval.sh` pipeline instead.

Paths used below:

| What | Path |
|---|---|
| Repo | `/scratch/mh8826/MLIR-RL` |
| Data | `/scratch/mh8826/data/ops_and_blocks` (8093 files, 6693 are `*_block_*`) |
| Checkpoint | `/scratch/mh8826/checkpoints/v5_model_1200.pt` |
| LLVM (to build) | `/scratch/mh8826/llvm-project` |
| Conda env | `MLIR-ENV` |

> The checkpoint has `summary_proj (64, 3491)`, which is the **12-loop** architecture. Every config you use with it must keep the architecture keys of `config/v5/v5_mlir_rl_v1_paper.json`. `v5_distributed.json` (7 loops) will **not** load.

---

## Step 0: Get a compute node

```bash
salloc --partition=compute --constraint=bergamo --cpus-per-task=32 --mem=64G --time=12:00:00
```
**Why:** the LLVM build is heavy. Don't run it on the login node.

## Step 1: Python environment

```bash
conda activate MLIR-ENV
cd /scratch/mh8826/MLIR-RL
pip install -r requirements.txt          # torch, numpy, dask-jobqueue, python-dotenv, ...
```
**Why:** `eval.py` needs torch, and `MLIR-ENV` currently has none.

## Step 2: Build LLVM/MLIR 19 with Python bindings and OpenMP

```bash
module load gcc/12.2.0                    # real compiler; on Jubail gcc/13.2.0 is a container wrapper without g++
cd /scratch/mh8826
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
pip install -r llvm-project/mlir/python/requirements.txt   # pybind11/nanobind for the bindings
cmake -S llvm-project/llvm -B llvm-project/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE="$(which python)"
ninja -C llvm-project/build -j 32
```
**Why:**
- It has to be LLVM **19**: `mlir-cpu-runner` was removed after 19.
- The Python bindings are used for transforms and execution.
- `openmp` provides `libomp.so`, which the tiled-parallel action needs.
- Add `-DLLVM_ENABLE_LLD=ON` only if `which ld.lld` finds a linker.

## Step 3: Build the C++ tools

```bash
cd /scratch/mh8826/MLIR-RL
for t in ast_dumper pre_vec vectorizer; do
  cmake -S tools/$t -B tools/$t/build -G Ninja \
    -DMLIR_DIR=/scratch/mh8826/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/scratch/mh8826/llvm-project/build/lib/cmake/llvm
  ninja -C tools/$t/build
done
ls tools/*/build/bin/
```
**Why:**
- `AstDumper` extracts features from every block. Blocks are multi-op files, so it decides which ops get scheduled and how they connect.
- `PreVec` is called by the vectorization action.

## Step 4: `.env`

```bash
cp .env.example .env
sed -i 's|/home/YOUR_NETID/.conda/envs/mlir|'"$CONDA_PREFIX"'|; s|/scratch/YOUR_NETID/Code/MLIR-RL-V2/llvm-project|/scratch/mh8826/llvm-project|; s|/scratch/YOUR_NETID/Code/MLIR-RL-V2|/scratch/mh8826/MLIR-RL|g' .env
cat .env
```
**Why:**
- `eval.sh` sources `.env` to find the conda env, the LLVM libraries, the tool binaries and the Python bindings.
- `CONDA_ENV` must be the env **directory**, which is what `$CONDA_PREFIX` contains while `MLIR-ENV` is active.

Check that it works:
```bash
set -a; source .env; set +a
python -c "from mlir.ir import Context; import torch; print('OK', torch.__version__)"
```

## Step 5: Link the data into the repo

```bash
cd /scratch/mh8826/MLIR-RL
ln -s /scratch/mh8826/data/ops_and_blocks data/ops_and_blocks
mkdir -p /scratch/mh8826/data/blocks_only
find /scratch/mh8826/data/ops_and_blocks -maxdepth 1 -name '*_block_*.mlir' \
  -exec ln -s {} /scratch/mh8826/data/blocks_only/ \;
ls /scratch/mh8826/data/blocks_only | wc -l        # expect 6693
```
**Why:**
- Configs use repo-relative paths (`data/<dataset>`), while the data stays outside git.
- `blocks_only` gives the baseline script a folder containing **only blocks**. It has no name filter of its own, so it would otherwise time all 8093 files.

## Step 6: Blocks config (a new JSON; no code changes)

```bash
mkdir -p config/ops_and_blocks/eval
python - <<'PY'
import json
c = json.load(open('config/v5/v5_mlir_rl_v1_paper.json'))      # keeps the 12-loop architecture
base = 'results/ops_and_blocks_results/v5_blocks_ckpt1200/baselines/base_blocks.json'
c.update(benchmarks_folder_path='data/ops_and_blocks', json_file=base, eval_json_file=base,
         results_dir='results/ops_and_blocks_results/v5_blocks_ckpt1200',
         tags=['v5', 'blocks', 'ckpt1200'])
json.dump(c, open('config/ops_and_blocks/eval/v5_ckpt1200_blocks.json', 'w'), indent=2)
PY
```
**Why:**
- The eval reads the benchmark names from `eval_json_file`, loads `<benchmarks_folder_path>/<name>.mlir`, and computes speedup = baseline time ÷ optimized time.
- `json_file` points to the same file so that nothing tries to read the paper-set baselines.
- The checkpoint never saw these blocks, so the full set counts as held-out. You don't need `split_json.py`.

## Step 7: Baseline timings for the blocks (Bergamo, 8 parallel chunks)

```bash
mkdir -p logs
sbatch --partition=compute --constraint=bergamo --cpus-per-task=64 --mem=32G --time=24:00:00 \
  --array=0-7 --output=logs/base_blocks_%A_%a.out --wrap '
cd /scratch/mh8826/MLIR-RL && set -a && source .env && set +a
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
export CONFIG_FILE_PATH=config/ops_and_blocks/eval/v5_ckpt1200_blocks.json
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular:$PYTHONPATH"
python scripts/baseline/get_base.py --implementation rl_autoschedular_v5 \
  --benchmarks-dir /scratch/mh8826/data/blocks_only \
  --output results/ops_and_blocks_results/v5_blocks_ckpt1200/baselines/base_blocks.json \
  --chunk-index $SLURM_ARRAY_TASK_ID --num-chunks 8'
```
**Why:**
- The speedup needs the time of the *unoptimized* block, measured with the same v5 execution path and on the same node type (Bergamo) as the eval job.
- `get_base.sh` can't be used as-is: its log paths point to `/scratch/mb10856`, and it `source`s the conda directory.

Merge the chunks and drop failed runs once all 8 tasks have finished:
```bash
python - <<'PY'
import json, glob
d = 'results/ops_and_blocks_results/v5_blocks_ckpt1200/baselines/'
m = {}
for f in sorted(glob.glob(d + 'base_blocks_chunk*.json')): m.update(json.load(open(f)))
ok = {k: v for k, v in m.items() if isinstance(v, (int, float)) and v > 0}
json.dump(ok, open(d + 'base_blocks.json', 'w'), indent=2)
print(len(m), 'timed,', len(ok), 'kept')
PY
```
**Why:** blocks that fail or time out here (≤ 0) would also crash or skew the eval. Dropping them now is the cheap safety net.

> Quick test first (optional): use 1 chunk on a copy of `blocks_only` with ~50 links to check the whole chain end-to-end before spending ~6.7k files of compute.

## Step 8: Put the checkpoint where `eval.sh` expects it

```bash
M=results/ops_and_blocks_results/v5_blocks_ckpt1200/models
mkdir -p $M
ln -s /scratch/mh8826/checkpoints/v5_model_1200.pt $M/model_1200.pt
```
**Why:** with `--checkpoint N`, `eval.sh` looks in `<results_dir>/models/`, and `eval.py` only recognises files named `model_<N>.pt`.

## Step 9: Run the evaluation

```bash
sbatch scripts/eval/eval.sh config/ops_and_blocks/eval/v5_ckpt1200_blocks.json --checkpoint 1200
squeue -u $USER
tail -f logs/eval_<JOBID>.out
```
**Why:**
- `eval.sh` sets `DASK_TRAINING_ONLY=1`, so the eval runs on a single node with no Dask workers.
- It writes `results/ops_and_blocks_results/v5_blocks_ckpt1200/eval/checkpoint_1200.json` plus CSVs with the per-block speedups.

For a median of 5 runs (more robust numbers), use `scripts/eval/final_eval.sh` with a wrapper config like `config/ops_and_blocks/eval/v5_distributed_final_eval.json`. Its `#SBATCH` log paths point to `/scratch/mb10856`, so pass `--output/--error` on the `sbatch` command line.

---

## Before you interpret the numbers

- **Out of distribution.** The policy was trained on single ops. Read block results as *generalisation*, not as an in-domain score.
- **One bad block stops the job.** An `AstDumper` failure is not caught by `Benchmarks`, so a single file it can't parse stops the run. Step 7's filter catches most of these. If it still happens, remove that name from `base_blocks.json`.
- **Loop limit is 12.** A block with more loops than the model supports can't be encoded. That's unverified for this dataset; the Step 7 quick test will show it.
- **Not comparable to your partner's single-op numbers.** Her harness uses 1 pinned thread, its own O3 baseline and host-side timing. This pipeline uses multithreaded OpenMP and in-kernel `nanoTime`. Compare against the paper-style eval, not against her tables.
