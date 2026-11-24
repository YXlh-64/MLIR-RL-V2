# Settings for the overnight pipeline. Edit here, nowhere else.
REPO=/scratch/mh8826/MLIR-RL
E=experiments/02_block_diagnosis                      # relative to REPO
CONDA_ENV=/home/mh8826/.conda/envs/MLIR-ENV           # torch 2.15 + numpy
GCC_ROOT=/share/apps/NYUAD5/gcc/12.2.0                # real compiler (gcc/13.2.0 here is not one)
LLVM_SRC=/scratch/mh8826/llvm-project                 # cloned + built by setup.sbatch
DATA_DIR=/scratch/mh8826/data/ops_and_blocks
CKPT_DIR=/scratch/mh8826/checkpoints

# Every .pt to evaluate. Each gets its own results tag: ckpt_<file name without .pt>.
CHECKPOINTS=(
    "$CKPT_DIR/v5_model_1200.pt"
)
# Tried in this order; the first one that loads with strict=True is used.
CANDIDATE_CONFIGS=(
    "$REPO/config/v5/v5_mlir_rl_v1_paper.json"
    "$REPO/config/v5/v5_no_transformer_mlir_rl_v1_paper.json"
    "$REPO/config/v5/v5_distributed.json"
    "$REPO/config/v5/v5_no_transformer.json"
)

# ALL blocks: no sampling and no size filter. run_blocks.py keeps one block per unique
# computation (1474 signatures cover all 6693 blocks; duplicates differ only in op tags) and
# map_duplicates.py attaches each result to every block with that signature.
PER_STRATUM=1000000
MAX_WORK=1e12                                          # largest block: 6.3e9 FLOPs
MAX_INPUT_MIB=4096                                     # largest block: 1337 MiB
SHARDS=48                                              # Slurm array tasks per checkpoint
MAX_PARALLEL_TASKS=24                                  # array throttle (%)
METHODS=(mlir_baseline mlir_rl torch_eager torch_inductor)

STATE=$REPO/$E/overnight/state                        # checkpoints.tsv, *.done markers
LOGS=$REPO/$E/overnight/logs

load_modules() {   # `module` may be undefined in a batch shell; its scripts break under `set -u`
    set +u
    type module >/dev/null 2>&1 || source /share/apps/NYUAD5/Modules/4.6.1/init/bash
    module load "$@"
}

# How the RL schedule is applied in the harness (see block_experiment.install_lenient_rl):
# lenient = like MLIR-RL's Env (failed action skipped with the rest of its op, recorded in
# methods/mlir_rl/rl_application.json); strict = experiment 01 (any failure = mlir_rl error).
RL_APPLY=lenient

# Retry stage: blocks whose latest run had a method time out are re-run with this limit (s).
RETRY_TIMEOUT=3600
RETRY_TASKS=30
