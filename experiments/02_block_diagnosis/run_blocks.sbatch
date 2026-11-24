#!/bin/bash
#SBATCH --job-name=exp02-blocks
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/02_block_diagnosis/results/slurm_%j.log
#SBATCH --open-mode=append

# Same launcher as experiment 01's run_all.sbatch. Submit from the repository root
# after `mkdir -p experiments/02_block_diagnosis/results`. Options after this
# filename are forwarded to run_blocks.py, e.g.
#   sbatch experiments/02_block_diagnosis/run_blocks.sbatch --tag main --threads 1 --repeat 0
#   sbatch --cpus-per-task=8 experiments/02_block_diagnosis/run_blocks.sbatch --tag main --threads 8 --repeat 0
# --cpus-per-task must be >= --threads (experiment.py pins the first N allocated CPUs).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
set -a
source .env
set +a

export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''

printf 'Experiment 02: job=%s node=%s CPUs=%s memory_MiB=%s args=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK" "${SLURM_MEM_PER_NODE:-unknown}" "$*"

exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    "$CONDA_ENV/bin/python" -u experiments/02_block_diagnosis/run_blocks.py "$@"
