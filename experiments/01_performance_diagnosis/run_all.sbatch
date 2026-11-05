#!/bin/bash
#SBATCH --job-name=exp01-all
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=experiments/01_performance_diagnosis/results/slurm_%j.log
#SBATCH --open-mode=append

# Submit from the repository root after creating the results directory.
# Options after this filename are forwarded to run_all.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
set -a
source .env
set +a

export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''

printf 'Experiment 01: job=%s node=%s CPUs=%s memory_MiB=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$SLURM_CPUS_PER_TASK" "${SLURM_MEM_PER_NODE:-unknown}"

exec srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    "$CONDA_ENV/bin/python" -u experiments/01_performance_diagnosis/run_all.py "$@"
