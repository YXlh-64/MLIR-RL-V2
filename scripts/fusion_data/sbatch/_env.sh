# Sourced by the fusion_data sbatch scripts: the same environment as scripts/train/train.sh
# (.env, MLIR-ENV conda env, PYTHONPATH; no OMP/BLAS thread overrides), so baselines are
# timed under the same conditions as training. Run sbatch from the repo root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "$PROJECT_ROOT"
set -a; source .env; set +a
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/slurm/default/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$(dirname "$(command -v python)")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export AUTOSCHEDULER_IMPL=rl_autoschedular_v6
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-config/fusion/A_prime.json}"
source "$PROJECT_ROOT/scripts/fusion_data/sbatch/_threads.sh"   # explicit CPU split, never an inherited OMP=1
mkdir -p logs
echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} exec_workers=$EXEC_WORKERS omp_threads=$OMP_NUM_THREADS config=$CONFIG_FILE_PATH start=$(date)"
