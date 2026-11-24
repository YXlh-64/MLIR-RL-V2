# Sourced by the export sbatch scripts: the export env (torch + torch-mlir), NOT MLIR-ENV.
# EXPORT_ENV / TORCH_MLIR_DIR as in scripts/fusion_data/setup_export_env.sh.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the repo root}"
cd "$PROJECT_ROOT"
EXPORT_ENV=${EXPORT_ENV:-$HOME/.conda/envs/mlir-export}
TORCH_MLIR_DIR=${TORCH_MLIR_DIR:-/scratch/mh8826/torch-mlir}
TORCH_MLIR_PKG=$(ls -d "$TORCH_MLIR_DIR"/build/python_packages/torch_mlir "$TORCH_MLIR_DIR"/build/tools/torch-mlir/python_packages/torch_mlir 2>/dev/null | head -1 || true)
export PYTHONPATH="${TORCH_MLIR_PKG}${PYTHONPATH:+:$PYTHONPATH}"   # empty for option A (wheel)
export EXPORT_PY="$EXPORT_ENV/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# torch export / torch-mlir are CPU-threaded: use every allocated core, never an inherited OMP=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs
echo "node=$(hostname) python=$EXPORT_PY torch_mlir_pkg=${TORCH_MLIR_PKG:-wheel} start=$(date)"
