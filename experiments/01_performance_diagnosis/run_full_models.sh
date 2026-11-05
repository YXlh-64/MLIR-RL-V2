#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
    echo "Usage: bash experiments/01_performance_diagnosis/run_full_models.sh <model_stem> [options]"
    echo "Runs one model in the foreground. Example model: gin_linalg"
    exit 2
fi
model="$1"
shift
set -a
source .env
set +a
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
exec "$CONDA_ENV/bin/python" -u experiments/01_performance_diagnosis/run_suite.py \
    --dataset full_models --category "$model" "$@"
