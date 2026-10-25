#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export CONFIG_FILE_PATH=config/v5/v5_fullmodel_relowered.json
export AUTOSCHEDULER_IMPL=rl_autoschedular_v5

MODEL="$1"
MODELS_DIR="${2:-data/full_models/relowered}"
OUTPUT_DIR="${3:-results/mlir_rl_v1_paper_results/baselines/fullmodels}"
TIMEOUT="${4:-7200}"

[[ "$MODELS_DIR" != /* ]] && MODELS_DIR="$PROJECT_ROOT/$MODELS_DIR"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

export MAX_EXEC_TIMEOUT=$TIMEOUT
export COMPILE_MIN_TIMEOUT=$TIMEOUT

OUT_FILE="$OUTPUT_DIR/baseline_${MODEL}.json"

echo "=== Running baseline for $MODEL on $(hostname) ==="
echo "Models dir: $MODELS_DIR"
echo "Output:     $OUT_FILE"
echo "Timeout:    ${TIMEOUT}s"

python scripts/baseline/baseline_fullmodels.py \
    --models-dir "$MODELS_DIR" \
    --pattern "${MODEL}.mlir" \
    --output "$OUT_FILE" \
    --timeout "$TIMEOUT"

echo "=== Done baseline for $MODEL ==="
