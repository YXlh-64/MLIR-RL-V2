#!/bin/bash
# Submit parallel baseline timing jobs to Slurm compute partition.
# One job per model to isolate JIT execution and maximize parallelism.
#
# Usage:
#   bash scripts/baseline/submit_baselines.sh \
#       --models-dir data/full_models/relowered \
#       --models "bart_linalg vit_b_16_linalg yolov8m_linalg llama3_2_1b_linalg" \
#       --output-dir results/mlir_rl_v1_paper_results/baselines/fullmodels \
#       --timeout 7200

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

MODELS_DIR="data/full_models/relowered"
MODELS=""
OUTPUT_DIR="results/mlir_rl_v1_paper_results/baselines/fullmodels"
TIMEOUT=7200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2;;
        --models) MODELS="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --timeout) TIMEOUT="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

[[ "$MODELS_DIR" != /* ]] && MODELS_DIR="$PROJECT_ROOT/$MODELS_DIR"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/logs"

if [[ -z "$MODELS" ]]; then
    MODELS=$(cd "$MODELS_DIR" && ls *_linalg.mlir | sed 's/\.mlir$//')
fi

echo "=========================================="
echo "Submitting baseline timing jobs"
echo "Models dir:  $MODELS_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Timeout:     ${TIMEOUT}s"
echo "Models:      $MODELS"
echo "=========================================="

for model in $MODELS; do
    JOB_ID=$(sbatch --job-name="base-${model}" \
        --partition=compute \
        --mem=64G \
        --cpus-per-task=32 \
        --time=04:00:00 \
        --output="$PROJECT_ROOT/logs/baseline_${model}_%j.out" \
        --error="$PROJECT_ROOT/logs/baseline_${model}_%j.err" \
        "$SCRIPT_DIR/run_baseline_single.sh" "$model" "$MODELS_DIR" "$OUTPUT_DIR" "$TIMEOUT" | awk '{print $4}')
    echo "  $model -> Slurm job $JOB_ID"
done
