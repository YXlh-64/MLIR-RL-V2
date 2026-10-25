#!/bin/bash
# Submit one Slurm job per full model, each evaluating all requested
# checkpoints sequentially via scripts/eval/eval_fullmodel.sh.
#
# Replaces the retired v4_5 ckpt_scan driver (see scripts/checkpoint/retired/).
# Cell outputs: <agent>/eval/checkpoint_<C>_<model>.json (via --label).
# Summarize with: python scripts/eval/merge_fullmodel_matrix.py --eval-dir <agent>/eval
#
# Usage:
#   bash scripts/eval/submit_fullmodel_matrix.sh --checkpoints "6400 6300"
#   bash scripts/eval/submit_fullmodel_matrix.sh --checkpoints "6400" --models "model_vgg resnet50_linalg"
#   bash scripts/eval/submit_fullmodel_matrix.sh --checkpoints "6400" --config config/v5/custom_fullmodel.json
#
# Monitor:
#   squeue -u $USER -n fm-matrix

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

CHECKPOINTS=""
ONLY_MODELS=""
CONFIG="$PROJECT_ROOT/config/v5/v5_mlir_rl_v1_paper_fullmodel.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints) CHECKPOINTS="$2"; shift 2;;
        --models) ONLY_MODELS="$2"; shift 2;;
        --config) CONFIG="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ "$CONFIG" != /* ]] && CONFIG="$PROJECT_ROOT/$CONFIG"
if [[ -z "$CHECKPOINTS" ]]; then
    echo "Usage: $0 --checkpoints \"6400 [6300 ...]\" [--models \"...\"] [--config ...]"
    exit 1
fi

EVAL_JSON="$(python3 -c "import json; print(json.load(open('$CONFIG'))['eval_json_file'])")"
[[ "$EVAL_JSON" != /* ]] && EVAL_JSON="$PROJECT_ROOT/$EVAL_JSON"
if [[ -z "$ONLY_MODELS" ]]; then
    ONLY_MODELS="$(python3 -c "import json; print(' '.join(json.load(open('$EVAL_JSON')).keys()))")"
fi

echo "Checkpoints: $CHECKPOINTS"
echo "Models: $ONLY_MODELS"
echo ""

for model in $ONLY_MODELS; do
    JOB_ID=$(sbatch --job-name="fm-matrix" \
        --partition=compute \
        --mem=64G \
        --cpus-per-task=64 \
        --time=1-00:00:00 \
        --output="$PROJECT_ROOT/logs/fm_matrix_${model}_%j.out" \
        --error="$PROJECT_ROOT/logs/fm_matrix_${model}_%j.err" \
        --wrap="
            cd $PROJECT_ROOT
            for ckpt in $CHECKPOINTS; do
                bash scripts/eval/eval_fullmodel.sh --checkpoint \$ckpt --models $model --label $model $CONFIG || echo \"CELL FAILED: ckpt=\$ckpt model=$model\"
            done
        " | awk '{print $4}')
    echo "  $model -> job $JOB_ID"
done

echo ""
echo "All jobs submitted. Summarize when done:"
echo "  python scripts/eval/merge_fullmodel_matrix.py --eval-dir results/mlir_rl_v1_paper_results/v5_fullmodel_agent/eval"
