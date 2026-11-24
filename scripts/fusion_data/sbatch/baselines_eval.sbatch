#!/bin/bash
#SBATCH --job-name=fd-base-eval
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/fd_base_eval_%j.out
#SBATCH --error=logs/fd_base_eval_%j.err
# root_exec_time of the runnable eval models (config/fusion/eval_set.json) on the training/eval node type,
# then the eval JSON the *_eval_fullmodel configs read. Models that fail (e.g. bert/bart/vit_b_16 raw
# files hit "op was not bufferized" in V5.2) get -1 and are left out of eval_fullmodel.json.
# Large files: llama3_2_1b is 19 GB of text (hence 256G); full-model JIT can take hours (timeout 4 h each).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
OUT=results/fusion_units_results/baselines/mlir
mkdir -p "$OUT"
python scripts/fusion_data/eval_set_tools.py link-runnable --full-models data/full_models --out data/fusion_units/eval_links
python scripts/baseline/baseline_fullmodels.py --models-dir data/fusion_units/eval_links \
    --pattern "*_linalg.mlir" --output "$OUT/eval_fullmodel_times.json" --timeout 14400
python - "$OUT/eval_fullmodel_times.json" "$OUT/eval_fullmodel_measured.json" <<'PY'
import json, sys
times = json.load(open(sys.argv[1]))
json.dump({k: v for k, v in times.items() if v and v > 0}, open(sys.argv[2], 'w'), indent=2)
PY
python scripts/fusion_data/eval_set_tools.py eval-json --times "$OUT/eval_fullmodel_measured.json" \
    --out "$OUT/eval_fullmodel.json" --allow-missing
