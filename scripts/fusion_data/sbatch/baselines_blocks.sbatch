#!/bin/bash
#SBATCH --job-name=fd-base-blocks
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/fd_base_blocks_%j.out
#SBATCH --error=logs/fd_base_blocks_%j.err
# Row A (status quo): root_exec_time of all 8,093 files of data/ops_and_blocks, on the training node
# type (compute + bergamo, 64 cores). The repo had no ops_and_blocks baselines on this machine.
# Uses baseline_fullmodels.py (v5+ Execution API). Resumable: re-submit to continue, finished files
# are skipped. Then (login):
#   python scripts/data/split_json.py --input results/ops_and_blocks_results/baselines/mlir/base.json
# -> base_train.json / base_eval.json next to it (the paths in config/fusion/A.json).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
OUT=results/ops_and_blocks_results/baselines/mlir
mkdir -p "$OUT"
python scripts/baseline/baseline_fullmodels.py --models-dir data/ops_and_blocks \
    --pattern "*.mlir" --output "$OUT/base.json" --timeout 300
