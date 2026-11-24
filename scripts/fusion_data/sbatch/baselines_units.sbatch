#!/bin/bash
#SBATCH --job-name=fd-base-units
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/fd_base_units_%j.out
#SBATCH --error=logs/fd_base_units_%j.err
# root_exec_time of every UNFUSED unit (R12), on the training node type (compute + bergamo, 64 cores:
# the single-node training allocation). Uses scripts/baseline/baseline_fullmodels.py (v5+ Execution API;
# the legacy get_base.py calls the pre-V5 string API and cannot be used). Resumable: finished units are
# skipped. Output: results/fusion_units_results/baselines/mlir/all_units.json ({unit: ns}, -1 = failed).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
OUT=results/fusion_units_results/baselines/mlir
mkdir -p "$OUT"
python scripts/baseline/baseline_fullmodels.py --models-dir data/fusion_units/units \
    --pattern "*_unit_*.mlir" --output "$OUT/all_units.json" --timeout 600
