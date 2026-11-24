#!/bin/bash
#SBATCH --job-name=fd-config-b
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/fd_config_b_%j.out
#SBATCH --error=logs/fd_config_b_%j.err
# Config B (design doc §6): the fusion layer alone, no agent. Prefuses every unit (M2) into
# data/fusion_units/units_prefused/ and times them like the unfused units (same tool, same node type).
# B speedup per unit = mlir/all_units.json[u] / mlir_prefused/B_units_prefused.json[u] (default lowering, R12).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
SRC=data/fusion_units/units
DST=data/fusion_units/units_prefused
# Separate folder: baseline_fullmodels.py caches times by benchmark NAME in <output dir>/.baseline_exec_cache.json,
# and prefused units keep the unfused names, so sharing the folder would return the unfused times.
OUT=results/fusion_units_results/baselines/mlir_prefused
mkdir -p "$DST" "$OUT"
for f in "$SRC"/*_unit_*.mlir; do
    [[ -f "$DST/$(basename "$f")" ]] || python -m rl_autoschedular_v6.fusion.prefuse "$f" "$DST/$(basename "$f")"
done
python scripts/baseline/baseline_fullmodels.py --models-dir "$DST" --pattern "*_unit_*.mlir" \
    --output "$OUT/B_units_prefused.json" --timeout 600
