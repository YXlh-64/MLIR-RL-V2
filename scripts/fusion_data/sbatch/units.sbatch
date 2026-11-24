#!/bin/bash
#SBATCH --job-name=fd-units
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fd_units_%j.out
#SBATCH --error=logs/fd_units_%j.err
# Step 4: closed units (fusion M3, leakage guard on) + sampling weights.
# Writes data/fusion_units/units/*.mlir, units/units_manifest.json, data/fusion_units/sampling_weights.json.
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
python scripts/fusion_data/build_units.py "$@"
