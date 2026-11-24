#!/bin/bash
#SBATCH --job-name=fd-validate
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fd_validate_%j.out
#SBATCH --error=logs/fd_validate_%j.err
# Step 2: validate the exports with the v6 MLIR (writes data/fusion_units/validation.json).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
python scripts/fusion_data/validate.py "$@"
