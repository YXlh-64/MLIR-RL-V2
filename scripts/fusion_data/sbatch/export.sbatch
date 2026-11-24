#!/bin/bash
#SBATCH --job-name=fd-export
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fd_export_%j.out
#SBATCH --error=logs/fd_export_%j.err
# Step 1: export every enabled model of data/fusion_units/models.json (idempotent; extra args are
# passed to export.py, e.g. --only a,b or --force). Writes data/fusion_units/raw{,_generic}/.
set -eo pipefail
source scripts/fusion_data/sbatch/_export_env.sh
"$EXPORT_PY" scripts/fusion_data/export.py --generic "$@"
