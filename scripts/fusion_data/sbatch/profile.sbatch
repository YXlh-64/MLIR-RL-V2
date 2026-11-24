#!/bin/bash
#SBATCH --job-name=fd-profile
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fd_profile_%j.out
#SBATCH --error=logs/fd_profile_%j.err
# Step 3: linalg profile of the validated models vs the runnable eval models, weighted comparison.
# Read data/fusion_units/profile/report/report.md (section 4b). The eval files are large (GBs).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
python scripts/fusion_data/linalg_profile.py "$@"
