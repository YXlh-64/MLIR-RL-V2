#!/bin/bash
#SBATCH --job-name=fd-step
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fd_step_%j.out
#SBATCH --error=logs/fd_step_%j.err
# A light pipeline step that must wait for a previous job (split, bench-json, split_json), run as a
# dependent job so run_pipeline.sh can chain everything: sbatch --dependency=afterok:<id> step.sbatch <cmd...>
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
echo "+ $*"
"$@"
