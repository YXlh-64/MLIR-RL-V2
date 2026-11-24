#!/bin/bash
#SBATCH --job-name=gap1
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/gap1/logs/%x_%j.log
# Usage (from the repo root): sbatch scripts/analysis/gap1/run.sbatch <script.py> [args...]
cd "$SLURM_SUBMIT_DIR"
source scripts/analysis/gap1/env.sh
python "scripts/analysis/gap1/$1" "${@:2}"
