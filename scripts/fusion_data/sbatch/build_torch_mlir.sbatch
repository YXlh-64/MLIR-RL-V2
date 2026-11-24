#!/bin/bash
#SBATCH --job-name=fd-build-torch-mlir
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fd_build_torch_mlir_%j.out
#SBATCH --error=logs/fd_build_torch_mlir_%j.err
# Part 2, option B: build torch-mlir against the agent's LLVM build. Needs TORCH_MLIR_COMMIT
# (from `setup_export_env.sh find-commit`) and the env from `setup_export_env.sh create-env`:
#   TORCH_MLIR_COMMIT=<sha> sbatch --export=ALL scripts/fusion_data/sbatch/build_torch_mlir.sbatch
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
set -a; source .env; set +a
bash scripts/fusion_data/setup_export_env.sh build
