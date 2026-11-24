#!/bin/bash
#SBATCH --job-name=fd-compat
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fd_compat_%j.out
#SBATCH --error=logs/fd_compat_%j.err
# Part 2.2: export resnet18, distilgpt2, deit_tiny (pretty + generic form) with the export env,
# then validate them with the v6 MLIR (bindings, AstDumper, features, no tm_tensor, static).
# Outputs: data/fusion_units/compat/, compat_generic/, compat/validation.json.
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_export_env.sh
"$EXPORT_PY" scripts/fusion_data/export.py --only resnet18,distilgpt2,deit_tiny --generic \
    --out data/fusion_units/compat --generic-out data/fusion_units/compat_generic --force
source scripts/fusion_data/sbatch/_env.sh
python scripts/fusion_data/validate.py --only resnet18,distilgpt2,deit_tiny \
    --raw-dir data/fusion_units/compat --generic-dir data/fusion_units/compat_generic \
    --out data/fusion_units/compat/validation.json
echo "tm_tensor lines per file:"; grep -c "tm_tensor\." data/fusion_units/compat/*.mlir || true
