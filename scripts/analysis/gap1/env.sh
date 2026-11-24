# Source from the repo root: `source scripts/analysis/gap1/env.sh`
# Same setup as experiments/02_block_diagnosis/overnight/env.sh (conda env, MLIR bindings, .env).
type module >/dev/null 2>&1 || source /share/apps/NYUAD5/Modules/4.6.1/init/bash
module load gcc/12.2.0 binutils/2.46.0
set -a; source .env; set +a
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-config/v5/v5_mlir_rl_v1_paper.json}"
