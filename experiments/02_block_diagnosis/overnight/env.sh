# Runtime environment for every overnight job (sourced after settings.sh, from REPO).
load_modules gcc/12.2.0 binutils/2.46.0                              # g++ for TorchInductor, libstdc++ for LLVM
set -a
source "$REPO/.env"
set +a
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$REPO:$REPO/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export CXX="$GCC_ROOT/bin/g++" CC="$GCC_ROOT/bin/gcc"
