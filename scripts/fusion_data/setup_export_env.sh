#!/bin/bash
# Export environment for scripts/fusion_data/export.py (torch + torch-mlir), kept apart from MLIR-ENV.
#
# The agent's MLIR is LLVM 19.1.7 (llvm-project release/19.x @ cd708029, tag llvmorg-19.1.7): branched
# from mainline in late July 2024. torch-mlir must therefore be one written for mainline LLVM of mid-2024:
#   TORCH_MLIR_COMMIT=43506726853b35ae9c253aa1d1c61b76ad9b4c13  (2024-08-07, pins llvm d16b21b of 2024-06,
#   the last commit before torch-mlir's first LLVM bump after the 19.x branch point), with torch 2.5.0.
# It is built out-of-tree against the existing LLVM build, so the exporter uses the agent's MLIR.
#
#   bash scripts/fusion_data/setup_export_env.sh find-commit     # login node (network): clone + list LLVM bumps
#   bash scripts/fusion_data/setup_export_env.sh create-env      # login node (network): conda env + pip
#   TORCH_MLIR_COMMIT=<sha> sbatch --export=ALL scripts/fusion_data/sbatch/build_torch_mlir.sbatch
#   bash scripts/fusion_data/setup_export_env.sh check           # login node: imports work
#
# Each commit builds into its own folder build-<sha8>; a successful build records it in
# $TORCH_MLIR_DIR.active_build, which check and scripts/fusion_data/sbatch/_export_env.sh read.
# Option A (approximate LLVM match): bash setup_export_env.sh wheel <YYYYMMDD>
set -eo pipefail   # no -u: conda's shell scripts read unset variables

LLVM_COMMIT=cd708029e0b2869e80abe31ddb175f7c35361f90
LLVM_BUILD=${LLVM_BUILD_PATH:-/scratch/mh8826/llvm-project/build}
TORCH_MLIR_DIR=${TORCH_MLIR_DIR:-/scratch/mh8826/torch-mlir}
TORCH_MLIR_COMMIT=${TORCH_MLIR_COMMIT:-}          # see the header
EXPORT_ENV=${EXPORT_ENV:-$HOME/.conda/envs/mlir-export}
PY=$EXPORT_ENV/bin/python
ACTIVE_BUILD_FILE="$TORCH_MLIR_DIR.active_build"

torch_mlir_package() {   # python package folder of a build folder
    ls -d "$1"/python_packages/torch_mlir "$1"/tools/torch-mlir/python_packages/torch_mlir 2>/dev/null | head -1
}

case "${1:-}" in
  find-commit)
    [[ -d $TORCH_MLIR_DIR ]] || git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_DIR"
    cd "$TORCH_MLIR_DIR"
    echo "torch-mlir commits that changed externals/llvm-project in 2024 (commit, date, llvm):"
    for c in $(git log --format=%H --since=2024-05-01 --until=2024-12-31 -- externals/llvm-project); do
      echo "$c $(git log -1 --format=%ci "$c" | cut -d' ' -f1) $(git ls-tree "$c" externals/llvm-project | awk '{print $3}')"   # %ci: old git has no %cs
    done | tee torch_mlir_llvm_bumps.txt
    echo "=> LLVM 19.x branched ~2024-07-23: use the parent of the first bump after that date"
    ;;
  create-env)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -y -p "$EXPORT_ENV" python=3.11
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.0" "torchvision==0.20.0"
    "$PY" -m pip install "transformers==4.48.3" "timm==1.0.14" huggingface_hub numpy pybind11 nanobind \
        cmake ninja "PyYAML" packaging
    ;;
  build)   # run inside build_torch_mlir.sbatch (compute node)
    [[ -n $TORCH_MLIR_COMMIT ]] || { echo "set TORCH_MLIR_COMMIT (see the header)"; exit 1; }
    # One build at a time: two jobs would check out different commits into one tree.
    exec 9>"$TORCH_MLIR_DIR.build.lock"
    flock -n 9 || { echo "another torch-mlir build is running (lock $TORCH_MLIR_DIR.build.lock); wait for it"; exit 1; }
    cd "$TORCH_MLIR_DIR"
    git checkout "$TORCH_MLIR_COMMIT"
    BUILD_DIR="$TORCH_MLIR_DIR/build-${TORCH_MLIR_COMMIT:0:8}"
    export PATH="$EXPORT_ENV/bin:$PATH"
    cmake -GNinja -B"$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE="$PY" \
      -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=OFF \
      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
      -DLLVM_TARGETS_TO_BUILD=host
    cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-16}"
    echo "$BUILD_DIR" > "$ACTIVE_BUILD_FILE"
    echo "built $TORCH_MLIR_COMMIT in $BUILD_DIR; package: $(torch_mlir_package "$BUILD_DIR")"
    ;;
  wheel)   # option A
    DATE=${2:?usage: wheel YYYYMMDD}
    "$PY" -m pip install --pre "torch-mlir==${DATE}.*" \
        -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
    ;;
  check)
    PKG=$( [[ -f $ACTIVE_BUILD_FILE ]] && torch_mlir_package "$(cat "$ACTIVE_BUILD_FILE")" || true )
    PYTHONPATH="${PKG:-}" "$PY" -c "import torch, torch_mlir; from torch_mlir.fx import export_and_import; print('torch', torch.__version__, 'torch_mlir OK', torch_mlir.__file__)"
    ;;
  *) echo "usage: $0 find-commit | create-env | build | wheel YYYYMMDD | check"; exit 1 ;;
esac
