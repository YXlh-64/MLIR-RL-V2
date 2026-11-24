#!/bin/bash
# Export environment for scripts/fusion_data/export.py (torch + torch-mlir), kept apart from MLIR-ENV.
#
# Recommended option B: build torch-mlir at the commit whose externals/llvm-project is the agent's
# LLVM (cd708029, 2025-01-14), out-of-tree against the existing LLVM build, so the exporter and the
# agent use the same MLIR. Nothing here has been run; run the steps in order after approval:
#
#   bash scripts/fusion_data/setup_export_env.sh find-commit     # login node (network): clone + list candidates
#   bash scripts/fusion_data/setup_export_env.sh create-env      # login node (network): conda env + pip
#   sbatch scripts/fusion_data/sbatch/build_torch_mlir.sbatch    # compute node: cmake + ninja (~30-60 min)
#   bash scripts/fusion_data/setup_export_env.sh check           # login node: imports work
#
# Option A (fallback, faster, approximate LLVM match): instead of build, install a torch-mlir nightly
# wheel dated at TORCH_MLIR_COMMIT's day and the torch nightly pinned in that commit's
# pytorch-requirements.txt:  bash setup_export_env.sh wheel <YYYYMMDD>
set -eo pipefail   # no -u: conda's shell scripts read unset variables

LLVM_COMMIT=cd708029e0b2869e80abe31ddb175f7c35361f90
LLVM_BUILD=${LLVM_BUILD_PATH:-/scratch/mh8826/llvm-project/build}
TORCH_MLIR_DIR=${TORCH_MLIR_DIR:-/scratch/mh8826/torch-mlir}
TORCH_MLIR_COMMIT=${TORCH_MLIR_COMMIT:-}          # set after find-commit
EXPORT_ENV=${EXPORT_ENV:-$HOME/.conda/envs/mlir-export}
PY=$EXPORT_ENV/bin/python

case "${1:-}" in
  find-commit)
    [[ -d $TORCH_MLIR_DIR ]] || git clone https://github.com/llvm/torch-mlir.git "$TORCH_MLIR_DIR"
    cd "$TORCH_MLIR_DIR"
    echo "torch-mlir commits that changed externals/llvm-project around 2025-01 (commit, date, llvm):"
    for c in $(git log --format=%H --since=2024-12-20 --until=2025-02-10 -- externals/llvm-project); do
      echo "$c $(git log -1 --format=%ci "$c" | cut -d' ' -f1) $(git ls-tree "$c" externals/llvm-project | awk '{print $3}')"   # %ci: old git has no %cs
    done | tee torch_mlir_llvm_bumps.txt
    grep "$LLVM_COMMIT" torch_mlir_llvm_bumps.txt \
      && echo "=> exact match above: export TORCH_MLIR_COMMIT=<first column>" \
      || echo "=> no exact match: pick the bump just after $LLVM_COMMIT (option B still builds against OUR llvm)"
    ;;
  create-env)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda create -y -p "$EXPORT_ENV" python=3.11
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0" "torchvision==0.21.0"
    "$PY" -m pip install "transformers==4.48.3" "timm==1.0.14" huggingface_hub numpy pybind11 nanobind \
        cmake ninja "PyYAML" packaging
    ;;
  build)   # run inside build_torch_mlir.sbatch (compute node)
    [[ -n $TORCH_MLIR_COMMIT ]] || { echo "set TORCH_MLIR_COMMIT (see find-commit)"; exit 1; }
    # One build at a time: two jobs would check out different commits into one tree and share build/.
    exec 9>"$TORCH_MLIR_DIR.build.lock"
    flock -n 9 || { echo "another torch-mlir build is running (lock $TORCH_MLIR_DIR.build.lock); wait for it"; exit 1; }
    cd "$TORCH_MLIR_DIR"
    git checkout "$TORCH_MLIR_COMMIT"
    export PATH="$EXPORT_ENV/bin:$PATH"
    cmake -GNinja -Bbuild \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE="$PY" \
      -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=OFF \
      -DTORCH_MLIR_ENABLE_STABLEHLO=OFF -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
      -DLLVM_TARGETS_TO_BUILD=host
    cmake --build build -j "${SLURM_CPUS_PER_TASK:-16}"
    echo "PYTHONPATH entry: $TORCH_MLIR_DIR/build/python_packages/torch_mlir (or build/tools/torch-mlir/python_packages/torch_mlir)"
    ;;
  wheel)   # option A
    DATE=${2:?usage: wheel YYYYMMDD}
    "$PY" -m pip install --pre "torch-mlir==${DATE}.*" \
        -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
    ;;
  check)
    PKG=$(ls -d "$TORCH_MLIR_DIR"/build/python_packages/torch_mlir "$TORCH_MLIR_DIR"/build/tools/torch-mlir/python_packages/torch_mlir 2>/dev/null | head -1)
    PYTHONPATH="${PKG:-}" "$PY" -c "import torch, torch_mlir; from torch_mlir.fx import export_and_import; print('torch', torch.__version__, 'torch_mlir OK', torch_mlir.__file__)"
    ;;
  *) echo "usage: $0 find-commit | create-env | build | wheel YYYYMMDD | check"; exit 1 ;;
esac
