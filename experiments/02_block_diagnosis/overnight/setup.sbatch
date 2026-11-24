#!/bin/bash
#SBATCH --job-name=ovn-setup
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --open-mode=append

# Overnight stage 1: everything the harness needs, then freeze the experiment inputs.
# Idempotent: each step is skipped if its output already exists, so resubmitting after a
# failure continues where it stopped. Log: experiments/02_block_diagnosis/overnight/logs/
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit through overnight.sh}"
source experiments/02_block_diagnosis/overnight/settings.sh

step() { printf '\n===== [%(%F %T)T] %s =====\n' -1 "$*"; }
load_modules gcc/12.2.0 binutils/2.46.0 cmake/3.28.0 ninja/1.13.0   # system ld 2.30 cannot link gcc 12 libgcc
export PATH="$CONDA_ENV/bin:$PATH"
PY="$CONDA_ENV/bin/python"
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK g++=$(which g++) cmake=$(cmake --version | head -1)"

step "1/8 Python packages (pybind11 2.11 is required by LLVM 19's bindings; matplotlib for plots)"
"$PY" -m pip install --quiet "pybind11==2.11.1" "PyYAML" "matplotlib" "pytz"   # pytz: imported by rl_autoschedular_v5
"$PY" -c "import pybind11, yaml, matplotlib, torch, numpy; print('python packages OK, torch', torch.__version__)"

step "2/8 LLVM 19 source"
if [ ! -d "$LLVM_SRC/.git" ]; then
    git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git "$LLVM_SRC"
fi

step "3/8 LLVM/MLIR build (mlir + clang + openmp + Python bindings; in-tree openmp needs the clang target)"
if [ ! -f "$LLVM_SRC/build/bin/mlir-opt" ] || [ ! -d "$LLVM_SRC/build/tools/mlir/python_packages/mlir_core/mlir" ]; then
    cmake -S "$LLVM_SRC/llvm" -B "$LLVM_SRC/build" -G Ninja \
        -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER="$(which gcc)" -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE="$PY" \
        -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_LINKER="$(which ld)" \
        -DLLVM_PARALLEL_LINK_JOBS=6
    ninja -C "$LLVM_SRC/build" -j "$SLURM_CPUS_PER_TASK"
fi
B="$LLVM_SRC/build"
for f in bin/mlir-opt bin/mlir-cpu-runner bin/mlir-translate lib/libmlir_c_runner_utils.so lib/libmlir_runner_utils.so; do
    [ -e "$B/$f" ] || { echo "MISSING after build: $B/$f"; exit 1; }
done
LIBOMP="$(find "$B" -name 'libomp.so' -not -path '*/CMakeFiles/*' | head -1)"
[ -n "$LIBOMP" ] || { echo "MISSING after build: libomp.so"; exit 1; }
echo "libomp: $LIBOMP"

step "4/8 C++ tools (AstDumper, PreVec, Vectorizer)"
for t in ast_dumper pre_vec vectorizer; do
    cmake -S "tools/$t" -B "tools/$t/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$(which gcc)" -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DLLVM_DIR="$B/lib/cmake/llvm" -DMLIR_DIR="$B/lib/cmake/mlir"
    ninja -C "tools/$t/build"
done
ls -l tools/ast_dumper/build/bin/AstDumper tools/pre_vec/build/bin/PreVec tools/vectorizer/build/bin/Vectorizer

step "5/8 .env and repo links"
if [ -f .env ] && ! grep -q 'written by overnight/setup.sbatch' .env; then
    cp .env ".env.backup.$(date +%s)"; echo "existing .env backed up"
fi
cat > .env <<EOF
# written by overnight/setup.sbatch
CONDA_ENV=$CONDA_ENV
LLVM_BUILD_PATH=$B
GCC_ROOT=$GCC_ROOT
MLIR_SHARED_LIBS=$LIBOMP,\${LLVM_BUILD_PATH}/lib/libmlir_c_runner_utils.so,\${LLVM_BUILD_PATH}/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=$REPO/tools/ast_dumper/build/bin/AstDumper
PRE_VEC_BIN_PATH=$REPO/tools/pre_vec/build/bin/PreVec
VECTORIZER_BIN_PATH=$REPO/tools/vectorizer/build/bin/Vectorizer
PATH=\${GCC_ROOT}/bin:\${LLVM_BUILD_PATH}/bin:\${PATH}
PYTHONPATH=\${LLVM_BUILD_PATH}/tools/mlir/python_packages/mlir_core\${PYTHONPATH:+:\${PYTHONPATH}}
LD_LIBRARY_PATH=\${LLVM_BUILD_PATH}/lib:$(dirname "$LIBOMP"):\${CONDA_ENV}/lib:\${GCC_ROOT}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF
cat .env
[ -e data/ops_and_blocks ] || ln -s "$DATA_DIR" data/ops_and_blocks
[ -e checkpoints ] || ln -s "$CKPT_DIR" checkpoints
ls -l data/ops_and_blocks checkpoints

step "6/8 Environment checks (MLIR bindings, TorchInductor, harness import)"
source experiments/02_block_diagnosis/overnight/env.sh
"$PY" -c "from mlir.ir import Context, Module; print('MLIR bindings OK')"
"$PY" -c "import torch; f=torch.compile(lambda x: x*2+1); print('TorchInductor OK', f(torch.ones(3)).tolist())"
"$PY" -c "import sys; sys.path.insert(0,'experiments/01_performance_diagnosis'); import experiment; print('harness import OK')"

step "7/8 Match each checkpoint to its architecture config (strict load, as the harness does)"
mkdir -p "$STATE"
: > "$STATE/checkpoints.tsv"
for CK in "${CHECKPOINTS[@]}"; do
    found=""
    for CFG in "${CANDIDATE_CONFIGS[@]}"; do
        if "$PY" - "$CK" "$CFG" <<'PY'
import json, os, sys, tempfile, torch
ck, cfg = sys.argv[1], sys.argv[2]
c = json.load(open(cfg)); c.update(dask_node_count=0, results_dir=tempfile.mkdtemp())
p = tempfile.mktemp(suffix='.json'); json.dump(c, open(p, 'w'))
os.environ.update(CONFIG_FILE_PATH=p, DASK_NODES='0', DASK_TRAINING_ONLY='1')
from rl_autoschedular_v5.model import HiearchyModel
w = torch.load(ck, map_location='cpu', weights_only=True); w = w.get('model', w)
HiearchyModel().load_state_dict(w, strict=True)
PY
        then found="$CFG"; break; fi 2>"$STATE/match_$(basename "$CK" .pt)__$(basename "$CFG" .json).err"
    done
    tag="ckpt_$(basename "$CK" .pt)"
    if [ -n "$found" ]; then
        printf '%s\t%s\t%s\n' "$tag" "$CK" "$found" >> "$STATE/checkpoints.tsv"
        echo "MATCH  $CK  ->  $found"
    else
        echo "NO CONFIG MATCHES $CK (its run jobs will skip; see RUN_CHECKPOINTS.md step 2)"
        "$PY" -c "import torch,sys; w=torch.load(sys.argv[1],map_location='cpu',weights_only=True); w=w.get('model',w); [print('   ',k,tuple(v.shape)) for k,v in list(w.items())[:25]]" "$CK"
    fi
done
cat "$STATE/checkpoints.tsv"

step "8/8 Freeze the block selection (identical for every checkpoint) and translate"
first=""
while IFS=$'\t' read -r tag ck cfg; do
    mkdir -p "$E/results/$tag"
    if [ -n "$first" ] && [ ! -f "$E/results/$tag/selection.json" ]; then
        cp "$E/results/$first/selection.json" "$E/results/$tag/selection.json"
    fi
    "$PY" "$E/run_blocks.py" --tag "$tag" --per-stratum "$PER_STRATUM" \
        --max-work "$MAX_WORK" --max-input-mib "$MAX_INPUT_MIB" --compositions 0 --ablation-per-class 0 \
        --checkpoint "$ck" --config "$cfg" --threads 1 --repeat 0 --prepare-only
    [ -n "$first" ] || first="$tag"
done < "$STATE/checkpoints.tsv"
touch "$STATE/setup.done"
step "setup finished"
