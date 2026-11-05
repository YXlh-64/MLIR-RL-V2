# LLVM and MLIR setup for MLIR-RL-V2

This is a standalone guide for the installation at
`/scratch/ab13817/Code/MLIR-RL-V2`. The environment is named **mlir**.
The original `MLIR_Setup.md` is preserved separately as the historical guide.

## 1. What you are setting up

The Python RL agent proposes a schedule: a sequence of compiler optimizations.
MLIR applies those transformations to a program. LLVM turns the lowered program
into executable machine code. Running that code supplies timings used to judge
the schedule. Installing PyTorch alone does not install this compiler toolchain.

| Component | Purpose | Actual location |
| --- | --- | --- |
| V2 repository | Agent, configuration, scripts, docs | `/scratch/ab13817/Code/MLIR-RL-V2` |
| Original repository | Source of the initial copy; not needed to run V2 | `/scratch/ab13817/Code/MLIR-RL-Official` |
| Conda environment | Python and installed dependencies | `/home/ab13817/.conda/envs/mlir` |
| LLVM source | Compiler framework and code generation | `/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/llvm` |
| MLIR source | IR, dialects, transformations and bindings | `/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/mlir` |
| Existing compiler build | Compiled tools and libraries, version 19.1.7 | `/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build` |
| MLIR Python package | Python interface to the native compiler | `/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build/tools/mlir/python_packages/mlir_core` |
| AstDumper | Extract operation/loop/dependency features | `/scratch/ab13817/Code/MLIR-RL-V2/tools/ast_dumper/build/bin/AstDumper` |
| PreVec | Prepare a selected operation for vectorization | `/scratch/ab13817/Code/MLIR-RL-V2/tools/pre_vec/build/bin/PreVec` |
| Vectorizer | Apply vectorization to a selected operation | `/scratch/ab13817/Code/MLIR-RL-V2/tools/vectorizer/build/bin/Vectorizer` |
| Local settings | Paths loaded into the process environment | `/scratch/ab13817/Code/MLIR-RL-V2/.env` |

The LLVM tree and helper builds were copied into V2, not linked to Official.
175 internal build links referring to the old checkout were repaired to point
within V2. Normal internal LLVM library/binding links remain. LLVM's nested
Git metadata was excluded. The `mlir` Conda environment is still shared by the
two projects: V2 is independent of Official's files, but package changes in
that environment can affect both. No dataset or results were copied.

## 2. HPC access: account setup, not repository setup

The earlier VPN, SSH and VS Code steps let your laptop reach the HPC and open
an editor there. An SSH key authenticates your account. The `hpc-job` SSH
configuration requests a compute allocation through Slurm before connecting.
The allocation has CPU, RAM and time limits; its resources also bound builds.

These steps are already working in this session. Opening a second repository
does not require new SSH keys, VPN configuration or another Conda environment.
Use an allocated compute node for expensive builds. Keep the interactive job
hosting the editor running.

## 3. Daily startup: use the existing installation

Run this in a terminal on the HPC:

```bash
cd /scratch/ab13817/Code/MLIR-RL-V2
conda activate mlir
set -a
source /scratch/ab13817/Code/MLIR-RL-V2/.env
set +a
export PYTHONPATH=/scratch/ab13817/Code/MLIR-RL-V2:/scratch/ab13817/Code/MLIR-RL-V2/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}
```

If `conda activate` is unavailable in a fresh non-interactive shell, initialize
Conda first, then repeat activation:

```bash
source /share/apps/NYUAD5/miniconda/3-4.11.0/etc/profile.d/conda.sh
conda activate /home/ab13817/.conda/envs/mlir
```

Activation selects this environment's Python and packages. `source .env` reads
settings into the current shell. `set -a` exports them so child Python and
compiler processes receive them. `PYTHONPATH` adds the repository and its
agent-package parent; the train/eval wrappers also add these automatically.
This startup does not compile or install anything and is safe to repeat.

## 4. Understand the local .env

The local file contains the following settings:

```bash
CONDA_ENV=/home/ab13817/.conda/envs/mlir
LLVM_BUILD_PATH=/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build
MLIR_SHARED_LIBS=${LLVM_BUILD_PATH}/lib/libomp.so,${LLVM_BUILD_PATH}/lib/libmlir_c_runner_utils.so,${LLVM_BUILD_PATH}/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/ab13817/Code/MLIR-RL-V2/tools/ast_dumper/build/bin/AstDumper
PRE_VEC_BIN_PATH=/scratch/ab13817/Code/MLIR-RL-V2/tools/pre_vec/build/bin/PreVec
VECTORIZER_BIN_PATH=/scratch/ab13817/Code/MLIR-RL-V2/tools/vectorizer/build/bin/Vectorizer
PATH=${LLVM_BUILD_PATH}/bin:${PATH}
PYTHONPATH=${LLVM_BUILD_PATH}/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}
LD_LIBRARY_PATH=${LLVM_BUILD_PATH}/lib:${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

| Setting | Why it exists |
| --- | --- |
| `CONDA_ENV` | Environment prefix accepted by the updated train/eval wrappers. |
| `LLVM_BUILD_PATH` | Root containing compiler `bin/`, libraries and generated bindings. |
| `PATH` | Makes `mlir-opt` and other compiler commands discoverable. |
| `PYTHONPATH` | Makes the built `mlir` package importable; it is not a regular pip dependency here. |
| `LD_LIBRARY_PATH` | Lets the dynamic loader find LLVM/MLIR and Conda runtime libraries. |
| `MLIR_SHARED_LIBS` | Explicit libraries given to the execution engine/runner: OpenMP and MLIR runtime utilities. |
| Helper paths | Tell the Python code which compiled C++ executable to launch. |

`LD_PRELOAD` is different: it forces a library to load first. The historical
build used Conda's `libstdc++.so.6` to resolve C++ runtime conflicts. The copied
installation passed runtime checks without setting it globally.

`.env.example` is a versioned template; `.env` is local and ignored by Git.
Neither installs dependencies. `CONFIG_FILE_PATH` is separate: it chooses an
experiment JSON, not a compiler installation.

## 5. Check that the installation works

After daily startup:

```bash
which python
python --version
which mlir-opt
mlir-opt --version
python -c "import numpy, torch, dotenv, typeguard, dask_jobqueue; print('Core Python dependencies loaded')"
python -c "from mlir.ir import Context, Module; from mlir.execution_engine import ExecutionEngine; print('MLIR bindings loaded')"
```

Expected Python: `/home/ab13817/.conda/envs/mlir/bin/python`.
Expected compiler: `/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build/bin/mlir-opt`.
Expected compiler version for the current copy: **19.1.7**.

A parser check:

```bash
python - <<'PYTHON'
from mlir.ir import Context, Module
from mlir.dialects import arith, func
with Context():
    module = Module.parse("""module {
      func.func @add(%a: i32, %b: i32) -> i32 {
        %0 = arith.addi %a, %b : i32
        return %0 : i32
      }
    }""")
    assert module.operation.verify()
    print(module)
PYTHON
```

The current installation was also checked with all three native helpers on a
tagged `linalg.add` and actual JIT execution (`2 + 3 = 5`) with the copied
runtime libraries. Those checks establish compiler functionality, not training
readiness or a complete compiler regression-test pass.

## 6. Python packages and optional model conversion

Core training dependencies are listed in
`/scratch/ab13817/Code/MLIR-RL-V2/requirements.txt`. They were import-checked
in the existing environment. If provisioning an environment that lacks them:

```bash
cd /scratch/ab13817/Code/MLIR-RL-V2
conda activate mlir
python -m pip install -r /scratch/ab13817/Code/MLIR-RL-V2/requirements.txt
```

Data conversion has separate requirements in
`/scratch/ab13817/Code/MLIR-RL-V2/data_utils/requirements.txt`.
It can require torchvision, transformers, ONNX, torch-mlir or GNN-specific
packages depending on the converter. These are not needed just to import MLIR
or use pre-existing benchmark files. torch-mlir must match the chosen PyTorch
and conversion workflow; the requirements file alone does not pin a compatible
combination. The helper
`scripts/setup/install_and_test_torch_mlir.sh` documents an existing workflow;
inspect its assumptions before running it.

`environment.yml` is an older export with another user's environment path and
compiler-package versions. Do not treat it as the specification of this copied
LLVM 19 build or blindly recreate it over the working `mlir` environment.

## 7. Optional: build LLVM/MLIR from source at V2's paths

**Skip this section to use the working copied installation.** Copied
`CMakeCache.txt` and Ninja files still record old build/source paths. Do not
rebuild in `llvm-project/build` or the copied helper `build/` directories.
Use the fresh `build-v2` paths below, keeping the existing copy intact.

### 7.1 Environment and sources

The environment already exists at `/home/ab13817/.conda/envs/mlir`. A new
installation would first create an environment named `mlir` with Python 3.11,
C/C++ compilers, CMake, Ninja, NumPy, PyYAML and pybind11 2.11. Do not recreate
or replace the existing environment for daily use. pybind11 2.11.1 is installed;
this LLVM branch was previously incompatible with pybind11 3.x.

For a fresh installation only, when no environment named `mlir` exists,
the corresponding provisioning command is:

```bash
conda create -n mlir -c conda-forge python=3.11 \
  gcc_linux-64=13.2.0 gxx_linux-64=13.2.0 \
  cmake=3.28.3 ninja numpy pyyaml pybind11=2.11
conda activate mlir
```

This follows the historical compiler recipe; it is not a description of every
package version currently installed. On this account the `mlir` environment
already exists, so use activation and skip creation. Install core Python
requirements from section 6 when provisioning a new environment.

The LLVM 19.1.7 sources are already under V2's `llvm-project/`. On a fresh
clone where that ignored folder is absent, obtain the matching source first:

```bash
cd /scratch/ab13817/Code/MLIR-RL-V2
# Only when llvm-project does not already exist:
git clone --depth 1 --branch llvmorg-19.1.7 https://github.com/llvm/llvm-project.git /scratch/ab13817/Code/MLIR-RL-V2/llvm-project
```

Conda's compiler executable names include the target platform. Explicit paths
below avoid needing `gcc`/`g++` symbolic-link aliases.

### 7.2 Configure a fresh build directory

```bash
conda activate mlir
export LD_PRELOAD=/home/ab13817/.conda/envs/mlir/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/ab13817/.conda/envs/mlir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake -S /scratch/ab13817/Code/MLIR-RL-V2/llvm-project/llvm \
  -B /scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2 -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=/home/ab13817/.conda/envs/mlir/bin/x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/home/ab13817/.conda/envs/mlir/bin/x86_64-conda-linux-gnu-g++ \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=/home/ab13817/.conda/envs/mlir/bin/python \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

CMake generates build instructions; it does not compile the full project.
`-S` is the source directory; `-B` is where generated output belongs.

| Option | Meaning |
| --- | --- |
| `LLVM_ENABLE_PROJECTS` | Enable MLIR, Clang (C/C++ frontend), and OpenMP support. |
| `LLVM_TARGETS_TO_BUILD=X86` | Generate code for the x86 target without building every backend. |
| `Release` | Optimize the compiler itself. |
| Assertions | Keep internal consistency checks enabled. |
| Examples/integration tests | Build additional examples and enable test coverage. |
| Python bindings/interpreter/pybind11 | Build the native Python API for the selected environment. |
| Install RPATH | Configure embedded library search paths; retained from the working build recipe. |

### 7.3 Compile, then validate

```bash
cmake --build /scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2 \
  --parallel "${SLURM_CPUS_PER_TASK:-4}"
```

Ninja now compiles and links the source. This is the expensive step; parallelism
must fit the allocation's RAM as well as CPU count. Four compiler processes
can still need substantial memory. After compilation, optional full checks:

```bash
export PYTHONPATH=/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2/tools/mlir/python_packages/mlir_core
export LD_LIBRARY_PATH=/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2/lib:/home/ab13817/.conda/envs/mlir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake --build /scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2 --target check-mlir
cmake --build /scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2 --target check-mlir-python
```

Inspect any failures rather than assuming a failing test is harmless.

### 7.4 Build the three helpers against that LLVM build

```bash
for tool in ast_dumper pre_vec vectorizer; do
  cmake -S "/scratch/ab13817/Code/MLIR-RL-V2/tools/$tool" \
    -B "/scratch/ab13817/Code/MLIR-RL-V2/tools/$tool/build-v2" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/home/ab13817/.conda/envs/mlir/bin/x86_64-conda-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=/home/ab13817/.conda/envs/mlir/bin/x86_64-conda-linux-gnu-g++ \
    -DLLVM_DIR=/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2/lib/cmake/llvm \
    -DMLIR_DIR=/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2/lib/cmake/mlir
  cmake --build "/scratch/ab13817/Code/MLIR-RL-V2/tools/$tool/build-v2" \
    --parallel "${SLURM_CPUS_PER_TASK:-4}"
done
```

Only after successful builds, edit V2's `.env`: change `LLVM_BUILD_PATH` to
`/scratch/ab13817/Code/MLIR-RL-V2/llvm-project/build-v2`, and the three helper
paths to their `build-v2/bin/` equivalents. `MLIR_SHARED_LIBS`, compiler PATH
and binding paths derive from `LLVM_BUILD_PATH`. Load the edited file in a
fresh terminal and repeat section 5. These optional rebuild commands are
instructions, not steps already executed during the copy.

## 8. Connect the installation to an experiment

For a small single-node configuration, inspect:
`/scratch/ab13817/Code/MLIR-RL-V2/config/v5/v5_single_node_smoke.json`.
The paper config `v5_mlir_rl_v1_paper.json` currently requests 16 Dask workers;
its filename does not mean it is a small local test.

Before training you need the chosen config's benchmark directory and baseline
timing JSONs, plus a registry entry in `experiments.json`. The current V2
`data/` and `results/` contain only ignore placeholders. Choose or generate the
data and timings; do not assume they were included in the compiler copy.

Once those inputs are prepared, submit from the V2 root:

```bash
cd /scratch/ab13817/Code/MLIR-RL-V2
# Example only, after data/baselines/registry are ready:
sbatch scripts/train/train.sh config/v5/v5_single_node_smoke.json
```

The train/eval wrappers load `.env`, activate its Conda prefix and set import
paths. They write Slurm output under V2's `logs/`. Other older shell scripts
can contain different environment assumptions; inspect them separately.
Metrics are written to local files; the standard train/eval flow does not
require a Neptune account. No training job was launched for this setup.

## 9. What Git stores

V2's root `.gitignore` excludes `llvm-project/`, native helper build folders,
`data/`, `results/`, `logs/`, `dask-logs/`, local environments, caches, weights
and `.env`. Source code, CMake files, configuration JSONs and `.env.example`
remain versioned. Ignoring preserves files on disk; it does not remove
already tracked files. A GitHub clone therefore does not include the heavy
local toolchain or dataset. Keep separate copies of irreplaceable data.

## 10. Common problems

| Symptom | Check |
| --- | --- |
| `mlir-opt: command not found` | Activate `mlir` and load V2's `.env`; check `which mlir-opt`. |
| `No module named mlir` | Check `PYTHONPATH` and the built binding directory; pip training requirements do not provide it. |
| `No module named rl_autoschedular_v5` | Add V2's `rl_autoschedular/` to `PYTHONPATH`, as in daily startup. |
| Missing `.so` or C++ runtime version | Check `LD_LIBRARY_PATH`, environment activation and library paths. |
| CMake refers to the old checkout | Use a fresh build directory; the copied caches are not relocatable build instructions. |
| Missing benchmarks/baseline JSON | Prepare experiment inputs; this is separate from installing the compiler. |
| Unexpected cluster jobs | Inspect `dask_node_count` and `DASK_NODES` before submission. |

For the code and folder map, read [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md).
