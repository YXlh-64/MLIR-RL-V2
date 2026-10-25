# MLIR Setup

## Local Setup

### New Installation (HPC)

#### 1. Install and Connect to Cisco VPN

```
vpn.abudhabi.nyu.edu
```

**Note:** you may need to enable an external VPN to connect to the NYU VPN.

---

#### 2. Generate SSH key

```bash
ssh-keygen
```

Check this link for more info: [Enabling Passwordless Authentication — CRC Documentation](https://crc-docs.abudhabi.nyu.edu/hpc/system/passwordless_auth.html#passwordless-ssh)

---

#### 3. Put this SSH config file in the local user folder ".ssh"

```
Host hpc
    HostName jubail.abudhabi.nyu.edu
    User <your NYU NetID>
    IdentityFile ~/.ssh/id_rsa

Host hpc-job
    ForwardAgent yes
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ProxyCommand ssh hpc "/opt/slurm/default/bin/salloc --nodes=1 -c 4 --mem=16G --time=9:00:00 /bin/bash -c 'nc $SLURM_NODELIST 22'"
    User <your NYU NetID>
    IdentityFile ~/.ssh/id_rsa
```

---

#### 4. Download the "Remote - SSH" extension on VSCode

---

#### 5. Launch a Remote Connection

Sometimes you will need to retry several times to connect.

---

#### 6. Conda Environment Setup

**Create Conda environment**

```bash
conda create -n mlir-build
conda activate mlir-build
```

**Install needed packages**

```bash
conda install -c conda-forge python=3.11 gxx_linux-64=13.2.0 gcc_linux-64=13.2.0 \
  cmake=3.28.3 ninja numpy pybind11=2.11 pyyaml
```

> ⚠️ **Updated:** pin `pybind11=2.11` explicitly. The latest `pybind11` (3.x) fails to
> compile against this LLVM branch (`static assertion failed: def_property family
> does not currently support keep_alive`). If you already ran the install without
> the pin and hit this error later during the build, fix it with:
> ```bash
> conda install -c conda-forge "pybind11=2.11" "pybind11-global=2.11"
> ```
> then re-run cmake (see step 7) and rebuild.

> ℹ️ If `conda install` fails with a `binutils_impl_linux-64` / `ld_impl_linux-64`
> version conflict, remove and recreate the environment from scratch
> (`conda env remove -n mlir-build`, then repeat the steps above) — this has
> been observed to resolve it in a clean state.

**Create `gcc` and `g++` symbolic links**

```bash
cd ~/.conda/envs/mlir-build/bin/
ln -s x86_64-conda-linux-gnu-gcc gcc
ln -s x86_64-conda-linux-gnu-g++ g++
ln -s x86_64-conda-linux-gnu-ar ar
```

**Verify the symbolic links and compilation**

```bash
# verify
which gcc
which g++
gcc --version
g++ --version

# test compilation
echo '#include <cstddef>' | g++ -x c++ -c - -o /dev/null
```

---

#### 7. LLVM Installation

**Clone MLIR-RL repo**

```bash
cd $SCRATCH
git clone https://github.com/Modern-Compilers-Lab/MLIR-RL.git
```

**Clone LLVM repo inside MLIR-RL**

```bash
cd MLIR-RL
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
```

**Build LLVM with MLIR and Python Bindings**

```bash
cd llvm-project
mkdir build
cd build

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
cmake --version

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

ninja
```

> ⚠️ **Updated (2 changes vs. the original guide):**
> 1. `LD_PRELOAD` now points to the generic symlink `libstdc++.so.6` instead of a
>    hardcoded version number (e.g. `libstdc++.so.6.0.34`). The exact patch version
>    shipped by conda-forge changes over time (`6.0.34`, `6.0.35`, ...), so hardcoding
>    it breaks with `ERROR: ld.so: object '...' cannot be preloaded`. Check yours with:
>    ```bash
>    ls -la $CONDA_PREFIX/lib/libstdc++*
>    ```
> 2. Added `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`. Without it, cmake fails to
>    generate build files entirely (`CMake Error: ... requires changing an RPATH
>    from the build tree, but this is not supported with the Ninja generator`).
>
> For a fresh terminal session later on, remember to re-export `LD_PRELOAD` before
> running any `cmake`/`ninja` command in the build directory — it doesn't persist
> across shells.

```bash
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:/scratch/<NetID>/MLIR-RL/llvm-project/mlir/python
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
cmake --build . --target check-mlir
```

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
cmake --build . --target check-mlir-python
```

> ℹ️ A single failing test out of ~2900 (`MLIR :: Examples/standalone/test.toy`)
> is a known harmless false positive tied to build-tool output formatting, not a
> real regression — safe to ignore if everything else passes.

**Add MLIR binaries and Python bindings to `PATH`**

Make sure to put your NetID in the path.

```bash
export PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/bin:$PATH
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
```

**Test MLIR and Python bindings**

```bash
mlir-opt --version
python -c "import mlir; print('MLIR Python bindings loaded successfully')"
```

```bash
python << 'EOF'
from mlir.ir import Context, Module
with Context() as ctx:
    module = Module.parse("""
    module {
      func.func @add(%arg0: i32, %arg1: i32) -> i32 {
        %0 = arith.addi %arg0, %arg1 : i32
        return %0 : i32
      }
    }
    """)
    print(module)
    print("\nMLIR working correctly!")
EOF
```

**To make these paths permanent, add to your `~/.bashrc`**

```bash
cat >> ~/.bashrc << 'EOF'
# MLIR paths (only when mlir-build env is active)
if [[ "$CONDA_DEFAULT_ENV" == "mlir-build" ]]; then
    export PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/bin:$PATH
    export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:/scratch/<NetID>/MLIR-RL/llvm-project/mlir/python:$PYTHONPATH
fi
EOF
source ~/.bashrc
conda activate mlir-build
```

> ⚠️ **Updated:** the `PYTHONPATH` line now includes both the compiled bindings
> path **and** the `mlir/python` source path (the original guide only listed the
> first one).

---

#### 8. Tools Installation

Within `MLIR-RL` repository execute:

```bash
cd $SCRATCH/MLIR-RL
PROJECT_ROOT=$(pwd)
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

**AST Dumper**

```bash
cd tools/ast_dumper
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/mlir \
  ..
cmake --build .
```

**Vectorizer**

```bash
cd ${PROJECT_ROOT}/tools/vectorizer
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/mlir \
  ..
cmake --build .
```

**Pre-Vectorizer** *(NEW — not present in the original guide, but required)*

```bash
cd ${PROJECT_ROOT}/tools/pre_vec
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/mlir \
  ..
cmake --build .
```

> ⚠️ **Updated:** the repo now has a third native tool, `tools/pre_vec/`, producing
> a binary called `PreVec`. It's referenced by the `PRE_VEC_BIN_PATH` environment
> variable (see step 11) and is required by the code — without it, block
> extraction / training will crash. It was missing from the original guide.

---

#### 9. Install Python Packages

Make sure your Conda environment is activated.

```bash
cd $SCRATCH/MLIR-RL
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

> ℹ️ `requirements.txt` covers the core training loop, but a few extra packages
> are needed depending on what you run next (data generation, baselines, etc.)
> and are **not** listed there yet:
> ```bash
> pip install pytz dill
> ```
> If you also plan to generate your own training data from real models
> (see the `data_utils/` pipeline), you'll additionally need:
> ```bash
> pip install torchvision onnx
> pip install torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
> ```
> (`torch-mlir` isn't on PyPI — it's distributed as nightly wheels from the LLVM
> GitHub releases. Double-check it doesn't try to downgrade your installed
> `torch` version before confirming.)

---

#### 10. ~~Create a Neptune Project~~ *(REMOVED — see note below)*

> ⚠️ **Removed.** The original guide asked to create an account on
> [neptune.ai](https://neptune.ai/) for experiment tracking. **This step is now
> obsolete for two reasons:**
> 1. Neptune.ai was **acquired by OpenAI** (announced Dec 3, 2025) and the
>    service is scheduled to shut down; **new account creation currently
>    appears to be disabled**, with the site redirecting to OpenAI's
>    announcement page.
> 2. More importantly: **the current MLIR-RL codebase no longer uses Neptune
>    at all.** Experiment logging is handled internally via `utils/file_logger.py`
>    (`FileLogger`), which writes locally — no external account or token needed.
>    The word "neptune" only survives in a couple of stale docstrings/comments.
>
> **Action: skip this step entirely.** Do not set `NEPTUNE_PROJECT` /
> `NEPTUNE_TOKEN` — they are not read by the code.

---

#### 11. Setup Environment Variables

Set `.env` (looking like the following):

```env
CONDA_ENV=/home/<NetID>/.conda/envs/mlir-build
LLVM_BUILD_PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build
MLIR_SHARED_LIBS=/scratch/<NetID>/MLIR-RL/llvm-project/build/lib/libomp.so,/scratch/<NetID>/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/<NetID>/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/<NetID>/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
PRE_VEC_BIN_PATH=/scratch/<NetID>/MLIR-RL/tools/pre_vec/build/bin/PreVec
VECTORIZER_BIN_PATH=/scratch/<NetID>/MLIR-RL/tools/vectorizer/build/bin/Vectorizer
```

> ⚠️ **Updated:** `NEPTUNE_PROJECT` / `NEPTUNE_TOKEN` removed (see step 10), and
> `PRE_VEC_BIN_PATH` added (see step 8). Always cross-check against the repo's
> own `.env.example` at the root of `MLIR-RL/`, since it reflects the current
> code more reliably than this guide.

**Note:** Make sure to put your NetID in the path.
