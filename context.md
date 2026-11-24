# CONTEXT.md: the MLIR-RL codebase and its environment

This file is background context about the code, the data and the cluster. The research background (theses, MDP formulations, open questions) is in `CLAUDE.md`.

Sources:
- **[local]**: read from the local checkout `/scratch/mh8826/MLIR-RL` on 6–7 Nov 2026. This is the authoritative source for what will actually run.
- **[repo]**: read from the public repo `Modern-Compilers-Lab/MLIR-RL`, branch `tayeb-mehdi/v4` (last commit 3 Jul 2026), on 6 Nov 2026. It differs from the local copy (§3).
- **[setup]**: from the user's own setup notes and sessions.
- **[HF]**: from the Hugging Face dataset page.

---

## 1. The user and the goal

- Maria is a Master's (PFE) student at ENSIA (Algiers) working in Dr. Riyadh Baghdadi's research line (NYUAD). She has a strong RL and PyTorch background, and is new to MLIR, LLVM builds and this codebase.
- **Goal:** build, baseline, train and evaluate the MLIR-RL agent on the NYUAD Jubail cluster, then extend it for her thesis.
- **Preferences:** rigorous, sourced explanations; she wants to understand *why*, not only *what*; unverified points should be stated as such.

## 2. The cluster (NYUAD Jubail, Slurm)

- **Login nodes** `login1`–`login4` are shared. CRC policy forbids heavy work on them (builds, training, evaluation, benchmark execution, large installs) as well as reverse tunnels and persistent services. Violations can get the account suspended.
- **Compute nodes** (`cn…`) are reached only through Slurm.
  - The user's interactive route is an SSH `ProxyCommand` that runs `salloc` (4 CPUs, 16 GB, 9 h). The allocation ends when the SSH session ends.
  - Inside that session `SLURM_JOB_ID` is set [local], so the Python entry points can also be run interactively there (§5).
  - Long work goes through `sbatch`.
- **Node types:**
  - Jubail: AMD EPYC 7742, 128 cores, 480 GB.
  - Bergamo: AMD EPYC 9754, 256 cores, ~1 TB; selected with `--constraint=bergamo`.
  - Dalma: Intel Xeon.
- **Slurm defaults:** walltime 5 h; 3.75 GB per CPU.
- **Partitions:**
  - `compute`: max 7 days.
  - `preempt`: starts sooner; default 2 h; jobs can be killed with a 30-min grace period.
  - `nvidia`; `bigmem` (> 480 GB).
- **Storage:**
  - `$HOME` (`/home/mh8826`): 50 GB / 500K files; jobs must not run from it.
  - `$SCRATCH` (`/scratch/mh8826`): 5 TB / 500K files; files not accessed for 90 days are deleted.
  - `$ARCHIVE`: not visible on compute nodes.
- **Network [setup]:**
  - Login nodes cannot reach github.com (the repo was cloned on the laptop and `rsync`ed).
  - Login nodes can reach `pypi.org` and `conda.anaconda.org`.
  - **Compute nodes can reach `huggingface.co`** (the dataset was downloaded from `cn145` on 7 Nov 2026). Whether they can reach github.com is still untested.
  - Unauthenticated HF downloads hit HTTP 429 rate limits (the dataset has ~9,450 small files). The CLI waits and retries; a read token (`hf auth login`) raises the limit.
- **Conda [setup]:**
  - The CRC miniconda is at `/share/apps/NYUAD5/miniconda/3-4.11.0` (conda 4.11, slow solver). `conda init` was run, so interactive shells start in `(base)`.
  - Batch jobs do not read `.bashrc`, so a job script must source `…/3-4.11.0/etc/profile.d/conda.sh` (or `…/bin/activate`) before `conda activate`.
- **GCC 14 libstdc++ [local]** exists at `/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64` (a second copy is under `spack/0.23.0/`).
- **Measurement methodology:** speedup = baseline time / optimized time. Baselines and evaluations are only comparable when measured on the same node type. The authors' `train.sh` and `eval.sh` request `--constraint=bergamo`.

## 3. Code identity: what the local checkout is

- **[local]** The local checkout is **not** a clone of `Modern-Compilers-Lab/MLIR-RL`. Its remote is `https://github.com/YXlh-64/MLIR-RL-V2.git`, branch `main`, with 3 commits:
  - `b4702e3` "Initial Commit: Tayeb & Mehdi Checkpoint" (YXlh, 25 Oct 2026);
  - `a8e8766` "Adding Setup Guide" (YXlh, 25 Oct 2026);
  - `c7c38e7` "first experiment: analysing where performance is dropped" (Aya BENALI KHODJA, 5 Nov 2026).
- **[local]** It is a snapshot of the authors' working tree (`/scratch/mb10856/MLIR-RL`), so it has more packages than the public `tayeb-mehdi/v4` branch. `rl_autoschedular/` contains:

  | Package | Encoder | Hardware vector | Shaped reward | Notes |
  |---|---|---|---|---|
  | `rl_autoschedular_paper` | LSTM | no | no | Port of the published artifact (Tirichine et al., arXiv:2409.11068); interchange mode `pointers` |
  | `rl_autoschedular_paper_transformer` | Transformer | no | no | Same as `paper`, except the encoder |
  | `rl_autoschedular_v0` … `v4`, `v2_5` | | | | Legacy versions |
  | `rl_autoschedular_v4_5` | Transformer | yes | yes | Process-isolated execution |
  | `rl_autoschedular_v4_9` | Transformer | yes | forced to 0 | Fix for the entropy collapse caused by shaped reward |
  | `rl_autoschedular_v45_no_hw`, `v45_no_shaped_reward`, `v45_no_transformer` | | | | Ablations of v4_5 |
  | `rl_autoschedular_v5` | Transformer | no | no | V5 platform: CPU-only, optional Dask-distributed collection (`distributed.py`), full-model eval (`full_model.py`) |
  | `rl_autoschedular_v5_no_transformer` | LSTM | no | no | V5 ablation |

  - `paper`, `paper_transformer`, `v5` and `v5_no_transformer` each ship their own `utils/` (Config, FileLogger, DaskManager, …). Packages never import from each other.
  - `rl_autoschedular_v5/README.md` is a stale copy of the `paper` README; ignore it.
- **[local]** **`rl_autoschedular_v5` exists locally**, with 14 configs in `config/v5/` (`v5_single_node.json`, `v5_single_node_smoke.json`, `v5_distributed.json`, `v5_mlir_rl_v1_paper.json`, full-model variants, …). `v5/model.py` contains the `TransformerEmbedding.summary_proj` layer.
- **[repo]** By contrast, no public branch contains `rl_autoschedular_v5`. Public `main` is the older single-package layout; other public branches are `v1.0`, `v2.0`, `v3.0`, `iql`, `nassim/aas`, `mohammed/*`, `kamel/*` (LLM-action and IQL experiments) and `v4/mlir-rl/action-synthesis` (Aug 2026; includes `iql/`, `llm_action/`, `train_ppo.py`).
- **[setup]** The user has the checkpoints `v5_model_1200.pt` and `v5_3_set1_base_p_*`. **They are not on the cluster yet** (no `.pt` file under `$HOME` or `$SCRATCH` as of 6 Nov).
  - `v5_model_1200.pt` should need `rl_autoschedular_v5` + `config/v5/v5_distributed.json` (7 loops); the expected key shape is `summary_proj.0.weight (64, 711)`. The local code has that layer, but the shape match has not been checked yet.
  - `v5_3_*` probably needs V5.3 code; locally V5.3 exists only as the design doc `docs/design/todo/v5_3_expanded_action_space.md`.
- **[local]** `AGENTS.md` describes the authors' own setup (`~/envs/mlir`, `rtk`, `graphify`, `.agents/skills/`). Those tools and paths do not exist for this user.

## 4. Code map [local]

| Path | Role |
|---|---|
| `scripts/train/train.sh` → `scripts/train/train.py` | Slurm wrapper and the generic PPO loop for any package. Usage: `sbatch scripts/train/train.sh <config> [impl] [--resume <results_dir>]`; `FORCE_NEW=1` overwrites. Default resources: `compute`, Bergamo, **8 CPUs, 8 GB**, 7 days; the script's comment says to use `--cpus-per-task=64 --mem=16G` for single-node runs |
| `scripts/eval/eval.sh` → `scripts/eval/eval.py` | Evaluates the checkpoints in `<results_dir>/models/` (64 CPUs, 16 GB, Bergamo). `--checkpoint N` evaluates one checkpoint (run id `ckpt_N`) and **forces** `EVAL_DIR=<repo>/<results_dir>/models`; without it, an existing `EVAL_DIR` is respected. Sets `DASK_TRAINING_ONLY=1`, so evaluation never starts Dask workers |
| `scripts/baseline/get_base.{sh,py}` | Times the unoptimized MLIR of each benchmark (`--timeout`, default 15 s; resumable; `--chunk-index` / `--num-chunks`). `get_base.py` does **not** call `load_dotenv`, so the environment must already be set |
| `scripts/data/split_json.py` | Stratified train/eval split by model family (`--eval-ratio 0.2 --seed 42` by default); drops entries ≤ 0; writes `<stem>_train.json` and `<stem>_eval.json` next to its input |
| `utils/config.py` | Root `Config` singleton, read from `$CONFIG_FILE_PATH` on first instantiation; keys not declared in the class are ignored |
| `utils/implementation.py` | Picks the package: `$AUTOSCHEDULER_IMPL`, else the config's `"implementation"`, else `rl_autoschedular`. The shell wrappers fall back to `rl_autoschedular_v0` instead |
| `utils/file_logger.py` | Writes directly into `results_dir`: `models/`, `logs/` (incl. `logs/exec_data.json`), `train_results.json` |
| `rl_autoschedular/<pkg>/` | `env`, `state`, `observation`, `actions/` (NT, T, TP, TF, I, V), `model` (`HiearchyModel`), `ppo`, `trajectory`, `execution`, `transforms`, `benchmarks` |
| `tools/{ast_dumper,pre_vec,vectorizer}/` | C++ tools built against the LLVM build (identical CMake files); binaries `AstDumper`, `PreVec`, `Vectorizer` |
| `config/ops_and_blocks/train/` | `paper_original`, `paper_transformer_{small,large}`, `v4_9_{small,large}`, `v0` |
| `config/ops_and_blocks/eval/` | The matching `*_eval.json`, plus `v5_distributed_final_eval`, `v5_no_transformer_final_eval` |
| `config/mlir_rl_v1_paper/eval/` | Full-model and v5 final-eval configs |
| `config/v5/` | Unified v5 configs (one JSON drives train and eval) |
| `data_utils/` | Dataset creation pipeline (model → MLIR → blocks) |
| `docs/` | Authors' docs: `hpc/HPC_SETUP.md`, `pipeline/{PIPELINE,COMMANDS,TRAINING_AND_EVALUATION,TRAINING_MANUAL}.md`, `design/{VERSIONS,CONFIG}.md`, `design/done/`, `design/todo/`, `onboarding/01–09`, `investigations/` |
| `documentation/` | Added in the V2 fork: `MLIR_LLVM_SETUP_V2.md` (another user's working install, §7), `CODE_WALKTHROUGH.md`, `REPOSITORY_STRUCTURE.md`, `V5_LEARNING_FROM_DOWNLOADED_DATA.md` |

**Execution path [local, `paper` package].**
1. Features are extracted by `AstDumper` (`state.py`), reading each benchmark file.
2. Transforms are applied through the Transform dialect, in a child `python -c` subprocess (`transforms.py`). The V action calls `PreVec` through `$PRE_VEC_BIN_PATH` (`transforms.py:276`), so `PreVec` must be built.
3. Code is lowered and run through `mlir.execution_engine` with `MLIR_SHARED_LIBS`, in a `multiprocessing.Process` child, as the median of 2 runs. Timeout: `min(300, max(MIN_EXEC_TIMEOUT, 5 × base time))` s.
4. If that fails, a CLI fallback runs `$LLVM_BUILD_PATH/bin/mlir-opt … | mlir-cpu-runner`.
   - **`mlir-cpu-runner` only exists in LLVM ≤ 19** (later releases renamed it), so LLVM 19 is required.
5. The lowering contains `convert-scf-to-openmp`, so `libomp.so` is required at runtime.

**Training loop details [local]:**
- `torch.set_num_threads(4)`.
- Checkpoints are saved every 50 iterations as `models/model_N.pt`, a dict with keys `model`, `optimizer`, `step`.
- A fresh run refuses to start if `<results_dir>/train_results.json` exists and is non-empty (unless `--resume` or `FORCE_NEW`).
- SIGABRT from native MLIR code is turned into a Python exception, and the iteration is skipped.
- `BindingsProcess` is off unless `ENABLE_BINDINGS_PROCESS=1` (it would fork, which corrupts MLIR state).
- `DaskManager`: off in `paper`. In `v5` it is **on whenever `dask_node_count > 0`** in the config (or `DASK_NODES` is set). `config/v5/v5_distributed.json` sets `dask_node_count: 16`, which submits 16 extra Slurm worker jobs. Use `v5_single_node*.json` for single-job runs.

**Known pitfalls [local]:**
- `Benchmarks` only catches `RuntimeError` (`benchmarks.py:53`), but an `AstDumper` failure raises a plain `Exception` (`state.py:303`). One listed benchmark that `AstDumper` cannot parse aborts data loading.
- `eval.py:107` sorts every `.pt` file in the models folder by `name.split('_')[1]`, so a file such as `v5_model_1200.pt` crashes evaluation. External checkpoints must be renamed `model_<N>.pt`.
- Train and eval configs of the same agent share `results_dir`, and every start resets `logs/exec_data.json` (the execution cache). Evaluating while training runs empties the training cache (slower, not incorrect).

## 5. Runtime environment variables [local]

| Variable | Used for |
|---|---|
| `LLVM_BUILD_PATH` | `mlir-opt`, `mlir-cpu-runner`, and the Python bindings at `$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core` |
| `MLIR_SHARED_LIBS` | Comma list: `libomp.so`, `libmlir_c_runner_utils.so`, `libmlir_runner_utils.so` |
| `AST_DUMPER_BIN_PATH`, `PRE_VEC_BIN_PATH`, `VECTORIZER_BIN_PATH` | The three tools. `paper` and `v5` use `AST_DUMPER` and `PRE_VEC`; older packages also use `VECTORIZER`. The authors' HPC doc never builds `pre_vec`, and their copy has no `tools/pre_vec/build` |
| `CONFIG_FILE_PATH` | The config JSON; the wrappers set it from their first argument |
| `CONDA_ENV` | `train.sh` and `eval.sh`: if it is a **directory**, they run `conda activate "$CONDA_ENV"`; otherwise they `source` it as a file (default `~/envs/mlir/bin/activate`). `get_base.sh` only does the `source`, so a directory fails there |
| `SLURM_JOB_ID` | `scripts/train/train.py` and `scripts/eval/eval.py` build the log filename from `os.environ['SLURM_JOB_ID']`, so they raise `KeyError` outside a Slurm job or `salloc` session |
| `PYTHONPATH` | The wrappers set `<mlir_core>:<repo root>:<repo root>/rl_autoschedular` |
| `MIN_EXEC_TIMEOUT` | Lower bound on the execution timeout (`eval.sh` sets 300) |
| `EVAL_DIR`, `EVAL_START`, `EVAL_END`, `EVAL_STRIDE` (default 100), `EVAL_LAST_ONLY`, `EVAL_FORCE` | Checkpoint selection in `eval.py` |
| `DISABLE_IMG2COL` | `1` disables the img2col rewrite of `conv_2d_*` benchmarks |

`.env` sits at the repo root (`.env.example` is the template; its paths point to `/scratch/YOUR_NETID/Code/MLIR-RL-V2` and must become `/scratch/mh8826/MLIR-RL`). The wrappers load it with `set -a; source .env`, and the Python entry points reload it with `load_dotenv(override=True)`.

## 6. Data

- **[HF]** Source: Hugging Face dataset **`Tay-eb/mlir-rl-dataset`** (https://huggingface.co/datasets/Tay-eb/mlir-rl-dataset), published by Tayeb Bessai (the 2026 thesis). License listed as "other"; 14.1 GB in total. Top-level layout:
  - `ops_and_blocks/`: about 8,093 `.mlir` files **directly in the folder** (e.g. `add_112_112_120_150.mlir`). Single ops plus multi-op blocks from 18 NN models; the primary training set.
  - `mlir_rl_v1_paper/`: about 1,357 `.mlir` files (2.6 GB). NN ops, LQCD kernels and 3 full models; the paper reproduction set.
  - `full_models/`: about 70 `.onnx`, `.onnx.data`, `_linalg.mlir` and `_torch.mlir` files. Large; only needed to re-run extraction.
  - No baseline or execution-time JSONs. The `.mlir` files include the `@nanoTime()` timing wrappers.
- **Layout [local]:** the configs set `benchmarks_folder_path: data/ops_and_blocks` (relative to the repo root), and `Benchmarks` opens `<benchmarks_folder_path>/<bench_name>.mlir`. The flat HF layout therefore matches without changes. (The `data/ops_and_blocks/all` path seen in the public repo is not used locally.)
- **Local copy [setup]:** the dataset is kept **outside the git repo**, at `/scratch/mh8826/data/{ops_and_blocks,mlir_rl_v1_paper}`. It was downloaded on 7 Nov 2026 with `hf download … --local-dir /scratch/mh8826/data`; `mlir_rl_v1_paper` is complete (1,357 files) and `ops_and_blocks` was still downloading.
  - It is planned to be linked into the repo with `ln -s /scratch/mh8826/data/<set> /scratch/mh8826/MLIR-RL/data/<set>`; `MLIR-RL/data/.gitignore` ignores everything in that folder except itself.
  - `/scratch/mh8826/data/data/` is a leftover of a failed first attempt and can be deleted. `/scratch/mh8826/data/.cache/` is HF resume metadata.
- **Authors' copy [local]:** `/scratch/mb10856/MLIR-RL/data/ops_and_blocks/` is world-readable (8,093 flat files). Their LLVM build, tool binaries and `results/` (baselines) are not readable.
- **File counts disagree [repo].** The public repo's `AGENTS.md` / `docs/RESULTS.md` say ~8,962 files and 8,280 timed / 8,240 valid / 6,553 train / 1,687 eval baselines. Those numbers are not in the local copy; the local `AGENTS.md` and the HF card say ~8,093.
- **Benchmark file format [local]:** `{tag = "operation_NNN"}` on linalg ops, an `@nanoTime()` wrapper, weights passed as function arguments, and `@main` returning `(tensor, i64)`. The 3 `model_*` files in `mlir_rl_v1_paper/` are exceptions, adapted in memory by `rl_autoschedular_v5/full_model.py`.
- **Baselines [local].**
  - The `ops_and_blocks` configs read `results/ops_and_blocks_results/baselines/mlir/base_train.json` (training) and `base_eval.json` (eval).
  - `get_base.py --config` writes to `<results_dir>/exec_times/base.json`, a different location. Passing `--output results/ops_and_blocks_results/baselines/mlir/base.json` and then `split_json.py --input` on that file produces exactly the two paths the configs expect.
  - Chunked runs write `base_chunk<i>.json`; the repo has no script to merge them.
  - The authors timed their `mlir_rl_v1_paper` baselines with `--implementation rl_autoschedular_v5` (`scripts/baseline/run_paper_original_baseline.sh`). Whether `paper` and `v5` lowering give the same timings has not been checked.
  - The HF dataset ships no baselines, so they must be measured.

## 7. Toolchain: decisions and an open conflict

- **[local]** No usable environment exists yet. `~/.conda/envs/MLIR-ENV` was found empty (only `conda-meta/`) on 6 Nov. By 7 Nov it had Python 3.11 and `huggingface_hub` installed (used for the data download). Whether to build LLVM against it or create a fresh env is undecided. `/scratch/mh8826/envs/` is empty.
- **[setup]** `pybind11` is pinned to 2.11, because 3.x broke the LLVM 19 bindings build. The Python packages are torch (CPU wheel), python-dotenv, tqdm, typeguard, dask-jobqueue, numpy, pyyaml (`requirements.txt` lists numpy, python-dotenv, torch, tqdm, dask-jobqueue, typeguard) and `huggingface_hub[cli]`.
- **Recipe A: gcc 13.2** (README, the user's original plan, `documentation/MLIR_LLVM_SETUP_V2.md`):
  - conda `gcc_linux-64` / `gxx_linux-64` 13.2, cmake 3.28.3, ninja;
  - LLVM `release/19.x` / `llvmorg-19.1.7` with `mlir;clang;openmp`, assertions and Python bindings (`-DPython3_EXECUTABLE`, `-Dpybind11_DIR`);
  - `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` during the build;
  - `libomp.so` ends up in `$LLVM_BUILD_PATH/lib`, which is what `.env.example` expects.
  - **[local]** `MLIR_LLVM_SETUP_V2.md` reports that another user (`ab13817`) built LLVM 19.1.7 with this recipe on this cluster and verified the bindings, the three tools and JIT execution. Their install is not readable from this account.
- **Recipe B: clang** (the authors' `docs/hpc/HPC_SETUP.md`, which says conda gcc 13.2 "did not work, faced recurring linking issues"):
  - conda-forge `clang clangxx lld`;
  - `-DLLVM_ENABLE_PROJECTS=mlir`;
  - `-DCMAKE_CXX_FLAGS="-include cstdint"` and `-DCMAKE_C_FLAGS="-include stdint.h"`, because clang 21 no longer transitively includes `<cstdint>` in LLVM 19 headers;
  - linking against the cluster's GCC 14 libstdc++ (§2) with `-Wl,-rpath`;
  - `libomp.so` taken from the conda env's `lib/`.
- **The choice between the two recipes is open.** The two sources contradict each other.
- **[repo]** A `llvm-project/build` compiled by another user is unusable: its CMake cache and Python symlinks point to that user's paths.
- **[local]** Wrappers with stale hard-coded paths:
  - `scripts/baseline/get_base.sh` and `run_paper_original_baseline.sh`: `#SBATCH --output/--error` point to `/scratch/mb10856/...`;
  - `get_base.sh` also adds `$HOME/envs/mlir/lib` to `LD_LIBRARY_PATH`;
  - `quick_commands.txt` uses `cd /scratch/mb10856/MLIR-RL`.
  - `train.sh` and `eval.sh` log to the relative `logs/` folder and are fine.

## 8. User-written job scripts [setup]

The scripts below were written in an earlier session but **are not in the local checkout** [local]. They would have to be recreated.

| Script | What it does | Known issue |
|---|---|---|
| `scripts/setup_env.sh` | 4 CPUs, 4 h. Creates `mlir-build`, pip-installs the packages, downloads the HF folders `ops_and_blocks/` and `mlir_rl_v1_paper/` into `data/` | Written for the gcc 13 recipe; the data now lives in `/scratch/mh8826/data` |
| `build_llvm.slurm` | 32 cores, 12 h. Builds LLVM 19 | Written for the gcc 13 recipe; contains `conda activate /scratch/mh8826/envs/mlir-build`, which does not exist |
| `scripts/baseline/baseline_job.sh` | Bergamo, 64 CPUs, 100 GB, 2 days. Runs `get_base.py`, then `split_json.py` | Same stale `conda activate` path |
| `scripts/eval/eval_v5_ckpt.sh` | Bergamo, 64 CPUs, 24 h. Evaluates a `rl_autoschedular_v5` checkpoint | Same stale `conda activate` path; the v5 code does exist locally (§3) |

## 9. Status as of 7 Nov 2026 [setup, local]

- **Working:**
  - SSH key login;
  - VS Code Remote-SSH, with the server installed in `$SCRATCH`;
  - interactive access to compute nodes;
  - `conda init` (`(base)` in every shell);
  - internet from compute nodes to huggingface.co.
- **In progress:**
  - the data download into `/scratch/mh8826/data` (`mlir_rl_v1_paper` complete, `ops_and_blocks` partial).
- **Not done:**
  - the build environment (decision pending: reuse `MLIR-ENV` or create a new one);
  - symlinks from `MLIR-RL/data/` to the dataset;
  - `.env`;
  - the LLVM build;
  - the three C++ tools;
  - baselines;
  - any training or evaluation run;
  - copying the v5 checkpoints to the cluster and inspecting them.
- **Resolved:** the local checkout is the `YXlh-64/MLIR-RL-V2` fork (`main`), and it contains `rl_autoschedular_v5`.
- **Intended first target:** `rl_autoschedular_paper` on `ops_and_blocks`, which reproduces the published agent before any extension.

## 10. References

- Local fork: https://github.com/YXlh-64/MLIR-RL-V2 (branch `main`)
- Upstream repo: https://github.com/Modern-Compilers-Lab/MLIR-RL (branch `tayeb-mehdi/v4`)
- Dataset: https://huggingface.co/datasets/Tay-eb/mlir-rl-dataset
- Paper: https://arxiv.org/abs/2409.11068
- CRC docs: https://crc-docs.abudhabi.nyu.edu
- MLIR Python bindings: https://mlir.llvm.org/docs/Bindings/Python/
