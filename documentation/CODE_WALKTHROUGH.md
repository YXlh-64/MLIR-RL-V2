# MLIR-RL: how the code fits together

Scope: how the modules interact, which package matches the last manuscript, and how much of the manuscripts' concepts the code contains. HPC/Slurm setup is left out on purpose.

Everything below was checked by reading the code in `MainGit/` (not only the docs). Where a repo doc and the code disagree, the code wins and I say so. Paths are relative to `code/`.

---

## 0. The short answers

**Which package is "the later one"?** There are two different answers, because the repo keeps every version side by side:

| Question | Answer |
|---|---|
| Newest code in the repo | `rl_autoschedular_v5` (design docs dated Aug to Sep 2026, after the thesis defence on 17 Jun 2026) |
| Package that matches the last manuscript (Benchikh and Bessai, 2026) | `rl_autoschedular_v4_5` (all three novelties on) and `rl_autoschedular_v4_9` (their final agent: Transformer + hardware vector, shaped reward switched off) |

The v5 package is **not** a descendant of v4.9. A file-by-file diff shows that `model.py`, `state.py`, `observation.py` and `trajectory.py` in v5 are identical to `rl_autoschedular_paper_transformer` (after renaming the package). v5 = paper_transformer + engineering (split rollout/execution, Dask hooks, full-model ABI adapter, no GPU occupier). The v4.9 design doc says the same thing: v5 "inherits the lesson" (no shaped reward, drop hardware features) but not the code.

**How much of the manuscripts is in the code?** Of the 40 concepts I tracked across the five manuscripts (table in section 6), v5 implements 20 fully and 11 partially; 8 are absent and 1 does not apply. The partial ones are mostly small deviations (tile-size rule, no constant column, timeout value). The big absences are the recursive AST/LSTM state, AlphaZero/MCTS, per-operation reward, the LQCD front-end, and the "future work" actions (unroll, pad, pack). The thesis' own contributions (Transformer with 17 tokens, hardware vector, shaped reward) exist in v4.5 exactly as described; v5 keeps only the Transformer.

---

## 1. Map of the repository

```
MainGit/
├── rl_autoschedular/        THE AGENTS. One folder per version, each a full copy of the code.
├── scripts/                 Entry points: train/train.py, eval/eval.py, baseline timing, data splitting, reports
├── utils/                   Shared helpers used by scripts and by the OLD packages (config, implementation router, logging)
├── config/                  JSON experiment configs (one JSON = one run)
├── data_utils/              Build the dataset: PyTorch model -> MLIR -> single ops / multi-op blocks
├── tools/                   Three C++ programs built against LLVM: AstDumper, PreVec, Vectorizer
├── dashboard/               Streamlit app that reads results/ and plots them
├── docs/                    Design notes per version, onboarding guides, manuscripts as markdown
├── tests/                   One unittest file (reporting helpers only, not the RL code)
├── data/                    EMPTY in this checkout (gitignored). No benchmark can be loaded from here.
└── results/ logs/ plots/    Outputs of past runs (also not needed to read the code)
```

Two things to know before reading anything:

1. **Packages are copy-pasted, not layered.** `rl_autoschedular_v0`, `_v1`, ..., `_v5` each contain their own `env.py`, `model.py`, `ppo.py`, etc. There is no shared base class. A scientific variant is a different folder. This is why the repo looks huge but each folder is small (about 3,500 lines).
2. **Which package runs is decided by one JSON field**, `"implementation": "rl_autoschedular_v5"`. [utils/implementation.py](MainGit/utils/implementation.py) reads it and `importlib`-imports the package. The scripts never import a package by name.

---

## 2. The family of packages

```
v0  (baseline: LSTM over [consumer, producer], no HW, no shaped reward)
 ├─ v1   + hardware vector in the observation
 ├─ v2   + shaped reward ──────────── v2.5 (hardened: process isolation, timeouts)
 ├─ v3   + Transformer encoder
 └─ v4   = v1 + v2 + v3 ── v4.5 (robust) ── v4.6/4.7/4.8 (same package, different configs)
                                          └─ v4.9 (shaped reward hard-wired to 0.0)   <-- last manuscript's final agent

paper               LSTM, interchange = pointers, process-isolated, own utils/
 └─ paper_transformer   same, with TransformerEmbedding
      └─ v5             = paper_transformer + rollout/execute split + Dask + full-model ABI   <-- newest code
v5_no_transformer       v5 platform with the LSTM encoder (ablation)
v45_no_hw, v45_no_shaped_reward, v45_no_transformer     ablations of v4.5
```

| Package | Encoder | Hardware vector | Shaped reward | Matches |
|---|---|---|---|---|
| `v0` | LSTM (2-step) | no | no | merged paper's agent (arXiv v2 / CGO), in its first form |
| `paper` | LSTM | no | no | same, with reliability fixes |
| `paper_transformer` | Transformer | no | no | paper + Benchikh and Bessai's encoder only |
| `v1`, `v2`, `v2_5`, `v3` | LSTM or Transformer | one novelty each | one novelty each | single-novelty ablations of the thesis |
| `v4`, `v4_5` | Transformer | yes | yes | **the thesis' full agent** (all three novelties) |
| `v4_9` | Transformer | yes | **off** | **the thesis' final agent** (shaped reward removed) |
| `v5` | Transformer | **no** | no | newest code |
| `v5_no_transformer` | LSTM | no | no | v5 ablation |

Practical consequences:
- Do not try to learn the repo from all 16 folders. Learn **one** (v5), then read the differences (section 2.1).
- `AGENTS.md` claims "each package is fully standalone, no cross-imports". That is true for imports *between packages*, but v0 to v4.9 all import `utils.config` from the **top-level** `utils/`, while `paper*` and `v5*` carry their own `utils/` folder with their own `Config` class. So the older packages share one config class; the newer ones each have their own.

### 2.1 What actually differs between versions (verified by reading)

| Change | Where it lives | Present in |
|---|---|---|
| Hardware vector (L1/L2/L3, cores, SIMD, MHz), read from `/sys` and `/proc/cpuinfo` at import, normalised, appended after the encoder | `observation.py` (`HardwareFeatures`, `_build_hardware_vector`) and `model.py` | v1, v4, v4_5, v4_9 |
| Shaped reward: static score `w_ai*log10(1+AI) + w_vec*vectorizable + w_par*parallel_ratio`, per-step delta, clipped, +0.2 bonus on V | `env.py` (`__shaped_reward`) | v2 to v4_5; in v4_9 the function is kept but returns `0.0` |
| Success-contingent zeroing (all shaped rewards set to 0 if the final execution fails) | `env.py` `apply_and_run_sequence` | v2_5, v4_5, v4_9 |
| Transformer encoder (17 tokens) | `model.py` (`TransformerEmbedding`) | v3 and later, `paper_transformer`, v5 |
| Transforms applied on `Module` objects instead of code strings | `transforms.py` | `paper*`, v5 (v0 passes `code: str`) |
| `rollout_benchmarks()` split from `collect_trajectory()`, Dask distributed collection, `full_model.py` | `ppo.py`, `distributed.py`, `full_model.py` | v5 only |

`rl_autoschedular_v4_9/evaluation.py` and `ablation.py` look like leftovers: they import a `config` module that does not exist in the folder and a class spelled `HierarchyModel` (the real class is `HiearchyModel`). Ignore them.

---

## 3. The architecture of v5 (the package to learn)

### 3.1 The MDP in the code's vocabulary

| RL object | In the code |
|---|---|
| Episode | one **benchmark** (a `.mlir` file holding one op or a block of up to 5 ops) |
| Sub-episode | one **operation** inside the benchmark. Operations are visited from the **last to the first** |
| State | `OperationState` ([state.py](MainGit/rl_autoschedular/rl_autoschedular_v5/state.py)): features of the current op, features of its (last) producer, and the transformation history |
| Observation | flat tensor built by [observation.py](MainGit/rl_autoschedular/rl_autoschedular_v5/observation.py) (2,689 numbers with the default sizes) |
| Action | a multi-discrete choice: which transformation, then its parameters ([actions/](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/)) |
| Transition | `Env.step`: updates the *features analytically*. **No MLIR is run during the rollout** |
| Reward | zero everywhere except the last transition of the episode: `log10(T_base / T_opt)`, or -5 / -20 on failure |
| Discount | `gamma = 1.0` in both returns and GAE (the defaults in [trajectory.py:300](MainGit/rl_autoschedular/rl_autoschedular_v5/trajectory.py#L300), never overridden); `lambda = 0.95` |
| Algorithm | PPO, clipped, separate policy and value networks, Adam |

The key design idea: **the agent never sees real MLIR during sampling.** The environment keeps a numerical description of the loop nest (bounds, access matrices) and edits it by rule after each action. Only after the whole schedule is chosen is the schedule replayed on the real MLIR module and timed. That is why the reward is sparse and why one rollout of 64 benchmarks is cheap while the execution step is the bottleneck.

### 3.2 One training iteration, step by step

```
scripts/train/train.py   (the real entry point; imports the package named in the config)
 │
 │  for step in range(nb_iterations):
 ▼
ppo.collect_trajectory(data, model, step)
 │
 ├─ 1. pick bench_count (=64) random benchmarks
 │
 ├─ 2. ppo.rollout_benchmarks()                       <- sampling only, no MLIR
 │      for each benchmark: Env.reset() -> OperationState of the LAST op
 │      loop until every benchmark is terminal:
 │         obs = Observation.from_state(state)          observation.py
 │         action_idx = model.sample(obs)               model.py + actions/__init__.py
 │         action = ActionSpace.action_by_index(...)    actions/__init__.py
 │         state = Env.step(state, action)              env.py  (analytic feature update)
 │         if state.terminal: Env.get_next_op_state()   move to the previous op, keep history
 │         TrajectoryCollector.append(...)              reward = 0.0 for now
 │
 ├─ 3. execute the schedules, in parallel (DaskManager.map_objs)
 │      ppo.__execute_states(state):
 │         Env.reset(same benchmark)
 │         Env.apply_and_run_sequence(state.transformation_history)
 │            ├─ __apply_sequence: for each action -> action.apply(module)
 │            │        └─ transforms.py: build a Transform-dialect script, run it in a Python subprocess
 │            └─ Execution.execute_code(module)           execution.py
 │                     ├─ cache hit? (keyed by the schedule string)  -> return
 │                     └─ child process: lower to LLVM, JIT, run, read the nanoTime delta
 │         reward = log10(T_root / T_new)  (or -20)  -> put on the LAST transition
 │
 ├─ 4. write rewards into the collectors, update the execution cache (logs/exec_data.json)
 └─ return TrajectoryData
 
ppo.ppo_update(trajectory, model, optimizer)
 ├─ trajectory.update_attributes(model): values, old log-probs, returns, GAE advantages   trajectory.py
 └─ 4 epochs of minibatch (64) PPO with entropy bonus + value loss                        ppo.py, model.py

every 50 (scripts) or 100 (package train.py) steps: torch.save(model + optimizer)
```

### 3.3 Module dependency map (v5)

```
                         utils/config.py (Config singleton, reads CONFIG_FILE_PATH)
                                   │  read by almost everything
        ┌──────────────────────────┼───────────────────────────────────────────┐
        │                          │                                           │
   benchmarks.py ──► state.py ◄── actions/* ──► transforms.py ──► (MLIR Transform dialect)
        │   │             ▲           ▲                                   ▲
        │   │             │           │                                   │
        │   └─► full_model.py        observation.py ──► model.py          execution.py ──► (MLIR ExecutionEngine)
        │                             ▲                   ▲                  ▲
        └────────────────────────►  env.py ◄────────────────┘ (uses actions, state)   │
                                      ▲                                       │
                                      │                                       │
                                    ppo.py ───────────────────────────────────┘
                                   ▲   │      ▲
                    trajectory.py ─┘   │      └─ distributed.py (v5.1: same rollout, run on Dask workers)
                                       ▼
                     scripts/train/train.py   scripts/eval/eval.py
```

Reading it top to bottom: `state.py` has no internal dependencies and everything depends on it. `actions/` and `observation.py` depend on each other lazily (an import inside functions in `actions/__init__.py` avoids the circular import). `model.py` depends on `actions` (to build the heads) and `observation` (to slice the input). `env.py` is the only module that touches both the symbolic world (`state`, `actions`) and the real one (`Execution`, MLIR `Module`).

---

## 4. Module by module (v5)

### 4.1 [state.py](MainGit/rl_autoschedular/rl_autoschedular_v5/state.py): the data model

Four dataclasses, no logic beyond copying:

- `NestedLoopFeatures`: one loop (`arg`, bounds, step, `parallel`/`reduction`).
- `OperationFeatures`: one Linalg op: type, op counts, `load_data` / `store_data` (access expressions as strings like `"%d0 + %d2 * 3"`), nested loops, `producers` / `consumers` as `(tag, operand_index)` lists, a `vectorizable` flag, and `pre_actions` (transformations already applied before the agent started, e.g. the implicit tiling a fused producer receives).
- `BenchmarkFeatures`: the whole file: its MLIR text, ordered `operation_tags`, a dict of `OperationFeatures`, and `root_exec_time` (untransformed time in ns).
- `OperationState`: what the agent sees at one step. Its `transformation_history` is a **list of lists** (one list per operation visited so far, current op first). `record_action` handles the multi-step interchange (see 4.3).

It also holds the **feature extractor**: it runs the C++ `AstDumper` binary on the `.mlir` file and parses its text output (`#START_OPERATION`, `#START_NESTED_LOOPS`, ..., `#BEGIN_GRAPH`) into these dataclasses. Loops beyond `max_num_loops` make the op unsupported (it is skipped, with an error message). Loads beyond `max_num_stores_loads` are silently truncated.

### 4.2 [benchmarks.py](MainGit/rl_autoschedular/rl_autoschedular_v5/benchmarks.py): the dataset loader

`Benchmarks.__init__` reads `json_file` (a dict `{bench_name: root_exec_time_ns}`; the baseline times come from `scripts/baseline/get_base.py`), and for each name runs the AstDumper on `<benchmarks_folder_path>/<name>.mlir`. Two special cases:
- names starting with `conv_2d_` are rewritten with `img2col` (convolution becomes a matmul) before features are extracted;
- full-model files are patched in memory by [full_model.py](MainGit/rl_autoschedular/rl_autoschedular_v5/full_model.py) so they have the timed `@main` that the rest of the pipeline expects, and their elided weights are replaced by zero constants (fine for timing).

### 4.3 [actions/](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/): the action space (the most important folder)

Six actions, in this order (the order is the index the network uses):

| Symbol | Class | Parameters | Effect |
|---|---|---|---|
| `NT` | `NoTransformation` | none | terminal, ends the op |
| `T` | `Tiling` | one tile-size index per loop | `tile_using_for` |
| `TP` | `TiledParallelization` (subclass of `Tiling`) | same | parallel (non-reduction) loops tiled with `tile_using_forall`; reduction loops with `tile_using_for` |
| `TPF` | `TiledFusion` (subclass of `TP`) | same, plus the producer | tile the consumer, then fuse the producer into the tiled loop (this is the manuscripts' "TF") |
| `I` | `Interchange` | a loop index (pointers mode) | reorder loops |
| `V` | `Vectorization` | none | terminal; pre-processing (transpose/decompose conv, `PreVec` tool) then `vectorize` |

Each action class implements the same contract (see [base.py](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/base.py)): `params_size`, `network_output_size`, `action_mask`, `action_history`, `distribution`, `sample`, `distribution_stats`, `apply(module)` (real MLIR), `update_features(op_features)` (symbolic edit). `ActionSpace` in `__init__.py` glues them together.

**Hierarchical sampling** ([actions/__init__.py](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/__init__.py)). The policy outputs one categorical for "which action" and one distribution per action that has parameters. `sample()` draws the action, then draws parameters for all heads and keeps those of the chosen action. The log-probability is `log P(a) + log P(params | a)` (the `distributions_stats` method adds the parameter term only where `selection_index == i`). This is the multi-discrete factorisation `pi(o, x | s) = P(o|s) * prod P(x_i | s, o)` from the Ameur and Tirichine thesis and the paper.

**Masks.** `ActionSpace.action_mask(state)` builds a boolean vector in two parts: which actions are allowed, then per-action parameter masks (for tiling, only tile sizes that divide the loop bound; for interchange, loops not yet chosen). Invalid logits are set to `-inf` before the `Categorical`. The mask is **appended to the observation**, so the model sees it.

**The `order` config field** is the schedule template. The v5 configs use:

```
step 0: ["I"]               interchange is the only allowed action
step 1: ["!", "I", "NT"]    anything except interchange and "no transformation"
step 2: ["!", "I"]          anything except interchange
step 3: ["V", "NT"]         only the two terminal actions
```

So a schedule for one op is at most `I -> (T|TP|TPF|V) -> (...) -> (V|NT)`. `truncate: 5` is never reached because `order` has four entries. This is the thesis' "order" mechanism (Benchikh and Bessai p. 49-50) in code.

**Tile sizes.** Parameter `j` means tile size `0` (no tiling) for `j = 0`, or `2^(j-1)` (1, 2, 4, ..., 64). Only powers of two that divide the loop bound and are smaller than it are unmasked ([tiling.py:131](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/tiling.py#L131)). The manuscripts say "divisors of the bound" (Ameur and Tirichine) or a fixed list (Bendib); the code uses this narrower power-of-two rule.

**Interchange in pointers mode** builds the permutation one position per environment step. The partial action is stored with `ready = False`; the next call to `record_action` *replaces* it and keeps the earlier pieces in `sub_actions`. This is why rewards are allocated as `len(sub_actions) + 1` per action: one transition per pointer choice. `enumerate` (3n-6 swaps at distance 1, 2, 3) and `continuous` (factoradic/Lehmer code) are also implemented but not used by the configs.

**Fusion** (`tiled_fusion.py`) is the most intricate file. After a fusion, the benchmark features are edited in place: the original producer is removed (or kept if it has other consumers), a renamed copy `<producer>_<consumer>` is inserted before the consumer, and it gets an implicit tiling as a `pre_action` computed from the consumer's access pattern. The agent will later visit this renamed op. Only the **last** producer in the list and only a single use are supported (`# NOTE: To change with multiple producers support`).

### 4.4 [observation.py](MainGit/rl_autoschedular/rl_autoschedular_v5/observation.py): turning state into numbers

`Observation` is a list of parts, concatenated into one vector (sizes with `L = D = LS = 7`, pointers mode):

| Part | Size | Content |
|---|---|---|
| `OpFeatures` (consumer) | 711 | op-type one-hot (6) + loop upper bounds (7, divided by 4096) + iterator types (7) + load access matrices (7x7x7) + store access matrices (7x7x7) + op counts (5: `+ - * / exp`) |
| `ProducerOpFeatures` | 711 | same, zeros if no producer |
| `ActionHistory` | 1,085 | tiling-type history in three blocks (5 steps x 7 loops x 8 sizes = 280 each) + interchange history (5 x 7 x 7 = 245) |
| `NumLoops` | 1 | loop count of the current op |
| `ActionMask` | 181 | 6 action flags + 3 x 56 tile masks + 7 interchange flags |
| **Total** | **2,689** | |

Things worth knowing:
- Features are built from `state.original_operation_features` for the consumer (not the edited one), and the updated `operation_features` only feed `NumLoops` and the masks. The policy therefore sees the original op plus the history of what was done to it.
- Access matrices store only the **coefficient of each loop index** per dimension. There is no "constant" column (the manuscripts' `N+1`).
- The `vectorizable` flag is parsed but its line in the feature vector is commented out.
- By my reading, the three 280-wide tiling blocks are identical: `TiledParallelization` and `TiledFusion` inherit `Tiling.action_history`, which checks `isinstance(action, Tiling)`, so every T, TP or TPF fills all three blocks. The sizes match the thesis' "1,085", but the information content is one block repeated three times.

### 4.5 [model.py](MainGit/rl_autoschedular/rl_autoschedular_v5/model.py): the networks

`HiearchyModel` (sic) = `PolicyModel` + `ValueModel`, with **separate** encoders (no shared trunk, as in Ameur and Tirichine).

`TransformerEmbedding` (identical in structure to the thesis):
- 17 tokens for `L = 7`: `[CLS]`, a consumer summary, a producer summary, 7 consumer loop tokens, 7 producer loop tokens.
- Summary token: `OpFeatures` (711) -> MLP -> `d_model`. Loop token: per-loop slice (bound, iterator type, that loop's column of every access matrix, op type, op counts = 111 numbers) -> MLP -> `d_model`.
- Three additive learned embeddings: role (global/consumer/producer), token type (cls/summary/loop), loop depth.
- `nn.TransformerEncoder`, pre-norm, GELU, dropout 0.1, key-padding mask for absent loops.
- Output = CLS vector, concatenated with the **raw** action history (the history bypasses the Transformer).

Heads: a 3 x 512 ReLU backbone, then one 512 -> 512 -> N head per action (6 for selection, 56 each for T/TP/TPF, 7 for interchange; NT and V have no head). The value network is 3 x 512 ReLU -> 1.

Sizes: the **configs in `config/v5/` use a small Transformer** (`d_model 64`, 2 heads, 2 layers, `ffn 128`). The thesis' numbers (256 / 8 / 3 / 1024) only appear in the `*_large.json` configs of `config/ops_and_blocks/`.

PPO loss ([model.py:213](MainGit/rl_autoschedular/rl_autoschedular_v5/model.py#L213)): standard clipped surrogate, with the behaviour-policy ratio handled so that experience replay would work; with `reuse_experience: none` the extra term is 1.

### 4.6 [env.py](MainGit/rl_autoschedular/rl_autoschedular_v5/env.py): the environment

Small and readable. Four public methods:
- `reset(benchs, idx)` picks a benchmark and builds the state for its **last** op (`__init_op_state(-1)`), choosing the last producer as the fusion candidate.
- `step(state, action)` copies the state, records the action, calls `action.update_features` (and `update_producer_features` for fusion), and sets `terminal` if the action is terminal, if the symbolic update raised, or if `step_count == truncate`.
- `get_next_op_state(state)` moves to the previous op (or returns `None` when the first op is done), carrying the history forward.
- `apply_and_run_sequence(seq)` is the **only place real MLIR is touched** for training: parse the benchmark, apply every action of every op in visit order, execute, and compute the reward.

Reward detail ([env.py:231](MainGit/rl_autoschedular/rl_autoschedular_v5/env.py#L231)): `log10(T_root / T_new)`; -5 if a transformation raised; -20 if execution failed or timed out. All earlier transitions of the episode get 0.0. One subtlety: if the very last action fails to apply, its -5 is overwritten by the final execution reward (`rewards[-1] = ...`).

### 4.7 [transforms.py](MainGit/rl_autoschedular/rl_autoschedular_v5/transforms.py): speaking to MLIR

Each `transform_*` function builds a **Transform-dialect script as a string** (`transform.structured.match attributes{tag="operation_007"}`, `tile_using_forall`, `fuse_into_containing_op`, `interchange`, `vectorize`, ...) and calls `__run_transform_code`, which writes the module and the script to temp files and runs the transform interpreter in a **fresh Python subprocess**, then re-parses the printed result back into the in-memory module. Slow, but a crash in MLIR's C++ cannot kill the trainer. `BindingsProcess` (a forked-process wrapper) exists but is disabled by default (`ENABLE_BINDINGS_PROCESS=0`; `AGENTS.md` warns that fork corrupts MLIR state).

Operations are found by their `tag` attribute, which is why every benchmark needs `{tag = "operation_NNN"}` on each Linalg op (the data pipeline adds them, see `data_utils/` and `scripts/full_model/preprocess_model.py`). `transform_bufferize_and_lower_v` is the pre-execution cleanup (bufferization, vector lowering patterns).

### 4.8 [execution.py](MainGit/rl_autoschedular/rl_autoschedular_v5/execution.py): timing

`Execution` (singleton) does:
1. **Cache lookup** by `bench_name` + a key made of the schedule's string form (`"I(1,0)TP(...)V(True)|..."`). Order: preloaded `main_exec_data`, an in-memory snapshot, then the JSON file `logs/exec_data.json`.
2. **Timeout**: `min(MAX, max(MIN, COMPILE_FLOOR, 5 x root_time))`, where the three bounds default to 300 s and can be raised by environment variables. With the defaults it is a flat 300 s.
3. **Isolated execution**: a `multiprocessing.Process` parses the module, bufferizes, runs a pass pipeline (linalg to loops, `scf.forall` to OpenMP, everything to LLVM), builds an `ExecutionEngine` at `-O3`, calls `main` **twice** and takes the median of the two (so the mean). Inputs are zero-filled arrays.
4. If the child fails or times out, **fallback** to `mlir-opt | mlir-cpu-runner`.

The measured time is not wall-clock around the call: the benchmark's `@main` returns an `i64` computed by calling `@nanoTime()` before and after the body, so JIT compilation is excluded. This is the "timed ABI" the data pipeline enforces.

### 4.9 [trajectory.py](MainGit/rl_autoschedular/rl_autoschedular_v5/trajectory.py) and [ppo.py](MainGit/rl_autoschedular/rl_autoschedular_v5/ppo.py): the learning loop

- `TrajectoryCollector` accumulates one transition per `Env.step`; `TrajectoryData` is the PyTorch `Dataset` over them, with `update_attributes(model)` computing `values`, `next_values`, `actions_old_log_p`, `returns` and GAE `advantages` in plain Python loops (backwards over the batch, resetting at `done`).
- `ppo.py` holds `rollout_benchmarks`, `collect_trajectory`, `ppo_update`, `value_update`, `evaluate_benchmarks` (greedy rollout of the whole eval set, then execute all schedules, write `eval_exec_times_<ckpt>.json`).
- `ppo_update`: 4 epochs, minibatch 64, per-minibatch advantage standardisation, loss = policy + 0.5 x value + 0.01 x (-entropy), gradient clipped at 0.5. With `value_epochs: 0` the value loss is trained jointly (the case in all v5 configs).

### 4.10 Distributed collection ([distributed.py](MainGit/rl_autoschedular/rl_autoschedular_v5/distributed.py), [utils/dask_manager.py](MainGit/rl_autoschedular/rl_autoschedular_v5/utils/dask_manager.py))

Not described in any manuscript (parallel training is listed as future work in Bouloudene and Bouchama). The driver keeps the model and the PPO update. Worker tasks (`rollout_group`) receive the current weights, run `rollout_benchmarks` plus the execution for a group of 4 benchmarks, and return trajectory segments, which the driver concatenates. `DaskManager` is a wrapper that, when Dask is disabled, runs the same `map_objs` with a plain thread pool, so the single-node path needs no cluster. `scripts/train/train.py` swaps `collect_trajectory` for `collect_distributed_trajectory` when Dask is enabled.

### 4.11 `utils/` inside the package

`config.py` (typed singleton filled from `CONFIG_FILE_PATH`; fields without a default raise if missing), `singleton.py` (metaclass), `file_logger.py` (appends metrics as text lines under `results_dir/logs/`, creates `models/` and `exec_data.json`), `log.py` (coloured prints), `bindings_process.py` (disabled fork wrapper), `dask_manager.py`, `gpu_occupier.py` (only in `paper*`).

---

## 5. What lives outside the package

| Folder / file | Role | Talks to the package through |
|---|---|---|
| [scripts/train/train.py](MainGit/scripts/train/train.py) | Operational training loop: resume, SIGABRT catch, seeding, Dask switch, checkpoint every 50 steps | `utils.implementation.import_autoschedular_module` |
| [scripts/eval/eval.py](MainGit/scripts/eval/eval.py) | Evaluates every (strided) checkpoint with `ppo.evaluate_benchmarks`, resumable | same |
| [rl_autoschedular_v5/train.py](MainGit/rl_autoschedular/rl_autoschedular_v5/train.py), `evaluate.py`, `baseline.py` | Simpler stand-alone versions of the above (no resume; `baseline.py` times the untransformed eval files) | direct imports |
| `scripts/baseline/get_base.py`, `scripts/data/split_json.py` | Produce `{bench: root_time_ns}` JSONs and the train/eval split (stratified by model family, 80/20) | JSON files |
| `scripts/eval/eval_fullmodel.sh` etc. | Run the block-trained policy on whole models (V5.2) | `Benchmarks` + `full_model.py` |
| `config/` | One JSON per experiment; `v5/` ones drive both train and eval | `Config` |
| `data_utils/` | `convert_model.py` (PyTorch/HF/GNN to Linalg MLIR), `extract_from_model.py` (single ops and sliding-window blocks, using AstDumper), `generate/` (synthetic ops), `postprocess/` (add the timed `@main`, strip weights) | produces the `.mlir` files `Benchmarks` reads |
| `tools/ast_dumper`, `pre_vec`, `vectorizer` | C++ MLIR programs. `AstDumper` = feature extractor and producer graph; `PreVec` = pre-vectorisation rewrite called by `Vectorization`; `Vectorizer` = older alternative (config flag in README) | subprocess via `AST_DUMPER_BIN_PATH`, `PRE_VEC_BIN_PATH` |
| `dashboard/` | Streamlit pages (version comparison, ablation, multi-hardware, full model) over `results/` | files only |
| `docs/` | `design/VERSIONS.md` and `design/done/*` explain each version and are accurate about intent; `onboarding/` is a gentler intro. Treat as notes, not specification: the READMEs inside `rl_autoschedular/` describe a file layout (`agent.py`, `environment.py`) that does not exist | n/a |

---

## 6. How much of the manuscripts is in the code

Legend: **Y** implemented as described, **P** partial or different, **N** not in the code. "v5" column is the newest package; "v4.5/4.9" notes where the thesis' own features live.

### Aouadj (search-based scheduler, 2023)
| # | Concept | v5 | Where / note |
|---|---|---|---|
| 1 | Schedule through the Transform dialect on Linalg | Y | `transforms.py` |
| 2 | Tiling, parallelisation, interchange, vectorisation | Y | `actions/` |
| 3 | C++ search scheduler (graph API, beam/exhaustive search) | N | not in the repo |
| 4 | MLP cost model (744 inputs) | N | no cost model anywhere |

### Bendib (first RL environment, 2024)
| # | Concept | v5 | Where / note |
|---|---|---|---|
| 5 | Hierarchical action space | Y | `actions/__init__.py` |
| 6 | Interchange of consecutive loops | P | generalised: pointers / swaps at distance 1-3 / continuous |
| 7 | Observation: bounds, load/store matrices, op counts, history | P | no constant column; 5 op kinds, not 6 (no `log`) |
| 8 | Final reward = log speedup | Y | `env.py` |
| 9 | Env rules: TP once, decompose conv before vectorising, timeout | P | TP once Y; decompose Y; timeout is 300 s, not 1 min |
| 10 | PPO | Y | `ppo.py` (DQN not present) |
| 11 | Network: shared 3x1024 MLP | P | separate actor/critic, 3x512 |

### Ameur and Tirichine (generalisation, 2025)
| # | Concept | v5 | Where / note |
|---|---|---|---|
| 12 | Operations in reverse order | Y | `Env.get_next_op_state` |
| 13 | Features updated analytically after each action | Y | `update_features` |
| 14 | Execution cache keyed by the schedule | Y | `Execution.get_code_cache_key` |
| 15 | conv2d to matmul at initialisation | P | img2col, only for files named `conv_2d_*` |
| 16 | Tile sizes = divisors of the bound | P | powers of two that divide it, up to 64 |
| 17 | Separate parallel and reduction tiling | Y | `TiledParallelization` |
| 18 | Three interchange encodings | Y | all in `interchange.py`; pointers used |
| 19 | Action masks and configurable `order` | Y | `ActionSpace.action_mask` |
| 20 | Reward: log10 speedup, -5, -20 | Y | `env.py` |
| 21 | Separate actor and critic | Y | `model.py` |
| 22 | Vectorisability flag in the state | P | parsed and used to invalidate, but not fed to the network |
| 23 | AlphaZero / MCTS | N | no trace in the repo |
| 24 | LQCD front-end (Halide IR to Linalg) | N | only name matching in the dashboard |

### Bouloudene and Bouchama (whole programs, 2025)
| # | Concept | v5 | Where / note |
|---|---|---|---|
| 25 | Tiling + fusion action (TF) | Y | `TiledFusion` (symbol `TPF`) |
| 26 | "Next" terminal action | Y | `NoTransformation` (`NT`) |
| 27 | Consumer to producer traversal, last op first | Y | `Env` |
| 28 | Consumer and producer in the state | P | yes, but as two flat vectors; only the last producer |
| 29 | Recursive AST / LSTM loop embedding | N | v5 has none. `LSTMEmbedding` in `v0`, `paper`, `v5_no_transformer` is a plain 2-step LSTM over the two flat vectors |
| 30 | MLIR Python bindings | Y | `transforms.py`, `execution.py` |
| 31 | Persistent incremental cache | P | keyed by schedule string, not a SHA-256 of the code |
| 32 | Reward once per operation | N | one reward per benchmark, at the end |
| 33 | Synthetic chains + NN-block dataset | Y | `data_utils/generate`, `extract_blocks.py` |

### Benchikh and Bessai (2026)
| # | Concept | v5 | v4.5 / v4.9 | Where / note |
|---|---|---|---|---|
| 34 | PyTorch to ONNX to Linalg pipeline, sliding-window blocks | Y | Y | `data_utils/` (window/stride in `extract_blocks.py`) |
| 35 | Hardware vector (7 values), concatenated after the encoder | N | Y | `observation.py` of v1/v4/v4.5/v4.9 |
| 36 | Transformer encoder, 17 tokens, role/type/depth, CLS | Y | Y | `model.py`; v5 configs use the small size |
| 37 | Terminal masking; V forbidden at depth > 6 | P | P | order-based terminal masking Y; V rule is "iteration space <= 512", not depth |
| 38 | Shaped reward (formula, weights, clip, +0.2 on V) | N (removed) | Y in v4.5, off in v4.9 | matches the thesis' decision to drop it |
| 39 | Zero shaped rewards if execution fails | N/A | Y | `env.py` of v2.5+ |
| 40 | Measurement: zero buffers, warm-up, one run | P | P | zero buffers Y; two runs, median, no discarded warm-up |

**Tally for v5** (rows 1-40): Y = 20, P = 11, N = 8 (rows 3, 4, 23, 24, 29, 32, 35, 38), N/A = 1 (row 39, which only exists when a shaped reward exists).
Counting the thesis' own three novelties (rows 35, 36, 38): v5 has 1 of 3; v4.5 has 3 of 3; v4.9 has 2 of 3 (hardware, Transformer), which is exactly what the thesis reports as its final agent.

### Not in any manuscript but in the code
Distributed PPO with Dask (v5.1); full-model evaluation by running the per-op policy over whole models with the timed ABI adapter (v5.2, 18 of 18 models in the docs); epsilon-greedy as an alternative to entropy exploration; experience replay (`topk`, `random`); CLI fallback for execution; hyper-parameter search scripts (`scripts/hpo`); the dashboard.

### Named in the manuscripts' "future work" and still missing
Unrolling, padding, packing, skewing (only `num_pad_multiples` / `num_unroll_factors` placeholders in `Config`, and a design doc `docs/design/todo/v5_3_expanded_action_space.md`); a learned cost model; GPU targets; tree search at inference time.

---

## 7. Traps when reading or running this code

1. **Config keys that do nothing in v5.** `gae_lambda`, `ppo_clip_range`, `max_grad_norm`, `eval_runs`, `eval_aggregation` and `transformer_use_action_history_token` are declared in `Config` but never read by the v5 code. The real values are hard-coded: lambda 0.95, clip 0.2, grad-norm 0.5, one eval run. Changing them in the JSON changes nothing.
2. **`gamma = 1`.** Manuscript tables list `gamma = 0.99`. The code uses 1.0 for returns and GAE, and the config has no gamma field.
3. **Two `Config` singletons.** `scripts/` use `utils.config.Config`; the v5 package uses `rl_autoschedular_v5.utils.config.Config`. They read the same JSON but are different classes with different fields (the top-level one knows `seed`, the package one does not).
4. **Failed speedup convention.** `Env.failed_seq` returns speedup 1.0 in v5 but 0.0 in v0 and v4.9. Averages across versions are not computed the same way.
5. **Speedups against what?** `root_exec_time` is the time of the *unscheduled* code lowered by the same pipeline, taken from `base_train.json` / `base_eval.json`. Large speedups on tiny kernels are baseline artefacts (the v4.9 doc itself flags `124,209x` outliers).
6. **Entry-point duplication.** Both `scripts/train/train.py` and `rl_autoschedular_v5/train.py` exist. The scripts one is the maintained one; its checkpoint cadence (50) and resume logic differ from the package one (100, none).
7. **`data/` is empty here.** Nothing can be sampled or timed from this checkout; reading is all you can do until the dataset and the MLIR build are available.

---

## 8. A reading order that works

1. [context.md](context.md) (what each manuscript claims) then section 3 of this file.
2. [state.py](MainGit/rl_autoschedular/rl_autoschedular_v5/state.py) (just the dataclasses).
3. [env.py](MainGit/rl_autoschedular/rl_autoschedular_v5/env.py) (short; shows the whole control flow).
4. [actions/tiling.py](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/tiling.py), then [actions/__init__.py](MainGit/rl_autoschedular/rl_autoschedular_v5/actions/__init__.py) (masks, sampling), then `interchange.py` and `tiled_fusion.py`.
5. [observation.py](MainGit/rl_autoschedular/rl_autoschedular_v5/observation.py) and [model.py](MainGit/rl_autoschedular/rl_autoschedular_v5/model.py) (the `TransformerEmbedding`; compare with `rl_autoschedular_v0/model.py` `LSTMEmbedding`).
6. [ppo.py](MainGit/rl_autoschedular/rl_autoschedular_v5/ppo.py) and [trajectory.py](MainGit/rl_autoschedular/rl_autoschedular_v5/trajectory.py) (you know PPO; look for how sparse reward and `done` interact).
7. [transforms.py](MainGit/rl_autoschedular/rl_autoschedular_v5/transforms.py) and [execution.py](MainGit/rl_autoschedular/rl_autoschedular_v5/execution.py) (the MLIR-facing code; skim now, return after you have a working MLIR build).
8. `rl_autoschedular_v4_9/observation.py` (hardware vector) and `env.py` (shaped reward) to see what the thesis added.
