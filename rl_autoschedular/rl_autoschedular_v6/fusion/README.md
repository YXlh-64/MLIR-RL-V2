# Fusion layer (rl_autoschedular_v6)

A fusion layer in front of the MLIR-RL agent, following `/scratch/mh8826/fusion_plan.md`
(DNNFusion-style edge classification adapted to tile-and-fuse).
It classifies every producer→consumer edge and pre-fuses the always-safe (green) edges.
It builds closed, representative training units from full models, gives Tiled Fusion
the best *data* producer, and can mask unprofitable TF tile sizes.
The agent's network, observation, action space and PPO loop are unchanged.

Every module is switched by the `fusion` section of the config. With the section absent
or `{}`, v6 behaves exactly like v5; `tests/test_regression_v5.py` checks this.
Every change to existing v6 files, and every decision, is logged in [CHANGES.md](CHANGES.md).

## Modules

| Module | File | Design rules | What it does |
|---|---|---|---|
| Rules (single source of truth) | `rules.py` | R1–R6 | Op types (One-to-One, Broadcast, Shuffle, Reduction, Window, Init) and edge classes (`green`, `red`, `init`, `yellow:<epilogue\|partial\|halo\|shuffle\|unparsed>`), as plain functions and constants. Edit the rules here only. |
| Views | `views.py` | R1 | Builds the op graph from (a) MLIR bindings, (b) the gap1 text parser (fallback), or (c) v6 `BenchmarkFeatures`. Tags match the AstDumper. |
| M1 classify | `classify.py` | R1–R6 | Per-op types and per-edge classes. CLI writes JSON. |
| M2 prefuse | `prefuse.py`, `check.py` | R2, R7, R12, R14 | MLIR elementwise fusion restricted to green edges (everything else is pinned). Reports op counts. `--check` compares outputs on random inputs. |
| M3 units | `units.py`, `unit_extract.py`, `mlir_io.py` | §4 (U1) | Splits a full model at residual/layer boundaries, groups repeated layers by structural signature, and writes one closed timed unit per distinct layer (weight = occurrences) plus `units_manifest.json`. |
| M4 producer select | `producer_select.py` | R3, R4, R8 | TF gets the best data producer (no init/red, largest intermediate result) instead of `producers[-1]`. Load-time statistics answer A8. |
| M5 tile guard | `tile_guard.py` | R10 | Removes TF tile sizes with too much recomputation (partial edges) or an output tile larger than L2. |
| Integration | `config.py`, `hooks.py` | — | `FusionConfig` (switches, cache guard) and `apply_fusion` (runs at the end of `Benchmarks.__init__`). |
| Eval set | `eval_set.py` | — | Loader of `config/fusion/eval_set.json`, the single source of truth for the evaluation/exclusion list. |
| Sampling | `sampling.py` | §4.2 (weighting) | Weighted draw of the iteration's benchmarks (`fusion.sampling_weights`); off = v5 uniform `randperm`. |

Out of scope: **multi-producer TF (R11)**. TF still fuses one producer, and D+ = D + tile guard only.
A set of producer tokens in the observation is also out of scope (it would change the network).

## Switches (`fusion` section of the config)

```jsonc
"fusion": {
  "prefuse": false,            // M2: pre-fuse green edges of every loaded benchmark
  "producer_select": false,    // M4: TF producer = best data producer
  "tile_guard": false,         // M5: mask unprofitable TF tile sizes
  "tile_guard_params": {"max_recompute": 8.0, "l2_bytes": 1048576, "element_bytes": 4},
  "sampling_weights": "",      // JSON {benchmark: weight}; "" = uniform as v5 (fusion/sampling.py)
  "units": {"min_ops": 5, "max_ops": 25, "global_span_fraction": 0.25, "global_min_users": 3,
            "eval_set": "config/fusion/eval_set.json"}   // offline M3 builder only (leakage guard)
}
```
Unknown keys are rejected. `units` and `tile_guard_params` are passive: they don't turn anything on.

**Evaluation set:** `config/fusion/eval_set.json` is the single source of truth: all 18 models of `data/full_models`, 13 of them runnable in v6. The units leakage guard, the eval JSON (`scripts/fusion_data/eval_set_tools.py`), the survey comparison and the config tests all read it.

## Integration points (existing v6 files, each change is small and guarded)
- `utils/config.py`: new field `fusion: dict = {}`.
- `benchmarks.py`: at the end of `__init__`, `apply_fusion(...)` runs only if a module is on. It prefuses, then **re-extracts features from the fused code**, keeps `root_exec_time` (R12), and logs the M4 statistics.
- `env.py`: in `__init_op_state`, if `producer_select` is on, `select_producer(...)` replaces the `producers[-1]` result. No loaded feature is modified (see CHANGES.md, MUST FIX 4).
- `actions/tiled_fusion.py`: `TiledFusion.action_mask` ANDs the tile-guard mask with the inherited Tiling mask when `tile_guard` is on.
- `ppo.py` / `distributed.py`: the iteration's benchmarks come from `sample_bench_indices` (uniform `randperm` as v5 unless `sampling_weights` is set).
- Step 0: all v6 imports changed from `rl_autoschedular_v5` to `rl_autoschedular_v6`.

The standard entry points work unchanged: `scripts/train/train.sh` and `scripts/eval/eval.sh` with a config whose `implementation` is `rl_autoschedular_v6`. No wrapper scripts are needed.

## How to run

Environment (from the repo root): `source scripts/analysis/gap1/env.sh`.

```bash
# M1: classify edges
python -m rl_autoschedular_v6.fusion.classify data/full_models/resnet50_linalg.mlir --out classes.json

# M2: pre-fuse one file (never in place). --check compiles and runs both programs (do this on a compute node).
python -m rl_autoschedular_v6.fusion.prefuse in.mlir out.mlir --report report.json [--check]

# M3: units for the training data are built by the pipeline in scripts/fusion_data (from the
# new models of data/fusion_units/models.json); every model of config/fusion/eval_set.json is refused.
python -m rl_autoschedular_v6.fusion.units some_model_linalg.mlir --out DIR --config config/fusion/A_prime.json

# Tests (fixtures only; nothing is timed or trained)
python -m pytest rl_autoschedular/rl_autoschedular_v6/fusion/tests -q
```

## Ablation rows (`config/fusion/`, design doc §6)

| Row | Train config | Data | prefuse | producer_select | tile_guard |
|---|---|---|---|---|---|
| A | `A.json` | current blocks (`data/ops_and_blocks`) | off | off | off |
| A′ | `A_prime.json` | units (`data/fusion_units/units`, weighted sampling) | off | off | off |
| B | `B.json` | units | on | — | — |
| C | `C.json` | units | on | off | off |
| D | `D.json` | units | on | on | off |
| D+ | `D_plus.json` | units | on | on | on |

- **Evaluation:** use `<row>_eval_fullmodel.json` with `scripts/eval/eval_fullmodel.sh --checkpoints-dir <train results_dir>/models`. It has the same `fusion` section and limits as its training config, its own `results_dir` (`<train results_dir>_fullmodel_eval`), and runs on `data/full_models` with `results/fusion_units_results/baselines/mlir/eval_fullmodel.json`.
- **B (module alone, no agent):** no training. Prefuse the unit files and the full models with the M2 CLI into new folders, then time them with `scripts/baseline/baseline_fullmodels.py --models-dir <prefused dir>`. (`get_base.py` uses the pre-V5 API and cannot be used.) Compare with the unfused baselines (R12).
- **Comparisons:** A vs A′ = data, A′ vs C = fusion layer, C vs D = producer selection, D vs D+ = tile guard. Use at least 3 seeds (`seed`).
- **All rows train from scratch** (limits 12 → a different observation shape from the v5 checkpoints). Never `--resume` from a v5 run.
- **Hardware:** baselines, training and evaluation all run on `--partition=compute --constraint=bergamo` (EPYC 9754, 1 MiB L2 per core). Single-node training needs `--cpus-per-task=64 --mem=16G`.

### Prerequisites before launching A′–D+
The training data pipeline is in `scripts/fusion_data/` (control file `data/fusion_units/models.json`): export → validate → profile → units → split → baselines. Every step has an sbatch script in `scripts/fusion_data/sbatch/`. See CHANGES.md ("Training data for A′–D+") for the full command sequence.

## Known limitations
- **R7 is conservative:** MLIR fuses single-use producers only, so green producers with ≥2 consumers stay unfused (counted as `multi_use`). Green edges into named ops (e.g. `linalg.transpose`) are not fused either (`named_op`).
- Models with `tm_tensor` ops can't be parsed by this MLIR build. The classifier falls back to text there (`unparsed` edges); v6 cannot load them at all.
- **Tile guard:** the TF mask is per loop, so the joint limits are enforced through per-loop budgets. "Don't tile" (column 0) stays allowed. The L2 size (1 MiB) and f32 element size are assumptions; set them per machine.
- **Unit sizes:** the estimate excludes cloned free ops (fills), so a few units exceed `max_ops` slightly. The manifest records the actual counts.
