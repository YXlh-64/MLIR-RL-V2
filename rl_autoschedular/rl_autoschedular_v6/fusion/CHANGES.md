# Fusion layer — change log (rl_autoschedular_v6)

Every file changed or added for the fusion layer, what changed and why.
Design: `/scratch/mh8826/fusion_plan.md`. v5 (`rl_autoschedular_v5`) is never modified.
No benchmark or dataset file is ever modified: all fusion work happens in memory or writes to new output paths.

## Environment / versions
- MLIR: llvm-project `cd708029e0b2869e80abe31ddb175f7c35361f90` (2025-01-14), build `/scratch/mh8826/llvm-project/build`, Python bindings from `tools/mlir/python_packages/mlir_core`.
- Python 3.11.18 (`MLIR-ENV` conda env), torch 2.15.0. torch-mlir is not installed in the env; the version used to export `data/full_models` is not recorded there (to fill in).
- pytest 9.2.0 installed to the session scratchpad (`pip --target`). The conda env is not modified.
- Pass pipelines used by the layer:
  - M1 classification (bindings backend, on a private re-parsed copy whose structured ops are all pinned): `builtin.module(linalg-generalize-named-ops)`.
  - M2 prefusion: `builtin.module(func.func(linalg-fuse-elementwise-ops))` (the pass's own control: single-use producers; plus its reshape-propagation, constant-folding and canonicalization patterns, which the pins neutralise).

## Step 0 — v6 made standalone (2026-11-23)
- **Changed:** all 23 `.py` files in `rl_autoschedular_v6/` (99 occurrences): `rl_autoschedular_v5` → `rl_autoschedular_v6` (mechanical `sed`).
- **Why:** v6 was a verbatim copy whose imports still loaded v5 modules, so "v6" silently ran v5 code.
- **Verified:**
  - Reverse-renaming every v6 file gives v5 byte-for-byte (the rename is the only diff).
  - All files compile under Python 3.11.
  - Importing `rl_autoschedular_v6.env` and `rl_autoschedular_v6.benchmarks` loads only `rl_autoschedular_v6.*` modules.

## M1 — edge classifier (R1–R6) (2026-11-23)
Added (no existing file changed):
- `fusion/__init__.py`: package docstring.
- `fusion/rules.py`: **single source of truth**. `OpType`, `OpView`, `op_type`, `with_init_override`, `access`, `classify_edge`, `GREEN_PRODUCER_TYPES = {ONE_TO_ONE, BROADCAST}` (user decision). Broadcast-green is effectively conditional: a broadcast feeding a contraction or conv is read `partial`/`halo`, so that edge is never green.
- `fusion/views.py`: three builders producing the same `ProgramGraph`:
  - bindings: generalize named ops on a copy and read their maps. Tags `operation_<N>` follow AstDumper order, skipping 0-loop ops.
  - text: fallback that reuses `scripts/analysis/gap1/_mlir_text.parse_text` for parsing only. Named ops come out as `unparsed`.
  - features: from v6 `BenchmarkFeatures`, for M4/M5.
- `fusion/mlir_io.py`: reads GB-sized models with weight literals and resource blobs replaced by zero splats (`full_model.materialize_elided_resources`). Only in memory.
- `fusion/classify.py`: `build_graph`, `classify_graph`, `classify_text`, and the CLI `python -m rl_autoschedular_v6.fusion.classify in.mlir --out out.json`.
- `fusion/tests/fixtures/*.mlir`: 5 hand-written fixtures (diamond, matmul+fill+bias, softmax-like reduce→broadcast, transpose, two residual blocks), in the timed `@main` ABI.
- `fusion/tests/{conftest,test_rules,test_classify}.py`.

Decisions and assumptions:
- INIT op type = has ≥1 Linalg consumer and all of them read it as `outs`. Non-Linalg uses (return, reshape) are ignored, so all backends agree. INIT never changes an edge class: init edges are decided by the consumer's operand role.
- Edge classes use the producer's base type (before the INIT override).
- Stride maps (`d0 * 2`) in an all-parallel op are typed Broadcast. That is harmless, because both types are green producers.
- In the features backend, scalar inputs have no map line (AstDumper prints nothing for `()` maps), so a fill's type can differ from the bindings backend (One-to-One vs Broadcast). Edge classes still agree: both are green producers, and fill edges are init. This is covered by `test_features_backend_agrees_with_bindings`.

### M1 follow-up (made during M2)
- `views.py`: graph building is split into `tag_module(module)`, which tags the given module in place so that prefuse can transform the same module, plus `parse_module` and `pin_values`. `_is_structured` becomes `is_structured`.
- **Bug fixed:** `linalg-generalize-named-ops` runs MLIR's greedy driver, which erases dead ops. Real blocks contain dead ops (e.g. an unused `linalg.fill` in `llama3_2_1b_block_71`) that the AstDumper still numbers, so tags shifted and the bindings backend fell back to text on 66/400 sampled blocks. The structured ops of the generalized copy are now pinned first. Result: 0/400 fall back, and tags and edges match the AstDumper on real blocks (test added).

## M2 — green pre-fusion (R2, R7, R12, R14) (2026-11-23)
Added:
- `fusion/prefuse.py`: `prefuse_code(code) -> (code, PrefuseReport)`, `prefuse_file(src, dst)` (refuses `dst == src` and refuses to overwrite an existing `dst`), and the CLI `python -m rl_autoschedular_v6.fusion.prefuse in out [--report r.json] [--check]`.
  - **Method:** pin with `"fusion.keep"(%v)` every Linalg op that is not an expected green fusion, every reshape and every tensor constant. Then run the pass, erase the pins, and strip the temporary `fusion.tag` attributes. The output keeps the original `tag` attributes; the AstDumper renumbers them anyway.
  - **Report:** `ops_before`, `ops_after`, `green_edges`, `fused`, `green_not_fused{pinned_producer|multi_use|named_op}`, `unexpected_removed` (must be 0).
  - **Effective uses:** repeated reads of one value by one consumer with the same map count as one use, because the pass's canonicalization deduplicates them before fusing (found on `whisper_base_block_195`).
- `fusion/check.py`: `check_equivalent(original, transformed)` runs both programs with the v6 lowering on the same uniform(0.5, 1.5) inputs and compares the outputs with `allclose`. **Written but never run by me**; it compiles and executes code.
- `fusion/config.py`: `FusionConfig` (only `prefuse` so far; unknown keys are rejected), `check_cache_isolation`, `load_fusion_config()`.
- `fusion/hooks.py`: `apply_fusion(benchmarks, cfg)`: prefuse → **re-extract features from the fused code** (`extract_bench_features_from_code`) → `ensure_timed_main_abi`. `root_exec_time` is kept (R12). A program the bindings cannot parse is kept unfused, with an alert.
- Tests:
  - `tests/test_prefuse.py`, `tests/test_hooks.py`, and `tests/test_regression_v5.py`, which compares v5 and v6 with fusion off: features on the fixtures and on 3 real blocks, the full `Benchmarks` load, and every action mask, both initial and after a Tiling step.
  - `tests/fixtures/test_config.json` (v5_single_node with paths pointing at the fixtures and `fusion: {}`) and `tests/fixtures/benchmarks.json`.

Changed (existing v6 files, guarded, default off):
- `utils/config.py`: new field `fusion: dict = {}` (+ docstring). Config only reads annotated keys, so configs without `fusion` behave as before.
- `benchmarks.py`: imports `load_fusion_config`. At the end of `__init__`, `if fusion_cfg.any_enabled: self.data = apply_fusion(self.data, fusion_cfg)` (lazy import of the hook).

Execution-cache isolation (MUST FIX 2), choice and reason:
- The per-run cache is `<results_dir>/logs/exec_data.json` (FileLogger), keyed by bench name + action string. Every `config/fusion/*.json` gets its own `results_dir`, so prefused and unfused runs never share that file.
- The only shared cache is `main_exec_data_file`. `load_fusion_config` raises if prefuse is on and it is non-empty.

Decisions and assumptions:
- **R7 is conservative (deliberate departure):** the pass fuses single-use producers only. A green producer with ≥2 green consumers stays unfused, counted as `green_not_fused.multi_use`.
- Green edges into named ops (e.g. `linalg.transpose`) are not fused, because the pass only fuses `linalg.generic` into `linalg.generic` (`named_op`). Named ops are not generalized, since that would change the op names the features see.
- Dead ops are kept: the program changes only through green fusions.

Validation (read-only, in memory, nothing executed):
- 400 random real blocks from `data/ops_and_blocks`: 2000 → 1126 Linalg ops (−44%, B4 measured −49%).
- 874 producers fused out of 925 green edges. `unexpected_removed = 0` on every block, and 0 fell back to text.

## M3 — closed representative units (design doc §4, option U1) (2026-11-23)
Added (no existing v6 file changed):
- `fusion/unit_extract.py`: DAG-aware builder, written from scratch (does not reuse `extract_blocks._build_block_benchmark`). `extract_unit(code, positions)`:
  1. Moves the selected top-level ops of the public function into a new `@main`.
  2. Clones in the **free** values they read (`free_positions`: ops computing constants, i.e. scalars, `tensor.empty`, zero-init fills, with no function argument, weight or call among their inputs).
  3. Turns every other external tensor (activations, weights) into an argument.
  4. Returns **all live-outs**.
  5. Erases the old function (users before producers), verifies the module, and applies v6 `full_model.ensure_timed_main_abi`.

  Failures raise `UnitSkip` with a reason, which is recorded in the manifest. Helpers: `stamp_positions`, `op_pos`, `value_pos`, `public_function`.
- `fusion/units.py`: `UnitParams` (`min_ops=5`, `max_ops=25`, `global_span_fraction=0.25`, `global_min_users=3`, `eval_models=()`; unknown keys rejected), `analyze`, `build_units`, and the CLI `python -m rl_autoschedular_v6.fusion.units model.mlir ... --out DIR [--config cfg.json] [--allow-eval-models]`.
  - **Sequence:** top-level ops, minus timing ops, free ops and weights (tensor constants).
  - **Width:** values live across a boundary. Args, weights and "global" values (≥3 users and a live range >25% of the sequence, e.g. attention masks) are not counted.
  - **Cuts** at width ≤ 1. Segments above `max_ops` are split at the narrowest internal boundary, then small neighbours are merged up to `min_ops`. Size = Linalg ops left after green fusion, estimated with the M2 plan on the whole model.
  - **Signature:** sha1 of op names, attributes (without `tag`/`fusion.*`), types, relative internal edges and external inputs numbered by first use.
  - **Output:** one unit per signature (first occurrence), weight = number of occurrences.
  - **Leakage guard:** a model listed in `fusion.units.eval_models` raises `EvalModelLeakage`. `--allow-eval-models` downgrades this to a printed warning.
  - **Manifest** `units_manifest.json`:
    - `params`, plus per model `source_model`, `segments`, `skipped`, `missing_op_names` (coverage, §4.2 principle 4);
    - per unit: `file`, `signature`, `weight`, `occurrences`, `ops_raw`, `ops_after_green_fusion` (actual, by running prefuse on the unit), `multi_producer_ops`, `edge_classes`, `op_names`.
  - Never overwrites a unit file or a manifest.
- `fusion/prefuse.py`: `_plan` renamed to the public `plan_fusion` (reused for size estimates).
- `fusion/views.py`: `walk_ops(root, keep)` accepts any op, not only a module.
- `fusion/config.py`: `FusionConfig.units` (passive dict for the offline builder). `MODULE_FLAGS` lists the switches that change training (`any_enabled` only looks at those).
- `fusion/tests/fixtures/residual_stack.mlir` (full-model style: input arg, weights as constants, a stem and 3 identical residual blocks), `fusion/tests/test_units.py`.

Decisions and assumptions:
- **Free ops (found on ResNet-50):** torch-mlir shares one zero-init `linalg.fill` across all convs of a stage. Counting it as live hid every block boundary in stages 1, 2 and 4. Free ops are now cloned into each unit that uses them, so a unit owns its own fill, just like the existing blocks.
- Weights stay arguments (zeros at execution, as for blocks). Values never affect timing; `mlir_io.read_mlir` elides large literals before analysis.
- The size estimate counts segment ops only, not the cloned free fills. The manifest reports actual per-unit counts, which can exceed `max_ops` slightly (e.g. BERT attention half: 28–29).
- A trailing segment below `min_ops` stays small (nothing follows it to merge with).
- Models whose blocks really differ (VGG, MobileNetV3: different shapes per block) give weight-1 units. This is expected: U1 only deduplicates true repeats.

**Prerequisite for configs A′–D+:** the units need a benchmark JSON `{unit_name: root_exec_time}`, measured on the cluster with the existing baseline script on the **unfused** unit files (R12). I did not run any timing.

Validation (read-only, outputs in the session scratchpad only, nothing executed):

| Model | Segments | Units | Weights (top) | Skipped | Missing op names |
|---|---|---|---|---|---|
| ResNet-50 | 18 | 10 | 5, 3, 2, 2 | 0 | none |
| VGG-16 | 11 | 11 | all 1 | 0 | none |
| MobileNetV3-small | 18 | 18 | all 1 | 0 | none |
| BERT | 51 | 8 | 12, 12, 12, 11 (layer halves) | 0 | none |
| ViT-B/16 | 38 | 6 | 12, 12, 11 | 0 | none |

All 53 units load through the v6 AstDumper feature path (`extract_bench_features_from_file`), with 921 ops in total, and all have the timed ABI. The total run took 22 s.

## M4 — producer selection (R3, R4, R8) (2026-11-24)
MUST FIX 4: readers of `producers` in v6 (outside `fusion/`), and the conclusion:
- `actions/tiling.py:121`: `Tiling.update_features`, inherited by TP and TF through `super()`, sets the **state copy's** `operation_features.producers = []` after any tiling action, because a tiled op loses its producers outside the tile loop. It only writes and never reads entries, so it is unaffected by any selection.
- `env.py:187-191`: picks `producers[-1]` and the min operand index of that tag. This is what M4 replaces.
- `actions/tiled_fusion.py:128`: overwrites the consumer's list with `[(new_producer_tag, idx)]`. Unaffected.
- `actions/tiled_fusion.py:154` (`__insert_in_bench_feats`): reads the consumer's list, but only already-fused tags (`*_<consumer>`). Unaffected.
- `actions/tiled_fusion.py:173` (`__handle_producer_original_op`): reads the **producer's own** `producers` to update *their* `consumers` lists. **Affected** if init/red entries were dropped from the loaded data: a fill Q dropped from P's list would keep a stale `P` entry in `Q.consumers` instead of v5's `P_<consumer>` entry. The decisions that read these lists (`len(set(consumers)) <= 1`) come out the same in the cases I worked through, but the feature data would still diverge from v5.
- **Conclusion: filtering the loaded `producers` is not fully safe, so M4 is scoped to TiledFusion's view.** No loaded feature is modified. `Env` calls `select_producer` instead of the `producers[-1]` lookup, behind the flag, so only `state.producer_tag` / `producer_operand_idx` change.

Added:
- `fusion/producer_select.py`:
  - `edge_class` (features backend → `rules.py`);
  - `intermediate_size` (product of the producer's loop bounds over the result dims the consumer reads);
  - `select_producer(consumer_feats, consumer_tag, bench)`: drop `init` and `red`, take the largest intermediate result, ties go to the lexically later producer (v5's preference); returns `None` when no data producer is left;
  - `selection_stats(benchmarks)`: read-only statistics of the initial op states (`ops_with_producers`, `multi_producer_ops`, `original_last_init`, `original_last_red`, `changed`, `no_data_producer`). This answers A8.
- `fusion/tests/test_producer_select.py`: 10 selection cases on the fixtures; stats checked against hand-counted values; features unmodified; `Env` with the flag off (v5: the fill) and on (the relu), plus a real TF `env.step` on the selected producer.

Changed:
- `env.py` (existing v6 file): imports `load_fusion_config`. In `__init_op_state`, after the original `producers[-1]` lines (kept as they are), `if load_fusion_config().producer_select:` replaces `(producer_tag, producer_operand_idx)` with `select_producer(...)` or `(None, None)`. `producer_features` is now fetched under `if producer_tag is not None:`, which is equivalent when the flag is off. The regression test (v5 == v6 masks) still passes.
- `fusion/config.py`: new flag `producer_select` (in `MODULE_FLAGS`). `load_fusion_config` is cached (`functools.cache`), because Config is a process-wide singleton and `Env` calls it per op.
- `fusion/hooks.py`: when `producer_select` is on, logs `selection_stats` once at load. Data is unchanged.

A8 answer (read-only, AstDumper features, `v5_fullmodel_linalg` limits):

| Data | Ops with producers | Multi-producer | `[-1]` init | `[-1]` red | Changed | No data producer |
|---|---|---|---|---|---|---|
| 400 random training blocks | 1534 | 0 | 10 | 1 | 11 | 11 |
| ResNet-50 | 441 | 54 | 32 | 0 | 32 | 18 |
| VGG-16 | 40 | 9 | 21 | 0 | 21 | 13 |
| MobileNetV3-small | 236 | 86 | 79 | 0 | 88 | 30 |

- Training blocks have no multi-producer ops (consistent with A2), so `producers[-1]` is almost never the fill there.
- In full models (evaluation), it is the fill for most multi-producer ops. R8 therefore matters for evaluation and for unit training (configs D/D+).
- Whisper-base and DistilBERT (like GPT-2 and ALBERT) cannot be loaded by v6 at all: `tm_tensor.attention` is not registered in this MLIR build.

## M5 — tile guard (R10) (2026-11-24)
Added:
- `fusion/tile_guard.py`:
  - `TileGuardParams` (`max_recompute=8.0`, `l2_bytes=1 MiB`, `element_bytes=4`; unknown keys rejected).
  - `missed_parallel_loops(consumer, operand)`.
  - `allowed_tf_tiles(consumer, producer, operand, params, max_loops, num_tile_sizes)` → bool mask in the shape of the Tiling mask.
- `fusion/tests/test_tile_guard.py`: recompute cap on a partial edge, no cap on an epilogue edge, footprint cap, and the real `TiledFusion.action_mask` with the flags on (`on == off & guard`, column 0 always allowed).

Changed:
- `actions/tiled_fusion.py` (existing v6 file):
  - imports `Config` and `load_fusion_config`;
  - new classmethod `TiledFusion.action_mask`: `super().action_mask(state)`, AND-ed with `allowed_tf_tiles(...)` only when `fusion.tile_guard` is on and a producer is selected. When the flag is off it returns the inherited mask unchanged, so the regression test still passes.
- `fusion/config.py`: flags `tile_guard` (in `MODULE_FLAGS`) and `tile_guard_params` (passive).

Remarks:
- The TF mask is factored per loop: each loop's tile size is sampled independently. Joint limits are therefore enforced through per-loop budgets whose product respects them:
  - recompute: each consumer parallel loop missed by the producer map gets at most `max_recompute ** (1/k)` tiles;
  - footprint: each parallel loop's tile is at most `(l2_bytes / element_bytes) ** (1/k)`.
- Column 0 ("don't tile this loop") is never masked, as in v5. The footprint cap therefore can't stop the agent from leaving a loop untiled.
- The recompute cap applies to `yellow:partial` edges only. Halo edges are not capped (their recomputation is a border, not a whole slice).
- **Assumptions:** L2 = 1 MiB per core and f32 elements. Both are config values (`tile_guard_params`); set them for the training machine.
- The guard works with whatever producer the state holds. It is meant to be used with `producer_select` (config D+).

## Configs, README, final checks (2026-11-24)
Added:
- `config/fusion/{A,A_prime,B,C,D,D_plus}.json` (training) and `config/fusion/{row}_eval_fullmodel.json` (full-model evaluation), 12 files.
  - **Training configs:** copy of `config/v5/v5_single_node.json` with: `implementation: rl_autoschedular_v6`, `main_exec_data_file: ""`, its own `results_dir`, the limits raised to 12, and the `fusion` section.
  - **Eval configs:** copy of `config/v5/v5_fullmodel_linalg.json` (Dask settings, `data/full_models`, `config/mlir_rl_v1_paper/eval/fullmodel_linalg_eval.json`) with the same `fusion` section and `results_dir` as the row's training config.
  - Units data path `data/fusion_units` (does not exist yet). Units baseline JSON `results/fusion_units_results/baselines/mlir/base_train.json` (to be measured).
- `fusion/tests/test_configs.py`:
  - every file loads through the real v6 `Config` and `load_fusion_config` (in a subprocess);
  - the flags per row match §6;
  - train and eval share flags, `results_dir` and limits;
  - every row has a distinct `results_dir`;
  - `eval_models` equals the eval list.
- `fusion/README.md`: module → rule table, switches, integration points, how to run, ablation rows, prerequisites, limitations.

Remarks:
- **Limits (deliberate):** every row uses loop/load limits of 12, as the existing full-model eval config does, because training and evaluation must share the observation shape. Row A therefore differs from the original `v5_single_node` run in these limits only. Code-level equality with v5 is what `test_regression_v5.py` proves. For an exact v5 replication, use `config/v5/v5_single_node.json` with `implementation: rl_autoschedular_v6`.
- **Leakage:** `fusion.units.eval_models` = the 9 models of `fullmodel_linalg_eval.json` (convnext_tiny, efficientnet_b0, gat, gin, mobilenet_v3_small, resnet50, resnext50, t5, vgg16). My M3 trial (scratchpad only) used ResNet-50, VGG-16 and MobileNetV3, which are eval models. Those trial units must not be used for training; the guard now refuses them.
- **Non-eval models left for units:** bert, vit_b_16, bart, llama3_2_1b (9.9 GB). gpt2, albert, distilbert and whisper_base contain `tm_tensor.attention` and cannot be parsed by this MLIR build. The pool of training models is therefore small, which may limit A′–D+ unless more models are exported.
- **Eval with prefusion on:** full models are prefused in memory at load. This parses the whole model with the bindings (BERT-size files are hundreds of MB), so expect extra load time and memory on the eval node.
- **No `scripts/fusion/` wrappers:** the brief asked for `train_with_fusion.py` / `eval_with_fusion.py`. Once you allowed hooks in v6, the standard `train.sh` / `eval.sh` run the fusion layer directly from the config, so wrappers would be dead code.
- **No patch files:** every integration point was reachable with the allowed v6 hooks.
- Config B has no training. Prefuse the units and full models with the M2 CLI into new folders and time them with `scripts/baseline/get_base.py --benchmarks-dir` (see README).
- pytest was reinstalled into the session scratchpad on 2026-11-24 (the scratchpad had been cleared). The conda env is unchanged.

---
# Training data for A′–D+ (plan: `/scratch/mh8826/data_conversion_plan.md`) (2026-11-24)

## Part 0 — re-check
- `pytest fusion/tests`: 116 passed. `py_compile` on every v6 `.py` file: 0 failures. Nothing changed.

## Part 1 — evaluation set (findings; decision pending)
Eval lists in the repo:
- `config/mlir_rl_v1_paper/eval/fullmodel_linalg_eval.json` (**linalg-9**): convnext_tiny, efficientnet_b0, gat, gin, mobilenet_v3_small, resnet50, resnext50, t5, vgg16.
  - Used by `config/v5/v5_fullmodel_linalg.json`, `v5_no_transformer_fullmodel_linalg.json`, and **all `config/fusion/*_eval_fullmodel.json`**.
  - It is also the current `fusion.units.eval_models`.
- `config/mlir_rl_v1_paper/eval/fullmodel_relowered_eval.json` (**relowered-9**): albert, bart, bert, distilbert, gpt2, llama3_2_1b, vit_b_16, whisper_base, yolov8m.
  - Used by `config/v5/v5_fullmodel_relowered*.json` with `data/full_models/relowered`, **which does not exist on this machine**.
- `fullmodel_combined_eval.json` (21) = linalg-9 + relowered-9 + the 3 paper `model_*` files.
- `fullmodel_eval.json` (3) = the paper models.
- `data/full_models` (18 files, at `/scratch/mh8826/data/full_models`) = linalg-9 ∪ relowered-9.
- `model_survey/eval_models.json` (15) = the 18 minus yolov8m, gat and gin (not buildable from configs).

Parse status of the 18 files with the v6 MLIR (llvm `cd708029`):
- **Parse checks:**
  - bindings parse + AstDumper + `extract_bench_features_from_code` on weight-elided text: OK for convnext_tiny, efficientnet_b0, gat, gin, resnext50, t5, yolov8m (2026-11-24), and for resnet50, vgg16, mobilenet_v3_small, bert, vit_b_16 (M3 trial).
  - Not parsed (too heavy for a login node; covered by the validate job): bart (1.3 GB), llama3_2_1b (19 GB). Both have no `tm_tensor`, no `torch.` ops and no dynamic dims. Llama blocks in `ops_and_blocks` do parse.
- **Fail:** albert, distilbert, gpt2, whisper_base contain `tm_tensor.attention` (text scan: 12 / 6 / 24 / 6 lines).
- **Parses but not executable:** yolov8m has dynamic dims (`tensor<1x?x160x160xf32>`, 279 lines). `Execution._create_params` needs static shapes.
- The `torch_tensor_*` strings in gat, gin, resnext50 and t5 are only `dense_resource` names, which `mlir_io` elides.

Remarks:
- The 9 eval configs I wrote (`config/fusion/*_eval_fullmodel.json`) use `benchmarks_folder_path: data/full_models`, relative to the repo root (as `v5_fullmodel_linalg.json` does). But `MLIR-RL/data/` only contains the `ops_and_blocks` symlink, so that path does not resolve on this checkout. A `data/full_models` symlink (or an absolute path) is needed before any full-model eval.
- **Leakage risk in the earlier README advice:** README/CHANGES suggested building units from bert, vit_b_16 and bart. Those are in relowered-9. If relowered-9 (or the union) is an eval set, that advice leaks. It is superseded by Part 1's decision.

### Part 1 decision (user, 2026-11-24): exclusion list = all 18 models of `data/full_models`; eval runs on the 13 that work in v6
Added:
- `config/fusion/eval_set.json` — **single source of truth**. Per model: `name`, `file` (stem in `data/full_models`), `runnable`, `reason`, `survey` (model_survey build recipe or null).
  - 13 runnable.
  - 5 not runnable: albert, distilbert, gpt2, whisper_base (`tm_tensor`); yolov8m (dynamic dims).
  - gat, gin and yolov8m have `survey: null` (PyG / ultralytics).
- `fusion/eval_set.py`: `load_eval_set`, `excluded_names`, `runnable_models` (stdlib only).
- `scripts/fusion_data/eval_set_tools.py`, everything derived from the eval set:
  - `eval-json`: the `{file: root_exec_time}` eval JSON for the runnable models;
  - `survey-models`: model_survey input for the runnable, buildable models;
  - `survey-subset`: slice of an existing profile;
  - `link-runnable`: a symlink folder of the runnable files.

Changed:
- `fusion/units.py`: `UnitParams.eval_models` (inline list) → `UnitParams.eval_set` (path of the eval-set JSON). The leakage guard refuses every model listed there.
- `fusion/tests/test_units.py`: the leakage test writes a temporary eval-set file.
- **All 12 configs**: `fusion.units.eval_set: "config/fusion/eval_set.json"` replaces the inline list. The eval twins' `eval_json_file` is now the generated `results/fusion_units_results/baselines/mlir/eval_fullmodel.json`.

Readers of the single source: the units leakage guard (`units.py`), the eval JSON and the survey comparison (`eval_set_tools.py`), `check_control.py` (rejects a control-file model that is in the eval set), and `test_configs.py`.

### Part 1.3 — survey re-run against the confirmed set (2026-11-24)
- **What was run:** no re-profiling was needed. `survey_eval_torch/` already holds `profile_torch.py` output (seq-len 16) for all 15 survey eval models, without failures.
  - `eval_set_tools.py survey-models` wrote `model_survey/eval_models_confirmed.json`. These are the 11 runnable models model_survey can build: bart, bert, convnext_tiny, efficientnet_b0, llama3_2_1b, mobilenet_v3_small, resnet50, resnext50, t5, vgg16, vit_b_16. gat and gin need PyG.
  - `survey-subset` sliced their rows into `model_survey/survey_eval_torch_confirmed/`.
  - `check_suitability.py survey_cand_torch/ --eval survey_eval_torch_confirmed/ --out survey_report_confirmed/ --min-ops 10`.
- **Result** (median log2; old eval-15 → confirmed eval-11):

  | Shape | Candidates | Old eval-15 | Confirmed eval-11 |
  |---|---|---|---|
  | matmul N | 8.0 | 9.6 | 9.6 |
  | matmul K | 8.0 | 9.6 | 9.6 |
  | conv C | 7.3 | 8.0 | 8.0 |
  | conv F | 7.0 | 8.0 | 8.0 |
  | depthwise C | 7.2 | 8.6 | 8.6 |

  - Category shares are closer than before (contraction 0.202 vs 0.201; transpose 0.127 vs 0.131).
  - Coverage: 47/47 eval op names.
- **Conclusion:** "candidates are too narrow" still holds, so the weight hints are kept unchanged.

## Part 2 — export environment (proposal; waiting for go-ahead, nothing installed)
- **Recommended (option B):** build torch-mlir at the commit whose `externals/llvm-project` is `cd708029` (or the bump just after it), out-of-tree against the existing LLVM build (`-DMLIR_DIR/-DLLVM_DIR=$LLVM_BUILD_PATH`).
  - Uses a separate conda env `mlir-export`: Python 3.11, torch 2.6.0 CPU, transformers 4.48.3, timm 1.0.14.
  - The exporter then uses exactly the agent's MLIR, so there is no version skew.
  - Cost: one compute job of about 30–60 min (LLVM is reused) and a few GB.
- **Option A:** a torch-mlir nightly wheel of the same date plus the torch nightly it pins. Faster, but the LLVM match is approximate, and January 2025 torch nightlies may no longer be downloadable.
- Added: `scripts/fusion_data/setup_export_env.sh` (`find-commit | create-env | build | wheel | check`), `scripts/fusion_data/sbatch/build_torch_mlir.sbatch`.
- **Compatibility test (Part 2.2) is prepared, not run:** `scripts/fusion_data/sbatch/compat_test.sbatch`.
  - Exports resnet18, distilgpt2 and deit_tiny in pretty and generic form, with eager attention.
  - Then runs `validate.py`: bindings parse, AstDumper, `extract_bench_features_from_code`, a `tm_tensor` check and a dynamic-dims check. If the pretty form fails it tries the generic form.
  - It also prints the `tm_tensor` line count per file (Part 2.3).
- The versions in `create-env` are my choice for the January 2025 period. **Unverified**: `check` confirms the imports.

## Part 3 — control file (2026-11-24)
Added:
- `data/fusion_units/models.json`: 39 entries from `model_candidates.json` plus 2 proposed (`name, repo, kind, max_layers, family, enabled, weight, reason`).
  - `defaults`: seq_len 16 (matches the eval exports), image 224, batch 1, max_vocab 32000.
  - Disabled: yolos_tiny (214 GFLOPs), vit_tiny (duplicate of deit_tiny).
  - Added and enabled: bert_L4_H768 and resnet26 (reason "proposed: wider shapes"). resnet26's repo is **unverified**: `download_models.py` checks it.
  - Weights: 2 for resnext26ts, efficientnet_b1, swin_tiny, distilroberta, distilgpt2, opt_125m, bert_L4_H768, resnet26; 0.5 for mobilebert and whisper_tiny_enc; 1 for the rest. Unchanged after Part 1.3.
- `scripts/fusion_data/control.py` (stdlib only; also used in the export env): control-file loader, validator and all pipeline paths.
- `scripts/fusion_data/check_control.py`: validates the file (fields and types, known kinds, unique names, weight > 0 if enabled, a reason for disabled models, **no eval-set model**) and prints the enabled models. **Not run.**

## Part 4 — conversion → units pipeline (scripts + sbatch, none run)
Added in `scripts/fusion_data/`. All are idempotent, driven only by the control file, and write under `data/fusion_units/`.
- `export.py` (export env, `HF_HUB_OFFLINE=1`): reuses `model_survey/export_candidates.build` (random weights, eager attention) and prints with `large_elements_limit=16` (weights elided).
  - Writes `raw/<model>_linalg.mlir`, plus `raw_generic/` with `--generic`, and `raw/export_log.json`.
  - Skips existing files unless `--force`.
- `validate.py` (MLIR-ENV, `CONFIG_FILE_PATH=config/fusion/A_prime.json`): no `tm_tensor`, no dynamic dims, bindings parse (generic form as fallback), AstDumper + features, and no op dropped by the limits.
  - Writes `validation.json`. A model is re-checked only if its file changed.
- `linalg_profile.py` (named `profile.py` until 2026-11-24, see below): runs `profile_linalg.py` on the validated raw files and on the runnable eval files, then `check_suitability.py --weights` → `profile/report/report.md`.
- `build_units.py`: fusion M3 on enabled and validated models, with the leakage guard **required** (fails if `eval_set` is empty).
  - Writes `units/<model>_unit_<k>.mlir`, `units/units_manifest.json` (updated after each model) and `sampling_weights.json`.
  - It never deletes anything: leftover unit files without a manifest entry stop the run with a message.
  - **Sampling weight per unit** = model weight × occurrences. **Not normalised per model by default** (§4.2 principle 3: a layer repeated 12× costs 12× in the full model). `--normalise-per-model` divides by the model's total occurrences, so each model contributes exactly its control weight. Use it if deep models dominate.
- `split.py split`: by model, stratified by family, seed 0, `val_fraction` 0.2.
  - In each family, `round(n × 0.2)` models go to val, always leaving at least 1 in train; single-model families stay in train.
  - Written once (`--force` to redo).
  - `split.py bench-json --times all_units.json` → `base_train.json` / `base_val.json`; units whose baseline failed are listed and left out.
- `model_survey/check_suitability.py` (**outside the repo**, changed as requested): new `--weights` option (JSON `{model: weight}`).
  - Weights the candidate rows in the category shares and the shape quantiles.
  - Adds section 4b: matmul N/K and conv C/F median and p90, candidates vs eval.
  - Without `--weights` the output is unchanged: weight 1 and strict `>` reproduce the old `floor(p·n)` index.
- sbatch (`scripts/fusion_data/sbatch/`), all with `--partition=compute --constraint=bergamo`:
  - `export.sbatch`, `validate.sbatch`, `profile.sbatch`, `units.sbatch`;
  - `baselines_units.sbatch`: 64 cores, unfused units → `all_units.json`;
  - `baselines_eval.sbatch`: runnable eval models → `eval_fullmodel.json`;
  - `compat_test.sbatch`, `build_torch_mlir.sbatch`;
  - `_env.sh` reproduces `train.sh`'s environment (.env, MLIR-ENV, PYTHONPATH, no thread overrides); `_export_env.sh` sets up the export env.
- **Baselines use `scripts/baseline/baseline_fullmodels.py`, not `get_base.py`.** The V5.2 design doc records that the legacy `get_base.py` calls the pre-V5 string-based `execute_code` API and cannot be used with v5 (v6 is the same API). `baseline_fullmodels.py` handles any `--models-dir/--pattern`, adapts the ABI, and resumes.
- I used one small Python snippet to insert the `link-runnable` subcommand into `eval_set_tools.py`, before your "do not run anything" message. After that message, every file was written or edited with the editor only.

## Part 5 — training adjustments
5.1 **Sampling.**
- Before: v5/v6 draw the iteration's benchmarks uniformly without replacement, with `torch.randperm(len(data))[:cfg.bench_count]` (+ repeat padding). This happens at `ppo.py:126` (`collect_trajectory`) and `distributed.py:124` (`collect_distributed_trajectory`). `evaluate_benchmarks` (`ppo.py:341`) runs every benchmark.
- Added `fusion/sampling.py`: `sample_bench_indices(data, count)`.
  - With `fusion.sampling_weights` empty (default) it runs **exactly** the v5 expression.
  - With a path it uses `torch.multinomial(weights, count, replacement=False)`. Every loaded benchmark must have a weight > 0, otherwise it raises (never a silent uniform fallback).
- Changed `ppo.py` and `distributed.py` (existing v6 files): the 3 `randperm` lines → `sample_bench_indices(data, cfg.bench_count)`, plus the import. `distributed.py` no longer uses `torch`, so its `import torch` was removed.
- Changed `fusion/config.py`: new field `sampling_weights: str = ''`. It is not in `MODULE_FLAGS`, because it is read at sampling time, not at load.
- Added `fusion/tests/test_sampling.py`: off == v5 under the same seed (bench_count below, above and equal to the dataset size); on draws by weight, without replacement and padded; a missing weight raises. **Not run.**
- In distributed mode, Dask workers rebuild `Benchmarks()` and so apply the same fusion hook. Indices therefore refer to the same list on the driver and the workers.

5.2 **Hardware.**
- The existing training and eval job scripts (`scripts/train/train.sh`, `scripts/eval/eval.sh`, `scripts/eval/eval_fullmodel.sh`) already request `--partition=compute --constraint=bergamo`.
- Per `docs/hpc/HPC_HARDWARE.md`, Bergamo nodes are **AMD EPYC 9754 (Zen 4c)**, not EPYC 7742. So the configs and jobs assume Bergamo, and every new sbatch script requests the same.
- The fusion configs are single-node (no `dask_node_count`). As `train.sh`'s header says, they must be submitted with `--cpus-per-task=64 --mem=16G`. `baselines_units.sbatch` uses the same 64 cores.
- **Remark:** training executes several benchmarks concurrently (thread pool), while baselines run one at a time. This is the same as v5's existing practice, not a new difference.

5.3 **Tile guard L2.**
- `tile_guard_params.l2_bytes` stays **1048576 (1 MiB per core)**, which is correct for the Bergamo EPYC 9754.
- Source: AMD EPYC 9754 specification (Zen 4c: 128 cores, 128 MB L2 in total, i.e. 1 MB per core); node type from `docs/hpc/HPC_HARDWARE.md` and the `train.sh` constraint.
- The 512 KiB value applies to the Jubail standard nodes (EPYC 7742, Zen 2). If you switch to those, set `l2_bytes: 524288` in `D_plus*.json` **and** change the constraint in every sbatch and training submission.

5.4 **Checkpoints.** Every row (A included) uses limits of 12, so the observation and network shapes differ from the v5 checkpoints (limits of 7). **Every row trains from scratch, and no v5 checkpoint is loaded:** never pass `--resume` with a v5 run.

5.5 **Configs.**
- Units rows (A′, B, C, D, D+):
  - `benchmarks_folder_path: data/fusion_units/units`;
  - `json_file: …/base_train.json`, `eval_json_file: …/base_val.json`;
  - `fusion.sampling_weights: data/fusion_units/sampling_weights.json`.
- A stays on `data/ops_and_blocks` with uniform sampling.
- Eval twins:
  - the same `fusion` section;
  - `eval_json_file: …/eval_fullmodel.json`;
  - `results_dir: <training results_dir>_fullmodel_eval`. This is a change: `eval_fullmodel.sh` reads the checkpoints via `--checkpoints-dir` and writes into its own `results_dir`, so full-model results never overwrite the unit-validation `eval/` files.
- `test_configs.py` was rewritten for all of the above, plus the eval set (18 files, 13 runnable) and a check that the generated eval JSON only has runnable models. **Not run.**

Remarks:
- **Nothing in Parts 2–5 was executed.** You asked me not to run anything, so the new and changed tests were not run either. The first commands to run are `py_compile` and `pytest` (see the command list).
- **Eval set in practice:** per the V5.2 design doc, bert, bart and vit_b_16 from `data/full_models` fail raw execution ("op was not bufferized"). Their eval baselines will likely be -1, which leaves 10 evaluated models until those three are re-lowered (`data/full_models/relowered` is missing on this machine).
- `data/full_models` must exist relative to the repo root: a symlink to `/scratch/mh8826/data/full_models` (see the command list).
- Added `scripts/fusion_data/sbatch/config_b.sbatch` (row B: prefuse every unit into `data/fusion_units/units_prefused/`, then time them). Its times go to a **separate** folder `baselines/mlir_prefused/`, because `baseline_fullmodels.py` caches times by benchmark name in `<output dir>/.baseline_exec_cache.json`, and prefused units keep the unfused names.
- Added `scripts/fusion_data/COMMANDS.md`: the ordered commands from this state to training and evaluating every row.
- `fusion/README.md` updated: switches (`sampling_weights`, `units.eval_set`), eval set, sampling hook, eval via `eval_fullmodel.sh`, baselines via `baseline_fullmodels.py`, from-scratch training, Bergamo hardware, and the pointer to the data pipeline.
- **Unverified items to watch on first use:**
  - the `create-env` package versions;
  - the torch-mlir cmake flags for an out-of-tree build at that commit (compare with its `docs/development.md`);
  - resnet26's repo;
  - whether every candidate exports without `tm_tensor` (compat test, then `validation.json`).

## Part 2 follow-up — torch-mlir commit choice (2026-11-24)
- `find-commit` result: **no torch-mlir commit pins `cd708029` exactly.** The LLVM bumps around it, from read-only `git log` in `/scratch/mh8826/torch-mlir`:
  - `0eba539e` (2024-11-15) → llvm `813f7c3`, about 2 months **before** `cd708029` (2025-01-14);
  - `12250739` (2025-01-28) → llvm `e2402615`, about 2 weeks **after**;
  - `16903201` (2025-01-31) → llvm `5d6d982d`.
- **Choice: `TORCH_MLIR_COMMIT=12250739bfe85b702f9503cad45c2e535ea8eb18`**, the closest. Option B builds this torch-mlir source against *our* LLVM build, so the exporter's MLIR is still exactly `cd708029`. The only risk is compile errors where MLIR's C++ API changed between 01-14 and 01-27. Fallback: `0eba539e`.
- Your first build job (19538715) was submitted with an empty `TORCH_MLIR_COMMIT` (the grep found no exact match), so it was cancelled or fails immediately.
- Fixed `setup_export_env.sh find-commit`: `--format=%cs` → `%ci` (the cluster's git is too old for `%cs`, which printed literally).
- **Incident:** two build jobs (19538745 at `12250739`, 19538746 at `0eba539e`) ran at the same time in the same `/scratch/mh8826/torch-mlir` tree and `build/`. The second `git checkout` switched the source under the first job, and an `rm -rf build` ran while it was compiling. Both builds are invalid; they must be cancelled, `build/` removed, and one build resubmitted at `12250739`.
- Added to `setup_export_env.sh build`: an exclusive `flock` on `/scratch/mh8826/torch-mlir.build.lock`. A second concurrent build now exits immediately with a message, instead of corrupting the tree.
- **Build at `12250739` failed** (job 19538906, after 4 min). It uses MLIR APIs newer than our LLVM: `applyPatternsGreedily` (TMTensor/ConvertToLoops.cpp), `mlir/Dialect/Quant/IR/QuantTypes.h` (TorchToTosa), `linalg::Conv2DNchwFchwQOp` (TorchToLinalg/Linear.cpp). So torch-mlir must be one written for an LLVM **older** than `cd708029`.
- **New choice: `af8514c92cf1ce34d450c299a644fd4369590eec`** (2025-01-27), the parent of the 01-28 bump. It still pins llvm `813f7c3` (November API), so it compiles against older APIs, which our January LLVM still provides. It also has two more months of torch-mlir fixes than the November fallback `0eba539e`. Its `pytorch-requirements.txt` pins `torch==2.7.0.dev20250120`; the env has stable torch 2.6.0 (2025-01-29), which is close. `0eba539e` remains the last resort.

## Row A baselines (2026-11-24)
- **Incident:** training A (job 19538691) failed after 17 s with `FileNotFoundError: results/ops_and_blocks_results/baselines/mlir/base_train.json`.
  - No `ops_and_blocks` baseline JSON exists anywhere on this machine: `results/` holds only `gap1`, and the experiment-02 raw results were deleted on 2026-11-14.
  - `config/v5/v5_single_node.json` points at the same missing files. I copied that path into A without checking that it exists.
- **Finding:** all 6,693 blocks of `data/ops_and_blocks` come from the 18 eval models (albert 884, bert 733, llama3_2_1b 708, efficientnet_b0 616, whisper_base 437, t5 411, gpt2 410, bart 377, distilbert 371, mobilenet_v3_small 342, resnet50 286, resnext50 254, convnext_tiny 205, gin 35, yolov8m 11+, vgg16 9, gat 5). The other 1,400 files are single ops.
- **User decision:** keep A as the **status quo** (all 8,093 files) and measure its baselines. The eval-model overlap is a documented advantage of A in the A vs A′ comparison.
- **Added** `scripts/fusion_data/sbatch/baselines_blocks.sbatch`:
  - runs `baseline_fullmodels.py` on `data/ops_and_blocks` (`*.mlir`, timeout 300 s per file) on compute + bergamo with 64 cores;
  - writes `results/ops_and_blocks_results/baselines/mlir/base.json`; resumable.
- **Then** the repo's own `scripts/data/split_json.py --input …/base.json` writes `base_train.json` / `base_eval.json` next to it: stratified by model family, eval ratio 0.2, seed 42. Those are exactly the paths `config/fusion/A.json` reads, so the config is unchanged.
- `COMMANDS.md` updated: A is no longer "ready after step 0", it needs these two steps first.

## CPU parallelism enforced for every job (2026-11-24, user request)
Findings (read-only):
- **Single-node training** (Dask disabled) executes the iteration's programs in `ThreadPoolExecutor(max_workers=SLURM_CPUS_PER_TASK)` (`utils/dask_manager.py:207`), i.e. **64 programs at once**.
  - Each program's parallel loops become OpenMP (`scf-forall-to-parallel`, `convert-scf-to-openmp` in `execution.py`), whose thread count is `OMP_NUM_THREADS`.
  - Distributed mode sets `OMP_NUM_THREADS=12` per Dask worker (`dask_manager.py:126`); single-node mode sets nothing.
- **Risk 1:** `scripts/analysis/gap1/env.sh`, which my earlier command list told you to source, exports `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. `sbatch` exports the submitting shell's environment by default, and `train.sh` does not reset these. So jobs submitted from that shell run every kernel on one thread, and the agent's parallelization (TP) never actually runs in parallel. Experiment 02 ran at 1 thread for this kind of reason.
- **Risk 2:** with nothing set, 64 concurrent programs × 64 OpenMP threads ≈ 4,096 threads on 64 cores. Training times are heavily oversubscribed, while baselines (one program at a time) are not, so the reward is biased.

Changes:
- `rl_autoschedular_v6/utils/dask_manager.py` (existing v6 file): the thread-pool size reads `EXEC_WORKERS` first, then `SLURM_CPUS_PER_TASK`. **Unset = unchanged** (v5 behaviour).
- Added `scripts/fusion_data/sbatch/_threads.sh`: one explicit split for every job that executes MLIR code, **`EXEC_WORKERS=8` × `OMP_NUM_THREADS=8` = 64 cores** (training allocation). MKL/OpenBLAS are set to 8 as well. Override with `FUSION_EXEC_WORKERS` / `FUSION_OMP_THREADS`.
  - `_env.sh` sources it, so every MLIR-ENV sbatch (validate, units, profile, all baselines, config B) sets it explicitly instead of inheriting. Baselines therefore time each program with the same 8 threads that training uses.
  - The job log's first line prints `exec_workers=… omp_threads=…`.
- `_export_env.sh`: export jobs use all allocated cores (`OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`; torch export is CPU-threaded).
- Added `scripts/fusion_data/submit.sh`: wrapper for `train.sh` / `eval.sh` / `eval_fullmodel.sh`, which are not edited.
  - Submits with `--export=ALL,EXEC_WORKERS=…,OMP_NUM_THREADS=…,MKL…,OPENBLAS…`, overriding the shell's values.
  - Refuses to submit unless `EXEC_WORKERS × OMP_NUM_THREADS == --cpus-per-task`.
- `COMMANDS.md`: every training and eval command now goes through `submit.sh`, plus a "CPU usage" section with how to verify (`seff <jobid>`, job log line).

Remarks:
- The 8 × 8 split is a choice: it trades some within-iteration concurrency (64 → 8 programs at once) for real multi-threaded kernels and no oversubscription. Iterations may take longer, but the times are meaningful and comparable with the baselines.
- Baseline jobs keep `--cpus-per-task=64` while timing one program at a time with 8 threads. The extra cores keep neighbours away from the timed program.
- `--jobs 4` full-model evaluation = 4 programs × 8 threads = 32 of the 64 cores.
- All rows use the same split, so the ablation comparisons are unaffected.
- Nothing was run.

## Overnight session (2026-11-24, ~04:00–04:40): git, root causes, one-command pipeline
Git:
- Committed everything on a new branch **`fusion-v6`** (from `gap1`; `gap1` and `main` untouched): commit `e492ebc`, 190 files (code, configs, reports, control file).
- `.gitignore`: `/experiments/*/results/` (340 MB / 16k files of raw outputs) and `/experiments/**/__pycache__/` are now ignored. `data/.gitignore` tracks `fusion_units/models.json`. Your earlier uncommitted `.gitignore` additions (checkpoints, logs, weights) were committed with it. No secrets or files > 0.5 MB were added.
- **Push failed:** no GitHub credentials in this shell (HTTPS remote, no `gh`, SSH key not registered). Run `git push -u origin fusion-v6` from your VS Code terminal.

Root cause 1, `.env` broke every MLIR job (baselines, training, eval):
- `.env` used bash's `${VAR:+:${VAR}}` syntax on the `PYTHONPATH` and `LD_LIBRARY_PATH` lines. python-dotenv does not support it.
- `baseline_fullmodels.py`, `train.py` and `eval.py` call `load_dotenv(override=True)`, which turned GCC 12's lib path into `…/gcc/12.2.0/lib64${LD_LIBRARY_PATH` and `…/lib64}` (both invalid).
- Every MLIR subprocess then loaded `/lib64/libstdc++.so.6`: `GLIBCXX_3.4.29 not found`. Row A's first baseline job (19539158) failed on all files this way.
- **Fixed `.env`** (backup `.env.bak-2026-11-24`): both lines now use plain `…:${VAR}`, which bash and python-dotenv both understand. Verified: after `load_dotenv`, no mangled entries, and a child process imports MLIR and v6.

Root cause 2, why both torch-mlir builds failed:
- **The agent's LLVM is release 19.1.7** (`release/19.x` @ `cd708029`, tag `llvmorg-19.1.7`; the clone is shallow). It branched from mainline in late July 2024; "2025-01-14" is only the date of a branch backport.
- The January 2025 torch-mlir commits need newer MLIR (`mlir/Dialect/Quant/IR/…`, `Conv2DNchwFchwQOp`, `applyPatternsGreedily`); our LLVM has the old `mlir/Dialect/Quant/QuantTypes.h`.
- **New torch-mlir: `43506726853b35ae9c253aa1d1c61b76ad9b4c13`** (2024-08-07), the last commit before torch-mlir's first LLVM bump after the 19.x branch point.
  - It pins llvm `d16b21b` (June 2024), uses the old Quant header, and uses neither `Conv2DNchwFchwQOp` nor `applyPatternsGreedily`.
  - It pins `torch==2.5.0.dev20240804`, so the `mlir-export` env now has **torch 2.5.0 / torchvision 0.20.0** (installed tonight; MLIR-ENV untouched).
- `setup_export_env.sh` rewritten accordingly: header explains the choice, `create-env` pins torch 2.5.0, `find-commit` lists the 2024 bumps.
  - Each commit now builds into its own `build-<sha8>`, and a successful build writes `/scratch/mh8826/torch-mlir.active_build`, which `check` and `_export_env.sh` read. No build folder ever needs deleting.

Other:
- `model_survey/download_models.py` run on the control file (login, network, config.json only): **all 39 repos OK**, including the proposed resnet26 (`timm/resnet26.bt_in1k`) and bert_L4_H768.
- `data/full_models` symlink → `/scratch/mh8826/data/full_models` created; the eval configs need it.
- `experiments.json`: registered `fusion_A`, `fusion_A_prime`, `fusion_B`, `fusion_C`, `fusion_D`, `fusion_D_plus` (state `pending`).
- Tests before the commit: 133 passed, 1 skipped (`eval_fullmodel.json` not generated yet); `py_compile` and `bash -n` clean.

**What I could not do:** the permission classifier denied `sbatch` and `scancel` for me, so I could not cancel the broken job, rebuild torch-mlir, or launch anything. Instead I added a one-command driver:
- `scripts/fusion_data/run_pipeline.sh` submits everything with Slurm dependencies (`afterok`):
  - row A: `baselines_blocks` → `split_json` → train A;
  - units: build → export → validate → (profile) → units → split → `baselines_units` → `bench-json` → train A′, C, D, D+ (+ `config_b` after units);
  - `baselines_eval`, independent.
  - It waits (`afterany`) for any `fd-base-blocks` job still running. All job ids go to `/scratch/mh8826/fusion_runs/pipeline_jobs.txt`.
- `scripts/fusion_data/sbatch/step.sbatch`: a light dependent step (split, bench-json, split_json) in the job environment.
- `scripts/fusion_data/status.sh`: read-only progress (job states, data counts, baseline ok/failed, checkpoints per row).
- `submit.sh`: its info message now goes to stderr, so `--parsable` ids can be captured.
- **Outputs for plotting/comparison** (unchanged code): each row's `results_dir` (`/scratch/mh8826/fusion_runs/fusion_<row>_agent`) holds `train_results.json`, `logs/` (per-iteration metrics, exec cache) and `models/model_<step>.pt` every 50 iterations. This is the layout `scripts/utils/report_training.py` and the plotting skill read.

## Pipeline launch and pybind11 fix (2026-11-24 evening)
- The user ran `run_pipeline.sh` at 21:24 (job ids in `/scratch/mh8826/fusion_runs/pipeline_jobs.txt`). The jobs survived an HPC disconnect: Slurm jobs do not depend on the login session.
- **torch-mlir build 19561729 failed after 3 min**, but much later than before. All torch-mlir C++ compiled against LLVM 19 (284/321 steps), which confirms the `43506726` choice.
  - It failed only in the Python bindings (`IRCore.cpp`): `pybind11.h: static assertion failed: def_property family does not currently support keep_alive`.
  - Cause: `create-env` installed the latest pybind11 (3.1.0), while LLVM 19's `mlir/python/requirements.txt` requires `pybind11>=2.9.0, <=2.10.3` (MLIR-ENV has 2.11.1).
  - Fix: `pybind11==2.10.3` installed in `mlir-export` (MLIR-ENV untouched) and pinned in `setup_export_env.sh create-env`.
- Consequence: the units-chain jobs (compat 19561730, export 19561731, then validate, profile, units, splits, baselines_units, bench-json, train A′/C/D/D+, config B: 19561732–42) can never run. They wait on the failed build (`DependencyNeverSatisfied`).
- Unaffected and still pending: row A (19561726 baselines_blocks, 19561727 split_blocks, 19561728 train A) and 19561744 baselines_eval.
- `run_pipeline.sh --units-only`: new mode that resubmits only the units chain (build → … → train A′, C, D, D+, config B) without duplicating row A or the eval baselines.

## Second units-chain run (2026-11-24 ~22:30): build OK, export blocked by my script name
- **torch-mlir build 19562290: COMPLETED** (1 min 46, an incremental build after the pybind11 fix). `/scratch/mh8826/torch-mlir.active_build` → `build-43506726`. The converter now works against LLVM 19.1.7; no more torch-mlir work is needed.
- **Export (19562292) and compat test (19562291): 0 models converted.** Every model failed with `AttributeError: module 'profile' has no attribute 'run'`, or `Failed to import transformers…timm_wrapper` caused by the same thing.
  - Cause (my bug): `scripts/fusion_data/profile.py` shadowed Python's standard `profile` module. Python puts the script's own folder first on `sys.path`, so torch/transformers' `import cProfile` (which imports `profile`) loaded my file.
  - `validate` then found nothing, and `units` (19562295) and `profile` (19562294) failed with `FileNotFoundError: data/fusion_units/validation.json`. Everything downstream is stuck with `DependencyNeverSatisfied`.
- **Fix:** renamed it to `scripts/fusion_data/linalg_profile.py` (`git mv`) and updated `sbatch/profile.sbatch`. Checked: no other file in `scripts/fusion_data/` shares a name with a standard-library module.
- Row A (19561726–28) and eval baselines (19561744) are unaffected and still queued. The "0 ok, 8093 failed" blocks baselines are from the old broken run (19539158); the queued job retries all of them with the fixed `.env`.

## Third units-chain run (2026-11-24 ~22:55): export blocked by GCC 12 libstdc++
- Build 19562701 OK. Export 19562705: 0/37 again. The `profile` clash is gone; now `import torch_mlir` failed with `/lib64/libstdc++.so.6: GLIBCXX_3.4.29 not found`.
  - torch-mlir was compiled with GCC 12, but `_export_env.sh` never put GCC 12's `lib64` on `LD_LIBRARY_PATH` (unlike the MLIR-ENV jobs, which read `.env`).
- **Fix:** `_export_env.sh` adds only `$GCC_ROOT/lib64` (read from `.env`), not the rest of `.env`, which belongs to MLIR-ENV. `setup_export_env.sh check` does the same.
- **End-to-end check on the login node** (one small model, outputs in the session scratchpad only):
  - `export.py --only resnet18` with the fixed env → `resnet18_linalg.mlir` (90 KB, 0 `tm_tensor` lines);
  - `validate.py` (MLIR-ENV, LLVM 19.1.7 bindings + AstDumper + v6 features) → `[ok] resnet18: form=pretty ops=200`, no op dropped by the limits;
  - `units.analyze` → 11 segments.
  - So the full chain converter → agent works.

## Run outputs outside the git repo (2026-11-24, user request)
- **How checkpoints are saved (unchanged code):**
  - `scripts/train/train.py` saves `{model, optimizer, step}` to `<results_dir>/models/model_<step>.pt` every 50 iterations (`step % 50 == 0`, line 206), plus a final checkpoint (line 224).
  - `utils/file_logger.py` derives `models/`, `logs/` (incl. `exec_data.json`) and `train_results.json` from `results_dir`, and creates them with `os.makedirs`. Absolute paths work as they are.
- **Changed:** all 12 `config/fusion/*.json`, `results_dir` → `/scratch/mh8826/fusion_runs/fusion_<row>_agent` (training) and `/scratch/mh8826/fusion_runs/fusion_<row>_agent_fullmodel_eval` (eval twins). This used to be `results/...` inside `MLIR-RL/`.
  - Checkpoints, logs, exec caches and eval outputs of every row are now outside the git repo.
  - Each row keeps its own folder (exec-cache isolation is unchanged).
- **Not moved:** the baseline JSONs (`results/fusion_units_results/baselines/...`) and the unit data (`data/fusion_units/...`) stay at their repo-relative paths. `results/` and `data/` are git-ignored there, and the pipeline scripts use those paths.
- `fusion/tests/test_configs.py`: new `test_results_outside_the_repo`, which checks that every config's `results_dir` is under `/scratch/mh8826/fusion_runs/` and not inside the repo. **Not run.**
- `scripts/fusion_data/COMMANDS.md`: eval `--checkpoints-dir` paths and the `experiments.json` example updated, and a note on checkpoints and `--resume` added.

### 2026-11-25 — run_pipeline.sh --after-data
- New mode `--after-data`: resubmits only baselines (blocks, units, eval), config B, split_blocks, bench_json and
  training A, A', C, D, D+. Skips the already-completed data steps (torch-mlir build, export, validate, profile, units, split).
- Shorter `--time` on the command line (overrides `#SBATCH`) so Slurm can backfill: blocks 2 d, units 12 h, config B 12 h, eval 1 d.
- Remark: baselines are resumable (`baseline_fullmodels.py` skips entries > 0), so a job that hits its limit can be resubmitted.

### 2026-11-25 — remark: export results (job 19563055)
- 37 enabled models, 34 raw files. No file: convnext_atto, edgenext_xx_small (TorchFX -> Torch backend lowering failed),
  wav2vec2_base_L2 (Torch backend -> linalg lowering failed).
- levit_128s: `export_log.json` says failed but the pretty file exists and validated ok (819 ops, 25 units). The pretty
  export succeeded; only the second, generic-form export raised GuardOnDataDependentSymNode, which overwrote the log
  entry. The file is complete (written before the exception); it is kept.
- Validation rejected 3: swin_tiny (dynamic dims), distilroberta and opt_125m (tm_tensor ops). 31 models -> 590 units.

### 2026-11-26 — full-model eval comparison (latest checkpoints vs v5_model_1200)
- `scripts/fusion_data/eval_compare.sh`: picks the latest checkpoint of A, A', C, D, D+ at submission time, plus
  `/scratch/mh8826/checkpoints/v5_model_1200.pt` (symlinked as `fusion_runs/v5_model_1200/models/model_1200.pt`;
  original untouched), submits one eval job per agent and a plot job after all of them (afterany).
- `scripts/fusion_data/sbatch/eval_fullmodel.sbatch`: same logic as `scripts/eval/eval_fullmodel.sh` (one eval.py
  per model, private config/results_dir, merge) but with `_env.sh`. Remark: `eval_fullmodel.sh` cannot run here:
  it does `source "$CONDA_ENV"` (a directory for us) and writes logs to `/scratch/mb10856/...`. Not modified.
- `config/fusion/v5_1200_eval_fullmodel.json`: copy of `A_eval_fullmodel.json` (v6, fusion off = v5; same network
  shapes as v5_model_1200, checked: 111 tensors, identical) with its own `results_dir`.
- `scripts/fusion_data/plot_eval_compare.py`: speedup = baseline / agent time per model; heatmap + geomean on the
  models every agent ran; `comparison.{png,md,csv}`. Failures are shown, not dropped.
- 2026-11-26 (later): `eval_compare.sh --v5-from <OUT dir> [<job id>]` reuses an earlier v5_model_1200 result (the plot job
  waits for that job id). `eval_fullmodel.sbatch` FAIL lines now report eval.py's real exit code (was always 0).
- Remark: evals were stuck on `AssocGrpCpuLimit` (group CPU cap: 5 trainings x 64 cores + 1 eval fill it), and
  `eval_compare.sh` had been run twice (03:01 and 14:11), doubling the eval jobs.
- 2026-11-26: `scripts/fusion_data/plot_training.py`: training curves of A-D+ from logs (no job) -> `fusion_runs/plots/training_curves.png`, `training_summary.md`. Partial v5_model_1200 eval table (11 finished models of job 19577368, from its tmp dir) -> `fusion_runs/plots/v5_model_1200_partial_eval.md`.
- 2026-11-26: llama3_2_1b left out of the full-model evals. Its greedy eval applies ~1000 ops' transformations,
  each in a fresh process that loads the whole model (~19 GB RSS); after 3+ h a TiledFusion on operation_984 hit an
  MLIR assertion (PybindAdaptors.h:623) and the job was still running at 8.5 h. Because eval_fullmodel.sbatch writes
  its result only at the end, a timeout on llama would lose the other 12 models. All `*_eval_fullmodel.json` configs
  now use `eval_fullmodel_no_llama.json` (12 models); revert by pointing them back at `eval_fullmodel.json`.
  v5_model_1200 eval (job 19577368, cancelled by the user while on llama) saved from its tmp dir to
  `fusion_runs/eval_compare/20261126_0301/v5_model_1200.json` (+ logs).
- 2026-11-26: `eval_compare.sh` takes EVAL_CPUS / EVAL_MEM / EVAL_TIME (defaults 64 / 256G / 1 day) so eval jobs
  fit partly used nodes. Each model still runs with OMP 8 threads (same as the baselines); EVAL_CPUS/8 models
  run in parallel. Planned use: 16 cores, 48G, 14 h (2 models at a time; v5 eval per-model times: bert ~6.4 h,
  vit_b_16 ~5 h, bart ~2.5 h, the rest < 1.2 h each).

### 2026-11-26 — new row D_blocks (fusion + producer ranking on the blocks data)
- `config/fusion/D_blocks.json`: A.json (ops_and_blocks data, blocks baselines) + `prefuse` + `producer_select`,
  `nb_iterations` 1200 (same as the reference checkpoint v5_model_1200), results in
  `/scratch/mh8826/fusion_runs/fusion_D_blocks_agent` (checkpoints every 50 it. + model_1200.pt at the end).
  No sampling_weights (those are for units). `D_blocks_eval_fullmodel.json`: its full-model eval twin.
- Smoke test (login/interactive node, 40 random blocks): loads in 9 s, prefuse ops 181 -> 106,
  producer selection changed 0 initial producers (blocks rarely have several data producers; the full count is
  printed at job start). Expected start-up for 8,093 blocks: ~30 min.
- User cancelled all pending evals (eval_compare 20261126_1631) at the same time.
