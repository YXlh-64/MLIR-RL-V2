# Gap 1 analysis scripts

Checks for Gap 1: each consumer sees and can fuse only one producer (`env.py:188`, `producers[-1]`), and training blocks are linear 5-op chains (`data_utils/extract/extract_blocks.py`). Each script answers one question (see its docstring). Each one writes CSV files and a `*_summary.md` to `--out`; `plot_results.py` turns them into PNGs.

Recommended order: **B1 → A1, A2, A5 → B3, B2 → B4, B5** (A3, A4, A6 and A7 can run any time).

## Setup

Run everything from the repo root (`cd /scratch/mh8826/MLIR-RL`).

```bash
FULL=/scratch/mh8826/data/full_models        # folder with <model>_linalg.mlir (not in this checkout)
BLOCKS=data/ops_and_blocks            # block files <model>_block_<N>.mlir
R=results/gap1                        # all outputs go here (gitignored)
G=scripts/analysis/gap1
mkdir -p $R/logs
```

- **A scripts** need only the Python standard library. Run them directly on a login node with any `python3`.
- **B scripts** and **plots** need the project conda env and MLIR bindings: `source $G/env.sh`. This is the same setup as experiment 02: modules, `.env`, `$CONDA_ENV`, `PYTHONPATH` and `CONFIG_FILE_PATH`.
- **B scripts compile and run in-process**, so an MLIR crash kills the script. Submit them with `sbatch $G/run.sbatch <script.py> [args]`, never on the login node. Logs go to `$R/logs/`.

Arguments named `models`, `blocks` and `programs` accept files, directories (searched recursively) or quoted globs.

## Run everything (recommended)

```bash
# All A checks, then plots + results/gap1/REPORT_A.md (compute node; ~minutes, llama adds the most)
bash $G/run_all_a.sh                 # or: bash $G/run_all_a.sh --skip-llama

# All B checks as one Slurm job, then plots + results/gap1/REPORT_B.md
mkdir -p $R/logs
sbatch $G/run_all_b.sbatch            # B2 uses schedules generated from the v5_model_1200 checkpoint
# options: CKPT=/scratch/mh8826/checkpoints/<other>.pt  SCHED_BENCHES="$BLOCKS/bert_block_*.mlir"
#          EXP=results/<exp>_agent  (use that experiment's exec_data.json instead of the checkpoint)

# Re-make plots + reports any time from existing results
source $G/env.sh && python $G/plot_results.py --results $R --only a    # or --only b, or no flag for both
```

`REPORT_A.md` / `REPORT_B.md` put every check's summary table next to its plots; open them in the IDE's Markdown preview.

## Commands, one per script

### A: data structure (stdlib only, MLIR parsed as text)

```bash
# A1  ops with 0/1/2/>=3 producers, by op type
python3 $G/a1_producer_counts.py "$FULL/*_linalg.mlir" --out $R/a1

# A2  multi-producer ops that keep >=2 producers inside a block (blocks: .mlir files or *_blocks_manifest.json)
python3 $G/a2_multi_producers_in_blocks.py --models "$FULL/*_linalg.mlir" --blocks "$BLOCKS/*_block_*.mlir" --out $R/a2

# A3  number of blocks each model op appears in
python3 $G/a3_block_membership.py --models "$FULL/*_linalg.mlir" --blocks "$BLOCKS/*_block_*.mlir" --out $R/a3

# A4  max_paths cap hits and producers never reached
python3 $G/a4_path_cap.py "$FULL/*_linalg.mlir" --max-depth 64 --max-paths 5000 --out $R/a4

# A5  green / yellow / red edge census
python3 $G/a5_edge_classes.py "$FULL/*_linalg.mlir" --out $R/a5

# A6  block vs model shape drift
python3 $G/a6_shape_drift.py --models "$FULL/*_linalg.mlir" --blocks "$BLOCKS/*_block_*.mlir" --out $R/a6

# A7  merged green groups vs feature limits
python3 $G/a7_merged_group_limits.py "$FULL/*_linalg.mlir" --config config/v5/v5_mlir_rl_v1_paper.json --out $R/a7
```

### B: execution (MLIR bindings, no agent), via Slurm

```bash
# B1  timing noise (run this first among the B scripts; it sets the noise threshold)
sbatch $G/run.sbatch b1_timing_noise.py "$BLOCKS/resnet50_block_1*.mlir" "$BLOCKS/bert_block_1*.mlir" --runs 20 --repeats 3 --out $R/b1

# B2  TPF schedule correctness, in two steps:
#   1) schedules: run the trained checkpoint greedily on some blocks (the only step that uses the agent)
sbatch $G/run.sbatch gen_schedules.py "$BLOCKS/resnet50_block_*.mlir" "$BLOCKS/gpt2_block_*.mlir" --checkpoint /scratch/mh8826/checkpoints/v5_model_1200.pt --out $R/schedules/exec_data.json
#   2) replay the TPF ones and compare outputs (no agent). An experiment's logs/exec_data.json also works here.
sbatch $G/run.sbatch b2_tpf_correctness.py --exec-data $R/schedules/exec_data.json --benchmarks-dir $BLOCKS --per-bench 1 --out $R/b2

# B3  two producers fused into one consumer (toy diamond, no data needed)
sbatch $G/run.sbatch b3_diamond_fusion.py --shape 256 512 --tiles 32 64 --out $R/b3

# B4  elementwise pre-fusion alone: all edges, then green edges only
sbatch $G/run.sbatch b4_elementwise_prefusion.py "$BLOCKS/resnet50_block_*.mlir" --runs 5 --out $R/b4
sbatch $G/run.sbatch b4_elementwise_prefusion.py "$BLOCKS/resnet50_block_*.mlir" --green-only --runs 5 --out $R/b4_green

# B5  additivity: full model vs sum of blocks vs sum of single ops (static-shape models)
sbatch $G/run.sbatch b5_additivity.py "$FULL/resnet50_linalg.mlir" --runs 3 --out $R/b5
```

### Plots (after any of the above)

```bash
source $G/env.sh
python $G/plot_results.py --results $R          # PNGs go to $R/plots/
```

It plots whatever results exist and skips the rest. B3 has no plot (it is a single row; read `$R/b3/b3_summary.md`). For the green-only B4 run add `--b4 b4_green` (writes `b4_green_prefusion.png`).

| Plot | What it shows |
|---|---|
| `a1_producers.png` | per model, share of ops with 0 / 1 / 2 / ≥3 producers |
| `a2_kept_in_blocks.png` | per model, % of multi-producer ops that keep ≥2 producers in some block |
| `a3_membership.png` | histogram of block appearances per op; most repeated op types |
| `a4_path_cap.png` | per model, start ops hitting the cap / with unreached producers |
| `a5_edge_classes.png` | per model, green/yellow/red share; edges by class and reason |
| `a6_shape_drift.png` | per model, share of block ops with same / batch-only / other shape drift |
| `a7_group_limits.png` | % of merged groups over each limit; body ops the features ignore |
| `b1_noise.png` | distribution of run CV and of the 2-run-median spread (p95 marked) |
| `b2_tpf_correctness.png` | TPF schedules: outputs match / differ / could not run |
| `b4_prefusion.png` | speedup per program (log axis, wrong outputs in red); op count before/after |
| `b5_additivity.png` | per model, Σ ops / full and Σ blocks / full (1.0 = additive) |

## Assumptions about data layout

- **Full models** are `<model>_linalg.mlir` or `<model>.mlir` files. The model name is the basename without that suffix. These files are not in this repo checkout (see `experiments/03_full_models/PLAN.md` for where they live).
- **Blocks** are `<model>_block_<N>.mlir` files, as in `data/ops_and_blocks/` (a single flat folder). Their linalg ops carry the **model** tags (`tag = "operation_K"`), which link them to the model's ops. A2 and A3 also accept extractor manifests (`<model>_blocks_manifest.json`, read from `entries[].path_consumer_to_producer`). With a manifest, the only in-block edges are the consecutive pairs of the path.
- **Tags in full models** are recomputed the way the AST dumper does it: linalg ops in lexical order, ops with zero loops skipped, numbered `operation_0, 1, …`. Producers are direct SSA uses between those ops, so nothing is followed through `tensor.expand_shape` or similar. This matches what the extractor saw.
- **Text parsing** expects the pretty-printed form that `mlir-opt` writes. Named ops get their access pattern from a small table in `_mlir_text.py`. Any linalg op not in that table is reported as unparsed (A1, A5), and its edges count as yellow/`unparsed`.
- **Edge classes (A5, A7, B4 `--green-only`)**
  - Green: the producer is all-parallel, and the consumer reads the operand with a pure dim map that covers all of the consumer's parallel dims.
  - Red: the producer has a reduction, and the consumer broadcasts its result (a parallel dim is missing, or the map has a constant index).
  - Yellow: everything else, with a sub-reason (`init_operand`, `ew_producer_consumer_partial|halo`, `reduction_producer_consumer_full|halo`, `unparsed`).
- **A7** counts a merged group's loads as the distinct external input values, and its stores as the members whose result is used outside the group. "Unseen ops" are arith/math body ops other than `addf`, `subf`, `mulf`, `divf` and `exp`, which are the only ones the AST dumper counts.
- **B1** times with zero inputs, as v5 does. The "2-run median" is v5's reward measurement (`execution.py`, median of 2 invocations).
- **Schedules for B2** come from `gen_schedules.py`, which runs a trained checkpoint greedily (same rollout as `experiments/02_block_diagnosis/ablation.py`, read-only, no agent code changed). `v5_model_1200.pt` needs `config/v5/v5_mlir_rl_v1_paper.json` (`--config`, the default). It writes the v5 exec-cache layout `{bench: {"<op0 actions>|<op1 actions>|…": 0}}` (time unknown = 0). B2 also accepts a real `logs/exec_data.json` from training/eval; `eval_exec_times_*.json` holds only times, not schedules.
- **B2** Only keys containing `TPF(` are replayed. B2 rebuilds the actions through `Env` and `ActionSpace.action_from_str`, then applies them in v5 order: a transform that fails skips the rest of that op's actions, as `Env` does.
- **B2, B3, B4** check outputs using the same seeded inputs: floats drawn from uniform(0.5, 1.5), so sqrt, log and division stay finite, and integers set to 0. Comparison is `np.allclose(..., equal_nan=True)`.
- **B4 `--green-only`**: `linalg-fuse-elementwise-ops` has no per-edge option, and its default control fuses only single-use producers. So the script gives each producer of a non-green edge a temporary extra use (`"gap1.keep"`), runs the pass, then deletes those lines. Full models get `prepare()` first (timed `@main` ABI, elided weights set to zeros). The pass can then constant-fold those zero weights.
- **B5** builds single-op and block programs from the model with the extractor's own code (`_build_block_benchmark`), at model shapes (`--batch` only fills `?` dims). For blocks it uses a greedy cover with no shared ops, and ops left uncovered are counted as single ops. The full model must have static shapes to run.
