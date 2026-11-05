# Experiment 01 — CPU performance comparison

Experiment 01 evaluates the **same computation, weights and inputs** through:

1. **MLIR baseline:** V5 bufferization/lowering with LLVM optimization level 3, without a learned schedule.
2. **MLIR RL:** the same pipeline after applying the downloaded policy's schedule.
3. **PyTorch eager.**
4. **CPU TorchInductor**, using `torch.compile`.

This is checkpoint inference and benchmarking, not training. Methods use one CPU thread and the same CPU affinity within a workload. The principal measurement is median host-call latency after warmup.

Generated outputs live under this experiment's `results/` directory. Existing benchmark attempts are preserved and normally skipped on subsequent runs. Read each category's `REPORT.md` for its actual outcomes; the tables below describe the sampling plan.

## Start here: files and setup

| Enclosed file | Responsibility |
| --- | --- |
| [run_all.py](run_all.py) | Run **all sampled categories and full models**, sequentially in the foreground. |
| [run_all.sbatch](run_all.sbatch) | Submit that complete plan as one separately allocated Slurm batch job. |
| [run_suite.py](run_suite.py) | List samples or run **one dataset/category**. |
| [experiment.py](experiment.py) | Execute one workload: realize inputs, infer/apply the schedule, compile methods, check outputs and measure latency. |
| [plot_speedups.py](plot_speedups.py) | Plot a category's existing CSV; never starts benchmarks. Requires Matplotlib. |
| [run_full_models.sh](run_full_models.sh) | Run **one full model in the foreground**. It no longer submits Slurm jobs or arrays. |

The selected policy is [checkpoints/v5_model_1200.pt](../../checkpoints/v5_model_1200.pt), with [config/v5/v5_mlir_rl_v1_paper.json](../../config/v5/v5_mlir_rl_v1_paper.json). This is the matching 12-loop V5 architecture. The seven-action V5.3 checkpoint is not used because its matching action implementation has not been established here. `v5_single_node.json` does not match the downloaded V5 checkpoint.

**You do not need to write JSON to run a category.** The runner derives argument shapes and equivalent Torch computations from the selected MLIR forms, constructs each specification in memory, and saves one `manifest.json` inside the actual run. No dataset baseline-timing JSON, train/eval split JSON or pre-existing `inputs/experiment.json` is needed.

Use an existing interactive compute-node allocation. Set up a foreground terminal from the repository root:

```bash
cd /scratch/ab13817/Code/MLIR-RL-V2
set -a
source .env
set +a
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PWD:$PWD/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The local `.env` supplies the installed Python environment and LLVM build paths. That environment needs NumPy, PyTorch, MLIR Python bindings and the compiler used by TorchInductor.

**The foreground Python/bash examples remain attached to your terminal until they finish.** The driver waits for one isolated method worker before starting the next. Batch submission is a separate option described below. Progress names the active method and its output directory; compiler details go into `worker.log`.

Ctrl+C stops the active worker process group and prevents later workloads from starting. Partial diagnostics remain. Do not cancel the interactive Slurm allocation to stop an experiment.

## Run everything with one command

After the environment setup above, preview the entire plan:

```bash
python experiments/01_performance_diagnosis/run_all.py --dry-run
```

The preview lists commands, selected benchmark names and known unsupported groups. It writes nothing and performs no compilation or timing. Full-model weights/state are audited when actually executing their comparisons.

Run the complete plan in the foreground:

```bash
python -u experiments/01_performance_diagnosis/run_all.py
```

The launcher reuses `run_suite.py` and its existing sampler, measurements and output layout. It runs:

1. All five simple-operation categories from `mlir_rl_v1_paper`.
2. All six simple-operation categories from `ops_and_blocks`.
3. Every `.mlir` file in `data/full_models/`, one model at a time. GIN/GAT use all four methods; other weight-complete models use explicitly labelled **MLIR-only diagnostics**, because their Torch references are not implemented. Models with missing weights/state are refused and recorded as blocked.

The default simple-operation selection remains **33 benchmarks**, not every file in the datasets. Increasing `--samples` selects more eligible distinct signatures while retaining the resource bounds. QCD families are included in the plan/status table as blocked until valid input fixtures and independent references exist. The three legacy paper `model_*` files are also recorded as blocked because the automatic category runner does not implement their comparisons; they need custom manifests. No arbitrary multi-operation block comparisons are invented.

Each method, benchmark and category finishes before the next starts. Failed/blocked categories do not stop later categories. **Ctrl+C stops the active worker and the whole plan**, leaving later entries pending. Completed benchmark attempts are skipped by default, including failed attempts; add `--rerun` to retry them. A skipped prior failure is still reported as failed in the overall status.

Optional example with more samples and longer full-model limits:

```bash
python -u experiments/01_performance_diagnosis/run_all.py \
  --samples 5 --full-model-timeout-seconds 1800
```

Other options: `--seed`, `--max-input-mib`, `--max-work`, `--timeout-seconds` (simple-operation per-method cap), and `--rerun`. Defaults remain 1,200 seconds per method for simple operations and 600 seconds for full models. Changing limits does not fix missing weights or compiler failures.

Start with **`results/RUN_ALL.csv`** for overall progress. It updates before/after each category, lists selected benchmarks and the category report path relative to `results/`. Reports appear once benchmark results have been saved. This CSV is the only additional file generated by `run_all.py`; no new JSON configuration is needed. A later invocation refreshes it while retaining benchmark attempts. Batch submission also produces a Slurm log.

| Overall status | Meaning |
| --- | --- |
| `pending` / `running` | Not yet started / currently executing. |
| `passed` | All requested methods have measurements and passed independent cross-backend checks. |
| `unverified` | Measurements exist, but independent validation is unavailable, as in an MLIR-only diagnostic. |
| `failed` / `incomplete` | Saved method failures, failed correctness, missing requested methods or an unexpected driver failure. |
| `blocked` | Comparison cannot run with the supplied weights/state/reference or resource bounds. Inspect terminal diagnostics for preflight refusals. |
| `interrupted` | Stopped with Ctrl+C. Remaining entries were not started. |

Exit code 0 means every planned comparison passed; 1 means at least one entry is blocked, failed, incomplete or unverified; 130 means interrupted. **With the currently missing full-model/QCD inputs/references, a complete traversal will still return 1.** That accurately reports incomplete validation rather than claiming all models succeeded.

## Run everything through Slurm (sbatch)

From the repository root:

```bash
mkdir -p experiments/01_performance_diagnosis/results
sbatch experiments/01_performance_diagnosis/run_all.sbatch
```

The results directory must exist before submission because Slurm opens the log before the script starts. The script loads `.env`, selects its Python environment and sets MLIR/library paths itself; interactive environment setup is not required for this submission command. Submit with `sbatch`, not `bash`: Bash does not interpret `#SBATCH` resource requests.

Defaults are **one node, one task, 2 CPUs, 192 GB RAM, three days**, on `compute` with the `bergamo` node constraint. The task runs all categories/models sequentially. Each benchmark remains single-threaded; allocating 16 CPUs does not make this runner evaluate 16 models concurrently. The memory request gives large full models headroom over the earlier observed approximately 44 GB peak; it is not a measured guarantee that every remaining compiler stage fits. The three-day job limit covers the complete plan; individual methods retain their own shorter timeouts.

Slurm resource overrides go **before** the script filename. Python options go **after** it:

```bash
sbatch --cpus-per-task=16 --mem=256G --time=3-00:00:00 \
  experiments/01_performance_diagnosis/run_all.sbatch \
  --full-model-timeout-seconds 1800
```

This creates a new job allocation; it does not enlarge or borrow resources from an existing interactive job. Slurm selects an eligible node with sufficient available resources, possibly another node or spare resources on the same physical node. Both allocations count separately toward account limits. `--mem=192G` with one node requests that memory on one machine, not RAM pooled from multiple nodes. Plain Python, or `srun` within an existing allocation, uses that allocation's resources. Inside this batch file, `srun` uses the **new batch allocation**.

Slurm uses the resources requested in the submission; it does not infer the model's future peak memory or automatically expand an allocation when more is needed. Larger requirements should be requested before launching. See the official [sbatch resource documentation](https://slurm.schedmd.com/sbatch.html) and [job/step allocation guide](https://slurm.schedmd.com/job_launch.html).

The submission prints a job ID. Replace `JOB_ID` below with that number:

```bash
squeue -j JOB_ID
tail -f experiments/01_performance_diagnosis/results/slurm_JOB_ID.log
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

The log appears after the job starts. `results/RUN_ALL.csv` and category reports use the same locations as foreground runs. Because the complete runner returns 1 for blocked/unverified comparisons, Slurm can report `FAILED` with exit code `1:0` after the plan was fully traversed; inspect the CSV/log to distinguish that from a crash, time limit or OOM. Do not run two launchers against these same output directories simultaneously.

## Sampling decisions

Categories come from the actual single MLIR operation:

| Category | MLIR form | Matching Torch computation |
| --- | --- | --- |
| `add` | `linalg.add` | `A + B`; initial output is overwritten. |
| `relu` | The dataset's single `linalg.generic` zero/compare/select body | `relu(A)` |
| `matmul` | `linalg.matmul` | `A @ B + C_initial` |
| `batch_matmul` | `linalg.batch_matmul` | Batched `A @ B + C_initial` |
| `conv2d` | `linalg.conv_2d_nchw_fchw` | NCHW convolution with source stride/dilation, plus initial output. |
| `max_pool2d` | `linalg.pooling_nchw_max` | Pool with source window/stride/dilation, then maximum with initial output. |

Including the initial-output tensors is essential for equivalent accumulation/pooling semantics.

Defaults select **three distinct input/computation signatures per category**, uniformly without replacement, using seed **2026**. Eligibility is bounded to **32 MiB of argument tensors** and **100 million estimated arithmetic/reduction operations** per benchmark. Selection never depends on previously observed performance or success. Repeating the same options over the same dataset files gives the same sample.

These are manageable bounded samples, not coverage of the largest shapes or every originating model family. The work estimate does not include compilation or guarantee a successful run. Sampling uses the whole local dataset, **not a verified held-out split** from checkpoint training.

Preview categories and exact sampled stems without creating output files:

```bash
python experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --list
python experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --list
```

### Paper dataset: five categories

All sampled stems below identify `.mlir` files in `data/mlir_rl_v1_paper/`.

| Category | Available files | Default sampled stems |
| --- | ---: | --- |
| `add` | 281 | `add_15_56_15_28`, `add_150_7_15_14`, `add_112_28_14_14` |
| `relu` | 163 | `relu_256_64_7_7`, `relu_256_128_7_7`, `relu_128_256` |
| `matmul` | 202 | `matmul_128_1024_128`, `matmul_512_128_256`, `matmul_1024_128_256` |
| `conv2d` | 296 | `conv_2d_nchw_fchw_256_32_7_7_128_1_1_4_4`, `conv_2d_nchw_fchw_256_48_7_7_64_1_1_4_4`, `conv_2d_nchw_fchw_128_64_7_7_64_1_1_7_7` |
| `max_pool2d` | 260 | `pooling_nchw_max_128_48_15_15_7_5_5`, `pooling_nchw_max_128_96_15_15_3_7_7`, `pooling_nchw_max_128_96_7_7_3_3_3` |

Run **one line**, wait for the prompt to return, read its category report, then run the next:

```bash
python -u experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --category add
python -u experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --category relu
python -u experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --category matmul
python -u experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --category conv2d
python -u experiments/01_performance_diagnosis/run_suite.py --dataset mlir_rl_v1_paper --category max_pool2d
```

This selects 15 workloads, up to 60 method attempts. The local paper dataset has no simple `batch_matmul` category.

### Ops-and-blocks dataset: six categories

These stems identify single-operation files in `data/ops_and_blocks/`. Extracted multi-operation `*_block_*` files are outside this selection.

| Category | Available files | Default sampled stems |
| --- | ---: | --- |
| `add` | 281 | `add_28_15_112_7`, `add_240_15_7_14`, `add_240_14_7_28` |
| `relu` | 163 | `relu_128_1024`, `relu_128_3072`, `relu_256_96_7_7` |
| `matmul` | 226 | `vgg16_matmul_2`, `matmul_128_128_768`, `matmul_256_1024_128` |
| `batch_matmul` | 53 | `vit_b_16_batch_matmul_2`, `albert_batch_matmul_4`, `llama3_2_1b_batch_matmul_2` |
| `conv2d` | 410 | `efficientnet_b0_conv_2d_nchw_fchw_31`, `resnext50_conv_2d_nchw_fchw_1`, `efficientnet_b0_conv_2d_nchw_fchw_1` |
| `max_pool2d` | 267 | `pooling_nchw_max_128_384_7_7_3_3_3`, `pooling_nchw_max_128_128_7_7_3_3_3`, `pooling_nchw_max_128_64_7_7_3_3_3` |

Run each category synchronously:

```bash
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category add
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category relu
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category matmul
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category batch_matmul
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category conv2d
python -u experiments/01_performance_diagnosis/run_suite.py --dataset ops_and_blocks --category max_pool2d
```

This selects 18 workloads, up to 72 method attempts. Together the selections contain **33 workloads across 11 dataset/category combinations**. Some kernels occur in both datasets; these are separate dataset selections, not 33 necessarily unrelated computations.

### Paper QCD families: sampled, with references still needed

The paper dataset also has 152 multi-operation QCD kernels and three `model_*` full-model files. These are not additional simple operation types. `--list` also samples three stems per QCD family:

| Family | Sampled stems |
| --- | --- |
| `baryon` | `baryon_128_4`, `baryon_16_4`, `baryon_8_64` |
| `baryon_loop_lin` | `baryon_loop_lin_16_1`, `baryon_loop_lin_64_4`, `baryon_loop_lin_8_1` |
| `baryon_loop_lin_rev` | `baryon_loop_lin_rev_16_64`, `baryon_loop_lin_rev_64_1`, `baryon_loop_lin_rev_8_4` |
| `baryon_rewrite` | `baryon_rewrite_16_128`, `baryon_rewrite_32_8`, `baryon_rewrite_64_16` |
| `dibaryon_dibaryon` | `dibaryon_dibaryon_12`, `dibaryon_dibaryon_3`, `dibaryon_dibaryon_48` |
| `dibaryon_hexaquark` | `dibaryon_hexaquark_16`, `dibaryon_hexaquark_256`, `dibaryon_hexaquark_64` |
| `hexaquark_hexaquark` | `hexaquark_hexaquark_12_true`, `hexaquark_hexaquark_3_true`, `hexaquark_hexaquark_6_true` |

These are **catalogued, not ready for automatic four-way execution**: independent Torch contractions and appropriate domain-specific integer/index/coefficient input fixtures remain to be provided. They are not covered by the simple-operation resource bounds. Arbitrary generic MLIR cannot safely be replaced by a guessed Torch expression.

## Preview, adjust and retry

Preview one category; writes nothing and runs no compiler or benchmark:

```bash
python experiments/01_performance_diagnosis/run_suite.py \
  --dataset ops_and_blocks --category batch_matmul --dry-run
```

Optional overrides:

- `--samples 5`: select five rather than three eligible distinct signatures.
- `--seed 2027`: choose a different reproducible sample/input seed.
- `--only matmul_128_128_128`: choose a named benchmark within the selected category.
- `--max-input-mib 64 --max-work 200000000`: explicitly expand resource bounds, including for `--only`.
- `--timeout-seconds 1800`: change the per-method cap, including compilation/setup.
- `--rerun`: preserve a new attempt even if the benchmark already has a completed summary.

Simple-operation defaults are three correctness cases, ten warmups, 100 latency repeats and a **1,200-second cap per method**. The methods and then sampled benchmarks run sequentially. Errors/timeouts are recorded and remaining selected methods/workloads are attempted. A new failed attempt gives category exit code 1.

A completed `summary.json` causes that benchmark to be skipped next time, **including completed failed attempts**. Use `--rerun` to retry. Interrupted attempts without a summary do not count as completed. Reports select the latest completed attempt per benchmark. Changing seed/sample size adds results to the category; it does not erase earlier selections. Running without arguments prints help and starts nothing.

## Outputs: which files matter?

Example destination for the paper matmul category:

```text
results/mlir_rl_v1_paper/matmul/
├── REPORT.md                    Read first: timings, speedups, failures
├── comparison.csv               Measurements for analysis/plotting
├── SCHEDULES.md                  Readable policy choices and schedule links
└── artifacts/<benchmark>/<timestamp_id>/
    ├── manifest.json
    ├── checkpoint_config.json
    ├── environment.json
    ├── workload.mlir
    ├── inputs/case_N.npz
    ├── summary.json
    ├── summary.csv
    └── methods/{mlir_baseline,mlir_rl,torch_eager,torch_inductor}/
```

You normally need only **REPORT.md, comparison.csv and SCHEDULES.md**. Everything below `artifacts/` is grouped by benchmark and attempt for reproduction or diagnosis.

| Generated file | Why it exists; no manual preparation needed |
| --- | --- |
| `manifest.json` | Exact run specification: copied source, input shapes/types/generators, Torch graph, checkpoint/config, seed, CPU settings, repeats and tolerances. |
| `checkpoint_config.json` | Snapshot of the matching policy configuration. |
| `environment.json` | Host, allocation, compiler/Python details and source/input/checkpoint hashes. |
| `summary.json` / `summary.csv` | All method outcomes for one benchmark, written after its method attempts finish. |
| `methods/<method>/result.json` | One method's latency statistics/samples, compilation costs, correctness checks or failure reason. |
| `methods/<method>/worker.log` | Compiler/Python log; inspect for failures or long silent compilation. |
| `methods/mlir_rl/policy_config.json` | Runtime copy disabling distributed execution and redirecting policy logs into this attempt. Not another training configuration to tune. |
| `methods/mlir_rl/schedule.json` | All planned policy decisions, including intermediate pointer choices. `ready: true` means complete action parameters, not successful application. |
| `methods/mlir_rl/scheduled.mlir` | MLIR after the complete schedule applies successfully; can be absent after transform failure. |
| `inputs/case_N.npz` | Exact arrays shared by methods, including accumulation/initial-output arguments. |
| `methods/<method>/outputs_N.npz` | Actual outputs for every correctness case. |
| Other MLIR/LLVM/JIT files, `inductor_cache/`, `torch_debug/`, `tmp/` | Compiler artifacts and temporary work contained within the attempt. Can consume substantial disk space, especially for full models. |

GIN/GAT also produce a category-level `<model>_weights.npz` and a run copy `weights.npz`, containing the exact embedded weights for Torch reference execution.

Reports update after each completed benchmark. Refresh saved reports without running anything:

```bash
python experiments/01_performance_diagnosis/run_suite.py \
  --dataset mlir_rl_v1_paper --category matmul --report-only
```

Generate exactly three figures, one per data family:

```bash
for dataset in mlir_rl_v1_paper ops_and_blocks full_models; do
  python experiments/01_performance_diagnosis/plot_speedups.py \
    --results-dir "experiments/01_performance_diagnosis/results/$dataset"
done
```

Each command reads the family's category CSVs and writes **one file**, `results/<dataset>/speedup_comparison.png`. The three families are single operations (`mlir_rl_v1_paper`), operations and blocks (`ops_and_blocks`), and full models (`full_models`). Each figure combines all available validated comparisons and shows MLIR baseline (1×), MLIR RL, PyTorch eager and TorchInductor. Missing/failed methods are labelled N/A; workloads without a validated baseline/backend pair are excluded. No benchmarks are rerun. Matplotlib is required and is not installed in the MLIR environment; this machine has an isolated copy, usable by exporting `PYTHONPATH="/scratch/ab13817/codex/cache/experiment01-plotting${PYTHONPATH:+:$PYTHONPATH}"` before plotting. Other diagnostic figures are deferred until their comparison questions and measurements are agreed.

## How measurements and correctness work

Inputs use fixed seeded nonzero normal values (standard deviation 0.1 for floats), saved once for all methods. Each call restores inputs outside latency timing, including initial-output tensors. Compilation and warmup are excluded. Host-call latency includes dispatch/result allocation, but excludes input restoration, output copying and disposal. The separately saved MLIR `nanoTime` is not divided by Torch host time.

**Speedup = MLIR baseline median host latency / candidate median host latency.** Higher than 1× is faster. A validated speedup requires baseline and candidate to agree with independent eager Torch on every saved case: same output count/order, shape, dtype and finite values, with `rtol=1e-4`, `atol=1e-5` or exact integer equality. A measured timing with failed correctness is not a valid speedup. MLIR-only ratios remain unverified. Errors/timeouts never become zero-latency measurements.

These are single-process trials, not repeated independent trials with confidence intervals. Optional hardware counters are disabled for ordinary sampled runs; separate profiling-loop CPU usage and generated-code artifacts remain available. Static SIMD/vector counts are not dynamic vector utilization.

Schedule application in `experiment.py:policy_schedule()` is strict. A failed transformation fails that RL method. Standard V5 `Env.__apply_sequence` can skip failed transformations and continue, so its failure behavior differs. A saved planned schedule alone does not establish successful application.

## Why full models did not all work

The downloaded `data/full_models/` contains **18 MLIR files**, without the original Torch/ONNX model bundles. This table records the previous attempts, before their generated outputs were deleted:

| Models | Previous outcome | What is needed |
| --- | --- | --- |
| ALBERT, ConvNeXt-Tiny, DistilBERT, EfficientNet-B0, GPT-2, ResNet50, VGG16, Whisper-base | Excluded: resources contain `__elided__`; numerical weights are absent. | Exact missing weight resources, or re-export from the matching original model and weights. |
| ResNeXt50 | Convolution weights embedded, but 160 entry arguments include external BatchNorm state with no saved values. | Exact running means/variances, counters and remaining argument/state values plus their mapping and inference semantics; preferably a complete inference re-export. |
| GIN | All four methods completed and agreed. | No missing weights for the implemented reference. |
| GAT | Baseline/eager/Inductor agreed; RL schedule application aborted. | Diagnose/fix the MLIR transformation failure. |
| MobileNetV3-Small, T5 | Baseline measured; RL transform failed; full Torch reference not implemented. | Repair transformation application and provide matching Torch graphs using the exact embedded weights. |
| BART, BERT | Baseline exceeded the 600-second method cap; RL transform failed; Torch reference not implemented. | Diagnose setup/compilation time, increase limits where justified, fix transforms and provide equivalent Torch graphs. |
| LLaMA 3.2 1B, ViT-B/16, YOLOv8m | Both MLIR methods exceeded the 600-second cap; Torch reference not implemented. | Identify the slow stage, allow adequate resources/time and provide Torch equivalents. LLaMA used about 44 GB RSS in a 192 GB allocation: the previous failure was not established as OOM. |

The transform abort was a `PybindAdaptors.h` **unchecked-error-message assertion** in the MLIR transform interpreter. That is a compiler/application failure; it does not establish missing weights or a numerical mismatch. The precise failing transformation still requires investigation.

A method timeout covers parsing, ABI adaptation, policy inference, transformations, lowering, JIT setup, correctness calls and measurements. It does not prove that one model inference takes ten minutes. More RAM/time cannot recover missing weights, implement an absent Torch graph or repair a compiler assertion.

This experiment rejects elided constants. The ordinary V5 full-model helper contains a zero-materialization path for timing-only evaluation; it is not used here because that cannot establish equivalence to the original weighted model.

### What information is needed for a fair comparison of every full model?

1. **Exact graph, parameters and buffers:** preferably the matching PyTorch architecture/source and state dictionary, or ONNX plus all external tensor-data files. Elided values cannot be recovered from the MLIR text. A public pretrained model with the same name is insufficient unless its graph and weights match.
2. **An executable equivalent Torch reference:** original PyTorch is the direct route for eager/Inductor. Reconstructing from complete MLIR is possible, as for GIN/GAT, but remains unimplemented for the other seven weight-complete models. Having embedded MLIR weights alone does not supply a Torch implementation.
3. **Input/output and state semantics:** argument order, shapes, dtypes, layouts, index ranges, attention masks, preprocessing, output order and evaluation mode. Supply any external/mutable buffers and representative saved inputs/expected outputs.
4. **Working MLIR compilation and transformations:** establish baseline correctness, diagnose failing actions, and distinguish a complete schedule from partial application.
5. **Matched execution conditions and sufficient resources:** same values, precision, threading, CPU affinity and host-call scope; compilation reported separately; adequate memory, disk and wall time for large models.

Training learning rates/optimizer settings are not the missing information. The matching architecture/action configuration for the selected V5 policy is already available.

### Inspect or run one full model, in the foreground

These commands inspect sources only and write no JSON. Auditing reads large files and can take time:

```bash
python experiments/01_performance_diagnosis/run_suite.py --dataset full_models --list
python experiments/01_performance_diagnosis/run_suite.py --audit-only
```

Run the implemented full GIN comparison:

```bash
bash experiments/01_performance_diagnosis/run_full_models.sh gin_linalg
```

The same launcher accepts `gat_linalg`, whose RL failure still needs repair. Full-model defaults: two correctness cases, two warmups, 20 repeats and a 600-second cap per method.

An explicitly **MLIR-only diagnostic**, not a validated four-way result:

```bash
bash experiments/01_performance_diagnosis/run_full_models.sh mobilenet_v3_small_linalg \
  --allow-mlir-only --timeout-seconds 1800
```

Models with absent weights/state are refused. Other weight-complete models require `--allow-mlir-only` until Torch references exist. Outputs go to `results/full_models/<model_stem>/` with the same report/artifact layout. Nothing is submitted to Slurm.

## Advanced: a custom block or full-model reference

Only for extending automatic coverage, generate an editable example (creates input files, runs nothing):

```bash
python experiments/01_performance_diagnosis/experiment.py init
```

This optional workflow creates `inputs/workload.mlir` and `inputs/experiment.json`. Here JSON is user input describing the source, argument arrays, exact Torch graph and checkpoint/config paths. The template has no checkpoint selected; set those paths for RL. Relative paths resolve against the input JSON. This is separate from the category workflow, which needs no hand-written JSON.

Run the custom workload synchronously:

```bash
python -u experiments/01_performance_diagnosis/experiment.py run \
  experiments/01_performance_diagnosis/inputs/experiment.json \
  --output experiments/01_performance_diagnosis/results/custom
```

Use `python experiments/01_performance_diagnosis/experiment.py explain` for the schema and measurement contracts. A custom reference must match accumulator, layout, convolution, reduction, constant and output semantics; a matching filename alone is not evidence of equivalence.
