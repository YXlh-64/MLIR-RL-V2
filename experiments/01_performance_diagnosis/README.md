# CPU benchmark diagnosis

Compare **MLIR baseline, MLIR RL, PyTorch eager, TorchInductor and TensorFlow XLA** on the same computation and inputs. Run the three logical categories independently. This is checkpoint inference and CPU measurement, not training.

There are four Python files and this one README:

- `execution_profiling.py`: execution latency, speedups, execution hardware counters, validation and shared worker/CSV utilities.
- `compilation_profiling.py`: separate optimization/compilation trials, phase wall/CPU/memory statistics, per-action timing and optional Python function profiling.
- `backends.py`: native backend execution, checked translation of the actual MLIR graph into framework operations, and V5 MLIR compilation.
- `plot_speedups.py`: three category figures comparing all five backends against MLIR baseline, using validated paired-trial speedups from the execution CSVs.

Dataset files and the existing `scripts/eval/` and `scripts/baseline/` programs are unchanged. Results belong to this experiment. Generate figures explicitly after execution results are available:

```bash
python experiments/01_performance_diagnosis/plot_speedups.py \
  --note 'Blocks combine bn001/bn004; original campaigns retain the documented OpenMP placement caveat.'
```

This writes exactly three PNGs to `results/figures/speedups_<category>.png`. Each grouped horizontal bar plot shows MLIR baseline, MLIR RL, TorchInductor, PyTorch eager and TensorFlow XLA at eight threads. A shared **logarithmic** scale keeps both sub-baseline and very large speedups visible. All three figures have identical limits, power-of-ten ticks, positive bar origins, image dimensions and axes positions, determined from the combined validated results. Equal spacing represents equal ratios; bar endpoint labels give the actual speedups. The dashed reference is 1×; higher is faster. Failed, missing or incomplete trials are marked × in the right-hand N/A column. All sampled benchmarks remain visible, including blocks not yet measured and full models without valid results. Use `--input`, `--output`, `--threads` or `--samples` to select another result root, figure folder, recorded CPU budget or frozen selection. Use `--note` to label measurement context; the note above applies to the archived default campaigns.

The block reporting outputs now combine the original nine benchmarks with the 13 continuation benchmarks: **22 blocks, 110 backend rows, 109 validated results**. MLIR RL scheduling failed for `efficientnet_b0_block_500`; its speedup remains unavailable. Raw artifacts remain in their source campaigns, and each merged row identifies its source campaign, hostname and CPU IDs. Original aggregates are backed up under the directory recorded in [merge provenance](results/execution/multi_operation_blocks/merge_provenance.json). The original `campaign.json` still describes the original allocation; the merged `summary.json` records both campaigns. These merged aggregates are reporting outputs; collect further measurements into a separate campaign root. See [analysis](results/analysis.md) and [backend statistics](results/backend_summary.csv) for coverage and comparisons on common validated benchmarks.

## Inputs and the fixed selection

`benchmark_inventory.csv` is the complete inventory. `sampled_benchmarks.csv` is the **frozen selection used by every run**. Both have the same six columns:

```text
dataset,benchmark,operation_count,operation_types,operation_type_counts,category
```

- `dataset`: `mlir_rl_v1_paper`, `ops_and_blocks` or `full_models`.
- `benchmark`: filename including `.mlir`; resolved as `data/<dataset>/<benchmark>`.
- `operation_count`: number of static Linalg operations, including fill and excluding index/yield.
- `operation_types`: JSON list, for example `["linalg.generic", "linalg.matmul"]`.
- `operation_type_counts`: JSON dictionary, for example `{"linalg.generic": 3, "linalg.matmul": 2}`.
- `category`: `single_operations`, `multi_operation_blocks` or `full_models`.

The current selection contains **24 single-operation files, 22 blocks and 12 full models**. Sampling targets four examples per MLIR operation type within each category, pooled across both source datasets. A mixed-operation file counts towards each type it contains, so common types can appear more than four times. Rare full-model Conv1D and grouped Conv2D have only one available example each; those are retained. The small GIN/GAT models are included as full-program controls.

Kernel sampling limits input tensors to 32 MiB and excludes QCD kernels, which need domain-specific index/input fixtures. The full inventory remains intact. Full models are sampled without hiding known missing-weight/resource limitations; those limitations become explicit outcomes during preparation. This is a bounded diagnostic selection, not an unbiased estimate of all model performance.

To create a **different** frozen selection, write a new CSV:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py sample \
  --per-type 3 --seed 42 --output /scratch/$USER/another_sample.csv
```

The sample command preserves exact inventory entries and refuses to overwrite an existing selection. The run command never resamples. Use `--samples /path/to/selection.csv` to select another CSV explicitly.

## Environment

Use the MLIR Conda environment and a reserved compute-node allocation. Commands below run sequentially in the foreground; they do not submit or cancel Slurm jobs. Reserve at least **eight physical cores** for the single eight-thread CPU budget, with enough memory for the selected compilation workloads. Source-file and input-buffer limits do not bound compiler peak memory.

From the repository root:

```bash
set -a
source .env
set +a
export PATH="$CONDA_ENV/bin:$PATH"

python experiments/01_performance_diagnosis/execution_profiling.py doctor
```

The runner also sources the repository `.env` through Bash before launching workers. This supports the shell expansions used in this repository. V5 config imports occur only after worker environment setup. The default checkpoint/config are `checkpoints/v5_model_1200.pt` and `config/v5/v5_mlir_rl_v1_paper.json`; they must match. Only the standalone V5 implementation is imported.

Verify OpenMP placement before running a new campaign:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py check-openmp
```

This requires eight allocated physical cores and the existing LLVM build, Torch and Clang. It generates a small C control in `results/openmp_control/`, links the **same LLVM libomp** used by MLIR, and imports Torch first to exercise the problematic initialization order. Eight independent arithmetic lanes run with one thread and then eight threads, using identical work and checking identical checksums. A pass requires eight distinct allocated CPUs, eight correctly bound threads, useful work by every thread, at least six average busy cores during the parallel call, and at least 2× serial-to-parallel speedup. These conservative utilization/speedup thresholds tolerate some system noise; a busy allocation can fail the check. `result.json` retains each lane's CPU, Linux thread ID, affinity and CPU time, both wall times, checksums and every pass/fail condition. `worker.log` and `compiler.log` retain diagnostics; the command exits nonzero on failure. `--python` and `--output` select another interpreter or control directory. No dataset files are needed.

**Placement fix:** V5 imports Torch's GNU OpenMP runtime, which can bind the main thread to its first CPU. If LLVM's separate OpenMP runtime initializes with that narrowed mask, it discovers only one valid place and ignores the other seven CPU IDs. The shared `MLIRRunner` now restores the worker's assigned CPU set **after V5/Torch imports and before LLVM OpenMP initialization**, initializes the configured LLVM runtime explicitly, and rejects a place list/thread budget that differs from the assignment. This applies to both execution and compilation profiling, including compilation workers that previously ran the policy at one thread. The launcher removes inherited `KMP_AFFINITY`, `KMP_HW_SUBSET` and `GOMP_CPU_AFFINITY` overrides so the explicit `OMP_PLACES` controls placement. Only CPUs selected from the inherited allocation are used. The [LLVM runtime documentation](https://openmp.llvm.org/design/Runtimes.html) describes affinity places and the initial-mask behavior.

This enables actual parallel regions to use eight cores; it does not add parallel transformations to serial schedules or guarantee useful scaling for small/unbalanced workloads. The main thread's one-CPU mask after initialization is normal: inspect all thread masks and CPU time together. Existing measurements are preserved. Results collected with the earlier placement problem need remeasurement; code hashes prevent mixing them with fixed workers. For example:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py run \
  --benchmark gin_block_21 --methods mlir_baseline mlir_rl torch_eager torch_inductor \
  --output experiments/01_performance_diagnosis/results/execution_openmp_fixed
```

TensorFlow CPU **2.21.0** is installed directly in the `mlir` Conda environment (`$CONDA_ENV/bin/python`). XLA CPU support is included in that package, following the official [TensorFlow pip installation instructions](https://www.tensorflow.org/install/pip). No separate TensorFlow interpreter is needed by default. To reproduce this setup and check it:

```bash
"$CONDA_ENV/bin/python" -m pip install tensorflow-cpu==2.21.0
export TF_BENCH_PYTHON="$CONDA_ENV/bin/python"
python experiments/01_performance_diagnosis/execution_profiling.py doctor \
  --tensorflow-python "$TF_BENCH_PYTHON"
```

`doctor` checks MLIR/PyTorch imports and then TensorFlow in separate subprocesses. Its TensorFlow check compiles a matrix-multiply/add/ReLU function with `jit_compile=True`, executes it, verifies the numerical output and retrieves optimized HLO. It exits nonzero if an import or this XLA check fails. Benchmark backends likewise keep separate workers even when sharing the same Python environment.

An optional separate environment is still supported with `--tensorflow-python`; the existing `/scratch/ab13817/codex/cache/benchmark-tensorflow-env/bin/python` remains available. For another machine, an example isolated installation is:

```bash
"$CONDA_ENV/bin/python" -m venv --system-site-packages /scratch/$USER/tensorflow-benchmark-env
/scratch/$USER/tensorflow-benchmark-env/bin/python -m pip install tensorflow-cpu==2.21.0
export TF_BENCH_PYTHON=/scratch/$USER/tensorflow-benchmark-env/bin/python
```

TensorFlow uses `tf.function(jit_compile=True)` with CPU execution and synchronous calls. Compilation failures are recorded; the runner does not fall back to ordinary TensorFlow execution. See the official [tf.function documentation](https://www.tensorflow.org/api_docs/python/tf/function) and [synchronous execution setting](https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_synchronous_execution).

## Commands for each category

Preview a category, including the exact CPU budgets, without writing results:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py run \
  --category single_operations --dry-run
```

Run each category separately, using the same frozen CSV and five backends:

```bash
python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category single_operations --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category multi_operation_blocks --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category full_models --tensorflow-python "$TF_BENCH_PYTHON"
```

The agreed execution protocol is:

| Setting | Default |
| --- | --- |
| CPU budget | **Eight threads**, pinned to eight allocated physical cores; no thread-count sweep |
| Process trials | **Three** fresh workers per benchmark/backend |
| First invocation | One, recorded separately from execution latency |
| Correctness | Three shared input cases |
| Warmup | **Five** additional untimed calls |
| Measured execution | **Five** timed calls per worker; take their median |
| CPU/hardware window | **At least one second AND at least three complete calls**, separate from timed execution |

One measurement worker means one Python process handling **one benchmark, one backend and one trial** at the campaign's CPU budget. It can use eight CPU threads; a worker is not a CPU core. Workers run sequentially, with shuffled backend order per trial. With five backends, one benchmark has **15 measurement workers and 75 timed calls**, plus preparation, scheduling, correctness, warmup and profiling. These fresh processes repeat backend setup/compilation, but execution calls inside a worker reuse its executable. CPU allocation allows parallel work; it does not guarantee that a backend uses all eight cores.

Both profiling scripts accept the same category/type/name filters. Restrict the selection to contained operation types or a sampled benchmark:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py run \
  --category single_operations --op-type add matmul \
  --tensorflow-python "$TF_BENCH_PYTHON"

python experiments/01_performance_diagnosis/execution_profiling.py run \
  --category multi_operation_blocks --benchmark gin_block_21 \
  --tensorflow-python "$TF_BENCH_PYTHON"

# Category is inferred when the matching CSV rows belong to one category.
python experiments/01_performance_diagnosis/compilation_profiling.py run \
  --benchmark resnext50_linalg --dry-run
```

`--op-type` selects files **containing** any requested MLIR type; combine it with category/name filters to intersect the selections. `linalg.generic` is not relabelled as an assumed semantic operation. Names accept a stem (`gin_block_21`), a filename (`gin_block_21.mlir`) or `dataset/name` to disambiguate. `--only` remains an alias for `--benchmark`; `--operation-category` and `--operation-type` are aliases for the shorter options. Filters only select from the frozen CSV and never trigger new sampling. An unknown name, conflicting filter or selection spanning multiple categories is rejected. Selecting ResNeXt is supported; its missing external BatchNorm fixtures still block an actual run.

### Separate compilation and optimization runs

The compilation script repeats **schedule inference, action application and compilation** in each fresh trial. It makes one invocation, with no execution warmups, latency repeats or steady-state hardware profiling. Defaults are **eight threads and three trials**, giving 15 compilation workers per benchmark with all five backends. RL inference uses one Torch thread, then the worker restores the eight-thread budget before lowering/JIT/execution. Compiler passes may remain serial.

```bash
python -u experiments/01_performance_diagnosis/compilation_profiling.py run \
  --category single_operations --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/compilation_profiling.py run \
  --category multi_operation_blocks --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/compilation_profiling.py run \
  --category full_models --tensorflow-python "$TF_BENCH_PYTHON"

# Small block investigation, restricted using the sampled CSV.
python -u experiments/01_performance_diagnosis/compilation_profiling.py run \
  --category multi_operation_blocks --op-type matmul --benchmark gin_block_21 \
  --trials 1 --methods mlir_baseline mlir_rl

# Optional Python hotspots: use a different output root for instrumented results.
python -u experiments/01_performance_diagnosis/compilation_profiling.py run \
  --benchmark gin_block_21 --trials 1 \
  --methods mlir_baseline mlir_rl --python-profile \
  --output /scratch/$USER/compilation_python_profile
```

Default roots are `results/execution/<category>/` and `results/compilation/<category>/`. `--output` replaces the root before the category is appended. Compilation success means compilation and the first invocation completed; numerical validation and performance claims belong to the execution script.

A short end-to-end check, writing to a separate output root:

```bash
python experiments/01_performance_diagnosis/execution_profiling.py run \
  --category single_operations --only add_112_15_15_15.mlir \
  --trials 1 \
  --output /scratch/$USER/benchmark_smoke \
  --tensorflow-python "$TF_BENCH_PYTHON"
```

Other controls:

| Option | Meaning |
| --- | --- |
| `--threads 8` | One CPU budget, default eight. A single explicit override is allowed for another machine; lists/sweeps are rejected. |
| `--trials 3` | Fresh process trials per benchmark/backend; all trials use the same CPU budget. |
| `--warmup 5` | Additional untimed calls after the first invocation and correctness checks; execution script only. |
| `--repeats 5` | Timed execution calls in each worker; execution script only. |
| `--profile-seconds 1 --profile-repeats 3` | Minimum elapsed wall time and complete calls for the separate execution profiling window; both conditions must hold. |
| `--methods mlir_baseline mlir_rl torch_eager` | Restrict backends; full validation requires baseline and eager. |
| `--timeout 1800` | Whole-worker cap in seconds, including setup, compilation and profiling. |
| `--max-input-mib 64` | Maximum summed input-buffer size at preparation. |
| `--max-source-mib 256` | Refuse larger source files before loading them into Python memory. |
| `--preprocess none` | Disable the common standard V5 standalone-Conv2D img2col preprocessing. |
| `--no-perf` | Execution script: keep latency/CPU-usage/static evidence; mark hardware counters unavailable. |
| `--python-profile` | Compilation script: save `python.prof` and a compact `python_profile.txt`; instrumentation perturbs timings. |
| `--python /path/to/python` | Override the MLIR/PyTorch worker interpreter. |
| `--output /path/to/root` | Separate campaign output root; category directories are appended. |
| `--retry-failed` | Reattempt saved failed workers; retain successful measurements. |

Reusing the same command resumes the same category campaign and skips completed worker attempts, including recorded failures. Use `--retry-failed` for failures. Settings, host/CPU budgets, source code, sample CSV, checkpoint/config hashes and interpreter paths must match the campaign. For changed settings or code, use a new output root. **Do not run two launchers against the same category output at once.**

Ctrl+C stops only the current worker group and launcher, preserving completed results. It does not cancel the interactive Slurm allocation. For execution, exit 0 means every requested backend/trial passed numerical validation; for compilation it means every requested trial compiled and completed one invocation. Exit 1 means some requested trials did not meet that criterion; 130 means interrupted. Missing execution counters alone do not invalidate a numerically correct latency measurement.

Each category is registered automatically in root `experiments.json` **before** worker launch, as required by repository instructions. Only this registry is updated outside the experiment's output directory.

## Implementation and execution order

`execution_profiling.py` owns argument parsing, CSV selection, CPU allocation, preparation, scheduling, isolated process launch, validation and aggregation. `compilation_profiling.py` reuses that infrastructure and supplies its own compile worker and phase summary. `backends.py` supplies the faithful framework adapters, MLIR compiler/JIT and static IR analysis. Backend imports occur inside workers, after environment/config setup. TensorFlow and MLIR/PyTorch use separate workers to avoid conflicting native libraries.

Both commands accept a single integer `--threads` (default 8). The campaign JSON retains `threads: [8]` as a singleton list to preserve the existing summary/artifact schema. Results still use `threads_8/trial_<K>/`; no additional thread settings are launched. Too few allocated physical cores is an error, rather than a silent change of CPU budget. The backend thread limits and CPU affinity are set before native imports. `run --dry-run` lists matching CSV rows and the chosen CPU set without writing results.

For execution profiling, the lifecycle is:

1. **Prepare once per benchmark.** Read/adapt the source in memory, construct the reference plan, save the same three seeded input cases, and apply any recorded common V5 preprocessing to both MLIR methods.
2. **Generate the RL schedule once.** A dedicated one-core worker extracts features/dependencies, loads the checkpoint and greedily chooses actions. Policy decisions are written to `schedule.json`. State updates during planning do not execute the benchmark.
3. **Apply the schedule in that same worker.** After planning, apply the ready actions to actual MLIR, audit successes/fallbacks/failures in `applied_actions.json`, and save `scheduled.mlir`. Generation and application both precede measurement workers. Failed scheduling blocks RL measurements.
4. **Launch each backend/trial worker at eight threads.** MLIR baseline loads `prepared.mlir`; MLIR RL loads the same saved `scheduled.mlir` for every trial. Each worker performs its own setup/compilation, restores case 0 inputs, and records one first invocation.
5. **Check outputs, then warm up.** Execute each of the three correctness cases and save outputs; the parent compares them across backends. Restore case 0 before each of five additional warmup calls.
6. **Measure five execution calls.** Restore case 0 before each call and record synchronous host-call nanoseconds. Input reset, output checking/copying/disposal, compilation and scheduling are outside each timed call. Preserve all five samples and calculate their median.
7. **Profile CPU/hardware use separately.** Repeatedly restore/call/dispose until at least one second and three calls have completed. Preserve actual elapsed time, call count, CPU time, per-thread activity and available perf totals. This window does not contribute execution-latency samples.
8. **Save artifacts and aggregate trials.** Save compiler evidence and rebuild the category CSV, JSON summary and schedule index. A repeated command resumes saved workers rather than generating another schedule or rerunning completed trials.

Compilation boundaries depend on the backend:

| Backend | During worker setup | During the separate first invocation |
| --- | --- | --- |
| MLIR baseline / MLIR RL | MLIR parsing, bufferization, lowering and JIT engine creation | Remaining lazy LLVM/JIT materialization as needed, then execution |
| TorchInductor | Build the native reference graph and create the `torch.compile` wrapper | Lazy compilation and execution |
| TensorFlow XLA | Create the `tf.function(jit_compile=True)` wrapper | Graph tracing/XLA compilation as needed and execution |
| PyTorch eager | Build the native-operation reference graph; no whole-program optimizing compiler | Execute native operators; runtime/library initialization may still occur |

The first invocation is **additional to** the five warmups and five timed calls. It includes an execution and only the compilation work deferred until that call; it is not total compilation time. In compilation profiling, MLIR symbol lookup explicitly materializes the JIT before the first invocation. Inductor/XLA retain a combined lazy-compilation-and-first-call interval. RL scheduling is regenerated and applied inside **each compilation trial**, whereas execution trials reuse one saved schedule. No policy inference or action application happens during warmup, measured calls or hardware profiling.

## Measurement definitions

### A. Execution performance

The primary metric is **synchronous host-call latency** around one invocation. It includes framework/Python/FFI dispatch and result allocation. Input restoration, output copying/checking/disposal, compilation and policy inference are outside the timed call. The reference interpreter runs once during setup: PyTorch FX produces an eager native-operation forward graph, which Inductor also consumes; TensorFlow traces the equivalent graph for XLA. The measured call does not traverse inventory/plan dictionaries. The graphs are faithful translations of these MLIR programs, not unrelated canonical pretrained framework implementations; generic scalar computations can still use more native operators than a hand-written framework model. TensorFlow synchronous execution is enabled explicitly. All backends execute on CPU. All three saved cases are used for correctness; warmup, latency and profiling use case 0 to keep the working set fixed.

Every trial retains its five nanosecond samples, median, mean, standard deviation, minimum and maximum. `result.json.call_counts` records the first invocation, correctness, warmup, measured and profiling call counts separately. `comparison.csv` reports the median of the three process-trial medians and their range, a separate validated-only median, profiling counters/IPC/cache-miss rate, observed busy cores, median profiling call count/duration and main compilation/scheduling costs. Complete details remain in per-worker JSON. Calls within one process are not treated as independent trials or confidence intervals. MLIR internal `nanoTime()` deltas are saved separately and never divided by framework host times.

`median_validated_speedup` is the median of paired-trial ratios `baseline host median / candidate host median` at the **same eight-thread budget**. Greater than 1 is faster. Each trial median comes from its five measured calls. Additional columns compare a backend against eager, Inductor and XLA using validated groups only. There is no one-thread scaling column because this protocol does not collect a one-thread reference.

For fairness:

- One allocated hardware thread per physical core; all methods receive the same CPU set and thread budget. The OpenMP/MKL/BLAS/framework limits are set together. Requesting additional Slurm CPUs alone does not change the benchmark budget.
- Worker execution is sequential. Method order is reproducibly shuffled within each thread/trial group.
- Exact seeded nonzero arrays are generated once, hashed, saved and reused. All arguments, including initial output/accumulator tensors, are restored before every call. Actual embedded weights are decoded, never replaced with random or zero tensors.
- Baseline and RL share V5 bufferization/lowering and LLVM **O3**. "Baseline" means no learned schedule, not O0. Standard V5 standalone Conv2D img2col preprocessing is common to both and recorded. `--preprocess none` provides a separate raw-Conv2D experiment.
- Every saved correctness case/output is checked for count, shape, dtype, finite values and equality within `rtol=1e-4`, `atol=1e-5` (integer/bool exact). Baseline and eager must agree; other methods must agree with both. Invalid or unavailable comparisons have null validated speedups.

### B. Hardware profiling

Profiling is a **separate post-warmup window**, outside primary latency samples:

- `perf stat`: cycles, instructions, cache references/misses and branches/misses, with raw events, units, per-call normalization and multiplexing information retained.
- Process CPU time / wall time estimates average busy cores. Per-thread CPU-tick deltas show which workers actually consumed CPU time; assigned CPU sets are recorded separately from the main thread's post-binding affinity. `result.json.thread_cpu_affinity` records every surviving thread's Linux affinity mask, keyed by the same thread IDs. MLIR workers also save `openmp_placement` with the exact runtime path, pre-restoration main-thread mask, assigned CPUs and validated LLVM place list.
- MLIR scheduled/bufferized/lowered IR, actual JIT object and disassembly provide static vectorization/parallel-region evidence. Inductor retains its generated C++/cache. XLA retains optimized HLO when its inspection API is available.
- `dependencies.json` retains extracted loop bounds, load/store indexing and producer/consumer information; `reference_plan.json` retains original graph/indexing maps. These support static investigation of memory access patterns and fusion opportunities.

The stop rule is **elapsed wall time >= 1 second AND completed calls >= 3**. One second is a minimum, not a deadline: a two-second benchmark needs at least six seconds for its three complete calls, while a fast kernel may execute many more than three calls. `parallel_execution` saves the actual `profile_calls`, `profile_wall_ns`, minimum thresholds and CPU-use statistics. Hardware totals are accumulated over the complete window and normalized per call; CPU utilization is computed from aggregate CPU time divided by window wall time (and by eight for allocated-utilization percent). These are window aggregates, not the median of three individual profiling calls. The category CSV takes medians of the three workers' window metrics. `--no-perf` disables hardware counters but retains the CPU-use window and its stop rule.

The perf window includes the repeated-call harness, input restoration and output disposal. Counts are **not kernel-only**. Generic cache events do not identify a specific cache level or measure DRAM bandwidth; static SIMD mentions do not measure dynamic vector utilization. No such measurements are fabricated. Unsupported counters remain null with diagnostics in `perf.log`/`perf.csv`. Profiling itself can perturb cache behavior, which is why it is separated from primary timing.

### C. Scheduling behavior

`SCHEDULES.md` keeps the previous collapsible per-benchmark schedule format with action parameters and artifact links. Each workload has a shared `methods/mlir_rl/` folder:

- `schedule.json`: all policy choices, including incomplete pointer decisions, parameters, extras and current producer selection.
- `applied_actions.json`: actual ready-action application order, parameters, elapsed time and `applied`, `fallback` or `failed` outcome.
- `dependencies.json`: operation types, dependencies, loop bounds and static memory-access features.
- `dependency_graph.json`: explicit operation nodes and producer → consumer edges, with consumer operand indices, before scheduling.
- `fusion_report.json`: each applied fusion's producer, consumer, operand, generated fused tag, before/after tag locations and whether the original producer remains. An empty list means no ready fusion was applied.
- `schedule_ir_analysis.json`: before/after scheduling graphs and static intermediate/allocation counts.
- `scheduled.mlir`: the resulting program, reused for every execution trial at the eight-thread budget.
- `result.json`: scheduling status and feature/model/inference/application costs.

The planned `V(True)` string alone is not evidence of vectorization. V5 can restore the original IR and change it to `V(False)`; that is recorded as `fallback`. Other transformation exceptions fail RL scheduling rather than being silently counted as applied. Fusion is `confirmed_fused_clone` only when the ready action succeeds and its expected producer/consumer fused tag actually appears in the IR. Its loop locations and any residual original producer are retained; this does not assert that all producer computation or all intermediate storage disappeared. Failed actions retain unconfirmed fusion evidence.

For every MLIR backend, `ir_analysis.json` records input, bufferized and LLVM-lowered snapshots. It includes operation counts, direct shaped-SSA dependency graphs (region arguments are explicit), intermediate tensor-result values, memref-result values, `memref.alloc`/`alloca`/`realloc` sites and LLVM heap-call/stack-allocation sites. It also counts written buffer identities, merging explicit memref views, including caller-provided destination arguments. `intermediate_written_buffer_identities` excludes directly returned buffers/views: zero local allocations therefore does not imply zero intermediate storage. Return-related classification follows direct returns and explicit memref views; it is not a complete loop/interprocedural alias/escape analysis. Static sites inside loops may execute repeatedly, and views can share storage. These are **static counts, not peak live buffers or peak tensor memory**. The execution CSV exposes the main counts; complete graphs and allocation locations stay in the artifact JSON. Graphs describe the actual IR at each stage; LLVM lowering naturally removes tensor/memref SSA types. Framework artifacts remain FX/generated C++ or optimized HLO, without invented MLIR buffer counts.

### D. Compilation and scheduling cost

Use `compilation_profiling.py` for independent, repeated optimization profiles. Execution results still retain setup costs as context, but reuse one schedule across their execution trials.

The compilation script saves `compilation.csv` with **one row per benchmark/backend/thread/trial/phase**, plus a process-wall row for every attempt, including failures. Columns identify the phase, wall nanoseconds, user/system CPU nanoseconds, CPU time of waited-for child processes, page faults, context switches and process high-water RSS before/after the phase (KiB on this Linux system). RSS is a process-wide high-water mark, not phase-local allocation or current memory. CPU times include native worker threads; child CPU covers compiler subprocesses that have been waited for. `summary.json` aggregates phase medians over successful trials only. Individual failed stages and their reasons remain visible.

| Requested phase | Recorded boundary |
| --- | --- |
| MLIR parsing and feature extraction | `mlir_parse_ns`, `schedule_ir_parse_ns`, `feature_extraction_ns` for RL; feature extraction includes V5's AST extraction subprocesses. |
| RL inference | `policy_inference_ns` sums observation/model/action selection; `schedule.json` has per-decision wall/CPU costs, and the result has summed inference CPU time. `checkpoint_loading_ns` is separate. |
| Transformations | `action_00001_ns`, etc. wrap only each actual `action.apply`; `applied_actions.json` maps these phases to operations and parameters. |
| Bufferization and LLVM lowering | `bufferization_ns` is the existing V5 helper, which also performs cleanup/vector lowering; `mlir_lowering_ns` applies the saved execution pass pipeline. |
| OpenMP runtime initialization | `openmp_initialization_ns` restores the assigned mask and initializes/validates LLVM OpenMP separately from lowering and JIT materialization. |
| LLVM/JIT and first invocation | `jit_engine_creation_ns`, then `llvm_jit_materialization_ns` performs symbol lookup without invoking main; `first_invoke_ns` records the subsequent first call. |
| Framework compilation | Native graph construction/wrapper setup and TensorFlow graph tracing are separate. Inductor/XLA `lazy_compile_and_first_invoke_ns` includes compilation plus one execution; pure compiler time is not inferred from it. Eager has only setup and its first call. |

`backend_setup_total_ns` overlaps its nested MLIR/setup phases. `policy_rollout_ns` overlaps inference and includes state updates/audit writes. `schedule_application_ns` overlaps per-action phases and includes auditing. Do not add these inclusive totals to their children. IR inspection and file writes are outside the individual parse/bufferize/lower/action timing intervals, but still affect process wall time and cache state. Preparation costs (source/ABI/reference translation/input generation/common preprocessing) are recorded once in `preparation/result.json`, outside per-trial compiler statistics. These profiles use fresh processes and per-worker Inductor caches; filesystem/OS caches are not flushed.

Optional `--python-profile` saves a cProfile file readable with `python -m pstats path/to/python.prof` and a top-40 cumulative-time text report. It profiles the compile worker including scheduling; Python function/call time can expose wrappers and waits, but does not resolve internal C++ MLIR/LLVM passes. Wall/CPU/resource phase measurements are collected without that option as well. Instrumented campaigns have a distinct setting and should use a separate output root. Compilation profiles are not numerical correctness checks and produce no speedups.

`result.json.compilation` records completed stages separately: source read, ABI adaptation, reference translation, input generation, common preprocessing, feature extraction, checkpoint loading, policy inference, schedule application, MLIR parsing, bufferization, lowering, JIT engine creation, native reference-graph construction, backend-wrapper creation and first invocation, as applicable.

In execution profiles, first invocation may include lazy compilation. It includes one actual execution, so engine/wrapper creation plus first invocation is a **setup upper bound**, not pure compiler time. Baseline/eager do not receive fictitious policy costs. Execution scheduling is measured once per benchmark on one core, rather than hidden in every execution sample. The compilation script regenerates it for every RL trial instead. Per-action `application_ns` and their `transformation_execution_ns` sum separate transform execution from the full audited application phase.

Before each stage the worker persists its status and stage name. Timeouts/crashes retain the last stage and completed costs; a 600-second timeout is a configured whole-worker limit, not an observed execution latency. Partial samples from an interrupted worker are never promoted into successful speedups.

`active_phases` records the nested intervals still running; `phase_status` distinguishes completed and failed intervals. `last_stage` is the most recently entered stage and can already be complete when a failure occurs during intervening artifact work.

## Output structure and statuses

```text
experiments/01_performance_diagnosis/
├── README.md
├── execution_profiling.py
├── compilation_profiling.py
├── backends.py
├── benchmark_inventory.csv
├── sampled_benchmarks.csv
└── results/
    ├── execution/<category>/
    └── compilation/<category>/
        ├── campaign.json
        ├── comparison.csv                       # execution only
        ├── compilation.csv                      # compilation only
        ├── summary.json
        ├── SCHEDULES.md
        └── artifacts/<dataset>/<benchmark_without_extension>/
            ├── manifest.json
            ├── policy_config.json
            ├── inputs.npz
            ├── constants.npz
            ├── reference_plan.json
            ├── prepared.mlir
            ├── preparation/{result.json,worker.log}
            └── methods/
                ├── mlir_rl/{schedule.json,applied_actions.json,dependency_graph.json,
                │            fusion_report.json,dependencies.json,scheduled.mlir,...}
                │                                        # shared schedule: execution only
                └── <backend>/threads_8/trial_<K>/
                    ├── result.json
                    ├── worker.log
                    ├── outputs.npz                       # execution only
                    ├── perf.csv, perf.log                 # execution, if attempted
                    ├── ir_analysis.json                  # MLIR stage graphs/counts
                    ├── schedule.json, applied_actions.json,
                    │   dependency_graph.json, fusion_report.json,
                    │   schedule_ir_analysis.json, scheduled.mlir  # compilation RL only
                    ├── python.prof, python_profile.txt   # compilation --python-profile
                    └── available compiler artifacts        # backend-specific
```

`<category>` is `single_operations`, `multi_operation_blocks` or `full_models`. Both profiles share the same preparation layout; each campaign has its own fixtures. Execution shares one schedule per workload. Compilation stores a new RL schedule beside each trial. `SCHEDULES.md` links the appropriate schedules in either profile. Inductor's necessary compiler cache can create additional generated files. Outputs/caches are ignored by git; the inventory and frozen sample CSV are ordinary reviewable files.

`measured` means an execution worker completed timing; inspect its separate correctness field. `compiled` means a compilation worker completed setup and one invocation, without a numerical claim. `unsupported` identifies an unsupported program/reference/ABI. `unavailable` identifies a missing backend module. `blocked` identifies a failed prerequisite such as weights, preparation or RL scheduling. `error`, `timeout` and `interrupted` retain diagnostics. Missing latency is absent/null, never zero. A measured MLIR-only diagnostic can be `unverified` and has no validated five-way speedup.

Reference support is intentionally checked: static ranked f32/f64/i32/i64/bool tensors; named add, matmul/batch-matmul, NCHW/FCHW and depthwise convolution, NCHW pooling, fill, broadcast and transpose; projected-permutation generic indexing with supported arithmetic/associative reductions; static reshape/concat/slice forms. A generic's provably unused secondary outputs can be removed from the reference plan; a live result depending on such an accumulator is rejected. Arbitrary indexing/control flow, unusual layouts, unsupported dtypes, missing real weights or unsupported full graphs require another reference adapter or valid fixtures. The runner reports the exact unsupported operation. It does not benchmark unrelated pretrained framework models as substitutes for the stored MLIR programs.

## Verification and known outcomes

The fixed selection was checked against the complete inventory: all 58 rows are unchanged inventory entries. Numerical reference checks passed for all 24 selected single operations, all 22 selected blocks, and the GIN/GAT full programs, comparing the actual MLIR computation with eager Torch and CPU XLA. These checks used one fixed test case and are not the full three-trial performance campaign.

Earlier protocol smoke artifacts are stored outside this experiment under `/scratch/ab13817/codex/cache/eight-thread-protocol-execution/` and `eight-thread-protocol-compilation/`. The populated execution campaigns in `results/execution/` predate the OpenMP placement fix and are preserved for reference. Rerun into a new output root to obtain corrected multi-core comparisons.

All three Python files pass `py_compile`. The new default execution protocol passed on `add_112_15_15_15` for **all five backends and all three trials**: 15 workers, 75 latency samples, five warmups per worker, one-second/three-call profiling minima, numerical correctness and median aggregation were checked. Resume preserved worker logs, inputs, policy decisions, applied actions and saved scheduled MLIR, without rerunning measurements. CLI checks verified eight-thread defaults for both scripts and rejection of thread-count lists.

MLIR baseline/RL compilation profiles passed on `gin_block_21` at eight threads, including cProfile, per-action/phase costs and JIT materialization followed by one invocation. They produced no steady-state latency or hardware windows. The fusion audit confirmed two fused clones while retaining evidence that the original producers remained. Static buffer counts retained four intermediate written buffer identities despite zero local allocation sites. Earlier checks also covered missing-weight and whole-worker timeout outcomes. These earlier checks did not verify actual multi-core placement; smoke timings are not campaign performance conclusions.

The placement control in `results/openmp_control/result.json` passed on eight allocated physical cores: **7.48 average busy cores and 7.52× serial-to-parallel speedup**, with identical checksums and one correctly bound worker per core. Replaying the existing `gin_block_21` schedule through the fixed execution worker used **7.96 average busy cores**, with all eight active threads bound to distinct assigned CPUs and all three output cases matching the saved eager reference. The serial MLIR baseline used about one busy core, as expected. These diagnostic replays are stored beneath `results/openmp_control/gin_block_21/`; they do not replace the original campaign.

Baseline and RL compilation workers also completed successfully with all eight LLVM places, including a newly generated/applied RL schedule, lowering, JIT materialization and first invocation. Their artifacts are under `results/openmp_control/compilation_gin_block_21/`. A separate negative control deliberately bypassed the fix in a fresh process: it discovered one place, emitted seven invalid-CPU warnings, used **0.98 busy cores**, and achieved **0.99× speedup**. Its expected failure is saved under `results/openmp_control/before_fix/` with `fix_disabled_for_negative_control: true`; this confirms that the control detects the original bug.

GAT's baseline/eager/Inductor/XLA outputs agree, but this checkpoint's RL application fails on `operation_22: TPF(operation_21;0,16,2)` with a native MLIR diagnostic assertion. This is retained as a failed transformation and blocked RL measurement, with no invented speedup. The underlying evaluation/transform implementation was not changed. Other full models can be blocked by missing real weights, source-size limits or unsupported references, as documented above.
