#!/bin/bash
#SBATCH --job-name=eval-fullmodel
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#
# Memory sizing: one full-model rollout + transformed-module compile peaks at
# ~8-10GB for large models (t5/resnext OOM-killed a 32G --jobs 4 run).
# 128G covers --jobs 7-8 on typical mixes.
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_fullmodel_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_fullmodel_%j.err
#SBATCH --mail-type=END,FAIL
#
# V5.2 full-model evaluation: greedy rollout of block-trained checkpoints on
# the full .mlir models (MobileNet-v2, ResNet, VGG), reusing the standard
# block eval machinery (scripts/eval/eval.py -> evaluate_benchmarks). The
# per-op policy + full-Module transform application in Env IS the greedy
# full-model rollout (see docs/design/todo/v5_2_full_model_eval.md).
#
# One model per eval.py process (sequential): MLIR native state does not
# survive multiple full-model executions in a single process - the worker
# dies silently partway through. Same reason ckpt_scan_all.sh uses one
# process per model.
#
# Outputs land in results/mlir_rl_v1_paper_results/v5_fullmodel_agent/
# (separate agent dir so eval/checkpoint_N.json never collides with block eval).
# Checkpoints are read from v5_agent/models/ via EVAL_DIR below.
#
# Usage:
#   sbatch scripts/eval/eval_fullmodel.sh --checkpoint 6400 --label smoke
#   sbatch scripts/eval/eval_fullmodel.sh --checkpoint 6400 --models model_vgg
#   sbatch scripts/eval/eval_fullmodel.sh --checkpoint 6400 --jobs 8
#          (--jobs N: up to N models in parallel as separate processes;
#           crash-isolated, same pattern as ckpt_scan_all.sh)
#
# Monitor:
#   tail -f logs/eval_fullmodel_<jobid>.out

set -e
trap 'echo "FULLMODEL EVAL FAILED"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

export MIN_EXEC_TIMEOUT=${MIN_EXEC_TIMEOUT:-300}
# Cap for single-model compile+exec (execution.py MAX_EXEC_TIMEOUT, default 7200).
# Large full models (distilbert/gpt2/bert/vit class) need ample time for LLVM -O3 JIT.
export MAX_EXEC_TIMEOUT=${MAX_EXEC_TIMEOUT:-18000}
# Floor for the same window: execute_code calibrates the join() timeout as
# min(MAX, max(MIN, root*5)), which collapses to 300s for fast-running large
# models (gpt2: 6.7s baseline, ~90-min LLVM JIT -> every eval killed at 5 min).
export COMPILE_MIN_TIMEOUT=${COMPILE_MIN_TIMEOUT:-14400}

# Native-crash diagnostics: MLIR C++ aborts/segfaults otherwise kill the worker
# silently. Faulthandler dumps the Python traceback to stderr on fatal signals.
export PYTHONFAULTHANDLER=1

# Extract flags before positional args
REQ_CHECKPOINT=""
REQ_EVAL_DIR=""
EVAL_LABEL=""
ONLY_MODELS=""
POS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) REQ_CHECKPOINT="$2"; shift 2;;
        --checkpoints-dir) REQ_EVAL_DIR="$2"; shift 2;;
        --label) EVAL_LABEL="$2"; shift 2;;
        --models) ONLY_MODELS="$2"; shift 2;;  # space-separated subset
        --jobs) MAX_JOBS="$2"; shift 2;;       # max parallel models (default 1)
        *) POS_ARGS+=("$1"); shift;;
    esac
done
set -- "${POS_ARGS[@]}"
MAX_JOBS="${MAX_JOBS:-1}"

# NOTE: EVAL_CHECKPOINT is deliberately NOT exported: eval.py's --checkpoint
# post-process derives the destination agent dir from EVAL_DIR (which points
# at v5_agent/models), so it would write into the block-eval agent dir and
# pollute its checkpoint CSVs. Checkpoint filtering uses EVAL_START/END/STRIDE
# and this script copies the result file itself (see merge step below).
if [[ -n "${REQ_CHECKPOINT:-}" ]]; then
    export EVAL_START="$REQ_CHECKPOINT"
    export EVAL_END="$REQ_CHECKPOINT"
    export EVAL_STRIDE=1
fi

CONFIG="${1:-$PROJECT_ROOT/config/v5/v5_mlir_rl_v1_paper_fullmodel.json}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

CONFIG_IMPL="$(python3 -c "import json; print(json.load(open('$CONFIG')).get('implementation', 'rl_autoschedular_v5'))")"
export AUTOSCHEDULER_IMPL="${AUTOSCHEDULER_IMPL:-$CONFIG_IMPL}"

if [[ "$AUTOSCHEDULER_IMPL" == *"no_transformer"* ]]; then
    DEFAULT_AGENT_DIR="v5_no_transformer_agent"
    DEFAULT_FULLMODEL_DIR="v5_no_transformer_fullmodel_agent"
else
    DEFAULT_AGENT_DIR="v5_agent"
    DEFAULT_FULLMODEL_DIR="v5_fullmodel_agent"
fi

# Checkpoints live in the training agent dir; the fullmodel agent dir has no models/
export EVAL_DIR="${REQ_EVAL_DIR:-${EVAL_DIR:-$PROJECT_ROOT/results/mlir_rl_v1_paper_results/$DEFAULT_AGENT_DIR/models}}"
# Each per-model run shares the agent results_dir, so force re-evaluation
# (otherwise run 2+ would see run 1's resumption marker and skip).
export EVAL_FORCE=1
if [[ ! -d "$EVAL_DIR" ]]; then
    echo "ERROR: checkpoint dir not found at $EVAL_DIR"
    exit 1
fi

FULLMODEL_AGENT="$(python3 -c "import json; print(json.load(open('$CONFIG')).get('results_dir', '$PROJECT_ROOT/results/mlir_rl_v1_paper_results/$DEFAULT_FULLMODEL_DIR'))")"
[[ "$FULLMODEL_AGENT" != /* ]] && FULLMODEL_AGENT="$PROJECT_ROOT/$FULLMODEL_AGENT"
EVAL_JSON="$(python3 -c "import json; print(json.load(open('$CONFIG'))['eval_json_file'])")"
[[ "$EVAL_JSON" != /* ]] && EVAL_JSON="$PROJECT_ROOT/$EVAL_JSON"

echo "=========================================="
echo "Full-model eval started at $(date)"
echo "Config:     $CONFIG"
echo "Checkpoints: $EVAL_DIR"
[[ -n "${REQ_CHECKPOINT:-}" ]] && echo "Checkpoint: $REQ_CHECKPOINT"
[[ -n "${EVAL_LABEL:-}" ]] && echo "Label:      $EVAL_LABEL"
[[ -n "${ONLY_MODELS:-}" ]] && echo "Models:     $ONLY_MODELS"
echo "Jobs:       $MAX_JOBS"
echo "Node:       $(hostname)"
echo "=========================================="

cd "$PROJECT_ROOT"
TMPDIR_RUN=$(mktemp -d "${TMPDIR:-/tmp}/fullmodel_eval_XXXXXX")
trap 'rm -rf "$TMPDIR_RUN"; echo "FULLMODEL EVAL FAILED"' ERR

run_model() {
    local MODEL="$1"
    echo "[$(date +%T)] --- $MODEL ---"
    # Per-model config: same hyperparams (checkpoint compat), 1-bench eval set,
    # private results_dir (parallel jobs must not share logs/ or resumption state)
    python3 - "$CONFIG" "$EVAL_JSON" "$MODEL" "$TMPDIR_RUN" <<'PY'
import json, sys
base_cfg, benches_file, model, tmpdir = sys.argv[1:5]
cfg = json.load(open(base_cfg))
benches = json.load(open(benches_file))
one = {model: benches[model]}
bench_json = f"{tmpdir}/{model}.eval.json"
json.dump(one, open(bench_json, "w"))
cfg["json_file"] = bench_json
cfg["eval_json_file"] = bench_json
cfg["results_dir"] = f"{tmpdir}/agent_{model}"
cfg_path = f"{tmpdir}/{model}.config.json"
json.dump(cfg, open(cfg_path, "w"))
print(cfg_path)
PY
    MODEL_CFG="$TMPDIR_RUN/${MODEL}.config.json"
    MODEL_AGENT="$TMPDIR_RUN/agent_${MODEL}"
    if CONFIG_FILE_PATH="$MODEL_CFG" python scripts/eval/eval.py; then
        cp "$MODEL_AGENT/logs/eval/eval_exec_times.json" "$TMPDIR_RUN/${MODEL}.result.json"
    else
        return 1  # reported with real exit code by record_result
    fi
}

FAILED=0
FAILED_MODELS=""
PIDS=()
declare -A PID_MODEL=()
record_result() {  # $1=pid -> records FAILED + model name, always returns 0
    local pid="$1" rc=0
    wait "$pid" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        FAILED=$((FAILED + 1))
        FAILED_MODELS="$FAILED_MODELS ${PID_MODEL[$pid]:-unknown}"
        echo "[$(date +%T)] !!! ${PID_MODEL[$pid]:-unknown} FAILED (exit $rc)"
    fi
    return 0
}
for MODEL in $(python3 -c "import json; print(' '.join(json.load(open('$EVAL_JSON')).keys()))"); do
    if [[ -n "${ONLY_MODELS:-}" ]] && [[ " $ONLY_MODELS " != *" $MODEL "* ]]; then
        continue
    fi
    run_model "$MODEL" &
    PID_MODEL[$!]="$MODEL"
    PIDS+=($!)
    # At capacity: block on the oldest job (rolling window, max MAX_JOBS live)
    while [[ "${#PIDS[@]}" -ge "$MAX_JOBS" ]]; do
        oldest="${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
        record_result "$oldest"
    done
done
# Drain remaining jobs
for pid in "${PIDS[@]}"; do
    record_result "$pid"
done
[[ -n "$FAILED_MODELS" ]] && echo "Failed models:$FAILED_MODELS"

# Merge per-model results -> eval/checkpoint_<N>.json
if [[ -n "${REQ_CHECKPOINT:-}" ]]; then
    _suffix=""
    [[ -n "${EVAL_LABEL:-}" ]] && _suffix="_${EVAL_LABEL}"
    DST="$FULLMODEL_AGENT/eval/checkpoint_${REQ_CHECKPOINT}${_suffix}.json"
    mkdir -p "$(dirname "$DST")"
    python3 - "$DST" "$EVAL_JSON" "$TMPDIR_RUN" <<'PY'
import json, sys, glob, os
dst, benches_file, tmpdir = sys.argv[1:4]
merged = {}
# Preserve existing entries (e.g. other models evaluated in earlier runs)
if os.path.exists(dst):
    try:
        merged.update(json.load(open(dst)))
    except json.JSONDecodeError:
        pass
for f in glob.glob(os.path.join(tmpdir, "*.result.json")):
    merged.update(json.load(open(f)))
baseline = json.load(open(benches_file))
json.dump(merged, open(dst, "w"), indent=2)
print(f"Saved full-model results to {dst}")
ok = {k: v for k, v in merged.items() if v}
print(f"benches: {len(merged)}, succeeded: {len(ok)}")
for k, v in merged.items():
    base = baseline.get(k)
    print(f"  {k}: exec={v}ns" + (f" speedup={base / v:.2f}x" if v and base else " FAILED"))
PY
fi

rm -rf "$TMPDIR_RUN"
trap - ERR
if [[ "$FAILED" -gt 0 ]]; then
    echo "Full-model eval completed with $FAILED failure(s) at $(date)"
    exit 1
fi
echo "Full-model eval completed at $(date)"
