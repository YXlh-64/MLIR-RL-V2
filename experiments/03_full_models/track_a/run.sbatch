#!/bin/bash
#SBATCH --job-name=fm-trackA
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append

# Track A, one array task = one model, all steps on the same node:
#   1. MLIR baseline   (scripts/baseline/baseline_fullmodels.py, V5 execution API)
#   2. MLIR-RL         (scripts/eval/eval.py greedy rollout of v5_model_1200 on the whole module,
#                       exactly as scripts/eval/eval_fullmodel.sh does per model)
#   3. MLIR baseline again (measures run-to-run noise)
# Submit via launch.sh (sets THREADS and --cpus-per-task). Resumable: a step whose output exists
# is skipped; a model is done when state/<threads>t_<model>.done exists.
set -eo pipefail
cd /scratch/mh8826/MLIR-RL
source experiments/02_block_diagnosis/overnight/settings.sh
source "$E/overnight/env.sh"
T=/scratch/mh8826/MLIR-RL/experiments/03_full_models/track_a
THREADS="${THREADS:?set by launch.sh}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export AUTOSCHEDULER_IMPL=rl_autoschedular_v5 DASK_TRAINING_ONLY=1
export MIN_EXEC_TIMEOUT=300 MAX_EXEC_TIMEOUT=14400 COMPILE_MIN_TIMEOUT=14400 EVAL_FORCE=1
export EVAL_DIR=/scratch/mh8826/ckpt_dirs/v5_1200 EVAL_START=1200 EVAL_END=1200 EVAL_STRIDE=1

IFS=$'\t' read -r MODEL DIR FILE BASE_NS CFG < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$T/models.tsv")
OUT="$T/results/t${THREADS}/$MODEL"; mkdir -p "$OUT"
DONE="$T/state/t${THREADS}_${MODEL}.done"
printf '[%(%F %T)T] model=%s threads=%s node=%s cpus=%s\n' -1 "$MODEL" "$THREADS" "$(hostname)" "$SLURM_CPUS_PER_TASK"
[ -e "$DONE" ] && { echo "already done"; exit 0; }
run() { srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores "$@"; }

baseline() {   # $1 = output folder (own exec cache, so the second run is a fresh measurement)
    mkdir -p "$1"
    [ -s "$1/baseline.json" ] && grep -q "\"$MODEL\": [1-9]" "$1/baseline.json" && return 0
    local t0=$SECONDS
    run python -u scripts/baseline/baseline_fullmodels.py --models-dir "$DIR" --pattern "$FILE" \
        --output "$1/baseline.json" --timeout 14400 > "$1/stdout.log" 2>&1 || true
    echo "{\"wall_seconds\": $((SECONDS - t0))}" > "$1/wall.json"
    printf '[%(%F %T)T] %s: %s\n' -1 "$1" "$(tr -d '\n ' < "$1/baseline.json" 2>/dev/null)"
}

baseline "$OUT/baseline_1"

mkdir -p "$OUT/rl"
if [ ! -s "$OUT/rl/eval_exec_times.json" ]; then
    sed "s#__RESULTS_DIR__#$OUT/rl/agent#" "$CFG" > "$OUT/rl/config.json"
    t0=$SECONDS
    CONFIG_FILE_PATH="$OUT/rl/config.json" SLURM_JOB_NAME=fm-trackA \
        run python -u scripts/eval/eval.py > "$OUT/rl/stdout.log" 2>&1 || echo "eval.py exited non-zero"
    echo "{\"wall_seconds\": $((SECONDS - t0))}" > "$OUT/rl/wall.json"
    for f in "$OUT/rl/agent/logs/eval/eval_exec_times"*.json; do [ -e "$f" ] && cp "$f" "$OUT/rl/eval_exec_times.json"; done
    printf '[%(%F %T)T] rl: %s\n' -1 "$(tr -d '\n ' < "$OUT/rl/eval_exec_times.json" 2>/dev/null || echo FAILED)"
fi

baseline "$OUT/baseline_2"

if [ -s "$OUT/rl/eval_exec_times.json" ] && grep -q "\"$MODEL\": [1-9]" "$OUT/baseline_1/baseline.json"; then
    touch "$DONE"; printf '[%(%F %T)T] done\n' -1
else
    printf '[%(%F %T)T] INCOMPLETE (see %s)\n' -1 "$OUT"; exit 1
fi
