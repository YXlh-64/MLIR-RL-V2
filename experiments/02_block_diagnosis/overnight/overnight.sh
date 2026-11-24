#!/bin/bash
# One command for the whole overnight evaluation. Run from anywhere:
#     bash /scratch/mh8826/MLIR-RL/experiments/02_block_diagnosis/overnight/overnight.sh
# Submits, with Slurm dependencies:
#   1. setup.sbatch   build LLVM 19 + tools, .env, checks, checkpoint config, block selection
#                     (skipped if state/setup.done exists)
#   2. run.sbatch     array of $SHARDS tasks per checkpoint, after setup succeeded
#   3. report.sbatch  after every run task ended (ok or not): tables, stats, plots, MORNING_REPORT.md
# Safe to re-run: finished stages and finished blocks are skipped. Job ids: state/jobs.log
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/settings.sh"
cd "$REPO"
mkdir -p "$LOGS" "$STATE" "$E/results"

dep=()
if [ -f "$STATE/setup.done" ]; then
    echo "setup already done ($STATE/setup.done)"
else
    setup=$(sbatch --parsable --output="$LOGS/setup_%j.log" "$E/overnight/setup.sbatch")
    echo "setup  job $setup"
    dep=(--dependency="afterok:$setup")
fi

runs=()
for CK in "${CHECKPOINTS[@]}"; do
    tag="ckpt_$(basename "$CK" .pt)"
    id=$(sbatch --parsable "${dep[@]}" --array="0-$((SHARDS - 1))%$MAX_PARALLEL_TASKS" \
         --job-name="ovn-$tag" --output="$LOGS/run_${tag}_%A_%a.log" "$E/overnight/run.sbatch" "$CK")
    echo "run    array $id  ($tag, $SHARDS shards)"
    runs+=("$id")
done

report=$(sbatch --parsable --dependency="afterany:$(IFS=:; echo "${runs[*]}")" \
         --output="$LOGS/report_%j.log" "$E/overnight/report.sbatch")
echo "report job $report"
printf '%(%F %T)T setup=%s runs=%s report=%s\n' -1 "${setup:-done}" "${runs[*]}" "$report" >> "$STATE/jobs.log"
echo "Watch: squeue -u $USER ; logs in $LOGS ; result summary later in $E/overnight/MORNING_REPORT.md"
