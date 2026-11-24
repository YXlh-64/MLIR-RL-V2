#!/bin/bash
# Submit Track A: one array per thread setting, one task per model (models.tsv, fastest first).
#   bash launch.sh            all models, 1 thread and 16 threads
#   bash launch.sh 0          only task 0 (smoke test)
#   MEM=128G MAXPAR=4 bash launch.sh 5,6,8,10   rerun given tasks with more memory
# Safe to rerun: finished models (state/t<threads>_<model>.done) exit immediately.
set -eo pipefail
T=/scratch/mh8826/MLIR-RL/experiments/03_full_models/track_a
N=$(wc -l < "$T/models.tsv")
ARRAY="${1:-0-$((N - 1))}"
for THREADS in 1 16; do
    CPUS=$(( THREADS == 1 ? 2 : THREADS ))
    id=$(sbatch --parsable --array="$ARRAY%${MAXPAR:-6}" --mem="${MEM:-48G}" --cpus-per-task="$CPUS" --export=ALL,THREADS="$THREADS" \
         --job-name="fmA-t$THREADS" --output="$T/logs/t${THREADS}_%A_%a.log" "$T/run.sbatch")
    echo "threads=$THREADS cpus=$CPUS array=$ARRAY -> job $id"
    printf '%(%F %T)T threads=%s array=%s job=%s\n' -1 "$THREADS" "$ARRAY" "$id" >> "$T/state/jobs.log"
done
