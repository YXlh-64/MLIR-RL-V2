#!/bin/bash
# Submit a training / evaluation job with the same explicit CPU split as the baselines
# (scripts/fusion_data/sbatch/_threads.sh): EXEC_WORKERS programs at once x OMP_NUM_THREADS each.
# The split is passed with --export=ALL,..., so it overrides anything the submitting shell exports
# (e.g. OMP_NUM_THREADS=1 from scripts/analysis/gap1/env.sh).
#
# Usage (repo root): scripts/fusion_data/submit.sh <sbatch options> <job script> <job args>
#   scripts/fusion_data/submit.sh --cpus-per-task=64 --mem=16G scripts/train/train.sh config/fusion/A.json
# Change the split with FUSION_EXEC_WORKERS / FUSION_OMP_THREADS (their product must equal --cpus-per-task).
set -eo pipefail
source "$(dirname "$0")/sbatch/_threads.sh"

cpus=""
for arg in "$@"; do
    [[ $arg == --cpus-per-task=* ]] && cpus="${arg#--cpus-per-task=}"
done
if [[ -z $cpus ]]; then
    echo "give --cpus-per-task=N explicitly (needed to check the CPU split)"; exit 1
fi
if (( EXEC_WORKERS * OMP_NUM_THREADS != cpus )); then
    echo "EXEC_WORKERS ($EXEC_WORKERS) x OMP_NUM_THREADS ($OMP_NUM_THREADS) != --cpus-per-task ($cpus); set FUSION_EXEC_WORKERS / FUSION_OMP_THREADS"
    exit 1
fi
echo "submitting with EXEC_WORKERS=$EXEC_WORKERS OMP_NUM_THREADS=$OMP_NUM_THREADS on $cpus cores"
exec sbatch --export="ALL,EXEC_WORKERS=$EXEC_WORKERS,OMP_NUM_THREADS=$OMP_NUM_THREADS,MKL_NUM_THREADS=$MKL_NUM_THREADS,OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" "$@"
