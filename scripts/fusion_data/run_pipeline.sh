#!/bin/bash
# Submit the whole A, A', B, C, D, D+ pipeline at once, chained with Slurm dependencies (afterok).
# Run once, from the repo root, on a login node:   bash scripts/fusion_data/run_pipeline.sh
#
#   Row A : baselines_blocks -> split_json -> train A
#   Units : build torch-mlir -> export -> validate -> units -> split -> baselines_units -> bench-json
#           -> train A', C, D, D+        (+ profile after validate, + config B after units)
#   Eval  : baselines_eval (independent) -> eval_fullmodel.json for the *_eval_fullmodel configs
# Every job id goes to /scratch/mh8826/fusion_runs/pipeline_jobs.txt; see status.sh for progress.
# A failed job leaves its dependents pending with reason DependencyNeverSatisfied (squeue).
set -eo pipefail
cd "$(dirname "$0")/../.."
export TORCH_MLIR_COMMIT=${TORCH_MLIR_COMMIT:-43506726853b35ae9c253aa1d1c61b76ad9b4c13}
RUNS=/scratch/mh8826/fusion_runs
JOBS=$RUNS/pipeline_jobs.txt
mkdir -p "$RUNS" logs
SB=scripts/fusion_data/sbatch
UNITS_BASE=results/fusion_units_results/baselines/mlir
BLOCKS_BASE=results/ops_and_blocks_results/baselines/mlir

submit() {   # submit NAME <sbatch args...>  -> sets LAST to the job id
    LAST=$(sbatch --parsable "${@:2}")
    echo "$(date '+%F %T') $1 $LAST" >> "$JOBS"
    echo "$1 -> $LAST"
}
train() {    # train ROW DEPENDENCY  (explicit 8 x 8 CPU split via submit.sh)
    LAST=$(bash scripts/fusion_data/submit.sh --parsable --dependency="afterok:$2" --job-name="train-$1" \
           --cpus-per-task=64 --mem=16G scripts/train/train.sh "config/fusion/$1.json")
    echo "$(date '+%F %T') train_$1 $LAST" >> "$JOBS"
    echo "train_$1 -> $LAST"
}

MODE=${1:-all}   # all | --units-only (resubmit only the units chain: build ... train A', C, D, D+, config B)
echo "$(date '+%F %T') ---- run_pipeline.sh $MODE (TORCH_MLIR_COMMIT=$TORCH_MLIR_COMMIT)" >> "$JOBS"

if [[ $MODE != --units-only ]]; then
# Row A: wait for any baselines_blocks job still running (it writes the same output file).
running=$(squeue -u "$USER" -h -n fd-base-blocks -o %i | paste -sd: -)
submit baselines_blocks ${running:+--dependency=afterany:$running} $SB/baselines_blocks.sbatch;  BB=$LAST
submit split_blocks --dependency="afterok:$BB" $SB/step.sbatch \
    python scripts/data/split_json.py --input "$BLOCKS_BASE/base.json";                             SPB=$LAST
train A "$SPB"
fi

# Units rows.
submit build_torch_mlir --export=ALL $SB/build_torch_mlir.sbatch;                                  BUILD=$LAST
submit compat_test --dependency="afterok:$BUILD" $SB/compat_test.sbatch
submit export --dependency="afterok:$BUILD" $SB/export.sbatch;                                     EXP=$LAST
submit validate --dependency="afterok:$EXP" $SB/validate.sbatch;                                   VAL=$LAST
submit profile --dependency="afterok:$VAL" $SB/profile.sbatch
submit units --dependency="afterok:$VAL" $SB/units.sbatch;                                         UNI=$LAST
submit split_units --dependency="afterok:$UNI" $SB/step.sbatch python scripts/fusion_data/split.py split;  SPU=$LAST
submit baselines_units --dependency="afterok:$UNI" $SB/baselines_units.sbatch;                     BU=$LAST
submit bench_json --dependency="afterok:$BU:$SPU" $SB/step.sbatch \
    python scripts/fusion_data/split.py bench-json --times "$UNITS_BASE/all_units.json";            BJ=$LAST
for row in A_prime C D D_plus; do train "$row" "$BJ"; done
submit config_b --dependency="afterok:$UNI" $SB/config_b.sbatch

# Full-model eval baselines (needed later by the *_eval_fullmodel configs).
[[ $MODE != --units-only ]] && submit baselines_eval $SB/baselines_eval.sbatch

echo "All submitted; ids in $JOBS. Progress: bash scripts/fusion_data/status.sh"
