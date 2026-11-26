#!/bin/bash
# Full-model eval of the LATEST checkpoint of every training row (A, A', C, D, D+) and of the
# reference checkpoint v5_model_1200, then a comparison plot. Run from the repo root (login node):
#   bash scripts/fusion_data/eval_compare.sh
#   bash scripts/fusion_data/eval_compare.sh --v5-from <earlier OUT dir> [<its v5 eval job id, if still running>]
#        (reuse an earlier v5_model_1200 result instead of evaluating it again)
# Size per eval job (env): EVAL_CPUS (default 64), EVAL_MEM (default 256G), EVAL_TIME (default 1-00:00:00).
# Each model always runs with 8 OpenMP threads (as the baselines were timed); EVAL_CPUS/8 models run at once.
#   EVAL_CPUS=16 EVAL_MEM=48G EVAL_TIME=14:00:00 bash scripts/fusion_data/eval_compare.sh ...
# Training keeps running: the checkpoint step is fixed at submission time.
# Output: /scratch/mh8826/fusion_runs/eval_compare/<timestamp>/  (one JSON per agent, logs, plot, table)
set -eo pipefail
cd "$(dirname "$0")/../.."
RUNS=/scratch/mh8826/fusion_runs
OUT=$RUNS/eval_compare/$(date +%Y%m%d_%H%M)
mkdir -p "$OUT" logs
# v5_model_1200 in the models/model_<N>.pt layout eval.py expects (symlink, the original is untouched)
mkdir -p "$RUNS/v5_model_1200/models"
ln -sf /scratch/mh8826/checkpoints/v5_model_1200.pt "$RUNS/v5_model_1200/models/model_1200.pt"

: > "$OUT/agents.tsv"   # label  config  checkpoint_dir  step  result_json
add() { echo -e "$1\t$2\t$3\t$4\t$OUT/$1.json" >> "$OUT/agents.tsv"; }
V5_FROM=""; V5_JOB=""
if [[ ${1:-} == --v5-from ]]; then V5_FROM=${2:?earlier OUT dir}; V5_JOB=${3:-}; fi
if [[ -n $V5_FROM ]]; then   # v5 row points at the earlier result file; no new v5 eval job
    echo -e "v5_model_1200\tconfig/fusion/v5_1200_eval_fullmodel.json\t$RUNS/v5_model_1200/models\t1200\t$V5_FROM/v5_model_1200.json" >> "$OUT/agents.tsv"
else
    add v5_model_1200 config/fusion/v5_1200_eval_fullmodel.json "$RUNS/v5_model_1200/models" 1200
fi
for row in A A_prime C D D_plus; do
    dir=$RUNS/fusion_${row}_agent/models
    step=$(ls "$dir"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    [[ -n $step ]] || { echo "no checkpoint for $row, skipped"; continue; }
    add "$row" "config/fusion/${row}_eval_fullmodel.json" "$dir" "$step"
done

EVAL_CPUS=${EVAL_CPUS:-64}; EVAL_MEM=${EVAL_MEM:-256G}; EVAL_TIME=${EVAL_TIME:-1-00:00:00}
(( EVAL_CPUS % 8 == 0 )) || { echo "EVAL_CPUS must be a multiple of 8"; exit 1; }
PARALLEL=$(( EVAL_CPUS / 8 ))
export FUSION_OMP_THREADS=8 FUSION_EXEC_WORKERS=$PARALLEL
ids=()
while IFS=$'\t' read -r label cfg dir step res; do
    [[ -n $V5_FROM && $label == v5_model_1200 ]] && continue
    id=$(bash scripts/fusion_data/submit.sh --parsable --cpus-per-task="$EVAL_CPUS" --mem="$EVAL_MEM" \
         --time="$EVAL_TIME" --job-name="eval-$label" \
         scripts/fusion_data/sbatch/eval_fullmodel.sbatch "$cfg" "$dir" "$step" "$res" "$PARALLEL")
    echo "eval $label (checkpoint $step) -> $id"; ids+=("$id")
    echo "$(date '+%F %T') eval_$label $id" >> "$RUNS/pipeline_jobs.txt"
done < "$OUT/agents.tsv"
[[ -n $V5_JOB ]] && ids+=("$V5_JOB")
dep=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --dependency="afterany:$dep" --job-name=eval-plot scripts/fusion_data/sbatch/step.sbatch \
     python scripts/fusion_data/plot_eval_compare.py "$OUT")
echo "plot (after all evals) -> $id"
echo "$(date '+%F %T') eval_plot $id" >> "$RUNS/pipeline_jobs.txt"
echo "Results will be in $OUT  (comparison.png, comparison.md, comparison.csv)"
