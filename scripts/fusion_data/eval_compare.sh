#!/bin/bash
# Full-model eval of the LATEST checkpoint of every training row (A, A', C, D, D+) and of the
# reference checkpoint v5_model_1200, then a comparison plot. Run from the repo root (login node):
#   bash scripts/fusion_data/eval_compare.sh
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
add v5_model_1200 config/fusion/v5_1200_eval_fullmodel.json "$RUNS/v5_model_1200/models" 1200
for row in A A_prime C D D_plus; do
    dir=$RUNS/fusion_${row}_agent/models
    step=$(ls "$dir"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    [[ -n $step ]] || { echo "no checkpoint for $row, skipped"; continue; }
    add "$row" "config/fusion/${row}_eval_fullmodel.json" "$dir" "$step"
done

ids=()
while IFS=$'\t' read -r label cfg dir step res; do
    id=$(bash scripts/fusion_data/submit.sh --parsable --cpus-per-task=64 --job-name="eval-$label" \
         scripts/fusion_data/sbatch/eval_fullmodel.sbatch "$cfg" "$dir" "$step" "$res")
    echo "eval $label (checkpoint $step) -> $id"; ids+=("$id")
    echo "$(date '+%F %T') eval_$label $id" >> "$RUNS/pipeline_jobs.txt"
done < "$OUT/agents.tsv"
dep=$(IFS=:; echo "${ids[*]}")
id=$(sbatch --parsable --dependency="afterany:$dep" --job-name=eval-plot scripts/fusion_data/sbatch/step.sbatch \
     python scripts/fusion_data/plot_eval_compare.py "$OUT")
echo "plot (after all evals) -> $id"
echo "$(date '+%F %T') eval_plot $id" >> "$RUNS/pipeline_jobs.txt"
echo "Results will be in $OUT  (comparison.png, comparison.md, comparison.csv)"
