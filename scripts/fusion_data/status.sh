#!/bin/bash
# Progress of the A-D+ pipeline (read-only). Run from the repo root:  bash scripts/fusion_data/status.sh
cd "$(dirname "$0")/../.."
RUNS=/scratch/mh8826/fusion_runs
JOBS=$RUNS/pipeline_jobs.txt

echo "== jobs (latest submission of each step)"
if [[ -f $JOBS ]]; then
    ids=$(grep -v -- '----' "$JOBS" | awk '{last[$3]=$4} END {for (s in last) print last[s]}' | paste -sd, -)
    [[ -n $ids ]] && sacct -j "$ids" -X -n --format=JobID%10,JobName%22,State%14,Elapsed,ExitCode | sort
else
    echo "no $JOBS yet (run scripts/fusion_data/run_pipeline.sh)"
fi
squeue -u "$USER" -h -o "%.10i %.22j %.8T %.10M %R" | grep -v interactive

count() {   # count FILE: ok / failed entries of a {name: ns} baseline JSON, and when it was last written
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); ok=sum(v>0 for v in d.values()); print(f'{ok} ok, {len(d)-ok} failed, {len(d)} total', end='')" "$1" 2>/dev/null \
        && echo "  (last written $(date -r "$1" '+%F %T'))" || echo "not yet"
}
echo; echo "== data"
echo "raw exports      : $(ls data/fusion_units/raw/*_linalg.mlir 2>/dev/null | wc -l) / $(python3 -c "import json; print(sum(m['enabled'] for m in json.load(open('data/fusion_units/models.json'))['models']))")"
echo "validated ok     : $(python3 -c "import json; d=json.load(open('data/fusion_units/validation.json')); print(sum(r['ok'] for r in d.values()), '/', len(d))" 2>/dev/null || echo 'not yet')"
echo "units            : $(ls data/fusion_units/units/*_unit_*.mlir 2>/dev/null | wc -l)"
echo "blocks baselines : $(count results/ops_and_blocks_results/baselines/mlir/base.json)"
echo "units baselines  : $(count results/fusion_units_results/baselines/mlir/all_units.json)"
echo "eval baselines   : $(count results/fusion_units_results/baselines/mlir/eval_fullmodel_times.json)"
for f in results/ops_and_blocks_results/baselines/mlir/base_train.json results/fusion_units_results/baselines/mlir/base_train.json results/fusion_units_results/baselines/mlir/eval_fullmodel.json; do
    [[ -f $f ]] && echo "ready            : $f ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f") entries)"
done

echo; echo "== training (checkpoints every 50 iterations)"
for row in A A_prime C D D_plus; do
    dir=$RUNS/fusion_${row}_agent
    n=$(ls "$dir"/models/model_*.pt 2>/dev/null | wc -l)
    last=$(ls "$dir"/models/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
    printf "%-8s checkpoints=%-4s latest=%-7s %s\n" "$row" "$n" "${last:--}" "$([[ -d $dir ]] && echo "$dir" || echo 'not started')"
done
