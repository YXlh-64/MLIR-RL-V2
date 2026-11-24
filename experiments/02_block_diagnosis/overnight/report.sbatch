#!/bin/bash
#SBATCH --job-name=ovn-report
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

# Overnight stage 3 (runs after all run jobs, even failed ones): collect, statistics, plots,
# experiment 01's report and plot, and overnight/MORNING_REPORT.md. Safe to resubmit any time.
set -o pipefail
cd "${SLURM_SUBMIT_DIR:?submit through overnight.sh}"
source experiments/02_block_diagnosis/overnight/settings.sh
source "$E/overnight/env.sh"

for d in "$E"/results/ckpt_*/; do
    TAG="$(basename "$d")"
    echo "===== $TAG"
    python "$E/run_blocks.py" --tag "$TAG" --collect-only || continue
    for c in "$d"t1_warm/r0/*/; do        # experiment 01's REPORT.md / comparison.csv per class
        python -c "import sys; sys.path.insert(0,'experiments/01_performance_diagnosis'); import run_suite; run_suite.collect_report(__import__('pathlib').Path(sys.argv[1]))" "$c" || true
    done
    python "$E/stats.py" --all-results "$d/all_results.csv" || echo "stats failed for $TAG"
    python "$E/plot_blocks.py" --tag "$TAG" --config t1_warm || echo "plot_blocks failed for $TAG"
    python experiments/01_performance_diagnosis/plot_speedups.py --results-dir "$d/t1_warm/r0" \
        || echo "plot_speedups failed for $TAG"
done
python "$E/overnight/map_duplicates.py" $(cd "$E/results" && ls -d ckpt_* 2>/dev/null)
python "$E/overnight/morning_report.py"
echo "Report: $E/overnight/MORNING_REPORT.md"
