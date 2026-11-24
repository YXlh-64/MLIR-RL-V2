#!/bin/bash
#SBATCH --job-name=gap1-b
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/gap1/logs/%x_%j.log
# Run B1-B5 one after another, then write plots + results/gap1/REPORT_B.md.
# A failing check is logged and the job moves on to the next one.
# Usage (repo root): mkdir -p results/gap1/logs && sbatch scripts/analysis/gap1/run_all_b.sbatch
# Optional: CKPT=<.pt> (default v5_model_1200), SCHED_BENCHES="<globs>", or EXP=results/<exp>_agent to use its exec cache.
cd "$SLURM_SUBMIT_DIR"
source scripts/analysis/gap1/env.sh
FULL=${FULL:-/scratch/mh8826/data/full_models}
BLOCKS=${BLOCKS:-data/ops_and_blocks}
R=${R:-results/gap1}
G=scripts/analysis/gap1

echo "== B1" && python $G/b1_timing_noise.py "$BLOCKS/resnet50_block_1*.mlir" "$BLOCKS/bert_block_1*.mlir" --runs 20 --repeats 3 --out $R/b1
# B2 replays TPF schedules: from an experiment's exec cache (EXP=...) or, by default,
# generated here by running the checkpoint greedily (CKPT) on the SCHED_BENCHES blocks.
CKPT=${CKPT:-/scratch/mh8826/checkpoints/v5_model_1200.pt}
SCHED_BENCHES=${SCHED_BENCHES:-"$BLOCKS/resnet50_block_*.mlir $BLOCKS/gpt2_block_*.mlir"}
if [ -n "$EXP" ]; then
  SCHED=$EXP/logs/exec_data.json
else
  SCHED=$R/schedules/exec_data.json
  echo "== schedules from $CKPT" && python $G/gen_schedules.py $SCHED_BENCHES --checkpoint $CKPT --out $SCHED
fi
echo "== B2" && python $G/b2_tpf_correctness.py --exec-data $SCHED --benchmarks-dir $BLOCKS --per-bench 1 --out $R/b2
echo "== B3" && python $G/b3_diamond_fusion.py --out $R/b3
echo "== B4" && python $G/b4_elementwise_prefusion.py "$BLOCKS/resnet50_block_*.mlir" --runs 5 --out $R/b4
echo "== B4 green" && python $G/b4_elementwise_prefusion.py "$BLOCKS/resnet50_block_*.mlir" --green-only --runs 5 --out $R/b4_green
echo "== B5" && python $G/b5_additivity.py "$FULL/resnet50_linalg.mlir" --runs 3 --out $R/b5

echo "== plots + report" && python $G/plot_results.py --results $R --only b
