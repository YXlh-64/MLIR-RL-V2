#!/bin/bash
# Run A1-A7, then write plots + results/gap1/REPORT_A.md.
# Usage (repo root, on a compute node): bash scripts/analysis/gap1/run_all_a.sh [--skip-llama]
set -e
FULL=${FULL:-/scratch/mh8826/data/full_models}
BLOCKS=${BLOCKS:-data/ops_and_blocks}
R=${R:-results/gap1}
G=scripts/analysis/gap1
MODELS="$FULL/*_linalg.mlir"
[ "$1" = "--skip-llama" ] && MODELS=$(ls $FULL/*_linalg.mlir | grep -v llama)

echo "== A1" && python3 $G/a1_producer_counts.py $MODELS --out $R/a1
echo "== A2" && python3 $G/a2_multi_producers_in_blocks.py --models $MODELS --blocks "$BLOCKS/*_block_*.mlir" --out $R/a2
echo "== A3" && python3 $G/a3_block_membership.py --models $MODELS --blocks "$BLOCKS/*_block_*.mlir" --out $R/a3
echo "== A4" && python3 $G/a4_path_cap.py $MODELS --max-depth 64 --max-paths 5000 --out $R/a4
echo "== A5" && python3 $G/a5_edge_classes.py $MODELS --out $R/a5
echo "== A6" && python3 $G/a6_shape_drift.py --models $MODELS --blocks "$BLOCKS/*_block_*.mlir" --out $R/a6
echo "== A7" && python3 $G/a7_merged_group_limits.py $MODELS --config config/v5/v5_mlir_rl_v1_paper.json --out $R/a7

echo "== plots + report" && source $G/env.sh && python $G/plot_results.py --results $R --only a
