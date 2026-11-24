#!/bin/bash
#SBATCH --job-name=ovn-run
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append

# Overnight stage 2 (Slurm array): shard <task id> of the checkpoint's block selection
# through experiment 01's harness with all four methods, 1 pinned thread (like experiment 01).
# Usage: sbatch --array=0-<SHARDS-1>%<N> run.sbatch <checkpoint .pt>
# Shard k runs blocks k, k+SHARDS, k+2*SHARDS, ... of the sorted selection (mixes classes).
# Resumable: a block with a summary.json is skipped, so resubmitting the array continues.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit through overnight.sh}"
source experiments/02_block_diagnosis/overnight/settings.sh
source "$E/overnight/env.sh"

CK="$1"
K="${SLURM_ARRAY_TASK_ID:?submit as an array}"
TAG="ckpt_$(basename "$CK" .pt)"
LINE="$(awk -F'\t' -v t="$TAG" '$1 == t' "$STATE/checkpoints.tsv" 2>/dev/null || true)"
if [ -z "$LINE" ]; then
    echo "SKIP $TAG: setup found no config that loads this checkpoint (see setup log, step 7/8)."
    exit 0
fi
CFG="$(cut -f3 <<<"$LINE")"
SEL="$E/results/$TAG/selection.json"
mapfile -t BLOCKS < <(python -c "
import json, sys
names = sorted(b['name'] for b in json.load(open(sys.argv[1]))['blocks'])
print('\n'.join(names[int(sys.argv[2])::int(sys.argv[3])]))" "$SEL" "$K" "$SHARDS")
printf '[%(%F %T)T] job=%s task=%s node=%s tag=%s blocks=%s config=%s\n' -1 \
    "$SLURM_ARRAY_JOB_ID" "$K" "$(hostname)" "$TAG" "${#BLOCKS[@]}" "$CFG"
[ "${#BLOCKS[@]}" -gt 0 ] || exit 0

missing() {   # blocks of this shard without a finished run in the current RL mode
    python -c "
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / 't1_warm' / 'r0'
n = 0
for name in sys.argv[3:]:
    runs = sorted(p.parent for p in root.glob(f'*/artifacts/{name}/*/summary.json'))
    mode = json.loads((runs[-1] / 'manifest.json').read_text()).get('rl_apply', 'strict') if runs else None
    n += mode != sys.argv[2]
print(n)" "$E/results/$TAG" "$RL_APPLY" "${BLOCKS[@]}"
}
# srun can fail to launch the step (seen once: "timeout waiting for task launch"); retry.
for attempt in 1 2 3; do
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python -u "$E/run_blocks.py" --tag "$TAG" --checkpoint "$CK" --config "$CFG" \
        --threads 1 --repeat 0 --methods "${METHODS[@]}" --rl-apply "$RL_APPLY" --only "${BLOCKS[@]}" \
        || echo "run_blocks exited non-zero (a block failed or srun did not launch)."
    left=$(missing)
    printf '[%(%F %T)T] attempt %s: %s of %s blocks still without results\n' -1 "$attempt" "$left" "${#BLOCKS[@]}"
    [ "$left" -eq 0 ] && break
    sleep 60
done
if [ "$left" -eq 0 ]; then
    touch "$STATE/run_${TAG}_shard${K}.done"
    printf '[%(%F %T)T] shard %s finished\n' -1 "$K"
else
    printf '[%(%F %T)T] shard %s INCOMPLETE (%s blocks missing): rerun overnight.sh\n' -1 "$K" "$left"
    exit 1
fi
