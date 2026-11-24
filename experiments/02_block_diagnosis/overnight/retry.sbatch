#!/bin/bash
#SBATCH --job-name=ovn-retry
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append

# Overnight stage 2b: re-run blocks where a method hit the per-method time limit.
#   sbatch retry.sbatch <checkpoint .pt>                      (no array) -> PLAN: write state/retry_list.txt
#   sbatch --array=0-<N-1> retry.sbatch <checkpoint .pt>     -> task k re-runs list[k::N]
# A block is listed when its LATEST run has a method with status "timeout" and that run used a
# limit below $RETRY_TIMEOUT. After a retry the latest run has the longer limit, so it is not
# listed again: re-planning / resubmitting only picks up what is still missing.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit through overnight.sh or retry_launch.sh}"
source experiments/02_block_diagnosis/overnight/settings.sh
source "$E/overnight/env.sh"

CK="$1"
TAG="ckpt_$(basename "$CK" .pt)"
LIST="$STATE/retry_list_${TAG}.txt"
CFG="$(awk -F'\t' -v t="$TAG" '$1 == t {print $3}' "$STATE/checkpoints.tsv")"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    python - "$E/results/$TAG/t1_warm/r0" "$RETRY_TIMEOUT" > "$LIST" <<'PY'
import json, sys
from pathlib import Path
root, limit = Path(sys.argv[1]), int(sys.argv[2])
latest = {}
for p in sorted(root.glob('*/artifacts/*/*/summary.json')):
    latest[p.parents[1].name] = p
for name, p in sorted(latest.items()):
    try:
        s = json.loads(p.read_text())
        timeout = json.loads((p.parent / 'manifest.json').read_text()).get('timeout_seconds', 0)
    except (OSError, json.JSONDecodeError):
        continue
    if timeout < limit and any(r.get('status') == 'timeout' for r in s['results'].values()):
        print(name)
PY
    printf '[%(%F %T)T] planned %s blocks to retry with a %s s limit -> %s\n' -1 "$(wc -l < "$LIST")" "$RETRY_TIMEOUT" "$LIST"
    cat "$LIST"
    exit 0
fi

K="$SLURM_ARRAY_TASK_ID"; N="${SLURM_ARRAY_TASK_COUNT:?}"
mapfile -t BLOCKS < <(awk -v k="$K" -v n="$N" '(NR - 1) % n == k' "$LIST")
printf '[%(%F %T)T] retry task %s/%s node=%s blocks=%s: %s\n' -1 "$K" "$N" "$(hostname)" "${#BLOCKS[@]}" "${BLOCKS[*]}"
[ "${#BLOCKS[@]}" -gt 0 ] || exit 0
pending() {   # this task's blocks that have no run newer than the plan (i.e. not retried yet)
    for b in "${BLOCKS[@]}"; do
        [ -n "$(find "$E/results/$TAG/t1_warm/r0" -path "*/artifacts/$b/*/summary.json" -newer "$LIST" | head -1)" ] || echo "$b"
    done
}
for attempt in 1 2 3; do       # srun launch can fail transiently: retry only what is still pending
    mapfile -t TODO < <(pending)
    [ "${#TODO[@]}" -eq 0 ] && break
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
        python -u "$E/run_blocks.py" --tag "$TAG" --checkpoint "$CK" --config "$CFG" \
        --threads 1 --repeat 0 --methods "${METHODS[@]}" --rl-apply "$RL_APPLY" \
        --timeout-seconds "$RETRY_TIMEOUT" --rerun --only "${TODO[@]}" \
        || echo "run_blocks exited non-zero (a block failed or srun did not launch)"
    sleep 30
done
printf '[%(%F %T)T] retry task %s finished\n' -1 "$K"
