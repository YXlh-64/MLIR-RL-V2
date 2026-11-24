#!/bin/bash
# Submit the retry stage: plan (list blocks with a timed-out method) -> retry array -> report.
#     bash .../overnight/retry_launch.sh [job ids to wait for, e.g. 19168867:19191342]
# Safe to run again any time: blocks already retried with $RETRY_TIMEOUT are not listed again.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/settings.sh"
cd "$REPO"
after=()
[ -n "${1:-}" ] && after=(--dependency="afterany:$1")
for CK in "${CHECKPOINTS[@]}"; do
    tag="ckpt_$(basename "$CK" .pt)"
    plan=$(sbatch --parsable "${after[@]}" --job-name="ovn-plan-$tag" \
           --output="$LOGS/retry_plan_${tag}_%j.log" "$E/overnight/retry.sbatch" "$CK")
    arr=$(sbatch --parsable --dependency="afterok:$plan" --array="0-$((RETRY_TASKS - 1))" \
          --job-name="ovn-retry-$tag" --output="$LOGS/retry_${tag}_%A_%a.log" "$E/overnight/retry.sbatch" "$CK")
    echo "$tag: plan $plan -> retry array $arr ($RETRY_TASKS tasks, ${RETRY_TIMEOUT}s per method)"
    arrays+=("$arr")
done
report=$(sbatch --parsable --dependency="afterany:$(IFS=:; echo "${arrays[*]}")" \
         --output="$LOGS/report_%j.log" "$E/overnight/report.sbatch")
echo "report $report"
printf '%(%F %T)T retry: plan->arrays=%s report=%s (after %s)\n' -1 "${arrays[*]}" "$report" "${1:-nothing}" >> "$STATE/jobs.log"
