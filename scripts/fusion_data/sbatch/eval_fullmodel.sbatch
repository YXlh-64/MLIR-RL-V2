#!/bin/bash
#SBATCH --job-name=fd-eval-full
#SBATCH --partition=compute
#SBATCH --constraint=bergamo
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/fd_eval_full_%j.out
#SBATCH --error=logs/fd_eval_full_%j.err
# Greedy full-model evaluation of ONE checkpoint on the runnable eval models (eval_json_file of CONFIG).
# Same logic as scripts/eval/eval_fullmodel.sh (one eval.py process per model, per-model config and
# results_dir, merge), but with this repo's environment (_env.sh: MLIR-ENV via conda, explicit CPU split).
# eval_fullmodel.sh itself is not usable here: it sources "$CONDA_ENV" as an activate script and writes
# its logs to another user's folder.
#
#   sbatch scripts/fusion_data/sbatch/eval_fullmodel.sbatch CONFIG CHECKPOINTS_DIR STEP OUT_JSON [PARALLEL]
# Writes OUT_JSON = {model: exec_time_ns or null}. PARALLEL models run at once (default 8 = 64 cores / 8 OMP).
set -eo pipefail   # no -u: conda activate reads unset variables
source scripts/fusion_data/sbatch/_env.sh
CONFIG=${1:?config}; CKPT_DIR=${2:?checkpoints dir}; STEP=${3:?checkpoint step}; OUT=${4:?output json}
PARALLEL=${5:-8}
export AUTOSCHEDULER_IMPL=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['implementation'])" "$CONFIG")
export EVAL_DIR="$CKPT_DIR" EVAL_START="$STEP" EVAL_END="$STEP" EVAL_STRIDE=1 EVAL_FORCE=1
export MIN_EXEC_TIMEOUT=300 MAX_EXEC_TIMEOUT=18000 COMPILE_MIN_TIMEOUT=14400 PYTHONFAULTHANDLER=1
EVAL_JSON=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['eval_json_file'])" "$CONFIG")
TMP=$(mktemp -d "/scratch/mh8826/fusion_runs/.eval_tmp_${SLURM_JOB_ID}_XXXX")
echo "config=$CONFIG checkpoint=$CKPT_DIR/model_$STEP.pt out=$OUT parallel=$PARALLEL tmp=$TMP"
[[ -f $CKPT_DIR/model_$STEP.pt ]] || { echo "missing $CKPT_DIR/model_$STEP.pt"; exit 1; }

run_model() {   # one model: private config + results_dir, one eval.py process
    local m=$1
    python - "$CONFIG" "$EVAL_JSON" "$m" "$TMP" <<'PY'
import json, sys
cfg_path, benches_file, m, tmp = sys.argv[1:5]
cfg = json.load(open(cfg_path))
one = {m: json.load(open(benches_file))[m]}
json.dump(one, open(f"{tmp}/{m}.bench.json", "w"))
cfg.update(json_file=f"{tmp}/{m}.bench.json", eval_json_file=f"{tmp}/{m}.bench.json", results_dir=f"{tmp}/agent_{m}")
json.dump(cfg, open(f"{tmp}/{m}.config.json", "w"))
PY
    echo "[$(date +%T)] start $m"
    if CONFIG_FILE_PATH="$TMP/$m.config.json" python scripts/eval/eval.py > "$TMP/$m.log" 2>&1; then
        cp "$TMP/agent_$m/logs/eval/eval_exec_times.json" "$TMP/$m.result.json" && echo "[$(date +%T)] done  $m"
    else
        echo "[$(date +%T)] FAIL  $m (exit $?; log kept in ${OUT%.json}_logs/$m.log)"
    fi
}

for m in $(python -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))))" "$EVAL_JSON"); do
    run_model "$m" &
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n || true; done
done
wait || true

mkdir -p "$(dirname "$OUT")" "${OUT%.json}_logs"
cp "$TMP"/*.log "${OUT%.json}_logs/" 2>/dev/null || true
python - "$EVAL_JSON" "$TMP" "$OUT" <<'PY'
import json, sys, glob, os
benches, tmp, out = sys.argv[1:4]
base = json.load(open(benches))
res = {m: None for m in base}
for f in glob.glob(os.path.join(tmp, "*.result.json")):
    for k, v in json.load(open(f)).items():
        res[k] = v[0] if isinstance(v, list) and v else v   # eval_exec_times may hold a list
json.dump(res, open(out, "w"), indent=2)
for m, t in res.items():
    print(f"  {m:28s} " + (f"{base[m] / t:7.2f}x" if t and t > 0 else "  FAILED"))
PY
rm -rf "$TMP"
echo "done $(date)"
