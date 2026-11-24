# One CPU split for every job that executes MLIR code (training, evaluation, baselines), so that
# baselines and training time each program under the same conditions and the cores are really used:
#   EXEC_WORKERS     programs executed at the same time (v6 dask_manager thread pool)
#   OMP_NUM_THREADS  OpenMP threads inside each program (the parallel loops the agent creates)
#   EXEC_WORKERS x OMP_NUM_THREADS = cores of the job (default 8 x 8 = 64, the training allocation).
# Always set explicitly: never inherit OMP_NUM_THREADS=1 from an interactive shell
# (scripts/analysis/gap1/env.sh exports it), which would run every kernel on one core.
export OMP_NUM_THREADS="${FUSION_OMP_THREADS:-8}"
export EXEC_WORKERS="${FUSION_EXEC_WORKERS:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
