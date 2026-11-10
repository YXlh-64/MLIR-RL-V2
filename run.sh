python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category single_operations --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category multi_operation_blocks --tensorflow-python "$TF_BENCH_PYTHON"

python -u experiments/01_performance_diagnosis/execution_profiling.py run \
  --category full_models --tensorflow-python "$TF_BENCH_PYTHON"