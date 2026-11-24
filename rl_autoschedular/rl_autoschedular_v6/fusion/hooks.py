"""Applies the enabled fusion modules to loaded benchmarks (integration point).

Called once at the end of `Benchmarks.__init__` when any module is on.
Order: prefuse (M2) -> re-extract features from the fused code -> producer
selection statistics (M4, read-only; the selection itself happens in Env).
Features are always re-extracted after prefusion, because the ones built by
`Benchmarks.__init__` describe the unfused program. `root_exec_time` is kept
(R12: speedups stay relative to the original program).
"""

from rl_autoschedular_v6.fusion.config import FusionConfig
from rl_autoschedular_v6.fusion.prefuse import prefuse_code
from rl_autoschedular_v6.fusion.producer_select import selection_stats
from rl_autoschedular_v6.fusion.views import BackendError
from rl_autoschedular_v6.full_model import ensure_timed_main_abi
from rl_autoschedular_v6.state import BenchmarkFeatures, extract_bench_features_from_code
from rl_autoschedular_v6.utils.log import print_alert, print_info


def prefuse_benchmark(bench: BenchmarkFeatures) -> BenchmarkFeatures:
    """Fused copy of `bench`, with features re-extracted from the fused code."""
    fused_code, _ = prefuse_code(bench.code)
    fused = extract_bench_features_from_code(bench.bench_name, fused_code, bench.root_exec_time)
    fused.code = ensure_timed_main_abi(fused.code)
    return fused


def apply_fusion(benchmarks: list[BenchmarkFeatures], cfg: FusionConfig) -> list[BenchmarkFeatures]:
    """Return the benchmarks with every enabled module applied (input list is not modified)."""
    result = []
    for bench in benchmarks:
        if cfg.prefuse:
            try:
                bench = prefuse_benchmark(bench)
            except BackendError as e:
                print_alert(f'Prefuse skipped for {bench.bench_name} (kept unfused): {e}')
        result.append(bench)
    ops_before = sum(len(b.operation_tags) for b in benchmarks)
    ops_after = sum(len(b.operation_tags) for b in result)
    print_info(f'Fusion layer {cfg}: {len(result)} benchmarks, ops {ops_before} -> {ops_after}')
    if cfg.producer_select:
        print_info(f'Producer selection (R8), initial op states: {selection_stats(result).as_dict()}')
    return result
