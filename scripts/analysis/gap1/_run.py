"""Shared helper for the B scripts: compile and run MLIR with the repo's v5 lowering (no agent/policy).

Same steps as rl_autoschedular_v5/execution.py::_exec_child_worker, but in-process and returning
every run time plus the outputs of the first run. Needs the MLIR env (.env loaded).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / 'rl_autoschedular')]

import numpy as np  # noqa: E402
from mlir.execution_engine import ExecutionEngine  # noqa: E402
from mlir.ir import Context, Module  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402
from rl_autoschedular_v5.execution import _EXEC_PASS_PIPELINE, Execution  # noqa: E402
from rl_autoschedular_v5.full_model import ensure_timed_main_abi, materialize_elided_resources  # noqa: E402
from rl_autoschedular_v5.transforms import transform_bufferize_and_lower_v  # noqa: E402


def prepare(code):
    """Full models -> timed @main ABI with elided weights materialized (no-op for blocks)."""
    return ensure_timed_main_abi(materialize_elided_resources(code))


def run(code, runs=2, seed=None):
    """Lower + JIT + invoke `runs` times. seed=None: zero inputs (as v5); else uniform(0.5, 1.5) floats.
    Returns (times_ns, outputs_of_first_run)."""
    with Context():
        module = Module.parse(code)
        transform_bufferize_and_lower_v(module)
        inputs, outs = Execution._create_params(module)
        if seed is not None:
            rng = np.random.default_rng(seed)
            for a in inputs:
                if a.dtype.kind == 'f':
                    a[...] = rng.uniform(0.5, 1.5, a.shape)
        args = Execution._convert_to_args(inputs, outs)
        PassManager.parse(_EXEC_PASS_PIPELINE, module.context).run(module.operation)
        libs = [s for s in os.getenv('MLIR_SHARED_LIBS', '').split(',') if s]
        engine = ExecutionEngine(module, opt_level=3, shared_libs=libs)
        times, results = [], None
        for _ in range(runs):
            engine.invoke('main', *args)
            if results is None:
                results = outs.get_results()
            times.append(outs.delta)
            outs.free_outputs()
    return times, results


def compare(a, b, rtol, atol):
    """(all outputs allclose, max abs error)."""
    ok = len(a) == len(b) and all(x.shape == y.shape and np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
                                  for x, y in zip(a, b))
    err = max((float(np.nanmax(np.abs(x.astype(np.float64) - y))) for x, y in zip(a, b) if x.shape == y.shape and x.size),
              default=float('nan'))
    return ok, err
