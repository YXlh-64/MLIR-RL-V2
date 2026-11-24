"""Output-equivalence check on random inputs (design rule R14).

Compiles and runs two programs with the v6 lowering (same steps as
`execution._exec_child_worker`, in-process) on the same random inputs, then
compares every output with `allclose`. Used by `prefuse --check`.
This compiles and executes code: it is meant for offline validation runs,
never inside training.
"""

import os

import numpy as np

from rl_autoschedular_v6.execution import _EXEC_PASS_PIPELINE, Execution
from rl_autoschedular_v6.full_model import ensure_timed_main_abi, materialize_elided_resources
from rl_autoschedular_v6.transforms import transform_bufferize_and_lower_v


def run_outputs(code: str, seed: int) -> list[np.ndarray]:
    """Lower, JIT and run `code` once on uniform(0.5, 1.5) float inputs; return its outputs."""
    from mlir.execution_engine import ExecutionEngine
    from mlir.ir import Context, Module
    from mlir.passmanager import PassManager

    code = ensure_timed_main_abi(materialize_elided_resources(code))
    with Context():
        module = Module.parse(code)
        transform_bufferize_and_lower_v(module)
        inputs, outputs = Execution._create_params(module)
        rng = np.random.default_rng(seed)
        for array in inputs:
            if array.dtype.kind == 'f':
                array[...] = rng.uniform(0.5, 1.5, array.shape)
        args = Execution._convert_to_args(inputs, outputs)
        PassManager.parse(_EXEC_PASS_PIPELINE, module.context).run(module.operation)
        libs = [lib for lib in os.getenv('MLIR_SHARED_LIBS', '').split(',') if lib]
        ExecutionEngine(module, opt_level=3, shared_libs=libs).invoke('main', *args)
        results = outputs.get_results()
        outputs.free_outputs()
    return results


def check_equivalent(original: str, transformed: str, seed: int = 0,
                     rtol: float = 1e-3, atol: float = 1e-4) -> tuple[bool, float]:
    """(all outputs allclose, max abs error) of two programs on the same random inputs."""
    a, b = run_outputs(original, seed), run_outputs(transformed, seed)
    same_shapes = len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b))
    if not same_shapes:
        return False, float('nan')
    ok = all(np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True) for x, y in zip(a, b))
    err = max((float(np.nanmax(np.abs(x.astype(np.float64) - y))) for x, y in zip(a, b) if x.size), default=0.0)
    return ok, err
