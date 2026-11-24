#!/usr/bin/env python3
"""Run one block workload with experiment 01's harness, unchanged.

experiment 01 (experiment.py) already does everything we need: seeded inputs,
one fresh worker process per method, CPU pinning, warmup + 100 timed calls,
correctness against eager Torch, schedule.json, summary.json/csv.
This file only adds three things, without editing experiment.py:

  1. BlockTorchRunner: experiment.py's TorchRunner with the larger op table from
     block_to_torch.torch_operations() (exp, rsqrt, where, reductions, ...).
     The constructor body is the same as hers; only `operations` differs.
  2. Optional cold-cache mode (manifest key `cache_flush_mib`): after every
     input reset (outside the timer) a buffer larger than the L3 is rewritten,
     so each timed call starts with cold caches. Used by experiment D4.
  3. A `_worker` entry point. experiment.run_experiment() starts each worker as
     `python <experiment module __file__> _worker ...`; run() points that path at
     this file for the duration of the call, so the workers apply 1-2 and then
     call her experiment.worker() exactly as her own entry point does.

Manual schedule variants (experiment D1) need nothing extra: her `manual`
manifest entries already measure any pre-scheduled .mlir with the same pipeline.
"""
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP01 = HERE.parent / '01_performance_diagnosis'
for p in (str(HERE), str(EXP01)):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiment          # noqa: E402  experiment 01, imported as a library (not modified)
import block_to_torch      # noqa: E402

_FLUSH = {}


class BlockTorchRunner(experiment.TorchRunner):
    """Same as experiment.TorchRunner.__init__, with the block op table."""

    def __init__(self, c, method, out):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np, self.torch, self.spec, self.times, self.out = np, torch, c['torch'], {}, out
        torch.set_num_threads(c['threads'])
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = True
        torch.set_float32_matmul_precision('highest')
        operations = block_to_torch.torch_operations(torch, F)
        names = [s['name'] for s in c['inputs']]
        constants = {}
        if self.spec.get('constants'):
            with np.load(self.spec['constants'], allow_pickle=False) as weights:
                constants = {k: torch.from_numpy(weights[k].copy()) for k in weights.files}
            if set(names) & constants.keys():
                raise ValueError('Constant names must differ from input names')
        defined = set(names) | constants.keys()
        for node in self.spec['graph']:
            if node['op'] not in operations or node['id'] in defined:
                raise ValueError(f"Unknown Torch graph operation or duplicate value: {node['op']}")
            if any(n not in defined for n in node['args']):
                raise ValueError('Torch graph arguments must refer to inputs or earlier nodes')
            defined.add(node['id'])
        if not self.spec['outputs'] or any(n not in defined for n in self.spec['outputs']):
            raise ValueError('Invalid Torch graph outputs')

        def graph(*args):
            values = dict(constants)
            values.update(zip(names, args))
            for node in self.spec['graph']:
                kwargs = dict(node.get('kwargs', {}))
                if isinstance(kwargs.get('dtype'), str):
                    kwargs['dtype'] = getattr(torch, kwargs['dtype'])
                operands = [values[n] for n in node['args']]
                values[node['id']] = (torch.cat(operands, **kwargs) if node['op'] == 'cat'
                                      else operations[node['op']](*operands, **kwargs))
            return tuple(values[n] for n in self.spec['outputs'])

        self.fn = graph
        t = time.perf_counter_ns()
        if method == 'torch_inductor':
            self.fn = torch.compile(graph, backend='inductor', fullgraph=True, dynamic=False)
        self.times['compile_wrapper_creation_ns'] = time.perf_counter_ns() - t


def _flush_caches():
    import numpy as np
    buffer = _FLUSH['buffer']
    np.add(buffer, 1.0, out=buffer)       # read + write every cache line


def with_cache_flush(cls):
    class Flushing(cls):
        def reset(self, arrays):
            super().reset(arrays)          # restore inputs (outside the timer) ...
            _flush_caches()                # ... then evict them: the next call starts cold
    Flushing.__name__ = 'Flushing' + cls.__name__
    return Flushing


_RL_LOG = {'mode': 'strict', 'failed': [], 'skipped': [], 'applied': 0}


def install_lenient_rl(impl):
    """Apply the RL schedule the way the RL environment does (manifest rl_apply='lenient').

    experiment.policy_schedule() applies every action and fails the whole mlir_rl method
    on the first transform error. On multi-op blocks some TiledFusion actions fail to
    apply, so strict mode reports an error instead of a schedule. MLIR-RL's own Env
    (env.py __apply_sequence) instead leaves the code unchanged when an action fails and
    skips the rest of that operation's actions. This reproduces that rule, ONLY for the
    final application inside policy_schedule (detected from the caller), and records every
    failed / skipped action in rl_application.json next to the method's result.
    """
    import importlib
    base = importlib.import_module(f'{impl}.actions.base')
    original = base.Action.apply
    failed_ops = set()
    _RL_LOG['mode'] = 'lenient'

    def apply(self, module):
        if sys._getframe(1).f_code.co_name != 'policy_schedule':
            return original(self, module)           # Env during inference: unchanged
        op = getattr(self, 'operation_tag', None)
        if op in failed_ops:
            _RL_LOG['skipped'].append(dict(operation=op, action=str(self)))
            return None
        try:
            result = original(self, module)          # transform runs in a subprocess and
            _RL_LOG['applied'] += 1                  # only replaces the module on success
            return result
        except Exception as error:
            failed_ops.add(op)
            _RL_LOG['failed'].append(dict(operation=op, action=str(self),
                                          error=str(error).strip().splitlines()[0][:400] if str(error).strip() else type(error).__name__))
            return None
    base.Action.apply = apply


def _lenient_policy_schedule(original):
    """Wrap experiment.policy_schedule: importing the v5 actions needs CONFIG_FILE_PATH,
    which policy_schedule only sets itself. Write the identical policy_config.json first
    (same updates as experiment.policy_schedule), patch Action.apply, then run the original,
    which rewrites the same file and does everything else unchanged."""
    import os

    def wrapped(code, c, out, compile_times):
        rl = c['rl']
        impl = rl.get('implementation', 'rl_autoschedular_v5')
        cfg = json.loads(Path(rl['config']).read_text())
        cfg.update(dask_node_count=0, results_dir=str(out / 'policy_logs'))
        experiment.write_json(out / 'policy_config.json', cfg)
        os.environ['CONFIG_FILE_PATH'] = str(out / 'policy_config.json')
        os.environ['DASK_NODES'] = '0'
        os.environ['DASK_TRAINING_ONLY'] = '1'
        install_lenient_rl(impl)
        return original(code, c, out, compile_times)
    return wrapped


def install(c):
    """Patch experiment's runner classes inside a worker process."""
    if c.get('rl_apply', 'strict') == 'lenient':
        experiment.policy_schedule = _lenient_policy_schedule(experiment.policy_schedule)
    experiment.TorchRunner = BlockTorchRunner
    mib = c.get('cache_flush_mib')
    if mib:
        import numpy as np
        _FLUSH['buffer'] = np.zeros(int(mib) * 2**20 // 8)
        experiment.MLIRRunner = with_cache_flush(experiment.MLIRRunner)
        experiment.TorchRunner = with_cache_flush(experiment.TorchRunner)


def run(manifest, methods, output):
    """experiment.run_experiment() with workers started through this file."""
    original = experiment.__file__
    experiment.__file__ = str(Path(__file__).resolve())
    try:
        return experiment.run_experiment(manifest, methods, output)
    finally:
        experiment.__file__ = original


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '_worker':
        manifest, method, out = sys.argv[2], sys.argv[3], Path(sys.argv[4])
        try:
            c = json.loads(Path(manifest).read_text())
            install(c)
            try:
                result = experiment.worker(c, method, out, Path(manifest).parent)
            finally:
                if method == 'mlir_rl':
                    experiment.write_json(out / 'rl_application.json', _RL_LOG)
            experiment.write_json(out / 'result.json', result)
        except Exception as e:
            traceback.print_exc()
            experiment.write_json(out / 'result.json', {'status': 'error', 'reason': str(e)})
            return 1
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == 'run':
        # Single manifest file, e.g. one written by run_blocks.py --prepare-only.
        return run(sys.argv[2], sys.argv[3:] or None, str(HERE / 'results' / 'manual_runs'))
    print(__doc__)
    print('Usage: block_experiment.py run <manifest.json> [methods...]')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
