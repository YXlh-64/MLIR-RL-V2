#!/usr/bin/env python3
"""One-workload CPU MLIR/RL/PyTorch execution and measurement.

For automatic category selection, use run_suite.py (see README.md).
For an optional custom workload, first create/edit an input manifest:
  python experiments/01_performance_diagnosis/experiment.py init
  python experiments/01_performance_diagnosis/experiment.py run \
    experiments/01_performance_diagnosis/inputs/experiment.json

Run `explain` for input contracts, measurement scope and artifact layout.
Only init creates an example MLIR; run never modifies the supplied workload.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import select
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor')

EXPLANATION = """
INPUT CONTRACT
  A JSON manifest describes one workload. All paths resolve relative to that JSON.
  mlir: ranked static tensor arguments; tensor outputs plus trailing i64 nanoTime
  delta at @main, with llvm.emit_c_interface. A simple raw tensor-returning entry
  is adapted with V5 ensure_timed_main_abi. Scalars/dynamic shapes/unranked or
  strided arguments are rejected. Use f32/f64/i32/i64 contiguous inputs.
  Missing/elided weights are rejected, never silently replaced by zeros.
  inputs: one specification per MLIR argument, including initial output tensors.
  Each has name/shape/dtype and generator normal|uniform|zeros|ones|integers,
  or file (a .npy). The manifest seed realizes exact arrays once as case_N.npz.
  Multiple correctness cases vary generated inputs; file inputs remain fixed.
  torch.graph: explicit equivalent computation, NOT an inferred MLIR translation.
  Values are input names or earlier node ids. Nodes have id/op/args/kwargs.
  Supported ops: matmul, add, mul, relu, sum, conv2d, permute, transpose,
  contiguous, reshape, expand, cat, softmax, leaky_relu, elu, max_pool2d,
  maximum. Optional
  torch.constants is an NPZ of immutable graph tensors, loaded before timing.
  outputs lists node/input names in MLIR return order,
  excluding the timer. Explicitly represent bias, initial accumulators, layouts,
  padding, reduction axes and dtype. No executable expressions are evaluated.
  rl: checkpoint plus a matching architecture/observation/action config; the
  adapter supports rl_autoschedular_v5 and rl_autoschedular_v5_no_transformer.
  Trusted local torch checkpoints only. Missing checkpoint/config is a recorded
  skip. No fake optimized numbers are generated. Policy runs greedy on CPU.
  manual: optional list of {name, mlir}; user-supplied scheduled variants must
  have identical signatures/semantics. No automatic schedule search is performed.

MEASUREMENTS
  Baseline means no learned schedule, NOT LLVM -O0. Baseline, RL and manual MLIR
  use the selected package's transform_bufferize_and_lower_v and exact
  _EXEC_PASS_PIPELINE, then ExecutionEngine(opt_level=3). No timing cache is read.
  Common latency is host perf_counter_ns around a single synchronous invocation:
  includes Python/FFI dispatch, result allocation and any workload timing wrapper;
  excludes resetting inputs, output copying, verification and disposal. PyTorch
  outputs remain live until the timer stops. MLIR's internal delta is secondary
  and is NEVER divided by PyTorch host latency. No CUDA, TF32 or fast-math override.
  Fresh subprocess per method; same affinity, threads, arrays, warmups/repeats.
  Input buffers are restored outside each timed call to prevent accumulation.
  Steady-state allocation is included; setup and warmup are excluded.
  compile: MLIR parse/ABI adaptation, RL feature extraction/loading/inference/
  application, bufferization, lowering, engine creation, and first invoke reported
  separately. LLVM materialization can be lazy: engine+first invoke is a useful
  compile/setup upper bound (includes first execution). Inductor records wrapper
  creation and first invocation separately using a fresh per-run cache.
  Correctness compares every case/output to baseline and torch eager, checks
  output count/shape/dtype, integer exactness and float allclose (NaNs fail).
  A failed or missing independent reference means no validated cross-backend
  speedup. Baseline remains a reference, not a proof of mathematical correctness.
  thread utilization: process user+system CPU time / wall time, and per-thread
  /proc CPU-tick deltas during a separate repeated-call profiling window.
  perf stat is attached AFTER compilation and warmup and enabled via control fd.
  Counters cover a repeated invocation loop INCLUDING Python, input restoration,
  output disposal, and timer/control overhead, not pure kernel-only instructions.
  Hardware support/permissions may prevent counts. Unsupported counts are null
  with diagnostics, never zero. cache-misses is generic PMU cache-misses, not a
  guaranteed L1/LLC event. Cache misses alone do not measure memory bandwidth.
  Vectorization: static IR/vector instruction evidence and actual JIT object
  disassembly; these are not dynamic vector utilization measurements.

OUTPUT
  results/<UTC timestamp>_<uuid>/ contains manifest.json, workload.mlir, exact
  inputs/case_N.npz, environment.json, summary.json, summary.csv, and methods/.
  Each method has result.json, worker.log, outputs_N.npz and available scheduled,
  bufferized, lowered MLIR, LLVM IR, actual JIT object/disassembly and perf CSV.
  TorchInductor debug/cache directories retain generated source. Raw .npz/.pt/
  objects/generated artifacts stay local via this experiment's .gitignore.
  Errors and skips are preserved independently; subprocess timeout/crash cannot
  turn into a successful result. Artifacts may still exist after a worker crash.
  This is one workload per run. run_suite.py builds sampled category manifests
  in memory; custom input manifests remain supported. See README.md.
"""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(args, timeout=60):
    try:
        p = subprocess.run(list(map(str, args)), capture_output=True, text=True,
                           timeout=timeout)
        return {'command': list(map(str, args)), 'returncode': p.returncode,
                'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': list(map(str, args)), 'error': str(e)}


def stats(values):
    return {'samples_ns': values, 'median_ns': statistics.median(values),
            'min_ns': min(values), 'max_ns': max(values),
            'mean_ns': statistics.mean(values),
            'stdev_ns': statistics.stdev(values) if len(values) > 1 else 0}


def initialize(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'name': 'add_example', 'mlir': 'workload.mlir',
        'seed': 2026, 'threads': 1, 'cpu_affinity': None,
        'warmup': 5, 'repeats': 30, 'correctness_cases': 3,
        'rtol': 1e-4, 'atol': 1e-5, 'timeout_seconds': 600,
        'profile_repeats': 100, 'profile_min_seconds': 0.25, 'perf': True,
        'pipeline_implementation': 'rl_autoschedular_v5',
        'inputs': [{'name': n, 'shape': [128], 'dtype': 'float32',
                    'generator': 'normal'} for n in ('a', 'b')],
        'torch': {'graph': [{'id': 'out', 'op': 'add', 'args': ['a', 'b']}],
                  'outputs': ['out']},
        'rl': {'checkpoint': None, 'config': None,
               'implementation': 'rl_autoschedular_v5', 'max_steps': 1000},
        'manual': [],
    }
    example = '''module {
  func.func @main(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> attributes {llvm.emit_c_interface} {
    %empty = tensor.empty() : tensor<128xf32>
    %out = linalg.add {tag = "operation_0"} ins(%a, %b : tensor<128xf32>, tensor<128xf32>) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
    return %out : tensor<128xf32>
  }
}
'''
    for name, content in [('experiment.json', json.dumps(manifest, indent=2)+'\n'),
                          ('workload.mlir', example)]:
        path = folder/name
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        path.write_text(content)
    print(f'Created runnable add example in {folder}. Edit experiment.json for your workload.')


def load_manifest(path):
    # Category runs construct a spec in memory; save only the final run manifest.
    if isinstance(path, dict):
        c = json.loads(json.dumps(path))
        source_manifest, base = None, REPO
    else:
        source_manifest = Path(path).resolve()
        c = json.loads(source_manifest.read_text())
        base = source_manifest.parent
    if c.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    for key in ('warmup', 'repeats', 'correctness_cases', 'threads', 'profile_repeats'):
        minimum = 0 if key == 'warmup' else 1
        if type(c.get(key)) is not int or c[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if c.get('profile_min_seconds', 0.25) < 0:
        raise ValueError('profile_min_seconds must be nonnegative')
    if c.get('timeout_seconds', 600) <= 0 or c.get('rtol', 1e-4) < 0 or c.get('atol', 1e-5) < 0:
        raise ValueError('Invalid timeout or correctness tolerance')
    def resolve(value):
        return str((base/value).resolve()) if value else None
    c['mlir'] = resolve(c['mlir'])
    if c.get('torch', {}).get('constants'):
        c['torch']['constants'] = resolve(c['torch']['constants'])
    if not Path(c['mlir']).is_file():
        raise FileNotFoundError(c['mlir'])
    if c.get('pipeline_implementation', 'rl_autoschedular_v5') not in (
            'rl_autoschedular_v5', 'rl_autoschedular_v5_no_transformer'):
        raise ValueError('Supported pipelines: V5 and V5 no-transformer')
    names = set()
    for spec in c['inputs']:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', spec['name']) or spec['name'] in names:
            raise ValueError('Input names must be unique identifiers')
        names.add(spec['name'])
        if spec['dtype'] not in ('float32', 'float64', 'int32', 'int64'):
            raise ValueError('Supported dtypes: float32, float64, int32, int64')
        if any(type(n) is not int or n <= 0 for n in spec['shape']):
            raise ValueError('Input shapes must be static positive integers')
        if spec.get('file'):
            spec['file'] = resolve(spec['file'])
    for k in ('checkpoint', 'config'):
        c.setdefault('rl', {})[k] = resolve(c.get('rl', {}).get(k))
    for manual in c.get('manual', []):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', manual['name']):
            raise ValueError('Manual schedule names must be path-safe')
        manual['mlir'] = resolve(manual['mlir'])
        if not Path(manual['mlir']).is_file():
            raise FileNotFoundError(manual['mlir'])
    if len({m['name'] for m in c.get('manual', [])}) != len(c.get('manual', [])):
        raise ValueError('Duplicate manual schedule names')
    available = sorted(os.sched_getaffinity(0))
    cpus = c.get('cpu_affinity') or available[:c['threads']]
    if not isinstance(cpus, list) or not set(cpus).issubset(available) or len(set(cpus)) < c['threads']:
        raise ValueError('cpu_affinity must contain >= threads distinct allocated CPUs')
    c['cpu_affinity'] = sorted(set(cpus))
    c['source_manifest'] = str(source_manifest) if source_manifest else None
    return c


def prepare_inputs(c, out):
    import numpy as np
    out.mkdir()
    for case in range(c['correctness_cases']):
        rng = np.random.default_rng(c.get('seed', 2026)+case)
        arrays = {}
        for spec in c['inputs']:
            shape, dtype = tuple(spec['shape']), np.dtype(spec['dtype'])
            if spec.get('file'):
                a = np.load(spec['file'], allow_pickle=False)
                if a.shape != shape or a.dtype != dtype:
                    raise ValueError(f"Input file shape/dtype mismatch: {spec['name']}")
            else:
                generator = spec.get('generator', 'normal')
                if generator == 'normal':
                    a = rng.normal(spec.get('mean', 0), spec.get('std', 1), shape)
                elif generator == 'uniform':
                    a = rng.uniform(spec.get('low', -1), spec.get('high', 1), shape)
                elif generator == 'integers':
                    a = rng.integers(spec.get('low', -3), spec.get('high', 4), shape)
                elif generator in ('zeros', 'ones'):
                    a = getattr(np, generator)(shape)
                else:
                    raise ValueError(f'Unknown generator {generator}')
                a = np.asarray(a, dtype=dtype)
            if not np.isfinite(a).all():
                raise ValueError('Input values must be finite')
            arrays[spec['name']] = np.array(a, order='C', copy=True)
        np.savez(out/f'case_{case}.npz', **arrays)


def policy_schedule(code, c, out, compile_times):
    rl = c['rl']
    if not rl.get('checkpoint') or not rl.get('config'):
        raise FileNotFoundError('Supply rl.checkpoint and its matching rl.config')
    impl = rl.get('implementation', 'rl_autoschedular_v5')
    if impl != c.get('pipeline_implementation', 'rl_autoschedular_v5'):
        raise ValueError('Policy and common pipeline implementations must match')
    # Import exactly one standalone implementation; no config/logger/train side effects.
    def mod(name):
        return importlib.import_module(f'{impl}.{name}')
    t = time.perf_counter_ns()
    cfg = json.loads(Path(rl['config']).read_text())
    if cfg.get('implementation', impl) != impl:
        raise ValueError('Checkpoint config implementation does not match rl.implementation')
    cfg.update(dask_node_count=0, results_dir=str(out/'policy_logs'))
    write_json(out/'policy_config.json', cfg)
    os.environ['CONFIG_FILE_PATH'] = str(out/'policy_config.json')
    os.environ['DASK_NODES'] = '0'
    os.environ['DASK_TRAINING_ONLY'] = '1'
    import torch
    model = mod('model').HiearchyModel().cpu().eval()
    checkpoint = torch.load(rl['checkpoint'], map_location='cpu', weights_only=True)
    weights = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=True)
    compile_times['checkpoint_and_model_load_ns'] = time.perf_counter_ns()-t
    t = time.perf_counter_ns()
    features = mod('state').extract_bench_features_from_code(c['name'], code, 1)
    compile_times['feature_extraction_ns'] = time.perf_counter_ns()-t
    env = mod('env').Env()
    state = env.reset([features], 0)
    actions, steps = [], 0
    t = time.perf_counter_ns()
    with torch.inference_mode():
        while True:
            obs = mod('observation').Observation.from_state(state)
            indices, _, _ = model.sample(obs, greedy=True)
            action = mod('actions').ActionSpace.action_by_index(indices[0].cpu(), state)
            actions.append({'operation': state.operation_tag, 'action': str(action),
                            'ready': action.ready})
            state = env.step(state, action)
            steps += 1
            if steps > rl.get('max_steps', 1000):
                raise RuntimeError('Policy exceeded max_steps')
            if state.terminal:
                nxt = env.get_next_op_state(state)
                if nxt is None:
                    break
                state = nxt
    compile_times['policy_inference_ns'] = time.perf_counter_ns()-t
    write_json(out/'schedule.json', actions)
    # Apply actions directly, failing loudly. Env's private application method can
    # swallow transformation failures and return partially optimized code.
    from mlir.ir import Context, Module
    t = time.perf_counter_ns()
    with Context():
        # Feature extraction assigns fresh operation tags. Apply the policy to
        # that same tagged code, including inputs with absent or different tags.
        module = Module.parse(features.code)
        for sequence in reversed(state.transformation_history):
            for action in sequence:
                action.apply(module)
        scheduled = str(module)
    compile_times['schedule_application_ns'] = time.perf_counter_ns()-t
    return scheduled


class MLIRRunner:
    """Reuse project lowering with explicit input/output ownership and no cache."""
    def __init__(self, c, method, out):
        import numpy as np
        from mlir.ir import Context, Module, MemRefType, IntegerType
        from mlir.dialects import func, arith, linalg  # register common parsers
        from mlir.passmanager import PassManager
        from mlir.execution_engine import ExecutionEngine
        from mlir.runtime import make_nd_memref_descriptor, as_ctype
        self.np, self.out, self.times = np, out, {}
        impl = c.get('pipeline_implementation', 'rl_autoschedular_v5')
        full = importlib.import_module(f'{impl}.full_model')
        transforms = importlib.import_module(f'{impl}.transforms')
        execution = importlib.import_module(f'{impl}.execution')
        self.pipeline = execution._EXEC_PASS_PIPELINE
        t = time.perf_counter_ns()
        source = c['mlir']
        if method.startswith('manual_'):
            source = next(m['mlir'] for m in c['manual'] if 'manual_'+m['name'] == method)
        code = Path(source).read_text()
        if '__elided__' in code:
            raise ValueError('Elided constants cannot establish numerical equivalence; supply real weights')
        code = full.ensure_timed_main_abi(code)
        self.times['abi_adaptation_ns'] = time.perf_counter_ns()-t
        (out/'adapted.mlir').write_text(code)
        if method == 'mlir_rl':
            code = policy_schedule(code, c, out, self.times)
        (out/'scheduled.mlir').write_text(code)
        self.context = Context()
        with self.context:
            t = time.perf_counter_ns()
            self.module = Module.parse(code)
            self.times['parse_ns'] = time.perf_counter_ns()-t
            t = time.perf_counter_ns()
            transforms.transform_bufferize_and_lower_v(self.module)
            self.times['bufferization_ns'] = time.perf_counter_ns()-t
            (out/'bufferized.mlir').write_text(str(self.module))
            main = next(op for op in self.module.body.operations
                        if isinstance(op, func.FuncOp) and op.name.value == 'main')
            if 'llvm.emit_c_interface' not in main.attributes:
                raise ValueError('@main needs llvm.emit_c_interface')
            self.input_types = list(main.type.inputs)
            results = list(main.type.results)
            if (len(results) < 2 or not isinstance(results[-1], IntegerType)
                    or results[-1].width != 64):
                raise ValueError('Expected @main -> (tensor/memref output(s), i64 timer)')
            mapping = {'f32': 'float32', 'f64': 'float64', 'i32': 'int32', 'i64': 'int64'}
            def signature(ty):
                if not isinstance(ty, MemRefType) or any(n < 0 for n in ty.shape):
                    raise ValueError(f'Only static ranked memrefs supported after bufferization: {ty}')
                dtype = mapping.get(str(ty.element_type))
                if not dtype:
                    raise ValueError(f'Unsupported element type: {ty.element_type}')
                return list(ty.shape), np.dtype(dtype)
            if len(self.input_types) != len(c['inputs']):
                raise ValueError('Manifest inputs must match ALL @main arguments in order')
            for ty, spec in zip(self.input_types, c['inputs']):
                shape, dtype = signature(ty)
                if shape != spec['shape'] or dtype != np.dtype(spec['dtype']):
                    raise ValueError(f'MLIR signature disagrees with input {spec["name"]}: {ty}')
            fields = []
            for i, ty in enumerate(results[:-1]):
                shape, dtype = signature(ty)
                fields.append((f'out_{i}', make_nd_memref_descriptor(len(shape), as_ctype(dtype))))
            self.fields = fields
            self.output_type = type('Outputs', (ctypes.Structure,), {'_fields_': fields+[('delta', ctypes.c_int64)]})
            t = time.perf_counter_ns()
            PassManager.parse(self.pipeline, self.context).run(self.module.operation)
            self.times['lowering_ns'] = time.perf_counter_ns()-t
            (out/'lowered.mlir').write_text(str(self.module))
            write_json(out/'pipeline.json', {'implementation': impl, 'pass_pipeline': self.pipeline,
                                            'llvm_opt_level': 3, 'timing_cache': False})
            t = time.perf_counter_ns()
            self.engine = ExecutionEngine(self.module, opt_level=3,
                shared_libs=[s for s in os.environ.get('MLIR_SHARED_LIBS', '').split(',') if s],
                enable_object_dump=True)
            self.times['engine_creation_ns'] = time.perf_counter_ns()-t
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None
        self.current = None

    def reset(self, arrays):
        from mlir.runtime import get_ranked_memref_descriptor
        if not hasattr(self, 'inputs'):
            self.inputs = [self.np.array(a, copy=True, order='C') for a in arrays]
            self.outputs = self.output_type()
            self.args = [ctypes.pointer(ctypes.pointer(self.outputs))]
            self.args += [ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a))) for a in self.inputs]
        else:
            for destination, source in zip(self.inputs, arrays):
                self.np.copyto(destination, source)

    def call(self):
        self.engine.invoke('main', *self.args)
        return self.outputs

    def snapshot(self):
        from mlir.runtime import ranked_memref_to_numpy
        return [ranked_memref_to_numpy([getattr(self.outputs, n)]).copy() for n, _ in self.fields]

    def release(self, value):
        # Returned buffers can alias each other OR borrowed input arrays. Do not
        # free borrowed memory, and free each owned allocation at most once.
        borrowed = [(a.ctypes.data, a.ctypes.data+max(1, a.nbytes)) for a in self.inputs]
        seen = set()
        for name, descriptor in self.fields:
            desc = getattr(self.outputs, name)
            address = ctypes.cast(desc.allocated, ctypes.c_void_p).value
            if address and address not in seen and not any(lo <= address < hi for lo, hi in borrowed):
                self.libc.free(address)
                seen.add(address)
            setattr(self.outputs, name, descriptor())

    def artifacts(self):
        result = {}
        object_path = self.out/'jit.o'
        try:
            self.engine.dump_to_object_file(str(object_path))
            result['object'] = str(object_path)
            tool = shutil.which('objdump')
            if tool:
                p = command([tool, '-d', object_path])
                (self.out/'jit.asm').write_text(p.get('stdout', '')+p.get('stderr', ''))
                result['disassembly'] = {k: v for k, v in p.items() if k != 'stdout'}
        except Exception as e:
            result['object_error'] = str(e)
        translate = Path(os.environ.get('LLVM_BUILD_PATH', ''))/'bin/mlir-translate'
        if translate.is_file():
            p = command([translate, '--mlir-to-llvmir', self.out/'lowered.mlir'])
            if p.get('returncode') == 0:
                (self.out/'generated.ll').write_text(p['stdout'])
            result['llvm_translation'] = {k: v for k, v in p.items() if k != 'stdout'}
        evidence = {}
        for file in ('scheduled.mlir', 'lowered.mlir', 'generated.ll', 'jit.asm'):
            path = self.out/file
            if path.exists():
                text = path.read_text()
                evidence[file] = {'vector_mlir_ops': len(re.findall(r'\bvector\.', text)),
                    'llvm_vector_types': len(re.findall(r'<\d+ x ', text)),
                    'x86_simd_register_mentions': len(re.findall(r'\b[xyz]mm\d+\b', text))}
        result['vectorization_evidence'] = evidence
        result['interpretation'] = 'Static evidence only; LLVM IR is before JIT optimization; object assembly is actual JIT code.'
        return result


class TorchRunner:
    def __init__(self, c, method, out):
        import numpy as np
        import torch
        import torch.nn.functional as F
        self.np, self.torch, self.spec, self.times, self.out = np, torch, c['torch'], {}, out
        torch.set_num_threads(c['threads'])
        torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = True
        torch.set_float32_matmul_precision('highest')
        operations = {'matmul': torch.matmul, 'add': torch.add, 'mul': torch.mul,
            'relu': torch.relu, 'sum': torch.sum, 'conv2d': F.conv2d,
            'max_pool2d': F.max_pool2d, 'maximum': torch.maximum,
            'permute': torch.permute, 'transpose': torch.transpose,
            'contiguous': lambda a: a.contiguous(), 'reshape': torch.reshape,
            'expand': lambda a, size: a.expand(size), 'cat': torch.cat,
            'softmax': torch.softmax, 'leaky_relu': F.leaky_relu, 'elu': F.elu}
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
                raise ValueError('Unknown Torch graph operation or duplicate value')
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
        self.times['compile_wrapper_creation_ns'] = time.perf_counter_ns()-t

    def reset(self, arrays):
        if not hasattr(self, 'inputs'):
            self.inputs = [self.torch.from_numpy(self.np.array(a, copy=True, order='C')) for a in arrays]
        else:
            for destination, source in zip(self.inputs, arrays):
                destination.copy_(self.torch.from_numpy(source))

    def call(self):
        self.current = self.fn(*self.inputs)
        return self.current

    def snapshot(self):
        return [a.detach().cpu().numpy().copy() for a in self.current]

    def release(self, value):
        self.current = None

    def artifacts(self):
        sources = list(self.out.rglob('*.cpp'))
        evidence = {str(p.relative_to(self.out)): {'vectorized_mentions': len(re.findall(
            r'Vectorized|#pragma (?:omp )?simd|[xyz]mm\d+', p.read_text(errors='replace')))} for p in sources}
        return {'inductor_cache': str(self.out/'inductor_cache'),
                'vectorization_evidence': evidence,
                'note': 'Generated Inductor source/cache retained. Evidence is static, not vector utilization.'}


def thread_ticks():
    ticks = {}
    for p in Path('/proc/self/task').glob('*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            ticks[p.parent.name] = int(fields[11])+int(fields[12])
        except (OSError, ValueError):
            pass
    return ticks


class PerfCounters:
    def __init__(self, out, enabled):
        self.out, self.proc, self.error = out, None, None
        self.fds = []
        self.log = None
        if not enabled:
            self.error = 'Disabled in manifest'
            return
        if not shutil.which('perf'):
            self.error = 'perf executable unavailable'
            return
        try:
            r, w = os.pipe()
            ar, aw = os.pipe()
            self.fds = [r, w, ar, aw]
            self.write_fd, self.read_fd = w, ar
            self.log = open(out/'perf.log', 'w')
            self.proc = subprocess.Popen(['perf', 'stat', '-p', str(os.getpid()),
                '--delay=-1', f'--control=fd:{r},{aw}', '-x', ';', '--no-big-num',
                '-e', 'task-clock,cycles,instructions,cache-references,cache-misses',
                '-o', str(out/'perf.csv')], pass_fds=(r, aw),
                stdout=self.log, stderr=self.log)
            self.control('enable')
        except Exception as e:
            self.error = str(e)

    def control(self, message):
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError('perf exited before acknowledging control; see perf.log')
        os.write(self.write_fd, (message+'\n').encode())
        ready, _, _ = select.select([self.read_fd], [], [], 5)
        if not ready or b'ack' not in os.read(self.read_fd, 128):
            raise RuntimeError('perf control acknowledgement unavailable; see perf.log')

    def finish(self):
        if self.proc:
            try:
                if not self.error:
                    self.control('disable')
            except Exception as e:
                self.error = str(e)
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            if self.proc.returncode not in (0, -signal.SIGINT):
                self.error = self.error or f'perf exited {self.proc.returncode}; see perf.log'
        for fd in self.fds:
            os.close(fd)
        if self.log:
            self.log.close()
        counters = {}
        path = self.out/'perf.csv'
        if path.exists():
            for row in csv.reader(path.read_text().splitlines(), delimiter=';'):
                if len(row) < 3 or row[0].startswith('#'):
                    continue
                try:
                    value = float(row[0].strip())
                    if not __import__('math').isfinite(value):
                        value = None
                except ValueError:
                    value = None
                if self.error:
                    value = None
                counters[row[2].strip().split(':')[0]] = {'value': value, 'event_selector': row[2].strip(), 'unit': row[1].strip(),
                    'raw': row, 'status': 'available' if value is not None else 'unavailable'}
        available = sum(v['value'] is not None for v in counters.values())
        return {'status': 'unavailable' if self.error or not available else ('available' if available == len(counters) else 'partial'),
                'reason': self.error, 'counters': counters,
                'scope': 'Repeated call loop including input restoration and output disposal; excludes compilation/warmup'}


def worker(c, method, out, run):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    (out/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(out/'tmp')
    os.sched_setaffinity(0, c['cpu_affinity'])
    sys.path[:0] = [str(REPO), str(REPO/'rl_autoschedular')]
    import numpy as np
    import torch
    torch.set_num_threads(c['threads'])
    torch.manual_seed(c.get('seed', 2026))
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        return measure_worker(c, method, out, run, np, torch)


def measure_worker(c, method, out, run, np, torch):
    cases = []
    for case in range(c['correctness_cases']):
        with np.load(run/'inputs'/f'case_{case}.npz', allow_pickle=False) as f:
            cases.append([f[s['name']].copy() for s in c['inputs']])
    print(f'Preparing {method}', flush=True)
    runner = TorchRunner(c, method, out) if method.startswith('torch_') else MLIRRunner(c, method, out)
    runner.reset(cases[0])
    t = time.perf_counter_ns()
    value = runner.call()
    runner.times['first_invoke_ns'] = time.perf_counter_ns()-t
    runner.release(value)
    del value
    for case, arrays in enumerate(cases):
        runner.reset(arrays)
        value = runner.call()
        output = runner.snapshot()
        np.savez(out/f'outputs_{case}.npz',
                 **{f'output_{i}': a for i, a in enumerate(output)})
        runner.release(value)
        del value
    for _ in range(c['warmup']):
        runner.reset(cases[0])
        value = runner.call()
        runner.release(value)
        del value
    samples, kernel_samples = [], []
    with torch.inference_mode():
        for _ in range(c['repeats']):
            runner.reset(cases[0])
            t = time.perf_counter_ns()
            value = runner.call()
            samples.append(time.perf_counter_ns()-t)
            if isinstance(runner, MLIRRunner):
                kernel_samples.append(int(runner.outputs.delta))
            runner.release(value)
            del value
    perf = PerfCounters(out, c.get('perf', True))
    before_ticks = thread_ticks()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter_ns()
    profile_calls = 0
    with torch.inference_mode():
        while profile_calls < c['profile_repeats'] or (time.perf_counter_ns()-start)/1e9 < c.get('profile_min_seconds', 0.25):
            runner.reset(cases[0])
            value = runner.call()
            runner.release(value)
            del value
            profile_calls += 1
    wall = time.perf_counter_ns()-start
    after = resource.getrusage(resource.RUSAGE_SELF)
    after_ticks = thread_ticks()
    counters = perf.finish()
    counters['profile_calls'] = profile_calls
    for event in counters['counters'].values():
        event['per_call'] = event['value']/profile_calls if event['value'] is not None else None
    cpu = (after.ru_utime-usage.ru_utime)+(after.ru_stime-usage.ru_stime)
    vector = runner.artifacts()
    return {'status': 'measured', 'method': method, 'latency': stats(samples),
            'mlir_internal_timer': stats(kernel_samples) if kernel_samples else None,
            'compilation': runner.times,
            'thread_utilization': {'scope': 'Separate repeated-call loop, including harness overhead',
                'wall_ns': wall, 'process_cpu_seconds': cpu,
                'average_busy_cores': cpu/(wall/1e9),
                'allocated_cpu_utilization_percent': 100*cpu/(wall/1e9)/len(c['cpu_affinity']),
                'per_thread_cpu_seconds': {tid: (v-before_ticks.get(tid, v))/os.sysconf('SC_CLK_TCK')
                                          for tid, v in after_ticks.items()},
                'note': 'Tick resolution limits short profiles; not instantaneous thread occupancy.'},
            'profiling': counters, 'code_artifacts': vector,
            'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                         'python': sys.version, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                         'torch_num_threads': torch.get_num_threads(), 'torch_build': torch.__config__.show(),
                         'CXX': os.environ.get('CXX'), 'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_PROC_BIND')}}}


def compare_outputs(reference, candidate, c):
    import numpy as np
    comparisons = []
    for case in range(c['correctness_cases']):
        with np.load(reference/f'outputs_{case}.npz', allow_pickle=False) as ref, \
             np.load(candidate/f'outputs_{case}.npz', allow_pickle=False) as got:
            if ref.files != got.files:
                return {'passed': False, 'reason': 'Output count/order mismatch'}
            for key in ref.files:
                a, b = ref[key], got[key]
                match = a.shape == b.shape and a.dtype == b.dtype
                finite = np.isfinite(a).all() and np.isfinite(b).all()
                close = bool(match and finite and (np.array_equal(a, b) if a.dtype.kind in 'iu'
                             else np.allclose(a, b, rtol=c.get('rtol', 1e-4), atol=c.get('atol', 1e-5), equal_nan=False)))
                error = float(np.max(np.abs(a.astype('float64')-b.astype('float64')))) if match and a.size and finite else None
                comparisons.append({'case': case, 'output': key, 'passed': close,
                    'shape': list(b.shape), 'dtype': str(b.dtype), 'max_abs_error': error})
    return {'passed': all(x['passed'] for x in comparisons), 'comparisons': comparisons}


def summarize(c, run, results):
    baseline = results.get('mlir_baseline', {})
    eager = results.get('torch_eager', {})
    for method, result in results.items():
        result['correctness'] = {'status': 'unavailable'}
        result['validated_speedup_vs_mlir_baseline'] = None
        result['observed_latency_ratio_vs_mlir_baseline'] = None
        if result['status'] != 'measured':
            continue
        if baseline.get('status') == 'measured':
            result['correctness']['vs_mlir_baseline'] = compare_outputs(
                run/'methods/mlir_baseline', run/'methods'/method, c)
            result['observed_latency_ratio_vs_mlir_baseline'] = baseline['latency']['median_ns']/result['latency']['median_ns']
        if eager.get('status') == 'measured':
            result['correctness']['vs_torch_eager'] = compare_outputs(
                run/'methods/torch_eager', run/'methods'/method, c)
        comparisons = result['correctness']
        baseline_matches_eager = (baseline.get('status') == 'measured' and eager.get('status') == 'measured'
            and compare_outputs(run/'methods/torch_eager', run/'methods/mlir_baseline', c)['passed'])
        passed = (baseline_matches_eager and comparisons.get('vs_mlir_baseline', {}).get('passed', False)
                  and comparisons.get('vs_torch_eager', {}).get('passed', False))
        result['correctness']['status'] = 'passed' if passed else ('failed' if
            any(not v['passed'] for v in comparisons.values() if isinstance(v, dict) and 'passed' in v) else 'unverified')
        if passed:
            result['validated_speedup_vs_mlir_baseline'] = result['observed_latency_ratio_vs_mlir_baseline']
        write_json(run/'methods'/method/'result.json', result)
    for method, result in results.items():
        write_json(run/'methods'/method/'result.json', result)
    write_json(run/'summary.json', {'workload': c['name'], 'results': results,
        'latency_scope': 'Host single-call latency, including dispatch/allocation; excludes input restore/output copies/free',
        'counter_scope': 'Repeated profiling loop includes Python, input restores and disposal',
        'correctness_scope': 'Baseline and independent eager graph, multiple fixed cases; not a formal proof',
        'source_mlir_sha256': sha256(run/'workload.mlir'), 'method_order': list(results)})
    with open(run/'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'status', 'correctness', 'median_host_ns', 'validated_speedup',
                         'compile_and_first_invoke_ns', 'average_busy_cores', 'instructions_per_profile_call', 'cache_misses_per_profile_call'])
        for method, result in results.items():
            counts = result.get('profiling', {}).get('counters', {})
            writer.writerow([method, result['status'], result.get('correctness', {}).get('status'),
                result.get('latency', {}).get('median_ns'), result.get('validated_speedup_vs_mlir_baseline'),
                sum(result.get('compilation', {}).values()) if result.get('compilation') else None,
                result.get('thread_utilization', {}).get('average_busy_cores'),
                counts.get('instructions', {}).get('per_call'), counts.get('cache-misses', {}).get('per_call')])


def run_experiment(path, selected, output):
    # .env is shell syntax and uses ${VAR:+...}; python-dotenv does not expand
    # that syntax correctly. Source it exactly as the repository launchers do.
    env_file = REPO/'.env'
    if env_file.is_file():
        loaded = subprocess.run(['bash', '-c', 'set -ea; source "$1" >&2; env -0',
            'bash', str(env_file)], check=True, capture_output=True)
        for entry in loaded.stdout.split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)
    c = load_manifest(path)
    root = Path(output).resolve() if output else HERE/'results'
    run = root/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'_'+uuid.uuid4().hex[:8])
    (run/'methods').mkdir(parents=True)
    prepare_inputs(c, run/'inputs')
    shutil.copy2(c['mlir'], run/'workload.mlir')
    c['mlir'] = str(run/'workload.mlir')
    if c.get('torch', {}).get('constants'):
        target = run/'weights.npz'
        shutil.copy2(c['torch']['constants'], target)
        c['torch']['constants'] = str(target)
    for i, m in enumerate(c.get('manual', [])):
        target = run/f'manual_{i}.mlir'
        shutil.copy2(m['mlir'], target)
        m['mlir'] = str(target)
    if c['rl'].get('config') and Path(c['rl']['config']).is_file():
        target = run/'checkpoint_config.json'
        shutil.copy2(c['rl']['config'], target)
        c['rl']['config'] = str(target)
    write_json(run/'manifest.json', c)
    write_json(run/'environment.json', {'platform': platform.platform(), 'python': sys.version,
        'hostname': platform.node(), 'allocation': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
        'lscpu': command(['lscpu']), 'git': command(['git', '-C', REPO, 'rev-parse', 'HEAD']),
        'git_status': command(['git', '-C', REPO, 'status', '--short']),
        'mlir_version': command([Path(os.environ['LLVM_BUILD_PATH'])/'bin/mlir-opt', '--version']),
        'perf_paranoid': Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
        'checkpoint_sha256': sha256(c['rl']['checkpoint']) if c['rl'].get('checkpoint') and Path(c['rl']['checkpoint']).is_file() else None,
        'runner_sha256': sha256(__file__),
        'torch_constants_sha256': sha256(c['torch']['constants']) if c.get('torch', {}).get('constants') else None,
        'input_sha256': {p.name: sha256(p) for p in (run/'inputs').glob('*.npz')}})
    methods = list(selected or METHODS)
    methods += ['manual_'+m['name'] for m in c.get('manual', [])]
    results = {}
    for method in methods:
        out = run/'methods'/method
        out.mkdir()
        if method == 'mlir_rl' and not all(c['rl'].get(k) and Path(c['rl'][k]).is_file() for k in ('checkpoint', 'config')):
            results[method] = {'status': 'skipped', 'reason': 'Upload rl.checkpoint and its matching rl.config'}
            write_json(out/'result.json', results[method])
            continue
        env = dict(os.environ)
        # Also work when launched with the env's absolute Python executable,
        # without an interactive conda activation hook (common in batch jobs).
        env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
        compiler = Path(sys.executable).parent/'x86_64-conda-linux-gnu-g++'
        if not env.get('CXX') and compiler.is_file():
            env['CXX'] = str(compiler)
        env.update({k: str(c['threads']) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        env.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', OMP_PROC_BIND='TRUE',
                   TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),
                   TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=str(out/'torch_debug'),
                   TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONHASHSEED=str(c.get('seed', 2026)),
                   DASK_NODES='0', DASK_TRAINING_ONLY='1', CUDA_VISIBLE_DEVICES='')
        print(f'Running {method}: {out}', flush=True)
        with open(out/'worker.log', 'w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                str(run/'manifest.json'), method, str(out)], env=env, stdout=log, stderr=log,
                start_new_session=True, cwd=REPO)
            try:
                process.wait(timeout=c.get('timeout_seconds', 600))
                result_path = out/'result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    result = {'status': 'error', 'reason': 'Worker exited without results; inspect worker.log'}
                if process.returncode != 0:
                    result.update(status='error', returncode=process.returncode)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result = {'status': 'timeout', 'reason': 'Worker process group exceeded timeout'}
            except KeyboardInterrupt:
                # Workers have their own process group for compiler isolation.
                # Stop it too when the foreground driver is interrupted.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                # A compiler child may outlive the worker that launched it.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                write_json(out/'result.json', {'status': 'interrupted',
                    'reason': 'Foreground execution interrupted by the user'})
                raise
            results[method] = result
            write_json(out/'result.json', result)
    summarize(c, run, results)
    print(f'Results: {run}/summary.json')
    for method, result in results.items():
        print(f"  {method}: {result['status']}; correctness={result.get('correctness', {}).get('status', 'unavailable')}")
    return 1 if any(r['status'] in ('error', 'timeout') or r.get('correctness', {}).get('status') == 'failed' for r in results.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    init = subs.add_parser('init', help='Write an editable manifest and runnable add example')
    init.add_argument('--directory', default=str(HERE/'inputs'))
    subs.add_parser('explain', help='Print complete input/measurement/output contracts')
    run = subs.add_parser('run', help='Execute an experiment in isolated CPU workers')
    run.add_argument('manifest')
    run.add_argument('--methods', nargs='+', choices=METHODS)
    run.add_argument('--output')
    internal = subs.add_parser('_worker', help='Internal subprocess entry point')
    internal.add_argument('manifest')
    internal.add_argument('method')
    internal.add_argument('out')
    args = parser.parse_args()
    if args.command == 'init':
        initialize(args.directory)
    elif args.command == 'explain':
        print(EXPLANATION)
    elif args.command == 'run':
        return run_experiment(args.manifest, args.methods, args.output)
    else:
        out = Path(args.out)
        try:
            c = json.loads(Path(args.manifest).read_text())
            result = worker(c, args.method, out, Path(args.manifest).parent)
            write_json(out/'result.json', result)
        except Exception as e:
            traceback.print_exc()
            write_json(out/'result.json', {'status': 'error', 'reason': str(e)})
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
