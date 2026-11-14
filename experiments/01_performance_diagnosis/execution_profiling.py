#!/usr/bin/env python3
"""Profile execution of one sampled CPU benchmark category at a time.

One worker handles one benchmark/backend/trial at one CPU budget (default 8).
The RL schedule is generated and applied once before execution trials. Each
trial uses five warmups, five timed calls, and a separate CPU/counter window
lasting at least one second AND three complete calls. See README.md for scope.

Results default to this directory; --output selects another root. Dataset files
and the project's training/evaluation scripts are never changed. See README.md for measurement
definitions, supported MLIR forms, backend setup and the artifact contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import select
import shutil
import signal
import statistics
import subprocess
import sys
import time # for measuring latency
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATEGORIES = ('single_operations', 'multi_operation_blocks', 'full_models')
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor', 'tensorflow_xla')
INVENTORY = HERE / 'benchmark_inventory.csv'
SAMPLES = HERE / 'sampled_benchmarks.csv'


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


def source_path(row):
    """The inventory uses filenames; never interpret a CSV value as executable code."""
    if Path(row['benchmark']).name != row['benchmark']:
        raise ValueError('benchmark must be a filename, not a path')
    if row['dataset'] not in ('ops_and_blocks', 'mlir_rl_v1_paper', 'full_models'):
        raise ValueError(f"Unknown dataset: {row['dataset']}")
    return ROOT / 'data' / row['dataset'] / row['benchmark']


def affordable(row, max_input_mib):
    """Keep small-kernel samples runnable; full models retain their real limitations.

    This is only a sampling filter, not a claim about peak compilation memory.
    Inspect static tensor types without importing MLIR or loading model weights.
    The runner performs a second, exact ABI/input-memory check before execution.
    """
    if row['category'] == 'full_models':
        return True
    text = source_path(row).read_text()
    match = re.search(r'func\.func @main\((.*?)\)\s*->', text, re.S)
    if not match:
        return False
    inputs = re.findall(r'tensor<([0-9x]+)x(f32|f64|i32|i64)>', match[1])
    sizes = [math.prod(map(int, shape.split('x'))) * (8 if dtype.endswith('64') else 4)
             for shape, dtype in inputs]
    # Very large intermediate tensors occur in the QCD kernels even when their
    # arguments are small. They also need domain-specific index fixtures.
    if row['benchmark'].startswith(('baryon_', 'dibaryon_', 'hexaquark_')):
        return False
    return bool(inputs) and sum(sizes) <= max_input_mib * 1024**2


def choose_samples(rows, per_type=4, seed=2026, max_input_mib=32):
    """Deterministic coverage sampling: one benchmark may cover several op types.

    Cover rare types first, then fill each type's remaining quota. This avoids
    making four separate copies of a mixed-operation block in the sample CSV.
    Exact inventory rows are retained, including their category and JSON fields.
    """
    selected, coverage = [], []
    for category in CATEGORIES:
        candidates = [r for r in rows if r['category'] == category and affordable(r, max_input_mib)]
        types = {t for r in candidates for t in json.loads(r['operation_types'])}
        groups = {t: [r for r in candidates if t in json.loads(r['operation_types'])] for t in types}
        chosen = {}
        # These two small, weight-complete models provide useful full-program
        # controls alongside models whose missing resources must be reported.
        if category == 'full_models':
            chosen.update({(r['dataset'], r['benchmark']): r for r in candidates
                           if r['benchmark'] in ('gin_linalg.mlir', 'gat_linalg.mlir')})
        for op_type in sorted(groups, key=lambda t: (len(groups[t]), t)):
            group = groups[op_type]
            have = sum(op_type in json.loads(r['operation_types']) for r in chosen.values())
            pool = sorted((r for r in group if (r['dataset'], r['benchmark']) not in chosen),
                          key=lambda r: (r['dataset'], r['benchmark']))
            rng = random.Random(f'{seed}:{category}:{op_type}')
            for row in rng.sample(pool, min(max(0, per_type - have), len(pool))):
                chosen[row['dataset'], row['benchmark']] = row
        selected.extend(sorted(chosen.values(), key=lambda r: (r['dataset'], r['benchmark'])))
        for op_type in sorted(groups):
            have = sum(op_type in json.loads(r['operation_types']) for r in chosen.values())
            coverage.append((category, op_type, len(groups[op_type]), have))
    return selected, coverage


def sample_command(args):
    fields, rows = read_csv(args.inventory)
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; run commands reuse it without resampling')
    selected, coverage = choose_samples(rows, args.per_type, args.seed, args.max_input_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    for category, op_type, available, count in coverage:
        print(f'{category}: {op_type}: {count} selected / {available} eligible'
              + (' (all available; fewer than requested)' if available < args.per_type else ''))
    print('Unique file entries:', dict(Counter(r['category'] for r in selected)))
    print(args.output)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def load_environment():
    """Use Bash for .env's shell expansions, before any V5 configuration import."""
    if (ROOT / '.env').is_file():
        result = subprocess.run(['bash', '-c', 'set -ea; source "$1" >&2; env -0',
            'bash', str(ROOT / '.env')], cwd=ROOT, check=True, capture_output=True)
        for entry in result.stdout.split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)
    # The parent may have been launched through an absolute Python executable.
    # Child processes start with the compiler/library paths already configured.
    paths = [str(ROOT), str(ROOT / 'rl_autoschedular')]
    if os.environ.get('LLVM_BUILD_PATH'):
        paths.insert(0, os.environ['LLVM_BUILD_PATH'] + '/tools/mlir/python_packages/mlir_core')
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + [os.environ.get('PYTHONPATH', '')])


class Progress:
    """Persist completed costs and the current stage, including before a crash.

    A timeout cannot be assigned to a compiler stage merely from a missing final
    result. The parent reads this same result.json to report the last live stage.
    """
    def __init__(self, out, resources=False):
        self.path = Path(out) / 'result.json'
        self.resources = resources
        self.started = time.perf_counter_ns()
        self.result = dict(status='running', last_stage='worker_startup', compilation={})
        self.flush()

    def flush(self):
        self.result['worker_elapsed_ns'] = time.perf_counter_ns() - self.started
        write_json(self.path, self.result)

    @contextmanager
    def stage(self, name):
        self.result['last_stage'] = name
        self.result['stage_started_ns'] = time.perf_counter_ns()
        self.result.setdefault('active_phases', []).append(name)
        self.flush()
        started = time.perf_counter_ns()
        before = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed = False
        try:
            yield
            completed = True
        finally:
            self.result['compilation'][name] = time.perf_counter_ns() - started
            self.result.setdefault('phase_status', {})[name] = 'completed' if completed else 'failed'
            self.result['active_phases'].pop()
            if self.resources:
                after = resource.getrusage(resource.RUSAGE_SELF)
                child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                self.result.setdefault('phase_resources', {})[name] = dict(
                    user_cpu_ns=round((after.ru_utime - before.ru_utime) * 1e9),
                    system_cpu_ns=round((after.ru_stime - before.ru_stime) * 1e9),
                    child_cpu_ns=round((child_after.ru_utime + child_after.ru_stime - children.ru_utime - children.ru_stime) * 1e9),
                    process_peak_rss_kib_before=before.ru_maxrss, process_peak_rss_kib_after=after.ru_maxrss,
                    minor_faults=after.ru_minflt - before.ru_minflt,
                    major_faults=after.ru_majflt - before.ru_majflt,
                    voluntary_context_switches=after.ru_nvcsw - before.ru_nvcsw,
                    involuntary_context_switches=after.ru_nivcsw - before.ru_nivcsw)
            self.flush()


def prepare(config, out, progress):
    """Save shared nonzero inputs, exact reference graph and common MLIR input."""
    import numpy as np
    from backends import reference_plan, Unsupported
    from rl_autoschedular_v5.full_model import ensure_timed_main_abi

    out = Path(config['run'])
    path = Path(config['source'])
    if path.stat().st_size > config['max_source_mib'] * 1024**2:
        raise Unsupported(f'Source exceeds --max-source-mib ({path.stat().st_size / 1024**2:.1f} MiB)')
    with progress.stage('source_read_ns'):
        code = path.read_text()
    if '__elided__' in code:
        raise Unsupported('Missing/elided model weights; real weights are required for comparison')
    if config['row']['benchmark'] == 'resnext50_linalg.mlir':
        raise Unsupported('Full ResNeXt requires its missing external BatchNorm state/input fixtures')
    with progress.stage('abi_adaptation_ns'):
        code = ensure_timed_main_abi(code)
    with progress.stage('reference_translation_ns'):
        try:
            plan, constants = reference_plan(code)
            reference_error = None
            config['inputs'] = plan['inputs']
            write_json(out / 'reference_plan.json', plan)
            np.savez(out / 'constants.npz', **constants)
        except Unsupported as error:
            # An unsupported framework reference does not prohibit an honest
            # MLIR-only diagnostic. Such latency ratios remain unverified.
            reference_error = str(error)
            from mlir.ir import Context, Module
            from backends import tensor_type
            with Context():
                module = Module.parse(code)
                main = next(op for op in module.body.operations
                            if op.operation.name == 'func.func' and str(op.name.value) == 'main')
                config['inputs'] = [dict(name=f'a{i}', shape=tensor_type(a.type)[0],
                                        dtype=tensor_type(a.type)[1]) for i, a in enumerate(main.arguments)]
    input_bytes = sum(math.prod(a['shape']) * np.dtype(a['dtype']).itemsize for a in config['inputs'])
    if input_bytes > config['max_input_mib'] * 1024**2:
        raise Unsupported(f'Input buffers exceed --max-input-mib ({input_bytes / 1024**2:.1f} MiB)')
    cases = {}
    with progress.stage('input_generation_ns'):
        for case in range(config['correctness_cases']):
            rng = np.random.default_rng(config['seed'] + case)
            for spec in config['inputs']:
                dtype = np.dtype(spec['dtype'])
                if dtype.kind in 'iu':
                    value = rng.integers(0, 2, spec['shape']).astype(dtype)
                elif dtype.kind == 'b':
                    value = np.ones(spec['shape'], dtype=dtype)
                else:
                    value = rng.normal(0, 0.1, spec['shape']).astype(dtype)
                cases[f"case_{case}_{spec['name']}"] = value
        np.savez(out / 'inputs.npz', **cases)
    # Standard V5 loader preprocessing is common to both MLIR methods. It is
    # deliberately recorded: comparing raw vs img2col schedules mixes factors.
    if config['preprocess'] == 'standard' and config['name'].startswith('conv_2d_'):
        from rl_autoschedular_v5.state import extract_bench_features_from_code
        from rl_autoschedular_v5.transforms import transform_img2col
        from mlir.ir import Context, Module
        with progress.stage('common_preprocessing_ns'):
            features = extract_bench_features_from_code(config['name'], code, 1)
            with Context():
                module = Module.parse(features.code)
                for tag in features.operation_tags:
                    if 'conv_2d' in features.operations[tag].operation_name:
                        transform_img2col(module, tag)
                code = str(module)
    (out / 'prepared.mlir').write_text(code)
    config.update(reference_error=reference_error, input_bytes=input_bytes,
                  source_sha256=digest(path), inputs_sha256=digest(out / 'inputs.npz'),
                  prepared_sha256=digest(out / 'prepared.mlir'))
    write_json(out / 'manifest.json', config)
    progress.result.update(status='prepared', reference_error=reference_error)


def schedule(config, out, progress):
    """Infer once, then audit actual application separately from planned actions."""
    import torch
    from mlir.ir import Context, Module
    from rl_autoschedular_v5.state import extract_bench_features_from_code
    from rl_autoschedular_v5.model import HiearchyModel
    from rl_autoschedular_v5.observation import Observation
    from rl_autoschedular_v5.env import Env
    from rl_autoschedular_v5.actions import ActionSpace
    from backends import ir_inventory

    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    with progress.stage('feature_extraction_ns'):
        features = extract_bench_features_from_code(config['name'],
            (Path(config['run']) / 'prepared.mlir').read_text(), 1)
    dependencies = {tag: asdict(op) for tag, op in features.operations.items()}
    write_json(out / 'dependencies.json', dependencies)
    write_json(out / 'dependency_graph.json', dict(
        nodes=[dict(tag=tag, operation=op.operation_name) for tag, op in features.operations.items()],
        edges=[dict(producer=producer, consumer=tag, operand_index=index)
               for tag, op in features.operations.items() for producer, index in op.producers],
        scope='Extracted operation graph before scheduling'))
    with progress.stage('checkpoint_loading_ns'):
        model = HiearchyModel().cpu().eval()
        checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=True)
    environment = Env()
    state = environment.reset([features], 0)
    decisions = []
    inference_ns = 0
    inference_cpu_ns = 0
    with progress.stage('policy_rollout_ns'), torch.inference_mode():
        while True:
            cpu_start = time.process_time_ns()
            start = time.perf_counter_ns()
            indices, _, _ = model.sample(Observation.from_state(state), greedy=True)
            action = ActionSpace.action_by_index(indices[0].cpu(), state)
            decision_ns = time.perf_counter_ns() - start
            decision_cpu_ns = time.process_time_ns() - cpu_start
            inference_ns += decision_ns
            inference_cpu_ns += decision_cpu_ns
            progress.result['compilation']['policy_inference_ns'] = inference_ns
            progress.result['policy_inference_cpu_ns'] = inference_cpu_ns
            decisions.append(dict(operation=state.operation_tag, action=str(action), ready=action.ready,
                parameters=action.parameters, extras=dict(action.extras),
                producer_tag=state.producer_tag, producer_operand_idx=state.producer_operand_idx,
                inference_ns=decision_ns, inference_cpu_ns=decision_cpu_ns))
            write_json(out / 'schedule.json', decisions)
            state = environment.step(state, action)
            if len(decisions) > config['max_steps']:
                raise RuntimeError('Policy exceeded --max-steps')
            if state.terminal:
                following = environment.get_next_op_state(state)
                if following is None: break
                state = following
    applied = []
    with Context():
        with progress.stage('schedule_ir_parse_ns'):
            module = Module.parse(features.code)
        initial_ir = ir_inventory(module)
        write_json(out / 'schedule_ir_analysis.json', dict(before=initial_ir))
        with progress.stage('schedule_application_ns'):
            apply_actions(state, module, out, progress, applied)
        final_ir = ir_inventory(module)
        write_json(out / 'schedule_ir_analysis.json', dict(before=initial_ir, after=final_ir))
        (out / 'scheduled.mlir').write_text(str(module))
    progress.result.update(status='scheduled', schedule_status='partial' if any(
        a['status'] == 'fallback' for a in applied) else 'applied', decisions=len(decisions),
        checkpoint_sha256=digest(config['checkpoint']),
        transformation_execution_ns=sum(a['application_ns'] for a in applied))


def apply_actions(state, module, out, progress, applied):
    """Audit each real action and confirm fused clones in the resulting IR."""
    from backends import ir_inventory
    # Retain only one set of IR snapshots per schedule, not a file per action.
    def tags():
        return ir_inventory(module)['tagged_operations']

    try:
        for sequence in reversed(state.transformation_history):
            for action in sequence:
                if not action.ready: continue
                entry = dict(operation=action.operation_tag, planned=str(action),
                    parameters=action.parameters, extras=dict(action.extras), status='applying')
                fusion = getattr(action, 'symbol', None) == 'TPF'
                if fusion:
                    before = tags()
                    entry['fusion'] = dict(producer=action.producer_tag, consumer=action.consumer_tag,
                        operand_index=action.producer_operand_idx, fused_tag=action.new_producer_tag,
                        before={tag: before.get(tag, []) for tag in
                                (action.producer_tag, action.consumer_tag, action.new_producer_tag)})
                applied.append(entry)
                write_json(out / 'applied_actions.json', applied)
                phase = f'action_{len(applied):05d}_ns'
                try:
                    with progress.stage(phase):
                        action.apply(module)
                    entry.update(actual=str(action), status='fallback' if action.extras.get('vectorized') is False else 'applied',
                                 extras=dict(action.extras))
                except Exception as error:
                    entry.update(status='failed', reason=str(error))
                    raise
                finally:
                    entry['application_ns'] = progress.result['compilation'][phase]
                    entry['phase'] = phase
                    if fusion:
                        after = tags()
                        audit = entry['fusion']
                        audit['after'] = {tag: after.get(tag, []) for tag in
                            (action.producer_tag, action.consumer_tag, action.new_producer_tag)}
                        created = after.get(action.new_producer_tag, [])
                        audit['outcome'] = ('confirmed_fused_clone' if entry['status'] == 'applied'
                            and created and not before.get(action.new_producer_tag) else 'unconfirmed')
                        audit['original_producer_remaining'] = bool(after.get(action.producer_tag))
                    write_json(out / 'applied_actions.json', applied)
    finally:
        write_json(out / 'fusion_report.json', [dict(action_index=i + 1, status=a['status'], **a['fusion'])
            for i, a in enumerate(applied) if 'fusion' in a])


def thread_ticks():
    """Per-thread CPU ticks supplement perf counters with observed parallel use."""
    result = {}
    for path in Path('/proc/self/task').glob('*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[path.parent.name] = int(fields[11]) + int(fields[12])
        except (OSError, ValueError): pass
    return result


def thread_affinities():
    """Record actual Linux thread masks; a main-thread mask alone is misleading."""
    masks = {}
    for path in Path('/proc/self/task').iterdir():
        try:
            masks[path.name] = sorted(os.sched_getaffinity(int(path.name)))
        except ProcessLookupError: pass
    return masks


class PerfCounters:
    """Attach disabled perf events *after* compilation, then control the window.

    Counters include the repeated-call harness, restoration and disposal. They
    are not claimed to be kernel-only memory bandwidth or vector utilization.
    Missing events/permissions remain null, with raw diagnostics retained.
    """
    def __init__(self, out, enabled):
        self.out, self.process, self.fds, self.log, self.error = out, None, [], None, None
        if not enabled or not shutil.which('perf'):
            self.error = 'Disabled by --no-perf' if not enabled else 'perf executable unavailable'
            return
        try:
            read, write = os.pipe()
            ack_read, ack_write = os.pipe()
            self.fds = [read, write, ack_read, ack_write]
            self.write, self.read = write, ack_read
            self.log = (out / 'perf.log').open('w')
            self.process = subprocess.Popen(['perf', 'stat', '-p', str(os.getpid()), '--delay=-1',
                f'--control=fd:{read},{ack_write}', '-x', ';', '--no-big-num',
                '-e', 'cycles,instructions,cache-references,cache-misses,branches,branch-misses',
                '-o', str(out / 'perf.csv')], pass_fds=(read, ack_write), stdout=self.log, stderr=self.log)
            self.control('enable')
        except Exception as error:
            self.error = str(error)

    def control(self, message):
        if self.process.poll() is not None: raise RuntimeError('perf exited; see perf.log')
        os.write(self.write, (message + '\n').encode())
        ready, _, _ = select.select([self.read], [], [], 5)
        if not ready or b'ack' not in os.read(self.read, 128):
            raise RuntimeError('No perf control acknowledgement; see perf.log')

    def finish(self, calls):
        if self.process:
            try:
                if not self.error: self.control('disable')
            except Exception as error: self.error = str(error)
            if self.process.poll() is None: self.process.send_signal(signal.SIGINT)
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for fd in self.fds: os.close(fd)
        if self.log: self.log.close()
        counters = {}
        path = self.out / 'perf.csv'
        if path.exists():
            for row in csv.reader(path.read_text().splitlines(), delimiter=';'):
                if len(row) < 3 or row[0].startswith('#'): continue
                try: value = float(row[0].strip())
                except ValueError: value = None
                if self.error or (value is not None and not math.isfinite(value)): value = None
                counters[row[2].strip()] = dict(value=value, per_call=value / calls if value is not None else None,
                                               unit=row[1].strip(), raw=row)
        available = sum(v['value'] is not None for v in counters.values())
        return dict(status='available' if available and available == len(counters) else (
                    'partial' if available else 'unavailable'), reason=self.error or (
                    None if available else 'No readable events; inspect perf.log/perf.csv'), counters=counters,
                    calls=calls, scope='Post-warmup repeated loop; includes harness/reset/disposal')


def stats(values):
    return dict(samples_ns=values, median_ns=statistics.median(values), min_ns=min(values),
        max_ns=max(values), mean_ns=statistics.mean(values),
        stdev_ns=statistics.stdev(values) if len(values) > 1 else 0)


def measure(config, method, out, progress):
    """Separate correctness, latency, hardware profiling and artifact collection."""
    import numpy as np
    from backends import FrameworkRunner, MLIRRunner, Unsupported
    if method.startswith(('torch_', 'tensorflow_')) and config.get('reference_error'):
        raise Unsupported(config['reference_error'])
    with np.load(Path(config['run']) / 'inputs.npz', allow_pickle=False) as archive:
        cases = [[archive[f"case_{case}_{a['name']}"].copy() for a in config['inputs']]
                 for case in range(config['correctness_cases'])]
    runner_type = MLIRRunner if method.startswith('mlir_') else FrameworkRunner
    runner = runner_type(config, method, out, progress)
    runner.reset(cases[0])
    with progress.stage('first_invoke_ns'):
        value = runner.call()
    runner.release(value)
    del value
    outputs = {}
    for case, arrays in enumerate(cases):
        runner.reset(arrays)
        value = runner.call()
        outputs.update({f'case_{case}_out_{i}': a for i, a in enumerate(runner.snapshot())})
        runner.release(value)
        del value
    np.savez(out / 'outputs.npz', **outputs)
    progress.result['last_stage'] = 'warmup'
    progress.flush()
    for _ in range(config['warmup']):
        runner.reset(cases[0])
        value = runner.call()
        runner.release(value)
        del value
    samples, internal = [], []
    progress.result['last_stage'] = 'execution_latency'
    progress.flush()
    for _ in range(config['repeats']):
        runner.reset(cases[0])
        start = time.perf_counter_ns()
        value = runner.call()
        samples.append(time.perf_counter_ns() - start)
        if method.startswith('mlir_'): internal.append(int(runner.outputs.delta))
        runner.release(value)
        del value
    # Persist latency before optional profiling/artifact work, so a subsequent
    # timeout does not erase an already collected (still unvalidated) sample.
    progress.result.update(latency=stats(samples), internal_latency=stats(internal) if internal else None,
                           call_counts=dict(first_invocation=1, correctness=len(cases),
                                            warmup=config['warmup'], measured=len(samples)),
                           last_stage='hardware_profile')
    progress.flush()
    perf = PerfCounters(out, config['perf'])
    ticks = thread_ticks()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter_ns()
    calls = 0
    # Both minima must be satisfied. A slow call is allowed to finish; the
    # one-second minimum is not a deadline, and this window never feeds latency.
    while calls < config['profile_repeats'] or (time.perf_counter_ns() - start) / 1e9 < config['profile_seconds']:
        runner.reset(cases[0])
        value = runner.call()
        runner.release(value)
        del value
        calls += 1
    wall = time.perf_counter_ns() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    after_ticks = thread_ticks()
    cpu_seconds = after.ru_utime + after.ru_stime - usage.ru_utime - usage.ru_stime
    progress.result['call_counts']['profile'] = calls
    progress.result.update(hardware=perf.finish(calls), parallel_execution=dict(
        profile_wall_ns=wall, profile_calls=calls, minimum_calls=config['profile_repeats'],
        minimum_wall_ns=round(config['profile_seconds'] * 1e9),
        cpu_seconds=cpu_seconds, average_busy_cores=cpu_seconds / (wall / 1e9),
        allocated_utilization_percent=100 * cpu_seconds / (wall / 1e9) / config['threads'],
        per_thread_cpu_seconds={tid: (v - ticks.get(tid, 0)) / os.sysconf('SC_CLK_TCK') for tid, v in after_ticks.items()}))
    progress.result['last_stage'] = 'code_artifacts'
    progress.flush()
    progress.result.update(code_artifacts=runner.artifacts(), status='measured',
        cpu_affinity=sorted(os.sched_getaffinity(0)), threads=config['threads'],
        latency_scope='Synchronous host call; excludes input reset, output copies and disposal')


def worker(task):
    """Private subprocess entry: all native crashes/timeouts stay within a worker."""
    out = Path(task['out'])
    progress = Progress(out, resources=task['config'].get('profiling') == 'compilation')
    progress.result['assigned_cpu_affinity'] = task['cpus']
    config = task['config']
    config['cpus'] = task['cpus']
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if not set(task['cpus']).issubset(os.sched_getaffinity(0)):
        raise ValueError('Worker CPUs must be within the inherited allocation')
    os.sched_setaffinity(0, task['cpus'])
    try:
        if task['mode'] == 'prepare': prepare(config, out, progress)
        elif task['mode'] == 'schedule': schedule(config, out, progress)
        elif task['mode'] == 'compile':
            from compilation_profiling import compile_benchmark
            compile_benchmark(config, task['method'], out, progress)
        elif task['mode'] == 'openmp_control': openmp_control(config, out, progress)
        else: measure(config, task['method'], out, progress)
    except Exception as error:
        from backends import Unsupported
        progress.result.update(status='unsupported' if isinstance(error, Unsupported) else (
            'unavailable' if isinstance(error, ModuleNotFoundError) else 'error'),
            reason=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        progress.result['thread_cpu_affinity'] = thread_affinities()
        progress.flush()


def launch(task, python, timeout, threads):
    """Launch sequentially; terminate only this isolated worker process group."""
    out = Path(task['out'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'result.json', dict(status='running', last_stage='process_startup', compilation={}))
    env = dict(os.environ)
    # These vendor overrides take precedence over OMP_PLACES. Use the same
    # explicit, allocation-limited places for every backend instead.
    for key in ('KMP_AFFINITY', 'KMP_HW_SUBSET', 'GOMP_CPU_AFFINITY'):
        env.pop(key, None)
    env.update({k: str(threads) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                         'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS')})
    env.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', OMP_PROC_BIND='close', OMP_THREAD_LIMIT=str(threads),
        OMP_PLACES=','.join('{' + str(cpu) + '}' for cpu in task['cpus']),
        TF_NUM_INTEROP_THREADS='1', CUDA_VISIBLE_DEVICES='', DASK_NODES='0', DASK_TRAINING_ONLY='1',
        TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor_cache'), TORCHINDUCTOR_COMPILE_THREADS='1',
        PYTHONHASHSEED=str(task['config']['seed']), CONFIG_FILE_PATH=str(Path(task['config']['run']) / 'policy_config.json'))
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    compiler = Path(python).parent / 'x86_64-conda-linux-gnu-g++'
    if compiler.is_file(): env.setdefault('CXX', str(compiler))
    started = time.perf_counter_ns()
    with (out / 'worker.log').open('w') as log:
        try:
            process = subprocess.Popen([str(python), str(Path(__file__).resolve()), '_worker'],
                stdin=subprocess.PIPE, stdout=log, stderr=log, text=True, env=env, cwd=ROOT, start_new_session=True)
        except OSError as error:
            saved = dict(status='unavailable', last_stage='process_startup', reason=str(error))
            write_json(out / 'result.json', saved)
            return saved
        try:
            process.communicate(json.dumps(task), timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            saved = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
            saved.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'timeout',
                         reason=f'Worker wall-time cap: {timeout}s', timeout_seconds=timeout)
            write_json(out / 'result.json', saved)
            if isinstance(error, KeyboardInterrupt): raise
    path = out / 'result.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    # SIGKILL is expected after our own timeout. Preserve that classification
    # instead of overwriting it with the subprocess's resulting -9 exit code.
    if saved.get('status') not in ('timeout', 'interrupted') and (
            saved.get('status') in (None, 'running') or process.returncode):
        saved.update(status='error', reason=f'Worker exited {process.returncode}; inspect worker.log')
    saved['process_wall_ns'] = time.perf_counter_ns() - started
    write_json(path, saved)
    return saved


def physical_cpus():
    """Choose one allocated logical CPU per physical core, ordered by NUMA node."""
    chosen = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        topology = base / 'topology'
        key = ((topology / 'physical_package_id').read_text().strip(),
               (topology / 'core_id').read_text().strip())
        nodes = list(base.glob('node[0-9]*'))
        chosen.setdefault(key, (int(nodes[0].name[4:]) if nodes else 0, cpu))
    return [cpu for _, cpu in sorted(chosen.values())]


# Generated only by check-openmp; this links the exact libomp used by MLIR.
# Eight independent lanes do identical useful arithmetic in the serial and
# parallel runs. Checksums prevent dead-code elimination and verify equality.
OPENMP_CONTROL_SOURCE = r'''
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
typedef struct {
    int thread, team, tid, cpu, affinity_count, affinity_cpu;
    double cpu_seconds;
    uint64_t checksum;
} Record;
static double cpu_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
void run_control(int threads, uint64_t iterations, Record *records) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int lane = 0; lane < 8; ++lane) {
        Record *r = &records[lane];
        double start = cpu_time();
        uint64_t x = lane + 1;
        for (uint64_t i = 0; i < iterations; ++i) {
            x ^= x >> 13;
            x *= UINT64_C(2862933555777941757);
            x ^= x << 7;
        }
        r->checksum = x;
        r->cpu_seconds = cpu_time() - start;
        r->thread = omp_get_thread_num();
        r->team = omp_get_num_threads();
        r->tid = syscall(SYS_gettid);
        r->cpu = sched_getcpu();
        cpu_set_t mask;
        CPU_ZERO(&mask);
        sched_getaffinity(0, sizeof(mask), &mask);
        r->affinity_count = CPU_COUNT(&mask);
        r->affinity_cpu = -1;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) { r->affinity_cpu = cpu; break; }
    }
}
'''


def openmp_control(config, out, progress):
    """Exercise the Torch -> LLVM initialization order used by MLIR workers."""
    import ctypes
    import torch
    from backends import initialize_mlir_openmp
    torch.set_num_threads(1)  # Compilation profiling first uses a serial policy.
    torch.set_num_threads(8)
    progress.result['openmp_placement'] = initialize_mlir_openmp(config['cpus'], 8)

    class Record(ctypes.Structure):
        _fields_ = [(key, ctypes.c_int) for key in
            ('thread', 'team', 'tid', 'cpu', 'affinity_count', 'affinity_cpu')]
        _fields_ += [('cpu_seconds', ctypes.c_double), ('checksum', ctypes.c_uint64)]

    native = ctypes.CDLL(str(out / 'control.so'))
    native.run_control.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(Record)]
    native.run_control.restype = None

    def run(threads, iterations):
        records = (Record * 8)()
        start = time.perf_counter()
        native.run_control(threads, iterations, records)
        wall = time.perf_counter() - start
        rows = [{key: getattr(row, key) for key, _ in Record._fields_} for row in records]
        return dict(wall_seconds=wall, average_busy_cores=sum(r['cpu_seconds'] for r in rows) / wall,
            lanes=rows)

    run(8, 10000)  # Initialize the team outside the measurement.
    calibration = run(1, 1000000)
    # Aim for about half a CPU-second per lane, bounded even on unusual hosts.
    lane_seconds = sum(r['cpu_seconds'] for r in calibration['lanes']) / 8
    iterations = max(1000000, min(2000000000, round(1000000 * .5 / max(lane_seconds, 1e-9))))
    serial, parallel = run(1, iterations), run(8, iterations)
    speedup = serial['wall_seconds'] / parallel['wall_seconds']
    checks = dict(
        eight_threads={r['thread'] for r in parallel['lanes']} == set(range(8))
            and all(r['team'] == 8 for r in parallel['lanes']),
        eight_distinct_allocated_cpus={r['cpu'] for r in parallel['lanes']} == set(config['cpus']),
        exact_thread_binding=all(r['affinity_count'] == 1 and
            r['affinity_cpu'] == config['cpus'][r['thread']] for r in parallel['lanes']),
        every_thread_did_work=all(r['cpu_seconds'] >= .05 for r in parallel['lanes']),
        same_outputs=[r['checksum'] for r in serial['lanes']] == [r['checksum'] for r in parallel['lanes']],
        concurrent_cpu_use=parallel['average_busy_cores'] >= 6,
        useful_work_speedup=speedup >= 2)
    progress.result.update(status='verified' if all(checks.values()) else 'error', checks=checks,
        iterations_per_lane=iterations, serial=serial, parallel=parallel, speedup=speedup,
        reason=None if all(checks.values()) else 'OpenMP control failed; inspect checks and thread records')


def check_openmp_command(args):
    """Compile/run a small eight-core control without loading dataset files."""
    cpus = physical_cpus()[:8]
    if len(cpus) != 8:
        raise ValueError('check-openmp requires eight allocated physical cores')
    load_environment()
    output = args.output.resolve()
    if output.is_relative_to((ROOT / 'data').resolve()): raise ValueError('Output must be outside data/')
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(config=str(Path(__file__).resolve()), profiling='openmp_control',
        seed=2026, threads=[8], trials=1, run=str(output))
    register('openmp_control', output, settings, 'running')
    try:
        build = Path(os.environ['LLVM_BUILD_PATH'])
        (output / 'control.c').write_text(OPENMP_CONTROL_SOURCE)
        toolchain = []
        gcc = Path(os.environ.get('CONDA_ENV', sys.prefix)) / 'bin/x86_64-conda-linux-gnu-gcc'
        if gcc.is_file():
            # HPC nodes may have no system GCC installation for Clang to find.
            crt = subprocess.check_output([str(gcc), '-print-file-name=crtbeginS.o'], text=True).strip()
            sysroot = subprocess.check_output([str(gcc), '-print-sysroot'], text=True).strip()
            toolchain = ['--gcc-install-dir=' + str(Path(crt).resolve().parent), '--sysroot=' + sysroot]
        # The build-tree omp.h is generated from this LLVM checkout.
        with (output / 'compiler.log').open('w') as log:
            subprocess.run([str(build / 'bin/clang'), *toolchain, '-O2', '-fPIC', '-shared', '-fopenmp',
                '-I' + str(build / 'projects/openmp/runtime/src'), '-L' + str(build / 'lib'),
                '-Wl,-rpath,' + str(build / 'lib'), str(output / 'control.c'),
                '-o', str(output / 'control.so')], stdout=log, stderr=log, check=True, timeout=60)
        python = args.python or str(Path(os.environ.get('CONDA_ENV', sys.prefix)) / 'bin/python')
        result = launch(dict(mode='openmp_control', cpus=cpus, config=settings, out=str(output)),
                        python, timeout=120, threads=8)
        passed = result['status'] == 'verified'
        register('openmp_control', output, settings, 'done' if passed else 'failed')
        if passed:
            print(f"OpenMP verified: 8 threads on {cpus}; {result['parallel']['average_busy_cores']:.2f} busy cores; "
                  f"{result['speedup']:.2f}x serial speedup")
        else: print(result.get('reason', 'OpenMP control failed'))
        print(output / 'result.json')
        return 0 if passed else 1
    except Exception:
        register('openmp_control', output, settings, 'failed')
        raise


def compare(left, right, config):
    """Compare *all* saved cases/outputs; nonfinite values and dtype drift fail."""
    import numpy as np
    if not left.exists() or not right.exists(): return dict(passed=False, reason='Missing output archive')
    errors = []
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files): return dict(passed=False, reason='Output count/key mismatch')
        for key in a.files:
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                return dict(passed=False, reason=f'Shape/dtype mismatch: {key}')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                return dict(passed=False, reason=f'Nonfinite output: {key}')
            passed = np.array_equal(x, y) if x.dtype.kind in 'iub' else np.allclose(x, y, rtol=config['rtol'], atol=config['atol'])
            if not passed: return dict(passed=False, reason=f'Numerical mismatch: {key}')
            errors.append(float(np.max(np.abs(x.astype('float64') - y.astype('float64')))) if x.size else 0)
    return dict(passed=True, max_abs_error=max(errors, default=0))


def collect(output, config):
    """Rebuild compact category summaries and the familiar SCHEDULES.md index."""
    rows, schedules = [], []
    for manifest in sorted((output / 'artifacts').glob('*/*/manifest.json')):
        run = manifest.parent
        spec = json.loads(manifest.read_text())
        schedule_file = run / 'methods/mlir_rl/schedule.json'
        if schedule_file.exists():
            decisions = json.loads(schedule_file.read_text())
            applied_path = schedule_file.with_name('applied_actions.json')
            applied = json.loads(applied_path.read_text()) if applied_path.exists() else []
            links = [f'[Raw decisions]({schedule_file.relative_to(output)})']
            if applied_path.exists(): links.append(f'[Applied actions]({applied_path.relative_to(output)})')
            scheduled = schedule_file.with_name('scheduled.mlir')
            for name in ('dependency_graph.json', 'fusion_report.json', 'schedule_ir_analysis.json'):
                if schedule_file.with_name(name).exists():
                    links.append(f'[{name}]({schedule_file.with_name(name).relative_to(output)})')
            links.append(f'[Applied MLIR]({scheduled.relative_to(output)})' if scheduled.exists()
                         else f'[Schedule did not complete]({schedule_file.with_name("result.json").relative_to(output)})')
            schedules.append(f"<details>\n<summary>{spec['row']['dataset']}/{spec['row']['benchmark']}</summary>\n\n"
                + '\n'.join(f"- {d['operation']}: {d['action']}" for d in decisions if d['ready'])
                + '\n\n' + '\n'.join(f"- Applied: {a['operation']}: {a.get('actual', a['planned'])} — {a['status']}" for a in applied)
                + '\n\n' + ' · '.join(links) + '\n\n</details>\n')
        for threads in config['threads']:
            by_method = {method: [] for method in config['methods']}
            for trial in range(1, config['trials'] + 1):
                paths = {method: run / 'methods' / method / f'threads_{threads}' / f'trial_{trial}' for method in config['methods']}
                results = {method: json.loads((path / 'result.json').read_text())
                           for method, path in paths.items() if (path / 'result.json').exists()}
                baseline, eager = results.get('mlir_baseline', {}), results.get('torch_eager', {})
                reference_ok = (baseline.get('status') == eager.get('status') == 'measured' and
                    compare(paths['mlir_baseline'] / 'outputs.npz', paths['torch_eager'] / 'outputs.npz', spec)['passed'])
                for method, result in results.items():
                    correctness = 'unavailable'
                    if result.get('status') == 'measured':
                        checks = {reference: compare(paths[method] / 'outputs.npz', paths[reference] / 'outputs.npz', spec)
                                  for reference in ('mlir_baseline', 'torch_eager')
                                  if results.get(reference, {}).get('status') == 'measured'}
                        correctness = 'failed' if any(not c['passed'] for c in checks.values()) else (
                            'passed' if reference_ok else 'unverified')
                        result['output_checks'] = checks
                    result['correctness'] = correctness
                    result['validated_speedup'] = (baseline['latency']['median_ns'] / result['latency']['median_ns']
                                                  if correctness == 'passed' else None)
                    write_json(paths[method] / 'result.json', result)
                    by_method[method].append(result)
            for method, results in by_method.items():
                valid = [r for r in results if r.get('status') == 'measured' and r.get('correctness') == 'passed']
                measured = [r for r in results if r.get('status') == 'measured']
                med = lambda values: statistics.median(values) if values else None
                def event(result, name):
                    return next((c['per_call'] for key, c in result['hardware']['counters'].items()
                                 if key.split(':')[0] == name), None)
                events = {name: [event(r, name) for r in measured] for name in (
                    'cycles', 'instructions', 'cache-references', 'cache-misses')}
                schedule_result = run / 'methods/mlir_rl/result.json'
                costs = (json.loads(schedule_result.read_text()).get('compilation', {})
                         if method == 'mlir_rl' and schedule_result.exists() else {})
                def ir_count(stage, key):
                    values = [r.get('code_artifacts', {}).get('static_ir_counts', {}).get(stage, {}).get(key)
                              for r in measured]
                    return med([v for v in values if v is not None])
                rows.append(dict(dataset=spec['row']['dataset'], benchmark=spec['row']['benchmark'],
                    threads=threads, method=method, attempted_trials=len(results), valid_trials=len(valid),
                    median_host_ns=med([r['latency']['median_ns'] for r in measured]),
                    median_valid_host_ns=med([r['latency']['median_ns'] for r in valid]),
                    median_validated_speedup=med([r['validated_speedup'] for r in valid]),
                    trial_min_ns=min([r['latency']['median_ns'] for r in measured], default=None),
                    trial_max_ns=max([r['latency']['median_ns'] for r in measured], default=None),
                    average_busy_cores=med([r['parallel_execution']['average_busy_cores'] for r in measured]),
                    allocated_utilization_percent=med([r['parallel_execution']['allocated_utilization_percent'] for r in measured]),
                    median_profile_calls=med([r['parallel_execution']['profile_calls'] for r in measured]),
                    median_profile_wall_ns=med([r['parallel_execution']['profile_wall_ns'] for r in measured]),
                    cycles_per_profile_call=med([v for v in events['cycles'] if v is not None]),
                    instructions_per_profile_call=med([v for v in events['instructions'] if v is not None]),
                    cache_misses_per_profile_call=med([v for v in events['cache-misses'] if v is not None]),
                    instructions_per_cycle=med([i / c for i, c in zip(events['instructions'], events['cycles'])
                                                if i is not None and c]),
                    cache_miss_rate=med([m / r for m, r in zip(events['cache-misses'], events['cache-references'])
                                         if m is not None and r]),
                    feature_extraction_ns=costs.get('feature_extraction_ns'),
                    policy_inference_ns=costs.get('policy_inference_ns'),
                    schedule_application_ns=costs.get('schedule_application_ns'),
                    median_lowering_ns=med([r['compilation']['mlir_lowering_ns'] for r in measured
                                            if 'mlir_lowering_ns' in r['compilation']]),
                    median_jit_engine_creation_ns=med([r['compilation']['jit_engine_creation_ns'] for r in measured
                                                       if 'jit_engine_creation_ns' in r['compilation']]),
                    median_first_invoke_ns=med([r['compilation']['first_invoke_ns'] for r in measured]),
                    intermediate_tensor_values=ir_count('input', 'intermediate_tensor_result_values'),
                    bufferized_allocation_sites=ir_count('bufferized', 'allocation_sites'),
                    bufferized_intermediate_written_buffers=ir_count('bufferized', 'intermediate_written_buffer_identities'),
                    bufferized_non_returned_allocation_sites=ir_count('bufferized', 'non_returned_allocation_sites'),
                    lowered_heap_allocation_call_sites=ir_count('lowered', 'llvm_heap_allocation_call_sites'),
                    lowered_stack_allocation_sites=ir_count('lowered', 'llvm_stack_allocation_sites'),
                    status=';'.join(sorted({r['status'] for r in results})) or 'not_run',
                    correctness=';'.join(sorted({r['correctness'] for r in results})) or 'unavailable',
                    reason='; '.join(sorted({r['reason'] for r in results if r.get('reason')})),
                    artifacts=str(run.relative_to(output))))
    # Derive relative backend performance at the same thread budget only when
    # both groups passed validation. These are ratios of trial-median aggregates.
    for row in rows:
        peers = {r['method']: r for r in rows if (r['dataset'], r['benchmark'], r['threads']) ==
                 (row['dataset'], row['benchmark'], row['threads'])}
        for reference in ('torch_eager', 'torch_inductor', 'tensorflow_xla'):
            other = peers.get(reference, {})
            row['speedup_vs_' + reference] = (other['median_valid_host_ns'] / row['median_valid_host_ns']
                if row['valid_trials'] and other.get('valid_trials') else None)
    if rows:
        with (output / 'comparison.csv').open('w', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / 'summary.json', dict(rows=rows, settings=config))
    (output / 'SCHEDULES.md').write_text('# Generated V5 schedules\n\n'
        'Planned choices are followed by actual application outcomes. Pointer decisions '
        'remain in schedule.json. Inspect result.json for failures/timeouts.\n\n' + '\n'.join(schedules))
    return rows


def register(category, output, settings, state):
    """Register before launching, as required by the repository's AGENTS.md."""
    path = ROOT / 'experiments.json'
    registry = json.loads(path.read_text())
    profiling = settings.get('profiling', 'execution')
    name = ('openmp_placement_control' if profiling == 'openmp_control' else
        ('compilation_diagnosis_' if profiling == 'compilation' else 'performance_diagnosis_') + category)
    entry = next((r for r in registry['experiments'] if r['name'] == name), None)
    if entry is None:
        entry = dict(name=name)
        registry['experiments'].append(entry)
    entry.update(config=str(Path(settings['config']).relative_to(ROOT)) if Path(settings['config']).is_relative_to(ROOT) else settings['config'],
        results_dir=str(output), mode=profiling + '_profiling', seed=settings['seed'], state=state,
        description=('Eight-thread LLVM OpenMP placement and useful-work control after Torch initialization'
            if profiling == 'openmp_control' else
            f"Five-backend {profiling} profiling; fixed inventory samples; threads={settings['threads']}; "
            f"{settings['trials']} independent trials"))
    write_json(path, registry)


def selected_rows(args):
    """Apply the same CSV-only filters to both profiling entry points.

    A stem, filename or dataset/filename is accepted. Category can be inferred
    for a single workload, but a run always belongs to one logical category.
    """
    _, samples = read_csv(args.samples)
    rows = [r for r in samples if not args.category or r['category'] == args.category]
    if args.op_type:
        kinds = {s if s.startswith('linalg.') else 'linalg.' + s for s in args.op_type}
        rows = [r for r in rows if kinds.intersection(json.loads(r['operation_types']))]
    if args.only:
        def names(row):
            stem = Path(row['benchmark']).stem
            return {row['benchmark'], stem, row['dataset'] + '/' + row['benchmark'], row['dataset'] + '/' + stem}
        rows = [r for r in rows if names(r).intersection(args.only)]
        matched = set().union(*(names(r) for r in rows))
        if set(args.only) - matched:
            raise ValueError('--benchmark contains names outside the selected category/type/sample CSV: ' + ', '.join(sorted(set(args.only) - matched)))
    if not rows: raise ValueError('No sampled benchmarks match this selection')
    categories = {r['category'] for r in rows}
    if len(categories) != 1: raise ValueError('Selection spans categories; specify --category')
    args.category = next(iter(categories))
    return rows


def run_command(args):
    """Shared isolation/resume orchestration; each profiler measures its own work."""
    rows = selected_rows(args)
    compilation = args.profiling == 'compilation'
    if compilation:
        from compilation_profiling import collect_compilation
    collector = collect_compilation if compilation else collect
    cpus = physical_cpus()
    if args.threads > len(cpus):
        raise ValueError(f'Requested {args.threads} threads but only {len(cpus)} allocated physical cores are available')
    for row in rows: print(row['category'], row['dataset'], row['benchmark'], flush=True)
    print('CPU budget:', args.threads, 'threads on', cpus[:args.threads], flush=True)
    if args.dry_run: return 0
    load_environment()
    # Do not resolve executable symlinks: a venv's bin/python may point at the
    # base interpreter, but its *invocation path* selects the venv packages.
    python = Path(os.path.abspath(args.python or (Path(os.environ.get('CONDA_ENV', sys.prefix)) / 'bin/python')))
    tensorflow_python = Path(os.path.abspath(args.tensorflow_python or python))
    output = (args.output / args.category).resolve()
    if output.is_relative_to((ROOT / 'data').resolve()): raise ValueError('Output must be outside data/')
    settings = dict(profiling=args.profiling, python_profile=args.python_profile,
        # Keep the existing campaign/artifact schema, with exactly one budget.
        threads=[args.threads], trials=args.trials, repeats=args.repeats,
        warmup=args.warmup, correctness_cases=args.correctness_cases, seed=args.seed,
        timeout=args.timeout, methods=args.methods, perf=not args.no_perf, profile_seconds=args.profile_seconds,
        profile_repeats=args.profile_repeats, max_input_mib=args.max_input_mib, max_source_mib=args.max_source_mib,
        max_steps=args.max_steps, rtol=args.rtol, atol=args.atol, preprocess=args.preprocess,
        config=str(args.config.resolve()), config_sha256=digest(args.config), checkpoint=str(args.checkpoint.resolve()),
        sample_sha256=digest(args.samples), checkpoint_sha256=digest(args.checkpoint),
        hostname=platform.node(), cpu_budgets={str(args.threads): cpus[:args.threads]},
        python=str(python), tensorflow_python=str(tensorflow_python),
        runner_sha256=digest(__file__), backend_sha256=digest(HERE / 'backends.py'),
        compilation_runner_sha256=digest(HERE / 'compilation_profiling.py') if compilation else None)
    output.mkdir(parents=True, exist_ok=True)
    campaign = output / 'campaign.json'
    if campaign.exists() and json.loads(campaign.read_text()) != settings:
        raise ValueError('Campaign settings/source/host/CPU/checkpoint differ; choose a new --output directory')
    write_json(campaign, settings)
    register(args.category, output, settings, 'running')
    try:
        for row in rows:
            run = output / 'artifacts' / row['dataset'] / Path(row['benchmark']).stem
            run.mkdir(parents=True, exist_ok=True)
            spec = dict(settings, row=row, name=Path(row['benchmark']).stem,
                        source=str(source_path(row)), run=str(run))
            cfg = json.loads(Path(settings['config']).read_text())
            if cfg.get('implementation') != 'rl_autoschedular_v5':
                raise ValueError('This runner deliberately uses only the standalone V5 package')
            cfg.update(results_dir=str(run / 'policy_logs'), dask_node_count=0)
            write_json(run / 'policy_config.json', cfg)
            prep_out = run / 'preparation'
            prep_path = prep_out / 'result.json'
            prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
            if not prep or args.retry_failed and prep.get('status') != 'prepared':
                prep = launch(dict(mode='prepare', config=spec, out=str(prep_out), cpus=cpus[:1]), python, args.timeout, 1)
            if prep.get('status') != 'prepared':
                write_json(run / 'manifest.json', spec)
                print(f"  {row['benchmark']}: preparation {prep.get('status')}: {prep.get('reason')}", flush=True)
                for n in settings['threads']:
                    for trial in range(1, args.trials + 1):
                        for method in args.methods:
                            dest = run / 'methods' / method / f'threads_{n}' / f'trial_{trial}'
                            dest.mkdir(parents=True, exist_ok=True)
                            write_json(dest / 'result.json', dict(status='blocked', correctness='unavailable',
                                reason=prep.get('reason'), last_stage=prep.get('last_stage')))
                collector(output, settings)
                continue
            spec = json.loads((run / 'manifest.json').read_text())
            if spec['source_sha256'] != digest(spec['source']) or spec['inputs_sha256'] != digest(run / 'inputs.npz'):
                raise ValueError('Source or saved inputs changed; choose a new --output directory')
            schedule_out = run / 'methods/mlir_rl'
            schedule_result = {}
            if 'mlir_rl' in args.methods and not compilation:
                path = schedule_out / 'result.json'
                schedule_result = json.loads(path.read_text()) if path.exists() else {}
                if not schedule_result or args.retry_failed and schedule_result.get('status') != 'scheduled':
                    schedule_result = launch(dict(mode='schedule', config=spec, out=str(schedule_out), cpus=cpus[:1]),
                                             python, args.timeout, 1)
            for n in settings['threads']:
                for trial in range(1, args.trials + 1):
                    methods = list(args.methods)
                    random.Random(f"{args.seed}:{row['dataset']}:{row['benchmark']}:{n}:{trial}").shuffle(methods)
                    for method in methods:
                        dest = run / 'methods' / method / f'threads_{n}' / f'trial_{trial}'
                        path = dest / 'result.json'
                        previous = json.loads(path.read_text()) if path.exists() else None
                        if previous and (not args.retry_failed or previous.get('status') == ('compiled' if compilation else 'measured')): continue
                        if not compilation and method == 'mlir_rl' and schedule_result.get('status') != 'scheduled':
                            dest.mkdir(parents=True, exist_ok=True)
                            write_json(path, dict(status='blocked', reason='RL scheduling: ' + schedule_result.get('reason', ''),
                                                 last_stage=schedule_result.get('last_stage')))
                            continue
                        worker_config = dict(spec, threads=n)
                        print(f"  {row['benchmark']} / {n} threads / trial {trial} / {method}", flush=True)
                        result = launch(dict(mode='compile' if compilation else 'measure', config=worker_config,
                            method=method, out=str(dest), cpus=cpus[:n]),
                            tensorflow_python if method == 'tensorflow_xla' else python, args.timeout, n)
                        print('   ', result['status'], result.get('reason', ''), flush=True)
                    collector(output, settings)
        summaries = collector(output, settings)
        requested = {(r['dataset'], r['benchmark']) for r in rows}
        passed = all(r['compiled_trials' if compilation else 'valid_trials'] == args.trials for r in summaries
                     if (r['dataset'], r['benchmark']) in requested)
        register(args.category, output, settings, 'done' if passed else 'failed')
        print(output / ('compilation.csv' if compilation else 'comparison.csv'), flush=True)
        return 0 if passed else 1
    except KeyboardInterrupt:
        register(args.category, output, settings, 'stopped')
        print('Interrupted; completed workers retained. Reuse the same command to resume.', flush=True)
        return 130
    except Exception:
        register(args.category, output, settings, 'failed')
        raise


def doctor_command(args):
    """Check imports and real CPU XLA compilation in separate child processes."""
    load_environment()
    python = args.python or str(Path(os.environ.get('CONDA_ENV', sys.prefix)) / 'bin/python')
    checks = [('MLIR/PyTorch', python, ('numpy', 'torch', 'mlir.ir')),
              ('TensorFlow XLA', args.tensorflow_python or python, ('tensorflow',))]
    budget = min(8, len(physical_cpus()))
    env = dict(os.environ, OMP_NUM_THREADS=str(budget), OPENBLAS_NUM_THREADS=str(budget),
        TF_NUM_INTRAOP_THREADS=str(budget), TF_NUM_INTEROP_THREADS='1', CUDA_VISIBLE_DEVICES='')
    failed = False
    for label, executable, modules in checks:
        code = ('import importlib\n' + '\n'.join(f"m=importlib.import_module({name!r}); print({name!r}, getattr(m,'__version__','available'))" for name in modules))
        if label == 'TensorFlow XLA':
            code += '''
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
fn = tf.function(lambda x, y: tf.nn.relu(tf.matmul(x, y) + 1), jit_compile=True, autograph=False)
x = tf.constant([[1., 2.], [3., 4.]])
y = tf.eye(2)
assert fn(x, y).numpy().tolist() == [[2., 3.], [4., 5.]]
assert 'HloModule' in fn.experimental_get_compiler_ir(x, y)(stage='optimized_hlo')
print('XLA CPU compilation, execution and correctness: passed')
'''
        result = subprocess.run([executable, '-c', code], env=env, capture_output=True, text=True, timeout=120)
        failed |= result.returncode != 0
        print(label + ':', result.stdout.strip() if result.returncode == 0 else result.stderr[-1500:])
    print('Allocated physical cores:', physical_cpus())
    print('perf:', shutil.which('perf'))
    print('Checkpoint:', args.checkpoint, args.checkpoint.is_file())
    print('AST dumper:', os.environ.get('AST_DUMPER_BIN_PATH'))
    print('No benchmark workers launched.')
    return 1 if failed else 0


def main(profiling='execution'):
    parser = argparse.ArgumentParser(description=f'{profiling.capitalize()} profiling of fixed sampled CPU benchmarks')
    commands = parser.add_subparsers(dest='command', required=True)
    sample = commands.add_parser('sample', help='Create a fixed subset of the inventory; no benchmarks run')
    sample.add_argument('--inventory', type=Path, default=INVENTORY)
    sample.add_argument('--output', type=Path, default=SAMPLES)
    sample.add_argument('--per-type', type=int, choices=(3, 4), default=4)
    sample.add_argument('--seed', type=int, default=2026)
    sample.add_argument('--max-input-mib', type=float, default=32)
    sample.set_defaults(handler=sample_command)
    doctor = commands.add_parser('doctor', help='Check backend imports, allocated CPUs and tools')
    doctor.add_argument('--python')
    doctor.add_argument('--tensorflow-python')
    doctor.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/v5_model_1200.pt')
    doctor.set_defaults(handler=doctor_command)
    control = commands.add_parser('check-openmp', help='Verify useful parallel work on eight allocated physical cores')
    control.add_argument('--python', help='MLIR/PyTorch Python; defaults to .env CONDA_ENV/bin/python')
    control.add_argument('--output', type=Path, default=HERE / 'results/openmp_control')
    control.set_defaults(handler=check_openmp_command)
    run = commands.add_parser('run', help='Run one logical operation category from the fixed sample CSV',
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run.add_argument('--category', '--operation-category', choices=CATEGORIES,
                     help='Category from sampled CSV; inferred when the selection belongs to one category')
    run.add_argument('--samples', type=Path, default=SAMPLES)
    run.add_argument('--output', type=Path, default=HERE / 'results' / profiling)
    run.add_argument('--benchmark', '--only', dest='only', nargs='+',
                     help='Sampled stem or filename, optionally dataset/name; e.g. resnext50_linalg')
    run.add_argument('--op-type', '--operation-type', nargs='+', help='Contained CSV operation type, e.g. add matmul')
    run.add_argument('--threads', type=int, default=8,
                     help='One CPU budget per campaign; the standard protocol uses eight physical cores')
    run.add_argument('--trials', type=int, default=3)
    if profiling == 'execution':
        run.add_argument('--warmup', type=int, help='Untimed warmup calls after first invocation and correctness')
        run.add_argument('--repeats', type=int, help='Timed calls per worker; their median is the trial latency')
        run.add_argument('--profile-seconds', type=float, help='Minimum wall seconds in the separate CPU/counter window')
        run.add_argument('--profile-repeats', type=int, help='Minimum complete calls in that window; BOTH minima must be met')
        run.add_argument('--no-perf', action='store_true')
    else:
        run.add_argument('--python-profile', action='store_true',
                         help='Also save cProfile Python hotspots; perturbs timing, native compiler internals stay opaque')
    run.add_argument('--correctness-cases', type=int, default=3)
    run.add_argument('--timeout', type=int, default=600, help='Whole worker wall-time cap, including compilation')
    run.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    run.add_argument('--config', type=Path, default=ROOT / 'config/v5/v5_mlir_rl_v1_paper.json')
    run.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/v5_model_1200.pt')
    run.add_argument('--preprocess', choices=('standard', 'none'), default='standard')
    run.add_argument('--max-input-mib', type=float, default=32)
    run.add_argument('--max-source-mib', type=float, default=128)
    run.add_argument('--max-steps', type=int, default=100000)
    run.add_argument('--rtol', type=float, default=1e-4)
    run.add_argument('--atol', type=float, default=1e-5)
    run.add_argument('--seed', type=int, default=2026)
    run.add_argument('--python', help='MLIR/PyTorch Python; defaults to .env CONDA_ENV/bin/python')
    run.add_argument('--tensorflow-python', help='Optional separate TensorFlow environment executable')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--retry-failed', action='store_true')
    run.set_defaults(handler=run_command, profiling=profiling, python_profile=False,
        warmup=5, repeats=5, profile_seconds=1, profile_repeats=3, no_perf=False)
    args = parser.parse_args()
    if args.command == 'run':
        if min([args.threads, args.trials, args.repeats, args.correctness_cases, args.timeout, args.profile_repeats]) < 1:
            parser.error('Thread/trial/repeat/case/timeout/profile counts must be positive')
        if args.warmup < 0 or min(args.profile_seconds, args.rtol, args.atol) < 0:
            parser.error('Warmup, profile seconds and tolerances must be nonnegative')
    return args.handler(args)


if __name__ == '__main__':
    if sys.argv[1:] == ['_worker']:
        worker(json.load(sys.stdin))
    else:
        sys.exit(main())
