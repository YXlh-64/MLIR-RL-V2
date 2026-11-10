#!/usr/bin/env python3
"""Profile optimization and compilation, separately from steady-state execution.

Uses the execution runner's CSV filters, fixtures, worker isolation and resume
logic and its single CPU budget (eight threads by default). Every trial
regenerates the RL schedule and compiles in a fresh worker.
Only one invocation is made: there are no latency repeats or hardware windows.
See README.md for overlapping phases and the limits of Python/native profiling.
"""
from __future__ import annotations

import cProfile
import csv
import io
import json
from pathlib import Path
import pstats
import statistics

from execution_profiling import main, schedule, write_json


def compile_benchmark(config, method, out, progress):
    """Measure real V5 phases, or the corresponding framework setup/first call.

    The MLIR symbol lookup materializes lazy LLVM code without executing main.
    Inductor/XLA have no equivalent public, consistent compile-only boundary
    here, so their first call is explicitly compilation PLUS one execution.
    """
    profiler = cProfile.Profile() if config['python_profile'] else None
    progress.result.update(python_profile_enabled=bool(profiler),
        scope='Fresh worker; compilation and one invocation; no steady-state timing',
        policy_threads=1 if method == 'mlir_rl' else None, threads=config['threads'])
    if profiler:
        profiler.enable()
    try:
        with progress.stage('backend_imports_ns'):
            import numpy as np
            from backends import FrameworkRunner, MLIRRunner, Unsupported
        if method.startswith(('torch_', 'tensorflow_')) and config.get('reference_error'):
            raise Unsupported(config['reference_error'])
        if method == 'mlir_rl':
            # Write schedule artifacts beside this compilation trial. A shared
            # execution schedule would hide inference/application trial costs.
            schedule(config, out, progress)
            config = dict(config, ir_path=str(out / 'scheduled.mlir'))
            # Scheduling uses one Torch thread. Restore the trial budget before
            # lowering/JIT/first execution in this same worker.
            import torch
            torch.set_num_threads(config['threads'])
        runner_type = MLIRRunner if method.startswith('mlir_') else FrameworkRunner
        with progress.stage('backend_setup_total_ns'):
            runner = runner_type(config, method, out, progress)
        with progress.stage('input_binding_ns'):
            with np.load(Path(config['run']) / 'inputs.npz', allow_pickle=False) as archive:
                arrays = [archive[f"case_0_{a['name']}"].copy() for a in config['inputs']]
            runner.reset(arrays)
        if method.startswith('mlir_'):
            with progress.stage('llvm_jit_materialization_ns'):
                runner.engine.lookup('main')
            first_phase = 'first_invoke_ns'
        elif method == 'tensorflow_xla':
            with progress.stage('framework_graph_trace_ns'):
                runner.fn.get_concrete_function(*runner.inputs)
            first_phase = 'lazy_compile_and_first_invoke_ns'
        elif method == 'torch_inductor':
            first_phase = 'lazy_compile_and_first_invoke_ns'
        else:
            first_phase = 'first_invoke_ns'
        with progress.stage(first_phase):
            value = runner.call()
        runner.release(value)
        del value
        # Artifact inspection can itself call compiler inspection APIs. Keep it
        # outside compiler phases and report its cost separately.
        with progress.stage('artifact_collection_ns'):
            artifacts = runner.artifacts()
        progress.result.update(status='compiled', code_artifacts=artifacts,
            correctness='not_checked_in_compilation_profile',
            phase_overlap='backend_setup_total includes parse/bufferize/lower/JIT; policy_rollout includes policy_inference; schedule_application includes per-action phases')
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(out / 'python.prof'))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(40)
            (out / 'python_profile.txt').write_text(stream.getvalue())


def collect_compilation(output, config):
    """One phase CSV, one JSON summary and the familiar schedule index.

    Failed trials retain their completed/failed phases. Aggregate medians use
    successful trials only; status rows preserve failures before the first phase.
    """
    phases, groups, schedules = [], [], []
    for manifest in sorted((output / 'artifacts').glob('*/*/manifest.json')):
        spec = json.loads(manifest.read_text())
        row, run = spec['row'], manifest.parent
        for threads in config['threads']:
            for method in config['methods']:
                trials = []
                for trial in range(1, config['trials'] + 1):
                    folder = run / 'methods' / method / f'threads_{threads}' / f'trial_{trial}'
                    path = folder / 'result.json'
                    if not path.exists():
                        continue
                    result = json.loads(path.read_text())
                    trials.append(result)
                    identity = dict(dataset=row['dataset'], benchmark=row['benchmark'], category=row['category'],
                        backend=method, threads=threads, trial=trial, status=result['status'],
                        last_stage=result.get('last_stage'), reason=result.get('reason'),
                        python_profile_enabled=config['python_profile'], artifact_path=str(folder.relative_to(output)))
                    for phase, wall in result.get('compilation', {}).items():
                        phases.append(dict(identity, phase=phase, wall_ns=wall,
                            phase_status=result.get('phase_status', {}).get(phase),
                            **result.get('phase_resources', {}).get(phase, {})))
                    # Process wall includes Python/imports/audit output; it is
                    # independent of the overlapping compiler-stage intervals.
                    phases.append(dict(identity, phase='process_wall_ns', wall_ns=result.get('process_wall_ns')))
                    decisions = folder / 'schedule.json'
                    if decisions.exists():
                        choices = json.loads(decisions.read_text())
                        applied_path = folder / 'applied_actions.json'
                        applied = json.loads(applied_path.read_text()) if applied_path.exists() else []
                        links = [f'[Decisions]({decisions.relative_to(output)})']
                        for name in ('applied_actions.json', 'fusion_report.json', 'dependency_graph.json', 'scheduled.mlir'):
                            if (folder / name).exists():
                                links.append(f'[{name}]({(folder / name).relative_to(output)})')
                        schedules.append(f"<details>\n<summary>{row['dataset']}/{row['benchmark']} — {threads} threads, trial {trial}</summary>\n\n"
                            + '\n'.join(f"- {d['operation']}: {d['action']}" for d in choices if d['ready'])
                            + '\n\n' + '\n'.join(f"- Applied: {a['operation']}: {a.get('actual', a['planned'])} — {a['status']}" for a in applied)
                            + '\n\n' + ' · '.join(links) + '\n\n</details>\n')
                valid = [r for r in trials if r['status'] == 'compiled']
                names = {key for r in valid for key in r.get('compilation', {})}
                groups.append(dict(dataset=row['dataset'], benchmark=row['benchmark'], backend=method,
                    threads=threads, attempted_trials=len(trials), compiled_trials=len(valid),
                    median_phase_ns={key: statistics.median(r['compilation'][key] for r in valid
                        if key in r['compilation']) for key in sorted(names)},
                    statuses=[r['status'] for r in trials]))
    columns = ['dataset', 'benchmark', 'category', 'backend', 'threads', 'trial', 'status', 'phase',
        'phase_status', 'wall_ns', 'user_cpu_ns', 'system_cpu_ns', 'child_cpu_ns',
        'process_peak_rss_kib_before', 'process_peak_rss_kib_after', 'minor_faults', 'major_faults',
        'voluntary_context_switches', 'involuntary_context_switches', 'python_profile_enabled',
        'last_stage', 'reason', 'artifact_path']
    with (output / 'compilation.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, columns)
        writer.writeheader()
        writer.writerows(phases)
    write_json(output / 'summary.json', dict(settings=config, groups=groups))
    (output / 'SCHEDULES.md').write_text('# Compilation-trial schedules\n\n' + '\n'.join(schedules))
    return groups


if __name__ == '__main__':
    raise SystemExit(main('compilation'))
