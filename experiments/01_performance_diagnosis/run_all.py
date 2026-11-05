#!/usr/bin/env python3
"""Run all sampled Experiment 01 categories and full models sequentially.

Use --dry-run to preview without writing files or running benchmarks.
"""
import argparse
import csv
from pathlib import Path
import shlex
import sys
import traceback

import run_suite


def build_plan(args):
    tasks = []
    common = ['--samples', str(args.samples), '--seed', str(args.seed),
              '--max-input-mib', str(args.max_input_mib), '--max-work', str(args.max_work)]
    if args.rerun:
        common.append('--rerun')
    for dataset in run_suite.DATASETS[:2]:
        selection = argparse.Namespace(**vars(args), dataset=dataset)
        catalog = run_suite.operation_catalog(dataset)
        if not catalog:
            raise ValueError(f'No single-operation categories found in data/{dataset}')
        order = ('add', 'relu', 'matmul', 'batch_matmul', 'conv2d', 'max_pool2d')
        for category in sorted(catalog, key=lambda c: order.index(c) if c in order else len(order)):
            chosen, _ = run_suite.sample_category(catalog[category], selection)
            tasks.append(dict(dataset=dataset, category=category, comparison='four-method',
                benchmarks=[item['name'] for item in chosen],
                argv=['--dataset', dataset, '--category', category] + common,
                status='pending' if chosen else 'blocked',
                details='' if chosen else 'No benchmarks within the sampling limits.'))
    for family, (_, names) in run_suite.paper_extra_samples(args).items():
        tasks.append(dict(dataset='mlir_rl_v1_paper', category=family,
            comparison='unavailable', benchmarks=names, argv=[], status='blocked',
            details='QCD kernels need domain-specific input fixtures and an independent Torch reference.'))
    for path in sorted((run_suite.ROOT / 'data/mlir_rl_v1_paper').glob('model_*.mlir')):
        tasks.append(dict(dataset='mlir_rl_v1_paper', category=path.stem,
            comparison='unavailable', benchmarks=[path.stem], argv=[], status='blocked',
            details='Legacy paper full model has no automatic comparison adapter/reference; use a custom manifest.'))
    models = sorted((run_suite.ROOT / 'data/full_models').glob('*.mlir'))
    if not models:
        raise ValueError('No MLIR files found in data/full_models')
    for path in models:
        comparison = 'four-method' if path.stem in ('gin_linalg', 'gat_linalg') else 'MLIR-only'
        tasks.append(dict(dataset='full_models', category=path.stem, comparison=comparison,
            benchmarks=[path.stem], argv=['--dataset', 'full_models', '--category', path.stem,
                '--allow-mlir-only'] + common, status='pending', details=''))
    for task in tasks:
        timeout = (args.full_model_timeout_seconds if task['dataset'] == 'full_models'
                   else args.timeout_seconds)
        if timeout is not None:
            task['argv'] += ['--timeout-seconds', str(timeout)]
    return tasks


def save_status(tasks):
    run_suite.OUTPUT.mkdir(parents=True, exist_ok=True)
    path = run_suite.OUTPUT / 'RUN_ALL.csv'
    with path.open('w', newline='') as f:
        fields = ('dataset', 'category', 'comparison', 'benchmarks', 'status', 'exit_code', 'report', 'details')
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            row = {key: task.get(key, '') for key in fields}
            row['benchmarks'] = '; '.join(task['benchmarks'])
            row['report'] = f"{task['dataset']}/{task['category']}/REPORT.md" if task['argv'] else ''
            writer.writerow(row)


def classify(task, output):
    """Check saved outcomes too: run_suite can skip previously failed attempts."""
    path = output / 'comparison.csv'
    if not path.is_file():
        return 'incomplete', 'No comparison.csv was produced; inspect terminal diagnostics.'
    with path.open(newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['workload'] in task['benchmarks']]
    methods = (run_suite.experiment.METHODS if task['comparison'] == 'four-method'
               else ('mlir_baseline', 'mlir_rl'))
    expected = {(name, method) for name in task['benchmarks'] for method in methods}
    if not expected.issubset({(r['workload'], r['method']) for r in rows}):
        return 'incomplete', 'Some requested methods are absent; use --rerun to retry completed partial comparisons.'
    if any(r['status'] != 'measured' or r['correctness'] == 'failed' for r in rows):
        return 'failed', 'One or more saved methods failed, timed out, were skipped or failed correctness. See the category REPORT.md.'
    if any(r['correctness'] != 'passed' for r in rows):
        return 'unverified', 'Measurements exist without independent cross-backend validation (MLIR-only diagnostic).'
    return 'passed', 'All requested methods passed cross-backend correctness checks.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-input-mib', type=float, default=32)
    parser.add_argument('--max-work', type=int, default=100_000_000)
    parser.add_argument('--timeout-seconds', type=int, help='Simple-operation per-method limit (default: 1200)')
    parser.add_argument('--full-model-timeout-seconds', type=int, help='Full-model per-method limit (default: 600)')
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args(argv)
    if args.samples < 1 or args.max_input_mib <= 0 or args.max_work < 1 or any(
            value is not None and value <= 0 for value in (args.timeout_seconds, args.full_model_timeout_seconds)):
        parser.error('Samples, resource limits and timeouts must be positive')
    try:
        tasks = build_plan(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task['dataset']}/{task['category']} ({task['comparison']})")
            print('  Benchmarks: ' + ', '.join(task['benchmarks']))
            if task['status'] == 'blocked':
                print('  BLOCKED: ' + task['details'])
            else:
                print('  ' + shlex.join([sys.executable, '-u', str(Path(run_suite.__file__)), *task['argv']]))
        print('\nPreview only. Full-model weights/state are audited during execution; missing values are refused.')
        print('No files written and no benchmarks started.')
        return 0
    save_status(tasks)
    print(f'Status: {run_suite.OUTPUT / "RUN_ALL.csv"}', flush=True)
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['dataset']}/{task['category']} ({task['comparison']})", flush=True)
        if task['status'] == 'blocked':
            print('BLOCKED: ' + task['details'], flush=True)
            continue
        task['status'] = 'running'
        save_status(tasks)
        try:
            try:
                code = run_suite.main(task['argv']) or 0
            except SystemExit as error:
                code = error.code if isinstance(error.code, int) else 1
            task['exit_code'] = code
            if code == 130:
                raise KeyboardInterrupt
            if code == 2:
                task.update(status='blocked', details='Preflight refused this comparison; see terminal error (e.g. missing weights/state).')
            else:
                output = run_suite.OUTPUT / task['dataset'] / task['category']
                run_suite.collect_report(output)
                status, detail = classify(task, output)
                if code and status == 'passed':
                    status, detail = 'failed', f'Category driver returned exit code {code}; inspect terminal diagnostics.'
                task.update(status=status, details=detail)
        except KeyboardInterrupt:
            task.update(status='interrupted', exit_code=130, details='Interrupted; remaining pending entries were not started.')
            save_status(tasks)
            print('\nStopped. No later categories will start.', flush=True)
            return 130
        except Exception as error:
            traceback.print_exc()
            task.update(status='failed', exit_code=1, details=str(error))
        save_status(tasks)
        print(task['status'].upper() + ': ' + task['details'], flush=True)
    print(f'\nFinished. Read {run_suite.OUTPUT / "RUN_ALL.csv"}', flush=True)
    return int(any(task['status'] != 'passed' for task in tasks))


if __name__ == '__main__':
    raise SystemExit(main())
