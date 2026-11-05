#!/usr/bin/env python3
"""Select and run one dataset/category in the foreground; --list runs nothing."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re

import experiment

ROOT = experiment.REPO
OUTPUT = experiment.HERE / 'results'
DATASETS = ('mlir_rl_v1_paper', 'ops_and_blocks', 'full_models')
KINDS = {'add': 'add', 'matmul': 'matmul', 'batch_matmul': 'batch_matmul',
         'conv_2d_nchw_fchw': 'conv2d', 'pooling_nchw_max': 'max_pool2d'}
DTYPES = {'f32': 'float32', 'f64': 'float64', 'i32': 'int32', 'i64': 'int64'}


def node(name, op, *args, **kwargs):
    return dict(id=name, op=op, args=list(args), kwargs=kwargs)


def tensor_inputs(text):
    entry = re.search(r'func\.func\s+@\w+\(([^)]*)\)\s*->', text)
    if not entry:
        raise ValueError('Public tensor entry not found')
    inputs = []
    for i, arg in enumerate(entry[1].split(',')):
        match = re.fullmatch(r'\s*%\w+:\s*tensor<([0-9x]*)(f32|f64|i32|i64)>\s*', arg)
        if not match:
            raise ValueError(f'Unsupported input ABI: {arg.strip()}')
        shape = [int(d) for d in match[1].rstrip('x').split('x') if d]
        spec = dict(name=f'a{i}', shape=shape, dtype=DTYPES[match[2]],
                    generator='normal', std=0.1)
        if match[2].startswith('i'):
            spec.update(generator='ones' if i == 1 else 'integers', low=0, high=16)
        inputs.append(spec)
    return inputs


def single_operation(path):
    """Recognize only the dataset's five/six simple single-operation forms."""
    text = path.read_text()
    operations = [op for op in re.findall(r'\blinalg\.(\w+)\b', text)
                  if op not in ('yield', 'fill', 'index')]
    if len(operations) != 1:
        return None
    op = operations[0]
    kind = KINDS.get(op)
    if op == 'generic' and path.stem.startswith('relu_'):
        if 'arith.cmpf ugt' in text and 'arith.select' in text:
            kind = 'relu'
    if not kind:
        return None
    inputs = tensor_inputs(text)
    shapes = [s['shape'] for s in inputs]
    if kind == 'relu':
        if len(inputs) != 2 or shapes[0] != shapes[1]:
            raise ValueError(f'Unexpected ReLU signature: {path}')
        graph = [node('out', 'relu', 'a0')]
    else:
        # These dataset forms consume arguments in signature order; reject drift.
        operation = text[text.index('linalg.' + op):]
        ins = re.search(r'ins\(([^:]+):', operation)
        outs = re.search(r'outs\(([^:]+):', operation)
        expected = ['%arg0', '%arg1', '%arg2']
        actual = re.findall(r'%\w+', ins[1] + ',' + outs[1]) if ins and outs else []
        if actual != expected or len(inputs) != 3:
            raise ValueError(f'Unexpected operands in {path}')
        if kind == 'add':
            graph = [node('out', 'add', 'a0', 'a1')]
        elif kind in ('matmul', 'batch_matmul'):
            graph = [node('product', 'matmul', 'a0', 'a1'),
                     node('out', 'add', 'product', 'a2')]
        else:
            def pair(attribute):
                match = re.search(attribute + r'\s*=\s*dense<([^>]+)>', operation)
                if not match:
                    raise ValueError(f'Missing {attribute}: {path}')
                values = [int(v) for v in re.findall(r'\d+', match[1])]
                return values * 2 if len(values) == 1 else values
            kwargs = dict(stride=pair('strides'), dilation=pair('dilations'))
            if kind == 'conv2d':
                graph = [node('convolution', 'conv2d', 'a0', 'a1', **kwargs),
                         node('out', 'add', 'convolution', 'a2')]
            else:
                graph = [node('pool', 'max_pool2d', 'a0', kernel_size=shapes[1], **kwargs),
                         node('out', 'maximum', 'pool', 'a2')]
    work = sum(math.prod(shape) for shape in shapes)
    if kind in ('matmul', 'batch_matmul'):
        work = 2 * math.prod(shapes[0]) * shapes[1][-1]
    elif kind == 'conv2d':
        work = 2 * math.prod(shapes[2]) * math.prod(shapes[1][1:])
    elif kind == 'max_pool2d':
        work = math.prod(shapes[2]) * math.prod(shapes[1])
    input_bytes = sum(math.prod(s['shape']) * (8 if s['dtype'].endswith('64') else 4)
                      for s in inputs)
    return dict(name=path.stem, source=str(path), category=kind, inputs=inputs,
                graph=graph, work=work, input_bytes=input_bytes)


def operation_catalog(dataset):
    groups = {}
    for path in sorted((ROOT / 'data' / dataset).glob('*.mlir')):
        # Full models and extracted blocks are not simple single-operation kernels.
        if path.stem.startswith('model_') or '_block_' in path.stem:
            continue
        item = single_operation(path)
        if item:
            groups.setdefault(item['category'], []).append(item)
    return groups


def sample_category(items, args):
    eligible = [item for item in items
                if item['input_bytes'] <= args.max_input_mib * 1024**2
                and item['work'] <= args.max_work]
    # Avoid spending the small sample on identical shapes from different models.
    unique = {}
    for item in eligible:
        key = json.dumps([item['inputs'], item['graph']], sort_keys=True)
        unique.setdefault(key, item)
    candidates = list(unique.values())
    rng = random.Random(f'{args.seed}:{args.dataset}:{items[0]["category"]}')
    chosen = rng.sample(candidates, min(args.samples, len(candidates)))
    return chosen, len(candidates)


def manifest(item, args):
    return dict(schema_version=1, name=item['name'], mlir=item['source'],
        dataset=args.dataset, category=args.category, seed=args.seed,
        sampling=dict(samples=args.samples, max_input_mib=args.max_input_mib,
                      max_work=args.max_work) if args.dataset != 'full_models' else None,
        threads=1, cpu_affinity=[min(os.sched_getaffinity(0))],
        warmup=10, repeats=100, correctness_cases=3,
        rtol=1e-4, atol=1e-5, timeout_seconds=args.timeout_seconds or 1200,
        profile_repeats=100, profile_min_seconds=0.25, perf=False,
        pipeline_implementation='rl_autoschedular_v5', inputs=item['inputs'],
        torch=dict(graph=item['graph'], outputs=['out']) if item.get('graph') else None,
        rl=dict(checkpoint=str(ROOT / 'checkpoints/v5_model_1200.pt'),
                config=str(ROOT / 'config/v5/v5_mlir_rl_v1_paper.json'),
                implementation='rl_autoschedular_v5', max_steps=100000),
        manual=[])


def audit_full_models(paths=None):
    """Check embedded resources and input ABI without materializing large models."""
    inventory = []
    for path in paths if paths is not None else sorted((ROOT / 'data/full_models').glob('*.mlir')):
        elided, previous = False, b''
        with path.open('rb') as f:
            header = f.read(1024 * 1024)
            part = header
            while part:
                elided |= b'__elided__' in previous + part
                previous = part[-32:]
                part = f.read(8 * 1024 * 1024)
        inputs, reason = [], None
        if elided:
            reason = 'Missing/elided embedded weights.'
        else:
            try:
                inputs = tensor_inputs(header.decode('utf-8'))
            except ValueError as error:
                reason = str(error)
            if path.stem == 'resnext50_linalg' and len(inputs) > 1:
                reason = 'Missing saved external BatchNorm running state/counters.'
        inventory.append(dict(name=path.stem, source=str(path), inputs=inputs,
                              bytes=path.stat().st_size, exclusion=reason))
    return inventory


def gnn_reference(c, target):
    """Recover actual GIN/GAT weights; graphs were checked against their MLIR."""
    import numpy as np
    text = Path(c['mlir']).read_text()
    resources = dict(re.findall(r'([\w.]+): "0x([0-9A-Fa-f]+)"', text))
    weights = {}
    for name, resource, dims in re.findall(
            r'%(\w+) = arith.constant dense_resource<([^>]+)> : tensor<([0-9x]+)xf32>', text):
        raw = bytes.fromhex(resources[resource])
        shape = [int(d) for d in dims.split('x')]
        if int.from_bytes(raw[:4], 'little') != 4 or len(raw) != 4 + 4 * np.prod(shape):
            raise ValueError(f'Unexpected resource encoding for {resource}')
        weights[name] = np.frombuffer(raw, dtype='<f4', offset=4).reshape(shape).copy()
    np.savez(target, **weights)
    graph = []

    def emit(name, op, *args, **kwargs):
        graph.append(node(name, op, *args, **kwargs))
        return name

    def linear(value, weight, bias, name):
        w = emit(name + '_wt', 'transpose', weight, dim0=0, dim1=1)
        mm = emit(name + '_mm', 'matmul', value, w)
        return emit(name, 'add', mm, bias) if bias else mm

    if c['name'] == 'gin_linalg':
        x = 'a0'
        for layer in range(2):
            neigh = emit(f'neigh{layer}', 'matmul', 'a1', x)
            agg = emit(f'agg{layer}', 'add', x, neigh)
            base = 1 + 4 * layer
            h = linear(agg, f'cst_{base}', f'cst_{base+1}', f'l{layer}_1')
            h = emit(f'relu{layer}', 'relu', h)
            x = linear(h, f'cst_{base+2}', f'cst_{base+3}', f'l{layer}_2')
        h = linear(x, 'cst_9', 'cst_10', 'l2_1')
        h = emit('relu2', 'relu', h)
        linear(h, 'cst_11', 'cst_12', 'out')
    else:
        h = linear('a0', 'cst', None, 'first')
        h = linear(h, 'cst_4', None, 'second')
        emit('src0', 'reshape', h, shape=[128, 1, 4, 32])
        emit('dst0', 'reshape', h, shape=[1, 128, 4, 32])
        emit('src', 'expand', 'src0', size=[128, 128, 4, 32])
        emit('dst', 'expand', 'dst0', size=[128, 128, 4, 32])
        emit('pairs', 'cat', 'src', 'dst', dim=3)
        scores = linear('pairs', 'cst_5', None, 'scores')
        emit('scores3', 'reshape', scores, shape=[128, 128, 4])
        emit('leaky', 'leaky_relu', 'scores3', negative_slope=0.01)
        emit('alpha', 'softmax', 'leaky', dim=1)
        emit('alpha4', 'reshape', 'alpha', shape=[128, 128, 4, 1])
        emit('weighted', 'mul', 'alpha4', 'dst0')
        emit('aggregated', 'sum', 'weighted', dim=1)
        emit('flat', 'reshape', 'aggregated', shape=[128, 128])
        emit('activation', 'elu', 'flat')
        linear('activation', 'cst_6', 'cst_7', 'out')
    c['torch'] = dict(graph=graph, outputs=['out'], constants=str(target))


def collect_report(output):
    latest = {}
    for path in sorted((output / 'artifacts').glob('*/*/summary.json')):
        summary = json.loads(path.read_text())
        latest[summary['workload']] = path
    if not latest:
        print(f'No completed results in {output}')
        return
    rows, schedules, table = [], [], []
    for name, path in sorted(latest.items()):
        summary = json.loads(path.read_text())
        for method, result in summary['results'].items():
            latency = result.get('latency', {}).get('median_ns')
            rows.append(dict(workload=name, method=method, status=result['status'],
                correctness=result.get('correctness', {}).get('status'),
                median_us=latency / 1000 if latency else '',
                validated_speedup=result.get('validated_speedup_vs_mlir_baseline'),
                observed_ratio=result.get('observed_latency_ratio_vs_mlir_baseline'),
                reason=result.get('reason', ''),
                artifacts=str(path.parent.relative_to(output))))
        cells = []
        for method in experiment.METHODS:
            result = summary['results'].get(method, {})
            latency = result.get('latency', {}).get('median_ns')
            cells.append(f'{latency / 1000:,.3f}' if latency else result.get('status', 'unavailable'))
        rl = summary['results'].get('mlir_rl', {})
        speedup = rl.get('validated_speedup_vs_mlir_baseline')
        link = path.relative_to(output)
        table.append(f'| [{name}]({link}) | ' + ' | '.join(cells)
                     + f' | {f"{speedup:.3f}×" if speedup else "—"} |')
        schedule = path.parent / 'methods/mlir_rl/schedule.json'
        if schedule.exists():
            actions = json.loads(schedule.read_text())
            scheduled = schedule.with_name('scheduled.mlir')
            applied = (f'[Applied MLIR]({scheduled.relative_to(output)})'
                       if scheduled.exists() else 'Schedule application did not complete.')
            schedules.append(f'<details>\n<summary>{name}</summary>\n\n'
                + '\n'.join(f"- {a['operation']}: {a['action']}" for a in actions if a['ready'])
                + f'\n\n[Raw decisions]({schedule.relative_to(output)}) · {applied}\n\n</details>\n')
    with (output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'SCHEDULES.md').write_text('# Generated V5 schedules\n\n'
        'Complete parameter choices are listed below. Raw JSON also includes pointer '
        'decisions. A planned schedule can fail during application; inspect REPORT.md.\n\n'
        + '\n'.join(schedules))
    report = ['# Experiment 01 comparison', '',
        f"Recorded {len(latest)} workloads: {sum(r['status'] == 'measured' for r in rows)} measured methods, "
        f"{sum(r['correctness'] == 'passed' for r in rows)} validated, "
        f"{sum(r['status'] == 'error' for r in rows)} errors, "
        f"{sum(r['status'] == 'timeout' for r in rows)} timeouts.", '',
        '[Measurements](comparison.csv) · [Schedules](SCHEDULES.md)', '',
        'Median CPU host-call times in microseconds. One thread; shared saved inputs. '
        'Baseline uses the V5 lowering pipeline and LLVM optimization level 3. '
        'Validated speedup = baseline median / candidate median, requiring agreement '
        'with independent eager Torch on every saved correctness case.', '',
        '| Workload | MLIR baseline µs | V5 RL µs | Eager µs | Inductor µs | Validated RL speedup |',
        '| --- | ---: | ---: | ---: | ---: | ---: |', *table, '',
        '## Failures and interpretation', '',
        'Each workload link opens its detailed results. Measured methods with failed '
        'correctness checks are not valid speedups. MLIR-only runs are unverified. '
        'A missing latency is never zero. Schedule application is strict: a failed '
        'transform aborts that method. Standard V5 evaluation can skip failed '
        'transformations, so its failure behavior differs.', '',
        'Compilation/setup is outside latency timing. Input restoration, output '
        'copying and disposal are also outside it. These are single-process trials; '
        'no confidence intervals or repeated independent trials were collected.', '']
    for row in rows:
        if row['status'] != 'measured' or row['correctness'] != 'passed':
            detail = Path(row['artifacts']) / 'methods' / row['method'] / 'result.json'
            report.append(f"- {row['workload']} / {row['method']}: "
                          f"{row['status']}, correctness {row['correctness']}. [Details]({detail})")
    (output / 'REPORT.md').write_text('\n'.join(report))


def paper_extra_samples(args):
    groups = {}
    for path in sorted((ROOT / 'data/mlir_rl_v1_paper').glob('*.mlir')):
        if path.stem.startswith(('baryon_', 'dibaryon_', 'hexaquark_')):
            family = re.split(r'_\d', path.stem, maxsplit=1)[0]
            groups.setdefault(family, []).append(path.stem)
    samples = {}
    for family, names in sorted(groups.items()):
        rng = random.Random(f'{args.seed}:{family}')
        chosen = sorted(rng.sample(names, min(args.samples, len(names))))
        samples[family] = (len(names), chosen)
    return samples


def print_paper_extras(args):
    print('\nQCD families (multi-operation; domain inputs and independent Torch reference needed):')
    for family, (count, chosen) in paper_extra_samples(args).items():
        print(f'  {family}: {count} files; sample: {", ".join(chosen)}')
    print('  model_*: 3 full-model files; excluded from single-operation sampling.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--category', help='One operation category, or one full-model stem')
    parser.add_argument('--list', action='store_true', help='Show categories and sampled names; writes nothing')
    parser.add_argument('--dry-run', action='store_true', help='Preview one category; writes nothing')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-input-mib', type=float, default=32)
    parser.add_argument('--max-work', type=int, default=100_000_000,
                        help='Sampling limit: estimated arithmetic/reduction work')
    parser.add_argument('--timeout-seconds', type=int, help='Per-method cap, including setup/compilation')
    parser.add_argument('--only', nargs='+', help='Explicit benchmark stems within the selected operation category')
    parser.add_argument('--methods', nargs='+', choices=experiment.METHODS)
    parser.add_argument('--allow-mlir-only', action='store_true',
                        help='Allow a full model without an equivalent Torch graph')
    parser.add_argument('--audit-only', action='store_true', help='Read full-model weights/ABI; writes nothing')
    parser.add_argument('--rerun', action='store_true', help='Preserve a new attempt instead of skipping completed workloads')
    parser.add_argument('--report-only', action='store_true', help='Summarize saved results; runs nothing')
    args = parser.parse_args(argv)
    if args.samples < 1 or args.max_input_mib <= 0 or args.max_work < 1 or (
            args.timeout_seconds is not None and args.timeout_seconds <= 0):
        parser.error('Samples, resource limits and timeout must be positive')
    if args.audit_only:
        for item in audit_full_models():
            reference = 'Torch reference available' if item['name'] in ('gin_linalg', 'gat_linalg') else 'Torch reference missing'
            print(item['name'], item['exclusion'] or 'Embedded weights present; ' + reference)
        return 0
    if not args.dataset:
        parser.print_help()
        return 0
    if args.dataset == 'full_models':
        if args.list:
            for path in sorted((ROOT / 'data/full_models').glob('*.mlir')):
                print(path.stem)
            print('Use --audit-only to inspect weights and parameter state.')
            return 0
        if not args.category:
            parser.error('Choose one full-model stem with --category')
        path = ROOT / 'data/full_models' / (args.category + '.mlir')
        if path.parent != ROOT / 'data/full_models' or not path.is_file():
            parser.error('Unknown full-model stem')
        selected = []
        if not args.report_only:
            item = audit_full_models([path])[0]
            if item['exclusion']:
                parser.error(item['exclusion'])
            if item['name'] not in ('gin_linalg', 'gat_linalg'):
                requested_torch = args.methods and any(m.startswith('torch_') for m in args.methods)
                if not args.allow_mlir_only or requested_torch:
                    parser.error('Equivalent Torch graph missing. Supply a custom manifest, or explicitly use --allow-mlir-only.')
            selected = [item]
    else:
        catalog = operation_catalog(args.dataset)
        if args.list:
            for category, items in sorted(catalog.items()):
                chosen, count = sample_category(items, args)
                print(f'{category}: {len(items)} files; {count} distinct eligible signatures; sample:')
                for item in chosen:
                    print(' ', item['name'])
            if args.dataset == 'mlir_rl_v1_paper':
                print_paper_extras(args)
            else:
                print('\nMulti-operation *_block_* files are outside this single-operation selection.')
            return 0
        if args.category not in catalog:
            parser.error('Choose one category: ' + ', '.join(sorted(catalog)))
        items = catalog[args.category]
        selected, _ = sample_category(items, args)
        if args.only:
            selected = [item for item in items if item['name'] in args.only]
            if set(args.only) != {item['name'] for item in selected}:
                parser.error('--only contains names outside the selected category')
            if any(item['input_bytes'] > args.max_input_mib * 1024**2
                   or item['work'] > args.max_work for item in selected):
                parser.error('Requested benchmark exceeds sampling limits; adjust --max-input-mib / --max-work explicitly')
    output = OUTPUT / args.dataset / args.category
    if args.report_only:
        collect_report(output)
        return 0
    if not selected:
        parser.error('No benchmarks within resource limits')
    print(f'Category: {args.dataset}/{args.category}; {len(selected)} workload(s)', flush=True)
    for item in selected:
        print(' ', item['name'], flush=True)
    if args.dry_run:
        return 0
    failed = 0
    try:
        for item in selected:
            destination = output / 'artifacts' / item['name']
            if not args.rerun and any(destination.glob('*/summary.json')):
                print(f"Already finished (may include failures): {item['name']}; use --rerun to retry.", flush=True)
                continue
            c = manifest(item, args)
            if not c['torch']:
                del c['torch']
            if args.dataset == 'full_models':
                c.update(warmup=2, repeats=20, correctness_cases=2,
                         profile_repeats=5, timeout_seconds=args.timeout_seconds or 600)
                if item['name'] in ('gin_linalg', 'gat_linalg'):
                    output.mkdir(parents=True, exist_ok=True)
                    gnn_reference(c, output / (item['name'] + '_weights.npz'))
            methods = args.methods or (experiment.METHODS if c.get('torch') else ('mlir_baseline', 'mlir_rl'))
            failed |= experiment.run_experiment(c, methods, destination)
            collect_report(output)
    except KeyboardInterrupt:
        print('\nInterrupted. The active worker group was stopped; no further workloads will start.', flush=True)
        return 130
    return failed


if __name__ == '__main__':
    raise SystemExit(main())
