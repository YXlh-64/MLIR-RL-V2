#!/usr/bin/env python3
"""Pre-processing pass: translate every block to Torch, run it, check it, time it.

For each block in the catalog (results/catalog/blocks_catalog.json):
  1. translate with block_to_torch.py           -> translations/<name>.torch.json + <name>.py
  2. build the Torch graph exactly like the harness runner does and run it eagerly
     (1 thread; first call + median of a few calls -- indicative only, the real
     benchmark is run_blocks.py)
  3. execute the generated <name>.py and check it gives the same tensors as the graph
  4. run the independent float64 NumPy interpreter (reference_eval.py) on the same
     inputs and compare with the harness tolerance (rtol 1e-4, atol 1e-5)

Blocks with the same catalog signature are the same computation on the same shapes,
so by default step 4 runs once per signature and duplicates inherit the verdict
(`verified_via`). --verify-all checks every block.

Inputs are generated like experiment.py's prepare_inputs (seed 2026, case 0).

Writes results/preprocess/:
  preprocess.json   environment, settings, totals, per-stage times, per-block records
  preprocess.csv    one row per block
  translations/     <name>.torch.json (+ <name>.py when translated)
  failures.json     every block that is unsupported / errored / mismatched

Usage (from the repo root or this folder):
  python experiments/02_block_diagnosis/verify_blocks.py --jobs 4
  python experiments/02_block_diagnosis/verify_blocks.py --limit 50 --verify-all
"""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import block_to_torch as b2t   # noqa: E402
import reference_eval as ref   # noqa: E402

RTOL, ATOL = 1e-4, 1e-5        # experiment.py compare_outputs defaults
SEED = 2026
SETTINGS = {}


def make_inputs(specs):
    import numpy as np
    rng = np.random.default_rng(SEED)
    arrays = {}
    for s in specs:
        g = s.get('generator', 'normal')
        if g == 'normal':
            a = rng.normal(s.get('mean', 0), s.get('std', 1), s['shape'])
        elif g == 'uniform':
            a = rng.uniform(s.get('low', -1), s.get('high', 1), s['shape'])
        elif g == 'integers':
            a = rng.integers(s.get('low', -3), s.get('high', 4), s['shape'])
        else:
            a = getattr(np, g)(s['shape'])
        arrays[s['name']] = np.array(np.asarray(a, dtype=s['dtype']), order='C', copy=True)
    return arrays


def build_graph(t, torch, ops):
    """Same evaluation loop as the harness TorchRunner (block_experiment.BlockTorchRunner)."""
    names = [s['name'] for s in t['inputs']]

    def graph(*args):
        values = dict(zip(names, args))
        for node in t['graph']:
            kwargs = dict(node.get('kwargs', {}))
            if isinstance(kwargs.get('dtype'), str):
                kwargs['dtype'] = getattr(torch, kwargs['dtype'])
            values[node['id']] = ops[node['op']](*[values[n] for n in node['args']], **kwargs)
        return tuple(values[n] for n in t['outputs'])
    return graph


def ms(t0):
    return round((time.perf_counter() - t0) * 1e3, 4)


def compare(got, want):
    import numpy as np
    if len(got) != len(want):
        return dict(passed=False, reason=f'{len(got)} outputs vs {len(want)} in reference')
    details = []
    for k, (g, w) in enumerate(zip(got, want)):
        if tuple(g.shape) != tuple(w.shape):
            details.append(dict(output=k, passed=False, reason=f'shape {list(g.shape)} vs {list(w.shape)}'))
            continue
        g64 = g.astype(np.float64)
        finite = bool(np.isfinite(g64).all() and np.isfinite(w).all())
        err = np.abs(g64 - w)
        details.append(dict(
            output=k, shape=list(g.shape), finite=finite,
            passed=bool(finite and np.allclose(g64, w, rtol=RTOL, atol=ATOL)),
            # float32-scale criterion: error relative to the output's magnitude. Long f32
            # reductions (e.g. K=25088) leave ~1e-6*|max| absolute error on near-zero
            # elements, which elementwise rtol/atol flags although the program is right.
            scaled_error=float(err.max() / (np.abs(w).max() + ATOL)) if err.size else 0.0,
            max_abs_error=float(err.max()) if err.size else 0.0,
            max_rel_error=float((err / (np.abs(w) + ATOL)).max()) if err.size else 0.0,
            ref_max_abs=float(np.abs(w).max()) if w.size else 0.0,
            nonfinite_ref=int((~np.isfinite(w)).sum()), nonfinite_torch=int((~np.isfinite(g64)).sum())))
    return dict(passed=all(d['passed'] for d in details), outputs=details,
                passed_scaled=all(d.get('finite') and d.get('scaled_error', 1) <= RTOL for d in details),
                max_abs_error=max((d.get('max_abs_error', 0) for d in details), default=None),
                max_rel_error=max((d.get('max_rel_error', 0) for d in details), default=None))


def process(job):
    name, path, verify = job
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    out = Path(SETTINGS['out']) / 'translations'
    rec = dict(name=name)
    try:
        text = Path(path).read_text()
        t0 = time.perf_counter()
        t = b2t.translate_file(path)
        rec['translate_ms'] = ms(t0)
        rec['status'] = t['status']
        if t['status'] != 'ok':
            rec['reason'] = t['reason']
            (out / f'{name}.torch.json').write_text(json.dumps(t, indent=1))
            return rec
        t0 = time.perf_counter()
        source = b2t.render_python(name, t)
        rec['render_ms'] = ms(t0)
        (out / f'{name}.torch.json').write_text(json.dumps(t, indent=1))
        (out / f'{name}.py').write_text(source)
        rec.update(n_nodes=t['n_nodes'], unused_inputs=t['unused_inputs'],
                   input_domains=sorted({s.get('domain', s['generator']) for s in t['inputs']}),
                   py_standalone="OPS['" not in source)

        arrays = make_inputs(t['inputs'])
        ops = b2t.torch_operations(torch, F)
        graph = build_graph(t, torch, ops)
        args = [torch.from_numpy(arrays[s['name']].copy()) for s in t['inputs']]
        with torch.no_grad():
            t0 = time.perf_counter()
            got = graph(*args)
            rec['torch_first_call_ms'] = ms(t0)
            calls = []
            for _ in range(SETTINGS['torch_calls']):
                t0 = time.perf_counter()
                graph(*args)
                calls.append(ms(t0))
        rec['torch_eager_median_ms'] = statistics.median(calls) if calls else None
        got = [g.contiguous().numpy() for g in got]
        rec['output_shapes'] = [list(g.shape) for g in got]

        ns = {'OPS': ops}
        exec(compile(source, f'{name}.py', 'exec'), ns)
        with torch.no_grad():
            py_out = ns['Block']().forward(*[torch.from_numpy(arrays[s['name']].copy()) for s in t['inputs']])
        rec['py_matches_graph'] = bool(len(py_out) == len(got) and all(
            np.array_equal(p.numpy(), g, equal_nan=True) for p, g in zip(py_out, got)))

        if not verify:
            rec['reference'] = dict(status='inherited')
            return rec
        t0 = time.perf_counter()
        try:
            want = ref.evaluate(text, arrays, max_volume=SETTINGS['max_volume'])
            rec['reference'] = dict(status='ok', ms=ms(t0), volume=int(ref.evaluate.last_volume))
        except ref.TooLarge as e:
            rec['reference'] = dict(status='skipped_too_large', reason=str(e))
            return rec
        except ref.RefUnsupported as e:
            rec['reference'] = dict(status='unsupported', reason=str(e))
            return rec
        rec['correctness'] = compare(got, want)
    except Exception as e:
        rec.setdefault('status', 'error')
        rec['crash'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc(limit=4)
    return rec


def timing(records):
    """count / total / median / p95 / max per stage (ms), over blocks that ran the stage."""
    stages = {'translate': lambda r: r.get('translate_ms'),
              'render_python': lambda r: r.get('render_ms'),
              'torch_eager_first_call': lambda r: r.get('torch_first_call_ms'),
              'torch_eager_median_call': lambda r: r.get('torch_eager_median_ms'),
              'reference_numpy': lambda r: (r.get('reference') or {}).get('ms')}
    out = {}
    for stage, get in stages.items():
        v = sorted(x for x in map(get, records) if isinstance(x, (int, float)))
        if v:
            out[stage] = dict(count=len(v), total_s=round(sum(v) / 1e3, 3), median=round(statistics.median(v), 4),
                              p95=round(v[min(len(v) - 1, int(0.95 * len(v)))], 4), max=round(v[-1], 4))
    return out


def verdict(r):
    """One word per block for the summary."""
    if r.get('status') != 'ok':
        return r.get('status', 'error')        # unsupported / error / lost
    if r.get('crash'):
        return 'crash'
    c = r.get('correctness')
    if c is None:
        return 'reference_' + r.get('reference', {}).get('status', 'missing')
    if c['passed']:
        return 'correct'
    if c.get('passed_scaled'):
        return 'correct_fp32_tolerance'     # fails strict rtol/atol only on near-zero elements
    if any(not o.get('finite', True) for o in c.get('outputs', [])):
        return 'mismatch_nonfinite'
    return 'mismatch'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--catalog', type=Path, default=HERE / 'results' / 'catalog' / 'blocks_catalog.json')
    parser.add_argument('--out', type=Path, default=HERE / 'results' / 'preprocess')
    parser.add_argument('--jobs', type=int, default=max(1, min(3, os.cpu_count() - 1)),
                        help='Worker processes (each can use a few GB on the largest blocks)')
    parser.add_argument('--limit', type=int, help='Only the first N blocks (quick test)')
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--verify-all', action='store_true', help='Reference-check every block, not one per signature')
    parser.add_argument('--max-volume', type=float, default=2e8,
                        help='Skip the NumPy reference above this many loop iterations per block')
    parser.add_argument('--max-parallel', type=float, default=ref.MAX_PARALLEL,
                        help='Memory guard of the reference: max elements of one op\'s parallel space')
    parser.add_argument('--torch-calls', type=int, default=5, help='Timed eager calls after the first one')
    parser.add_argument('--stall-timeout', type=float, default=900,
                        help='Seconds without any finished block before giving up on the rest')
    parser.add_argument('--fresh', action='store_true', help='Ignore records.jsonl from a previous run')
    parser.add_argument('--rerun', nargs='*', metavar='VERDICT',
                        help='Re-run blocks with these verdicts, e.g. mismatch reference_unsupported')
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    blocks = catalog['blocks']
    if args.models:
        blocks = [b for b in blocks if b.get('model') in set(args.models)]
    if args.limit:
        blocks = blocks[:args.limit]
    meta = {b['name']: b for b in blocks}
    rep = {}
    jobs = []
    for b in blocks:
        sig = b.get('signature')
        first = sig not in rep
        rep.setdefault(sig, b['name'])
        jobs.append((b['name'], b['source'], args.verify_all or first or sig is None))

    (args.out / 'translations').mkdir(parents=True, exist_ok=True)
    ref.MAX_PARALLEL = args.max_parallel           # inherited by the forked workers
    SETTINGS.update(out=str(args.out), max_volume=args.max_volume, torch_calls=args.torch_calls)
    print(f'{len(jobs)} blocks, {sum(j[2] for j in jobs)} reference checks, {args.jobs} processes',
          flush=True)

    import torch
    from multiprocessing import TimeoutError as PoolTimeout
    started = datetime.now(timezone.utc)
    t_wall = time.perf_counter()
    # Resumable: every finished block is appended to records.jsonl; a rerun skips them.
    done_path = args.out / 'records.jsonl'
    if args.fresh:
        for f in (done_path, args.out / 'invocations.jsonl'):
            if f.exists():
                f.unlink()
    records = []
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            r = json.loads(line)
            if r.get('verdict_pending') != 'lost':
                records.append(r)
    if args.rerun:                                   # drop these verdicts so they run again
        records = [r for r in records if verdict(r) not in set(args.rerun)]
        done_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    done = {r['name'] for r in records}
    todo = [j for j in jobs if j[0] not in done]
    print(f'{len(done)} already done (records.jsonl), {len(todo)} to run', flush=True)
    lost = []
    with open(done_path, 'a') as sink, \
            Pool(args.jobs, initializer=SETTINGS.update, initargs=(dict(SETTINGS),),
                 maxtasksperchild=50) as pool:
        results = pool.imap_unordered(process, todo, chunksize=1)
        finished = set()
        for k in range(1, len(todo) + 1):
            try:
                r = results.next(timeout=args.stall_timeout)
            except PoolTimeout:
                # A worker died (e.g. killed for exceeding the job's memory limit): Pool
                # never returns its task. Record what is missing instead of hanging.
                lost = [j[0] for j in todo if j[0] not in finished]
                print(f'  no result for {args.stall_timeout}s: {len(lost)} blocks recorded as lost '
                      '(worker killed? see dmesg). Rerun with --jobs 1 to retry them.', flush=True)
                pool.terminate()
                break
            finished.add(r['name'])
            records.append(r)
            sink.write(json.dumps(r) + '\n')
            sink.flush()
            if k % 200 == 0 or k == len(todo):
                counts = Counter(verdict(x) for x in records)
                print(f'  {k}/{len(todo)}  {dict(counts)}  {time.perf_counter() - t_wall:.0f}s', flush=True)
    for name in lost:
        records.append(dict(name=name, status='lost', reason='worker process died (likely out of memory)'))
    wall_s = time.perf_counter() - t_wall
    runs_path = args.out / 'invocations.jsonl'
    with open(runs_path, 'a') as f:
        f.write(json.dumps(dict(started_utc=started.isoformat(timespec='seconds'), wall_s=round(wall_s, 2),
                                blocks_run=len(todo), jobs=args.jobs, argv=sys.argv[1:])) + '\n')
    invocations = [json.loads(l) for l in runs_path.read_text().splitlines()]

    by_name = {r['name']: r for r in records}
    for b in blocks:                                    # duplicates inherit the representative's verdict
        r = by_name[b['name']]
        r.update(model=b.get('model'), cls=b.get('class'), size_bucket=b.get('size_bucket'),
                 n_ops=b.get('n_ops'), signature=b.get('signature'), total_flops=b.get('total_flops'),
                 abi_bytes=b.get('abi_bytes'))
        rp = rep.get(b.get('signature'))
        if r.get('reference', {}).get('status') == 'inherited' and rp in by_name:
            src = by_name[rp]
            r['verified_via'] = rp
            if 'correctness' in src:
                r['correctness'] = dict(passed=src['correctness']['passed'],
                                        passed_scaled=src['correctness'].get('passed_scaled'),
                                        max_abs_error=src['correctness'].get('max_abs_error'),
                                        max_rel_error=src['correctness'].get('max_rel_error'))
            else:
                r['reference'] = dict(status=src.get('reference', {}).get('status', 'missing'))
        r['verdict'] = verdict(r)
    records = [by_name[b['name']] for b in blocks]

    per_class = defaultdict(Counter)
    for r in records:
        per_class[r.get('cls')][r['verdict']] += 1
    fails = [r for r in records if r['verdict'] not in ('correct', 'correct_fp32_tolerance')]
    summary = dict(
        blocks=len(records), unique_signatures=len(rep),
        verdicts=dict(Counter(r['verdict'] for r in records)),
        verdicts_by_class={k: dict(v) for k, v in per_class.items()},
        verdicts_unique=dict(Counter(by_name[n]['verdict'] for n in rep.values())),
        unsupported_reasons=dict(Counter(r.get('reason') for r in records if r.get('status') != 'ok')),
        py_matches_graph=dict(Counter(str(r.get('py_matches_graph')) for r in records if r.get('status') == 'ok')),
        py_standalone=dict(Counter(str(r.get('py_standalone')) for r in records if r.get('status') == 'ok')),
        worst_rel_error=max((r['correctness'].get('max_rel_error') or 0 for r in records
                             if r.get('correctness') and r['correctness']['passed']), default=None),
        timing_ms=timing(records),
        timing_ms_by_class={c: timing([r for r in records if r.get('cls') == c]) for c in sorted(per_class)},
        invocations=invocations)
    env = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
               numpy=__import__('numpy').__version__, cpus=os.cpu_count(), jobs=args.jobs,
               torch_threads_per_job=1)
    settings = dict(seed=SEED, rtol=RTOL, atol=ATOL, max_volume=args.max_volume,
                    torch_calls=args.torch_calls, verify_all=args.verify_all, catalog=str(args.catalog),
                    reference='reference_eval.py (float64, sequential reductions)')
    doc = dict(schema_version=1, started_utc=started.isoformat(timespec='seconds'),
               environment=env, settings=settings, summary=summary, blocks=records)
    (args.out / 'preprocess.json').write_text(json.dumps(doc, indent=1))
    (args.out / 'failures.json').write_text(json.dumps(fails, indent=1))

    cols = ['name', 'model', 'cls', 'size_bucket', 'n_ops', 'verdict', 'status', 'reason', 'n_nodes',
            'translate_ms', 'render_ms', 'torch_first_call_ms', 'torch_eager_median_ms',
            'py_matches_graph', 'py_standalone', 'reference_status', 'reference_ms', 'reference_volume',
            'max_abs_error', 'max_rel_error', 'verified_via', 'total_flops', 'abi_bytes', 'signature']
    with open(args.out / 'preprocess.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore')
        w.writeheader()
        for r in records:
            row = dict(r)
            row.update(reference_status=r.get('reference', {}).get('status'),
                       reference_ms=r.get('reference', {}).get('ms'),
                       reference_volume=r.get('reference', {}).get('volume'),
                       max_abs_error=(r.get('correctness') or {}).get('max_abs_error'),
                       max_rel_error=(r.get('correctness') or {}).get('max_rel_error'))
            w.writerow(row)
    print(json.dumps(summary, indent=1))
    print(f'Wrote {args.out}/preprocess.json, preprocess.csv, failures.json, translations/')


if __name__ == '__main__':
    main()
