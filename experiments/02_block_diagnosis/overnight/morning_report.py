#!/usr/bin/env python3
"""Write overnight/MORNING_REPORT.md from every results/ckpt_*/all_results.csv.

Per checkpoint: how many blocks finished per method, how many passed correctness, and the
geometric-mean latency ratio vs mlir_baseline (only on blocks where both ran correctly).
Across checkpoints: a paired, block-by-block comparison of the mlir_rl latency.
Standard library only.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor')


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def fmt(x, nd=3):
    return '–' if x is None else f'{x:.{nd}f}'


def load(tag_dir):
    rows = list(csv.DictReader((tag_dir / 'all_results.csv').open()))
    by = defaultdict(dict)                     # block -> method -> row (t1_warm, repeat 0)
    for r in rows:
        if r['config'] == 't1_warm' and r['repeat'] == '0':
            by[r['block']][r['method']] = r
    return by


def ok(r):
    return r is not None and r.get('status') == 'measured' and r.get('median_ns') not in ('', None)


def main():
    tags = sorted(p for p in RESULTS.glob('ckpt_*') if (p / 'all_results.csv').exists())
    out = ['# Overnight report', '']
    if not tags:
        out += ['No results yet (no `results/ckpt_*/all_results.csv`). Check the logs in '
                '`overnight/logs/`, starting with the setup log.']
    data = {t.name: load(t) for t in tags}
    for tag, by in data.items():
        sel = len(list((RESULTS / tag / 'translations').glob('*.py')))
        out += [f'## {tag}', '', f'Blocks with results: {len(by)} (selected and translated: {sel})', '',
                '| Method | ran ok | correctness passed | geomean speedup vs mlir_baseline* | blocks in geomean |',
                '|---|---|---|---|---|']
        for m in METHODS:
            ran = [b for b, d in by.items() if ok(d.get(m))]
            passed = [b for b in ran if by[b][m].get('correctness') == 'passed']
            pairs = [float(by[b]['mlir_baseline']['median_ns']) / float(by[b][m]['median_ns'])
                     for b in passed if ok(by[b].get('mlir_baseline'))
                     and by[b]['mlir_baseline'].get('correctness') == 'passed']
            out.append(f'| {m} | {len(ran)}/{len(by)} | {len(passed)} | {fmt(geomean(pairs))} | {len(pairs)} |')
        fails = defaultdict(int)
        for d in by.values():
            for m, r in d.items():
                if r.get('status') != 'measured':
                    fails[(m, (r.get('reason') or r.get('status') or '')[:90])] += 1
        if fails:
            out += ['', 'Failures (method, reason, count):', '']
            out += [f'- `{m}`: {reason} ({n})' for (m, reason), n in sorted(fails.items(), key=lambda x: -x[1])[:15]]
        out.append('')
    out.append('*speedup = baseline median latency / method median latency, over blocks where both '
               'passed correctness. >1 means faster than plain MLIR.')
    if len(data) >= 2:
        names = list(data)
        out += ['', '## Checkpoints compared on the same blocks (mlir_rl latency)', '']
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                common = [k for k in data[a] if k in data[b] and ok(data[a][k].get('mlir_rl'))
                          and ok(data[b][k].get('mlir_rl'))]
                ratios = [float(data[b][k]['mlir_rl']['median_ns']) / float(data[a][k]['mlir_rl']['median_ns'])
                          for k in common]
                wins = sum(r > 1.05 for r in ratios)
                losses = sum(r < 1 / 1.05 for r in ratios)
                out += [f'- {a} vs {b}: {len(common)} common blocks; geomean latency({b})/latency({a}) = '
                        f'{fmt(geomean(ratios))}; {a} faster by >5% on {wins}, slower by >5% on {losses}, '
                        f'within 5% on {len(ratios) - wins - losses}.']
    out += ['', 'Details: `results/<tag>/analysis/headline_table.md`, `results/<tag>/t1_warm/r0/<class>/REPORT.md`, '
            '`results/<tag>/figures/`, raw rows in `results/<tag>/all_results.csv`.']
    (HERE / 'MORNING_REPORT.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    sys.exit(main())
