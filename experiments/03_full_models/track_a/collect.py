#!/usr/bin/env python3
"""Collect Track A results into results/track_a.csv and results/track_a.md.

speedup = mean of the two same-node MLIR baseline runs / MLIR-RL time (each value is the median
of the 2 timed runs the repo's Execution does). noise = |baseline_1 - baseline_2| / mean.
"""
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def value(path, model):
    try:
        v = json.loads(Path(path).read_text()).get(model)
        return float(v) if v and float(v) > 0 else None
    except (OSError, ValueError):
        return None


def wall(path):
    try:
        return json.loads(Path(path).read_text())['wall_seconds']
    except (OSError, ValueError, KeyError):
        return None


models = [l.split('\t') for l in (HERE / 'models.tsv').read_text().splitlines()]
rows = []
for threads in (1, 16):
    for model, _, _, stored, _ in models:
        d = HERE / 'results' / f't{threads}' / model
        b1, b2 = value(d / 'baseline_1/baseline.json', model), value(d / 'baseline_2/baseline.json', model)
        rl = value(d / 'rl/eval_exec_times.json', model)
        log = (d / 'rl/stdout.log').read_text(errors='replace') if (d / 'rl/stdout.log').exists() else ''
        base = [x for x in (b1, b2) if x]
        bm = sum(base) / len(base) if base else None
        rows.append(dict(
            model=model, threads=threads, stored_baseline_s=float(stored) / 1e9,
            baseline_1_s=b1 and b1 / 1e9, baseline_2_s=b2 and b2 / 1e9,
            baseline_noise_pct=(abs(b1 - b2) / bm * 100) if b1 and b2 else None,
            mlir_rl_s=rl and rl / 1e9, speedup=(bm / rl) if bm and rl else None,
            rl_failed_actions=len(re.findall(r'Error applying action', log)) if log else None,
            rl_wall_min=(wall(d / 'rl/wall.json') or 0) / 60 or None,
            status='ok' if rl and bm else ('rl failed' if bm else 'missing')))

with open(HERE / 'results' / 'track_a.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def fmt(x, p=3):
    return '' if x is None else f'{x:.{p}g}'


lines = ['# Track A: MLIR-RL (v5_model_1200) vs MLIR baseline on full models', '',
         'Same node for both baseline runs and the RL run. Times in seconds.', '']
for threads in (1, 16):
    rs = [r for r in rows if r['threads'] == threads]
    sp = [r['speedup'] for r in rs if r['speedup']]
    gm = math.exp(sum(map(math.log, sp)) / len(sp)) if sp else None
    lines += [f'## {threads} thread{"s" if threads > 1 else ""}', '',
              '| Model | Baseline (run 1 / run 2) | Noise | MLIR-RL | Speedup | Failed RL actions | RL eval wall (min) | Status |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rs:
        lines.append(f"| {r['model']} | {fmt(r['baseline_1_s'])} / {fmt(r['baseline_2_s'])} | "
                     f"{fmt(r['baseline_noise_pct'], 2)}% | {fmt(r['mlir_rl_s'])} | "
                     f"{'**' + fmt(r['speedup']) + '×**' if r['speedup'] else ''} | "
                     f"{fmt(r['rl_failed_actions'])} | {fmt(r['rl_wall_min'], 2)} | {r['status']} |")
    lines += ['', f'Geomean speedup over {len(sp)} models: **{fmt(gm)}×**; '
              f'faster on {sum(s > 1.05 for s in sp)}, within 5% on {sum(0.95 <= s <= 1.05 for s in sp)}, '
              f'slower on {sum(s < 0.95 for s in sp)}.', '']
(HERE / 'results' / 'track_a.md').write_text('\n'.join(lines))
print('\n'.join(lines))
