"""Plot the Gap 1 results: one PNG per check, from the CSVs the a*/b* scripts wrote under --results/<a1..b5>/."""
import argparse
import csv
import os
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (dataviz skill, light mode): fixed categorical order, status colors, ink.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # ordinal blue ramp (steps 250..700)
GOOD, WARN, CRIT, NEUTRAL = '#0ca30c', '#fab219', '#d03b3b', '#b8b7b0'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d9d8d2', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


def read(results, sub, name):
    path = os.path.join(results, sub, name)
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print('wrote', name)


def stacked_share(ax, labels, parts, colors, names):
    """Horizontal 100% stacked bars, one per label; parts[i][j] = count of names[j] for labels[i]."""
    left = [0.0] * len(labels)
    for j, (name, color) in enumerate(zip(names, colors)):
        shares = [p[j] / sum(p) if sum(p) else 0 for p in parts]
        ax.barh(labels, shares, left=left, color=color, edgecolor=SURFACE, linewidth=1.5, height=0.6, label=name)
        left = [a + b for a, b in zip(left, shares)]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.legend(ncol=len(names), loc='lower left', bbox_to_anchor=(0, 1.0))


def h(n):
    return 1.2 + 0.32 * n


def a1(res, out):
    rows = [r for r in read(res, 'a1', 'a1_producer_counts.csv') or [] if r['op_type'] == 'ALL']
    if not rows:
        return
    keys = ['0', '1', '2', '>=3']
    fig, ax = plt.subplots(figsize=(8, h(len(rows))))
    stacked_share(ax, [r['model'] for r in rows], [[int(r[k]) for k in keys] for r in rows], SEQ,
                  [f'{k} producers' for k in keys])
    ax.set_title('A1  Share of linalg ops by number of linalg producers', loc='left', pad=24)
    save(fig, out, 'a1_producers.png')


def a2(res, out):
    rows = [r for r in read(res, 'a2', 'a2_summary.csv') or [] if r['model'] != 'ALL']
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, h(len(rows))))
    vals = [num(r['pct_keep_ge2']) for r in rows]
    ax.barh([f"{r['model']} (n={r['multi']})" for r in rows], vals, color=SERIES[0], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f' {v:.0%}', va='center', color=INK2)
    ax.set_xlim(0, max(0.05, max(vals) * 1.15))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.set_title('A2  Multi-producer ops that keep >=2 producers inside a block', loc='left')
    save(fig, out, 'a2_kept_in_blocks.png')


def a3(res, out):
    d = [r for r in read(res, 'a3', 'a3_distribution.csv') or [] if r['model'] == 'ALL']
    t = (read(res, 'a3', 'a3_op_types.csv') or [])[:15]
    if not d:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar([int(r['n_blocks']) for r in d], [int(r['n_ops']) for r in d], color=SERIES[0], width=0.8)
    ax1.set_xlabel('number of blocks the op appears in')
    ax1.set_ylabel('model ops')
    ax1.set_title('A3  Block appearances per model op', loc='left')
    ax2.barh([r['op_type'] for r in t], [num(r['mean_per_op']) for r in t], color=SERIES[0], height=0.6)
    ax2.invert_yaxis()
    ax2.set_xlabel('mean blocks per op')
    ax2.set_title('Most repeated op types', loc='left')
    save(fig, out, 'a3_membership.png')


def a4(res, out):
    rows = read(res, 'a4', 'a4_summary.csv') or []
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, h(len(rows))))
    y = range(len(rows))
    for k, (key, name) in enumerate([('n_capped', 'start ops hitting max_paths'),
                                     ('n_starts_with_unreached', 'start ops with unreached producers')]):
        ax.barh([i + (k - 0.5) * 0.36 for i in y], [int(r[key]) for r in rows], height=0.34,
                color=SERIES[k], label=name)
    ax.set_yticks(list(y), [r['model'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('start ops')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title('A4  Path-enumeration cap', loc='left', pad=24)
    save(fig, out, 'a4_path_cap.png')


def a5(res, out):
    rows = read(res, 'a5', 'a5_edges.csv') or []
    if not rows:
        return
    models = sorted({r['model'] for r in rows})
    c = Counter((r['model'], r['cls']) for r in rows)
    classes = [('green', GOOD, 'green: fuse before RL'), ('yellow', WARN, 'yellow: left to RL'),
               ('red', CRIT, 'red: never fuse')]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3.5, h(len(models)))))
    stacked_share(ax1, models, [[c[(m, k)] for k, _, _ in classes] for m in models],
                  [col for _, col, _ in classes], [n for _, _, n in classes])
    ax1.set_title('A5  Producer->consumer edges by class', loc='left', pad=24)
    why = Counter(f"{r['cls']}: {r['reason'] or '-'}" for r in rows).most_common()
    ax2.barh([w for w, _ in why], [n for _, n in why], height=0.6,
             color=[dict(green=GOOD, yellow=WARN, red=CRIT)[w.split(':')[0]] for w, _ in why])
    ax2.invert_yaxis()
    ax2.set_xlabel('edges (all models)')
    ax2.set_title('Edges by class and reason', loc='left')
    save(fig, out, 'a5_edge_classes.png')


def a6(res, out):
    rows = [r for r in read(res, 'a6', 'a6_summary.csv') or [] if r['model'] != 'ALL']
    if not rows:
        return
    keys = ['same', 'dim0', 'dynamic_specialized', 'other', 'tag_missing']
    fig, ax = plt.subplots(figsize=(9, h(len(rows))))
    stacked_share(ax, [r['model'] for r in rows], [[int(r[k]) for k in keys] for r in rows],
                  [SERIES[0], SERIES[1], SERIES[2], SERIES[3], NEUTRAL], keys)
    ax.set_title('A6  Block op shapes vs the same op in the model', loc='left', pad=24)
    save(fig, out, 'a6_shape_drift.png')


def a7(res, out):
    s = read(res, 'a7', 'a7_groups.csv') or []
    u = (read(res, 'a7', 'a7_unseen_ops.csv') or [])[:15]
    if not s:
        return
    n = len(s)
    checks = [('loads > limit', 'loads_over'), ('stores > limit', 'stores_over'), ('loops > limit', 'loops_over')]
    vals = [sum(r[k] == 'True' for r in s) / n for _, k in checks] + [sum(bool(r['unseen_ops']) for r in s) / n]
    names = [c for c, _ in checks] + ['uses unseen arith/math ops']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.barh(names, vals, color=SERIES[0], height=0.6)
    for i, v in enumerate(vals):
        ax1.text(v, i, f' {v:.0%}', va='center', color=INK2)
    ax1.set_xlim(0, 1.1)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.invert_yaxis()
    ax1.set_title(f'A7  Merged green groups (n={n})', loc='left')
    ax2.barh([r['op'] for r in u], [int(r['occurrences']) for r in u], color=SERIES[0], height=0.6)
    ax2.invert_yaxis()
    ax2.set_xlabel('occurrences in merged groups')
    ax2.set_title('Body ops the features do not count', loc='left')
    save(fig, out, 'a7_group_limits.png')


def b1(res, out):
    rows = [r for r in read(res, 'b1', 'b1_noise.csv') or [] if num(r.get('cv_runs')) is not None]
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.hist([100 * num(r['cv_runs']) for r in rows], bins=30, color=SERIES[0], edgecolor=SURFACE)
    ax1.set_xlabel('coefficient of variation of single runs (%)')
    ax1.set_ylabel('benchmarks')
    ax1.set_title('B1  Run-to-run noise', loc='left')
    ratios = sorted(num(r['med2_max_over_min']) for r in rows)
    ax2.hist(ratios, bins=30, color=SERIES[0], edgecolor=SURFACE)
    p95 = ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]
    ax2.axvline(p95, color=INK2, linewidth=1, linestyle='--')
    ax2.text(p95, ax2.get_ylim()[1] * 0.95, f' p95 = {p95:.2f}x', color=INK2, va='top')
    ax2.set_xlabel('max / min of the 2-run medians (x)')
    ax2.set_title('Spread of the 2-run median (reward timing)', loc='left')
    save(fig, out, 'b1_noise.png')


def outcome_bar(ax, labels, counts, colors, title):
    ax.barh(labels, counts, color=colors, height=0.6)
    for i, v in enumerate(counts):
        ax.text(v, i, f' {v}', va='center', color=INK2)
    ax.invert_yaxis()
    ax.set_title(title, loc='left')


def b2(res, out):
    rows = read(res, 'b2', 'b2_tpf.csv') or []
    if not rows:
        return
    ok = sum(r.get('allclose') == 'True' for r in rows)
    bad = sum(r.get('allclose') == 'False' for r in rows)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    outcome_bar(ax, ['outputs match', 'outputs differ', 'could not replay/run'], [ok, bad, len(rows) - ok - bad],
                [GOOD, CRIT, NEUTRAL], f'B2  TPF schedules vs untransformed program (n={len(rows)})')
    ax.set_xlabel('schedules')
    save(fig, out, 'b2_tpf_correctness.png')


def b4(res, out):
    rows = sorted((r for r in read(res, args.b4, 'b4_prefusion.csv') or [] if num(r.get('speedup'))),
                  key=lambda r: num(r['speedup']))
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3.5, min(14, h(len(rows))))))
    names = [r['program'].replace('.mlir', '') for r in rows]
    colors = [SERIES[0] if r['allclose'] == 'True' else CRIT for r in rows]
    sp = [num(r['speedup']) for r in rows]
    ax1.barh(names, [s - 1 for s in sp], left=1, color=colors, height=0.6)  # bars grow from 1x
    for i, s in enumerate(sp):
        ax1.text(s, i, f' {s:.3f}x ', va='center', ha='left' if s >= 1 else 'right',
                 color=INK2)
    lo, hi = min(sp + [1]), max(sp + [1])
    if hi / lo > 2:
        ax1.set_xscale('log')
    pad = (hi - lo) * 0.25 or 0.01
    ax1.set_xlim(lo - pad if hi / lo <= 2 else lo / 1.3, hi + pad if hi / lo <= 2 else hi * 1.3)
    dec = 3 if hi - lo < 0.1 else 2
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:.{dec}f}x'))
    ax1.axvline(1, color=INK2, linewidth=1)
    ax1.set_xlabel('speedup of pre-fused vs original (default lowering; 1.00x = no change)')
    ax1.set_title('B4  Elementwise pre-fusion alone', loc='left', pad=24)
    ax1.legend(handles=[matplotlib.patches.Patch(color=SERIES[0], label='outputs match'),
                        matplotlib.patches.Patch(color=CRIT, label='outputs differ')],
               loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    y = range(len(rows))
    ax2.barh([i + 0.18 for i in y], [int(r['ops_before']) for r in rows], height=0.34, color=SERIES[0], label='before')
    ax2.barh([i - 0.18 for i in y], [int(r['ops_after']) for r in rows], height=0.34, color=SERIES[1], label='after')
    ax2.set_yticks(list(y), names)
    ax2.set_xlabel('linalg ops')
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    ax2.set_title('Op count', loc='left', pad=24)
    save(fig, out, f'{args.b4}_prefusion.png')


def b5(res, out):
    rows = [r for r in read(res, 'b5', 'b5_summary.csv') or [] if num(r.get('ops_ratio'))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, h(len(rows))))
    y = range(len(rows))
    for k, (key, name) in enumerate([('ops_ratio', 'sum of single ops / full model'),
                                     ('blocks_ratio', 'sum of blocks / full model')]):
        ax.barh([i + (k - 0.5) * 0.36 for i in y], [num(r[key]) for r in rows], height=0.34, color=SERIES[k], label=name)
    ax.axvline(1, color=INK2, linewidth=1, linestyle='--')
    ax.text(1, -0.9, ' additive', color=INK2)
    ax.set_yticks(list(y), [r['model'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('time ratio (1.0 = additive)')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title('B5  Additivity of measured time', loc='left', pad=24)
    save(fig, out, 'b5_additivity.png')


def report(res, out, part):
    """REPORT_<A|B>.md in --results: every check's summary table followed by its plots."""
    subs = [f'a{i}' for i in range(1, 8)] if part == 'a' else ['b1', 'b2', 'b3', 'b4', 'b4_green', 'b5']
    text = [f'# Gap 1 results, part {part.upper()}\n']
    for sub in subs:
        md = [f for f in sorted(os.listdir(os.path.join(res, sub))) if f.endswith('_summary.md')] \
            if os.path.isdir(os.path.join(res, sub)) else []
        if not md:
            text.append(f'\n## {sub}\n\n_not run yet_\n')
            continue
        body = open(os.path.join(res, sub, md[0]), encoding='utf-8').read()
        text.append('\n' + '\n'.join('#' + ln if ln.startswith('#') else ln for ln in body.splitlines()) + '\n')
        for png in sorted(f for f in os.listdir(out) if f.startswith(sub + '_') and f.endswith('.png')
                          and not (sub == 'b4' and f.startswith('b4_green'))):
            text.append(f'\n![{png}]({os.path.relpath(os.path.join(out, png), res)})\n')
    path = os.path.join(res, f'REPORT_{part.upper()}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(text))
    print('wrote', path)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--results', default='out', help='folder holding a1/ ... b5/ (the --out folders of the scripts)')
ap.add_argument('--out', default=None, help='where to write PNGs (default: <results>/plots)')
ap.add_argument('--only', choices=['a', 'b'], default=None, help='only the A or only the B checks (default: both)')
args = ap.parse_args()
out = args.out or os.path.join(args.results, 'plots')
os.makedirs(out, exist_ok=True)
parts = [args.only] if args.only else ['a', 'b']
if 'a' in parts:
    for fn in (a1, a2, a3, a4, a5, a6, a7):
        fn(args.results, out)
if 'b' in parts:
    for fn in (b1, b2, b5):
        fn(args.results, out)
    for sub in ('b4', 'b4_green'):
        args.b4 = sub
        b4(args.results, out)
for part in parts:
    report(args.results, out, part)
