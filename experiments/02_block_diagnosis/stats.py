#!/usr/bin/env python3
"""Statistics for experiment 02: confidence intervals, paired tests, headline tables.

Library (used by ablation.py and plot_blocks.py):
  median, mean, geomean
  bootstrap(fn, groups)       percentile CI of fn(medians of resampled groups)
  ratio_ci(base, cand)        CI of median(base)/median(cand) from raw latency samples
  wilcoxon(values)            paired signed-rank test of values against 0 (two-sided)
  holm(pvalues)               Holm-Bonferroni adjustment

CLI:
  python experiments/02_block_diagnosis/stats.py --all-results results/<tag>/all_results.csv
writes next to it (or --out):
  per_block.csv       one row per (config, block, method): speedup vs MLIR baseline + 95% CI,
                      win/tie/loss, run-to-run variation
  headline_table.csv/.md  per (config, group, method): geomean speedup [CI], W/T/L,
                      Wilcoxon p, Holm-adjusted p, effect size
  per_model.csv, per_size.csv   geomean speedup per model / per size bucket
  stats.json          everything above plus coverage counts

Decision rules (fixed before looking at results):
  * a speedup only counts if correctness passed (experiment 01 rule);
  * win  = 95% CI of the per-block speedup above 1 AND point estimate >= 1 + THRESHOLD;
    loss = CI below 1 AND point <= 1 / (1 + THRESHOLD); otherwise tie;
  * across blocks: geometric mean with a bootstrap CI over blocks, Wilcoxon signed-rank
    on per-block log ratios, Holm correction over every test in the table,
    rank-biserial correlation as effect size.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path

THRESHOLD = 0.05
METHODS = ('mlir_baseline', 'mlir_rl', 'torch_eager', 'torch_inductor')
# Paired comparisons: (label, numerator method, denominator method); ratio = T_num / T_den,
# so > 1 means the denominator method is faster.
COMPARISONS = [('mlir_rl vs mlir_baseline', 'mlir_baseline', 'mlir_rl'),
               ('torch_eager vs mlir_baseline', 'mlir_baseline', 'torch_eager'),
               ('torch_inductor vs mlir_baseline', 'mlir_baseline', 'torch_inductor'),
               ('mlir_rl vs torch_inductor', 'torch_inductor', 'mlir_rl'),
               ('mlir_rl vs torch_eager', 'torch_eager', 'mlir_rl')]


# --------------------------------------------------------------------------- basic helpers

def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def bootstrap(fn, groups, n=2000, seed=2026, alpha=0.05):
    """Percentile CI of fn([median(resample(g)) for g in groups])."""
    import numpy as np
    rng = np.random.default_rng(seed)
    meds = []
    for g in groups:
        a = np.asarray(g, dtype=float)
        idx = rng.integers(0, len(a), size=(n, len(a)))
        meds.append(np.median(a[idx], axis=1))
    values = np.array([fn([m[b] for m in meds]) for b in range(n)])
    values = values[np.isfinite(values)]
    return (float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))) \
        if len(values) else (math.nan, math.nan)


def ratio_ci(base, cand, n=2000):
    return bootstrap(lambda m: m[0] / m[1], [base, cand], n=n)


def geomean_ci(values, n=2000, seed=2026):
    """Bootstrap over blocks of the geometric mean."""
    import numpy as np
    logs = np.log(np.asarray([v for v in values if v and v > 0], dtype=float))
    if len(logs) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(logs), size=(n, len(logs)))
    boot = np.exp(logs[idx].mean(axis=1))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def wilcoxon(values):
    """Two-sided Wilcoxon signed-rank test of `values` against 0.

    Uses scipy when installed (exact for small n); otherwise the normal
    approximation with tie and continuity correction. Effect size: matched-pairs
    rank-biserial correlation (+1 = all positive, -1 = all negative).
    """
    d = [v for v in values if v != 0 and math.isfinite(v)]
    n = len(d)
    if n == 0:
        return dict(n=0, statistic=None, p=None, effect_size=None, method='none')
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                       # average ranks for ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    w_minus = sum(r for r, v in zip(ranks, d) if v < 0)
    total = n * (n + 1) / 2
    effect = (w_plus - w_minus) / total
    try:
        from scipy.stats import wilcoxon as sp_wilcoxon
        res = sp_wilcoxon(d, alternative='two-sided')
        return dict(n=n, statistic=float(res.statistic), p=float(res.pvalue), effect_size=effect, method='scipy')
    except Exception:
        pass
    mu = total / 2
    ties = collections.Counter(ranks)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48)
    if sigma == 0:
        return dict(n=n, statistic=w_plus, p=1.0, effect_size=effect, method='normal')
    z = (abs(w_plus - mu) - 0.5) / sigma
    return dict(n=n, statistic=w_plus, p=min(1.0, math.erfc(max(z, 0) / math.sqrt(2))),
                effect_size=effect, method='normal-approx')


def holm(pvalues):
    """{key: p} -> {key: Holm-adjusted p}; None stays None."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m, adjusted, running = len(items), {}, 0.0
    for rank, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[k] = running
    return {k: adjusted.get(k) for k in pvalues}


def verdict(point, lo, hi):
    if point is None or lo is None or not math.isfinite(lo):
        return 'unknown'
    if lo > 1 and point >= 1 + THRESHOLD:
        return 'win'
    if hi < 1 and point <= 1 / (1 + THRESHOLD):
        return 'loss'
    return 'tie'


# --------------------------------------------------------------------------- summarise all_results.csv

def load_samples(path, cache={}):
    if path not in cache:
        try:
            cache[path] = json.loads(Path(path).read_text()).get('latency', {}).get('samples_ns') or []
        except (OSError, ValueError):
            cache[path] = []
    return cache[path]


def per_block_table(rows, bootstrap_n):
    """Aggregate repeats; one entry per (config, block, method) with validated data."""
    groups = collections.defaultdict(list)
    meta = {}
    for r in rows:
        config = f"t{r['threads']}_{r['cache']}"
        groups[(config, r['block'], r['method'])].append(r)
        meta[r['block']] = {k: r[k] for k in ('model', 'class', 'size_bucket', 'n_ops',
                                              'arithmetic_intensity', 'total_flops', 'role')}
    table = {}
    for (config, block, method), reps in groups.items():
        ok = [r for r in reps if r['status'] == 'measured' and r['correctness'] == 'passed']
        if not ok:
            continue
        medians = [float(r['median_ns']) for r in ok]
        pooled = [s for r in ok for s in load_samples(r['result_json'])]
        table[(config, block, method)] = dict(
            config=config, block=block, method=method, repeats=len(ok), median_ns=median(medians),
            run_cv=(math.sqrt(sum((m - mean(medians)) ** 2 for m in medians) / (len(medians) - 1)) / mean(medians))
            if len(medians) > 1 else None,
            samples=pooled, **meta[block])
    out = []
    for (config, block, method), e in table.items():
        base = table.get((config, block, 'mlir_baseline'))
        row = {k: v for k, v in e.items() if k != 'samples'}
        if base and method != 'mlir_baseline':
            point = base['median_ns'] / e['median_ns']
            lo, hi = ratio_ci(base['samples'], e['samples'], n=bootstrap_n) \
                if base['samples'] and e['samples'] else (math.nan, math.nan)
            flops = float(e['total_flops']) if e.get('total_flops') not in (None, '') else None
            row.update(speedup=point, ci_low=lo, ci_high=hi, verdict=verdict(point, lo, hi),
                       gflops=flops / e['median_ns'] if flops else None)
        out.append(row)
    return out, table


def headline(per_block, table, groups_of):
    tests, rows = {}, []
    configs = sorted({r['config'] for r in per_block})
    for config in configs:
        blocks = sorted({r['block'] for r in per_block if r['config'] == config})
        for group_name, members in groups_of(blocks):
            for label, num, den in COMPARISONS:
                ratios = []
                for b in members:
                    a, c = table.get((config, b, num)), table.get((config, b, den))
                    if a and c:
                        ratios.append(a['median_ns'] / c['median_ns'])
                if not ratios:
                    continue
                w = wilcoxon([math.log(x) for x in ratios])
                key = (config, group_name, label)
                tests[key] = w['p']
                pb = [r for r in per_block if r['config'] == config and r['block'] in members
                      and r['method'] == den and num == 'mlir_baseline']
                lo, hi = geomean_ci(ratios)
                rows.append(dict(config=config, group=group_name, comparison=label, n_blocks=len(ratios),
                                 geomean_ratio=geomean(ratios), ci_low=lo, ci_high=hi,
                                 wins=sum(r['verdict'] == 'win' for r in pb) if pb else None,
                                 ties=sum(r['verdict'] == 'tie' for r in pb) if pb else None,
                                 losses=sum(r['verdict'] == 'loss' for r in pb) if pb else None,
                                 wilcoxon_p=w['p'], effect_size=w['effect_size'], test=w['method']))
    adjusted = holm(tests)
    for r in rows:
        r['holm_p'] = adjusted.get((r['config'], r['group'], r['comparison']))
    return rows


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:.{digits}g}' if isinstance(x, float) else str(x)


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or ['empty'])
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Statistics over all_results.csv (reads files only).')
    parser.add_argument('--all-results', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    out = args.out or args.all_results.parent / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    with open(args.all_results, newline='') as f:
        rows = list(csv.DictReader(f))

    per_block, table = per_block_table(rows, args.bootstrap)
    meta = {r['block']: r for r in per_block}

    def by(key):
        def groups_of(blocks):
            yield 'all', blocks
            for value in sorted({meta[b][key] for b in blocks}):
                yield f'{key}={value}', [b for b in blocks if meta[b][key] == value]
        return groups_of

    head = headline(per_block, table, by('class'))
    per_model = headline(per_block, table, by('model'))
    per_size = headline(per_block, table, by('size_bucket'))

    write_csv(out / 'per_block.csv', per_block)
    write_csv(out / 'headline_table.csv', head)
    write_csv(out / 'per_model.csv', [r for r in per_model if r['group'] != 'all'])
    write_csv(out / 'per_size.csv', [r for r in per_size if r['group'] != 'all'])

    coverage = collections.Counter((r['method'], r['status'], r['correctness']) for r in rows)
    md = ['# Experiment 02 headline table', '',
          'Ratio = time(numerator method) / time(compared method): > 1 means the compared method is faster.',
          f'Win/tie/loss: per-block 95% CI vs 1 with a {THRESHOLD:.0%} practical threshold. '
          'p-values: Wilcoxon signed-rank on per-block log ratios, Holm-adjusted over the whole table.', '',
          '| Config | Group | Comparison | n | Geomean [95% CI] | W/T/L | p (Holm) | Effect |',
          '| --- | --- | --- | ---: | --- | --- | --- | ---: |']
    for r in head:
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}" if r['wins'] is not None else '—'
        md.append(f"| {r['config']} | {r['group']} | {r['comparison']} | {r['n_blocks']} | "
                  f"{fmt(r['geomean_ratio'])}× [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {wtl} | "
                  f"{fmt(r['holm_p'], 2)} | {fmt(r['effect_size'], 2)} |")
    md += ['', '## Status counts (method, status, correctness)', '']
    md += [f'- {m} / {s} / {c}: {n}' for (m, s, c), n in sorted(coverage.items())]
    (out / 'headline_table.md').write_text('\n'.join(md) + '\n')
    (out / 'stats.json').write_text(json.dumps(dict(
        threshold=THRESHOLD, headline=head, per_model=per_model, per_size=per_size,
        status_counts=[dict(method=m, status=s, correctness=c, n=n) for (m, s, c), n in coverage.items()]),
        indent=1, default=lambda x: None))
    print(f'Wrote {out}/headline_table.md and CSVs')


if __name__ == '__main__':
    main()
