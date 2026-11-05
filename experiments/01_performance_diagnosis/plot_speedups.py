#!/usr/bin/env python3
"""Plot one data family's validated backend speedups; runs no benchmarks.

Requires Matplotlib. Pass --results-dir results/<dataset>.
Reads category comparison.csv files and writes one speedup_comparison.png.
"""
import argparse
import csv
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

SERIES = (
    ('mlir_baseline', 'MLIR baseline', '#8A929E'),
    ('mlir_rl', 'MLIR RL', '#0072B2'),
    ('torch_eager', 'PyTorch eager', '#D55E00'),
    ('torch_inductor', 'TorchInductor', '#009E73'),
)


def validated(row):
    return (row.get('status') == 'measured' and row.get('correctness') == 'passed'
            and bool(row.get('validated_speedup')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True, type=Path,
                        help='Data-family directory containing category comparison.csv files')
    args = parser.parse_args()
    results = args.results_dir.resolve()
    groups = {}
    sources = sorted(results.glob('*/comparison.csv'))
    if not sources:
        raise SystemExit(f'No category comparison.csv files in {results}; pass results/<dataset>.')
    for path in sources:
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                groups.setdefault(row['workload'], {})[row['method']] = row
    comparable = {
        name: methods for name, methods in groups.items()
        if validated(methods.get('mlir_baseline', {}))
        and any(validated(methods.get(method, {})) for method, _, _ in SERIES[1:])
    }
    if not comparable:
        print(f'No validated backend comparisons in {results}; no figure generated.')
        return
    names = sorted(comparable)
    speedups = {}
    for method, _, _ in SERIES:
        for name, rows in comparable.items():
            row = rows.get(method, {})
            if not validated(row):
                continue
            ratio = float(rows['mlir_baseline']['median_us']) / float(rows[method]['median_us'])
            if (not math.isfinite(ratio) or ratio <= 0
                    or not math.isclose(ratio, float(row['validated_speedup']), rel_tol=1e-9)):
                raise ValueError(f'Speedup does not match latency ratio: {name}/{method}')
            speedups[name, method] = ratio
    lower = min(0.5, min(speedups.values()) * 0.65)
    upper = max(2, max(speedups.values()) * 1.6)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    height = max(5.5, 2.6 + 0.85 * len(names))
    fig, ax = plt.subplots(figsize=(14, height))
    for offset, (method, label, color) in zip((-0.3, -0.1, 0.1, 0.3), SERIES):
        positions, values = [], []
        for i, name in enumerate(names):
            value = speedups.get((name, method))
            if value is None:
                status = comparable[name].get(method, {}).get('status', 'not run')
                ax.text(lower * 1.05, i + offset, f'N/A ({status})',
                        va='center', color=color, fontsize=9)
            else:
                positions.append(i + offset)
                values.append(value)
        ax.barh(positions, values, height=0.16, color=color, label=label, zorder=3)
        for position, value in zip(positions, values):
            ax.annotate(f'{value:.3f}×' if 0.99 < value < 1.01 else f'{value:.2f}×',
                        (value, position), xytext=(5, 0), textcoords='offset points',
                        va='center', fontsize=9, color=color)
    ax.set_xscale('log')
    ax.set_xlim(lower, upper)
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator([tick for tick in ticks if lower <= tick <= upper]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}×'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1, color='#374151', linestyle='--', linewidth=1, zorder=2)
    labels = [textwrap.fill(name.replace('_', ' '), width=35) for name in names]
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Speedup over MLIR baseline (log scale) · higher is faster', labelpad=12)
    ax.grid(axis='x', color='#E5E7EB', zorder=0)
    ax.tick_params(axis='both', length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = {'mlir_rl_v1_paper': 'Single operations · paper dataset',
             'ops_and_blocks': 'Operations and blocks',
             'full_models': 'Full models'}.get(results.name, results.name)
    fig.suptitle(title, x=0.04, y=1 - 0.15 / height,
                 ha='left', fontsize=17, fontweight='bold')
    fig.text(0.04, 1 - 0.6 / height, 'Speedup = baseline median latency / backend median latency · one CPU thread',
             fontsize=10, color='#4B5563')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left',
               bbox_to_anchor=(0.035, 1 - 0.83 / height), ncol=4, frameon=False)
    fig.text(0.04, 0.12 / height,
             f'{len(names)}/{len(groups)} recorded workloads have validated comparisons. Baseline = 1×; below 1× is slower.\n'
             'N/A = no validated measurement. Blocked tasks have no recorded comparison. Source: category comparison.csv files.',
             fontsize=9, color='#4B5563')
    fig.subplots_adjust(left=0.28, right=0.94, top=1 - 1.35 / height, bottom=1.2 / height)
    path = results / 'speedup_comparison.png'
    fig.savefig(path, dpi=180, facecolor='white')
    print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
