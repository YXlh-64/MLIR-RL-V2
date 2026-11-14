#!/usr/bin/env python3
"""Create exactly three execution-speedup figures from validated CSV results.

Use the runner's median of paired-trial speedups, rather than recomputing a
ratio of aggregate latencies. Failed/incomplete results stay visibly missing.
All categories share a linear axis starting at zero and identical image
dimensions. Labels retain the exact values of small speedups.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

# Plotting needs no BLAS thread pool, especially on restricted login nodes.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
CATEGORIES = {'single_operations': 'Single operations',
              'multi_operation_blocks': 'Multi-operation blocks', 'full_models': 'Full models'}
METHODS = [('mlir_baseline', 'MLIR baseline', '#666666'),
           ('mlir_rl', 'MLIR RL', '#0072B2'),
           ('torch_inductor', 'TorchInductor', '#D55E00'),
           ('torch_eager', 'PyTorch eager', '#009E73'),
           ('tensorflow_xla', 'TensorFlow XLA', '#CC79A7')]


def validated_speedup(row, trials):
    if int(row.get('valid_trials', 0)) != trials or row.get('correctness') != 'passed':
        return None
    raw = row.get('median_validated_speedup')
    result = float(raw) if raw else None
    return result if result is not None and math.isfinite(result) and result > 0 else None


def plot_category(rows, trials, output, category, threads, ticks, figsize, keys, note):
    lookup = {(r['dataset'], r['benchmark'], r['method']): r for r in rows}

    values = {(key, method): validated_speedup(lookup.get((*key, method), {}), trials)
              for key in keys for method, *_ in METHODS}
    available = [v for v in values.values() if v is not None]
    labels = [Path(name).stem for _, name in keys]
    # Include dataset only when filenames collide; preserve benchmark names.
    labels = [f'{dataset}/{label}' if labels.count(label) > 1 else label
              for (dataset, _), label in zip(keys, labels)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.35, right=0.92, top=0.88, bottom=0.14)
    for i in range(len(keys)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#F3F5F7', zorder=0)
    ax.set_xscale('linear')
    ax.set_xlim(ticks[0], ticks[-1])
    ax.axvline(1, color='#555555', linestyle='--', linewidth=1, zorder=4)
    for j, (method, label, color) in enumerate(METHODS):
        for i, key in enumerate(keys):
            y = i + (j - (len(METHODS) - 1) / 2) * 0.16
            speedup = values[key, method]
            if speedup is None:
                ax.plot(1.045, y, marker='x', color=color, markersize=6,
                        transform=ax.get_yaxis_transform(), clip_on=False)
            else:
                # Every bar starts at zero and ends at its exact speedup.
                ax.barh(y, speedup, left=0, height=0.13,
                        color=color, edgecolor='white', linewidth=0.3, zorder=2)
                ax.annotate(f'{speedup:.3g}×', (speedup, y), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=7.5, color=color)
    ax.set_yticks(range(len(keys)), labels, fontsize=8.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0, pad=10)
    # Identical linear increments, limits and canvas/axes dimensions ensure
    # that the same speedup has the same pixel length in every saved figure.
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}×'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='x', which='major', color='#D8DDE3', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Execution speedup relative to MLIR baseline · shared linear scale starting at 0', labelpad=12)
    ax.text(1.045, 1.015, 'N/A', transform=ax.transAxes, ha='center', fontsize=9, color='#555555')
    fig.suptitle(f'{CATEGORIES[category]} — execution speedups', fontsize=19, fontweight='bold', y=0.98)
    fig.text(0.5, 0.947, f'{threads} CPU threads · {trials} validated trials per result · baseline = 1× · higher is faster',
             ha='center', fontsize=11, color='#555555')
    handles = [Patch(facecolor=color, label=label) for _, label, color in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.928), ncol=len(METHODS), frameon=False)
    missing = len(values) - len(available)
    fig.text(0.35, 0.025,
        f'Speedup = baseline latency / method latency, aggregated by paired-trial median.\n'
        f'Right-column × = unavailable, failed or incomplete; {missing} of {len(values)} results missing. '
        f'Missing results are never plotted as zero.\n{note}', fontsize=9, color='#555555')
    path = output / f'speedups_{category}.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'{path} ({len(available)}/{len(values)} valid results)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'results/execution')
    parser.add_argument('--output', type=Path, default=HERE / 'results/figures')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--samples', type=Path, default=HERE / 'sampled_benchmarks.csv')
    parser.add_argument('--note', default='', help='Measurement context printed below each figure')
    args = parser.parse_args()
    with args.samples.open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    data, all_speedups = {}, [1.0]
    methods = {method for method, *_ in METHODS}
    for category in CATEGORIES:
        folder = args.input / category
        with (folder / 'comparison.csv').open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['threads']) == args.threads]
        if not rows:
            raise ValueError(f'No {args.threads}-thread results for {category}')
        trials = json.loads((folder / 'campaign.json').read_text())['trials']
        keys = sorted({(r['dataset'], r['benchmark']) for r in rows} |
                      {(r['dataset'], r['benchmark']) for r in samples if r['category'] == category})
        data[category] = rows, trials, keys
        for row in rows:
            value = validated_speedup(row, trials) if row['method'] in methods else None
            if value is not None:
                all_speedups.append(value)
    # All figures use the same zero origin, upper limit and evenly spaced ticks.
    upper = 100 * math.ceil(max(all_speedups) * 1.1 / 100)
    ticks = list(range(0, upper + 1, 100))
    max_rows = max(len(keys) for _, _, keys in data.values())
    figsize = (18, max(8, 0.62 * max_rows + 2.4))
    print(f'Shared linear axis: 0× to {ticks[-1]:g}×; 100× ticks', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        rows, trials, keys = data[category]
        plot_category(rows, trials, args.output, category, args.threads, ticks, figsize, keys, args.note)
