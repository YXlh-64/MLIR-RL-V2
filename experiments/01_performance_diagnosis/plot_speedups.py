#!/usr/bin/env python3
"""Create exactly three execution-speedup figures from validated CSV results.

Use the runner's median of paired-trial speedups, rather than recomputing a
ratio of aggregate latencies. Failed/incomplete results stay visibly missing.
All categories share a logarithmic axis and identical image dimensions, so
small speedups remain visible alongside results hundreds of times faster.
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
           ('torch_eager', 'PyTorch eager', '#009E73')]


def validated_speedup(row, trials):
    if int(row.get('valid_trials', 0)) != trials or row.get('correctness') != 'passed':
        return None
    raw = row.get('median_validated_speedup')
    result = float(raw) if raw else None
    return result if result is not None and math.isfinite(result) and result > 0 else None


def plot_category(rows, trials, output, category, threads, ticks, figsize):
    keys = sorted({(r['dataset'], r['benchmark']) for r in rows})
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
    ax.set_xscale('log')
    ax.set_xlim(ticks[0], ticks[-1])
    ax.axvline(1, color='#555555', linestyle='--', linewidth=1, zorder=4)
    for j, (method, label, color) in enumerate(METHODS):
        for i, key in enumerate(keys):
            y = i + (j - 1.5) * 0.20
            speedup = values[key, method]
            if speedup is None:
                ax.plot(1.045, y, marker='x', color=color, markersize=6,
                        transform=ax.get_yaxis_transform(), clip_on=False)
            else:
                # Logarithmic axes have no zero. Use the same positive origin
                # for all bars and place each endpoint at its exact speedup.
                ax.barh(y, speedup - ticks[0], left=ticks[0], height=0.16,
                        color=color, edgecolor='white', linewidth=0.3, zorder=2)
                ax.annotate(f'{speedup:.3g}×', (speedup, y), xytext=(6, 0),
                            textcoords='offset points', va='center', fontsize=7.5, color=color)
    ax.set_yticks(range(len(keys)), labels, fontsize=8.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0, pad=10)
    # Identical multiplicative increments, limits and canvas/axes dimensions ensure
    # that the same speedup has the same pixel length in every saved figure.
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}×'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='x', which='major', color='#D8DDE3', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Execution speedup relative to MLIR baseline · shared log scale (each tick = 10×)', labelpad=12)
    ax.text(1.045, 1.015, 'N/A', transform=ax.transAxes, ha='center', fontsize=9, color='#555555')
    fig.suptitle(f'{CATEGORIES[category]} — execution speedups', fontsize=19, fontweight='bold', y=0.98)
    fig.text(0.5, 0.947, f'{threads} CPU threads · {trials} validated trials per result · baseline = 1× · higher is faster',
             ha='center', fontsize=11, color='#555555')
    handles = [Patch(facecolor=color, label=label) for _, label, color in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.928), ncol=4, frameon=False)
    missing = len(values) - len(available)
    fig.text(0.35, 0.025,
        f'Speedup = baseline latency / method latency, aggregated by paired-trial median.\n'
        f'Right-column × = unavailable, failed or incomplete; {missing} of {len(values)} results missing. '
        'Missing results are never plotted as zero.', fontsize=9, color='#555555')
    path = output / f'speedups_{category}.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'{path} ({len(available)}/{len(values)} valid results)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'results/execution')
    parser.add_argument('--output', type=Path, default=HERE / 'results/figures')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    data, all_speedups = {}, [1.0]
    methods = {method for method, *_ in METHODS}
    for category in CATEGORIES:
        folder = args.input / category
        with (folder / 'comparison.csv').open(newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if int(r['threads']) == args.threads]
        if not rows:
            raise ValueError(f'No {args.threads}-thread results for {category}')
        trials = json.loads((folder / 'campaign.json').read_text())['trials']
        data[category] = rows, trials
        for row in rows:
            value = validated_speedup(row, trials) if row['method'] in methods else None
            if value is not None:
                all_speedups.append(value)
    # One shared log range makes sub-baseline and very large speedups readable.
    # Retain equal image dimensions to prevent different preview resizing.
    low = math.floor(math.log10(min(all_speedups) * 0.9))
    high = math.ceil(math.log10(max(all_speedups) * 1.1))
    ticks = [10.0 ** exponent for exponent in range(low, high + 1)]
    max_rows = max(len({(r['dataset'], r['benchmark']) for r in rows}) for rows, _ in data.values())
    figsize = (18, max(8, 0.62 * max_rows + 2.4))
    print(f'Shared logarithmic axis: {ticks[0]:g}× to {ticks[-1]:g}×; decade ticks', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        plot_category(*data[category], args.output, category, args.threads, ticks, figsize)
