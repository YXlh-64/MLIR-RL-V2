#!/usr/bin/env python3
"""Figures for experiment 02. Reads result tables only; runs no benchmarks.

Inputs (all under results/<tag>/, any missing input just skips its figures):
  analysis/per_block.csv, per_model.csv, per_size.csv, headline_table.csv   (stats.py)
  analysis/interaction.csv, context.csv, composition.csv, cache.csv         (ablation.py)

Figures (results/<tag>/figures/), each answering one question:
  F1 speedup_distribution.png  How are per-block speedups vs the MLIR baseline distributed, per class?
  F2 win_tie_loss.png          How often is each method significantly faster / slower?
  F3 speedup_vs_size.png       Does the RL policy get worse as blocks get larger?
  F4 per_model.png             Which models' blocks benefit (geomean + 95% CI)?
  F5 rl_vs_inductor.png        Per block, who wins: MLIR RL or TorchInductor?
  F6 roofline.png              Achieved GFLOP/s vs arithmetic intensity: memory- or compute-bound?
  F7 threads.png               Does the ranking change with thread count?
  F8 d1_interaction.png        Do op schedules inside a block help each other (synergy) or interfere?
  F9 d2_context.png            Does the policy schedule the same op the same way in different blocks?
  F10 d3_composition.png       Does a block's speedup follow from the speedups of its parts?
  F11 d4_cache.png             How sensitive is each method to cold caches?
The per-class bar chart of experiment 01 is still available with its own script:
  python experiments/01_performance_diagnosis/plot_speedups.py --results-dir results/<tag>/t1_warm/r0
(paths relative to experiments/02_block_diagnosis).
"""
import argparse
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES = (('mlir_baseline', 'MLIR baseline', '#8A929E'),     # same colours as experiment 01
          ('mlir_rl', 'MLIR RL', '#0072B2'),
          ('torch_eager', 'PyTorch eager', '#D55E00'),
          ('torch_inductor', 'TorchInductor', '#009E73'))
LABEL = {m: l for m, l, _ in SERIES}
COLOR = {m: c for m, _, c in SERIES}
CANDIDATES = ('mlir_rl', 'torch_eager', 'torch_inductor')


def read(path):
    if not path.exists():
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def log_axis(ax, axis='y'):
    getattr(ax, f'set_{axis}scale')('log')
    (ax.axhline if axis == 'y' else ax.axvline)(1, color='#444', lw=0.8, ls='--')


def save(fig, out, name, made):
    fig.tight_layout()
    fig.savefig(out / name, dpi=170, facecolor='white')
    made.append(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='main')
    p.add_argument('--config', default='t1_warm', help='Config for single-config figures')
    p.add_argument('--peak-gflops', type=float, help='Roofline: peak compute of the allocated cores')
    p.add_argument('--peak-gbs', type=float, help='Roofline: peak memory bandwidth (GB/s)')
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    root = HERE / 'results' / args.tag
    analysis, out = root / 'analysis', root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    made = []

    per_block = read(analysis / 'per_block.csv') or []
    pb = [r for r in per_block if r['config'] == args.config]
    classes = sorted({r['class'] for r in pb})

    # F1 distribution per class and method
    if pb:
        fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(classes)), 4.8))
        width = 0.25
        for j, m in enumerate(CANDIDATES):
            data = [[num(r['speedup']) for r in pb if r['class'] == c and r['method'] == m and num(r['speedup'])]
                    for c in classes]
            pos = [i + (j - 1) * width for i in range(len(classes))]
            if any(data):
                parts = ax.violinplot([d or [math.nan] for d in data], positions=pos, widths=width * 0.9,
                                      showmedians=True)
                for body in parts['bodies']:
                    body.set_facecolor(COLOR[m])
                    body.set_alpha(0.45)
            ax.plot([], [], color=COLOR[m], lw=6, alpha=0.5, label=LABEL[m])
        ax.set_xticks(range(len(classes)), classes)
        log_axis(ax)
        ax.set_ylabel('Speedup vs MLIR baseline (log)')
        ax.set_title(f'F1 Per-block speedup distribution ({args.config}, validated only)')
        ax.legend(frameon=False)
        save(fig, out, 'speedup_distribution.png', made)

        # F2 win / tie / loss
        fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(classes)), 4.2))
        for j, m in enumerate(CANDIDATES):
            for i, c in enumerate(classes):
                rows = [r for r in pb if r['class'] == c and r['method'] == m]
                bottom = 0
                for verdict, alpha in (('win', 1.0), ('tie', 0.35), ('loss', 0.1)):
                    n = sum(r['verdict'] == verdict for r in rows)
                    ax.bar(i + (j - 1) * 0.27, n, 0.25, bottom=bottom, color=COLOR[m], alpha=alpha,
                           edgecolor=COLOR[m])
                    bottom += n
            ax.plot([], [], color=COLOR[m], lw=6, label=LABEL[m])
        ax.set_xticks(range(len(classes)), classes)
        ax.set_ylabel('Blocks (dark = win, light = tie, outline = loss)')
        ax.set_title('F2 Significant wins / ties / losses vs MLIR baseline (95% CI, 5% threshold)')
        ax.legend(frameon=False)
        save(fig, out, 'win_tie_loss.png', made)

        # F5 RL vs Inductor
        t = {(r['block'], r['method']): num(r['median_ns']) for r in pb}
        pts = [(t[(b, 'torch_inductor')], t[(b, 'mlir_rl')], c) for (b, m), _ in t.items() if m == 'mlir_rl'
               for c in [next(r['class'] for r in pb if r['block'] == b)]
               if t.get((b, 'torch_inductor')) and t.get((b, 'mlir_rl'))]
        if pts:
            fig, ax = plt.subplots(figsize=(5.6, 5.4))
            markers = dict(zip(classes, 'osD^v'))
            for c in classes:
                xs = [p[0] / 1e3 for p in pts if p[2] == c]
                ys = [p[1] / 1e3 for p in pts if p[2] == c]
                ax.scatter(xs, ys, s=18, marker=markers.get(c, 'o'), label=c, alpha=0.75)
            lo = min(min(p[0], p[1]) for p in pts) / 1e3 * 0.8
            hi = max(max(p[0], p[1]) for p in pts) / 1e3 * 1.25
            ax.plot([lo, hi], [lo, hi], color='#444', lw=0.8, ls='--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('TorchInductor median µs')
            ax.set_ylabel('MLIR RL median µs')
            ax.set_title('F5 Below the diagonal: MLIR RL faster')
            ax.legend(frameon=False, fontsize=8)
            save(fig, out, 'rl_vs_inductor.png', made)

        # F6 roofline
        fig, ax = plt.subplots(figsize=(6.4, 5))
        for m, label, color in SERIES:
            xs = [num(r['arithmetic_intensity']) for r in pb if r['method'] == m]
            if m == 'mlir_baseline':
                ys = [num(r['total_flops']) / num(r['median_ns']) if num(r['total_flops']) and num(r['median_ns'])
                      else None for r in pb if r['method'] == m]
            else:
                ys = [num(r['gflops']) for r in pb if r['method'] == m]
            pts2 = [(x, y) for x, y in zip(xs, ys) if x and y]
            if pts2:
                ax.scatter(*zip(*pts2), s=14, alpha=0.6, color=color, label=label)
        if args.peak_gflops and args.peak_gbs:
            xs = [10 ** (k / 10) for k in range(-30, 31)]
            ax.plot(xs, [min(args.peak_gflops, args.peak_gbs * x) for x in xs], color='black', lw=1, label='Roofline')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Arithmetic intensity (FLOP / ABI byte, estimate)')
        ax.set_ylabel('Achieved GFLOP/s')
        ax.set_title('F6 Roofline view')
        ax.legend(frameon=False, fontsize=8)
        save(fig, out, 'roofline.png', made)

    # F3 size trend and F4 per model (from stats.py tables)
    for fname, key, title, fig_name in (('per_size.csv', 'size_bucket', 'F3 Speedup vs block size (ops)', 'speedup_vs_size.png'),
                                        ('per_model.csv', 'model', 'F4 Speedup per model', 'per_model.png')):
        rows = [r for r in (read(analysis / fname) or []) if r['config'] == args.config
                and r['comparison'].endswith('vs mlir_baseline')]
        if not rows:
            continue
        groups = sorted({r['group'] for r in rows}, key=lambda g: (g.split('=')[1].replace('7+', '7'), g))
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(groups) + 3), 4.6))
        for j, m in enumerate(CANDIDATES):
            sel = {r['group']: r for r in rows if r['comparison'].startswith(m)}
            xs = [i + (j - 1) * 0.25 for i, g in enumerate(groups) if g in sel]
            gm = [num(sel[g]['geomean_ratio']) for g in groups if g in sel]
            err = [[max(0, (y or 0) - (num(sel[g]['ci_low']) or y or 0)) for g, y in zip([g for g in groups if g in sel], gm)],
                   [max(0, (num(sel[g]['ci_high']) or y or 0) - (y or 0)) for g, y in zip([g for g in groups if g in sel], gm)]]
            ax.errorbar(xs, gm, yerr=err, fmt='o', color=COLOR[m], label=LABEL[m], capsize=3)
        ax.set_xticks(range(len(groups)), [g.split('=')[1] for g in groups], rotation=45 if key == 'model' else 0,
                      ha='right' if key == 'model' else 'center')
        log_axis(ax)
        ax.set_ylabel('Geomean speedup vs MLIR baseline [95% CI]')
        ax.set_title(f'{title} ({args.config})')
        ax.legend(frameon=False)
        save(fig, out, fig_name, made)

    # F7 thread scaling (overall group, warm cache)
    head = read(analysis / 'headline_table.csv') or []
    configs = sorted({r['config'] for r in head if r['config'].endswith('_warm')},
                     key=lambda c: int(c.split('_')[0][1:]))
    if len(configs) > 1:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        for m in CANDIDATES:
            pts3 = [(int(c.split('_')[0][1:]), num(r['geomean_ratio']), num(r['ci_low']), num(r['ci_high']))
                    for c in configs for r in head
                    if r['config'] == c and r['group'] == 'all' and r['comparison'] == f'{m} vs mlir_baseline']
            if pts3:
                xs, ys, lo, hi = zip(*pts3)
                ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                            marker='o', color=COLOR[m], label=LABEL[m], capsize=3)
        ax.set_xscale('log', base=2)
        log_axis(ax)
        ax.set_xlabel('Threads')
        ax.set_ylabel('Geomean speedup vs MLIR baseline (same threads)')
        ax.set_title('F7 Thread scaling')
        ax.legend(frameon=False)
        save(fig, out, 'threads.png', made)

    # F8 D1 interaction
    inter = [r for r in (read(analysis / 'interaction.csv') or []) if r.get('status') == 'ok']
    if inter:
        inter.sort(key=lambda r: num(r['interaction_index']))
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.18 * len(inter) + 1.5)))
        ax = axes[0]
        ys = range(len(inter))
        xs = [num(r['interaction_index']) for r in inter]
        ax.errorbar(xs, ys, xerr=[[x - num(r['ci_low']) for x, r in zip(xs, inter)],
                                  [num(r['ci_high']) - x for x, r in zip(xs, inter)]],
                    fmt='o', color=COLOR['mlir_rl'], ms=3, capsize=2)
        ax.axvline(0, color='#444', ls='--', lw=0.8)
        ax.set_yticks(list(ys), [r['block'] for r in inter], fontsize=6)
        ax.set_xlabel('Interaction index  (>0 synergy, <0 interference)')
        ax.set_title('F8a Whole schedule vs sum of per-op parts')
        ax = axes[1]
        fx = [(num(r['interaction_index']), num(r.get('fusion_log_gain'))) for r in inter if num(r.get('fusion_log_gain')) is not None]
        if fx:
            ax.scatter(*zip(*fx), color=COLOR['mlir_rl'], s=16)
            ax.axhline(0, color='#444', ls='--', lw=0.8)
            ax.axvline(0, color='#444', ls='--', lw=0.8)
        ax.set_xlabel('Interaction index')
        ax.set_ylabel('log(time without fusion / time with fusion)')
        ax.set_title('F8b Is the synergy explained by fusion?')
        save(fig, out, 'd1_interaction.png', made)

    # F9 D2 context
    ctx = read(analysis / 'context.csv') or []
    if ctx:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([num(r['pairwise_agreement']) for r in ctx], bins=[i / 10 for i in range(11)],
                color=COLOR['mlir_rl'], edgecolor='white')
        ax.set_xlabel('Pairwise schedule agreement for the same model op across blocks')
        ax.set_ylabel('Model ops')
        ax.set_title('F9 1.0 = schedule independent of the surrounding block')
        save(fig, out, 'd2_context.png', made)

    # F10 D3 composition
    comp = [r for r in (read(analysis / 'composition.csv') or []) if r['config'].startswith(args.config)]
    if comp:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        ax = axes[0]
        for m in CANDIDATES:
            pts4 = [(num(r['speedup_predicted']), num(r['speedup_measured'])) for r in comp if r['method'] == m]
            if pts4:
                ax.scatter(*zip(*pts4), color=COLOR[m], s=18, label=LABEL[m], alpha=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        lims = ax.get_xlim()
        ax.plot(lims, lims, color='#444', ls='--', lw=0.8)
        ax.set_xlabel('Predicted from parts: (B_Y+B_Z)/(T_Y+T_Z)')
        ax.set_ylabel('Measured on whole block: B_X/T_X')
        ax.set_title('F10a Does block speedup compose?')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1]
        data = [[num(r['composition_gain']) for r in comp if r['method'] == m] for m, _, _ in SERIES]
        ax.boxplot([d or [math.nan] for d in data], labels=[l for _, l, _ in SERIES])
        log_axis(ax)
        ax.set_ylabel('(T_Y + T_Z) / T_X   (>1: whole block faster than its parts)')
        ax.set_title('F10b Cross-block benefit per method')
        save(fig, out, 'd3_composition.png', made)

    # F11 D4 cache
    cache = read(analysis / 'cache.csv') or []
    if cache:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        data = [[num(r['cold_over_warm']) for r in cache if r['method'] == m] for m, _, _ in SERIES]
        axes[0].boxplot([d or [math.nan] for d in data], labels=[l for _, l, _ in SERIES])
        log_axis(axes[0])
        axes[0].set_ylabel('Cold / warm median time')
        axes[0].set_title('F11a Cache sensitivity')
        for m in CANDIDATES:
            pts5 = [(num(r['speedup_warm']), num(r['speedup_cold'])) for r in cache if r['method'] == m]
            if pts5:
                axes[1].scatter(*zip(*pts5), color=COLOR[m], s=16, label=LABEL[m], alpha=0.8)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        lims = axes[1].get_xlim()
        axes[1].plot(lims, lims, color='#444', ls='--', lw=0.8)
        axes[1].set_xlabel('Speedup vs baseline, warm caches')
        axes[1].set_ylabel('Speedup vs baseline, cold caches')
        axes[1].set_title('F11b Does the advantage survive cold caches?')
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, out, 'd4_cache.png', made)

    print(f'Wrote {len(made)} figures to {out}: {", ".join(made) or "none (no inputs yet)"}')


if __name__ == '__main__':
    main()
