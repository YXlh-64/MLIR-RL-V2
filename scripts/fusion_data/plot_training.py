"""Training curves of the A-D+ rows from their logs (no job needed; light, runs anywhere).

    python scripts/fusion_data/plot_training.py [OUT_DIR]     (default /scratch/mh8826/fusion_runs/plots)

Reads <run>/logs/train/final_speedup (one value per episode, bench_count=64 episodes per iteration).
Per iteration: geometric-mean speedup and share of episodes faster than the baseline, smoothed over
20 iterations. Writes training_curves.png and training_summary.md.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/scratch/mh8826/fusion_runs"
ROWS = ["A", "A_prime", "C", "D", "D_plus"]
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]   # categorical slots 1-5, fixed order
TEXT, TEXT2 = "#191919", "#666666"
PER_ITER, SMOOTH = 64, 20


def per_iteration(path):
    v = [float(x) for x in open(path).read().split()]
    gm, win = [], []
    for i in range(0, len(v) - PER_ITER + 1, PER_ITER):
        it = [x for x in v[i:i + PER_ITER] if x > 0]
        gm.append(math.exp(sum(map(math.log, it)) / len(it)) if it else float("nan"))
        win.append(sum(x > 1 for x in v[i:i + PER_ITER]) / PER_ITER)
    return gm, win


def smooth(xs, k=SMOOTH):
    out = []
    for i in range(len(xs)):
        w = [x for x in xs[max(0, i - k + 1):i + 1] if not math.isnan(x)]
        out.append(sum(w) / len(w) if w else float("nan"))
    return out


def smooth_log(xs, k=SMOOTH):
    return [math.exp(x) for x in smooth([math.log(x) if x > 0 else float("nan") for x in xs], k)]


def main(out):
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.edgecolor": "#cccccc"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
    lines = ["| row | iterations | geomean speedup, first 50 it. | geomean speedup, last 50 it. | "
             "episodes faster than baseline, last 50 it. |", "|---|---|---|---|---|"]
    for row, c in zip(ROWS, COLORS):
        p = f"{RUNS}/fusion_{row}_agent/logs/train/final_speedup"
        if not os.path.exists(p):
            continue
        gm, win = per_iteration(p)
        x = range(1, len(gm) + 1)
        s, w = smooth_log(gm), smooth(win)
        ax.plot(x, s, color=c, linewidth=2, label=row)
        bx.plot(x, [100 * y for y in w], color=c, linewidth=2, label=row)
        for a, yy in ((ax, s[-1]), (bx, 100 * w[-1])):
            a.annotate(f" {row}", (len(gm), yy), color=TEXT, fontsize=8, va="center")
        first = math.exp(sum(math.log(g) for g in gm[:50]) / len(gm[:50]))
        last = math.exp(sum(math.log(g) for g in gm[-50:]) / len(gm[-50:]))
        lines.append(f"| {row} | {len(gm)} | {first:.2f}x | {last:.2f}x | {100 * sum(win[-50:]) / len(win[-50:]):.0f}% |")
    ax.set_yscale("log", base=2)
    ax.axhline(1, color=TEXT2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title(f"Training speedup over baseline (geomean per iteration, {SMOOTH}-iteration average)",
                 loc="left", fontsize=10)
    ax.set_ylabel("speedup (log scale, 1x = baseline)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}x"))
    bx.axhline(50, color=TEXT2, linewidth=1, linestyle=(0, (3, 3)))
    bx.set_ylim(0, 100)
    bx.set_title("Episodes faster than the baseline (%)", loc="left", fontsize=10)
    for a in (ax, bx):
        a.set_xlabel("training iteration")
        a.grid(color="#eeeeee"); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/training_curves.png", dpi=150)
    open(f"{out}/training_summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{RUNS}/plots")
