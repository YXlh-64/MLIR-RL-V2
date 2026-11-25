"""Compare full-model eval results of several agents (written by scripts/fusion_data/eval_compare.sh).

    python scripts/fusion_data/plot_eval_compare.py <OUT_DIR>

Reads <OUT_DIR>/agents.tsv (label, config, checkpoint dir, step, result json) and the baseline times
(eval_json_file of each config). Speedup = baseline time / agent time; failures are shown, never dropped
silently. Writes comparison.csv, comparison.md and comparison.png in OUT_DIR.
Summary statistics are geometric means (and medians), over (a) the models that agent ran and
(b) the models every agent ran, so agents are compared on the same set.
"""
import csv
import json
import math
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TEXT, TEXT2, SURFACE = "#191919", "#666666", "#ffffff"
BLUE, RED, MID, FAIL = "#1b64be", "#c43b3c", "#efefef", "#d9d9d9"   # diverging blue <-> gray <-> red


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main(out):
    agents = []
    for line in open(os.path.join(out, "agents.tsv")):
        label, cfg, _, step, res = line.rstrip("\n").split("\t")
        base = json.load(open(json.load(open(cfg))["eval_json_file"]))
        times = json.load(open(res)) if os.path.exists(res) else {}
        agents.append((f"{label}\n@{step}", base, times))
    models = sorted(agents[0][1])
    sp = {a: {m: (b[m] / t[m] if t.get(m) and t[m] > 0 else None) for m in models} for a, b, t in agents}
    names = [a for a, _, _ in agents]
    common = [m for m in models if all(sp[a][m] for a in names)]

    rows = []
    for a in names:
        ok = [sp[a][m] for m in models if sp[a][m]]
        rows.append(dict(agent=a.replace("\n", " "), ran=len(ok), failed=len(models) - len(ok),
                         gmean=gmean(ok), median=statistics.median(ok) if ok else float("nan"),
                         gmean_common=gmean([sp[a][m] for m in common])))

    with open(os.path.join(out, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [n.replace("\n", " ") for n in names])
        for m in models:
            w.writerow([m] + [f"{sp[a][m]:.4f}" if sp[a][m] else "FAILED" for a in names])
    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write(f"# Full-model eval: speedup over baseline (higher is better)\n\n"
                f"Models every agent ran: {len(common)} / {len(models)}\n\n"
                "| agent | ran | failed | geomean | median | geomean (common set) |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['agent']} | {r['ran']} | {r['failed']} | {r['gmean']:.2f}x | {r['median']:.2f}x | "
                    f"{r['gmean_common']:.2f}x |\n")
        f.write("\n| model | " + " | ".join(n.replace("\n", " ") for n in names) + " |\n|---|"
                + "---|" * len(names) + "\n")
        for m in models:
            f.write(f"| {m.replace('_linalg', '')} | " + " | ".join(
                f"{sp[a][m]:.2f}x" if sp[a][m] else "FAILED" for a in names) + " |\n")

    # Figure: left = per-model heatmap (log2 speedup, diverging around 1x), right = geomean on the common set
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.edgecolor": "#cccccc"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.42 * len(models) + 2.2),
                                 gridspec_kw={"width_ratios": [len(names) * 1.1, 4]}, facecolor=SURFACE)
    vals = [[math.log2(sp[a][m]) if sp[a][m] else float("nan") for a in names] for m in models]
    lim = max([abs(v) for r in vals for v in r if not math.isnan(v)] + [1.0])
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE]); cmap.set_bad(FAIL)
    ax.imshow(vals, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, m in enumerate(models):
        for j, a in enumerate(names):
            s = sp[a][m]
            dark = s and abs(math.log2(s)) > 0.6 * lim
            ax.text(j, i, f"{s:.2f}x" if s else "fail", ha="center", va="center", fontsize=8,
                    color=(SURFACE if dark else (TEXT if s else TEXT2)))
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(models)), [m.replace("_linalg", "") for m in models])
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(names))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title("Speedup over baseline per model  (blue = faster, red = slower, gray = failed)",
                 loc="left", fontsize=10, pad=28)

    g = [r["gmean_common"] for r in rows]
    y = range(len(names))
    bx.barh(y, g, color=BLUE, height=0.55)
    bx.axvline(1, color=TEXT2, linewidth=1, linestyle=(0, (3, 3)))
    for yi, v in zip(y, g):
        if not math.isnan(v):
            bx.text(v, yi, f"  {v:.2f}x", va="center", fontsize=8, color=TEXT)
    bx.set_yticks(list(y), [n.replace("\n", " ") for n in names]); bx.invert_yaxis()
    bx.set_xlabel("geometric-mean speedup (1x = baseline)")
    bx.set_title(f"Geomean on the {len(common)} models every agent ran", loc="left", fontsize=10)
    for s in ("top", "right"): bx.spines[s].set_visible(False)
    bx.grid(axis="x", color="#eeeeee"); bx.set_axisbelow(True)
    bx.set_xlim(0, max([v for v in g if not math.isnan(v)] + [1]) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparison.png"), dpi=150)
    print(open(os.path.join(out, "comparison.md")).read())


if __name__ == "__main__":
    main(sys.argv[1])
