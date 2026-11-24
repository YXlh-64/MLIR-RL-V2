#!/usr/bin/env python3
"""Mean and median speedup vs mlir_baseline of all 4 methods, per operation type.

Each block is counted in every operation-type group it contains (a block with an add, a matmul
and a relu counts once in each of those three groups). Per block and method:
    speedup = mlir_baseline median latency / method median latency,
over blocks where both that method and mlir_baseline passed correctness.

Usage:
    python plot_op_types.py --tag ckpt_v5_model_1200 [--unique]
--unique uses the 1474 distinct computations once each; default uses all 6693 blocks
(duplicates carry the result of the block they were measured as, see map_duplicates.py).
Writes results/<tag>/figures/speedup_by_op_type[_unique].png and analysis/speedup_by_op_type[_unique].csv.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METHODS = ["mlir_baseline", "mlir_rl", "torch_eager", "torch_inductor"]
LABELS = {"mlir_baseline": "MLIR baseline", "mlir_rl": "MLIR-RL",
          "torch_eager": "Torch eager", "torch_inductor": "Torch Inductor"}
COLORS = {"mlir_baseline": "#2a78d6", "mlir_rl": "#eb6834",   # validated default categorical
          "torch_eager": "#1baf7a", "torch_inductor": "#eda100"}  # palette, slots 1-4

NAMED = {"matmul": "matmul", "batch_matmul": "matmul",
         "conv_2d_nchw_fchw": "conv2d", "depthwise_conv_2d_nchw_chw": "depthwise conv",
         "pooling_nchw_max": "pooling", "pooling_nchw_sum": "pooling",
         "transpose": "copy/transpose", "broadcast": "copy/transpose"}
BODY = {("arith.addf",): "add", ("arith.mulf",): "mul", ("arith.subf",): "sub",
        ("arith.divf",): "div", ("arith.cmpf", "arith.select"): "relu/max",
        ("arith.negf", "math.exp", "arith.addf", "arith.divf"): "sigmoid",
        ("math.rsqrt",): "rsqrt", ("math.sqrt",): "sqrt", ("math.powf",): "pow",
        ("math.fpowi",): "pow", ("math.erf",): "erf (gelu)", ("math.exp",): "exp",
        ("math.tanh",): "tanh", (): "copy/transpose",
        ("arith.truncf", "arith.addf"): "add", ("arith.truncf", "arith.mulf"): "mul"}


def op_type(o):
    if o["kind"] == "fill":
        return None                                  # initialisation, not an operation
    if o["kind"] in NAMED:
        return NAMED[o["kind"]]
    body = tuple(o.get("body_ops") or [])
    if "reduction" in (o.get("iterators") or []) and body == ("arith.addf",):
        return "sum (reduction)"
    return BODY.get(body, "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ckpt_v5_model_1200")
    ap.add_argument("--unique", action="store_true")
    a = ap.parse_args()
    res = HERE / "results" / a.tag
    cat = json.loads((HERE / "results/catalog/blocks_catalog.json").read_text())["blocks"]
    types = {b["name"]: {t for t in map(op_type, b["ops"]) if t} for b in cat}

    rows = list(csv.DictReader(open(res / "all_blocks_results.csv")))
    if a.unique:
        rows = [r for r in rows if r["block"] == r["measured_as"]]
    lat = defaultdict(dict)                          # block -> method -> median ns (passed only)
    for r in rows:
        if r["status"] == "measured" and r["correctness"] == "passed" and r["median_ns"]:
            lat[r["block"]][r["method"]] = float(r["median_ns"])
    blocks = {r["block"] for r in rows}

    speed = defaultdict(lambda: defaultdict(list))   # type -> method -> [speedups]
    nblocks = defaultdict(int)
    for b in blocks:
        for t in types[b]:
            nblocks[t] += 1
        base = lat[b].get("mlir_baseline")
        if not base:
            continue
        for m in METHODS:
            if m in lat[b]:
                for t in types[b]:
                    speed[t][m].append(base / lat[b][m])

    order = sorted(speed, key=lambda t: -len(speed[t]["mlir_baseline"]))
    suffix = "_unique" if a.unique else ""
    (res / "analysis").mkdir(exist_ok=True)
    with open(res / "analysis" / f"speedup_by_op_type{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_type", "blocks_containing", "method", "n", "mean_speedup", "median_speedup"])
        for t in order:
            for m in METHODS:
                v = speed[t][m]
                w.writerow([t, nblocks[t], m, len(v), f"{mean(v):.4g}" if v else "",
                            f"{median(v):.4g}" if v else ""])

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 1.15 * len(order)), 10), sharex=True)
    width = 0.8 / len(METHODS)
    xs = range(len(order))
    for ax, (stat, fn) in zip(axes, [("Mean", mean), ("Median", median)]):
        for i, m in enumerate(METHODS):
            vals = [fn(speed[t][m]) if speed[t][m] else float("nan") for t in order]
            ax.bar([x + (i - 1.5) * width for x in xs], vals, width * 0.9, color=COLORS[m],
                   label=LABELS[m], zorder=3)
        ax.set_yscale("log")
        ax.axhline(1, color="#52514e", lw=1, zorder=4)
        ax.grid(axis="y", which="major", color="#e4e3df", lw=0.8, zorder=0)
        ax.set_ylabel(f"{stat} speedup vs MLIR baseline (log)")
        ax.set_title(f"{stat} speedup per operation type", loc="left", fontsize=12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[1].set_xticks(list(xs))
    def tick(t):
        n, nrl = len(speed[t]["mlir_baseline"]), len(speed[t]["mlir_rl"])
        return f"{t}\nn={n}" + (f"\nRL n={nrl}" if nrl != n else "")
    axes[1].set_xticklabels([tick(t) for t in order], rotation=0, fontsize=9)
    what = "1474 distinct computations" if a.unique else "all 6693 blocks"
    fig.suptitle(f"{a.tag}: speedup vs MLIR baseline by operation type ({what}, 1 thread)\n"
                 "A block counts in every operation-type group it contains; >1 = faster than "
                 "MLIR baseline; only blocks where both methods passed correctness.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = res / "figures" / f"speedup_by_op_type{suffix}.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
