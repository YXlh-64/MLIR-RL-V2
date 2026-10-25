#!/usr/bin/env python3
"""Merge V5.2 full-model matrix results into a summary.

Reads per-cell files written by eval_fullmodel.sh --label:
    <eval-dir>/checkpoint_<step>_<model>.json = {bench_name: exec_time_ns}

Also accepts unlabelled block-eval files (checkpoint_<step>.json) and merges
them as a pseudo-model entry, so one summary can cover both.

Usage:
    python scripts/eval/merge_fullmodel_matrix.py --eval-dir <agent>/eval
    python scripts/eval/merge_fullmodel_matrix.py --eval-dir <agent>/eval \\
        --baselines config/mlir_rl_v1_paper/eval/fullmodel_eval.json \\
        --output <agent>/eval/matrix_summary.json
"""
import argparse
import json
import os
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge full-model matrix results.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--baselines", default=None, help="Baseline {bench: ns} JSON for speedups")
    parser.add_argument("--output", default=None, help="Where to write summary JSON")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    baseline = json.load(open(args.baselines)) if args.baselines else {}

    per_checkpoint: dict[str, list] = {}
    for f in sorted(eval_dir.glob("checkpoint_*.json")):
        m = re.match(r"checkpoint_(\d+)(?:_(.+))?\.json$", f.name)
        if not m:
            continue
        step, model = m.group(1), m.group(2) or "block_eval"
        try:
            data = json.load(open(f))
        except json.JSONDecodeError:
            print(f"WARN: skipping unreadable {f.name}")
            continue
        entries = []
        for bench, exec_ns in data.items():
            base = baseline.get(bench)
            speedup = (base / exec_ns) if exec_ns and base else 0.0
            model_key = bench if (bench in baseline or model == "block_eval") else model
            entries.append({
                "model": model_key,
                "bench": bench,
                "speedup": speedup,
                "baseline_ns": base,
                "optimized_ns": exec_ns,
            })
        per_checkpoint.setdefault(step, []).extend(entries)

    summary = {"per_checkpoint": {}, "best_per_model": {}}
    best: dict[str, tuple] = {}
    for step in sorted(per_checkpoint, key=int):
        entries = per_checkpoint[step]
        ok = [e["speedup"] for e in entries if e["speedup"] > 0]
        summary["per_checkpoint"][step] = {
            "average_speedup": sum(ok) / len(ok) if ok else 0.0,
            "models_evaluated": len(entries),
            "models_succeeded": len(ok),
        }
        for e in entries:
            if e["speedup"] > best.get(e["model"], (None, 0.0))[1]:
                best[e["model"]] = (step, e["speedup"])
    summary["best_per_model"] = {
        model: {"checkpoint": step, "speedup": sp} for model, (step, sp) in best.items()
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.output, "w"), indent=2)
        print(f"Summary -> {args.output}")

    print(f"\n{'Checkpoint':<12} {'Avg Speedup':>12} {'Models':>8}")
    print("-" * 34)
    for step in sorted(summary["per_checkpoint"], key=int):
        cr = summary["per_checkpoint"][step]
        print(f"{step:<12} {cr['average_speedup']:>12.4f}x {cr['models_succeeded']:>3}/{cr['models_evaluated']:<3}")

    print(f"\n{'Model':<25} {'Best ckpt':<10} {'Speedup':>10}")
    print("-" * 47)
    for model, b in sorted(summary["best_per_model"].items()):
        print(f"{model:<25} {b['checkpoint'] or 'N/A':<10} {b['speedup']:>10.4f}x")


if __name__ == "__main__":
    main()
