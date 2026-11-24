"""B2: are TPF (tiled fusion) schedules correct? Replay them on random inputs and compare outputs to the untransformed program."""
import argparse
import json
import os
import re

from _run import compare, run
from _mlir_text import md_table, write_csv, write_md
from rl_autoschedular_v5.actions import ActionSpace
from rl_autoschedular_v5.env import Env
from rl_autoschedular_v5.state import extract_bench_features_from_file

ACTION = re.compile(r'[A-Z]+\([^)]*\)')


def replay(features, key):
    """Rebuild action objects from an exec-cache key ('|'-joined per-op action strings, first op first)."""
    env = Env()
    state = env.reset([features], 0)
    for seg in reversed(key.split('|')):  # v5 schedules the last op first
        for a in ACTION.findall(seg):
            state = env.step(state, ActionSpace.action_from_str(state, a))
            if state.terminal:
                break
        nxt = env.get_next_op_state(state)
        if nxt is None:
            break
        state = nxt
    return state.transformation_history


def apply(code, history):
    """Apply in v5 order (lenient, like Env): on failure skip that op's remaining actions."""
    from mlir.ir import Context, Module
    failures = []
    with Context():
        module = Module.parse(code)
        for op_seq in reversed(history):
            for action in op_seq:
                try:
                    action.apply(module)
                except Exception as e:  # noqa: BLE001
                    failures.append(f'{action}: {str(e)[:120]}')
                    break
        return str(module), failures


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--exec-data', nargs='+', required=True,
                help='exec_data.json from training/eval, or the JSON written by gen_schedules.py')
ap.add_argument('--benchmarks-dir', required=True, help='folder with <bench>.mlir')
ap.add_argument('--per-bench', type=int, default=1, help='fastest N TPF schedules per bench')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--rtol', type=float, default=1e-3)
ap.add_argument('--atol', type=float, default=1e-4)
ap.add_argument('--out', required=True)
args = ap.parse_args()

scheds = {}
for f in args.exec_data:
    for bench, d in json.load(open(f)).items():
        scheds.setdefault(bench, {}).update({k: v for k, v in d.items() if 'TPF(' in k and v is not None and v >= 0})  # 0 = from gen_schedules.py

rows = []
for bench, d in sorted(scheds.items()):
    path = os.path.join(args.benchmarks_dir, bench + '.mlir')
    if not d or not os.path.exists(path):
        continue
    feats = extract_bench_features_from_file(bench, path, 1)
    ref = None
    for key in sorted(d, key=d.get)[:args.per_bench]:
        row = dict(bench=bench, schedule=key, n_tpf=key.count('TPF('), cached_time_ns=d[key])
        try:
            code, fails = apply(feats.code, replay(feats, key))
            if ref is None:
                ref = run(feats.code, 1, args.seed)[1]
            ok, err = compare(ref, run(code, 1, args.seed)[1], args.rtol, args.atol)
            row.update(transform_failures=len(fails), allclose=ok, max_abs_err=err, failures=' ; '.join(fails))
        except Exception as e:  # noqa: BLE001
            row.update(error=str(e)[:200])
        rows.append(row)

done = [r for r in rows if 'allclose' in r]
summary = [dict(schedules=len(rows), replayed=len(done), allclose=sum(r['allclose'] for r in done),
                wrong=sum(not r['allclose'] for r in done), with_transform_failures=sum(r['transform_failures'] > 0 for r in done))]
write_csv(args.out, 'b2_tpf.csv', rows)
write_md(args.out, 'b2_summary.md', '# B2 TPF schedule correctness\n\n' + md_table(summary))
