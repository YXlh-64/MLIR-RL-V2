"""B4: what does graph-level elementwise fusion alone give (op count, default-lowering time, same outputs)?"""
import argparse
import os
from statistics import median

from _run import compare, prepare, run
from _mlir_text import files, classify, edges, md_table, parse_text, write_csv, write_md
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

KEEP = '"gap1.keep"'


def block_non_green(code):
    """Give producers of non-green edges a dummy extra use: the pass's default control only fuses
    single-use producers, so these edges are left alone. Returns (code, n_blocked)."""
    ops = parse_text(code)[0]
    lines = code.splitlines()
    blocked = {p.tag: p for p, c, k in edges(ops) if p.kind == 'ew' and classify(p, c, k)[0] != 'green'
               and p.res and p.nres == 1}
    for p in sorted(blocked.values(), key=lambda o: -o.end):
        lines.insert(p.end + 1, f'    {KEEP}({p.res}) : ({p.outs_types[0]}) -> ()')
    return '\n'.join(lines), len(blocked)


def fuse(code, green_only):
    n_blocked = 0
    if green_only:
        code, n_blocked = block_non_green(code)
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(code)
        PassManager.parse('builtin.module(func.func(linalg-fuse-elementwise-ops))', ctx).run(module.operation)
        out = str(module)
    return '\n'.join(ln for ln in out.splitlines() if KEEP not in ln), n_blocked


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('programs', nargs='+', help='model or block .mlir files or globs')
ap.add_argument('--green-only', action='store_true', help='only fuse edges A5 classifies green')
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--rtol', type=float, default=1e-3)
ap.add_argument('--atol', type=float, default=1e-4)
ap.add_argument('--out', required=True)
args = ap.parse_args()

rows = []
for f in files(args.programs):
    row = dict(program=os.path.basename(f))
    try:
        code = prepare(open(f).read())
        fused, n_blocked = fuse(code, args.green_only)
        row.update(ops_before=len(parse_text(code)[0]), ops_after=len(parse_text(fused)[0]), blocked_producers=n_blocked)
        t0, r0 = run(code, args.runs, args.seed)
        t1, r1 = run(fused, args.runs, args.seed)
        ok, err = compare(r0, r1, args.rtol, args.atol)
        row.update(base_ns=median(t0), fused_ns=median(t1), speedup=median(t0) / median(t1), allclose=ok, max_abs_err=err)
    except Exception as e:  # noqa: BLE001
        row.update(error=str(e)[:200])
    rows.append(row)

ok = [r for r in rows if 'speedup' in r]
summary = [dict(programs=len(rows), measured=len(ok), wrong_outputs=sum(not r['allclose'] for r in ok),
                median_speedup=median([r['speedup'] for r in ok]) if ok else '',
                ops_before=sum(r['ops_before'] for r in ok), ops_after=sum(r['ops_after'] for r in ok))]
write_csv(args.out, 'b4_prefusion.csv', rows)
write_md(args.out, 'b4_summary.md', f'# B4 elementwise pre-fusion (green_only={args.green_only})\n\n' + md_table(summary))
