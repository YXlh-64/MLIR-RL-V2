"""A1: per full model, how many linalg ops have 0 / 1 / 2 / >=3 linalg producers, by op type?"""
import argparse
from collections import Counter, defaultdict

from _mlir_text import files, md_table, model_name, parse_file, write_csv, write_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('models', nargs='+', help='full-model .mlir files, dirs or globs')
ap.add_argument('--out', required=True)
args = ap.parse_args()

rows, unparsed = [], []
for f in files(args.models):
    model = model_name(f)
    ops, bad = parse_file(f)
    unparsed += [dict(model=model, op=n) for n in bad]
    counts = defaultdict(Counter)
    for op in ops:
        n = len({t for t, _ in op.producers})
        b = '>=3' if n >= 3 else str(n)
        counts[op.label][b] += 1
        counts['ALL'][b] += 1
    for label, c in sorted(counts.items()):
        total = sum(c.values())
        rows.append({'model': model, 'op_type': label, 'total': total,
                     **{k: c[k] for k in ('0', '1', '2', '>=3')},
                     'pct_ge2': (c['2'] + c['>=3']) / total})

write_csv(args.out, 'a1_producer_counts.csv', rows)
write_csv(args.out, 'a1_unparsed.csv', unparsed)
write_md(args.out, 'a1_summary.md', '# A1 producers per op\n\n'
         + md_table([r for r in rows if r['op_type'] == 'ALL'])
         + f'\nUnparsed op names: {sorted({u["op"] for u in unparsed}) or "none"}\n')
