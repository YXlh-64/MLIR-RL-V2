"""A5: how many producer->consumer edges are green (fuse now), red (never fuse) or yellow (left to RL), and why?"""
import argparse
from collections import Counter

from _mlir_text import classify, edges, files, md_table, model_name, parse_file, write_csv, write_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('models', nargs='+', help='full-model (or block) .mlir files, dirs or globs')
ap.add_argument('--out', required=True)
args = ap.parse_args()

rows, unparsed = [], []
for f in files(args.models):
    ops, bad = parse_file(f)
    unparsed += [dict(model=model_name(f), op=n) for n in bad]
    for p, c, k in edges(ops):
        cls, why = classify(p, c, k)
        rows.append(dict(model=model_name(f), producer=p.tag, consumer=c.tag, operand=k,
                         producer_type=p.label, consumer_type=c.label, cls=cls, reason=why))

by_cls = Counter((r['cls'], r['reason']) for r in rows)
crows = [dict(cls=c, reason=w, edges=n, pct=n / len(rows)) for (c, w), n in by_cls.most_common()] if rows else []
pairs = Counter((r['producer_type'], r['consumer_type'], r['cls'], r['reason']) for r in rows)
prows = [dict(producer_type=a, consumer_type=b, cls=c, reason=w, edges=n) for (a, b, c, w), n in pairs.most_common()]
per_model = Counter((r['model'], r['cls']) for r in rows)
mrows = [dict(model=m, **{c: per_model[(m, c)] for c in ('green', 'yellow', 'red')})
         for m in sorted({r['model'] for r in rows})]

write_csv(args.out, 'a5_edges.csv', rows)
write_csv(args.out, 'a5_pairs.csv', prows)
write_csv(args.out, 'a5_unparsed.csv', unparsed)
write_md(args.out, 'a5_summary.md', '# A5 edge classes\n\n' + md_table(crows) + '\n## Per model\n\n' + md_table(mrows)
         + '\n## Top op-type pairs\n\n' + md_table(prows[:30])
         + f'\nUnparsed op names: {sorted({u["op"] for u in unparsed}) or "none"}\n')
