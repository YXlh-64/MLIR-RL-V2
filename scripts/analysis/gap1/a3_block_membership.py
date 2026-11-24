"""A3: how many blocks does each original model op appear in (distribution, most repeated op types)?"""
import argparse
from collections import Counter, defaultdict
from statistics import mean

from _mlir_text import files, load_blocks, md_table, model_name, parse_file, write_csv, write_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--models', nargs='+', required=True, help='full-model .mlir files/dirs/globs (op types, unseen ops)')
ap.add_argument('--blocks', nargs='+', required=True, help='block manifests or block .mlir files/dirs/globs')
ap.add_argument('--top', type=int, default=20)
ap.add_argument('--out', required=True)
args = ap.parse_args()

label = {(model_name(f), op.tag): op.label for f in files(args.models) for op in parse_file(f)[0]}
models = {m for m, _ in label}
count = Counter()
for b in load_blocks(args.blocks):
    count.update((b['model'], t) for t in set(b['tags']))

rows = [dict(model=m, tag=t, op_type=lab, n_blocks=count[(m, t)]) for (m, t), lab in sorted(label.items())]
missing = sorted(k for k in count if k not in label and k[0] in models)  # tags in blocks but not in model

dist = Counter((r['model'], r['n_blocks']) for r in rows) + Counter(('ALL', r['n_blocks']) for r in rows)
drows = [dict(model=m, n_blocks=n, n_ops=c) for (m, n), c in sorted(dist.items())]
by_type = defaultdict(list)
for r in rows:
    by_type[r['op_type']].append(r['n_blocks'])
trows = sorted((dict(op_type=k, n_ops=len(v), total_appearances=sum(v), mean_per_op=mean(v), max_per_op=max(v),
                     pct_unseen=sum(x == 0 for x in v) / len(v)) for k, v in by_type.items()),
               key=lambda r: -r['total_appearances'])

write_csv(args.out, 'a3_ops.csv', rows)
write_csv(args.out, 'a3_distribution.csv', drows)
write_csv(args.out, 'a3_op_types.csv', trows)
write_md(args.out, 'a3_summary.md', '# A3 block membership per model op\n\n'
         + '## Distribution (all models)\n\n' + md_table([r for r in drows if r['model'] == 'ALL'])
         + f'\n## Top {args.top} op types by appearances\n\n' + md_table(trows[:args.top])
         + f'\nBlock tags not found in the parsed model: {len(missing)}\n')
