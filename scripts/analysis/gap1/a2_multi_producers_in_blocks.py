"""A2: of the model ops with >=2 producers, what % keep >=2 producers inside at least one of their blocks?"""
import argparse
from collections import defaultdict

from _mlir_text import files, load_blocks, md_table, model_name, parse_file, write_csv, write_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--models', nargs='+', required=True, help='full-model .mlir files/dirs/globs')
ap.add_argument('--blocks', nargs='+', required=True,
                help='block manifests (*_blocks_manifest.json) or block .mlir files/dirs/globs')
ap.add_argument('--out', required=True)
args = ap.parse_args()

multi = {}  # (model, tag) -> (op, n_producers)
for f in files(args.models):
    for op in parse_file(f)[0]:
        n = len({t for t, _ in op.producers})
        if n >= 2:
            multi[(model_name(f), op.tag)] = (op, n)

n_blocks, max_kept = defaultdict(int), defaultdict(int)
for b in load_blocks(args.blocks):
    for tag in b['tags']:
        key = (b['model'], tag)
        if key in multi:
            n_blocks[key] += 1
            kept = len({p for p, c in b['edges'] if c == tag})
            max_kept[key] = max(max_kept[key], kept)

rows = [dict(model=m, tag=t, op_type=op.label, n_producers=n, n_blocks=n_blocks[(m, t)],
             max_kept_in_block=max_kept[(m, t)]) for (m, t), (op, n) in sorted(multi.items())]
summary = defaultdict(lambda: dict(multi=0, in_blocks=0, keep_ge2=0))
for r in rows:
    for key in (r['model'], 'ALL'):
        s = summary[key]
        s['multi'] += 1
        s['in_blocks'] += r['n_blocks'] > 0
        s['keep_ge2'] += r['max_kept_in_block'] >= 2
srows = [dict(model=k, **v, pct_keep_ge2=v['keep_ge2'] / v['multi']) for k, v in sorted(summary.items())]

write_csv(args.out, 'a2_ops.csv', rows)
write_csv(args.out, 'a2_summary.csv', srows)
write_md(args.out, 'a2_summary.md', '# A2 multi-producer ops kept inside blocks\n\n' + md_table(srows))
