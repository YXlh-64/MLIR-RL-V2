"""A6: how do operand shapes of block ops differ from the same (tagged) ops in the source models?"""
import argparse
import math
from collections import Counter, defaultdict
from statistics import median

from _mlir_text import files, md_table, model_name, parse_file, shape, write_csv, write_md


def elems(types):
    n = 0
    for t in types:
        s = shape(t)
        if s is not None and '?' not in s:
            n += math.prod(int(d) for d in s)
    return n


def drift(bt, mt):
    """'same', 'dynamic_specialized' (model has '?'), 'dim0' (only leading dim differs), 'other'."""
    if bt == mt:
        return 'same'
    bs, ms = [shape(t) for t in bt], [shape(t) for t in mt]
    if len(bs) != len(ms) or any(a is None or b is None or len(a) != len(b) for a, b in zip(bs, ms) if a != b):
        return 'other'
    diffs = [(y, j) for a, b in zip(bs, ms) if a is not None for j, (x, y) in enumerate(zip(a, b)) if x != y]
    if not diffs:
        return 'other'  # same shapes, different element type / layout
    if all(y == '?' for y, _ in diffs):
        return 'dynamic_specialized'
    return 'dim0' if all(j == 0 for _, j in diffs) else 'other'


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--models', nargs='+', required=True, help='full-model .mlir files/dirs/globs')
ap.add_argument('--blocks', nargs='+', required=True, help='block .mlir files/dirs/globs (with tag attributes)')
ap.add_argument('--out', required=True)
args = ap.parse_args()

model_ops = {(model_name(f), o.tag): o for f in files(args.models) for o in parse_file(f)[0]}
models = {m for m, _ in model_ops}
rows = []
for f in files(args.blocks):
    m = model_name(f)
    if m not in models:
        continue
    for o in parse_file(f, tags='attr')[0]:
        src = model_ops.get((m, o.tag))
        if src is None:
            rows.append(dict(model=m, block=f, tag=o.tag, block_op=o.name, model_op='', drift='tag_missing'))
            continue
        bt, mt = o.ins_types + o.outs_types, src.ins_types + src.outs_types
        e_b, e_m = elems(bt), elems(mt)
        rows.append(dict(model=m, block=f, tag=o.tag, block_op=o.name, model_op=src.name,
                         same_op=o.name == src.name, drift=drift(bt, mt), elems_ratio=e_b / e_m if e_m else '',
                         block_types=' '.join(bt), model_types=' '.join(mt)))

c = defaultdict(Counter)
ratios = defaultdict(list)
for r in rows:
    for k in (r['model'], 'ALL'):
        c[k][r['drift']] += 1
        c[k]['op_name_mismatch'] += r.get('same_op') is False
        if r.get('elems_ratio') not in ('', None):
            ratios[k].append(r['elems_ratio'])
keys = ['same', 'dim0', 'dynamic_specialized', 'other', 'tag_missing', 'op_name_mismatch']
srows = [dict(model=k, n=sum(v[x] for x in keys[:5]), **{x: v[x] for x in keys},
              median_elems_ratio=median(ratios[k]) if ratios[k] else '') for k, v in sorted(c.items())]

write_csv(args.out, 'a6_ops.csv', rows)
write_csv(args.out, 'a6_summary.csv', srows)
write_md(args.out, 'a6_summary.md', '# A6 block vs model shapes\n\n' + md_table(srows))
