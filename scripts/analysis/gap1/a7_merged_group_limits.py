"""A7: would green-merged op groups exceed the feature limits (loads/stores, loops) or use arith/math ops the features ignore?"""
import argparse
import json
import re
from collections import Counter, defaultdict

from _mlir_text import classify, edges, files, md_table, model_name, parse_file, read_text, write_csv, write_md

SEEN = {'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'math.exp'}  # ops counted by the AST dumper

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('models', nargs='+', help='full-model .mlir files, dirs or globs')
ap.add_argument('--config', required=True, help='v5 config JSON (max_num_stores_loads, max_num_loops, ...)')
ap.add_argument('--out', required=True)
args = ap.parse_args()
cfg = json.load(open(args.config))
LS, L = cfg['max_num_stores_loads'], cfg['max_num_loops']

rows, unseen = [], Counter()
for f in files(args.models):
    ops = parse_file(f)[0]
    text = read_text(f)
    parent = {o.tag: o.tag for o in ops}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for p, c, k in edges(ops):
        if classify(p, c, k)[0] == 'green':
            parent[find(p.tag)] = find(c.tag)
    groups = defaultdict(list)
    for o in ops:
        groups[find(o.tag)].append(o)

    for gid, members in groups.items():
        if len(members) < 2:
            continue
        internal = {o.res for o in members}
        loads = {v.split('#')[0] for o in members for v in o.ins} - internal
        # a member is a group output if its result has a use outside the group (text count, same function)
        stores = 0
        for o in members:
            if not o.res:
                continue
            uses =len(re.findall(re.escape(o.res) + r'(?![\w$.\-])', text)) - 1
            inner = sum(v.split('#')[0] == o.res for m in members for v in m.ins + m.outs)
            stores += uses > inner
        body = Counter(b for o in members for b in o.body if b not in SEEN and b != 'arith.constant')
        unseen.update(body)
        loops = max(o.n_loops for o in members)
        rows.append(dict(model=model_name(f), group=gid, size=len(members),
                         members=' '.join(o.label for o in members), n_loads=len(loads), n_stores=max(stores, 1),
                         loads_over=len(loads) > LS, stores_over=stores > LS, loops=loops, loops_over=loops > L,
                         unseen_ops=' '.join(sorted(body))))

n = len(rows) or 1
summary = [dict(groups=len(rows), pct_loads_over=sum(r['loads_over'] for r in rows) / n,
                pct_stores_over=sum(r['stores_over'] for r in rows) / n,
                pct_loops_over=sum(r['loops_over'] for r in rows) / n,
                pct_with_unseen_ops=sum(bool(r['unseen_ops']) for r in rows) / n,
                max_loads=max((r['n_loads'] for r in rows), default=0))]
urows = [dict(op=k, occurrences=v) for k, v in unseen.most_common()]

write_csv(args.out, 'a7_groups.csv', rows)
write_csv(args.out, 'a7_unseen_ops.csv', urows)
write_md(args.out, 'a7_summary.md', f'# A7 merged green groups vs limits (max_num_stores_loads={LS}, max_num_loops={L})\n\n'
         + md_table(summary) + '\n## Arith/math ops the features do not count\n\n' + md_table(urows))
