"""A4: which start ops hit the max_paths cap of _enumerate_backward_paths, and which of their producers are never reached?"""
import argparse
from collections import deque

from _mlir_text import files, md_table, model_name, parse_file, write_csv, write_md


def _enumerate_backward_paths(operations, start_tag, max_depth, max_paths):
    """Exact copy of data_utils/extract/extract_blocks.py::_enumerate_backward_paths
    (operations: tag -> producers [(producer_tag, operand_idx)])."""
    paths = []
    stack = [[start_tag]]

    while stack and len(paths) < max_paths:
        path = stack.pop()
        current = path[-1]

        if len(path) >= 2:
            paths.append(path)

        if len(path) >= max_depth:
            continue

        for producer_tag, _ in operations[current]:
            if producer_tag not in operations:
                continue
            if producer_tag in path:
                continue
            stack.append(path + [producer_tag])

    return paths


def ancestors(operations, start, max_depth):
    """Producers reachable within max_depth-1 edges (BFS)."""
    seen, q = {start: 0}, deque([start])
    while q:
        t = q.popleft()
        if seen[t] + 1 >= max_depth:
            continue
        for p, _ in operations[t]:
            if p in operations and p not in seen:
                seen[p] = seen[t] + 1
                q.append(p)
    return set(seen) - {start}


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('models', nargs='+', help='full-model .mlir files, dirs or globs')
ap.add_argument('--max-depth', type=int, default=64)
ap.add_argument('--max-paths', type=int, default=5000)
ap.add_argument('--out', required=True)
args = ap.parse_args()

rows, summary = [], []
for f in files(args.models):
    ops = parse_file(f)[0]
    operations = {o.tag: o.producers for o in ops}
    for op in ops:
        paths = _enumerate_backward_paths(operations, op.tag, args.max_depth, args.max_paths)
        reached = {t for p in paths for t in p[1:]}
        anc = ancestors(operations, op.tag, args.max_depth)
        unreached = sorted(anc - reached)
        rows.append(dict(model=model_name(f), start_tag=op.tag, op_type=op.label, n_producers=len(op.producers),
                         n_paths=len(paths), capped=len(paths) >= args.max_paths, n_ancestors=len(anc),
                         n_unreached=len(unreached), unreached=' '.join(unreached)))
    mr = [r for r in rows if r['model'] == model_name(f)]
    summary.append(dict(model=model_name(f), n_ops=len(mr), n_capped=sum(r['capped'] for r in mr),
                        n_starts_with_unreached=sum(r['n_unreached'] > 0 for r in mr),
                        total_unreached=sum(r['n_unreached'] for r in mr)))

write_csv(args.out, 'a4_starts.csv', rows)
write_csv(args.out, 'a4_summary.csv', summary)
write_md(args.out, 'a4_summary.md', f'# A4 path cap (max_paths={args.max_paths}, max_depth={args.max_depth})\n\n'
         + md_table(summary))
