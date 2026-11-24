"""B5: is time additive? Full-model time vs sum of block times and sum of single-op times (default lowering)."""
import argparse
import os
from statistics import median

from _run import prepare, run
from _mlir_text import files, md_table, write_csv, write_md
from data_utils.extract.extract_blocks import (_build_block_benchmark, _collect_candidate_windows, _parse_ast,
                                               _run_ast_dumper)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('models', nargs='+', help='full-model .mlir files or globs (static shapes)')
ap.add_argument('--batch', type=int, default=1, help="value for '?' dims in extracted ops/blocks")
ap.add_argument('--window-size', type=int, default=5)
ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--max-depth', type=int, default=64)
ap.add_argument('--max-paths', type=int, default=5000)
ap.add_argument('--runs', type=int, default=3)
ap.add_argument('--out', required=True)
args = ap.parse_args()


def time_of(code):
    try:
        return median(run(code, args.runs)[0])
    except Exception:  # noqa: BLE001
        return None


def unit(parsed, tags, model, i):
    """Time one extractor-built benchmark (tags consumer->producer), None if it cannot be built/run."""
    try:
        return time_of(_build_block_benchmark(parsed, tags, args.batch, model, i)[0])
    except Exception:  # noqa: BLE001
        return None


summary, details = [], []
for f in files(args.models):
    model = os.path.basename(f).replace('_linalg.mlir', '').replace('.mlir', '')
    parsed = _parse_ast(_run_ast_dumper(f))
    t_full = time_of(prepare(open(f).read()))
    op_t = {t: unit(parsed, [t], model, i) for i, t in enumerate(parsed.operation_tags)}
    details += [dict(model=model, kind='op', tags=t, time_ns=v) for t, v in op_t.items()]

    # greedy disjoint cover with extractor windows; uncovered ops are added as single ops
    covered, blocks_t = set(), 0
    for i, w in enumerate(_collect_candidate_windows(parsed, args.window_size, args.stride, args.max_depth, args.max_paths)):
        if covered & set(w):
            continue
        t = unit(parsed, w, model, i)
        if t is not None:
            covered |= set(w)
            blocks_t += t
            details.append(dict(model=model, kind='block', tags=' '.join(w), time_ns=t))
    ok_ops = {t: v for t, v in op_t.items() if v is not None}
    sum_ops = sum(ok_ops.values())
    sum_blocks = blocks_t + sum(v for t, v in ok_ops.items() if t not in covered)
    summary.append(dict(model=model, full_ns=t_full, n_ops=len(op_t), ops_failed=len(op_t) - len(ok_ops),
                        sum_ops_ns=sum_ops, sum_blocks_ns=sum_blocks, ops_in_blocks=len(covered),
                        ops_ratio=sum_ops / t_full if t_full else '', blocks_ratio=sum_blocks / t_full if t_full else ''))

write_csv(args.out, 'b5_units.csv', details)
write_csv(args.out, 'b5_summary.csv', summary)
write_md(args.out, 'b5_summary.md', '# B5 additivity (ratio = sum / full-model time; 1.0 = additive)\n\n'
         + md_table(summary))
