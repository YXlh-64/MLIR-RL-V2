"""B1: how noisy is v5 timing? Per benchmark: CV of single runs and spread of the 2-run median the reward uses."""
import argparse
import os
from statistics import mean, median, pstdev

from _run import prepare, run
from _mlir_text import files, md_table, write_csv, write_md

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('benchmarks', nargs='+', help='.mlir files or globs')
ap.add_argument('--runs', type=int, default=20, help='invocations per compile')
ap.add_argument('--repeats', type=int, default=3, help='fresh compiles per benchmark')
ap.add_argument('--out', required=True)
args = ap.parse_args()

cv = lambda xs: pstdev(xs) / mean(xs) if len(xs) > 1 and mean(xs) else float('nan')
rows = []
for f in files(args.benchmarks):
    code = prepare(open(f).read())
    times, per_compile = [], []
    try:
        for _ in range(args.repeats):
            t = run(code, args.runs)[0]
            times += t
            per_compile.append(median(t))
    except Exception as e:  # noqa: BLE001
        rows.append(dict(bench=os.path.basename(f), error=str(e)[:200]))
        continue
    med2 = [median(times[i:i + 2]) for i in range(0, len(times) - 1, 2)]  # v5: median of 2 runs
    rows.append(dict(bench=os.path.basename(f), n=len(times), median_ns=median(times), cv_runs=cv(times),
                     cv_runs_excl_first=cv([x for i, x in enumerate(times) if i % args.runs]),
                     cv_med2=cv(med2), med2_max_over_min=max(med2) / min(med2) if min(med2) > 0 else float('nan'),
                     cv_across_compiles=cv(per_compile)))

ok = [r for r in rows if 'error' not in r]
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')
summary = [dict(benches=len(ok), failed=len(rows) - len(ok),
                median_cv_runs=median([r['cv_runs'] for r in ok]) if ok else '',
                median_cv_med2=median([r['cv_med2'] for r in ok]) if ok else '',
                p95_med2_max_over_min=q([r['med2_max_over_min'] for r in ok], 0.95))]

write_csv(args.out, 'b1_noise.csv', rows)
write_md(args.out, 'b1_summary.md', '# B1 timing noise\n\n' + md_table(summary)
         + '\n`p95_med2_max_over_min`: a speedup below this is within noise for a single 2-run median.\n')
