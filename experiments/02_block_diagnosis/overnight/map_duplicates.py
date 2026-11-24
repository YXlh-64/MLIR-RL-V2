#!/usr/bin/env python3
"""Expand results from the 1474 measured unique computations to all 6693 blocks.

Blocks with the same catalog signature are the same MLIR computation on the same shapes
(they differ only in op tags / SSA names), so they have one measurement. This writes
results/<tag>/all_blocks_results.csv: one row per (block, method), with
`measured_as` = the block that was actually run. Standard library only.

Usage: python map_duplicates.py <tag> [<tag> ...]
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

E = Path(__file__).resolve().parents[1]


def main(tags):
    catalog = json.loads((E / 'results' / 'catalog' / 'blocks_catalog.json').read_text())['blocks']
    for tag in tags:
        root = E / 'results' / tag
        src = root / 'all_results.csv'
        if not src.exists():
            print(f'{tag}: no all_results.csv yet')
            continue
        rows = [r for r in csv.DictReader(src.open()) if r['config'] == 't1_warm' and r['repeat'] == '0']
        measured = defaultdict(list)
        sig_of = {b['name']: b.get('signature') for b in catalog}
        for r in rows:
            measured[sig_of.get(r['block'])].append(r)
        out_rows, missing = [], 0
        for b in catalog:
            reps = measured.get(b.get('signature'))
            if not reps:
                missing += 1
                continue
            for r in reps:
                row = dict(r)
                row.update(block=b['name'], model=b.get('model'), measured_as=r['block'],
                           signature=b.get('signature'))
                out_rows.append(row)
        if not out_rows:
            print(f'{tag}: nothing measured yet')
            continue
        path = root / 'all_blocks_results.csv'
        tmp = path.with_name('.all_blocks_results.tmp')
        with tmp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        os.replace(tmp, path)
        covered = len({r['block'] for r in out_rows})
        print(f'{tag}: {covered}/{len(catalog)} blocks covered by {len(measured)} measured computations; '
              f'{missing} not measured yet -> {path}')


if __name__ == '__main__':
    main(sys.argv[1:])
