"""Step 5 — train/val split by MODEL, then the benchmark JSONs read by Benchmarks.

  split       models in the units manifest -> data/fusion_units/split.json.
              Stratified by family (control file), fixed seed: in each family,
              round(n * val_fraction) models go to val, always leaving at least one
              in train. A family with one model stays in train. Written once; the
              split never changes unless --force.
  bench-json  measured unit baselines {unit: ns} -> base_train.json / base_val.json
              (results/fusion_units_results/baselines/mlir/, the paths in config/fusion).
              Units with a failed baseline (<= 0) are left out and listed.

Usage (repo root, any Python 3.9+):
    python scripts/fusion_data/split.py split [--val-fraction 0.2] [--seed 0]
    python scripts/fusion_data/split.py bench-json --times results/fusion_units_results/baselines/mlir/all_units.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from control import BASELINES, CONTROL, MANIFEST, SPLIT, load_control


def split_models(models: list[str], family: dict[str, str], val_fraction: float, seed: int) -> dict:
    by_family = defaultdict(list)
    for name in sorted(models):
        by_family[family[name]].append(name)
    rng = random.Random(seed)
    train, val = [], []
    for fam in sorted(by_family):
        names = by_family[fam][:]
        rng.shuffle(names)
        n_val = min(round(len(names) * val_fraction), len(names) - 1)
        val += names[:n_val]
        train += names[n_val:]
    return {'seed': seed, 'val_fraction': val_fraction, 'train': sorted(train), 'val': sorted(val)}


def bench_json(times: dict[str, int], manifest: dict, split: dict) -> tuple[dict, dict, list[str]]:
    unit_model = {u['file'][:-len('.mlir')]: model
                  for model, entry in manifest['models'].items() for u in entry['units']}
    train, val, failed = {}, {}, []
    for unit, model in sorted(unit_model.items()):
        t = times.get(unit, -1)
        if t is None or t <= 0:
            failed.append(unit)
        elif model in split['train']:
            train[unit] = t
        elif model in split['val']:
            val[unit] = t
    return train, val, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('split')
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p = sub.add_parser('bench-json')
    p.add_argument('--times', type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.cmd == 'split':
        if SPLIT.exists() and not args.force:
            raise SystemExit(f'{SPLIT} exists; the split is fixed once made (--force to redo)')
        family = {m['name']: m['family'] for m in load_control(CONTROL)['models']}
        split = split_models(list(manifest['models']), family, args.val_fraction, args.seed)
        SPLIT.write_text(json.dumps(split, indent=1))
        print(f'{SPLIT}: {len(split["train"])} train models, {len(split["val"])} val models: {split["val"]}')
        return

    split = json.loads(SPLIT.read_text())
    train, val, failed = bench_json(json.loads(args.times.read_text()), manifest, split)
    BASELINES.mkdir(parents=True, exist_ok=True)
    (BASELINES / 'base_train.json').write_text(json.dumps(train, indent=1))
    (BASELINES / 'base_val.json').write_text(json.dumps(val, indent=1))
    print(f'base_train.json: {len(train)} units, base_val.json: {len(val)} units, '
          f'left out (failed baseline): {len(failed)} {failed[:10]}')


if __name__ == '__main__':
    main()
