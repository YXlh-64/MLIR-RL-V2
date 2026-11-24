"""Step 4 — closed training units (fusion M3) for every enabled, validated model.

Runs in MLIR-ENV. Unit parameters (min/max ops, eval set for the leakage guard)
come from the `fusion.units` section of config/fusion/A_prime.json; the guard is
always on. Writes data/fusion_units/units/<model>_unit_<k>.mlir and, next to
them, units_manifest.json (updated after every model), then
data/fusion_units/sampling_weights.json.

Sampling weight of a unit = model weight (control file) x occurrences of the
unit in its model (TVM/Ansor-style task weight: time ~ sum of unit time x count).
With --normalise-per-model, each model's unit weights are divided by the model's
total occurrences, so every model contributes exactly its control-file weight
whatever its depth. Default: not normalised (design doc §4.2, principle 3).

Idempotent: models already in the manifest are skipped (--force rebuilds after you
remove their unit files yourself; nothing is ever deleted by this script).

Usage (repo root):
    CONFIG_FILE_PATH=config/fusion/A_prime.json python scripts/fusion_data/build_units.py [--only a,b] [--normalise-per-model]
"""

import argparse
import json
from pathlib import Path

from control import CONTROL, MANIFEST, RAW, SAMPLING_WEIGHTS, UNITS, UNITS_CONFIG, VALIDATION, enabled_models, load_control


def sampling_weights(manifest: dict, model_weights: dict[str, float], normalise: bool) -> dict[str, float]:
    """{unit benchmark name: sampling weight}."""
    weights = {}
    for model, entry in manifest['models'].items():
        total = sum(u['weight'] for u in entry['units']) or 1
        for unit in entry['units']:
            share = unit['weight'] / total if normalise else unit['weight']
            weights[unit['file'][:-len('.mlir')]] = model_weights[model] * share
    return weights


def main() -> None:
    from rl_autoschedular_v6.fusion.units import UnitParams, build_units

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--config', type=Path, default=UNITS_CONFIG)
    parser.add_argument('--only', default='', help='comma-separated model names')
    parser.add_argument('--force', action='store_true', help='rebuild models already in the manifest')
    parser.add_argument('--normalise-per-model', action='store_true')
    args = parser.parse_args()

    params = UnitParams.from_dict(json.loads(args.config.read_text())['fusion']['units'])
    if not params.eval_set:
        raise SystemExit(f'{args.config}: fusion.units.eval_set is empty; the leakage guard must be on')
    validation = json.loads(VALIDATION.read_text())
    control = load_control(args.control)
    UNITS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'params': params.__dict__, 'models': {}}
    only = {x for x in args.only.split(',') if x}

    for m in enabled_models(control):
        name = m['name']
        if (only and name not in only) or not validation.get(name, {}).get('ok'):
            continue
        if name in manifest['models'] and not args.force:
            print(f'[skip] {name} (in manifest)')
            continue
        leftovers = sorted(UNITS.glob(f'{name}_unit_*.mlir'))
        if leftovers:
            raise SystemExit(f'{name}: unit files exist without a manifest entry (or --force): {leftovers[:3]}... '
                             'remove them yourself, then re-run')
        entry = build_units(str(RAW / f'{name}_linalg.mlir'), str(UNITS), params)
        manifest['models'][name] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f'[ok] {name}: {entry["segments"]} segments -> {len(entry["units"])} units, '
              f'{len(entry["skipped"])} skipped, missing op names: {entry["missing_op_names"]}')

    model_weights = {m['name']: m['weight'] for m in control['models']}
    weights = sampling_weights(manifest, model_weights, args.normalise_per_model)
    SAMPLING_WEIGHTS.write_text(json.dumps(weights, indent=1))
    print(f'{SAMPLING_WEIGHTS}: {len(weights)} units (normalised per model: {args.normalise_per_model})')


if __name__ == '__main__':
    main()
