"""Step 2 — check that every exported model is usable by the v6 agent.

Runs in MLIR-ENV with CONFIG_FILE_PATH=config/fusion/A_prime.json (its loop and
load limits decide which ops the features keep). Per model:
- no `tm_tensor.` op (the AstDumper cannot parse it) and no dynamic dims (`?`);
- the v6 MLIR bindings parse it (weights elided by fusion.mlir_io); if the pretty
  form fails and raw_generic/ has the generic form, that one is tried;
- the AstDumper + `extract_bench_features_from_code` load it, and keep every op
  (ops beyond the config limits are dropped by state.py and reported here).

Writes data/fusion_units/validation.json {model: {ok, form, ops, feature_ops, reasons, mtime}}.
Idempotent: a model whose file did not change since its last check is skipped.

Usage (repo root):
    CONFIG_FILE_PATH=config/fusion/A_prime.json python scripts/fusion_data/validate.py [--raw-dir DIR] [--out FILE]
"""

import argparse
import json
import re
from pathlib import Path

from control import CONTROL, RAW, RAW_GENERIC, VALIDATION, enabled_models, load_control

DYNAMIC = re.compile(r'tensor<[^>]*\?')


def check_file(name: str, path: Path) -> dict:
    """Parse + AstDumper + features for one file; returns ops counts or raises."""
    from rl_autoschedular_v6.fusion.mlir_io import read_mlir
    from rl_autoschedular_v6.fusion.views import graph_from_bindings
    from rl_autoschedular_v6.state import extract_bench_features_from_code

    code = read_mlir(str(path))
    ops = len(graph_from_bindings(code).ops)
    feature_ops = len(extract_bench_features_from_code(name, code, 1).operation_tags)
    return {'ops': ops, 'feature_ops': feature_ops}


def validate(name: str, path: Path, generic_path: Path) -> dict:
    text = path.read_text()
    reasons = []
    if 'tm_tensor.' in text:
        reasons.append('contains tm_tensor ops')
    if DYNAMIC.search(text):
        reasons.append('dynamic dims')
    result = {'form': None, 'ops': None, 'feature_ops': None}
    errors = []
    for form, candidate in (('pretty', path), ('generic', generic_path)):
        if not candidate.exists():
            continue
        try:
            result.update(check_file(name, candidate), form=form)
            break
        except Exception as e:  # record why each form failed
            errors.append(f'{form}: {type(e).__name__}: {str(e).strip().splitlines()[0][:200]}')
    if result['form'] is None:
        reasons += errors or ['no file could be parsed']
    elif result['feature_ops'] != result['ops']:
        reasons.append(f'{result["ops"] - result["feature_ops"]} ops dropped by the config limits')
    return {**result, 'ok': not reasons, 'reasons': reasons, 'mtime': path.stat().st_mtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--raw-dir', type=Path, default=RAW)
    parser.add_argument('--generic-dir', type=Path, default=RAW_GENERIC)
    parser.add_argument('--out', type=Path, default=VALIDATION)
    parser.add_argument('--only', default='', help='comma-separated model names')
    args = parser.parse_args()

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    only = {x for x in args.only.split(',') if x}
    for m in enabled_models(load_control(args.control)):
        name = m['name']
        path = args.raw_dir / f'{name}_linalg.mlir'
        if (only and name not in only) or not path.exists():
            continue
        if name in results and results[name].get('mtime') == path.stat().st_mtime:
            print(f'[skip] {name} (unchanged)')
            continue
        results[name] = validate(name, path, args.generic_dir / path.name)
        r = results[name]
        print(f'[{"ok" if r["ok"] else "FAIL"}] {name}: form={r["form"]} ops={r["ops"]} {"; ".join(r["reasons"])}')
        args.out.write_text(json.dumps(results, indent=2))
    ok = sorted(n for n, r in results.items() if r['ok'])
    print(f'{len(ok)}/{len(results)} valid: {ok}')


if __name__ == '__main__':
    main()
