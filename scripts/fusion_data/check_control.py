"""Validate the control file (data/fusion_units/models.json) and print the enabled models.

Usage (repo root, any Python 3.9+):
    python scripts/fusion_data/check_control.py [--control data/fusion_units/models.json]
Exit code 1 if the file is invalid.
"""

import argparse
import sys
from pathlib import Path

from control import CONTROL, EVAL_SET, enabled_models, load_control, validate_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--eval-set', type=Path, default=EVAL_SET)
    args = parser.parse_args()

    control = load_control(args.control)
    errors = validate_control(control, args.eval_set)
    for error in errors:
        print(f'ERROR {error}')
    enabled = enabled_models(control)
    print(f'{len(enabled)} enabled / {len(control["models"])} models; defaults {control["defaults"]}')
    for m in enabled:
        layers = f' (first {m["max_layers"]} layers)' if m['max_layers'] else ''
        print(f'  {m["name"]:20s} {m["family"]:24s} weight={m["weight"]:<4} {m["kind"]}{layers}')
    for m in control['models']:
        if not m['enabled']:
            print(f'  [off] {m["name"]:14s} {m["reason"]}')
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
