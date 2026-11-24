"""Step 3 — linalg-level profile of the validated models vs the runnable eval models, with weights.

Runs model_survey/profile_linalg.py on data/fusion_units/raw (validated, enabled
models only) and on the runnable eval models of config/fusion/eval_set.json
(files in --full-models), then check_suitability.py with --weights (model
weights from the control file). Read report/report.md, section 4b: matmul N/K
and conv C/F median and p90, candidates vs eval (target: candidate median about
2^9, p90 about 2^11).

Outputs under data/fusion_units/profile/: cand/, eval/, weights.json, report/.
Idempotent: an existing cand/ or eval/ profile is reused unless --force.

Usage (repo root, any Python 3.9+; the eval files are large, run it as a job):
    python scripts/fusion_data/profile.py [--full-models data/full_models] [--force]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from control import CONTROL, EVAL_SET, PROFILE, RAW, VALIDATION, enabled_models, load_control

SURVEY_DIR = Path('/scratch/mh8826/model_survey')


def run_profile(files: list[Path], out: Path, survey: Path, force: bool) -> None:
    if (out / 'ops.csv').exists() and not force:
        print(f'[skip] {out} (exists)')
        return
    subprocess.run([sys.executable, str(survey / 'profile_linalg.py'), *map(str, files), '--out', str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--survey-dir', type=Path, default=SURVEY_DIR)
    parser.add_argument('--full-models', type=Path, default=Path('data/full_models'))
    parser.add_argument('--min-ops', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    validation = json.loads(VALIDATION.read_text())
    models = [m for m in enabled_models(load_control(args.control)) if validation.get(m['name'], {}).get('ok')]
    with open(EVAL_SET) as f:
        eval_files = [args.full_models / f'{m["file"]}.mlir' for m in json.load(f)['models'] if m['runnable']]

    PROFILE.mkdir(parents=True, exist_ok=True)
    run_profile([RAW / f'{m["name"]}_linalg.mlir' for m in models], PROFILE / 'cand', args.survey_dir, args.force)
    run_profile(eval_files, PROFILE / 'eval', args.survey_dir, args.force)
    weights = PROFILE / 'weights.json'
    # profile_linalg names a model after its file without "_linalg.mlir"
    weights.write_text(json.dumps({m['name']: m['weight'] for m in models}, indent=1))
    subprocess.run([sys.executable, str(args.survey_dir / 'check_suitability.py'), str(PROFILE / 'cand'),
                    '--eval', str(PROFILE / 'eval'), '--out', str(PROFILE / 'report'),
                    '--weights', str(weights), '--min-ops', str(args.min_ops)], check=True)


if __name__ == '__main__':
    main()
