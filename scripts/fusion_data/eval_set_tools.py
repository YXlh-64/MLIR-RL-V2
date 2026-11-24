"""Everything derived from the evaluation set (config/fusion/eval_set.json, the single source of truth).

Subcommands:
  eval-json      {file: root_exec_time} for the runnable eval models, the format of `eval_json_file`.
                 Times come from --times files, first file that has the model wins.
  survey-models  the runnable models that model_survey can build, in its models-JSON format
                 (input for profile_torch.py).
  survey-subset  keep only those models in an existing profile directory (ops.csv + models.json),
                 so check_suitability.py compares against exactly the confirmed eval set.
  link-runnable  a directory of symlinks to the runnable eval files (input of the eval baseline job).

Run from the repo root with the MLIR env (source scripts/analysis/gap1/env.sh).
"""

import argparse
import csv
import json
import os
from pathlib import Path

from rl_autoschedular_v6.fusion.eval_set import runnable_models

EVAL_SET = 'config/fusion/eval_set.json'


def eval_json(eval_set: str, time_files: list[str], out: str, allow_missing: bool) -> None:
    sources = []
    for path in time_files:
        with open(path) as f:
            sources.append(json.load(f))
    times, missing = {}, []
    for model in runnable_models(eval_set):
        found = next((src[model.file] for src in sources if model.file in src), None)
        if found is None:
            missing.append(model.file)
        else:
            times[model.file] = found
    if missing and not allow_missing:
        raise SystemExit(f'no root_exec_time for {missing}; measure them first or pass --allow-missing')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(times, f, indent=2)
    print(f'{out}: {len(times)} models; missing: {missing or "none"}')


def survey_names(eval_set: str) -> list[str]:
    return [m.name for m in runnable_models(eval_set) if m.survey is not None]


def survey_models(eval_set: str, out: str) -> None:
    models = [{'name': m.name, **m.survey} for m in runnable_models(eval_set) if m.survey is not None]
    doc = f'Generated from {eval_set} (runnable eval models buildable by model_survey). Do not edit.'
    with open(out, 'w') as f:
        json.dump({'_doc': doc, 'models': models}, f, indent=1)
    print(f'{out}: {[m["name"] for m in models]}')


def survey_subset(eval_set: str, profile_dir: str, out_dir: str) -> None:
    keep = set(survey_names(eval_set))
    src, dst = Path(profile_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=False)
    with open(src / 'ops.csv') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['model'] in keep]
        fields = reader.fieldnames
    with open(dst / 'ops.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(src / 'models.json') as f:
        models = [m for m in json.load(f) if m['model'] in keep]
    with open(dst / 'models.json', 'w') as f:
        json.dump(models, f, indent=1)
    found = {m['model'] for m in models}
    print(f'{dst}: kept {sorted(found)}; not in the profile: {sorted(keep - found) or "none"}')


def link_runnable(eval_set: str, full_models: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for model in runnable_models(eval_set):
        target = Path(full_models).resolve() / f'{model.file}.mlir'
        link = out / target.name
        if not target.exists():
            raise SystemExit(f'missing eval file {target}')
        if not link.exists():
            link.symlink_to(target)
    print(f'{out}: {sorted(p.name for p in out.iterdir())}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-set', default=EVAL_SET)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('eval-json')
    p.add_argument('--times', nargs='+', required=True, help='JSON files {file_stem: root_exec_time}')
    p.add_argument('--out', required=True)
    p.add_argument('--allow-missing', action='store_true')
    p = sub.add_parser('survey-models')
    p.add_argument('--out', required=True)
    p = sub.add_parser('survey-subset')
    p.add_argument('--profile', required=True, help='profile_torch.py / profile_linalg.py output dir')
    p.add_argument('--out', required=True, help='new directory (must not exist)')
    p = sub.add_parser('link-runnable')
    p.add_argument('--full-models', default='data/full_models')
    p.add_argument('--out', default='data/fusion_units/eval_links')
    args = parser.parse_args()

    if args.cmd == 'eval-json':
        eval_json(args.eval_set, args.times, args.out, args.allow_missing)
    elif args.cmd == 'survey-models':
        survey_models(args.eval_set, args.out)
    elif args.cmd == 'survey-subset':
        survey_subset(args.eval_set, args.profile, args.out)
    else:
        link_runnable(args.eval_set, args.full_models, args.out)


if __name__ == '__main__':
    main()
