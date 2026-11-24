"""Control file and paths of the A'-D+ training-data pipeline (stdlib only).

Every script in scripts/fusion_data reads the models only from the control file
(data/fusion_units/models.json). Stdlib only, because export.py imports it in the
export env, which has no MLIR-RL packages. Paths are relative to the repo root.
"""

import json
from pathlib import Path

DATA = Path('data/fusion_units')
CONTROL = DATA / 'models.json'
RAW = DATA / 'raw'                      # step 1: <model>_linalg.mlir (pretty form)
RAW_GENERIC = DATA / 'raw_generic'      # step 1, --generic: same in generic op form
VALIDATION = DATA / 'validation.json'   # step 2
PROFILE = DATA / 'profile'              # step 3
UNITS = DATA / 'units'                  # step 4: <model>_unit_<k>.mlir + units_manifest.json
MANIFEST = UNITS / 'units_manifest.json'
SAMPLING_WEIGHTS = DATA / 'sampling_weights.json'
SPLIT = DATA / 'split.json'             # step 5
EVAL_LINKS = DATA / 'eval_links'        # symlinks to the runnable eval models
BASELINES = Path('results/fusion_units_results/baselines/mlir')
EVAL_SET = Path('config/fusion/eval_set.json')
UNITS_CONFIG = Path('config/fusion/A_prime.json')   # limits + fusion.units params used by the pipeline

KINDS = {'timm', 'hf_vision', 'hf_text_encoder', 'hf_text_decoder', 'hf_seq2seq',
         'hf_whisper_encoder', 'hf_audio_encoder'}
FIELDS = {'name': (str,), 'repo': (str,), 'kind': (str,), 'max_layers': (int, type(None)),
          'family': (str,), 'enabled': (bool,), 'weight': (int, float), 'reason': (str,)}


def load_control(path: Path = CONTROL) -> dict:
    with open(path) as f:
        return json.load(f)


def enabled_models(control: dict) -> list[dict]:
    return [m for m in control['models'] if m['enabled']]


def eval_names(path: Path = EVAL_SET) -> set[str]:
    with open(path) as f:
        return {m['name'] for m in json.load(f)['models']}


def validate_control(control: dict, eval_set: Path = EVAL_SET) -> list[str]:
    """Errors in the control file (empty list = valid)."""
    errors = []
    if set(control) - {'_doc'} != {'defaults', 'models'}:
        errors.append(f'top-level keys must be defaults and models (+ _doc), got {sorted(control)}')
        return errors
    names = [m.get('name') for m in control['models']]
    errors += [f'duplicate name {n}' for n in sorted(set(names)) if names.count(n) > 1]
    excluded = eval_names(eval_set)
    for m in control['models']:
        name = m.get('name', '?')
        if set(m) != set(FIELDS):
            errors.append(f'{name}: fields must be {sorted(FIELDS)}, got {sorted(m)}')
            continue
        errors += [f'{name}: {key} must be {[t.__name__ for t in types]}' for key, types in FIELDS.items()
                   if not isinstance(m[key], types) or (key != 'enabled' and isinstance(m[key], bool))]
        if m['kind'] not in KINDS:
            errors.append(f'{name}: unknown kind {m["kind"]}')
        if m['enabled'] and not m['weight'] > 0:
            errors.append(f'{name}: enabled models need weight > 0')
        if not m['enabled'] and not m['reason'].startswith('disabled'):
            errors.append(f'{name}: disabled models need a reason starting with "disabled"')
        if name in excluded:
            errors.append(f'{name}: is in the evaluation set ({eval_set}); it cannot be a training model')
    return errors
