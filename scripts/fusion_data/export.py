"""Step 1 — export the enabled models of the control file to linalg-on-tensors MLIR.

Runs in the export env (torch + torch-mlir, NOT MLIR-ENV), with HF_HUB_OFFLINE=1:
only the cached config.json files are used (download them first with
model_survey/download_models.py). Models are built by model_survey's
export_candidates.build: random weights, eager (explicit) attention, so no
fused SDPA reaches torch-mlir (which would lower to tm_tensor). Large constants
are elided (dense_resource<__elided__>), as in data/full_models.

Writes data/fusion_units/raw/<name>_linalg.mlir and raw/export_log.json;
--generic also writes the generic-op form to raw_generic/ (fallback for parse
problems between MLIR versions). Idempotent: existing outputs are skipped
unless --force.

Usage (repo root):
    HF_HUB_OFFLINE=1 python scripts/fusion_data/export.py [--only a,b] [--generic] [--force]
"""

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from control import CONTROL, RAW, RAW_GENERIC, enabled_models, load_control

SURVEY_DIR = Path('/scratch/mh8826/model_survey')


def linalg_text(module, inputs, generic: bool) -> str:
    """torch-mlir export to linalg, printed with weights elided (as export_candidates.to_linalg_text)."""
    import torch
    from torch_mlir.compiler_utils import OutputType
    from torch_mlir.fx import export_and_import

    with torch.no_grad():
        mlir = export_and_import(module, *inputs, output_type=OutputType.LINALG_ON_TENSORS, func_name='main')
    text = mlir.operation.get_asm(large_elements_limit=16, print_generic_op_form=generic)
    text = text.split('{-#')[0]                                     # drop resource blobs
    return re.sub(r'dense_resource<[^>]+>', 'dense_resource<__elided__>', text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, default=CONTROL)
    parser.add_argument('--survey-dir', type=Path, default=SURVEY_DIR)
    parser.add_argument('--cache', type=Path, default=SURVEY_DIR / 'hf_cache')
    parser.add_argument('--out', type=Path, default=RAW)
    parser.add_argument('--only', default='', help='comma-separated model names')
    parser.add_argument('--generic', action='store_true', help='also write the generic op form to --generic-out')
    parser.add_argument('--generic-out', type=Path, default=RAW_GENERIC)
    parser.add_argument('--force', action='store_true', help='re-export existing files')
    args = parser.parse_args()

    sys.path.insert(0, str(args.survey_dir))
    from export_candidates import build   # model_survey: model + example inputs per kind

    control = load_control(args.control)
    defaults = control['defaults']
    build_args = SimpleNamespace(batch=defaults['batch'], seq_len=defaults['seq_len'], weights=False,
                                 cache=str(args.cache), max_vocab=defaults['max_vocab'])
    only = {x for x in args.only.split(',') if x}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / 'export_log.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    for m in enabled_models(control):
        if only and m['name'] not in only:
            continue
        path = args.out / f'{m["name"]}_linalg.mlir'
        generic_path = args.generic_out / path.name
        if path.exists() and (generic_path.exists() or not args.generic) and not args.force:
            print(f'[skip] {m["name"]} (exists)')
            continue
        start = time.time()
        try:
            module, inputs = build(m, build_args)
            module.eval()
            path.write_text(linalg_text(module, inputs, generic=False))
            if args.generic:
                args.generic_out.mkdir(parents=True, exist_ok=True)
                generic_path.write_text(linalg_text(module, inputs, generic=True))
            log[m['name']] = {'status': 'ok', 'file': str(path), 'inputs': [list(t.shape) for t in inputs],
                              'params_M': round(sum(p.numel() for p in module.parameters()) / 1e6, 2),
                              'seconds': round(time.time() - start, 1)}
            print(f'[ok]   {m["name"]} -> {path}')
        except Exception as e:  # one failing model must not stop the others; the log keeps the reason
            log[m['name']] = {'status': 'failed', 'error': f'{type(e).__name__}: {e}'[:500],
                              'trace': traceback.format_exc()[-2000:]}
            print(f'[FAIL] {m["name"]}: {type(e).__name__}: {str(e)[:150]}')
        log_path.write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
