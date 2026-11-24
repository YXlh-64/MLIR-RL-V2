#!/usr/bin/env python3
"""Track A setup: write one config + one-model benchmark list per model (repo files untouched).

Each config is a copy of the repo's full-model config (same hyperparameters, so v5_model_1200
loads) with absolute paths and a private results_dir under track_a/results/<threads>/<model>/agent.
Writes track_a/models.tsv: model<TAB>models_dir<TAB>file<TAB>stored_baseline_ns<TAB>config.
"""
import json
from pathlib import Path

REPO = Path('/scratch/mh8826/MLIR-RL')
HERE = Path(__file__).resolve().parent
GROUPS = [  # (repo config, repo eval list, folder holding the .mlir files)
    ('config/v5/v5_mlir_rl_v1_paper_fullmodel.json', 'config/mlir_rl_v1_paper/eval/fullmodel_eval.json',
     '/scratch/mh8826/data/mlir_rl_v1_paper'),
    ('config/v5/v5_fullmodel_linalg.json', 'config/mlir_rl_v1_paper/eval/fullmodel_linalg_eval.json',
     '/scratch/mb10856/MLIR-RL/data/full_models'),
]

rows = []
for cfg_path, list_path, folder in GROUPS:
    base_cfg = json.loads((REPO / cfg_path).read_text())
    for model, base_ns in json.loads((REPO / list_path).read_text()).items():
        f = Path(folder) / f'{model}.mlir'
        assert f.exists(), f
        bench = HERE / 'configs' / f'{model}.bench.json'
        bench.write_text(json.dumps({model: base_ns}))
        cfg = dict(base_cfg, benchmarks_folder_path=folder, json_file=str(bench),
                   eval_json_file=str(bench), results_dir='__RESULTS_DIR__')
        (HERE / 'configs' / f'{model}.config.json').write_text(json.dumps(cfg, indent=2))
        rows.append((model, folder, f.name, str(base_ns), str(HERE / 'configs' / f'{model}.config.json')))

rows.sort(key=lambda r: float(r[3]))   # fastest first
(HERE / 'models.tsv').write_text(''.join('\t'.join(r) + '\n' for r in rows))
print(f'{len(rows)} models -> {HERE / "models.tsv"}')
for r in rows:
    print(f'  {r[0]:28s} stored baseline {float(r[3]) / 1e9:8.3f} s')
