"""config/fusion/*.json: one per ablation row (design doc §6) + its full-model eval twin,
and the evaluation set (config/fusion/eval_set.json, single source of truth)."""

import json
import os
import subprocess
import sys

import pytest

from conftest import REPO
from rl_autoschedular_v6.fusion.config import FusionConfig
from rl_autoschedular_v6.fusion.eval_set import load_eval_set, runnable_models
from rl_autoschedular_v6.fusion.tile_guard import TileGuardParams
from rl_autoschedular_v6.fusion.units import UnitParams

CONFIG_DIR = REPO / 'config' / 'fusion'
EVAL_SET = 'config/fusion/eval_set.json'
UNITS_DIR = 'data/fusion_units/units'
BASELINES = 'results/fusion_units_results/baselines/mlir'
RUNS_DIR = '/scratch/mh8826/fusion_runs'   # checkpoints + logs of every row, outside the git repo
ROWS = {  # row -> (prefuse, producer_select, tile_guard, trains on units)
    'A': (False, False, False, False),
    'A_prime': (False, False, False, True),
    'B': (True, False, False, True),
    'C': (True, False, False, True),
    'D': (True, True, False, True),
    'D_plus': (True, True, True, True),
}
FILES = [f'{row}{suffix}.json' for row in ROWS for suffix in ('', '_eval_fullmodel')]


def load(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


@pytest.mark.parametrize('name', FILES)
def test_loads_through_v6_config(name):
    """The real (singleton) Config and load_fusion_config accept the file."""
    code = ('from rl_autoschedular_v6.utils.config import Config; Config(); '
            'from rl_autoschedular_v6.fusion.config import load_fusion_config; print(load_fusion_config())')
    env = {**os.environ, 'CONFIG_FILE_PATH': str(CONFIG_DIR / name)}
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, env=env, cwd=REPO)


@pytest.mark.parametrize('row', ROWS)
def test_flags_data_and_isolation(row):
    train, evaluation = load(f'{row}.json'), load(f'{row}_eval_fullmodel.json')
    prefuse, select, guard, units = ROWS[row]
    for cfg in (train, evaluation):
        fusion = FusionConfig.from_dict(cfg['fusion'])
        assert (fusion.prefuse, fusion.producer_select, fusion.tile_guard) == (prefuse, select, guard)
        assert UnitParams.from_dict(fusion.units).eval_set == EVAL_SET
        TileGuardParams.from_dict(fusion.tile_guard_params)
        assert cfg['implementation'] == 'rl_autoschedular_v6' and cfg['main_exec_data_file'] == ''
    assert train['fusion'] == evaluation['fusion']                 # eval applies the same modules
    # eval_fullmodel.sh takes the checkpoints with --checkpoints-dir <train results_dir>/models and writes
    # into its own results_dir, so full-model results never overwrite the unit-validation eval files.
    assert evaluation['results_dir'] == train['results_dir'] + '_fullmodel_eval'
    assert train['max_num_loops'] == evaluation['max_num_loops']   # same observation shape
    assert evaluation['benchmarks_folder_path'] == 'data/full_models'
    assert evaluation['eval_json_file'] == f'{BASELINES}/eval_fullmodel.json'
    if units:
        assert train['benchmarks_folder_path'] == UNITS_DIR
        assert train['json_file'] == f'{BASELINES}/base_train.json'
        assert train['eval_json_file'] == f'{BASELINES}/base_val.json'
        assert train['fusion']['sampling_weights'] == 'data/fusion_units/sampling_weights.json'
    else:
        assert train['benchmarks_folder_path'] == 'data/ops_and_blocks'
        assert FusionConfig.from_dict(train['fusion']).sampling_weights == ''


def test_rows_have_distinct_results_dirs():
    dirs = [load(f'{row}.json')['results_dir'] for row in ROWS]
    assert len(set(dirs)) == len(dirs)


@pytest.mark.parametrize('name', FILES)
def test_results_outside_the_repo(name):
    """Checkpoints (<results_dir>/models, every 50 iterations) and logs are written outside the git repo."""
    results_dir = load(name)['results_dir']
    assert results_dir.startswith(RUNS_DIR + '/')
    assert not os.path.abspath(results_dir).startswith(str(REPO) + os.sep)


def test_eval_set_covers_every_full_model():
    """The exclusion list is every model of data/full_models (decision 2026-11-24)."""
    models = load_eval_set(str(REPO / EVAL_SET))
    assert len(models) == len({m.name for m in models}) == 18
    assert {m.file for m in models} == {
        'albert_linalg', 'bart_linalg', 'bert_linalg', 'convnext_tiny_linalg', 'distilbert_linalg',
        'efficientnet_b0_linalg', 'gat_linalg', 'gin_linalg', 'gpt2_linalg', 'llama3_2_1b_linalg',
        'mobilenet_v3_small_linalg', 'resnet50_linalg', 'resnext50_linalg', 't5_linalg', 'vgg16_linalg',
        'vit_b_16_linalg', 'whisper_base_linalg', 'yolov8m_linalg'}
    assert all(m.reason for m in models if not m.runnable)
    assert len(runnable_models(str(REPO / EVAL_SET))) == 13


def test_generated_eval_json_only_has_runnable_models():
    path = REPO / BASELINES / 'eval_fullmodel.json'
    if not path.exists():
        pytest.skip('eval_fullmodel.json not generated yet (scripts/fusion_data/eval_set_tools.py eval-json)')
    runnable = {m.file for m in runnable_models(str(REPO / EVAL_SET))}
    assert set(json.loads(path.read_text())) <= runnable
