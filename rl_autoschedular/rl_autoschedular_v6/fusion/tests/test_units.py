"""M3: segmentation, signatures, closed-unit extraction, manifest, leakage guard. Nothing is executed."""

import json
import subprocess
import sys

import pytest

from conftest import fixture_path, fixture_text, needs_ast_dumper
from rl_autoschedular_v6.fusion.classify import classify_text
from rl_autoschedular_v6.fusion.unit_extract import extract_unit
from rl_autoschedular_v6.fusion.units import EvalModelLeakage, UnitParams, analyze, build_units
from rl_autoschedular_v6.full_model import _is_timed_main

STACK = 'residual_stack'


def test_repeated_blocks_share_one_signature():
    analysis = analyze(fixture_text(STACK), UnitParams(min_ops=1))
    spans = [(s.start, s.end) for s in analysis.segments]
    assert spans == [(0, 1), (1, 4), (4, 7), (7, 10)]           # stem + 3 blocks (fills are free), cut where width <= 1
    signatures = [s.signature for s in analysis.segments]
    assert signatures[1] == signatures[2] == signatures[3] != signatures[0]


def test_split_and_merge():
    split = analyze(fixture_text(STACK), UnitParams(min_ops=1, max_ops=1)).segments
    assert [(s.start, s.end, s.est_ops) for s in split][1:3] == [(1, 2, 1), (2, 4, 1)]  # matmul | relu+add
    merged = analyze(fixture_text(STACK), UnitParams(min_ops=5, max_ops=25)).segments
    assert [(s.start, s.end) for s in merged] == [(0, 7), (7, 10)]


def test_unit_is_closed_and_timed():
    analysis = analyze(fixture_text(STACK), UnitParams(min_ops=1))
    unit = extract_unit(fixture_text(STACK), analysis.positions(analysis.segments[1]))
    assert _is_timed_main(unit)
    graph = classify_text(unit, backend='bindings').graph
    # the shared zero-init fill is free: cloned into the unit, not an argument
    assert [v.name for v in graph.ops.values()] == ['linalg.fill', 'linalg.matmul', 'linalg.generic', 'linalg.generic']
    assert len(graph.edges) == 3                                 # every internal edge kept
    assert 'func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<64x64xf32>)' in unit  # input + weight


def test_all_live_outs_returned():
    """Stem + matmul + relu of block 1: the stem (read by the residual add) and relu are live-outs."""
    analysis = analyze(fixture_text(STACK), UnitParams(min_ops=1))
    unit = extract_unit(fixture_text(STACK), analysis.body_positions[0:3])
    assert '-> (tensor<32x64xf32>, tensor<32x64xf32>, i64)' in unit


@needs_ast_dumper
def test_unit_loads_as_v6_benchmark():
    from rl_autoschedular_v6.state import extract_bench_features_from_code
    analysis = analyze(fixture_text(STACK), UnitParams(min_ops=1))
    unit = extract_unit(fixture_text(STACK), analysis.positions(analysis.segments[1]))
    bench = extract_bench_features_from_code('unit', unit, 1)
    assert len(bench.operation_tags) == 4
    assert bench.operations['operation_1'].producers == [('operation_0', 2)]


def test_build_units_and_manifest(tmp_path):
    entry = build_units(fixture_path(STACK), str(tmp_path), UnitParams(min_ops=1))
    weights = sorted(u['weight'] for u in entry['units'])
    assert weights == [1, 3]
    block = next(u for u in entry['units'] if u['weight'] == 3)
    assert block['ops_raw'] == 4 and block['ops_after_green_fusion'] == 3
    assert block['edge_classes'] == {'init': 1, 'yellow:epilogue': 1, 'green': 1}
    assert block['occurrences'] == [[1, 4], [4, 7], [7, 10]]
    assert entry['skipped'] == [] and entry['missing_op_names'] == []
    assert (tmp_path / block['file']).exists()
    with pytest.raises(FileExistsError):
        build_units(fixture_path(STACK), str(tmp_path), UnitParams(min_ops=1))


def test_eval_model_leakage_guard(tmp_path, capsys):
    eval_set = tmp_path / 'eval_set.json'
    eval_set.write_text(json.dumps({'models': [{'name': STACK, 'file': f'{STACK}_linalg', 'runnable': True,
                                                'reason': '', 'survey': None}]}))
    params = UnitParams(min_ops=1, eval_set=str(eval_set))
    with pytest.raises(EvalModelLeakage):
        build_units(fixture_path(STACK), str(tmp_path), params)
    build_units(fixture_path(STACK), str(tmp_path), params, allow_eval_models=True)
    assert 'WARNING' in capsys.readouterr().out


def test_params_reject_unknown_keys():
    with pytest.raises(ValueError):
        UnitParams.from_dict({'max_op': 3})


def test_cli(tmp_path):
    config = tmp_path / 'cfg.json'
    config.write_text(json.dumps({'fusion': {'units': {'min_ops': 1}}}))
    out = tmp_path / 'units'
    subprocess.run([sys.executable, '-m', 'rl_autoschedular_v6.fusion.units', fixture_path(STACK),
                    '--out', str(out), '--config', str(config)], check=True, capture_output=True)
    manifest = json.loads((out / 'units_manifest.json').read_text())
    assert manifest['params']['min_ops'] == 1
    assert len(manifest['models'][STACK]['units']) == 2
