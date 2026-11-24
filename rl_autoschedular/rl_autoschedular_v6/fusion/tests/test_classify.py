"""M1: expected classes per fixture, AstDumper agreement, backend agreement."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FIXTURE_NAMES, ast_dump, fixture_features, fixture_path, fixture_text, needs_ast_dumper
from rl_autoschedular_v6.fusion.classify import classify_graph, classify_text
from rl_autoschedular_v6.fusion.rules import OpType
from rl_autoschedular_v6.fusion.views import graph_from_features

T = OpType


def _op(i: int) -> str:
    return f'operation_{i}'


EXPECTED_EDGES = {
    'diamond_ew': {(0, 2): 'green', (1, 2): 'green'},
    'matmul_fill_bias': {(0, 2): 'yellow:partial', (1, 2): 'init', (2, 3): 'yellow:epilogue'},
    'reduce_broadcast': {(0, 1): 'init', (1, 2): 'red', (2, 3): 'green', (3, 5): 'green',
                         (4, 5): 'init', (5, 6): 'red', (3, 6): 'green'},
    'transpose': {(0, 1): 'green', (1, 2): 'yellow:shuffle'},
    'residual_add': {(0, 1): 'init', (1, 2): 'yellow:epilogue', (2, 3): 'green', (3, 5): 'yellow:partial',
                     (4, 5): 'init', (5, 6): 'yellow:epilogue', (6, 7): 'green', (3, 7): 'green'},
}
EXPECTED_TYPES = {
    'diamond_ew': [T.ONE_TO_ONE] * 3,
    'matmul_fill_bias': [T.ONE_TO_ONE, T.INIT, T.REDUCTION, T.BROADCAST],
    'reduce_broadcast': [T.INIT, T.REDUCTION, T.BROADCAST, T.ONE_TO_ONE, T.INIT, T.REDUCTION, T.BROADCAST],
    'transpose': [T.ONE_TO_ONE, T.SHUFFLE, T.ONE_TO_ONE],
    'residual_add': [T.INIT, T.REDUCTION, T.ONE_TO_ONE, T.ONE_TO_ONE] * 2,
}


def _classes(classification) -> dict[tuple[str, str, int], str]:
    return {(e.producer, e.consumer, e.operand): cls for e, cls in classification.edge_classes}


@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_expected_edge_classes(name):
    result = classify_text(fixture_text(name), backend='bindings')
    got = {(e.producer, e.consumer): cls for e, cls in result.edge_classes}
    assert got == {(_op(p), _op(c)): cls for (p, c), cls in EXPECTED_EDGES[name].items()}


@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_expected_op_types(name):
    result = classify_text(fixture_text(name), backend='bindings')
    assert list(result.op_types.values()) == EXPECTED_TYPES[name]


@needs_ast_dumper
@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_tags_and_edges_match_ast_dumper(name):
    names, edges = ast_dump(fixture_path(name))
    graph = classify_text(fixture_text(name), backend='bindings').graph
    assert {tag: view.name for tag, view in graph.ops.items()} == names
    assert {(e.producer, e.result, e.consumer, e.operand) for e in graph.edges} == edges


@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_text_backend_agrees_or_is_unparsed(name):
    """The text fallback gives the same class, or 'unparsed' where a named op hides its maps."""
    bindings = _classes(classify_text(fixture_text(name), backend='bindings'))
    text = _classes(classify_text(fixture_text(name), backend='text'))
    assert text.keys() == bindings.keys()
    for key, cls in text.items():
        assert cls in (bindings[key], 'yellow:unparsed'), key
    if name in ('diamond_ew', 'reduce_broadcast'):  # only generics / fills: must agree exactly
        assert text == bindings


@needs_ast_dumper
@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_features_backend_agrees_with_bindings(name):
    """M4/M5 classify from OperationFeatures; they must match M1/M2 (bindings) on every edge."""
    bindings = _classes(classify_text(fixture_text(name), backend='bindings'))
    features = _classes(classify_graph(graph_from_features(fixture_features(name))))
    assert features == bindings


def test_cli_writes_json(tmp_path):
    out = tmp_path / 'classes.json'
    subprocess.run([sys.executable, '-m', 'rl_autoschedular_v6.fusion.classify',
                    fixture_path('matmul_fill_bias'), '--out', str(out)], check=True, capture_output=True)
    data = json.loads(out.read_text())['matmul_fill_bias.mlir']
    assert data['backend'] == 'bindings'
    assert data['counts'] == {'yellow:partial': 1, 'init': 1, 'yellow:epilogue': 1}
    assert data['ops']['operation_1'] == {'name': 'linalg.fill', 'type': 'init'}


@needs_ast_dumper
@pytest.mark.parametrize('name', ['llama3_2_1b_block_71', 'bert_block_1017', 'resnet50_block_0'])
def test_real_blocks_match_ast_dumper(name):
    """Real blocks (read-only), incl. a dead fill that a rewrite driver would erase."""
    path = os.path.join('/scratch/mh8826/data/ops_and_blocks', f'{name}.mlir')
    if not os.path.isfile(path):
        pytest.skip(f'{path} not available')
    names, edges = ast_dump(path)
    with open(path) as f:
        graph = classify_text(f.read(), backend='bindings').graph
    assert {tag: view.name for tag, view in graph.ops.items()} == names
    assert {(e.producer, e.result, e.consumer, e.operand) for e in graph.edges} == edges
