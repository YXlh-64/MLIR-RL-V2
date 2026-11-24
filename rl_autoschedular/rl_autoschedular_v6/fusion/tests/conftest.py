"""Shared test helpers: fixture paths, AstDumper output, v6 features of a fixture.

Run from the repo root after `source scripts/analysis/gap1/env.sh`.
"""

from pathlib import Path
import os
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
FIXTURE_NAMES = ['diamond_ew', 'matmul_fill_bias', 'reduce_broadcast', 'transpose', 'residual_add']

sys.path[:0] = [str(REPO), str(REPO / 'rl_autoschedular')]
# Config singletons (v5 and v6) read this file: v5_single_node.json pointed at the fixtures,
# with `fusion: {}` (all off). Its paths are relative to the repo root.
os.chdir(REPO)
os.environ['CONFIG_FILE_PATH'] = str(FIXTURES / 'test_config.json')

AST_DUMPER = os.getenv('AST_DUMPER_BIN_PATH', '')
needs_ast_dumper = pytest.mark.skipif(not os.path.isfile(AST_DUMPER), reason='AstDumper binary not available')


def fixture_path(name: str) -> str:
    return str(FIXTURES / f'{name}.mlir')


def fixture_text(name: str) -> str:
    return Path(fixture_path(name)).read_text()


def ast_dump(path: str) -> tuple[dict[str, str], set[tuple[str, int, str, int]]]:
    """({tag: op name}, {(producer, result, consumer, operand)}) straight from the AstDumper."""
    out = subprocess.run([AST_DUMPER, path], capture_output=True, text=True, check=True).stdout
    info = out.split('########################################')[0]
    ops_part, graph_part = info.split('#BEGIN_GRAPH')
    names = {}
    for block in ops_part.split('#START_OPERATION')[1:]:
        name = block.strip().split('\n')[0]
        tag = block.split('#START_TAG')[1].strip().split('\n')[0]
        names[tag] = name
    edges = set()
    for line in graph_part.replace('#END_GRAPH', '').strip().splitlines():
        (p, r), (c, k) = (side.split() for side in line.split(' --> '))
        edges.add((p, int(r), c, int(k)))
    return names, edges


def fixture_features(name: str, root_exec_time: int = 1):
    """v6 BenchmarkFeatures of a fixture (exactly what Benchmarks loads)."""
    from rl_autoschedular_v6.state import extract_bench_features_from_file
    return extract_bench_features_from_file(name, fixture_path(name), root_exec_time)
