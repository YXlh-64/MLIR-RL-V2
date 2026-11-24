"""M2: green-only pre-fusion on fixtures and (read-only) real blocks. Nothing is executed."""

import json
import os
import subprocess
import sys

import pytest

from conftest import FIXTURE_NAMES, ast_dump, fixture_path, fixture_text, needs_ast_dumper
from rl_autoschedular_v6.fusion.classify import classify_text
from rl_autoschedular_v6.fusion.prefuse import prefuse_code, prefuse_file

# name: (ops_before, ops_after, fused, green_not_fused)
EXPECTED = {
    'diamond_ew': (3, 1, 2, {}),
    'matmul_fill_bias': (4, 4, 0, {}),
    'reduce_broadcast': (7, 6, 1, {'multi_use': 2}),   # R7 conservative: exp has 2 green consumers
    'transpose': (3, 3, 0, {'named_op': 1}),            # the pass does not fuse into linalg.transpose
    'residual_add': (8, 6, 2, {'pinned_producer': 1}),  # R7: op3 also feeds a matmul (partial)
}
# whisper_195 reads one value twice in one op; llama_71 has a dead fill (kept, as the AstDumper numbers it).
REAL_BLOCKS = ['bert_block_1017', 'whisper_base_block_195', 'albert_block_445', 'llama3_2_1b_block_71']
REAL_DIR = '/scratch/mh8826/data/ops_and_blocks'


@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_report(name):
    _, report = prefuse_code(fixture_text(name))
    before, after, fused, not_fused = EXPECTED[name]
    assert (report.ops_before, report.ops_after, report.fused) == (before, after, fused)
    assert report.green_not_fused == not_fused
    assert report.unexpected_removed == 0


def test_non_green_edges_survive():
    """Red and init edges are never fused: fills and both reductions stay, and so do the red pairs."""
    fused, _ = prefuse_code(fixture_text('reduce_broadcast'))
    result = classify_text(fused, backend='bindings')
    names = [v.name for v in result.graph.ops.values()]
    assert names.count('linalg.fill') == 2
    assert result.counts().get('red') == 2
    assert result.counts().get('init') == 2


def test_output_has_no_pins_or_tags():
    fused, _ = prefuse_code(fixture_text('residual_add'))
    assert 'fusion.keep' not in fused and 'fusion.tag' not in fused


@needs_ast_dumper
@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_ast_dumper_reads_fused_output(name, tmp_path):
    fused, report = prefuse_code(fixture_text(name))
    path = tmp_path / f'{name}.mlir'
    path.write_text(fused)
    names, _ = ast_dump(str(path))
    assert len(names) == report.ops_after


def test_never_in_place(tmp_path):
    src = tmp_path / 'in.mlir'
    src.write_text(fixture_text('diamond_ew'))
    with pytest.raises(ValueError):
        prefuse_file(str(src), str(src))
    dst = tmp_path / 'out.mlir'
    dst.write_text('existing')
    with pytest.raises(FileExistsError):
        prefuse_file(str(src), str(dst))
    assert src.read_text() == fixture_text('diamond_ew')


def test_cli_writes_output_and_report(tmp_path):
    out, report = tmp_path / 'out.mlir', tmp_path / 'report.json'
    subprocess.run([sys.executable, '-m', 'rl_autoschedular_v6.fusion.prefuse', fixture_path('diamond_ew'),
                    str(out), '--report', str(report)], check=True, capture_output=True)
    assert json.loads(report.read_text())['ops_after'] == 1
    assert out.exists()


@pytest.mark.parametrize('name', REAL_BLOCKS)
def test_real_blocks_only_fuse_green(name):
    path = os.path.join(REAL_DIR, f'{name}.mlir')
    if not os.path.isfile(path):
        pytest.skip(f'{path} not available')
    with open(path) as f:
        _, report = prefuse_code(f.read())
    assert report.unexpected_removed == 0
    assert report.ops_after == report.ops_before - report.fused
