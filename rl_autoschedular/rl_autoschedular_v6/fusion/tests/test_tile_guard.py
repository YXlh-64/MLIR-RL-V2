"""M5: tile-guard caps on fixture features, and the TiledFusion.action_mask hook."""

import pytest
import torch

from conftest import fixture_features, needs_ast_dumper
from rl_autoschedular_v6.fusion.config import FusionConfig
from rl_autoschedular_v6.fusion.tile_guard import TileGuardParams, allowed_tf_tiles, missed_parallel_loops

MAX_LOOPS, NUM_SIZES = 7, 7   # test config values; column j >= 1 is tile size 2 ** (j - 1)


def _guard(fixture, consumer, producer, operand, **params):
    bench = fixture_features(fixture)
    return allowed_tf_tiles(bench.operations[consumer], bench.operations[producer], operand,
                            TileGuardParams(**params), MAX_LOOPS, NUM_SIZES)


@needs_ast_dumper
def test_recompute_cap_on_partial_edge():
    """relu -> matmul LHS (partial): the LHS map misses d1 (ub 128); <= 8 tiles means size >= 16."""
    bench = fixture_features('matmul_fill_bias')
    assert missed_parallel_loops(bench.operations['operation_2'], 0) == ['%d1']
    mask = _guard('matmul_fill_bias', 'operation_2', 'operation_0', 0)
    assert mask[1].tolist() == [True, False, False, False, False, True, True, True]
    assert mask[0].all() and mask[2].all()       # d0 is read; d2 is a reduction


@needs_ast_dumper
def test_no_recompute_cap_on_epilogue():
    mask = _guard('matmul_fill_bias', 'operation_3', 'operation_2', 0)
    assert mask.all()


@needs_ast_dumper
def test_footprint_cap():
    """L2 of 256 f32 over 2 parallel loops: each tile <= 16 (column 0 = untiled stays allowed)."""
    mask = _guard('matmul_fill_bias', 'operation_3', 'operation_2', 0, l2_bytes=1024)
    for loop in (0, 1):
        assert mask[loop].tolist() == [True, True, True, True, True, True, False, False]


def test_params_reject_unknown_keys():
    with pytest.raises(ValueError):
        TileGuardParams.from_dict({'l2': 1})


@needs_ast_dumper
def test_action_mask_hook(monkeypatch):
    from rl_autoschedular_v6 import env as env_module
    from rl_autoschedular_v6.actions import TiledFusion
    from rl_autoschedular_v6.actions import tiled_fusion as tf_module
    from rl_autoschedular_v6.benchmarks import Benchmarks

    benchs = Benchmarks(is_training=True)
    idx = [b.bench_name for b in benchs.data].index('matmul_fill_bias')
    monkeypatch.setattr(env_module, 'load_fusion_config', lambda: FusionConfig(producer_select=True))
    env = env_module.Env()
    state = env.reset(benchs, idx)
    while state.operation_tag != 'operation_2':
        state = env.get_next_op_state(state)

    off = TiledFusion.action_mask(state)
    monkeypatch.setattr(tf_module, 'load_fusion_config', lambda: FusionConfig(producer_select=True, tile_guard=True))
    on = TiledFusion.action_mask(state)
    guard = allowed_tf_tiles(state.operation_features, state.producer_features, state.producer_operand_idx,
                             TileGuardParams(), MAX_LOOPS, NUM_SIZES).reshape(-1)
    assert torch.equal(on, off & guard)
    assert not torch.equal(on, off)
    assert on.reshape(MAX_LOOPS, NUM_SIZES + 1)[:, 0].all()
