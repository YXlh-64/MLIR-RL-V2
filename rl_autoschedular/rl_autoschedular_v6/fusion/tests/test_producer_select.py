"""M4: producer selection on fixtures, A8 statistics, and the Env hook (flag on / off)."""

import pytest

from conftest import FIXTURE_NAMES, fixture_features, needs_ast_dumper
from rl_autoschedular_v6.fusion.config import FusionConfig
from rl_autoschedular_v6.fusion.producer_select import select_producer, selection_stats

# (fixture, consumer) -> expected selection; the comment gives v5's producers[-1]
EXPECTED = {
    ('matmul_fill_bias', 'operation_2'): ('operation_0', 0),   # v5: the fill (init)
    ('matmul_fill_bias', 'operation_3'): ('operation_2', 0),   # epilogue, unchanged
    ('residual_add', 'operation_1'): None,                     # v5: fill (init) -> TF not offered
    ('residual_add', 'operation_5'): ('operation_3', 0),       # v5: fill (init); partial edge kept
    ('residual_add', 'operation_7'): ('operation_6', 0),       # tie in size -> later producer, as v5
    ('reduce_broadcast', 'operation_2'): None,                 # v5: rowmax (red)
    ('reduce_broadcast', 'operation_5'): ('operation_3', 0),   # v5: fill (init)
    ('reduce_broadcast', 'operation_6'): ('operation_3', 0),   # v5: rowsum (red)
    ('diamond_ew', 'operation_2'): ('operation_1', 1),         # tie -> later producer, as v5
    ('transpose', 'operation_2'): ('operation_1', 0),          # shuffle (yellow) stays allowed
}


@needs_ast_dumper
@pytest.mark.parametrize('fixture,consumer', list(EXPECTED))
def test_select_producer(fixture, consumer):
    bench = fixture_features(fixture)
    assert select_producer(bench.operations[consumer], consumer, bench) == EXPECTED[(fixture, consumer)]


@needs_ast_dumper
def test_selection_stats_answer_a8():
    stats = selection_stats([fixture_features(n) for n in FIXTURE_NAMES]).as_dict()
    assert stats == {'ops_with_producers': 16, 'multi_producer_ops': 6, 'original_last_init': 5,
                     'original_last_red': 2, 'changed': 7, 'no_data_producer': 3}


@needs_ast_dumper
def test_selection_does_not_modify_features():
    bench = fixture_features('matmul_fill_bias')
    before = bench.copy()
    selection_stats([bench])
    assert bench == before


def _states(env, benchs, bench_idx):
    state = env.reset(benchs, bench_idx)
    while state is not None:
        yield state
        state = env.get_next_op_state(state)


@needs_ast_dumper
def test_env_hook(monkeypatch):
    from rl_autoschedular_v6 import env as env_module
    from rl_autoschedular_v6.actions import TiledFusion
    from rl_autoschedular_v6.benchmarks import Benchmarks

    benchs = Benchmarks(is_training=True)
    idx = [b.bench_name for b in benchs.data].index('matmul_fill_bias')
    env = env_module.Env()

    off = {s.operation_tag: s.producer_tag for s in _states(env, benchs, idx)}
    assert off['operation_2'] == 'operation_1'                    # v5 behaviour: the fill

    monkeypatch.setattr(env_module, 'load_fusion_config', lambda: FusionConfig(producer_select=True))
    states = {s.operation_tag: s for s in _states(env, benchs, idx)}
    matmul = states['operation_2']
    assert (matmul.producer_tag, matmul.producer_operand_idx) == ('operation_0', 0)
    assert states['operation_1'].producer_tag is None             # fill has no producer
    assert not TiledFusion.is_allowed(states['operation_0'])     # relu reads only args

    # TF on the selected producer goes through the existing feature update without failing.
    env.reset(benchs, idx)
    stepped = env.step(matmul, TiledFusion([2, 2, 0], matmul))
    assert not stepped.terminal
    assert stepped.operation_features.producers == [('operation_0_operation_2', 0)]
