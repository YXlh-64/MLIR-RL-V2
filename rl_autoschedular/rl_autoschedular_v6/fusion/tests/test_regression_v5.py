"""MUST FIX 3: v6 with the fusion layer off behaves exactly like v5.

Compares, for v5 and v6: extracted features (fixtures and a few real blocks),
the full `Benchmarks` load (test config, `fusion: {}`), and the action masks of
every op's initial state and of the state after a Tiling step.
"""

import importlib
import os

import pytest
import torch

from conftest import FIXTURE_NAMES, fixture_path, needs_ast_dumper

PACKAGES = ('rl_autoschedular_v5', 'rl_autoschedular_v6')
REAL_BLOCKS = ['bert_block_1017', 'whisper_base_block_195', 'add_112_112_14_15']
REAL_DIR = '/scratch/mh8826/data/ops_and_blocks'


def plain(bench) -> dict:
    """Package-independent view of a BenchmarkFeatures (enums -> values, actions -> str)."""
    return {
        'name': bench.bench_name, 'code': bench.code, 'tags': bench.operation_tags, 'root': bench.root_exec_time,
        'ops': {tag: {
            'name': op.operation_name, 'type': op.operation_type.value, 'op_count': op.op_count,
            'load': op.load_data, 'store': op.store_data,
            'loops': [(l.arg, l.lower_bound, l.upper_bound, l.step, l.iterator_type.value) for l in op.nested_loops],
            'producers': op.producers, 'consumers': op.consumers, 'vectorizable': op.vectorizable,
            'pre_actions': [str(a) for a in op.pre_actions],
        } for tag, op in bench.operations.items()},
    }


def features(package: str, name: str, path: str) -> dict:
    state = importlib.import_module(f'{package}.state')
    return plain(state.extract_bench_features_from_file(name, path, 1000))


def load_benchmarks(package: str):
    return importlib.import_module(f'{package}.benchmarks').Benchmarks(is_training=True)


def all_masks(package: str, benchs) -> list[torch.Tensor]:
    env_module = importlib.import_module(f'{package}.env')
    actions = importlib.import_module(f'{package}.actions')
    env, masks = env_module.Env(), []
    for idx in range(len(benchs)):
        state = env.reset(benchs, idx)
        while state is not None:
            masks.append(actions.ActionSpace.action_mask(state))
            loops = state.operation_features.nested_loops
            params = [1 if i == next((j for j, l in enumerate(loops) if l.upper_bound > 1), -1) else 0
                      for i in range(len(loops))]
            masks.append(actions.ActionSpace.action_mask(env.step(state, actions.Tiling(params, state))))
            state = env.get_next_op_state(state)
    return masks


@needs_ast_dumper
@pytest.mark.parametrize('name', FIXTURE_NAMES)
def test_fixture_features_identical(name):
    v5, v6 = (features(p, name, fixture_path(name)) for p in PACKAGES)
    assert v5 == v6


@needs_ast_dumper
@pytest.mark.parametrize('name', REAL_BLOCKS)
def test_real_block_features_identical(name):
    path = os.path.join(REAL_DIR, f'{name}.mlir')
    if not os.path.isfile(path):
        pytest.skip(f'{path} not available')
    v5, v6 = (features(p, name, path) for p in PACKAGES)
    assert v5 == v6


@needs_ast_dumper
def test_benchmarks_and_action_masks_identical():
    v5, v6 = (load_benchmarks(p) for p in PACKAGES)
    assert [plain(b) for b in v5.data] == [plain(b) for b in v6.data]
    masks5, masks6 = all_masks(PACKAGES[0], v5), all_masks(PACKAGES[1], v6)
    assert len(masks5) == len(masks6) > 0
    assert all(torch.equal(a, b) for a, b in zip(masks5, masks6))
