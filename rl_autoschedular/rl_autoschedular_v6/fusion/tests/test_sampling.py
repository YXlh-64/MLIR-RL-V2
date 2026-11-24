"""fusion/sampling.py: off reproduces the v5 draw exactly; on draws by weight."""

import json
from dataclasses import dataclass

import pytest
import torch

from rl_autoschedular_v6.fusion import sampling
from rl_autoschedular_v6.fusion.config import FusionConfig


@dataclass
class FakeBench:
    bench_name: str


class FakeBenchmarks:
    def __init__(self, n: int):
        self.data = [FakeBench(f'b{i}') for i in range(n)]

    def __len__(self) -> int:
        return len(self.data)


def v5_indices(n: int, count: int) -> list[int]:
    """The expression ppo.collect_trajectory / distributed.py used before the hook."""
    indices = torch.randperm(n)[:count].long().tolist()
    if len(indices) < count:
        indices = (indices * count)[:count]
    return indices


@pytest.mark.parametrize('n,count', [(100, 64), (10, 64), (64, 64)])
def test_off_reproduces_v5(monkeypatch, n, count):
    monkeypatch.setattr(sampling, 'load_fusion_config', lambda: FusionConfig())
    torch.manual_seed(7)
    expected = v5_indices(n, count)
    torch.manual_seed(7)
    assert sampling.sample_bench_indices(FakeBenchmarks(n), count) == expected


def _weights_file(tmp_path, weights: dict) -> str:
    path = tmp_path / 'weights.json'
    path.write_text(json.dumps(weights))
    return str(path)


def test_on_draws_by_weight(monkeypatch, tmp_path):
    path = _weights_file(tmp_path, {'b0': 1000.0, 'b1': 1.0, 'b2': 1.0, 'b3': 1.0})
    monkeypatch.setattr(sampling, 'load_fusion_config', lambda: FusionConfig(sampling_weights=path))
    data = FakeBenchmarks(4)
    torch.manual_seed(0)
    firsts = [sampling.sample_bench_indices(data, 1)[0] for _ in range(200)]
    assert firsts.count(0) > 180
    assert sorted(sampling.sample_bench_indices(data, 4)) == [0, 1, 2, 3]   # without replacement
    assert len(sampling.sample_bench_indices(data, 10)) == 10                # padded like v5


def test_missing_weight_is_an_error(monkeypatch, tmp_path):
    path = _weights_file(tmp_path, {'b0': 1.0})
    monkeypatch.setattr(sampling, 'load_fusion_config', lambda: FusionConfig(sampling_weights=path))
    with pytest.raises(ValueError):
        sampling.sample_bench_indices(FakeBenchmarks(2), 2)
