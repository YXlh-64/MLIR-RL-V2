"""Which benchmarks a training iteration explores (design doc §4.2, principle 3: weighting).

v5 explores `bench_count` benchmarks drawn uniformly without replacement
(`torch.randperm`, ppo.py collect_trajectory and distributed.py
collect_distributed_trajectory). With `fusion.sampling_weights` set to a JSON
file {benchmark name: weight} (written by scripts/fusion_data/build_units.py),
they are drawn without replacement with probability proportional to their
weight (`torch.multinomial`). Off (empty path): the exact v5 expression, so the
same RNG state gives the same indices.

Every loaded benchmark must have a weight > 0 (a missing name is an error, never
a silent fallback to uniform).
"""

import json
from functools import cache

import torch

from rl_autoschedular_v6.fusion.config import load_fusion_config


@cache
def _load_weights(path: str) -> dict[str, float]:
    with open(path) as f:
        return json.load(f)


def bench_weights(names: list[str], path: str) -> torch.Tensor:
    weights = _load_weights(path)
    missing = [n for n in names if weights.get(n, 0) <= 0]
    if missing:
        raise ValueError(f'{path}: no positive sampling weight for {len(missing)} benchmarks, e.g. {missing[:5]}')
    return torch.tensor([float(weights[n]) for n in names])


def sample_bench_indices(data, count: int) -> list[int]:
    """Indices of the benchmarks to explore this iteration (`data`: Benchmarks)."""
    path = load_fusion_config().sampling_weights
    if path:
        weights = bench_weights([b.bench_name for b in data.data], path)
        indices = torch.multinomial(weights, min(count, len(data)), replacement=False).long().tolist()
    else:
        indices = torch.randperm(len(data))[:count].long().tolist()
    if len(indices) < count:
        indices = (indices * count)[:count]
    return indices
