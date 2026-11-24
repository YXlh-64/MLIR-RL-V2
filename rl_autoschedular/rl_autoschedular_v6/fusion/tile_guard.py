"""M5 — tile guard for Tiled Fusion (design rule R10: legal does not mean profitable).

Masks TF tile sizes that are likely to lose:
- recomputation (partial edges only): the producer slice is recomputed once per
  tile of every consumer parallel loop that the producer map misses. The total
  is the product over those k loops of their tile counts, so each such loop is
  capped at `max_recompute ** (1/k)` tiles; the product stays <= `max_recompute`;
- footprint: the fused output tile (product of the parallel tile sizes) should
  fit in L2. Each of the k parallel loops is capped at
  `(l2_bytes / element_bytes) ** (1/k)`.

The TF mask is factored per loop (each loop's size is sampled independently),
so only per-loop caps can be expressed; the caps above are the per-loop budgets
whose product respects the joint limit. Column 0 (do not tile this loop) is
never masked, as in v5. Masked sizes are only removed from what Tiling already
allows: the guard is AND-ed with `Tiling.action_mask`.
"""

from dataclasses import dataclass, fields
from typing import Any

import torch

from rl_autoschedular_v6.fusion.producer_select import edge_class
from rl_autoschedular_v6.state import IteratorType, OperationFeatures


@dataclass(frozen=True)
class TileGuardParams:
    max_recompute: float = 8.0     # max producer recomputations per fused TF (partial edges)
    l2_bytes: int = 1 << 20        # L2 budget per core (assumption: 1 MiB, set per machine)
    element_bytes: int = 4         # f32

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> 'TileGuardParams':
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f'unknown fusion.tile_guard_params keys: {sorted(unknown)} (known: {sorted(known)})')
        return cls(**section)


def missed_parallel_loops(consumer: OperationFeatures, operand: int) -> list[str]:
    """Consumer parallel loops (args) absent from its map for `operand`."""
    accesses = consumer.load_data + consumer.store_data
    used = {term.strip() for expr in accesses[operand] for term in expr.split()}
    return [loop.arg for loop in consumer.nested_loops
            if loop.iterator_type == IteratorType.Parallel and loop.arg not in used]


def allowed_tf_tiles(consumer: OperationFeatures, producer: OperationFeatures, operand: int,
                     params: TileGuardParams, max_loops: int, num_tile_sizes: int) -> torch.Tensor:
    """Bool mask [max_loops, num_tile_sizes + 1]; column j >= 1 is tile size 2 ** (j - 1)."""
    mask = torch.ones((max_loops, num_tile_sizes + 1), dtype=torch.bool)
    loops = consumer.nested_loops
    parallel = [loop.arg for loop in loops if loop.iterator_type == IteratorType.Parallel and loop.upper_bound > 1]
    footprint_cap = (params.l2_bytes / params.element_bytes) ** (1 / max(len(parallel), 1))
    missed = missed_parallel_loops(consumer, operand) if edge_class(consumer, producer, operand) == 'yellow:partial' else []
    tiles_cap = params.max_recompute ** (1 / max(len(missed), 1))

    for i, loop in enumerate(loops[:max_loops]):
        for j in range(1, num_tile_sizes + 1):
            size = 2 ** (j - 1)
            too_big = loop.arg in parallel and size > footprint_cap
            too_many_tiles = loop.arg in missed and loop.upper_bound / size > tiles_cap
            mask[i, j] = not (too_big or too_many_tiles)
    return mask
