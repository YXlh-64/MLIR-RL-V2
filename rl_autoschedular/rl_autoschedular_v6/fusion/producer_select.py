"""M4 — producer selection for Tiled Fusion (design rules R3, R4, R8).

v5 gives TF the lexically last producer (`producers[-1]`, env.py), which is
often the init fill. Instead, `select_producer` returns the best *data* producer:
1. drop init producers (R4) and red producers (R3);
2. take the largest intermediate result (most memory traffic saved), with ties
   going to the lexically later producer (v5's preference).

Scope: no loaded feature is modified. `Env` calls `select_producer` in place of
its `producers[-1]` lookup (hook in env.py, behind `fusion.producer_select`), so
only TiledFusion's view (`state.producer_tag` / `producer_operand_idx`)
changes. `selection_stats` is read-only and answers open check A8 (how often
`producers[-1]` is an init producer).
"""

from dataclasses import asdict, dataclass
from typing import Optional

from rl_autoschedular_v6.fusion.rules import INIT, RED, access, classify_edge, op_type
from rl_autoschedular_v6.fusion.views import feature_view
from rl_autoschedular_v6.state import BenchmarkFeatures, OperationFeatures

EXCLUDED_CLASSES = (INIT, RED)


def edge_class(consumer: OperationFeatures, producer: OperationFeatures, operand: int) -> str:
    """Class of the edge producer -> consumer operand `operand` (aligned feature index)."""
    return classify_edge(op_type(feature_view(producer)), access(feature_view(consumer), operand))


def intermediate_size(producer: OperationFeatures, consumer_tag: str) -> int:
    """Elements of the producer result read by `consumer_tag` (product of its stored loop bounds)."""
    results = [res for tag, res in producer.consumers if tag == consumer_tag]
    store = producer.store_data[results[0] if results else 0] if producer.store_data else []
    bounds = {loop.arg: loop.upper_bound for loop in producer.nested_loops}
    size = 1
    for expr in store:
        size *= bounds.get(expr.strip(), 1)
    return size


def select_producer(consumer: OperationFeatures, consumer_tag: str,
                    bench: BenchmarkFeatures) -> Optional[tuple[str, int]]:
    """(producer tag, operand index) of the best data producer, or None if there is none."""
    candidates = []
    for order, (tag, operand) in enumerate(consumer.producers):
        producer = bench.operations.get(tag)
        if producer is None or edge_class(consumer, producer, operand) in EXCLUDED_CLASSES:
            continue
        candidates.append((intermediate_size(producer, consumer_tag), order, tag, operand))
    if not candidates:
        return None
    _, _, best_tag, _ = max(candidates)
    return best_tag, min(operand for _, _, tag, operand in candidates if tag == best_tag)


@dataclass
class SelectionStats:
    ops_with_producers: int = 0
    multi_producer_ops: int = 0       # >= 2 distinct producer tags
    original_last_init: int = 0       # producers[-1] is an init producer (A8)
    original_last_red: int = 0        # producers[-1] is a red producer
    changed: int = 0                  # selection differs from producers[-1]
    no_data_producer: int = 0         # all producers excluded: TF not offered

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def selection_stats(benchmarks: list[BenchmarkFeatures]) -> SelectionStats:
    """Read-only statistics of the selection over loaded benchmarks (initial op states)."""
    stats = SelectionStats()
    for bench in benchmarks:
        for tag in bench.operation_tags:
            op = bench.operations[tag]
            if not op.producers:
                continue
            stats.ops_with_producers += 1
            stats.multi_producer_ops += len({t for t, _ in op.producers}) >= 2
            last_tag = op.producers[-1][0]
            last_operand = min(i for t, i in op.producers if t == last_tag)
            last_class = edge_class(op, bench.operations[last_tag], last_operand)
            stats.original_last_init += last_class == INIT
            stats.original_last_red += last_class == RED
            selected = select_producer(op, tag, bench)
            stats.no_data_producer += selected is None
            stats.changed += selected != (last_tag, last_operand)
    return stats
