"""Edge-classification rules — the single source of truth (design rules R1–R6).

Everything that decides an op type or an edge class lives here, as plain
functions and constants with no MLIR dependency. The three view builders in
`views.py` (MLIR bindings, text parser, v6 OperationFeatures) only translate
their source into `OpView`s; they never decide anything themselves.

Map expressions use the loop names `d0..dN` (e.g. "d0", "d1 + d4", "0").
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class OpType(str, Enum):
    """Per-op type (design doc §2.2)."""
    ONE_TO_ONE = 'one_to_one'   # all parallel, every map identity
    BROADCAST = 'broadcast'     # all parallel, an input map omits a loop / uses a constant
    SHUFFLE = 'shuffle'         # all parallel, a non-identity permutation map
    REDUCTION = 'reduction'     # at least one reduction iterator
    WINDOW = 'window'           # a map term adds two loop dims (conv / pooling)
    INIT = 'init'               # only ever used as a consumer's `outs` operand
    UNKNOWN = 'unknown'         # maps or iterators not available


# R2: producer types whose edges may be green. Broadcast is included on purpose
# (DNNFusion "One-to-One with any type"; matches the A5 'ew' kind). This is
# effectively conditional: a broadcast feeding a contraction or conv is read
# with a `partial` / `halo` map there, so that edge is never green.
GREEN_PRODUCER_TYPES = frozenset({OpType.ONE_TO_ONE, OpType.BROADCAST})

# Many-to-Many producers (R3 red when read with a broadcast, R5 epilogue otherwise).
MANY_TO_MANY_TYPES = frozenset({OpType.REDUCTION, OpType.WINDOW})

# Edge classes.
GREEN = 'green'
RED = 'red'
INIT = 'init'
YELLOW_REASONS = ('epilogue', 'partial', 'halo', 'shuffle', 'unparsed')

# How a consumer reads one operand (§2.3).
ACCESS_FULL = 'full'        # pure dims covering all of the consumer's parallel loops
ACCESS_PARTIAL = 'partial'  # misses a parallel loop, or uses a constant / stride
ACCESS_HALO = 'halo'        # a term mixes two loop dims
ACCESS_INIT = 'init'        # the operand is an `outs` (init) operand

_DIM = re.compile(r'd\d+')


@dataclass(frozen=True)
class OpView:
    """Backend-independent description of one Linalg op.

    Attributes:
        name: op name as printed (e.g. "linalg.matmul").
        num_inputs: number of DPS inputs (operand index < num_inputs is an input).
        num_inits: number of DPS inits.
        iterators: 'parallel' / 'reduction' per loop, or None if unknown.
        input_maps: result expressions of each input map, or None if unknown.
        init_maps: result expressions of each init map, or None if unknown.
    """
    name: str
    num_inputs: int
    num_inits: int
    iterators: Optional[tuple[str, ...]]
    input_maps: Optional[tuple[tuple[str, ...], ...]]
    init_maps: Optional[tuple[tuple[str, ...], ...]]

    @property
    def known(self) -> bool:
        return self.iterators is not None and self.input_maps is not None and self.init_maps is not None


def normalize_expr(expr: str) -> str:
    """'%d0 +  %d3' -> 'd0 + d3' (feature strings carry '%')."""
    return ' '.join(expr.replace('%', '').split())


def is_pure_dim(expr: str) -> bool:
    return _DIM.fullmatch(expr) is not None


def is_halo_expr(expr: str) -> bool:
    return len(set(_DIM.findall(expr))) >= 2


def op_type(view: OpView) -> OpType:
    """Base op type from iterators and maps (R1). INIT is applied by `with_init_override`."""
    if not view.known:
        return OpType.UNKNOWN
    maps = view.input_maps + view.init_maps
    if any(is_halo_expr(e) for m in maps for e in m):
        return OpType.WINDOW
    if any(it != 'parallel' for it in view.iterators):
        return OpType.REDUCTION

    identity = tuple(f'd{i}' for i in range(len(view.iterators)))
    broadcast = shuffle = False
    for m in view.input_maps:
        if m == identity:
            continue
        if all(is_pure_dim(e) for e in m) and sorted(m) == sorted(identity):
            shuffle = True
        else:
            # omits a loop, uses a constant or a stride (strides fold here: both are green producers)
            broadcast = True
    if any(m != identity for m in view.init_maps):
        shuffle = True
    # DNNFusion order: One-to-One < Shuffle < One-to-Many, the more complex type wins.
    if broadcast:
        return OpType.BROADCAST
    if shuffle:
        return OpType.SHUFFLE
    return OpType.ONE_TO_ONE


def with_init_override(base: OpType, consumer_accesses: list[Optional[str]]) -> OpType:
    """INIT if the op has at least one Linalg consumer and all of them read it as init.

    Only Linalg consumers count (uses by return / reshape are ignored), so every
    backend reaches the same answer. INIT never changes an edge class: init
    edges are decided by the consumer's access alone.
    """
    if consumer_accesses and all(a == ACCESS_INIT for a in consumer_accesses):
        return OpType.INIT
    return base


def access(consumer: OpView, operand: int) -> Optional[str]:
    """How `consumer` reads its operand `operand` (index over inputs then inits)."""
    if operand >= consumer.num_inputs:
        return ACCESS_INIT if operand < consumer.num_inputs + consumer.num_inits else None
    if not consumer.known or operand >= len(consumer.input_maps):
        return None
    exprs = consumer.input_maps[operand]
    if any(is_halo_expr(e) for e in exprs):
        return ACCESS_HALO
    parallel = {f'd{i}' for i, it in enumerate(consumer.iterators) if it == 'parallel'}
    if all(is_pure_dim(e) for e in exprs) and parallel <= set(exprs):
        return ACCESS_FULL
    return ACCESS_PARTIAL


def classify_edge(producer_type: OpType, consumer_access: Optional[str]) -> str:
    """Edge class: 'green' | 'red' | 'init' | 'yellow:<reason>' (R2–R5).

    `producer_type` is the producer's base type (before the INIT override).
    """
    if consumer_access == ACCESS_INIT:
        return INIT                                                    # R4
    if producer_type in (OpType.UNKNOWN, OpType.INIT) or consumer_access is None:
        return 'yellow:unparsed'
    if producer_type in GREEN_PRODUCER_TYPES and consumer_access == ACCESS_FULL:
        return GREEN                                                   # R2
    if producer_type in MANY_TO_MANY_TYPES and consumer_access == ACCESS_PARTIAL:
        return RED                                                     # R3
    if producer_type == OpType.SHUFFLE:
        return 'yellow:shuffle'
    if producer_type in MANY_TO_MANY_TYPES:
        return 'yellow:epilogue' if consumer_access == ACCESS_FULL else 'yellow:halo'  # R5
    return f'yellow:{consumer_access}'                                 # R5 partial / halo
