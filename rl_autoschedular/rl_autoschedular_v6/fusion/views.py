"""Builders that turn a program into a `ProgramGraph` of `OpView`s (supports R1).

Three sources, one representation, so `rules.py` stays the only place that decides:
- `graph_from_bindings`: MLIR Python bindings (default). Named ops are generalized
  on a private copy to read their indexing maps.
- `graph_from_text`: fallback, reuses the stdlib text parser of
  `scripts/analysis/gap1/_mlir_text.py` (parsing only, not its classifier).
- `graph_from_features`: v6 `BenchmarkFeatures` (AstDumper output), used at
  training time by producer selection (M4) and the tile guard (M5).

Tags follow the AstDumper (`tools/ast_dumper/main.cpp`): Linalg ops in walk
order, ops with zero loops skipped, numbered `operation_<N>`; edge operand
index runs over ins then outs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import importlib.util
import re

from rl_autoschedular_v6.fusion.rules import OpView, normalize_expr

if TYPE_CHECKING:
    from rl_autoschedular_v6.state import BenchmarkFeatures, OperationFeatures

# `linalg.*` ops that are not structured (not LinalgOp): the AstDumper never sees them.
NON_STRUCTURED = frozenset({
    'linalg.yield', 'linalg.index', 'linalg.softmax', 'linalg.pack', 'linalg.unpack',
    'linalg.winograd_filter_transform', 'linalg.winograd_input_transform',
    'linalg.winograd_output_transform',
})
TAG_ATTR = 'fusion.tag'
KEEP_OP = 'fusion.keep'
_GAP1_TEXT_PARSER = Path(__file__).resolve().parents[3] / 'scripts' / 'analysis' / 'gap1' / '_mlir_text.py'


class BackendError(RuntimeError):
    """The bindings backend cannot reproduce AstDumper numbering for this program."""


@dataclass(frozen=True)
class Edge:
    """Producer result `result` feeds operand `operand` of `consumer` (-1: result unknown)."""
    producer: str
    consumer: str
    operand: int
    result: int


@dataclass
class ProgramGraph:
    ops: dict[str, OpView]  # tag -> view, in AstDumper order
    edges: list[Edge]
    backend: str


def is_structured(op) -> bool:
    return op.name.startswith('linalg.') and op.name not in NON_STRUCTURED


def walk_ops(root, keep) -> list:
    """All ops nested in `root` (a module or op; post-order, like the AstDumper's walk) for which `keep(op)` holds."""
    from mlir.ir import WalkResult
    found = []

    def visit(op):
        if keep(op):
            found.append(op)
        return WalkResult.ADVANCE

    root.operation.walk(visit)
    return found


def pin_values(ops: list) -> None:
    """Give every result of `ops` an extra use by an unregistered `fusion.keep` op, placed
    before the function terminator, so rewrite drivers neither erase nor fuse them.
    Only results defined at function top level are pinned. Must be called inside a Context."""
    from mlir.ir import InsertionPoint, Location, Operation

    with Location.unknown():
        for op in ops:
            parent = op.parent
            if parent.operation.name != 'func.func':
                continue
            with InsertionPoint.at_block_terminator(parent.regions[0].blocks[0]):
                for result in op.results:
                    Operation.create(KEEP_OP, operands=[result])


def tag_module(module) -> tuple[ProgramGraph, dict]:
    """Tag the structured ops of `module` in place (`fusion.tag` attribute) and build its graph.

    Maps are read from a generalized re-parse (`linalg-generalize-named-ops`
    rewrites ops 1:1 in place, so walk positions match). Every structured op of
    the copy is pinned first: the pass's greedy driver would otherwise erase dead
    ops, which the AstDumper still numbers. Must be called inside the module's
    Context. Returns (graph, {tag: original op}).
    """
    from mlir.ir import AffineMapAttr, Module, OpResult, StringAttr
    from mlir.passmanager import PassManager

    originals = walk_ops(module, is_structured)
    generalized = Module.parse(str(module))
    pin_values(walk_ops(generalized, is_structured))
    PassManager.parse('builtin.module(linalg-generalize-named-ops)').run(generalized.operation)
    generics = walk_ops(generalized, is_structured)
    if len(generics) != len(originals) or any(op.name != 'linalg.generic' for op in generics):
        raise BackendError('some structured ops did not generalize')

    ops: dict[str, OpView] = {}
    by_tag: dict = {}
    edges: list[Edge] = []
    for original, generic in zip(originals, generics):
        iterators = tuple(re.findall(r'parallel|reduction', str(generic.attributes['iterator_types'])))
        if not iterators:
            continue  # the AstDumper skips zero-loop ops
        tag = f'operation_{len(ops)}'
        original.attributes[TAG_ATTR] = StringAttr.get(tag)
        n_in, n_init = list(generic.attributes['operandSegmentSizes'])
        maps = [tuple(normalize_expr(str(e)) for e in AffineMapAttr(a).value.results)
                for a in generic.attributes['indexing_maps']]
        ops[tag] = OpView(original.name, n_in, n_init, iterators, tuple(maps[:n_in]), tuple(maps[n_in:]))
        by_tag[tag] = original
        for k, value in enumerate(original.operands):
            if OpResult.isinstance(value) and TAG_ATTR in value.owner.attributes:
                producer = StringAttr(value.owner.attributes[TAG_ATTR]).value
                edges.append(Edge(producer, tag, k, OpResult(value).result_number))
    return ProgramGraph(ops, edges, 'bindings'), by_tag


def parse_module(text: str):
    """Parse `text` in the current Context (unregistered dialects allowed); BackendError on failure."""
    from mlir.ir import Context, MLIRError, Module

    Context.current.allow_unregistered_dialects = True
    try:
        return Module.parse(text)
    except MLIRError as e:
        raise BackendError(f'bindings cannot parse the program: {e}') from e


def graph_from_bindings(text: str) -> ProgramGraph:
    """Build the graph with the MLIR Python bindings."""
    from mlir.ir import Context

    with Context():
        graph, _ = tag_module(parse_module(text))
    return graph


def _load_gap1_text_parser():
    spec = importlib.util.spec_from_file_location('gap1_mlir_text', _GAP1_TEXT_PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_from_text(text: str) -> ProgramGraph:
    """Build the graph with the gap1 text parser. Named ops get unknown maps (-> 'unparsed')."""
    parsed, _ = _load_gap1_text_parser().parse_text(text)
    ops: dict[str, OpView] = {}
    edges: list[Edge] = []
    for o in parsed:
        n_in = len(o.ins)
        known = o.maps is not None and o.iters is not None
        maps = [tuple(normalize_expr(e) for e in results) for _, results in o.maps] if known else None
        ops[o.tag] = OpView(
            name=f'linalg.{o.name}', num_inputs=n_in, num_inits=len(o.outs),
            iterators=tuple(o.iters) if known else None,
            input_maps=tuple(maps[:n_in]) if known else None,
            init_maps=tuple(maps[n_in:]) if known else None,
        )
        operands = o.ins + o.outs
        for producer, k in o.producers:
            result = operands[k].split('#')[1] if '#' in operands[k] else '0'
            edges.append(Edge(producer, o.tag, k, int(result)))
    return ProgramGraph(ops, edges, 'text')


def feature_view(features: 'OperationFeatures') -> OpView:
    """OpView of one v6 OperationFeatures. Operand indices are the aligned
    `load_data + store_data` positions used by `producers` (state.py)."""
    def maps(accesses: list[list[str]]) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(normalize_expr(e) for e in access) for access in accesses)

    return OpView(
        name=features.operation_name,
        num_inputs=len(features.load_data),
        num_inits=len(features.store_data),
        iterators=tuple(loop.iterator_type.value for loop in features.nested_loops),
        input_maps=maps(features.load_data),
        init_maps=maps(features.store_data),
    )


def graph_from_features(bench: 'BenchmarkFeatures') -> ProgramGraph:
    """Build the graph from loaded benchmark features (producers lists as loaded)."""
    ops = {tag: feature_view(bench.operations[tag]) for tag in bench.operation_tags}
    edges = [Edge(producer, tag, k, -1)
             for tag in bench.operation_tags
             for producer, k in bench.operations[tag].producers
             if producer in ops]
    return ProgramGraph(ops, edges, 'features')
