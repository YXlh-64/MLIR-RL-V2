"""M2 — offline green pre-fusion (design rules R2, R7, R12, R14).

Runs MLIR's elementwise fusion (`linalg-fuse-elementwise-ops`) restricted to
green edges. The pass has no per-edge control from Python, so every value that
must not be fused is *pinned* with an extra use by an unregistered
`"fusion.keep"` op (the pass only fuses single-use producers, and never erases
an op that still has a use). The pins are erased after the pass. Pinned values:
- results of every Linalg op that is not an expected green fusion. This covers
  producers with a non-green out-edge (R7: fuse only if all consumers are
  fusable), consumers, and dead ops (which the pass would otherwise erase);
- results of `tensor.expand_shape` / `tensor.collapse_shape` (blocks the pass's
  reshape propagation, R6);
- tensor `arith.constant`s (blocks its constant folding).

R7 is conservative here: the pass fuses single-use producers only, so a green
producer with >= 2 green consumers stays unfused (counted as `multi_use`).

R12: the result is only a different program for the same benchmark. Speedups
stay measured against the original program's time (`root_exec_time` is kept).

CLI (never in place; `--check` compares outputs on random inputs, R14):
    python -m rl_autoschedular_v6.fusion.prefuse in.mlir out.mlir [--report r.json] [--check]
"""

from dataclasses import asdict, dataclass, field
import argparse
import json
import os

from rl_autoschedular_v6.fusion.classify import Classification, classify_graph
from rl_autoschedular_v6.fusion.rules import GREEN
from rl_autoschedular_v6.fusion.views import KEEP_OP, TAG_ATTR, parse_module, pin_values, tag_module, walk_ops

FUSION_PIPELINE = 'builtin.module(func.func(linalg-fuse-elementwise-ops))'
_RESHAPES = ('tensor.expand_shape', 'tensor.collapse_shape')


@dataclass
class PrefuseReport:
    ops_before: int
    ops_after: int
    green_edges: int
    fused: int                      # producers expected to be fused away
    green_not_fused: dict[str, int] = field(default_factory=dict)  # reason -> count
    unexpected_removed: int = 0     # ops_before - fused - ops_after (should be 0)


def _effective_use_count(op, tag: str, classification: Classification) -> int:
    """Uses of `op`, counting repeated reads by one consumer with the same map once
    (the pass's canonicalization deduplicates such operands before fusing)."""
    graph = classification.graph
    reads = [(e.consumer, graph.ops[e.consumer].input_maps[e.operand]
              if e.operand < graph.ops[e.consumer].num_inputs else ('init', e.operand))
             for e, _ in classification.edge_classes if e.producer == tag]
    raw = sum(len(list(result.uses)) for result in op.results)
    return raw - (len(reads) - len(set(reads)))


def plan_fusion(classification: Classification, by_tag: dict) -> tuple[set[str], dict[str, int]]:
    """(tags expected to fuse, green-but-not-fused counts by reason)."""
    graph = classification.graph
    non_green = {e.producer for e, cls in classification.edge_classes if cls != GREEN}

    fusable, not_fused = set(), {}
    for edge, cls in classification.edge_classes:
        if cls != GREEN:
            continue
        producer, consumer = graph.ops[edge.producer], graph.ops[edge.consumer]
        if edge.producer in non_green:
            reason = 'pinned_producer'      # R7: another consumer is not fusable
        elif _effective_use_count(by_tag[edge.producer], edge.producer, classification) > 1:
            reason = 'multi_use'            # R7 conservative: MLIR fuses single-use only
        elif producer.name != 'linalg.generic' or consumer.name != 'linalg.generic':
            reason = 'named_op'             # the pass fuses linalg.generic only
        else:
            fusable.add(edge.producer)
            continue
        not_fused[reason] = not_fused.get(reason, 0) + 1
    return fusable, not_fused


def _strip_tags(module) -> None:
    for op in walk_ops(module, lambda op: TAG_ATTR in op.attributes):
        del op.attributes[TAG_ATTR]


def _ops_to_pin(module, kept_linalg_ops: list) -> list:
    extra = walk_ops(module, lambda op: op.name in _RESHAPES
                     or (op.name == 'arith.constant' and 'tensor<' in str(op.results[0].type)))
    return kept_linalg_ops + extra


def prefuse_code(code: str) -> tuple[str, PrefuseReport]:
    """Fuse green edges of `code` (R2, R7). Returns (fused code, report)."""
    from mlir.ir import Context
    from mlir.passmanager import PassManager

    with Context():
        module = parse_module(code)
        graph, by_tag = tag_module(module)
        classification = classify_graph(graph)
        fusable, not_fused = plan_fusion(classification, by_tag)

        pin_values(_ops_to_pin(module, [op for tag, op in by_tag.items() if tag not in fusable]))
        PassManager.parse(FUSION_PIPELINE).run(module.operation)
        for keep in walk_ops(module, lambda op: op.name == KEEP_OP):
            keep.erase()
        _strip_tags(module)
        ops_after = len(tag_module(module)[0].ops)
        _strip_tags(module)
        fused_code = str(module)

    report = PrefuseReport(
        ops_before=len(graph.ops),
        ops_after=ops_after,
        green_edges=sum(cls == GREEN for _, cls in classification.edge_classes),
        fused=len(fusable),
        green_not_fused=not_fused,
        unexpected_removed=len(graph.ops) - len(fusable) - ops_after,
    )
    return fused_code, report


def prefuse_file(src: str, dst: str) -> PrefuseReport:
    """Read `src`, write the fused program to a different path `dst`."""
    if os.path.realpath(src) == os.path.realpath(dst):
        raise ValueError('prefuse never writes in place: choose a different output path')
    if os.path.exists(dst):
        raise FileExistsError(f'{dst} exists; refusing to overwrite')
    with open(src) as f:
        fused, report = prefuse_code(f.read())
    with open(dst, 'w') as f:
        f.write(fused)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Green-only elementwise pre-fusion (R2, R7).')
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--report', help='write the report as JSON here')
    parser.add_argument('--check', action='store_true',
                        help='compare outputs of input and output on random inputs (compiles and runs both)')
    args = parser.parse_args()

    report = prefuse_file(args.input, args.output)
    result = asdict(report)
    if args.check:
        from rl_autoschedular_v6.fusion.check import check_equivalent
        with open(args.input) as f_in, open(args.output) as f_out:
            result['allclose'], result['max_abs_err'] = check_equivalent(f_in.read(), f_out.read())
    print(json.dumps(result))
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
