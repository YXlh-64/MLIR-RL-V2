"""M1 — edge classifier (design rules R1–R6).

Gives every Linalg op a type (One-to-One / Broadcast / Shuffle / Reduction /
Window / Init) and every producer->consumer edge a class
`green | red | init | yellow:<epilogue|partial|halo|shuffle|unparsed>`.
All decisions come from `rules.py`. This module only wires a graph builder
(`views.py`) to the rules.

CLI:
    python -m rl_autoschedular_v6.fusion.classify model.mlir [more.mlir ...] --out classes.json
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import json
import os

from rl_autoschedular_v6.fusion.rules import OpType, access, classify_edge, op_type, with_init_override
from rl_autoschedular_v6.fusion.views import BackendError, Edge, ProgramGraph, graph_from_bindings, graph_from_text


@dataclass
class Classification:
    graph: ProgramGraph
    op_types: dict[str, OpType]       # final type (with INIT override)
    base_types: dict[str, OpType]     # type used for edge classes
    edge_classes: list[tuple[Edge, str]]

    def counts(self) -> dict[str, int]:
        return dict(Counter(cls for _, cls in self.edge_classes))

    def to_json(self) -> dict:
        return {
            'backend': self.graph.backend,
            'ops': {tag: {'name': view.name, 'type': self.op_types[tag].value}
                    for tag, view in self.graph.ops.items()},
            'edges': [{'producer': e.producer, 'consumer': e.consumer, 'operand': e.operand,
                       'result': e.result, 'cls': cls} for e, cls in self.edge_classes],
            'counts': self.counts(),
        }


def build_graph(text: str, backend: str = 'auto') -> ProgramGraph:
    """backend: 'bindings', 'text', or 'auto' (bindings, falling back to text)."""
    if backend in ('auto', 'bindings'):
        try:
            return graph_from_bindings(text)
        except (ImportError, BackendError):
            if backend == 'bindings':
                raise
    return graph_from_text(text)


def classify_graph(graph: ProgramGraph) -> Classification:
    base = {tag: op_type(view) for tag, view in graph.ops.items()}
    consumer_accesses: dict[str, list] = defaultdict(list)
    edge_classes = []
    for edge in graph.edges:
        acc = access(graph.ops[edge.consumer], edge.operand)
        consumer_accesses[edge.producer].append(acc)
        edge_classes.append((edge, classify_edge(base[edge.producer], acc)))
    final = {tag: with_init_override(base[tag], consumer_accesses[tag]) for tag in graph.ops}
    return Classification(graph, final, base, edge_classes)


def classify_text(text: str, backend: str = 'auto') -> Classification:
    return classify_graph(build_graph(text, backend))


def main() -> None:
    from rl_autoschedular_v6.fusion.mlir_io import read_mlir

    parser = argparse.ArgumentParser(description='Classify producer->consumer edges (R1-R6).')
    parser.add_argument('inputs', nargs='+', help='linalg .mlir files')
    parser.add_argument('--out', required=True, help='output JSON file')
    parser.add_argument('--backend', choices=('auto', 'bindings', 'text'), default='auto')
    args = parser.parse_args()

    result = {os.path.basename(path): classify_text(read_mlir(path), args.backend).to_json()
              for path in args.inputs}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    for name, res in result.items():
        print(f'{name}: {len(res["ops"])} ops, {res["counts"]} ({res["backend"]})')


if __name__ == '__main__':
    main()
