"""M3 — representative closed units from full models (design doc §4, option U1).

1. Sequence: top-level ops of the model's public function, minus timing ops
   (`@nanoTime` calls and what uses them) and "free" ops (scalars,
   `tensor.empty`, zero-init fills: anything computing a constant without
   weights), which every unit that uses them clones in. torch-mlir shares one
   init fill across many convs, which would otherwise look live across whole stages.
2. Cut width at a boundary = number of values defined before it and used after
   it. Function args and weights (tensor constants) are not counted. "Global" values are not
   counted either (e.g. attention masks): those with at least `global_min_users`
   users and a live range over `global_span_fraction` of the sequence.
   A boundary with width <= 1 is a natural cut (end of a residual block or
   layer half).
3. Segments between cuts are split above `max_ops` at their narrowest internal
   boundary, then consecutive segments below `min_ops` are merged (staying
   <= `max_ops`). Size = Linalg ops left after green fusion (estimated with
   the M2 plan on the whole model).
4. Structural signature per segment (op names, attributes, types, relative
   internal edges, external inputs numbered by first use). One unit per
   distinct signature, with weight = number of occurrences (§4.2 principle 3).
5. Units are written with `unit_extract.extract_unit`, plus `units_manifest.json`.

Leakage guard: a model listed in the evaluation set (`eval_set`, the single
source of truth `config/fusion/eval_set.json`) is refused (or only warned about
with `--allow-eval-models`), so evaluation models never become training units.

CLI:
    python -m rl_autoschedular_v6.fusion.units model.mlir [...] --out DIR [--config cfg.json] [--allow-eval-models]
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any
import argparse
import hashlib
import json
import os

from rl_autoschedular_v6.fusion.classify import classify_graph, classify_text
from rl_autoschedular_v6.fusion.eval_set import excluded_names
from rl_autoschedular_v6.fusion.prefuse import plan_fusion, prefuse_code
from rl_autoschedular_v6.fusion.unit_extract import (
    POS_ATTR, UnitSkip, extract_unit, free_positions, is_tensor, op_pos, public_function, stamp_positions, value_pos,
)
from rl_autoschedular_v6.fusion.views import parse_module, tag_module, walk_ops


@dataclass(frozen=True)
class UnitParams:
    min_ops: int = 5
    max_ops: int = 25
    global_span_fraction: float = 0.25
    global_min_users: int = 3
    eval_set: str = ''   # path of the eval-set JSON (fusion/eval_set.py); '' = no guard

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> 'UnitParams':
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f'unknown fusion.units keys: {sorted(unknown)} (known: {sorted(known)})')
        return cls(**section)


@dataclass
class Segment:
    start: int               # sequence index, inclusive
    end: int                 # sequence index, exclusive
    est_ops: int             # Linalg ops left after green fusion
    signature: str = ''


@dataclass
class ModelAnalysis:
    body_positions: list[int]     # sequence index -> position in the function body
    widths: list[int]             # widths[k]: values live across the boundary before sequence op k
    segments: list[Segment] = field(default_factory=list)

    def positions(self, segment: Segment) -> list[int]:
        return self.body_positions[segment.start:segment.end]


class EvalModelLeakage(RuntimeError):
    """A model reserved for evaluation was given to the training-unit builder."""


def model_name(path: str) -> str:
    base = os.path.basename(path)
    return base[:-len('.mlir')].removesuffix('_linalg')


# ---------------------------------------------------------------- analysis

def _sequence(body: list) -> list[int]:
    """Body positions of the ops that are segmented: no free ops, no weights (tensor
    constants, which units take as arguments), no timing ops, no terminator."""
    free = free_positions(body)
    timing, keep = set(), []
    for p, op in enumerate(body[:-1]):
        name = op.operation.name
        is_timer = name == 'func.call' and str(op.attributes['callee']) == '@nanoTime'
        if is_timer or any(value_pos(v) in timing for v in op.operands):
            timing.add(p)
        elif p not in free and not (name == 'arith.constant' and is_tensor(op.results[0])):
            keep.append(p)
    return keep


def _widths(body: list, seq: list[int], params: UnitParams) -> list[int]:
    """Live values across each sequence boundary (index k = before sequence op k)."""
    seq_index = {p: i for i, p in enumerate(seq)}
    n = len(seq)
    diff = [0] * (n + 1)
    for i, p in enumerate(seq):
        for result in body[p].results:
            users = {op_pos(use.owner) for use in result.uses}
            last = max((seq_index.get(u, n) for u in users if u is not None), default=None)
            if last is None or last <= i:
                continue
            is_global = len(users) >= params.global_min_users and (last - i) > params.global_span_fraction * n
            if not is_global:
                diff[i + 1] += 1
                diff[last + 1 if last < n else n] -= 1
    widths, running = [], 0
    for k in range(n):
        running += diff[k]
        widths.append(running)
    return widths


def _est_ops(seq: list[int], by_tag: dict, fusable: set[str]) -> list[int]:
    """Linalg ops per sequence op that survive green fusion."""
    seq_index = {p: i for i, p in enumerate(seq)}
    counts = [0] * len(seq)
    for tag, op in by_tag.items():
        i = seq_index.get(op_pos(op))
        if i is not None and tag not in fusable:
            counts[i] += 1
    return counts


def _cut_segments(widths: list[int], est: list[int]) -> list[Segment]:
    bounds = [0] + [k for k in range(1, len(widths)) if widths[k] <= 1] + [len(widths)]
    return [Segment(s, e, sum(est[s:e])) for s, e in zip(bounds, bounds[1:]) if e > s]


def _split(seg: Segment, widths: list[int], est: list[int], max_ops: int) -> list[Segment]:
    """Split at the narrowest internal boundary (ties: nearest the middle) until <= max_ops."""
    inner = [k for k in range(seg.start + 1, seg.end)
             if sum(est[seg.start:k]) > 0 and sum(est[k:seg.end]) > 0]
    if seg.est_ops <= max_ops or not inner:
        return [seg]
    middle = (seg.start + seg.end) / 2
    k = min(inner, key=lambda b: (widths[b], abs(b - middle)))
    left = Segment(seg.start, k, sum(est[seg.start:k]))
    right = Segment(k, seg.end, sum(est[k:seg.end]))
    return _split(left, widths, est, max_ops) + _split(right, widths, est, max_ops)


def _merge(segments: list[Segment], min_ops: int, max_ops: int) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        last = merged[-1] if merged else None
        if last is not None and last.est_ops < min_ops and last.est_ops + seg.est_ops <= max_ops:
            merged[-1] = Segment(last.start, seg.end, last.est_ops + seg.est_ops)
        else:
            merged.append(seg)
    return merged


def _signature(body: list, positions: list[int], entry_block) -> str:
    """Hash of the segment's structure, independent of SSA names, tags and position."""
    from mlir.ir import BlockArgument, OpResult

    inside = {p: i for i, p in enumerate(positions)}
    externals: list = []   # external values, numbered by first use
    tokens = []
    for p in positions:
        for op in walk_ops(body[p], lambda _: True):
            refs = []
            for v in op.operands:
                if OpResult.isinstance(v) and op_pos(v.owner) in inside and POS_ATTR in v.owner.attributes:
                    refs.append(f'%{inside[op_pos(v.owner)]}.{OpResult(v).result_number}')
                elif OpResult.isinstance(v) and op_pos(v.owner) in inside:
                    refs.append(f'local:{v.type}')
                elif not OpResult.isinstance(v) and BlockArgument(v).owner != entry_block:
                    refs.append(f'local:{v.type}')
                else:
                    if v not in externals:
                        externals.append(v)
                    refs.append(f'ext{externals.index(v)}:{v.type}')
            attrs = sorted(f'{a.name}={a.attr}' for a in op.attributes
                           if a.name != 'tag' and not a.name.startswith('fusion.'))
            tokens.append(f'{op.operation.name}({",".join(refs)}){attrs}->{[str(r.type) for r in op.results]}')
    return hashlib.sha1('\n'.join(tokens).encode()).hexdigest()[:12]


def analyze(code: str, params: UnitParams) -> ModelAnalysis:
    """Segment a model and sign each segment."""
    from mlir.ir import Context

    with Context():
        module = parse_module(code)
        graph, by_tag = tag_module(module)
        fusable, _ = plan_fusion(classify_graph(graph), by_tag)
        main = public_function(module)
        body = list(main.regions[0].blocks[0].operations)
        stamp_positions(body)

        seq = _sequence(body)
        widths = _widths(body, seq, params)
        est = _est_ops(seq, by_tag, fusable)
        segments = [s for seg in _cut_segments(widths, est) for s in _split(seg, widths, est, params.max_ops)]
        analysis = ModelAnalysis(seq, widths, _merge(segments, params.min_ops, params.max_ops))
        for seg in analysis.segments:
            seg.signature = _signature(body, analysis.positions(seg), main.regions[0].blocks[0])
    return analysis


# ---------------------------------------------------------------- units + manifest

def _unit_stats(unit_code: str) -> dict:
    classification = classify_text(unit_code, backend='bindings')
    producers: dict[str, set] = {}
    for edge, _ in classification.edge_classes:
        producers.setdefault(edge.consumer, set()).add(edge.producer)
    return {
        'ops_raw': len(classification.graph.ops),
        'ops_after_green_fusion': prefuse_code(unit_code)[1].ops_after,
        'multi_producer_ops': sum(len(p) >= 2 for p in producers.values()),
        'edge_classes': classification.counts(),
        'op_names': sorted({v.name for v in classification.graph.ops.values()}),
    }


def build_units(path: str, out_dir: str, params: UnitParams, allow_eval_models: bool = False) -> dict:
    """Write one unit per distinct signature of `path`; return its manifest entry."""
    from rl_autoschedular_v6.fusion.mlir_io import read_mlir

    name = model_name(path)
    if params.eval_set and name in excluded_names(params.eval_set):
        message = f'{name} is an evaluation model ({params.eval_set}); refusing to build training units'
        if not allow_eval_models:
            raise EvalModelLeakage(message)
        print(f'WARNING: {message} -- continuing because --allow-eval-models was given')

    code = read_mlir(path)
    analysis = analyze(code, params)
    groups: dict[str, list[Segment]] = {}
    for seg in analysis.segments:
        groups.setdefault(seg.signature, []).append(seg)

    units, skipped, covered = [], [], set()
    for k, (signature, segs) in enumerate(groups.items()):
        try:
            unit_code = extract_unit(code, analysis.positions(segs[0]))
        except UnitSkip as e:
            skipped.append({'signature': signature, 'weight': len(segs), 'reason': str(e)})
            continue
        file_name = f'{name}_unit_{k}.mlir'
        file_path = os.path.join(out_dir, file_name)
        if os.path.exists(file_path):
            raise FileExistsError(f'{file_path} exists; refusing to overwrite')
        with open(file_path, 'w') as f:
            f.write(unit_code)
        stats = _unit_stats(unit_code)
        covered.update(stats['op_names'])
        units.append({'file': file_name, 'signature': signature, 'weight': len(segs),
                      'occurrences': [[s.start, s.end] for s in segs], **stats})

    model_ops = {v.name for v in classify_text(code, backend='bindings').graph.ops.values()}
    return {'source_model': os.path.abspath(path), 'segments': len(analysis.segments), 'units': units,
            'skipped': skipped, 'missing_op_names': sorted(model_ops - covered)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build closed representative units from full models (§4).')
    parser.add_argument('models', nargs='+', help='full-model .mlir files')
    parser.add_argument('--out', required=True, help='output directory (unit files + units_manifest.json)')
    parser.add_argument('--config', help='config JSON whose fusion.units section sets the parameters')
    parser.add_argument('--allow-eval-models', action='store_true', help='warn instead of failing on eval models')
    args = parser.parse_args()

    section = {}
    if args.config:
        with open(args.config) as f:
            section = json.load(f).get('fusion', {}).get('units', {})
    params = UnitParams.from_dict(section)

    manifest_path = os.path.join(args.out, 'units_manifest.json')
    if os.path.exists(manifest_path):
        raise FileExistsError(f'{manifest_path} exists; choose a new --out directory')
    os.makedirs(args.out, exist_ok=True)
    manifest = {'params': asdict(params), 'models': {}}
    for path in args.models:
        entry = build_units(path, args.out, params, args.allow_eval_models)
        manifest['models'][model_name(path)] = entry
        print(f'{model_name(path)}: {entry["segments"]} segments -> {len(entry["units"])} units, '
              f'{len(entry["skipped"])} skipped, missing op names: {entry["missing_op_names"]}')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)


if __name__ == '__main__':
    main()
