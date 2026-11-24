#!/usr/bin/env python3
"""Catalog the multi-operation blocks of a dataset (step 1 of experiment 02).

Reads `.mlir` text only. It compiles nothing, executes nothing and needs no MLIR
bindings or torch, so it can run on a login node in a plain Python 3.10+.

For every `<model>_block_<N>.mlir` it records:
  * the operations in producer->consumer order, with their model tag
    (`tag = "operation_13"`), kind, operand shapes, loop structure and body ops;
  * the dataflow edges between those operations inside the block;
  * a class (conv / matmul / reduction / elementwise) and a size bucket, used by
    run_blocks.py for stratified sampling;
  * static cost estimates (FLOPs, bytes crossing the function ABI, intensity);
  * static pre-checks: can the translator (block_to_torch.py) handle it, and
    does it fit the 12-loop limit of v5_model_1200?

Across blocks of the same model it records how blocks relate, which the
dependency experiments need:
  * overlaps     blocks sharing model ops (the extractor uses sliding windows)
  * contains     strict subset relations (nested windows)
  * feeds        disjoint blocks where an op of A produces an input of B
  * compositions X whose ops are exactly Y + Z with Y feeding Z (experiment D3)

Outputs (default: experiments/02_block_diagnosis/results/catalog/):
  blocks_catalog.json   one record per block
  blocks_index.json     relations between blocks, per model
  catalog_summary.csv   counts per model x class x size bucket
  coverage.json         totals and flag counts for the pre-checks

Usage:
  python experiments/02_block_diagnosis/block_catalog.py
  python experiments/02_block_diagnosis/block_catalog.py --models albert resnet50 --limit 50
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DATA = REPO / 'data' / 'ops_and_blocks'
DEFAULT_OUTPUT = HERE / 'results' / 'catalog'

# Architecture limit of v5_model_1200 (config/v5/v5_mlir_rl_v1_paper.json: max_num_loops).
MAX_NUM_LOOPS = 12

DTYPE_BYTES = {'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8,
               'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8, 'index': 8}

# What block_to_torch.py is planned to translate. Anything else is flagged, never guessed.
SUPPORTED_NAMED_OPS = {'generic', 'matmul', 'batch_matmul', 'conv_2d_nchw_fchw',
                       'depthwise_conv_2d_nchw_chw', 'pooling_nchw_max', 'pooling_nchw_sum',
                       'fill', 'transpose', 'broadcast'}
SUPPORTED_BODY_OPS = {
    'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.negf',
    'arith.maximumf', 'arith.minimumf', 'arith.maxnumf', 'arith.minnumf',
    'arith.cmpf', 'arith.select', 'arith.constant',
    'arith.truncf', 'arith.extf', 'arith.sitofp', 'arith.uitofp', 'arith.fptosi',
    'arith.addi', 'arith.subi', 'arith.muli', 'arith.index_cast',
    'math.exp', 'math.rsqrt', 'math.sqrt', 'math.tanh', 'math.erf',
    'math.powf', 'math.fpowi', 'math.log', 'math.absf',
    'linalg.index',
}
# Body ops that do not count as floating-point work in the FLOP estimate.
NON_FLOP_BODY_OPS = {'arith.constant', 'arith.index_cast', 'linalg.index',
                     'arith.truncf', 'arith.extf', 'arith.sitofp', 'arith.uitofp',
                     'arith.fptosi', 'arith.select'}

TRANSLATOR_BLOCKERS = ('parse_error', 'dynamic_shape', 'complex_affine_map',
                       'missing_affine_map', 'unknown_loop_size', 'unsupported_body_op',
                       'unsupported_named_op', 'operand_type_mismatch')
RL_BLOCKERS = ('parse_error', 'dynamic_shape', 'loops_exceed_max')

BLOCK_NAME_RE = re.compile(r'^(?P<model>.+)_block_(?P<index>\d+)$')
MAP_DEF_RE = re.compile(r'^\s*(#[\w$.-]+)\s*=\s*affine_map<(.*)>\s*$')
MAIN_RE = re.compile(r'func\.func\s+@main\s*\(')
OP_START_RE = re.compile(r'^\s*(?P<res>%[\w.$-]+)\s*=\s*linalg\.(?P<kind>\w+)\b')
CONST_RE = re.compile(r'^\s*(%[\w.$-]+)\s*=\s*arith\.constant\s+(.*?)\s*:\s*(\S.*?)\s*$')
RETURN_RE = re.compile(r'^\s*return\s+(.*?)\s*:')
TENSOR_RE = re.compile(r'tensor<((?:[0-9?]+x)*)([a-z]+[0-9]*)>')
SSA_RE = re.compile(r'%[\w.$-]+')


# --------------------------------------------------------------------------- text helpers

def split_top(text, sep=','):
    """Split on `sep` outside (), [], <>, {}; the '>' of '->' is not a bracket."""
    parts, current, depth = [], [], 0
    for i, ch in enumerate(text):
        if ch in '([{<':
            depth += 1
        elif ch in ')]}' or (ch == '>' and (i == 0 or text[i - 1] != '-')):
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = ''.join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def uses(name, line):
    """True if SSA value `name` appears in `line` as a whole token."""
    return re.search(re.escape(name) + r'(?![\w.$-])', line) is not None


def parse_type(text):
    text = text.strip()
    m = TENSOR_RE.fullmatch(text)
    if m:
        dims = [d for d in m[1].split('x') if d]
        if '?' in dims:
            return dict(kind='tensor', shape=None, dtype=m[2], dynamic=True)
        return dict(kind='tensor', shape=[int(d) for d in dims], dtype=m[2], dynamic=False)
    if text in DTYPE_BYTES:
        return dict(kind='scalar', shape=[], dtype=text, dynamic=False)
    return dict(kind='other', shape=None, dtype=None, dynamic=False, text=text)


def nbytes(t):
    if not t or t.get('shape') is None:
        return None
    return math.prod(t['shape']) * DTYPE_BYTES.get(t['dtype'], 4)


def parse_affine_map(text):
    """`(d0, d1)[s0] -> (d0, 0)` -> {'dims': [...], 'results': [('dim', 0), ('const', 0)]}."""
    m = re.fullmatch(r'\s*\(([^)]*)\)\s*(\[[^\]]*\])?\s*->\s*\((.*)\)\s*', text)
    if not m:
        return dict(text=text, dims=[], results=[], complex=True)
    dims = [d.strip() for d in m[1].split(',') if d.strip()]
    results, complex_ = [], bool(m[2] and m[2].strip('[] '))
    for expr in split_top(m[3]):
        if expr in dims:
            results.append(('dim', dims.index(expr)))
        elif re.fullmatch(r'-?\d+', expr):
            results.append(('const', int(expr)))
        else:  # d0 + d1, d0 * 2, floordiv, mod, symbols: not a plain projection
            results.append(('expr', expr))
            complex_ = True
    return dict(text=text.strip(), dims=dims, results=results, complex=complex_)


def operand_group(header, keyword):
    """Parse `ins(%a, %b : tensor<..>, tensor<..>)` into [{'name', 'type'}]."""
    m = re.search(rf'\b{keyword}\(([^)]*)\)', header)
    if not m:
        return [], True
    inner = m[1]
    if ':' not in inner:
        return [dict(name=n, type=None) for n in SSA_RE.findall(inner)], False
    names_part, types_part = inner.split(':', 1)
    names = SSA_RE.findall(names_part)
    types = [parse_type(t) for t in split_top(types_part)]
    return [dict(name=n, type=t) for n, t in zip(names, types)], len(names) == len(types)


def op_family(kind):
    if kind.startswith(('conv', 'depthwise_conv')):
        return 'conv'
    if kind.startswith('pooling'):
        return 'pool'
    if 'matmul' in kind or 'matvec' in kind:
        return 'matmul'
    if kind == 'generic':
        return 'generic'
    if kind in ('fill', 'transpose', 'broadcast', 'copy'):
        return 'data_movement'
    return 'other'


def int_list(pattern, text, rank=2):
    m = re.search(pattern, text)
    if not m:
        return None
    values = [int(v) for v in re.findall(r'-?\d+', m[1])]
    return values * rank if len(values) == 1 else values


# --------------------------------------------------------------------------- per-op parsing

def named_op_cost(kind, ins, outs):
    """(n_loops, iteration_points, flops) for named ops; None where unknown."""
    shape = lambda ops, i: (ops[i]['type'] or {}).get('shape') if len(ops) > i else None
    a, b, o = shape(ins, 0), shape(ins, 1), shape(outs, 0)
    try:
        if kind == 'matmul':                       # A[M,K] B[K,N] -> C[M,N]
            (m, k), n = a, b[1]
            return 3, m * n * k, 2 * m * n * k
        if kind == 'batch_matmul':                 # A[B,M,K] B[B,K,N] -> C[B,M,N]
            bt, m, k = a
            n = b[2]
            return 4, bt * m * n * k, 2 * bt * m * n * k
        if kind == 'conv_2d_nchw_fchw':            # in[N,C,H,W] w[F,C,KH,KW] -> [N,F,OH,OW]
            _, c, kh, kw = b
            points = math.prod(o) * c * kh * kw
            return 7, points, 2 * points
        if kind == 'depthwise_conv_2d_nchw_chw':   # in[N,C,H,W] w[C,KH,KW] -> [N,C,OH,OW]
            _, kh, kw = b
            points = math.prod(o) * kh * kw
            return 6, points, 2 * points
        if kind in ('pooling_nchw_max', 'pooling_nchw_sum'):   # in[N,C,H,W] window[KH,KW] -> [N,C,OH,OW]
            points = math.prod(o) * math.prod(b)
            return 6, points, points
        if kind in ('fill', 'transpose', 'broadcast', 'copy'):
            return len(o), math.prod(o), 0
    except (TypeError, ValueError, IndexError):
        pass
    return (len(o) if o else None), None, None


def parse_op(match, op_lines, maps, position):
    kind = match['kind']
    header = op_lines[0]
    flags = []
    tag = re.search(r'\btag\s*=\s*"([^"]+)"', header)
    ins, ok_in = operand_group(header, 'ins')
    outs, ok_out = operand_group(header, 'outs')
    if not (ok_in and ok_out):
        flags.append('operand_type_mismatch')
    result_types = re.findall(r'->\s*\(?\s*(tensor<[^>]*>)', op_lines[-1])
    result_type = parse_type(result_types[-1]) if result_types else None

    for operand in ins + outs:
        if operand['type'] and operand['type'].get('dynamic'):
            flags.append('dynamic_shape')
    if kind not in SUPPORTED_NAMED_OPS:
        flags.append(f'unsupported_named_op:{kind}')

    op = dict(position=position, tag=tag[1] if tag else None, kind=kind,
              family=op_family(kind), result=match['res'],
              ins=[dict(name=o['name'], **_shape_dtype(o['type'])) for o in ins],
              outs=[dict(name=o['name'], **_shape_dtype(o['type'])) for o in outs],
              result_type=_shape_dtype(result_type))
    if not tag:
        flags.append('no_tag')

    if kind == 'generic':
        maps_m = re.search(r'indexing_maps\s*=\s*\[(.*?)\]\s*,\s*iterator_types', header)
        map_texts = []
        for item in split_top(maps_m[1]) if maps_m else []:
            if item.startswith('#'):
                if item not in maps:
                    flags.append('missing_affine_map')
                map_texts.append(maps.get(item, ''))
            elif item.startswith('affine_map<'):
                map_texts.append(item[len('affine_map<'):-1])
        parsed = [parse_affine_map(t) for t in map_texts]
        if any(p['complex'] for p in parsed):
            flags.append('complex_affine_map')
        it_m = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', header)
        iterators = re.findall(r'(parallel|reduction|window)', it_m[1]) if it_m else []

        body = [l for l in op_lines[1:] if not re.match(r'^\s*\}', l)]
        bb = next((l for l in body if l.strip().startswith('^bb')), '')
        block_args = re.findall(r'(%[\w.$-]+)\s*:', bb)
        out_args = block_args[len(ins):]
        body_ops = [o for o in re.findall(r'=\s*((?:arith|math|linalg)\.\w+)', '\n'.join(body))]
        reads_out = any(uses(a, l) for a in out_args for l in body if not l.strip().startswith('^bb'))
        for b in sorted(set(body_ops) - SUPPORTED_BODY_OPS):
            flags.append(f'unsupported_body_op:{b}')

        n_loops = len(iterators)
        sizes = [None] * n_loops
        for operand, p in zip(ins + outs, parsed):
            shp = (operand['type'] or {}).get('shape') or []
            for pos, (k, v) in enumerate(p['results']):
                if k == 'dim' and v < n_loops and pos < len(shp):
                    if sizes[v] is not None and sizes[v] != shp[pos]:
                        flags.append('loop_size_conflict')
                    sizes[v] = shp[pos]
        if any(s is None for s in sizes):
            flags.append('unknown_loop_size')
        points = math.prod(sizes) if sizes and None not in sizes else None
        flop_ops = sum(1 for b in body_ops if b not in NON_FLOP_BODY_OPS)
        op.update(n_loops=n_loops, iterators=iterators, loop_sizes=sizes,
                  maps=[p['text'] for p in parsed], body_ops=body_ops, reads_out=reads_out,
                  iteration_points=points,
                  flops=points * flop_ops if points is not None else None)
    else:
        n_loops, points, flops = named_op_cost(kind, ins, outs)
        op.update(n_loops=n_loops, iteration_points=points, flops=flops,
                  # named linalg ops accumulate into `outs` (C += A*B); fill/transpose/broadcast overwrite
                  reads_out=op['family'] in ('matmul', 'conv', 'pool'))
        strides = int_list(r'strides\s*=\s*dense<([^>]*)>', header)
        dilations = int_list(r'dilations\s*=\s*dense<([^>]*)>', header)
        permutation = int_list(r'permutation\s*=\s*\[([^\]]*)\]', header, rank=1)
        dimensions = int_list(r'dimensions\s*=\s*\[([^\]]*)\]', header, rank=1)
        attrs = {k: v for k, v in dict(strides=strides, dilations=dilations,
                 permutation=permutation, dimensions=dimensions).items() if v is not None}
        if attrs:
            op['attrs'] = attrs

    if op.get('n_loops') and op['n_loops'] > MAX_NUM_LOOPS:
        flags.append('loops_exceed_max')
    op['flags'] = sorted(set(flags))
    return op


def _shape_dtype(t):
    if not t:
        return dict(shape=None, dtype=None)
    return dict(shape=t.get('shape'), dtype=t.get('dtype'))


# --------------------------------------------------------------------------- per-block parsing

def classify(ops):
    families = {o['family'] for o in ops}
    if families & {'conv', 'pool'}:
        return 'conv'
    if 'matmul' in families:
        return 'matmul'
    if any('reduction' in o.get('iterators', []) for o in ops):
        return 'reduction'
    return 'elementwise'


def size_bucket(n_ops):
    if n_ops <= 3:
        return '1-3'
    if n_ops <= 6:
        return '4-6'
    return '7+'


def canonical_signature(op_lines):
    """Hash of the computation with tags removed and SSA names renumbered.

    Two blocks with the same signature compute the same thing on the same shapes,
    even if they come from different models (used to avoid duplicate samples).
    """
    names = {}
    def rename(m):
        return names.setdefault(m[0], f'%{len(names)}')
    text = '\n'.join(op_lines)
    text = re.sub(r'\btag\s*=\s*"[^"]*"\s*,?\s*', '', text)
    text = SSA_RE.sub(rename, text)
    text = re.sub(r'\s+', ' ', text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_main(text):
    """Split a block file into its pieces (shared with block_to_torch.py).

    Returns dict(maps, args, results, ops, consts, returned) where
      maps     {'#map2': '(d0, d1) -> (d0)'}
      args     [{'name': '%arg_0_in_0', 'role': 'in'|'out', 'shape', 'dtype', 'bytes', 'dynamic'}]
      results  [{'shape', 'dtype'}] of @main (the trailing i64 is the nanoTime delta)
      ops      [(regex match of '%v = linalg.<kind>', [lines of that op, region included])]
      consts   [{'name', 'value', 'type'}] for @main-level arith.constant
      returned SSA names in the `return`
    """
    lines = text.splitlines()
    maps = {}
    for line in lines:
        mm = MAP_DEF_RE.match(line)
        if mm:
            maps[mm[1]] = mm[2]
    args, results, ops, consts, returned = [], [], [], [], []
    i, in_main = 0, False
    while i < len(lines):
        line = lines[i]
        if not in_main:
            if MAIN_RE.search(line):
                signature = line
                while 'attributes' not in signature and not signature.rstrip().endswith('{') \
                        and i + 1 < len(lines):
                    i += 1
                    signature += ' ' + lines[i].strip()
                arg_text = signature.split(')', 1)[0]
                for name, ty in re.findall(r'(%[\w.$-]+)\s*:\s*(tensor<[^>]*>|[a-z]+[0-9]*)', arg_text):
                    t = parse_type(ty)
                    args.append(dict(name=name, role='out' if '_out_' in name else 'in',
                                     **_shape_dtype(t), bytes=nbytes(t), dynamic=t.get('dynamic', False)))
                res_m = re.search(r'->\s*\((.*?)\)\s*(?:attributes|\{)', signature)
                if res_m:
                    results = [_shape_dtype(parse_type(t)) for t in split_top(res_m[1])]
                in_main = True
            i += 1
            continue
        om = OP_START_RE.match(line)
        if om:
            depth = line.count('{') - line.count('}')
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += lines[j].count('{') - lines[j].count('}')
            ops.append((om, lines[i:j + 1]))
            i = j + 1
            continue
        cm = CONST_RE.match(line)
        if cm:
            consts.append(dict(name=cm[1], value=cm[2], type=cm[3]))
        rm = RETURN_RE.match(line)
        if rm:
            returned = SSA_RE.findall(rm[1])
        i += 1
    return dict(maps=maps, args=args, results=results, ops=ops, consts=consts, returned=returned)


def parse_block(path):
    m = BLOCK_NAME_RE.match(path.stem)
    record = dict(name=path.stem, model=m['model'] if m else None,
                  block_index=int(m['index']) if m else None, source=str(path))
    main = read_main(path.read_text())
    maps, args, results, returned = main['maps'], main['args'], main['results'], main['returned']
    ops, op_texts = [], []
    for om, op_lines in main['ops']:
        ops.append(parse_op(om, op_lines, maps, len(ops)))
        op_texts.extend(op_lines)
    tensor_consts = sum(1 for c in main['consts'] if 'dense<' in c['value'] or c['type'].startswith('tensor<'))
    scalar_consts = len(main['consts']) - tensor_consts

    if not ops:
        raise ValueError('no linalg operations found in @main')

    # Dataflow edges inside the block (producer -> consumer), by model tag.
    producer_of = {o['result']: o for o in ops}
    edges = []
    for o in ops:
        for role, group in (('in', o['ins']), ('out', o['outs'])):
            for k, operand in enumerate(group):
                p = producer_of.get(operand['name'])
                if p is not None:
                    edges.append(dict(producer=p['tag'], consumer=o['tag'], role=role, operand=k))
    consumers = {e['producer'] for e in edges}
    producers = {e['consumer'] for e in edges}

    flags = sorted({f for o in ops for f in o['flags']}
                   | ({'dynamic_shape'} if any(a['dynamic'] for a in args) else set())
                   | ({'tensor_constant'} if tensor_consts else set()))
    # Bytes crossing the function boundary: all tensor arguments + tensor results
    # (the scalar i64 nanoTime delta is not data).
    abi_parts = [a['bytes'] for a in args] + [nbytes(r) for r in results if r.get('shape')]
    abi_bytes = sum(abi_parts) if None not in abi_parts else None
    flops = [o['flops'] for o in ops]
    total_flops = sum(flops) if None not in flops else None
    tags = [o['tag'] for o in ops]

    record.update(
        n_ops=len(ops),
        n_compute_ops=sum(1 for o in ops if o['family'] != 'data_movement'),
        size_bucket=size_bucket(len(ops)),
        **{'class': classify(ops)},
        tags=tags,
        ops=ops,
        args=args,
        results=results,
        edges=edges,
        sources=[o['tag'] for o in ops if o['tag'] not in producers],
        sinks=[producer_of[r]['tag'] for r in returned if r in producer_of],
        dangling=[o['tag'] for o in ops if o['tag'] not in consumers
                  and o['result'] not in returned],
        scalar_constants=scalar_consts,
        tensor_constants=tensor_consts,
        max_loops=max((o.get('n_loops') or 0) for o in ops),
        total_flops=total_flops,
        abi_bytes=abi_bytes,
        arithmetic_intensity=(total_flops / abi_bytes) if total_flops is not None and abi_bytes else None,
        signature=canonical_signature(op_texts),
        flags=flags,
        translator_precheck='ok' if not any(f.startswith(TRANSLATOR_BLOCKERS) for f in flags) else 'blocked',
        rl_precheck='ok' if not any(f.startswith(RL_BLOCKERS) for f in flags) else 'blocked',
    )
    return record


# --------------------------------------------------------------------------- relations between blocks

def build_index(records):
    """Relations between blocks of the same model, keyed by block name.

    Model-level edges are the union of edges seen inside blocks, so `feeds` is a
    lower bound: two ops never placed in the same block leave no visible edge.
    """
    by_model = collections.defaultdict(list)
    for r in records:
        if r.get('model') and r.get('tags'):
            by_model[r['model']].append(r)

    index, tag_conflicts = {}, []
    for model, blocks in by_model.items():
        tagsets = {b['name']: frozenset(b['tags']) for b in blocks}
        tag_to_blocks = collections.defaultdict(set)
        op_identity = collections.defaultdict(set)
        model_edges = set()
        for b in blocks:
            for o in b['ops']:
                tag_to_blocks[o['tag']].add(b['name'])
                op_identity[o['tag']].add(json.dumps([o['kind'], [x['shape'] for x in o['ins']],
                                                      [x['shape'] for x in o['outs']]]))
            model_edges.update((e['producer'], e['consumer']) for e in b['edges'])
        for tag, ids in op_identity.items():
            if len(ids) > 1:  # same model tag but different op: tags are not a reliable identity
                tag_conflicts.append(dict(model=model, tag=tag, variants=sorted(ids)))

        rel = {name: dict(model=model, tags=sorted(ts), overlaps={}, contains=[],
                          contained_in=[], feeds=[], fed_by=[], compositions=[])
               for name, ts in tagsets.items()}
        for name, ts in tagsets.items():
            others = set().union(*(tag_to_blocks[t] for t in ts)) - {name}
            for other in others:
                shared = len(ts & tagsets[other])
                rel[name]['overlaps'][other] = dict(
                    shared=shared, jaccard=round(shared / len(ts | tagsets[other]), 4))
                if tagsets[other] < ts:
                    rel[name]['contains'].append(other)
                    rel[other]['contained_in'].append(name)

        feeds = set()
        for p, c in model_edges:
            for a in tag_to_blocks[p]:
                for b in tag_to_blocks[c]:
                    if a != b and not (tagsets[a] & tagsets[b]):
                        feeds.add((a, b))
        for a, b in feeds:
            rel[a]['feeds'].append(b)
            rel[b]['fed_by'].append(a)

        # D3 candidates: whole block X == part Y + part Z, Y feeds Z.
        by_tagset = collections.defaultdict(list)
        for name, ts in tagsets.items():
            by_tagset[ts].append(name)
        for x, ts in tagsets.items():
            for y in rel[x]['contains']:
                for z in by_tagset.get(ts - tagsets[y], []):
                    if (y, z) in feeds:
                        rel[x]['compositions'].append(dict(producer_part=y, consumer_part=z))

        for r in rel.values():
            for key in ('contains', 'contained_in', 'feeds', 'fed_by'):
                r[key].sort()
        index.update(rel)
    return index, tag_conflicts


# --------------------------------------------------------------------------- summaries and main

def write_summary(records, path):
    rows = collections.defaultdict(lambda: dict(blocks=0, translator_ok=0, rl_ok=0, signatures=set()))
    for r in records:
        if 'class' not in r:
            continue
        row = rows[(r['model'], r['class'], r['size_bucket'])]
        row['blocks'] += 1
        row['translator_ok'] += r['translator_precheck'] == 'ok'
        row['rl_ok'] += r['rl_precheck'] == 'ok'
        row['signatures'].add(r['signature'])
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'class', 'size_bucket', 'blocks', 'unique_signatures',
                    'translator_precheck_ok', 'rl_precheck_ok'])
        for (model, cls, bucket), row in sorted(rows.items()):
            w.writerow([model, cls, bucket, row['blocks'], len(row['signatures']),
                        row['translator_ok'], row['rl_ok']])


def coverage(records, tag_conflicts):
    parsed = [r for r in records if 'class' in r]
    flag_counts = collections.Counter(f.split(':')[0] if not f.startswith('unsupported') else f
                                      for r in records for f in r.get('flags', []))
    count = lambda key, value: sum(1 for r in parsed if r[key] == value)
    return dict(
        blocks=len(records),
        parsed=len(parsed),
        parse_errors=len(records) - len(parsed),
        translator_precheck_ok=count('translator_precheck', 'ok'),
        rl_precheck_ok=count('rl_precheck', 'ok'),
        both_ok=sum(1 for r in parsed if r['translator_precheck'] == 'ok' and r['rl_precheck'] == 'ok'),
        unique_signatures=len({r['signature'] for r in parsed}),
        by_class=dict(collections.Counter(r['class'] for r in parsed)),
        by_size_bucket=dict(collections.Counter(r['size_bucket'] for r in parsed)),
        by_model=dict(collections.Counter(r['model'] for r in parsed)),
        flag_counts=dict(flag_counts.most_common()),
        tag_conflicts=len(tag_conflicts),
        note='Static pre-checks only. A block passing them can still fail translation, '
             'numerical validation or RL scheduling at run time.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA,
                        help='Folder with <model>_block_<N>.mlir files (default: repo data/ops_and_blocks)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Where to write the catalog')
    parser.add_argument('--models', nargs='*', help='Only these models (e.g. albert resnet50)')
    parser.add_argument('--limit', type=int, help='Only the first N blocks after filtering (quick test)')
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob('*_block_*.mlir'))
    if args.models:
        wanted = set(args.models)
        paths = [p for p in paths
                 if (m := BLOCK_NAME_RE.match(p.stem)) is not None and m['model'] in wanted]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f'No *_block_*.mlir files found in {args.data_dir}')

    records = []
    for k, path in enumerate(paths, 1):
        try:
            records.append(parse_block(path))
        except Exception as error:  # keep going: one bad file must not hide the rest
            m = BLOCK_NAME_RE.match(path.stem)
            records.append(dict(name=path.stem, model=m['model'] if m else None, source=str(path),
                                flags=[f'parse_error:{type(error).__name__}: {error}']))
        if k % 500 == 0:
            print(f'  parsed {k}/{len(paths)}', file=sys.stderr, flush=True)

    index, tag_conflicts = build_index(records)
    summary = coverage(records, tag_conflicts)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'blocks_catalog.json').write_text(json.dumps(
        dict(schema_version=1, data_dir=str(args.data_dir), max_num_loops=MAX_NUM_LOOPS,
             blocks=records), indent=1))
    (args.output / 'blocks_index.json').write_text(json.dumps(
        dict(schema_version=1, blocks=index, tag_conflicts=tag_conflicts), indent=1))
    (args.output / 'coverage.json').write_text(json.dumps(summary, indent=2))
    write_summary(records, args.output / 'catalog_summary.csv')

    print(f"Blocks: {summary['blocks']}  parsed: {summary['parsed']}  parse errors: {summary['parse_errors']}")
    print(f"Translator pre-check ok: {summary['translator_precheck_ok']}   "
          f"RL pre-check ok: {summary['rl_precheck_ok']}   both: {summary['both_ok']}")
    print(f"Unique computations: {summary['unique_signatures']}   tag conflicts: {summary['tag_conflicts']}")
    print(f"By class: {summary['by_class']}")
    print(f"Most common flags: {dict(list(summary['flag_counts'].items())[:8])}")
    print(f'Wrote {args.output}/')


if __name__ == '__main__':
    main()
