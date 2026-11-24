#!/usr/bin/env python3
"""Translate one multi-operation MLIR block into an equivalent PyTorch computation.

Output format is the one experiment 01 already uses (experiment.py TorchRunner):
    {'inputs': [{name, shape, dtype, generator, ...}],      one per @main argument, in order
     'graph':  [{'id', 'op', 'args', 'kwargs'}],            nodes in execution order
     'outputs': [node ids in MLIR return order, timer excluded]}
plus a readable Python rendering of the same graph (torch_block.py) for inspection
and for exporting to other compilers later.

Translation rules (MLIR semantics -> Torch):
  * named linalg ops ACCUMULATE into `outs`:
        batch_matmul / matmul      out = outs + a @ b
        conv_2d_nchw_fchw          out = outs + conv2d(x, w, stride, dilation)
        depthwise_conv_2d_nchw_chw out = outs + conv2d(x, w[:,None], groups=C, ...)
        pooling_nchw_max           out = max(outs, max_pool2d(x, window, ...))
        pooling_nchw_sum           out = outs + conv2d(x, ones[C,1,KH,KW], groups=C, ...)
    fill / transpose / broadcast overwrite `outs`.
  * linalg.generic is evaluated over its whole iteration space at once:
        each operand is moved into the loop space using its indexing map
        (projection/permutation -> permute + reshape with size-1 broadcast dims,
        constant index -> narrow), the scalar body becomes elementwise Torch ops,
        and reductions are recognised from the combining op that reads `%out`
        (addf -> sum, mulf -> prod, max/min forms -> amax/amin) and combined
        with the initial `outs` value, exactly like the MLIR loop does.
  * Anything else (d0+d1 maps, unknown body patterns, non-splat tensor
    constants, ...) raises Unsupported with a reason. Nothing is guessed.

Input generation follows experiment 01 (seeded normal, std 0.1), except:
  * a value that must be positive (rsqrt/sqrt/log operand, powf base, divisor) is
    traced back through positivity-preserving ops (add, mul, div, sum, matmul,
    transpose, ...; inside a body and across ops, see propagate_positive) and the
    block inputs it comes from are drawn from uniform(0.5, 1.5);
  * a scalar powf exponent passed as an input (GELU's x**3) is drawn as the
    integer 2 or 3, since a fractional power of a negative base is NaN.
Otherwise both MLIR and Torch would produce NaN/inf and the correctness check
would be meaningless. The domain used is recorded in each input spec.

This module only needs the standard library to translate. torch is imported
only by torch_operations() (used by the runner and by tests).

CLI (translates, writes JSON/py, runs nothing):
  python experiments/02_block_diagnosis/block_to_torch.py data/ops_and_blocks/albert_block_0.mlir --out /tmp/x
"""
import argparse
import json
import math
import re
import struct
from pathlib import Path

import block_catalog as cat

TORCH_DTYPES = {'f16': 'float16', 'bf16': 'bfloat16', 'f32': 'float32', 'f64': 'float64',
                'i1': 'bool', 'i8': 'int8', 'i16': 'int16', 'i32': 'int32', 'i64': 'int64',
                'index': 'int64'}
HARNESS_DTYPES = {'float32', 'float64', 'int32', 'int64'}   # what experiment.py can generate

BINARY = {'arith.addf': 'add', 'arith.addi': 'add', 'arith.subf': 'sub', 'arith.subi': 'sub',
          'arith.mulf': 'mul', 'arith.muli': 'mul', 'arith.divf': 'div',
          'arith.maximumf': 'maximum', 'arith.minimumf': 'minimum',
          'arith.maxnumf': 'fmax', 'arith.minnumf': 'fmin',
          'math.powf': 'pow', 'math.fpowi': 'pow'}
UNARY = {'arith.negf': 'neg', 'math.exp': 'exp', 'math.rsqrt': 'rsqrt', 'math.sqrt': 'sqrt',
         'math.tanh': 'tanh', 'math.erf': 'erf', 'math.log': 'log', 'math.absf': 'abs'}
CASTS = {'arith.truncf', 'arith.extf', 'arith.sitofp', 'arith.uitofp', 'arith.fptosi',
         'arith.index_cast'}
# Ordered/unordered variants are identical for finite values (inputs are always finite).
CMPF = {'oeq': 'eq', 'ueq': 'eq', 'one': 'ne', 'une': 'ne', 'ogt': 'gt', 'ugt': 'gt',
        'oge': 'ge', 'uge': 'ge', 'olt': 'lt', 'ult': 'lt', 'ole': 'le', 'ule': 'le'}
# Reduction combiners: body op that reads %out -> (reduce kind, combine op).
COMBINERS = {'arith.addf': ('sum', 'add'), 'arith.addi': ('sum', 'add'),
             'arith.mulf': ('prod', 'mul'), 'arith.muli': ('prod', 'mul'),
             'arith.maximumf': ('amax', 'maximum'), 'arith.maxnumf': ('amax', 'fmax'),
             'arith.minimumf': ('amin', 'minimum'), 'arith.minnumf': ('amin', 'fmin')}
POSITIVE_DOMAIN_OPS = {'math.rsqrt': [0], 'math.sqrt': [0], 'math.log': [0],
                       'math.powf': [0], 'arith.divf': [1]}

BODY_RE = re.compile(r'^\s*(%[\w.$-]+)\s*=\s*([a-z_]+\.[a-z_0-9]+)\s*(.*?)\s*:\s*(.+?)\s*$')
YIELD_RE = re.compile(r'^\s*linalg\.yield\s*(.*?)\s*(?::.*)?$')


class Unsupported(Exception):
    """The block uses a construct this translator does not handle (reason in the message)."""


# --------------------------------------------------------------------------- torch side

def torch_operations(torch, F):
    """The op table shared by the runner (block_experiment.py) and the tests.

    Superset of experiment.py's TorchRunner table; same calling convention:
    op(*tensor_args, **kwargs). `dtype` kwargs arrive already converted to torch dtypes.
    """
    def reduce_(a, kind, dims):
        if kind == 'sum':
            return a.sum(dim=dims, keepdim=True)
        if kind == 'amax':
            return a.amax(dim=dims, keepdim=True)
        if kind == 'amin':
            return a.amin(dim=dims, keepdim=True)
        if kind == 'prod':
            for d in dims:
                a = a.prod(dim=d, keepdim=True)
            return a
        raise ValueError(kind)

    return {
        # experiment 01 ops (kept so its graphs also run here)
        'matmul': torch.matmul, 'add': torch.add, 'mul': torch.mul, 'relu': torch.relu,
        'sum': torch.sum, 'conv2d': F.conv2d, 'max_pool2d': F.max_pool2d,
        'maximum': torch.maximum, 'permute': torch.permute, 'transpose': torch.transpose,
        'contiguous': lambda a: a.contiguous(), 'reshape': torch.reshape,
        'expand': lambda a, size: a.expand(size), 'cat': torch.cat, 'softmax': torch.softmax,
        'leaky_relu': F.leaky_relu, 'elu': F.elu,
        # block ops
        'sub': torch.sub, 'div': torch.div, 'minimum': torch.minimum,
        'fmax': torch.fmax, 'fmin': torch.fmin, 'pow': torch.pow, 'neg': torch.neg,
        'exp': torch.exp, 'rsqrt': torch.rsqrt, 'sqrt': torch.sqrt, 'tanh': torch.tanh,
        'erf': torch.erf, 'log': torch.log, 'abs': torch.abs, 'where': torch.where,
        'eq': torch.eq, 'ne': torch.ne, 'gt': torch.gt, 'ge': torch.ge, 'lt': torch.lt, 'le': torch.le,
        'to': lambda a, dtype: a.to(dtype),
        'const': lambda value, dtype: torch.tensor(float(value) if dtype.is_floating_point
                                                   else int(float(value)), dtype=dtype),
        'narrow': lambda a, dim, start, length: a.narrow(dim, start, length),
        'reduce': reduce_,
        'loop_index': lambda size, dim, rank: torch.arange(size, dtype=torch.int64).reshape(
            [size if i == dim else 1 for i in range(rank)]),
        'fill': lambda value, size, dtype: value.to(dtype).expand(size),
    }


# Readable rendering of each op (same semantics as torch_operations).
def _render(node):
    a, k = node['args'], node.get('kwargs', {})
    op = node['op']
    infix = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'eq': '==', 'ne': '!=',
             'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<='}
    if op in infix:
        return f'{a[0]} {infix[op]} {a[1]}'
    if op in ('maximum', 'minimum', 'fmax', 'fmin', 'pow', 'matmul'):
        return f'torch.{op}({a[0]}, {a[1]})'
    if op in ('exp', 'rsqrt', 'sqrt', 'tanh', 'erf', 'log', 'abs', 'relu'):
        return f'torch.{op}({a[0]})'
    if op == 'neg':
        return f'-{a[0]}'
    if op == 'where':
        return f'torch.where({a[0]}, {a[1]}, {a[2]})'
    if op == 'to':
        return f"{a[0]}.to(torch.{k['dtype']})"
    if op == 'const':
        value = k['value'] if isinstance(k['value'], (int, float)) else f"float('{k['value']}')"
        return f"torch.tensor({value}, dtype=torch.{k['dtype']})"
    if op == 'narrow':
        return f"{a[0]}.narrow({k['dim']}, {k['start']}, {k['length']})"
    if op == 'reshape':
        return f"{a[0]}.reshape({k['shape']})"
    if op == 'permute':
        return f"{a[0]}.permute({k['dims']})"
    if op == 'expand':
        return f"{a[0]}.expand({k['size']})"
    if op == 'reduce':
        if k['kind'] == 'prod':
            expr = a[0]
            for d in k['dims']:
                expr = f'{expr}.prod(dim={d}, keepdim=True)'
            return expr
        return f"{a[0]}.{k['kind']}(dim={k['dims']}, keepdim=True)"
    if op == 'loop_index':
        view = [k['size'] if i == k['dim'] else 1 for i in range(k['rank'])]
        return f"torch.arange({k['size']}).reshape({view})"
    if op == 'fill':
        return f"{a[0]}.to(torch.{k['dtype']}).expand({k['size']})"
    if op == 'conv2d':
        return (f"F.conv2d({a[0]}, {a[1]}, stride={k.get('stride', 1)}, "
                f"dilation={k.get('dilation', 1)}, groups={k.get('groups', 1)})")
    if op == 'max_pool2d':
        return (f"F.max_pool2d({a[0]}, kernel_size={k['kernel_size']}, stride={k['stride']}, "
                f"dilation={k['dilation']})")
    args = ', '.join(list(a) + [f'{key}={val!r}' for key, val in k.items()])
    return f"OPS['{op}']({args})"


def render_python(name, translation):
    """Readable PyTorch module equivalent to the graph (for inspection / export)."""
    params = [s['name'] for s in translation['inputs']]
    lines = ['"""Auto-generated by block_to_torch.py from ' + name + '.mlir. Do not edit."""',
             'import torch', 'import torch.nn.functional as F', '', '',
             'class Block(torch.nn.Module):',
             f'    def forward(self, {", ".join(params)}):']
    current_op = None
    for node in translation['graph']:
        if node.get('mlir_op') != current_op:
            current_op = node.get('mlir_op')
            lines.append(f'        # {current_op}')
        lines.append(f"        {node['id']} = {_render(node)}")
    lines.append(f"        return ({', '.join(translation['outputs'])},)")
    lines += ['', '', 'INPUTS = ' + json.dumps([{k: s[k] for k in ('name', 'shape', 'dtype')}
                                                for s in translation['inputs']])]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- graph builder

class Graph:
    def __init__(self):
        self.nodes, self.count, self.mlir_op = [], 0, None

    def emit(self, op, *args, **kwargs):
        node_id = f'n{self.count}'
        self.count += 1
        self.nodes.append(dict(id=node_id, op=op, args=list(args), kwargs=kwargs, mlir_op=self.mlir_op))
        return node_id


def parse_scalar(value, mlir_type):
    """MLIR scalar literal -> JSON-safe number (inf/nan as strings)."""
    value = value.strip()
    if value in ('true', 'false'):
        return 1 if value == 'true' else 0
    if value.startswith('0x'):
        raw = int(value, 16)
        if mlir_type == 'f64':
            result = struct.unpack('>d', raw.to_bytes(8, 'big'))[0]
        elif mlir_type == 'f32':
            result = struct.unpack('>f', raw.to_bytes(4, 'big'))[0]
        else:
            return raw
    else:
        result = float(value)
        if mlir_type.startswith('i') or mlir_type == 'index':
            return int(result)
    return result if math.isfinite(result) else str(result)


def to_loop_space(g, value, shape, amap, sizes):
    """Move a tensor indexed by `amap` into the n-D loop space (size-1 = broadcast)."""
    n = len(sizes)
    results = amap['results']
    if not shape:                     # scalar operand: broadcasts as a 0-d tensor
        return value
    if len(results) != len(shape):
        raise Unsupported('indexing map rank differs from operand rank')
    dims = [(pos, i) for pos, (kind, i) in enumerate(results) if kind == 'dim']
    if len({i for _, i in dims}) != len(dims):
        raise Unsupported('repeated loop dimension in one indexing map (diagonal access)')
    for pos, (kind, c) in enumerate(results):
        if kind == 'const':
            if not 0 <= c < shape[pos]:
                raise Unsupported('constant index out of range')
            value = g.emit('narrow', value, dim=pos, start=c, length=1)
    for pos, i in dims:
        if shape[pos] not in (1, sizes[i]):
            raise Unsupported('operand size disagrees with loop size')
    kept = [shape[pos] for pos, _ in dims]
    if len(kept) != len(shape):
        value = g.emit('reshape', value, shape=kept)
    order = sorted(range(len(dims)), key=lambda k: dims[k][1])
    if order != list(range(len(dims))):
        value = g.emit('permute', value, dims=order)
    if len(dims) != n:                # insert size-1 axes for loops this operand does not use
        full = [1] * n
        for pos, i in dims:
            full[i] = shape[pos]
        value = g.emit('reshape', value, shape=full)
    return value


def from_loop_space(g, value, out_shape, amap, sizes, reduced):
    """Inverse of to_loop_space for the output operand."""
    n = len(sizes)
    results = amap['results']
    if len(results) != len(out_shape):
        raise Unsupported('output map rank differs from output rank')
    dims = [(pos, i) for pos, (kind, i) in enumerate(results) if kind == 'dim']
    in_out = {i for _, i in dims}
    for pos, (kind, c) in enumerate(results):
        if kind == 'const' and (c != 0 or out_shape[pos] != 1):
            raise Unsupported('constant output index other than 0 into a size-1 dim')
    for i in range(n):
        if i not in in_out and i not in reduced:
            raise Unsupported('parallel loop missing from the output map')
    target = [sizes[i] if i in in_out else 1 for i in range(n)]
    value = g.emit('expand', value, size=target)
    ascending = sorted(in_out)
    value = g.emit('reshape', value, shape=[sizes[i] for i in ascending])
    perm = [ascending.index(i) for _, i in dims]
    if perm != list(range(len(perm))):
        value = g.emit('permute', value, dims=perm)
    return g.emit('reshape', value, shape=list(out_shape))


# --------------------------------------------------------------------------- per-op translation

def body_statements(op_lines):
    """[(result, opname, operand_names, attrs_text, type_text)], yields, block_args."""
    body = op_lines[1:]
    block_args, statements, yields = [], [], []
    for line in body:
        s = line.strip()
        if s.startswith('^bb'):
            block_args = re.findall(r'(%[\w.$-]+)\s*:', s)
        elif s.startswith('linalg.yield'):
            yields = cat.SSA_RE.findall(YIELD_RE.match(s)[1])
        elif s.startswith('%'):
            m = BODY_RE.match(s)
            if not m:
                raise Unsupported(f'unparsed body line: {s[:80]}')
            rest = re.sub(r'fastmath<[^>]*>', '', m[3])
            statements.append((m[1], m[2], cat.SSA_RE.findall(rest), rest, m[4]))
    return statements, yields, block_args


def result_dtype(opname, type_text):
    """MLIR scalar type of a body statement's result."""
    if ' to ' in type_text:                       # casts: 'f32 to f64'
        return type_text.split(' to ')[1].strip()
    if opname == 'arith.cmpf':
        return 'i1'
    return cat.split_top(type_text)[0].strip()


def translate_generic(g, op, op_lines, values, dtypes, maps, input_args, domains):
    header = op_lines[0]
    ins, _ = cat.operand_group(header, 'ins')
    outs, _ = cat.operand_group(header, 'outs')
    maps_m = re.search(r'indexing_maps\s*=\s*\[(.*?)\]\s*,\s*iterator_types', header)
    if not maps_m:
        raise Unsupported('generic without indexing_maps')
    texts = []
    for item in cat.split_top(maps_m[1]):
        if item.startswith('#'):
            if item not in maps:
                raise Unsupported(f'missing affine map {item}')
            texts.append(maps[item])
        else:
            texts.append(item[len('affine_map<'):-1])
    amaps = [cat.parse_affine_map(t) for t in texts]
    if any(m['complex'] for m in amaps):
        raise Unsupported('non-projected affine map (e.g. d0 + d1)')
    sizes, iterators = op['loop_sizes'], op['iterators']
    if any(s is None for s in sizes):
        raise Unsupported('unknown loop size')
    if any(it not in ('parallel', 'reduction') for it in iterators):
        raise Unsupported(f'iterator type {set(iterators)}')
    reduced = [i for i, it in enumerate(iterators) if it == 'reduction']

    statements, yields, block_args = body_statements(op_lines)
    if len(block_args) != len(ins) + len(outs) or len(yields) != len(outs):
        raise Unsupported('body arguments/yields do not match operands')
    operands = ins + outs
    out_args = block_args[len(ins):]
    local, local_dtype, depends_on_out = {}, {}, {}
    for k, (arg, operand) in enumerate(zip(block_args, operands)):
        shape = (operand['type'] or {}).get('shape')
        if shape is None:
            raise Unsupported('operand without a static type')
        local[arg] = to_loop_space(g, values[operand['name']], shape, amaps[k], sizes)
        local_dtype[arg] = (operand['type'] or {}).get('dtype')
        depends_on_out[arg] = arg in out_args

    def lookup(name):
        if name in local:
            return local[name]
        if name in values:            # captured from @main (constants)
            return values[name]
        raise Unsupported(f'unknown SSA value {name} in body')

    def operand_of(name):
        return operands[block_args.index(name)] if name in block_args else None

    def note_domain(opname, operand_names):
        # Values (block inputs OR intermediate results) that must be positive for the
        # op to be finite. Intermediates are traced back to inputs by propagate_positive().
        positions = POSITIVE_DOMAIN_OPS.get(opname, [])
        if opname == 'math.powf' and len(operand_names) > 1:
            exponent = operand_of(operand_names[1])
            if exponent and exponent['name'] in input_args and \
                    math.prod((exponent['type'] or {}).get('shape') or [1]) == 1:
                # scalar exponent given as an input (e.g. GELU's x**3): draw a small integer
                # exponent; then any base is valid and the base needs no positive domain.
                domains[exponent['name']] = 'exponent'
                positions = []
        for idx in positions:
            if idx < len(operand_names):
                require_positive(operand_names[idx], set())

    def require_positive(name, seen):
        # A block argument -> mark its operand; a value computed earlier in this body
        # by a positivity-preserving op -> require its own operands to be positive.
        if name in seen:
            return
        seen.add(name)
        operand = operand_of(name)
        if operand is not None:
            domains.setdefault(operand['name'], 'positive')
        elif name in defs and defs[name][1] in POSITIVE_PRESERVING:
            for n in defs[name][2]:
                require_positive(n, seen)

    # Which statements matter for each yielded value (and does it read %out?).
    defs = {s[0]: s for s in statements}
    for res, opname, names, _, _ in statements:
        depends_on_out[res] = any(depends_on_out.get(n, False) for n in names)

    def emit_stmt(res, opname, names, rest, type_text):
        note_domain(opname, names)
        args = [lookup(n) for n in names]
        dtype = result_dtype(opname, type_text)
        if opname in BINARY:
            out = g.emit(BINARY[opname], *args)
        elif opname in UNARY:
            out = g.emit(UNARY[opname], *args)
        elif opname in CASTS:
            if dtype not in TORCH_DTYPES:
                raise Unsupported(f'cast to {dtype}')
            out = g.emit('to', *args, dtype=TORCH_DTYPES[dtype])
        elif opname == 'arith.cmpf':
            pred = rest.split(',')[0].strip()
            if pred not in CMPF:
                raise Unsupported(f'cmpf predicate {pred}')
            out = g.emit(CMPF[pred], *args)
        elif opname == 'arith.select':
            out = g.emit('where', *args)
        elif opname == 'arith.constant':
            literal = rest.strip()
            if 'dense' in literal:
                raise Unsupported('tensor constant inside a generic body')
            out = g.emit('const', value=parse_scalar(literal, dtype), dtype=TORCH_DTYPES[dtype])
        elif opname == 'linalg.index':
            d = int(re.search(r'\d+', rest)[0])
            out = g.emit('loop_index', size=sizes[d], dim=d, rank=len(sizes))
        else:
            raise Unsupported(f'body op {opname}')
        local[res], local_dtype[res] = out, dtype
        return out

    def needed(roots):
        """Statements (in order) needed to compute `roots`."""
        need, stack = set(), list(roots)
        while stack:
            name = stack.pop()
            if name in defs and name not in need:
                need.add(name)
                stack.extend(defs[name][2])
        return [s for s in statements if s[0] in need]

    results = []
    for j, (yielded, out_operand) in enumerate(zip(yields, outs)):
        out_arg = out_args[j]
        out_shape = out_operand['type']['shape']
        out_dtype = out_operand['type']['dtype']
        out_map = amaps[len(ins) + j]
        if not reduced:
            for s in needed([yielded]):
                if s[0] not in local:
                    emit_stmt(*s)
            value = lookup(yielded)
        else:
            if yielded not in defs:
                raise Unsupported('reduction yields a block argument directly')
            res, opname, names, rest, _ = defs[yielded]
            kind = combine = partial = None
            if opname in COMBINERS and len(names) == 2 and out_arg in names:
                other = names[1 - names.index(out_arg)]
                if not depends_on_out.get(other, False):
                    (kind, combine), partial = COMBINERS[opname], other
            elif opname == 'arith.select' and len(names) == 3 and names[0] in defs:
                c_res, c_op, c_names, c_rest, _ = defs[names[0]]
                picked_if_true, picked_if_false = names[1], names[2]
                if (c_op == 'arith.cmpf' and set(c_names) == {picked_if_true, picked_if_false}
                        and out_arg in c_names):
                    other = c_names[1 - c_names.index(out_arg)]
                    pred = CMPF.get(c_rest.split(',')[0].strip())
                    if not depends_on_out.get(other, False) and pred in ('gt', 'ge', 'lt', 'le'):
                        larger_first = pred in ('gt', 'ge')        # cond true when c0 > c1
                        picks_first = picked_if_true == c_names[0]
                        kind = 'amax' if larger_first == picks_first else 'amin'
                        combine, partial = ('maximum' if kind == 'amax' else 'minimum'), other
            if kind is None:
                raise Unsupported('reduction body is not a recognised combiner pattern')
            for s in needed([partial]):
                if s[0] not in local:
                    emit_stmt(*s)
            part = g.emit('expand', lookup(partial), size=list(sizes))
            part = g.emit('reduce', part, kind=kind, dims=reduced)
            value = g.emit(combine, local[out_arg], part)
        if local_dtype.get(yielded, out_dtype) != out_dtype:
            value = g.emit('to', value, dtype=TORCH_DTYPES[out_dtype])
        results.append(from_loop_space(g, value, out_shape, out_map, sizes, set(reduced)))
    if len(results) != 1:
        raise Unsupported('generic with several results')
    return results[0]


def translate_named(g, op, op_lines, values):
    header = op_lines[0]
    kind = op['kind']
    ins, _ = cat.operand_group(header, 'ins')
    outs, _ = cat.operand_group(header, 'outs')
    x = [values[o['name']] for o in ins]
    out_val = values[outs[0]['name']]
    out_shape = outs[0]['type']['shape']
    out_dtype = outs[0]['type']['dtype']
    attrs = op.get('attrs', {})
    if kind in ('matmul', 'batch_matmul'):
        return g.emit('add', out_val, g.emit('matmul', x[0], x[1]))
    if kind == 'conv_2d_nchw_fchw':
        return g.emit('add', out_val, g.emit('conv2d', x[0], x[1], stride=attrs.get('strides', [1, 1]),
                                             dilation=attrs.get('dilations', [1, 1]), groups=1))
    if kind == 'depthwise_conv_2d_nchw_chw':
        c, kh, kw = ins[1]['type']['shape']
        w = g.emit('reshape', x[1], shape=[c, 1, kh, kw])
        return g.emit('add', out_val, g.emit('conv2d', x[0], w, stride=attrs.get('strides', [1, 1]),
                                             dilation=attrs.get('dilations', [1, 1]), groups=c))
    if kind == 'pooling_nchw_max':
        window = ins[1]['type']['shape']
        pooled = g.emit('max_pool2d', x[0], kernel_size=window, stride=attrs.get('strides', [1, 1]),
                        dilation=attrs.get('dilations', [1, 1]))
        return g.emit('maximum', out_val, pooled)
    if kind == 'pooling_nchw_sum':
        # out = outs + sum over each window: a depthwise conv with an all-ones kernel
        # (handles stride and dilation; the window operand only carries the shape).
        c = ins[0]['type']['shape'][1]
        kh, kw = ins[1]['type']['shape']
        ones = g.emit('fill', g.emit('const', value=1.0, dtype=TORCH_DTYPES[out_dtype]),
                      size=[c, 1, kh, kw], dtype=TORCH_DTYPES[out_dtype])
        return g.emit('add', out_val, g.emit('conv2d', x[0], ones, stride=attrs.get('strides', [1, 1]),
                                             dilation=attrs.get('dilations', [1, 1]), groups=c))
    if kind == 'fill':
        return g.emit('fill', x[0], size=out_shape, dtype=TORCH_DTYPES[out_dtype])
    if kind == 'transpose':
        return g.emit('permute', x[0], dims=attrs['permutation'])
    if kind == 'broadcast':
        added = set(attrs.get('dimensions', []))
        view = [1 if i in added else out_shape[i] for i in range(len(out_shape))]
        return g.emit('expand', g.emit('reshape', x[0], shape=view), size=out_shape)
    raise Unsupported(f'named op linalg.{kind}')


# --------------------------------------------------------------------------- input domains

# Body ops that map positive operands to a positive result (with non-negative constants).
POSITIVE_PRESERVING = {'arith.addf', 'arith.mulf', 'arith.divf', 'arith.truncf', 'arith.extf',
                       'math.sqrt', 'math.rsqrt', 'math.powf', 'math.fpowi',
                       'arith.maximumf', 'arith.maxnumf', 'arith.minimumf', 'arith.minnumf'}
POSITIVE_PRESERVING_NAMED = {'matmul', 'batch_matmul', 'conv_2d_nchw_fchw', 'depthwise_conv_2d_nchw_chw',
                             'pooling_nchw_max', 'pooling_nchw_sum', 'transpose', 'broadcast'}


def propagate_positive(domains, producers, consts):
    """Trace 'must be positive' from intermediate values back to block inputs.

    Example (LayerNorm tail):  rsqrt(sum(x) / 768 + eps).  rsqrt needs a positive
    operand; it comes from addf(+eps) <- divf(/768) <- sum(x), all of which keep
    positive values positive, so x (and the sum's initial `outs`) are drawn from
    the positive domain. In the real model x is a square, so this matches reality.
    Ops that can turn positive into negative (subf, negf, select, ...) stop the
    trace; such blocks may still produce NaN and verify_blocks.py reports them.
    """
    scalars = {}
    for c in consts:
        try:
            scalars[c['name']] = float(parse_scalar(c['value'], c['type'].strip()))
        except (ValueError, TypeError):
            pass
    for result in reversed(list(producers)):         # producers are in program order
        if domains.get(result) != 'positive':
            continue
        kind, op_lines = producers[result]
        header = op_lines[0]
        if kind == 'generic':
            statements, _, _ = body_statements(op_lines)
            if {s[1] for s in statements} - POSITIVE_PRESERVING:
                continue
            if any(scalars.get(n, 0.0) < 0 for s in statements for n in s[2]):
                continue
        elif kind not in POSITIVE_PRESERVING_NAMED:
            continue
        for o in cat.operand_group(header, 'ins')[0] + cat.operand_group(header, 'outs')[0]:
            domains.setdefault(o['name'], 'positive')


# --------------------------------------------------------------------------- whole block

def translate_text(name, text):
    """Translate block text. Raises Unsupported or returns the translation dict."""
    main = cat.read_main(text)
    g = Graph()
    values, dtypes, domains, producers = {}, {}, {}, {}
    inputs = []
    for a in main['args']:
        if a['dynamic'] or a['shape'] is None:
            raise Unsupported('dynamic or non-tensor argument')
        dtype = TORCH_DTYPES.get(a['dtype'])
        if dtype not in HARNESS_DTYPES:
            raise Unsupported(f'argument dtype {a["dtype"]} not supported by the harness')
        arg_name = a['name'].lstrip('%')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', arg_name):
            raise Unsupported(f'argument name {arg_name} is not an identifier')
        values[a['name']] = arg_name
        dtypes[a['name']] = a['dtype']
        inputs.append(dict(name=arg_name, shape=a['shape'], dtype=dtype, role=a['role']))
    input_args = set(values)

    g.mlir_op = '@main constants'
    for c in main['consts']:
        if 'dense<' in c['value']:
            splat = re.fullmatch(r'dense<([^\[\]<>,]+)>', c['value'].strip())
            t = cat.parse_type(c['type'])
            if not splat or t['shape'] is None:
                raise Unsupported('non-splat tensor constant')
            scalar = g.emit('const', value=parse_scalar(splat[1], t['dtype']), dtype=TORCH_DTYPES[t['dtype']])
            values[c['name']] = g.emit('expand', scalar, size=t['shape'])
        else:
            t = c['type'].strip()
            if t not in TORCH_DTYPES:
                raise Unsupported(f'constant type {t}')
            values[c['name']] = g.emit('const', value=parse_scalar(c['value'], t), dtype=TORCH_DTYPES[t])

    for position, (om, op_lines) in enumerate(main['ops']):
        op = cat.parse_op(om, op_lines, main['maps'], position)
        g.mlir_op = f"{op['tag']} linalg.{op['kind']}"
        if op['kind'] == 'generic':
            result = translate_generic(g, op, op_lines, values, dtypes, main['maps'], input_args, domains)
        else:
            result = translate_named(g, op, op_lines, values)
        values[op['result']] = result
        producers[op['result']] = (op['kind'], op_lines)
    propagate_positive(domains, producers, main['consts'])

    outputs = []
    for r in main['returned']:
        if r not in values:
            continue                  # the i64 nanoTime delta is computed outside linalg
        outputs.append(values[r])
    if not outputs:
        raise Unsupported('no tensor result')

    for spec in inputs:
        if spec['dtype'].startswith('int'):
            spec.update(generator='integers', low=0, high=16)
        elif domains.get('%' + spec['name']) == 'positive':
            spec.update(generator='uniform', low=0.5, high=1.5, domain='positive')
        elif domains.get('%' + spec['name']) == 'exponent':
            spec.update(generator='integers', low=2, high=4, domain='exponent')
        else:
            spec.update(generator='normal', std=0.1)
    used = {a for n in g.nodes for a in n['args']} | set(outputs)
    graph = [dict(id=n['id'], op=n['op'], args=n['args'], kwargs=n['kwargs'], mlir_op=n['mlir_op'])
             for n in g.nodes]
    return dict(name=name, inputs=inputs, graph=graph, outputs=outputs,
                unused_inputs=[s['name'] for s in inputs if s['name'] not in used],
                n_nodes=len(graph))


def translate_file(path):
    """{'status': 'ok', ...translation} or {'status': 'unsupported', 'reason': ...}."""
    path = Path(path)
    try:
        result = translate_text(path.stem, path.read_text())
        result['status'] = 'ok'
        return result
    except Unsupported as error:
        return dict(name=path.stem, status='unsupported', reason=str(error))
    except Exception as error:      # parser bug or malformed file: report, never guess
        return dict(name=path.stem, status='error', reason=f'{type(error).__name__}: {error}')


def harness_graph(translation):
    """The parts experiment.py's manifest expects under 'torch' (mlir_op labels dropped)."""
    return dict(graph=[{k: n[k] for k in ('id', 'op', 'args', 'kwargs')} for n in translation['graph']],
                outputs=translation['outputs'])


def main():
    parser = argparse.ArgumentParser(description='Translate block .mlir files to Torch graphs (runs nothing).')
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='Folder for <name>.torch.json and <name>.py')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        t = translate_file(path)
        (args.out / f'{path.stem}.torch.json').write_text(json.dumps(t, indent=1))
        if t['status'] == 'ok':
            (args.out / f'{path.stem}.py').write_text(render_python(path.stem, t))
        print(f"{path.stem}: {t['status']}" + (f" ({t['reason']})" if t['status'] != 'ok' else
                                               f" ({t['n_nodes']} nodes)"))


if __name__ == '__main__':
    main()
