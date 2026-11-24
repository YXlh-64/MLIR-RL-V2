#!/usr/bin/env python3
"""Independent NumPy reference interpreter for block .mlir files.

Used to check block_to_torch.py translations while no MLIR build is available.
It shares NO translation logic with block_to_torch.py: it does not recognise
patterns (no "this is a sum", no permute/reshape planning). Instead it executes
each linalg op literally, the way the MLIR loop nest does:

  * every op (named ones included) is turned into its generic form:
    indexing maps + iterator types + a scalar body;
  * the parallel loops are vectorised with NumPy index grids, the reduction
    loops are executed ONE ITERATION AT A TIME in lexicographic order, feeding
    the yielded value back as %out (exactly the MLIR sequential semantics);
  * operands are read with the indexing map evaluated as a real affine
    expression (so d0*2 + d4 conv windows are just index arithmetic);
  * the body is interpreted line by line (arith.*, math.*).

Arithmetic is float64 (inputs are the float32 arrays given to Torch, upcast),
so the result is a higher-precision version of what MLIR/Torch compute in f32.
truncf rounds to float32 as MLIR does. Only block_catalog.read_main is reused,
to split the file into ops (it does not interpret anything).

Cost is proportional to the iteration-space volume; evaluate() refuses blocks
above `max_volume` (raises TooLarge) instead of taking forever.
"""
import itertools
import math
import re
import struct

import numpy as np

import block_catalog as cat


class RefUnsupported(Exception):
    pass


class TooLarge(Exception):
    pass


SSA = re.compile(r'%[\w.$-]+')
BODY_LINE = re.compile(r'^\s*(%[\w.$-]+)\s*=\s*([a-z_]+\.[a-z_0-9]+)\s*(.*?)\s*:\s*(.+?)\s*$')
YIELD = re.compile(r'^\s*linalg\.yield\s*(.*?)\s*(?::.*)?$')


# ----------------------------------------------------------------------------- scalars / maps

def scalar_value(text, mlir_type):
    v = text.strip()
    if v in ('true', 'false'):
        return float(v == 'true')
    if v.lower().startswith('0x'):
        bits = int(v, 16)
        t = mlir_type.strip()
        if t == 'f32':
            return struct.unpack('<f', struct.pack('<I', bits))[0]
        if t == 'f64':
            return struct.unpack('<d', struct.pack('<Q', bits))[0]
        if t == 'f16':
            return float(np.frombuffer(struct.pack('<H', bits), np.float16)[0])
        raise RefUnsupported(f'hex constant of type {t}')
    return float(v)


def compile_map(text):
    """'(d0, d1)[s0] -> (d0 * 2 + d1, 0)' -> (n_dims, [callable(dims) for each result])."""
    m = re.fullmatch(r'\s*\(([^)]*)\)\s*(?:\[[^\]]*\])?\s*->\s*\((.*)\)\s*', text)
    if not m:
        raise RefUnsupported(f'affine map {text}')
    dims = [d.strip() for d in m[1].split(',') if d.strip()]
    results = [r.strip() for r in m[2].split(',')] if m[2].strip() else []
    fns = []
    for r in results:
        expr = r.replace('floordiv', '//').replace('mod', '%')
        if not re.fullmatch(r'[d0-9\s+*()/%-]*', expr):
            raise RefUnsupported(f'affine expression {r}')
        code = compile(re.sub(r'\bd(\d+)\b', r'D[\1]', expr), '<map>', 'eval')
        fns.append(lambda D, code=code: eval(code, {'__builtins__': {}}, {'D': D}))
    return len(dims), fns, results


# ----------------------------------------------------------------------------- body interpreter

def _f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


CMP = {'oeq': np.equal, 'ueq': np.equal, 'one': np.not_equal, 'une': np.not_equal,
       'ogt': np.greater, 'ugt': np.greater, 'oge': np.greater_equal, 'uge': np.greater_equal,
       'olt': np.less, 'ult': np.less, 'ole': np.less_equal, 'ule': np.less_equal}


def _erf(x):
    # math.erf element by element would be slow on big arrays; torch's erf in
    # float64 is a library function, not translation logic.
    import torch
    return torch.special.erf(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))).numpy()


def run_body(lines, env):
    """Interpret body lines with `env` {ssa: array}; returns the yielded values."""
    for line in lines:
        y = YIELD.match(line)
        if y:
            return [env[n] for n in SSA.findall(y[1])]
        if line.strip().startswith('cf.assert'):       # runtime check, no value
            continue
        m = BODY_LINE.match(line)
        if not m:
            if line.strip().startswith('^bb') or not line.strip() or line.strip().startswith('}'):
                continue
            raise RefUnsupported(f'body line {line.strip()}')
        res, opname, rest, types = m.groups()
        if opname == 'arith.constant':
            env[res] = scalar_value(rest, types.split()[-1])
            continue
        pred = None
        if opname in ('arith.cmpf', 'arith.cmpi'):
            pred, rest = rest.split(',', 1)
            pred = pred.strip()
        a = [env[n] for n in SSA.findall(rest)]
        if opname in ('arith.addf', 'arith.addi'):
            v = a[0] + a[1]
        elif opname in ('arith.subf', 'arith.subi'):
            v = a[0] - a[1]
        elif opname in ('arith.mulf', 'arith.muli'):
            v = a[0] * a[1]
        elif opname == 'arith.divf':
            v = a[0] / a[1]
        elif opname == 'arith.negf':
            v = -a[0]
        elif opname in ('arith.maximumf', 'arith.maxnumf'):
            v = (np.maximum if opname == 'arith.maximumf' else np.fmax)(a[0], a[1])
        elif opname in ('arith.minimumf', 'arith.minnumf'):
            v = (np.minimum if opname == 'arith.minimumf' else np.fmin)(a[0], a[1])
        elif opname == 'arith.cmpf':
            if pred not in CMP:
                raise RefUnsupported(f'cmpf {pred}')
            v = CMP[pred](a[0], a[1])
            if pred.startswith('u'):
                v = v | np.isnan(a[0]) | np.isnan(a[1])
        elif opname == 'arith.select':
            v = np.where(a[0], a[1], a[2])
        elif opname == 'math.exp':
            v = np.exp(a[0])
        elif opname == 'math.rsqrt':
            v = 1.0 / np.sqrt(a[0])
        elif opname == 'math.sqrt':
            v = np.sqrt(a[0])
        elif opname == 'math.tanh':
            v = np.tanh(a[0])
        elif opname == 'math.erf':
            v = _erf(a[0])
        elif opname == 'math.log':
            v = np.log(a[0])
        elif opname == 'math.absf':
            v = np.abs(a[0])
        elif opname in ('math.powf', 'math.fpowi'):
            v = np.power(a[0], a[1])
        elif opname == 'arith.truncf':
            v = _f32(a[0])
        elif opname in ('arith.extf', 'arith.sitofp', 'arith.uitofp'):
            v = np.asarray(a[0], dtype=np.float64)
        else:
            raise RefUnsupported(f'body op {opname}')
        env[res] = v
    raise RefUnsupported('body without linalg.yield')


# ----------------------------------------------------------------------------- loop engine

MAX_PARALLEL = 2e7   # elements of the vectorised parallel space (~8 float64 temporaries of it live at once)


def loop_nest(operands, maps, iterators, body, n_outs, max_volume):
    """Execute one op in generic form.

    operands  list of float64 arrays (ins then outs; a 0-d array for scalars)
    maps      list of (n_dims, [result fns], [result texts]) per operand
    iterators ['parallel'|'reduction'] per loop dim
    body      fn(list of operand element arrays) -> list of yielded arrays
    """
    n = len(iterators)
    sizes = [None] * n
    for arr, (_, _, texts) in zip(operands, maps):
        for axis, t in enumerate(texts):
            dm = re.fullmatch(r'd(\d+)', t)
            if dm and sizes[int(dm[1])] is None:
                sizes[int(dm[1])] = arr.shape[axis]
    if any(s is None for s in sizes):
        raise RefUnsupported(f'cannot infer loop sizes {sizes}')
    volume = math.prod(sizes)
    if volume > max_volume:
        raise TooLarge(volume)
    par = [k for k in range(n) if iterators[k] == 'parallel']
    red = [k for k in range(n) if iterators[k] != 'parallel']
    pshape = tuple(sizes[k] for k in par)
    if math.prod(pshape) > MAX_PARALLEL:          # memory guard (the job has a few GB per worker)
        raise TooLarge(f'parallel space {math.prod(pshape)} elements')
    D = [0] * n
    for i, k in enumerate(par):
        D[k] = np.arange(sizes[k]).reshape([-1 if j == i else 1 for j in range(len(par))])

    def read(arr, fns):
        if arr.ndim == 0:
            return np.broadcast_to(arr, pshape)
        return np.broadcast_to(arr[tuple(np.asarray(f(D)) for f in fns)], pshape)

    outs = operands[-n_outs:]
    out_maps = maps[-n_outs:]
    for k in red:                                  # outs indexing must not use reduction dims
        D[k] = 0
    acc = [read(o, m[1]).copy() for o, m in zip(outs, out_maps)]
    ins = operands[:-n_outs]
    in_maps = maps[:-n_outs]
    for point in itertools.product(*[range(sizes[k]) for k in red]):
        for k, v in zip(red, point):
            D[k] = v
        vals = [read(a, m[1]) for a, m in zip(ins, in_maps)]
        acc = [np.broadcast_to(np.asarray(v, dtype=np.float64), pshape).copy()
               for v in body(vals + acc)]
    for k in red:
        D[k] = 0
    results = []
    for o, m, a in zip(outs, out_maps, acc):
        r = o.copy()
        idx = tuple(np.broadcast_to(np.asarray(f(D)), pshape) for f in m[1])
        r[idx] = a
        results.append(r)
    return results, volume


def attr_pair(header, key):
    m = re.search(key + r'\s*=\s*dense<\s*\[?\s*([0-9,\s]+?)\s*\]?\s*>', header)
    if not m:
        return [1, 1]
    v = [int(x) for x in m[1].split(',')]
    return v * 2 if len(v) == 1 else v


def operand_names(header, keyword):
    m = re.search(keyword + r'\(([^:()]*?)\s*:', header)
    return SSA.findall(m[1]) if m else []


def named_generic(kind, header, ins, out):
    """Generic form of a named op: (maps, iterators, body)."""
    P, R = 'parallel', 'reduction'
    mac = lambda v: [v[2] + v[0] * v[1]]              # out + a*b
    if kind == 'matmul':                               # d0=m d1=n d2=k
        return ['(d0,d1,d2)->(d0,d2)', '(d0,d1,d2)->(d2,d1)', '(d0,d1,d2)->(d0,d1)'], [P, P, R], mac
    if kind == 'batch_matmul':                         # d0=b d1=m d2=n d3=k
        return (['(d0,d1,d2,d3)->(d0,d1,d3)', '(d0,d1,d2,d3)->(d0,d3,d2)', '(d0,d1,d2,d3)->(d0,d1,d2)'],
                [P, P, P, R], mac)
    sh, sw = attr_pair(header, 'strides')
    dh, dw = attr_pair(header, 'dilations')
    win = f'd2*{sh}+d5*{dh}, d3*{sw}+d6*{dw}'
    if kind == 'conv_2d_nchw_fchw':                    # n f oh ow c kh kw
        return ([f'(d0,d1,d2,d3,d4,d5,d6)->(d0,d4,{win})', '(d0,d1,d2,d3,d4,d5,d6)->(d1,d4,d5,d6)',
                 '(d0,d1,d2,d3,d4,d5,d6)->(d0,d1,d2,d3)'], [P, P, P, P, R, R, R], mac)
    win6 = f'd2*{sh}+d4*{dh}, d3*{sw}+d5*{dw}'
    if kind == 'depthwise_conv_2d_nchw_chw':           # n c oh ow kh kw
        return ([f'(d0,d1,d2,d3,d4,d5)->(d0,d1,{win6})', '(d0,d1,d2,d3,d4,d5)->(d1,d4,d5)',
                 '(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3)'], [P, P, P, P, R, R], mac)
    if kind in ('pooling_nchw_max', 'pooling_nchw_sum'):   # n c oh ow kh kw
        body = ((lambda v: [np.maximum(v[2], v[0])]) if kind.endswith('max')
                else (lambda v: [v[2] + v[0]]))
        return ([f'(d0,d1,d2,d3,d4,d5)->(d0,d1,{win6})', '(d0,d1,d2,d3,d4,d5)->(d4,d5)',
                 '(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3)'], [P, P, P, P, R, R], body)
    rank = out.ndim
    dims = ','.join(f'd{i}' for i in range(rank))
    if kind == 'transpose':
        perm = [int(x) for x in re.search(r'permutation\s*=\s*\[([^\]]*)\]', header)[1].split(',')]
        src = [None] * rank
        for i, p in enumerate(perm):                   # out[i] indexes input dim perm[i]
            src[p] = f'd{i}'
        return [f'({dims})->({",".join(src)})', f'({dims})->({dims})'], [P] * rank, lambda v: [v[0]]
    if kind == 'broadcast':
        added = {int(x) for x in re.search(r'dimensions\s*=\s*\[([^\]]*)\]', header)[1].split(',')}
        kept = ','.join(f'd{i}' for i in range(rank) if i not in added)
        return [f'({dims})->({kept})', f'({dims})->({dims})'], [P] * rank, lambda v: [v[0]]
    if kind == 'fill':
        return [f'({dims})->()', f'({dims})->({dims})'], [P] * rank, lambda v: [v[0]]
    raise RefUnsupported(f'named op linalg.{kind}')


def evaluate(text, inputs, max_volume=2e8):
    """Run a block. inputs: {arg name without %: array}. Returns [float64 arrays] in return order."""
    main = cat.read_main(text)
    env = {}
    for a in main['args']:
        env[a['name']] = np.asarray(inputs[a['name'].lstrip('%')], dtype=np.float64)
    for c in main['consts']:
        val = c['value'].strip()
        t = c['type'].strip()
        splat = re.fullmatch(r'dense<([^\[\]<>,]+)>', val)
        if splat:
            shape = [int(x) for x in re.search(r'tensor<((?:\d+x)*)', t)[1].split('x') if x]
            env[c['name']] = np.full(shape, scalar_value(splat[1], t.split('x')[-1].rstrip('>')))
        elif 'dense' in val:
            raise RefUnsupported('non-splat constant')
        else:
            env[c['name']] = np.asarray(scalar_value(val, t))
    total = 0
    for om, op_lines in main['ops']:
        header = op_lines[0]
        kind = om['kind']
        ins = [env[n] for n in operand_names(header, 'ins')]
        outs = [env[n] for n in operand_names(header, 'outs')]
        if kind == 'generic':
            map_names = re.search(r'indexing_maps\s*=\s*\[([^\]]*)\]', header)[1]
            map_texts = []
            for name in [x.strip() for x in map_names.split(',')]:
                inline = re.fullmatch(r'affine_map<(.*)>', name)
                map_texts.append(inline[1] if inline else main['maps'][name])
            iterators = re.findall(r'"(parallel|reduction)"',
                                   re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', header)[1])
            bb = next(l for l in op_lines if l.strip().startswith('^bb'))
            names = re.findall(r'(%[\w.$-]+)\s*:', bb)
            body_lines = op_lines[op_lines.index(bb) + 1:]

            def body(vals, names=names, body_lines=body_lines):
                local = {k: v for k, v in env.items() if np.ndim(v) == 0}   # @main scalar constants
                local.update(zip(names, vals))
                return run_body(body_lines, local)
        else:
            map_texts, iterators, body = named_generic(kind, header, ins, outs[0])
        maps = [compile_map(t) for t in map_texts]
        try:
            results, volume = loop_nest(ins + outs, maps, iterators, body, len(outs), max_volume - total)
        except TooLarge as e:
            raise TooLarge(f'parallel space too large for memory ({e})' if 'parallel' in str(e)
                           else f'block iteration volume exceeds {max_volume:.3g}')
        total += volume
        env[om['res']] = results[0]
    evaluate.last_volume = total
    return [env[r] for r in main['returned'] if r in env and isinstance(env[r], np.ndarray)
            and r not in [c['name'] for c in main['consts']]]
