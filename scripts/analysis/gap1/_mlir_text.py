"""Stdlib-only text parser for linalg ops in MLIR files, shared by the A scripts.

Mirrors the AST dumper (tools/ast_dumper/main.cpp): linalg ops in lexical order,
ops with zero loops skipped, tags `operation_<k>` numbered over the kept ops,
producer edges = direct SSA uses between kept linalg ops (operand index over ins+outs).
"""
import csv
import glob
import os
import re
from dataclasses import dataclass, field

# Named ops: name -> (kind, body ops). kind: 'ew' all-parallel, 'red' has reductions.
_EW = {
    'add': 'arith.addf', 'sub': 'arith.subf', 'mul': 'arith.mulf', 'div': 'arith.divf',
    'max': 'arith.maximumf', 'min': 'arith.minimumf', 'exp': 'math.exp', 'log': 'math.log',
    'abs': 'math.absf', 'ceil': 'math.ceil', 'floor': 'math.floor', 'negf': 'arith.negf',
    'tanh': 'math.tanh', 'rsqrt': 'math.rsqrt', 'sqrt': 'math.sqrt', 'square': 'arith.mulf',
    'reciprocal': 'arith.divf', 'erf': 'math.erf', 'powf': 'math.powf', 'select': 'arith.select',
    'copy': '', 'fill': '', 'transpose': '', 'broadcast': '', 'map': '',
    'elemwise_unary': '', 'elemwise_binary': '',
}
_CONTRACTION = re.compile(r'^(matmul|batch_matmul|batch_reduce_matmul|matvec|vecmat|batch_matvec|batch_vecmat'
                          r'|dot|contract|mmt4d|batch_mmt4d|quantized_\w+)')
_WINDOW = re.compile(r'^(conv_|depthwise_conv_|pooling_)')

_OP_START = re.compile(r'^\s*(?:(%[\w$.\-]+)(?::(\d+))?\s*=\s*)?linalg\.(\w+)')
_TAG = re.compile(r'\btag\s*=\s*"([^"]+)"')
_ALIAS = re.compile(r'^(#[\w$.\-]+)\s*=\s*(affine_map<.*>)\s*$')
_BODY = re.compile(r'\b((?:arith|math)\.\w+)')


def split_top(s, sep=','):
    """Split on `sep` outside (), <>, [], {}."""
    out, depth, cur = [], 0, ''
    for i, ch in enumerate(s):
        if ch in '(<[{':
            depth += 1
        elif ch in ')]}' or (ch == '>' and s[i - 1:i] != '-'):
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _balanced(text, start, open_ch, close_ch):
    """Content between text[start] == open_ch and its matching close_ch."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return ''


def _section(text, kw):
    m = re.search(r'\b' + kw + r'\(', text)
    if not m:
        return [], []
    vals, _, types = _balanced(text, m.end() - 1, '(', ')').partition(':')
    return [v.strip() for v in vals.split(',') if v.strip()], split_top(types)


def parse_map(s):
    """'affine_map<(d0, d1)[s0] -> (d0, 0)>' -> (['d0','d1'], ['d0','0'])."""
    m = re.search(r'affine_map<\(([^)]*)\)(?:\[[^\]]*\])?\s*->\s*\((.*)\)>', s)
    if not m:
        return None
    return [d.strip() for d in m.group(1).split(',') if d.strip()], split_top(m.group(2))


def shape(t):
    """'tensor<1x?x8xf32>' -> ['1','?','8']; scalars -> None."""
    m = re.match(r'(?:tensor|memref)<(.*)>', t.strip())
    if not m:
        return None
    return split_top(m.group(1))[0].split('x')[:-1]


@dataclass
class Op:
    tag: str
    name: str
    text: str
    start: int
    end: int
    func: int
    res: str = ''
    nres: int = 1
    ins: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    ins_types: list = field(default_factory=list)
    outs_types: list = field(default_factory=list)
    maps: list = None   # [(dims, results)] per operand, generic only
    iters: list = None  # ['parallel'|'reduction'|...], generic only
    body: list = field(default_factory=list)
    attr_tag: str = None
    producers: list = field(default_factory=list)  # [(producer_tag, operand_idx)] sorted like _parse_ast

    @property
    def kind(self):
        """'ew' (all parallel), 'red' (has reduction), or None (unparsed)."""
        if self.name == 'generic':
            if self.iters is None:
                return None
            return 'red' if any(i != 'parallel' for i in self.iters) else 'ew'
        if self.name in _EW:
            return 'ew'
        if self.name == 'reduce' or _CONTRACTION.match(self.name) or _WINDOW.match(self.name):
            return 'red'
        return None

    @property
    def label(self):
        return f'generic({self.kind or "?"})' if self.name == 'generic' else self.name

    @property
    def n_loops(self):
        if self.iters is not None:
            return len(self.iters)
        ranks = [len(shape(t)) for t in self.ins_types + self.outs_types if shape(t) is not None]
        return max(ranks, default=0)

    def access(self, k):
        """How this op reads its operand k: 'full', 'partial' (broadcast / misses a parallel dim),
        'halo' (compound index like d2+d5), 'init' (outs operand), or None (unknown)."""
        if k >= len(self.ins):
            return 'init'
        if self.maps is not None and self.iters is not None and k < len(self.maps):
            dims, results = self.maps[k]
            if any(not re.fullmatch(r'd\d+|\d+', r) for r in results):
                return 'halo'
            par = {d for d, it in zip(dims, self.iters) if it == 'parallel'}
            used = {r for r in results if r.startswith('d')}
            return 'full' if par <= used and len(used) == len(results) else 'partial'
        if self.name in ('broadcast',) or _CONTRACTION.match(self.name):
            return 'partial'
        if _WINDOW.match(self.name):
            return 'halo' if k == 0 else 'partial'
        if self.name in _EW or self.name == 'reduce':
            return 'full'
        return None


def parse_text(text, tags='ast'):
    """Parse all linalg ops. tags='ast' numbers like the AST dumper; tags='attr' keeps the
    existing `tag` attribute (blocks carry their model tags). Returns (ops, unparsed_names)."""
    lines = text.splitlines()
    aliases = {}
    for ln in lines:
        m = _ALIAS.match(ln.strip())
        if m:
            aliases[m.group(1)] = m.group(2)
    ops, func, i, k = [], -1, 0, 0
    while i < len(lines):
        ln = lines[i]
        if 'func.func' in ln:
            func += 1
        m = _OP_START.match(ln)
        if not m or m.group(3) in ('yield', 'index'):
            i += 1
            continue
        start, depth = i, 0
        while True:
            depth += lines[i].count('{') - lines[i].count('}')
            if depth <= 0 or i == len(lines) - 1:
                break
            i += 1
        body = '\n'.join(lines[start:i + 1])
        op = Op(tag='', name=m.group(3), text=body, start=start, end=i, func=func,
                res=m.group(1) or '', nres=int(m.group(2) or 1))
        op.ins, op.ins_types = _section(body, 'ins')
        op.outs, op.outs_types = _section(body, 'outs')
        t = _TAG.search(body)
        op.attr_tag = t.group(1) if t else None
        if op.name == 'generic':
            mm = re.search(r'indexing_maps\s*=\s*\[', body)
            it = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', body)
            if mm and it:
                items = split_top(_balanced(body, mm.end() - 1, '[', ']'))
                maps = [parse_map(aliases.get(x, x)) for x in items]
                op.maps = None if None in maps else maps
                op.iters = re.findall(r'parallel|reduction|window', it.group(1))
            region = body[body.find('^bb'):] if '^bb' in body else body
            op.body = _BODY.findall(region)
        else:
            op.body = _BODY.findall(body) or ([_EW[op.name]] if _EW.get(op.name) else [])
        i += 1
        if op.n_loops == 0:
            continue  # AST dumper skips zero-loop ops
        op.tag = op.attr_tag if tags == 'attr' and op.attr_tag else f'operation_{k}'
        k += 1
        ops.append(op)
    _link(ops)
    return ops, sorted({o.name for o in ops if o.kind is None})


def _link(ops):
    """Fill op.producers from SSA operands (per function)."""
    by_res = {}
    for op in ops:
        by_res[(op.func, op.res)] = op
    for op in ops:
        for idx, v in enumerate(op.ins + op.outs):
            p = by_res.get((op.func, v.split('#')[0]))
            if p is not None and p is not op:
                op.producers.append((p.tag, idx))
        op.producers.sort(key=lambda p: (p[1], p[0]))


def read_text(path, max_line=10000):
    """Read an MLIR file, truncating long non-linalg lines (weight constants can be GBs on one line)."""
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if len(line) > max_line and 'linalg.' not in line[:max_line]:
                line = line[:max_line] + '...\n'
            out.append(line)
    return ''.join(out)


def parse_file(path, tags='ast'):
    if os.path.getsize(path) > 1e6:
        print(f'parsing {os.path.basename(path)} ({os.path.getsize(path) / 1e9:.2f} GB)', flush=True)
    return parse_text(read_text(path), tags)


def classify(p, c, k):
    """DNNFusion-style edge class for producer p -> consumer c operand k: (class, reason)."""
    acc = c.access(k)
    if acc == 'init':
        return 'yellow', 'init_operand'
    if p.kind is None or acc is None:
        return 'yellow', 'unparsed'
    if p.kind == 'ew':
        return ('green', '') if acc == 'full' else ('yellow', f'ew_producer_consumer_{acc}')
    if acc == 'partial':
        return 'red', 'reduction_then_broadcast'
    return 'yellow', f'reduction_producer_consumer_{acc}'


def edges(ops):
    """[(producer_op, consumer_op, operand_idx)]."""
    by_tag = {o.tag: o for o in ops}
    return [(by_tag[t], c, k) for c in ops for t, k in c.producers]


def model_name(path):
    b = os.path.basename(path)
    return b.replace('_linalg.mlir', '').replace('.mlir', '').split('_block_')[0]


def files(paths, exts=('.mlir',)):
    """Expand files, directories (recursive) and globs."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs if f.endswith(exts)]
        else:
            out += [f for f in glob.glob(p) if f.endswith(exts)]
    if not out:
        raise SystemExit(f'No {"/".join(exts)} files found in: {" ".join(paths)}')
    print(f'{len(set(out))} input files')
    return sorted(set(out))


def load_blocks(paths):
    """Blocks from manifests (*_blocks_manifest.json) or block .mlir files.
    Returns [{model, name, tags, edges}] with edges = {(producer_tag, consumer_tag)} inside the block."""
    import json
    blocks = []
    for f in files(paths, ('.json', '.mlir')):
        if f.endswith('.json'):
            with open(f, encoding='utf-8') as fh:
                man = json.load(fh)
            for e in man.get('entries', []):
                path = e['path_consumer_to_producer']
                blocks.append(dict(model=man['model'], name=os.path.basename(e['output_file'])[:-5],
                                   tags=path, edges={(path[i + 1], path[i]) for i in range(len(path) - 1)}))
        else:
            ops, _ = parse_file(f, tags='attr')
            blocks.append(dict(model=model_name(f), name=os.path.basename(f)[:-5], tags=[o.tag for o in ops],
                               edges={(t, o.tag) for o in ops for t, _ in o.producers}))
    return blocks


def write_csv(out, name, rows):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def md_table(rows, keys=None):
    if not rows:
        return '_(none)_\n'
    keys = keys or list(rows[0])
    fmt = lambda v: f'{v:.3g}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
                     + ['| ' + ' | '.join(fmt(r.get(k, '')) for k in keys) + ' |' for r in rows]) + '\n'


def write_md(out, name, text):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), 'w', encoding='utf-8') as f:
        f.write(text)
