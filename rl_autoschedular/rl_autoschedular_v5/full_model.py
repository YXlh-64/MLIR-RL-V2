"""Full-model ABI adapter for timed execution.

Block benchmarks already conform to the timed ABI expected by
``Execution.execute_code``: a public ``@main`` returning
``(tensor..., i64)`` where the trailing ``i64`` is a ``@nanoTime()``
delta. Full-model files (``data/mlir_rl_v1_paper/model_*.mlir``) do not:
they expose a raw tensor entry (``@main``/``@resnet50``/``@vgg16``) with
no timing wrapper, so execution fails and eval records ``null``.

``ensure_timed_main_abi`` rewrites such code to the timed ABI. It is a
no-op for already-conforming code, so applying it to every loaded
benchmark is safe for block benchmarks.
"""

import re


def materialize_elided_resources(code: str) -> str:
    """Replace ``dense_resource<__elided__>`` constants with zero splats.

    ResNet/VGG full-model files were exported with elided large constants,
    so no weight data exists in the file and JIT fails with
    "resource does not exist". Constant *values* do not affect execution
    *time* (identical shapes, dtypes, and FLOPs), and eval only consumes
    the ``@nanoTime()`` delta — never the outputs. No-op when the module
    has no elided resources (all block benchmarks, MobileNet).

    The zero splat matches the element type (``dense<0>`` for integers /
    index / booleans, ``dense<0.0>`` for floats) — a float splat on an
    integer tensor fails to parse (seen in gpt2's i1 mask).
    """
    if '__elided__' not in code:
        return code
    return _ELIDED_RE.sub(_elided_repl, code)


# dense_resource<__elided__> : tensor<<shape>>
_ELIDED_RE = re.compile(r'dense_resource<__elided__>\s*:\s*tensor<([^>]+)>')


def _elided_repl(match: re.Match) -> str:
    shape = match.group(1)
    etype = shape.split('x')[-1]
    zero = '0' if (etype.startswith('i') or etype == 'index') else '0.0'
    return f'dense<{zero}> : tensor<{shape}>'

# func.func @name(args) -> ret [attributes {...}] {
_ENTRY_RE = re.compile(
    r'func\.func\s+@(\w+)\s*\(([^)]*)\)\s*->\s*'
    r'(\([^)]*\)|[^\s{]+?)(\s*attributes\s*\{[^{}]*\})?\s*\{'
)


def _is_timed_main(code: str) -> bool:
    """True if code already defines a timed @main (tensor..., i64)."""
    if '@nanoTime' not in code:
        return False
    return bool(
        re.search(
            r'func\.func\s+@main\s*\([^)]*\)\s*->\s*\([^)]*i64[^)]*\)',
            code,
        )
    )


def ensure_timed_main_abi(code: str) -> str:
    """Rewrite a raw-tensor full-model entry to the timed ``@main`` ABI.

    Returns the input unchanged when it already conforms or when the
    module does not have exactly one unambiguous public tensor entry.
    """
    if _is_timed_main(code):
        return code

    # Exactly one public (non-private) function must exist.
    public = [
        m for m in _ENTRY_RE.finditer(code)
        if not m.group(0).startswith('func.func private')
        and 'private' not in (m.group(4) or '')
    ]
    if len(public) != 1:
        return code
    entry = public[0]
    name, args_str, ret_str = entry.group(1), entry.group(2).strip(), entry.group(3).strip()

    # Entry args must all be ranked tensors (weights-as-args are fine).
    args = [a.strip() for a in args_str.split(',') if a.strip()]
    if not args or not all(re.fullmatch(r'%\w+\s*:\s*tensor<[^>]+>', a) for a in args):
        return code

    # Single or multi tensor return (e.g. albert returns hidden + pooled).
    # Execution._create_params already handles N memrefs + trailing i64.
    if ret_str.startswith('('):
        if not ret_str.endswith(')'):
            return code
        ret_str = ret_str[1:-1].strip()
    ret_types = [r.strip() for r in ret_str.split(',')]
    if not ret_types or not all(re.fullmatch(r'tensor<[^>]+>', r) for r in ret_types):
        return code

    # Renaming must be unambiguous: the symbol occurs only at its definition.
    if name != 'main':
        if code.count('@' + name) != 1:
            return code
        code = code[:entry.start()] + code[entry.start():].replace(
            f'func.func @{name}(', 'func.func @main(', 1
        )
        entry = _ENTRY_RE.search(code, entry.start())
        if entry is None:
            return code
        args_str = entry.group(2).strip()

    # Locate the entry body via brace matching.
    body_start = entry.end()
    depth, pos = 1, body_start
    while depth > 0 and pos < len(code):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
        pos += 1
    body, pre, post = code[body_start:pos - 1], code[:entry.start()], code[pos:]

    # Find the final tensor return.
    lines = body.split('\n')
    ret_idx = next(
        (i for i in range(len(lines) - 1, -1, -1)
         if lines[i].strip().startswith('return ') and ':' in lines[i]),
        None,
    )
    if ret_idx is None:
        return code
    ret_vals = lines[ret_idx].strip()[len('return '):].rsplit(':', 1)[0].strip()

    # Collision-free SSA names for the timer.
    t0, t1, dl = '%t0', '%t1', '%delta'
    if any(v in body for v in (t0, t1, dl)):
        t0, t1, dl = '%_t0', '%_t1', '%_delta'
        if any(v in body for v in (t0, t1, dl)):
            return code

    timed_body = (
        f'\n    {t0} = call @nanoTime() : () -> i64\n'
        + '\n'.join(lines[:ret_idx])
        + f'\n    {t1} = call @nanoTime() : () -> i64'
        + f'\n    {dl} = arith.subi {t1}, {t0} : i64'
        + f'\n    return {ret_vals}, {dl} : {ret_str}, i64'
    )
    attrs = entry.group(4) or ''
    if 'llvm.emit_c_interface' not in attrs:
        attrs = ' attributes {llvm.emit_c_interface}'
    new_main = f'func.func @main({args_str}) -> ({ret_str}, i64){attrs} {{{timed_body}\n}}'

    if '@nanoTime' not in pre:
        nano_decl = '  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}\n'
        pre = re.sub(r'(module\s*\{)', r'\1\n' + nano_decl, pre, count=1)

    return pre + new_main + post
