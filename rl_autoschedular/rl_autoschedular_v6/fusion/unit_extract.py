"""M3 — closed-unit builder: one slice of a model -> one runnable benchmark (design doc §4.2).

Given the positions of some top-level ops in the model's public function, moves
those ops into a new `@main`. Unlike the chain-only block builder
(`extract_blocks._build_block_benchmark`), it keeps the DAG as it is:
- "free" values the slice reads (scalars, `tensor.empty`, zero-init fills: any
  op computing a constant without weights) are cloned in;
- every other value it reads from outside becomes a function argument
  (activations, weights);
- **every** value it defines that the rest of the model uses is returned
  (all live-outs);
- ops and edges inside the slice are untouched: real shapes, all producers.
The result goes through v6 `full_model.ensure_timed_main_abi` (timed `@main`).
"""

from rl_autoschedular_v6.fusion.views import parse_module, walk_ops
from rl_autoschedular_v6.full_model import ensure_timed_main_abi

POS_ATTR = 'fusion.pos'


class UnitSkip(RuntimeError):
    """The slice cannot become a valid timed unit (reason in the message)."""


def public_function(module):
    """The single public `func.func` with a body."""
    funcs = [op for op in module.body.operations
             if op.operation.name == 'func.func' and 'sym_visibility' not in op.operation.attributes
             and len(op.regions[0].blocks) > 0]
    if len(funcs) != 1:
        raise UnitSkip(f'expected one public function, found {len(funcs)}')
    return funcs[0]


def is_tensor(value) -> bool:
    return str(value.type).startswith('tensor<')


def stamp_positions(body: list) -> None:
    """Record each top-level op's body position as an attribute (read back by `op_pos`)."""
    from mlir.ir import IntegerAttr, IntegerType

    for p, op in enumerate(body):
        op.attributes[POS_ATTR] = IntegerAttr.get(IntegerType.get_signless(64), p)


def op_pos(op) -> int | None:
    """Body position of the top-level ancestor of `op` (None for the function itself)."""
    from mlir.ir import IntegerAttr

    while op is not None and op.operation.name not in ('func.func', 'builtin.module'):
        if POS_ATTR in op.attributes:
            return IntegerAttr(op.attributes[POS_ATTR]).value
        op = op.parent
    return None


def value_pos(value) -> int | None:
    """Body position of the top-level op defining `value` (None for function arguments)."""
    from mlir.ir import BlockArgument, OpResult

    if OpResult.isinstance(value):
        return op_pos(value.owner)
    return op_pos(BlockArgument(value).owner.owner)


def operand_positions(op) -> set:
    """Positions of the top-level ops defining what `op` (incl. its regions) reads from outside itself."""
    own = op_pos(op)
    return {value_pos(v) for inner in walk_ops(op, lambda _: True) for v in inner.operands} - {own}


def free_positions(body: list) -> set[int]:
    """Top-level ops that compute constants: no function argument, tensor constant (weights
    stay arguments) or call among their transitive inputs. Requires `stamp_positions`."""
    free: set[int] = set()
    for p, op in enumerate(body[:-1]):
        name = op.operation.name
        if name == 'func.call' or (name == 'arith.constant' and is_tensor(op.results[0])):
            continue
        if operand_positions(op) <= free:
            free.add(p)
    return free


def _free_closure(positions: set[int], body: list, free: set[int]) -> set[int]:
    """`positions` plus every free op they depend on."""
    closure, todo = set(), list(positions)
    while todo:
        p = todo.pop()
        if p not in closure:
            closure.add(p)
            todo.extend(q for q in operand_positions(body[p]) if q in free)
    return closure


def extract_unit(code: str, positions: list[int]) -> str:
    """Timed unit made of the public function's top-level ops at `positions` (body order)."""
    from mlir.dialects import func
    from mlir.ir import Context, FunctionType, InsertionPoint, Location, MLIRError, StringAttr

    with Context(), Location.unknown():
        module = parse_module(code)
        main = public_function(module)
        block = main.regions[0].blocks[0]
        body = list(block.operations)
        stamp_positions(body)
        free = free_positions(body)
        inside = set(positions)
        unit_ops = [body[p] for p in sorted(inside)]

        external, args, free_deps = [], [], set()
        for top in unit_ops:
            for op in walk_ops(top, lambda _: True):
                for value in op.operands:
                    p = value_pos(value)
                    if p in inside or value in external:
                        continue
                    external.append(value)
                    if p in free:
                        free_deps.add(p)
                    elif is_tensor(value):
                        args.append(value)
                    else:
                        raise UnitSkip(f'external non-tensor value of type {value.type} is not a constant')
        live_outs = [r for op in unit_ops for r in op.results if any(op_pos(u.owner) not in inside for u in r.uses)]
        if not live_outs or not all(is_tensor(v) for v in live_outs):
            raise UnitSkip('unit must return at least one value, and only tensors')

        with InsertionPoint(module.body):
            new_func = func.FuncOp('fusion_unit_main', FunctionType.get([v.type for v in args],
                                                                        [v.type for v in live_outs]))
        entry = new_func.add_entry_block()
        for op in unit_ops:
            entry.append(op.operation)
        clones = [body[p] for p in sorted(_free_closure(free_deps, body, free))]
        with InsertionPoint(unit_ops[0].operation):
            copies = [op.operation.clone() for op in clones]
        for original, copy in zip(clones, copies):
            for old, new in zip(original.results, copy.results):
                old.replace_all_uses_with(new)
        for value, arg in zip(args, entry.arguments):
            value.replace_all_uses_with(arg)
        with InsertionPoint(entry):
            func.ReturnOp(live_outs)

        # Old function: erase users before producers, then the function itself.
        for op in reversed(list(block.operations)):
            op.operation.erase()
        main.operation.erase()
        new_func.attributes['sym_name'] = StringAttr.get('main')
        for op in walk_ops(module, lambda o: POS_ATTR in o.attributes):
            del op.attributes[POS_ATTR]
        try:
            module.operation.verify()
        except MLIRError as e:
            raise UnitSkip(f'extracted unit does not verify: {e}') from e
        unit = str(module)

    timed = ensure_timed_main_abi(unit)
    if 'call @nanoTime' not in timed:
        raise UnitSkip('ensure_timed_main_abi could not add the timed @main')
    return timed
