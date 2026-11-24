"""rules.py on hand-built views: op types, accesses and every edge class."""

from rl_autoschedular_v6.fusion.rules import (
    OpType, OpView, access, classify_edge, normalize_expr, op_type, with_init_override,
)

P2 = ('parallel', 'parallel')


def view(iterators, inputs, inits, name='linalg.generic'):
    return OpView(name, len(inputs), len(inits), tuple(iterators),
                  tuple(tuple(m) for m in inputs), tuple(tuple(m) for m in inits))


EW = view(P2, [('d0', 'd1')], [('d0', 'd1')])
BIAS = view(P2, [('d0', 'd1'), ('d1',)], [('d0', 'd1')])
CONST_IDX = view(P2, [('d0', '0')], [('d0', 'd1')])
TRANSPOSE = view(P2, [('d1', 'd0')], [('d0', 'd1')])
MATMUL = view(('parallel', 'parallel', 'reduction'), [('d0', 'd2'), ('d2', 'd1')], [('d0', 'd1')])
ROWSUM = view(('parallel', 'reduction'), [('d0', 'd1')], [('d0',)])
CONV = view(('parallel',) * 4 + ('reduction',) * 3,
            [('d0', 'd4', 'd1 + d5', 'd2 + d6'), ('d3', 'd4', 'd5', 'd6')], [('d0', 'd3', 'd1', 'd2')])
UNKNOWN = OpView('linalg.matmul', 2, 1, None, None, None)


def test_op_types():
    assert op_type(EW) == OpType.ONE_TO_ONE
    assert op_type(BIAS) == OpType.BROADCAST
    assert op_type(CONST_IDX) == OpType.BROADCAST
    assert op_type(TRANSPOSE) == OpType.SHUFFLE
    assert op_type(MATMUL) == OpType.REDUCTION
    assert op_type(CONV) == OpType.WINDOW
    assert op_type(UNKNOWN) == OpType.UNKNOWN


def test_init_override_needs_all_consumers_init():
    assert with_init_override(OpType.BROADCAST, ['init', 'init']) == OpType.INIT
    assert with_init_override(OpType.BROADCAST, ['init', 'full']) == OpType.BROADCAST
    assert with_init_override(OpType.ONE_TO_ONE, []) == OpType.ONE_TO_ONE


def test_access():
    assert access(EW, 0) == 'full'
    assert access(EW, 1) == 'init'
    assert access(BIAS, 1) == 'partial'
    assert access(CONST_IDX, 0) == 'partial'
    assert access(TRANSPOSE, 0) == 'full'
    assert access(MATMUL, 0) == 'partial'       # LHS misses the output-column loop
    assert access(ROWSUM, 0) == 'full'          # covers the only parallel loop
    assert access(CONV, 0) == 'halo'
    assert access(UNKNOWN, 0) is None
    assert access(UNKNOWN, 2) == 'init'         # init is known from operand counts alone
    assert access(EW, 5) is None


def test_edge_classes():
    T = OpType
    assert classify_edge(T.ONE_TO_ONE, 'full') == 'green'
    assert classify_edge(T.BROADCAST, 'full') == 'green'
    assert classify_edge(T.ONE_TO_ONE, 'init') == 'init'
    assert classify_edge(T.REDUCTION, 'partial') == 'red'
    assert classify_edge(T.WINDOW, 'partial') == 'red'
    assert classify_edge(T.REDUCTION, 'full') == 'yellow:epilogue'
    assert classify_edge(T.WINDOW, 'halo') == 'yellow:halo'
    assert classify_edge(T.ONE_TO_ONE, 'partial') == 'yellow:partial'
    assert classify_edge(T.BROADCAST, 'partial') == 'yellow:partial'   # broadcast into a contraction
    assert classify_edge(T.ONE_TO_ONE, 'halo') == 'yellow:halo'
    assert classify_edge(T.SHUFFLE, 'full') == 'yellow:shuffle'
    assert classify_edge(T.UNKNOWN, 'full') == 'yellow:unparsed'
    assert classify_edge(T.ONE_TO_ONE, None) == 'yellow:unparsed'


def test_normalize_expr():
    assert normalize_expr('%d1  + %d4') == 'd1 + d4'
