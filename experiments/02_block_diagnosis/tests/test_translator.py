#!/usr/bin/env python3
"""Unit tests for block_to_torch.py. Needs torch + numpy only (no MLIR build).

Each test writes a tiny block in the dataset's format, translates it, executes
the Torch graph AND the generated torch_block.py, and compares both with an
independent NumPy/Torch reference written by hand from the MLIR semantics.

Run:  python experiments/02_block_diagnosis/tests/test_translator.py
(pytest also collects the test_* functions if you prefer it.)
"""
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402

import block_to_torch as b2t            # noqa: E402

RNG = np.random.default_rng(0)


def block(maps, args, body, ret_type, ret, consts=''):
    sig = ', '.join(f'%{n}: {t}' for n, t in args)
    return f'''{maps}
module {{
  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
  func.func @main({sig}) -> ({ret_type}, i64) attributes {{llvm.emit_c_interface}} {{
{consts}    %t0 = call @nanoTime() : () -> i64
{body}
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return {ret}, %delta : {ret_type}, i64
  }}
}}
'''


def translate(text):
    t = b2t.translate_text('test_block_0', text)
    t['status'] = 'ok'
    return t


def inputs_for(t):
    arrays = []
    for s in t['inputs']:
        if s.get('generator') == 'uniform':
            a = RNG.uniform(s['low'], s['high'], s['shape'])
        else:
            a = RNG.normal(0, 0.5, s['shape'])
        arrays.append(np.asarray(a, dtype=s['dtype']))
    return arrays


def run_graph(t, arrays):
    ops = b2t.torch_operations(torch, F)
    values = {s['name']: torch.from_numpy(a.copy()) for s, a in zip(t['inputs'], arrays)}
    for node in t['graph']:
        kwargs = dict(node['kwargs'])
        if isinstance(kwargs.get('dtype'), str):
            kwargs['dtype'] = getattr(torch, kwargs['dtype'])
        values[node['id']] = ops[node['op']](*[values[a] for a in node['args']], **kwargs)
    return [values[o].numpy() for o in t['outputs']]


def run_generated(t, arrays):
    namespace = {}
    exec(b2t.render_python('test_block_0', t), namespace)
    out = namespace['Block']().forward(*[torch.from_numpy(a.copy()) for a in arrays])
    return [o.numpy() for o in out]


def check(text, reference, expect_positive=()):
    t = translate(text)
    arrays = inputs_for(t)
    named = {s['name']: a for s, a in zip(t['inputs'], arrays)}
    want = reference(named)
    for got in (run_graph(t, arrays), run_generated(t, arrays)):
        assert len(got) == 1, 'one output expected'
        assert got[0].shape == want.shape, (got[0].shape, want.shape)
        np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-5)
    for name in expect_positive:
        spec = next(s for s in t['inputs'] if s['name'] == name)
        assert spec['generator'] == 'uniform' and spec['low'] > 0, spec
    return t


ID3 = 'affine_map<(d0, d1, d2) -> (d0, d1, d2)>'


def test_elementwise_then_row_sum():
    # The albert_block_0 pattern: elementwise ops, then sum over the last dim into (d0, d1, 0).
    maps = f'#id = {ID3}\n#red = affine_map<(d0, d1, d2) -> (d0, d1, 0)>'
    body = '''    %v0 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg_0_in_0, %arg_0_in_1 : tensor<2x4x8xf32>, tensor<2x4x8xf32>) outs(%arg_0_out_0 : tensor<2x4x8xf32>) attrs =  {tag = "operation_1"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.subf %in, %in_1 : f32
      %1 = arith.mulf %0, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x4x8xf32>
    %v1 = linalg.generic {indexing_maps = [#id, #red], iterator_types = ["parallel", "parallel", "reduction"]} ins(%v0 : tensor<2x4x8xf32>) outs(%arg_1_out_0 : tensor<2x4x1xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<2x4x1xf32>'''
    args = [('arg_0_in_0', 'tensor<2x4x8xf32>'), ('arg_0_in_1', 'tensor<2x4x8xf32>'),
            ('arg_0_out_0', 'tensor<2x4x8xf32>'), ('arg_1_out_0', 'tensor<2x4x1xf32>')]
    check(block(maps, args, body, 'tensor<2x4x1xf32>', '%v1'),
          lambda a: a['arg_1_out_0'] + ((a['arg_0_in_0'] - a['arg_0_in_1']) ** 2).sum(axis=2, keepdims=True))


def test_batch_matmul_accumulates_into_outs():
    maps = f'#id = {ID3}'
    body = '''    %v0 = linalg.batch_matmul {tag = "operation_1"} ins(%arg_0_in_0, %arg_0_in_1 : tensor<2x3x4xf32>, tensor<2x4x5xf32>) outs(%arg_0_out_0 : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    %v1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%v0 : tensor<2x3x5xf32>) outs(%arg_1_out_0 : tensor<2x3x5xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %0 = math.tanh %in : f32
      linalg.yield %0 : f32
    } -> tensor<2x3x5xf32>'''
    args = [('arg_0_in_0', 'tensor<2x3x4xf32>'), ('arg_0_in_1', 'tensor<2x4x5xf32>'),
            ('arg_0_out_0', 'tensor<2x3x5xf32>'), ('arg_1_out_0', 'tensor<2x3x5xf32>')]
    check(block(maps, args, body, 'tensor<2x3x5xf32>', '%v1'),
          lambda a: np.tanh(a['arg_0_out_0'] + a['arg_0_in_0'] @ a['arg_0_in_1']))


def test_conv_then_channel_bias():
    # The resnet50_block_0 pattern: strided conv + per-channel bias broadcast via (d1, 0, 0).
    maps = ('#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>\n'
            '#ch = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>')
    body = '''    %v0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins(%arg_0_in_0, %arg_0_in_1 : tensor<1x3x9x9xf32>, tensor<4x3x3x3xf32>) outs(%arg_0_out_0 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    %v1 = linalg.generic {indexing_maps = [#id4, #ch, #id4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%v0, %arg_1_in_1 : tensor<1x4x4x4xf32>, tensor<4x1x1xf32>) outs(%arg_1_out_0 : tensor<1x4x4x4xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      linalg.yield %0 : f32
    } -> tensor<1x4x4x4xf32>'''
    args = [('arg_0_in_0', 'tensor<1x3x9x9xf32>'), ('arg_0_in_1', 'tensor<4x3x3x3xf32>'),
            ('arg_0_out_0', 'tensor<1x4x4x4xf32>'), ('arg_1_in_1', 'tensor<4x1x1xf32>'),
            ('arg_1_out_0', 'tensor<1x4x4x4xf32>')]

    def ref(a):
        conv = F.conv2d(torch.from_numpy(a['arg_0_in_0']), torch.from_numpy(a['arg_0_in_1']), stride=2).numpy()
        return a['arg_0_out_0'] + conv + a['arg_1_in_1'].reshape(1, 4, 1, 1)
    check(block(maps, args, body, 'tensor<1x4x4x4xf32>', '%v1'), ref)


def test_depthwise_conv():
    body = '''    %v0 = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_1"} ins(%arg_0_in_0, %arg_0_in_1 : tensor<1x3x6x6xf32>, tensor<3x3x3xf32>) outs(%arg_0_out_0 : tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>'''
    args = [('arg_0_in_0', 'tensor<1x3x6x6xf32>'), ('arg_0_in_1', 'tensor<3x3x3xf32>'),
            ('arg_0_out_0', 'tensor<1x3x4x4xf32>')]

    def ref(a):
        x, w = a['arg_0_in_0'], a['arg_0_in_1']
        out = a['arg_0_out_0'].copy()
        for c in range(3):
            for i in range(4):
                for j in range(4):
                    out[0, c, i, j] += (x[0, c, i:i + 3, j:j + 3] * w[c]).sum()
        return out
    check(block('', args, body, 'tensor<1x3x4x4xf32>', '%v0'), ref)


def test_max_reductions_select_and_maxnumf_with_neg_inf_fill():
    maps = ('#id2 = affine_map<(d0, d1) -> (d0, d1)>\n'
            '#row = affine_map<(d0, d1) -> (d0)>')
    consts = '    %cst = arith.constant 0xFF800000 : f32\n'
    body = '''    %v0 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%arg_0_out_0 : tensor<3xf32>) -> tensor<3xf32>
    %v1 = linalg.generic {indexing_maps = [#id2, #row], iterator_types = ["parallel", "reduction"]} ins(%arg_1_in_0 : tensor<3x7xf32>) outs(%v0 : tensor<3xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ogt, %in, %out : f32
      %1 = arith.select %0, %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<3xf32>
    %v2 = linalg.generic {indexing_maps = [#id2, #row], iterator_types = ["parallel", "reduction"]} ins(%arg_2_in_0 : tensor<3x7xf32>) outs(%v1 : tensor<3xf32>) attrs =  {tag = "operation_3"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxnumf %out, %in : f32
      linalg.yield %0 : f32
    } -> tensor<3xf32>'''
    args = [('arg_0_out_0', 'tensor<3xf32>'), ('arg_1_in_0', 'tensor<3x7xf32>'), ('arg_2_in_0', 'tensor<3x7xf32>')]
    check(block(maps, args, body, 'tensor<3xf32>', '%v2', consts),
          lambda a: np.maximum(a['arg_1_in_0'].max(axis=1), a['arg_2_in_0'].max(axis=1)))


def test_transpose_broadcast_and_loop_index():
    maps = '#id2 = affine_map<(d0, d1) -> (d0, d1)>'
    body = '''    %v0 = linalg.transpose ins(%arg_0_in_0 : tensor<2x3xf32>) outs(%arg_0_out_0 : tensor<3x2xf32>) permutation = [1, 0]  {tag = "operation_1"}
    %v1 = linalg.broadcast ins(%arg_1_in_0 : tensor<2xf32>) outs(%arg_1_out_0 : tensor<3x2xf32>) dimensions = [0]  {tag = "operation_2"}
    %v2 = linalg.generic {indexing_maps = [#id2, #id2, #id2], iterator_types = ["parallel", "parallel"]} ins(%v0, %v1 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%arg_2_out_0 : tensor<3x2xf32>) attrs =  {tag = "operation_3"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = linalg.index 0 : index
      %1 = arith.index_cast %0 : index to i64
      %2 = arith.sitofp %1 : i64 to f32
      %3 = arith.addf %in, %in_1 : f32
      %4 = arith.addf %3, %2 : f32
      linalg.yield %4 : f32
    } -> tensor<3x2xf32>'''
    args = [('arg_0_in_0', 'tensor<2x3xf32>'), ('arg_0_out_0', 'tensor<3x2xf32>'), ('arg_1_in_0', 'tensor<2xf32>'),
            ('arg_1_out_0', 'tensor<3x2xf32>'), ('arg_2_out_0', 'tensor<3x2xf32>')]
    check(block(maps, args, body, 'tensor<3x2xf32>', '%v2'),
          lambda a: a['arg_0_in_0'].T + a['arg_1_in_0'][None, :] + np.arange(3, dtype=np.float32)[:, None])


def test_positive_domain_for_rsqrt_input():
    maps = '#id2 = affine_map<(d0, d1) -> (d0, d1)>'
    body = '''    %v0 = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]} ins(%arg_0_in_0 : tensor<4x5xf32>) outs(%arg_0_out_0 : tensor<4x5xf32>) attrs =  {tag = "operation_1"} {
    ^bb0(%in: f32, %out: f32):
      %0 = math.rsqrt %in : f32
      linalg.yield %0 : f32
    } -> tensor<4x5xf32>'''
    args = [('arg_0_in_0', 'tensor<4x5xf32>'), ('arg_0_out_0', 'tensor<4x5xf32>')]
    check(block(maps, args, body, 'tensor<4x5xf32>', '%v0'),
          lambda a: 1 / np.sqrt(a['arg_0_in_0']), expect_positive=('arg_0_in_0',))


def test_positive_domain_traced_through_layernorm_tail():
    # albert_block_1002 pattern: rsqrt(sum(x) / 768 + eps). x is only indirectly under
    # rsqrt, so the domain must be traced back through divf/addf/sum to x and the sum's outs.
    maps = f'#id = {ID3}\n#red = affine_map<(d0, d1, d2) -> (d0, d1, 0)>'
    body = '''    %v0 = linalg.generic {indexing_maps = [#id, #red], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg_0_in_0 : tensor<2x4x8xf32>) outs(%arg_0_out_0 : tensor<2x4x1xf32>) attrs =  {tag = "operation_1"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<2x4x1xf32>
    %v1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%v0 : tensor<2x4x1xf32>) outs(%arg_1_out_0 : tensor<2x4x1xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.divf %in, %cst_n : f32
      %1 = arith.addf %0, %cst_eps : f32
      %2 = math.rsqrt %1 : f32
      linalg.yield %2 : f32
    } -> tensor<2x4x1xf32>'''
    consts = ('    %cst_n = arith.constant 8.000000e+00 : f32\n'
              '    %cst_eps = arith.constant 9.99999996E-13 : f32\n')
    args = [('arg_0_in_0', 'tensor<2x4x8xf32>'), ('arg_0_out_0', 'tensor<2x4x1xf32>'),
            ('arg_1_out_0', 'tensor<2x4x1xf32>')]
    check(block(maps, args, body, 'tensor<2x4x1xf32>', '%v1', consts),
          lambda a: 1 / np.sqrt((a['arg_0_out_0'] + a['arg_0_in_0'].sum(axis=2, keepdims=True)) / 8 + 1e-12),
          expect_positive=('arg_0_in_0', 'arg_0_out_0'))


def test_scalar_powf_exponent_input_is_integer():
    # albert_block_1015 pattern (GELU x**3): the exponent is a 0-d input. A random
    # fractional exponent on a negative base is NaN; it must be drawn as a small integer.
    maps = f'#id = {ID3}\n#s = affine_map<(d0, d1, d2) -> ()>'
    body = '''    %v0 = linalg.generic {indexing_maps = [#id, #s, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg_0_in_0, %arg_0_in_1 : tensor<2x4x8xf32>, tensor<f32>) outs(%arg_0_out_0 : tensor<2x4x8xf32>) attrs =  {tag = "operation_1"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %0 = math.powf %in, %in_1 : f32
      linalg.yield %0 : f32
    } -> tensor<2x4x8xf32>'''
    args = [('arg_0_in_0', 'tensor<2x4x8xf32>'), ('arg_0_in_1', 'tensor<f32>'),
            ('arg_0_out_0', 'tensor<2x4x8xf32>')]
    t = translate(block(maps, args, body, 'tensor<2x4x8xf32>', '%v0'))
    spec = {s['name']: s for s in t['inputs']}
    assert spec['arg_0_in_1']['generator'] == 'integers' and spec['arg_0_in_1']['low'] >= 2, spec
    assert spec['arg_0_in_0']['generator'] == 'normal', 'base may stay signed with an integer exponent'
    a = np.asarray(RNG.normal(0, 0.5, (2, 4, 8)), dtype='float32')
    e = np.asarray(3, dtype='float32')
    o = np.zeros((2, 4, 8), dtype='float32')
    for got in (run_graph(t, [a, e, o]), run_generated(t, [a, e, o])):
        assert np.isfinite(got[0]).all()
        np.testing.assert_allclose(got[0], a ** 3, rtol=1e-5, atol=1e-6)


def test_pooling_sum_accumulates_window_sums():
    # convnext_tiny_block_369 pattern (global 7x7 sum, stride 7) plus a strided 3x3 case.
    for (h, k, s, o) in ((7, 7, 7, 1), (9, 3, 2, 4)):
        body = f'''    %v0 = linalg.pooling_nchw_sum {{dilations = dense<1> : vector<2xi64>, strides = dense<{s}> : vector<2xi64>, tag = "operation_1"}} ins(%arg_0_in_0, %arg_0_in_1 : tensor<1x3x{h}x{h}xf32>, tensor<{k}x{k}xf32>) outs(%arg_0_out_0 : tensor<1x3x{o}x{o}xf32>) -> tensor<1x3x{o}x{o}xf32>'''
        args = [('arg_0_in_0', f'tensor<1x3x{h}x{h}xf32>'), ('arg_0_in_1', f'tensor<{k}x{k}xf32>'),
                ('arg_0_out_0', f'tensor<1x3x{o}x{o}xf32>')]

        def want(a, k=k, s=s, o=o):
            x = a['arg_0_in_0']
            r = a['arg_0_out_0'].copy()
            for i in range(o):
                for j in range(o):
                    r[:, :, i, j] += x[:, :, i * s:i * s + k, j * s:j * s + k].sum(axis=(2, 3))
            return r
        check(block('', args, body, f'tensor<1x3x{o}x{o}xf32>', '%v0'), want)


def test_non_projected_map_is_reported_unsupported():
    maps = ('#a = affine_map<(d0, d1) -> (d0 + d1)>\n'
            '#o = affine_map<(d0, d1) -> (d0)>')
    body = '''    %v0 = linalg.generic {indexing_maps = [#a, #o], iterator_types = ["parallel", "reduction"]} ins(%arg_0_in_0 : tensor<8xf32>) outs(%arg_0_out_0 : tensor<4xf32>) attrs =  {tag = "operation_1"} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<4xf32>'''
    args = [('arg_0_in_0', 'tensor<8xf32>'), ('arg_0_out_0', 'tensor<4xf32>')]
    try:
        b2t.translate_text('x', block(maps, args, body, 'tensor<4xf32>', '%v0'))
    except b2t.Unsupported as error:
        assert 'affine map' in str(error) or 'loop size' in str(error), error
        return
    raise AssertionError('expected Unsupported')


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception:
            failed += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
