// Transpose between two elementwise ops.
// operation_0 = relu(x)                 (one_to_one)
// operation_1 = transpose(op0) [1, 0]   (shuffle, named op)
// operation_2 = exp(op1)                (one_to_one)
// Expected: op0->op1 green (the transpose reads op0 with a full permutation),
//           op1->op2 yellow:shuffle.
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%x: tensor<32x64xf32>) -> (tensor<64x32xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %cst = arith.constant 0.000000e+00 : f32
    %e1 = tensor.empty() : tensor<32x64xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x64xf32>) outs(%e1 : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %e2 = tensor.empty() : tensor<64x32xf32>
    %tr = linalg.transpose ins(%relu : tensor<32x64xf32>) outs(%e2 : tensor<64x32xf32>) permutation = [1, 0]
    %exp = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%tr : tensor<64x32xf32>) outs(%e2 : tensor<64x32xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = math.exp %v : f32
      linalg.yield %r : f32
    } -> tensor<64x32xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %exp, %delta : tensor<64x32xf32>, i64
  }
}
