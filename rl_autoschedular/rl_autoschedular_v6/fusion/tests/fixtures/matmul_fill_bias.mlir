// Matmul with a fill init, an elementwise producer on its LHS and a bias epilogue.
// operation_0 = relu(a)           (one_to_one)
// operation_1 = fill(0)           (init)
// operation_2 = matmul(op0, w)    (reduction), outs = op1
// operation_3 = op2 + bias[d1]    (broadcast)
// Expected: op0->op2 yellow:partial, op1->op2 init, op2->op3 yellow:epilogue.
// AstDumper order gives op2.producers = [op0, op1], so producers[-1] is the fill.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%a: tensor<32x64xf32>, %w: tensor<64x128xf32>, %bias: tensor<128xf32>) -> (tensor<32x128xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %cst = arith.constant 0.000000e+00 : f32
    %ea = tensor.empty() : tensor<32x64xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<32x64xf32>) outs(%ea : tensor<32x64xf32>) {
    ^bb0(%x: f32, %o: f32):
      %r = arith.maximumf %x, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %e = tensor.empty() : tensor<32x128xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%e : tensor<32x128xf32>) -> tensor<32x128xf32>
    %mm = linalg.matmul ins(%relu, %w : tensor<32x64xf32>, tensor<64x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
    %out = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%mm, %bias : tensor<32x128xf32>, tensor<128xf32>) outs(%e : tensor<32x128xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %r = arith.addf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<32x128xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %out, %delta : tensor<32x128xf32>, i64
  }
}
