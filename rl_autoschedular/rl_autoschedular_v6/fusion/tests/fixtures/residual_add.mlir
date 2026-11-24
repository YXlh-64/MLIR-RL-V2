// Two identical residual blocks: y = relu(x @ W) + x.
// Block 1: operation_0 fill, operation_1 matmul, operation_2 relu, operation_3 add
// Block 2: operation_4 fill, operation_5 matmul, operation_6 relu, operation_7 add
// Expected: fill->matmul init, matmul->relu yellow:epilogue, relu->add green,
//           op3->op5 yellow:partial (matmul LHS), op3->op7 green (residual).
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%x: tensor<32x64xf32>, %w1: tensor<64x64xf32>, %w2: tensor<64x64xf32>) -> (tensor<32x64xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<32x64xf32>
    %f1 = linalg.fill ins(%cst : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %m1 = linalg.matmul ins(%x, %w1 : tensor<32x64xf32>, tensor<64x64xf32>) outs(%f1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%m1 : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %y1 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%r1, %x : tensor<32x64xf32>, tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.addf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %f2 = linalg.fill ins(%cst : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %m2 = linalg.matmul ins(%y1, %w2 : tensor<32x64xf32>, tensor<64x64xf32>) outs(%f2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r2 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%m2 : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %y2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%r2, %y1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.addf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %y2, %delta : tensor<32x64xf32>, i64
  }
}
