// Diamond: two elementwise producers feeding one elementwise consumer.
// operation_0 = exp(a), operation_1 = a * b, operation_2 = op0 + op1.
// Expected: op0->op2 green, op1->op2 green.
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%a: tensor<64x128xf32>, %b: tensor<64x128xf32>) -> (tensor<64x128xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %e = tensor.empty() : tensor<64x128xf32>
    %p1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%x: f32, %o: f32):
      %r = math.exp %x : f32
      linalg.yield %r : f32
    } -> tensor<64x128xf32>
    %p2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %r = arith.mulf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x128xf32>
    %c = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%p1, %p2 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%e : tensor<64x128xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %r = arith.addf %x, %y : f32
      linalg.yield %r : f32
    } -> tensor<64x128xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %c, %delta : tensor<64x128xf32>, i64
  }
}
