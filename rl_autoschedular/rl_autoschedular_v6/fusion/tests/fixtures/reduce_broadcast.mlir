// Softmax-like: reductions whose results are broadcast back (red edges).
// operation_0 = fill(-inf)                   (init)
// operation_1 = rowmax(x), outs = op0        (reduction)
// operation_2 = x - op1[d0]                  (broadcast)
// operation_3 = exp(op2)                     (one_to_one, two consumers)
// operation_4 = fill(0)                      (init)
// operation_5 = rowsum(op3), outs = op4      (reduction)
// operation_6 = op3 / op5[d0]                (broadcast)
// Expected: op1->op2 red, op5->op6 red, op2->op3 green, op3->op5 green, op3->op6 green,
//           op0->op1 init, op4->op5 init.
#id = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d0)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%x: tensor<16x256xf32>) -> (tensor<16x256xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %e1 = tensor.empty() : tensor<16xf32>
    %e2 = tensor.empty() : tensor<16x256xf32>
    %init_max = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
    %max = linalg.generic {indexing_maps = [#id, #col], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<16x256xf32>) outs(%init_max : tensor<16xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %r = arith.maximumf %v, %acc : f32
      linalg.yield %r : f32
    } -> tensor<16xf32>
    %sub = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %max : tensor<16x256xf32>, tensor<16xf32>) outs(%e2 : tensor<16x256xf32>) {
    ^bb0(%v: f32, %m: f32, %o: f32):
      %r = arith.subf %v, %m : f32
      linalg.yield %r : f32
    } -> tensor<16x256xf32>
    %exp = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%sub : tensor<16x256xf32>) outs(%e2 : tensor<16x256xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = math.exp %v : f32
      linalg.yield %r : f32
    } -> tensor<16x256xf32>
    %init_sum = linalg.fill ins(%zero : f32) outs(%e1 : tensor<16xf32>) -> tensor<16xf32>
    %sum = linalg.generic {indexing_maps = [#id, #col], iterator_types = ["parallel", "reduction"]} ins(%exp : tensor<16x256xf32>) outs(%init_sum : tensor<16xf32>) {
    ^bb0(%v: f32, %acc: f32):
      %r = arith.addf %v, %acc : f32
      linalg.yield %r : f32
    } -> tensor<16xf32>
    %div = linalg.generic {indexing_maps = [#id, #col, #id], iterator_types = ["parallel", "parallel"]} ins(%exp, %sum : tensor<16x256xf32>, tensor<16xf32>) outs(%e2 : tensor<16x256xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.divf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<16x256xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %div, %delta : tensor<16x256xf32>, i64
  }
}
