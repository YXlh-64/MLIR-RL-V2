// Full-model style (raw tensor entry, weights as constants): stem + 3 identical residual blocks.
// stem: h0 = relu(x)
// block k: f = fill(0); m = h @ Wk (outs f); r = relu(m); h' = r + h
// Expected units (min_ops=1): [stem] and one block unit with weight 3.
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%x: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %w1 = arith.constant dense<5.000000e-01> : tensor<64x64xf32>
    %w2 = arith.constant dense<2.500000e-01> : tensor<64x64xf32>
    %w3 = arith.constant dense<1.250000e-01> : tensor<64x64xf32>
    %e = tensor.empty() : tensor<32x64xf32>
    %h0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %f1 = linalg.fill ins(%cst : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %m1 = linalg.matmul ins(%h0, %w1 : tensor<32x64xf32>, tensor<64x64xf32>) outs(%f1 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%m1 : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %h1 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%r1, %h0 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.addf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %f2 = linalg.fill ins(%cst : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %m2 = linalg.matmul ins(%h1, %w2 : tensor<32x64xf32>, tensor<64x64xf32>) outs(%f2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r2 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%m2 : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %h2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%r2, %h1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.addf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %f3 = linalg.fill ins(%cst : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %m3 = linalg.matmul ins(%h2, %w3 : tensor<32x64xf32>, tensor<64x64xf32>) outs(%f3 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r3 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%m3 : tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %o: f32):
      %r = arith.maximumf %v, %cst : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    %h3 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%r3, %h2 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%e : tensor<32x64xf32>) {
    ^bb0(%v: f32, %s: f32, %o: f32):
      %r = arith.addf %v, %s : f32
      linalg.yield %r : f32
    } -> tensor<32x64xf32>
    return %h3 : tensor<32x64xf32>
  }
}
