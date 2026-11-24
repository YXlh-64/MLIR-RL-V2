"""B3: can two producers be fused into one consumer by tile_using_forall + two chained fuse_into_containing_op?"""
import argparse

from _run import compare, run
from _mlir_text import md_table, write_csv, write_md
from mlir.dialects.transform import interpreter
from mlir.ir import Context, Module
from mlir.passmanager import PassManager

# %a -> p1 = exp(a); (%a, %b) -> p2 = a * b; (p1, p2) -> c = p1 + p2
TOY = """#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%a: $T, %b: $T, %o1: $T, %o2: $T, %o3: $T) -> ($T, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %p1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%a : $T) outs(%o1 : $T) attrs = {tag = "p1"} {
    ^bb0(%x: f32, %y: f32):
      %e = math.exp %x : f32
      linalg.yield %e : f32
    } -> $T
    %p2 = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %b : $T, $T) outs(%o2 : $T) attrs = {tag = "p2"} {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %m = arith.mulf %x, %y : f32
      linalg.yield %m : f32
    } -> $T
    %c = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%p1, %p2 : $T, $T) outs(%o3 : $T) attrs = {tag = "c"} {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> $T
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %c, %delta : $T, i64
  }
}
"""

SCRIPT = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %c = transform.structured.match attributes{tag = "c"} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %c tile_sizes $TILES : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p1 = transform.structured.match attributes{tag = "p1"} in %root : (!transform.any_op) -> !transform.any_op
    %f1, %forall1 = transform.structured.fuse_into_containing_op %p1 into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p2 = transform.structured.match attributes{tag = "p2"} in %root : (!transform.any_op) -> !transform.any_op
    %f2, %forall2 = transform.structured.fuse_into_containing_op %p2 into %forall1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""


def tagged_parents(op, out):
    """{tag: [parent op names]} for every tagged op."""
    for region in op.regions:
        for block in region.blocks:
            for o in block.operations:
                if 'tag' in o.operation.attributes:
                    tag = str(o.operation.attributes['tag']).strip('"')
                    out.setdefault(tag, []).append(o.operation.parent.name)
                tagged_parents(o.operation, out)
    return out


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--shape', type=int, nargs=2, default=[256, 512])
ap.add_argument('--tiles', type=int, nargs=2, default=[32, 64])
ap.add_argument('--runs', type=int, default=5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
args = ap.parse_args()

code = TOY.replace('$T', f'tensor<{args.shape[0]}x{args.shape[1]}xf32>')
row = dict(shape='x'.join(map(str, args.shape)), tiles='x'.join(map(str, args.tiles)))
try:
    with Context():
        module = Module.parse(code)
        script = Module.parse(SCRIPT.replace('$TILES', str(args.tiles)))
        interpreter.apply_named_sequence(module, script.body.operations[0], script)
        PassManager.parse('builtin.module(canonicalize,cse)', module.context).run(module.operation)
        parents = tagged_parents(module.operation, {})
        fused = str(module)
    row.update(applied=True, parents=str(parents),
               both_fused=all(parents.get(t) and set(parents[t]) == {'scf.forall'} for t in ('p1', 'p2')))
    t0, r0 = run(code, args.runs, args.seed)
    t1, r1 = run(fused, args.runs, args.seed)
    ok, err = compare(r0, r1, 1e-5, 1e-6)
    row.update(allclose=ok, max_abs_err=err, base_min_ns=min(t0), fused_min_ns=min(t1))
    write_md(args.out, 'b3_fused.mlir', fused)
except Exception as e:  # noqa: BLE001
    row.update(applied=False, error=str(e)[:500])

write_csv(args.out, 'b3_diamond.csv', [row])
write_md(args.out, 'b3_summary.md', '# B3 two-producer fusion (toy diamond)\n\n' + md_table([row]))
