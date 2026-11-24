# Overnight report

## ckpt_v5_model_1200

Blocks with results: 1474 (selected and translated: 1474)

| Method | ran ok | correctness passed | geomean speedup vs mlir_baseline* | blocks in geomean |
|---|---|---|---|---|
| mlir_baseline | 1463/1474 | 1452 | 1.000 | 1452 |
| mlir_rl | 1422/1474 | 1405 | 0.885 | 1405 |
| torch_eager | 1474/1474 | 1452 | 10.122 | 1452 |
| torch_inductor | 1474/1474 | 1452 | 12.865 | 1452 |

Failures (method, reason, count):

- `mlir_rl`: Worker exited without results; inspect worker.log (15)
- `mlir_baseline`: Worker process group exceeded timeout (11)
- `mlir_rl`: Worker process group exceeded timeout (10)
- `mlir_rl`: Failure while creating the ExecutionEngine. (4)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":67:13: 'llvm.intr.powi' op operand #1 mu (3)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":114:13: 'llvm.intr.powi' op operand #1 m (2)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":125:13: 'llvm.intr.powi' op operand #1 m (2)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":105:15: 'llvm.intr.powi' op operand #1 m (2)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":116:15: 'llvm.intr.powi' op operand #1 m (2)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":105:13: 'llvm.intr.powi' op operand #1 m (1)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":118:13: 'llvm.intr.powi' op operand #1 m (1)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":1654:13: 'llvm.intr.powi' op operand #1  (1)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":122:15: 'llvm.intr.powi' op operand #1 m (1)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":188:15: 'llvm.intr.powi' op operand #1 m (1)
- `mlir_rl`: Failure while executing pass pipeline:
error: "-":1624:13: 'llvm.intr.powi' op operand #1  (1)

*speedup = baseline median latency / method median latency, over blocks where both passed correctness. >1 means faster than plain MLIR.

Details: `results/<tag>/analysis/headline_table.md`, `results/<tag>/t1_warm/r0/<class>/REPORT.md`, `results/<tag>/figures/`, raw rows in `results/<tag>/all_results.csv`.
