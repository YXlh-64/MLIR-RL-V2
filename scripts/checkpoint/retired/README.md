# Retired full-model eval scripts (pre-V5.2)

These scripts implemented the **v4_5 block-window method** (per-block schedules
with window/stride over pre-extracted blocks) and are superseded by the V5.2
greedy full-model rollout (`scripts/eval/eval_fullmodel.sh`, design doc
`docs/design/todo/v5_2_full_model_eval.md`).

Retired 2026-09-17, kept for reference (do not run — inputs no longer exist):

| Script | Why retired |
|---|---|
| `ckpt_scan_all.sh` | Drove `optimize_model_via_blocks.py` over `data/nn/` (gone) |
| `submit_ckpt_scan.sh` | Same; also pinned `experiment3/v4_5_agent` checkpoints (gone) |
| `merge_ckpt_scan.py` | Schema-bound to old per-model scan JSONs; replaced by `scripts/eval/merge_fullmodel_matrix.py` |
| `run_rl_eval.sh` (in `../eval/retired/`) | Same dead inputs (`experiment3`, `full_model_dalma`, `data/nn/blocks`) |
| `optimize_model_via_blocks.py`, `optimize_full_model.py/.sh` (in `../full_model/retired/`) | The retired method itself |

Kept active in `scripts/full_model/`: `preprocess_model.py`,
`add_timing_wrapper.py`, and the `get_pytorch_*` / `merge_full_model_results.sh`
baseline helpers.
