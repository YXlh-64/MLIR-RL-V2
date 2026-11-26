# Fusion layer (rl_autoschedular_v6) — progress snapshot, 2026-11-26

Numbers only; the full log of changes, remarks and results is
[`rl_autoschedular/rl_autoschedular_v6/fusion/CHANGES.md`](../../rl_autoschedular/rl_autoschedular_v6/fusion/CHANGES.md)
(section "Progress and results as of 2026-11-26"). How to run everything: [`scripts/fusion_data/COMMANDS.md`](../../scripts/fusion_data/COMMANDS.md).

| file | content |
|---|---|
| `plots/training_curves.png` | training speedup (geomean per iteration, 20-iteration moving average) and share of episodes faster than baseline, rows A, A', C, D, D+ |
| `tables/training_summary.md` | iterations reached, speedup first / last 50 iterations per row |
| `tables/v5_model_1200_partial_eval.md` | full-model eval of the reference checkpoint v5_model_1200 (12 of 13 models run, llama3_2_1b not finished) |
| `tables/v5_model_1200_fullmodel_eval.json` | the same, raw execution times in ns (null = failed) |
| `tables/training_models.json` | the 39 candidate training models for the units data (37 enabled) |
| `tables/data_profile_report.md`, `data_profile_suitability.csv`, `plots/data_profile_categories.png` | linalg op profile of the training models vs the eval models |

Not tracked (large / regenerable): checkpoints and run logs in `/scratch/mh8826/fusion_runs/`, data in `data/`, baselines in `results/`.
Regenerate the plot: `python scripts/fusion_data/plot_training.py`.
