#!/usr/bin/env python3
"""Plot 18 full-model evaluation results on Bergamo comparing V5 (Transformer) vs V5 No Transformer (LSTM).

Generates:
  1. full_model_results.png (linear scale)
  2. full_model_results_log.png (log scale, shows sub-1x to 100x+ clearly)
  3. full_model_results_no_outliers.png (linear scale excluding Whisper & LLaMA)
Also writes full_model_results.csv with the exact numerical speedups and execution times.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(
    PROJECT_ROOT, "plots", "experimentation_plots", "mlir_rl_v1_paper", "v5_v5_no_transformer"
)
os.makedirs(OUT_DIR, exist_ok=True)

# Colors matching Okabe-Ito paper palette
COLOR_V5 = "#0072B2"         # Deep academic blue
COLOR_V5_NO_TF = "#D55E00"   # Vermilion

FONT_SETTINGS = {
    "title_size": 16,
    "label_size": 13,
    "tick_size": 11,
    "legend_size": 11,
    "title_weight": "bold",
    "label_weight": "bold",
}

# 18 Full Models evaluated on Bergamo
MODELS = [
    "albert",
    "bart",
    "bert",
    "convnext_tiny",
    "distilbert",
    "efficientnet_b0",
    "gat",
    "gin",
    "gpt2",
    "llama3_2_1b",
    "mobilenet_v3_small",
    "resnet50",
    "resnext50",
    "t5",
    "vgg16",
    "vit_b_16",
    "whisper_base",
    "yolov8m",
]

# Exact evaluation results on Bergamo
RESULTS = {
    "albert": {
        "v5_speedup": 15.5882, "v5_exec_ms": 435.22,
        "no_speedup": 0.0, "no_exec_ms": None, "no_status": "Timeout"
    },
    "bart": {
        "v5_speedup": 17.3631, "v5_exec_ms": 194.68,
        "no_speedup": 8.9120, "no_exec_ms": 379.30, "no_status": "OK"
    },
    "bert": {
        "v5_speedup": 19.3155, "v5_exec_ms": 351.40,
        "no_speedup": 0.0, "no_exec_ms": None, "no_status": "Timeout"
    },
    "convnext_tiny": {
        "v5_speedup": 8.2140, "v5_exec_ms": 1173.70,
        "no_speedup": 15.9068, "no_exec_ms": 606.08, "no_status": "OK"
    },
    "distilbert": {
        "v5_speedup": 16.4547, "v5_exec_ms": 205.26,
        "no_speedup": 9.8659, "no_exec_ms": 342.35, "no_status": "OK"
    },
    "efficientnet_b0": {
        "v5_speedup": 1.8526, "v5_exec_ms": 251.40,
        "no_speedup": 1.1545, "no_exec_ms": 403.43, "no_status": "OK"
    },
    "gat": {
        "v5_speedup": 0.7836, "v5_exec_ms": 63.60,
        "no_speedup": 0.9510, "no_exec_ms": 52.41, "no_status": "OK"
    },
    "gin": {
        "v5_speedup": 0.8602, "v5_exec_ms": 25.48,
        "no_speedup": 0.8982, "no_exec_ms": 24.40, "no_status": "OK"
    },
    "gpt2": {
        "v5_speedup": 15.4667, "v5_exec_ms": 432.25,
        "no_speedup": 0.0, "no_exec_ms": None, "no_status": "Timeout"
    },
    "llama3_2_1b": {
        "v5_speedup": 86.4734, "v5_exec_ms": 10967.90,
        "no_speedup": 0.0, "no_exec_ms": None, "no_status": "Timeout"
    },
    "mobilenet_v3_small": {
        "v5_speedup": 0.4879, "v5_exec_ms": 154.56,
        "no_speedup": 0.4339, "no_exec_ms": 173.79, "no_status": "OK"
    },
    "resnet50": {
        "v5_speedup": 7.6045, "v5_exec_ms": 726.08,
        "no_speedup": 3.3099, "no_exec_ms": 1668.14, "no_status": "OK"
    },
    "resnext50": {
        "v5_speedup": 9.5084, "v5_exec_ms": 600.61,
        "no_speedup": 6.7131, "no_exec_ms": 850.71, "no_status": "OK"
    },
    "t5": {
        "v5_speedup": 8.3521, "v5_exec_ms": 239.75,
        "no_speedup": 0.0, "no_exec_ms": None, "no_status": "OOM"
    },
    "vgg16": {
        "v5_speedup": 40.0587, "v5_exec_ms": 616.54,
        "no_speedup": 34.3938, "no_exec_ms": 718.09, "no_status": "OK"
    },
    "vit_b_16": {
        "v5_speedup": 1.5333, "v5_exec_ms": 37798.71,
        "no_speedup": 8.0687, "no_exec_ms": 7182.77, "no_status": "OK"
    },
    "whisper_base": {
        "v5_speedup": 105.1450, "v5_exec_ms": 1873.16,
        "no_speedup": 7.4794, "no_exec_ms": 26332.91, "no_status": "OK"
    },
    "yolov8m": {
        "v5_speedup": 11.0201, "v5_exec_ms": 3037.81,
        "no_speedup": 18.6492, "no_exec_ms": 1795.08, "no_status": "OK"
    },
}


def export_csv():
    rows = []
    for m in sorted(MODELS):
        d = RESULTS[m]
        rows.append({
            "model": m,
            "v5_speedup": d["v5_speedup"],
            "v5_exec_ms": d["v5_exec_ms"],
            "v5_no_transformer_speedup": d["no_speedup"],
            "v5_no_transformer_exec_ms": d["no_exec_ms"],
            "v5_no_transformer_status": d["no_status"],
            "winner": "v5" if d["v5_speedup"] > d["no_speedup"] else "v5_no_transformer"
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "full_model_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")


def plot_bar_chart(models, filename, title, log_scale=False, show_annotations=True):
    fig, ax = plt.subplots(figsize=(max(14, len(models) * 0.9), 7.5))
    x = np.arange(len(models))
    width = 0.38

    v5_vals = [RESULTS[m]["v5_speedup"] for m in models]
    no_vals = [RESULTS[m]["no_speedup"] for m in models]

    # In log scale, zeros must be handled gracefully
    plot_v5_vals = v5_vals
    plot_no_vals = [max(v, 0.05) if log_scale and v == 0.0 else v for v in no_vals]

    rects1 = ax.bar(
        x - width / 2, plot_v5_vals, width,
        label="v5 (Transformer ckpt 1200)", color=COLOR_V5, edgecolor="white", linewidth=0.6, zorder=3
    )
    rects2 = ax.bar(
        x + width / 2, plot_no_vals, width,
        label="v5_no_transformer (LSTM ckpt 1250)", color=COLOR_V5_NO_TF, edgecolor="white", linewidth=0.6, zorder=3
    )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="MLIR Baseline (1.0×)", zorder=2)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Speedup over MLIR Baseline (×, log scale)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
        ax.set_ylim(bottom=0.2, top=max(v5_vals + no_vals) * 2.2)
    else:
        ax.set_ylabel("Geometric Mean Speedup (×)", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
        ax.set_ylim(bottom=0, top=max(v5_vals + no_vals) * 1.15)

    ax.set_title(title, fontsize=FONT_SETTINGS["title_size"], fontweight=FONT_SETTINGS["title_weight"], pad=14)
    ax.set_xlabel("Full Model / Benchmark", fontsize=FONT_SETTINGS["label_size"], fontweight=FONT_SETTINGS["label_weight"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=FONT_SETTINGS["tick_size"])
    ax.tick_params(axis="y", labelsize=FONT_SETTINGS["tick_size"])

    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=FONT_SETTINGS["legend_size"], loc="upper left", frameon=True)

    # Annotate values and statuses above bars
    if show_annotations:
        for i, m in enumerate(models):
            v5_sp = RESULTS[m]["v5_speedup"]
            no_sp = RESULTS[m]["no_speedup"]
            no_stat = RESULTS[m]["no_status"]

            # V5 label
            if not log_scale or v5_sp > 0:
                val_str = f"{v5_sp:.1f}×" if v5_sp >= 10 else f"{v5_sp:.2f}×"
                ax.annotate(
                    val_str,
                    xy=(x[i] - width / 2, v5_sp),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color="#00466E",
                    rotation=0 if len(models) <= 12 else 45
                )

            # LSTM label
            if no_stat == "OK":
                val_str = f"{no_sp:.1f}×" if no_sp >= 10 else f"{no_sp:.2f}×"
                ax.annotate(
                    val_str,
                    xy=(x[i] + width / 2, no_sp),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color="#8F3C00",
                    rotation=0 if len(models) <= 12 else 45
                )
            else:
                # Mark Timeout or OOM
                y_pos = 0.25 if log_scale else 0.8
                ax.annotate(
                    no_stat,
                    xy=(x[i] + width / 2, y_pos),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontstyle="italic", fontweight="bold", color="#B22222",
                    rotation=90
                )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    export_csv()

    # 1. Full 18-model linear comparison
    plot_bar_chart(
        MODELS,
        "full_model_results.png",
        "18 Full Models on Bergamo (AMD EPYC 9754): V5 (Transformer) vs. V5 No Transformer (LSTM)",
        log_scale=False,
    )

    # 2. Full 18-model log-scale comparison (highlights sub-1x to 105x)
    plot_bar_chart(
        MODELS,
        "full_model_results_log.png",
        "18 Full Models on Bergamo: V5 (Transformer) vs. V5 No Transformer (LSTM) [Log Scale]",
        log_scale=True,
    )

    # 3. 16-model comparison (excluding extreme outliers whisper_base and llama3_2_1b)
    models_no_outliers = [m for m in MODELS if m not in {"whisper_base", "llama3_2_1b"}]
    plot_bar_chart(
        models_no_outliers,
        "full_model_results_no_llama3_whisper.png",
        "Full Models on Bergamo (Excl. Whisper & LLaMA): V5 (Transformer) vs. V5 No Transformer (LSTM)",
        log_scale=False,
    )


if __name__ == "__main__":
    main()
