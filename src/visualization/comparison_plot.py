import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.style import set_thesis_style


RESULTS_PATH = Path("results/overall_metrics/all_overall_metrics.csv")
FALLBACK_RESULTS_PATH = Path("results/metrics")
SAVE_PATH = Path("plots/model_comparison")

MODEL_COLORS = {
    "resnet50": "#1F6FEB",
    "mobilenetv2": "#2DA44E",
    "efficientnetb0": "#D73A49",
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "roc_auc_macro",
    "roc_auc_weighted",
    "top_2_accuracy",
    "top_3_accuracy",
]

DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "Macro F1",
    "f1_weighted": "Weighted F1",
    "roc_auc_macro": "Macro AUC",
    "roc_auc_weighted": "Weighted AUC",
    "top_2_accuracy": "Top-2 Accuracy",
    "top_3_accuracy": "Top-3 Accuracy",
}


def load_results():
    if RESULTS_PATH.exists():
        df = pd.read_csv(RESULTS_PATH)
    else:
        records = []
        for file in FALLBACK_RESULTS_PATH.glob("*.json"):
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if "summary" in data:
                data = data["summary"]

            parts = file.stem.split("_")
            data["model"] = parts[0]
            data["resolution"] = int(parts[1])
            data["run"] = int(parts[2].replace("run", "")) if len(parts) > 2 else 1
            records.append(data)
        df = pd.DataFrame(records)

    df["resolution"] = df["resolution"].astype(int)
    df["run"] = df["run"].astype(int)
    return df.sort_values(["model", "resolution", "run"]).reset_index(drop=True)


def savefig(fig, folder, filename):
    output_dir = SAVE_PATH / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def style_axis(ax, title, xlabel=None, ylabel=None, ylim=None):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grouped_bar(df, metric):
    pivot = df.pivot_table(index="resolution", columns="model", values=metric, aggfunc="mean")
    pivot = pivot.sort_index()

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, model in enumerate(pivot.columns):
        values = pivot[model].values
        positions = x + (offset - (len(pivot.columns) - 1) / 2) * width
        bars = ax.bar(positions, values, width, label=model, color=MODEL_COLORS.get(model, "#6B7280"))
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_xticks(x, [f"{resolution}x{resolution}" for resolution in pivot.index])
    style_axis(ax, f"{DISPLAY_NAMES[metric]} by Model and Resolution", "Input resolution", DISPLAY_NAMES[metric], (0, 1.05))
    ax.legend(title="Model")
    fig.tight_layout()
    return savefig(fig, "grouped_bars", f"{metric}_grouped_bar.png")


def plot_heatmap(df, metric):
    pivot = df.pivot_table(index="model", columns="resolution", values=metric, aggfunc="mean")
    pivot = pivot.reindex(["resnet50", "mobilenetv2", "efficientnetb0"])
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=DISPLAY_NAMES[metric])

    ax.set_xticks(np.arange(len(pivot.columns)), [f"{col}x{col}" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(f"Heatmap of {DISPLAY_NAMES[metric]}", loc="left", fontsize=14, fontweight="bold", pad=12)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="#111827", fontsize=9)

    fig.tight_layout()
    return savefig(fig, "heatmaps", f"{metric}_heatmap.png")


def plot_resolution_line(df, metric):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for model, group in df.groupby("model"):
        group = group.sort_values("resolution")
        ax.plot(
            group["resolution"],
            group[metric],
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=MODEL_COLORS.get(model, "#6B7280"),
            label=model,
        )

    ax.set_xticks(sorted(df["resolution"].unique()))
    style_axis(ax, f"{DISPLAY_NAMES[metric]} Across Input Resolutions", "Input resolution", DISPLAY_NAMES[metric], (0, 1.05))
    ax.legend(title="Model")
    fig.tight_layout()
    return savefig(fig, "line_charts", f"{metric}_resolution_line.png")


def plot_metric_rank(df, metric):
    ranked = df.sort_values(metric, ascending=True).copy()
    ranked["label"] = ranked["model"] + " " + ranked["resolution"].astype(str) + "x" + ranked["resolution"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6.8))
    colors = [MODEL_COLORS.get(model, "#6B7280") for model in ranked["model"]]
    bars = ax.barh(ranked["label"], ranked[metric], color=colors)

    for bar in bars:
        ax.annotate(
            f"{bar.get_width():.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(5, 0),
            textcoords="offset points",
        )

    style_axis(ax, f"Model Ranking by {DISPLAY_NAMES[metric]}", DISPLAY_NAMES[metric], "Experiment", (0, 1.05))
    fig.tight_layout()
    return savefig(fig, "rankings", f"{metric}_ranking.png")


def plot_radar_for_best_resolution(df):
    radar_metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc_macro"]
    available_metrics = [metric for metric in radar_metrics if metric in df.columns]
    best_rows = df.sort_values("f1_macro", ascending=False).groupby("model", as_index=False).head(1)

    labels = [DISPLAY_NAMES[metric] for metric in available_metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for _, row in best_rows.iterrows():
        values = [row[metric] for metric in available_metrics]
        values += values[:1]
        model = row["model"]
        ax.plot(angles, values, linewidth=2.5, color=MODEL_COLORS.get(model, "#6B7280"), label=f"{model} {int(row['resolution'])}x")
        ax.fill(angles, values, color=MODEL_COLORS.get(model, "#6B7280"), alpha=0.12)

    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 1)
    ax.set_title("Best Experiment Profile by Model", fontsize=14, fontweight="bold", pad=18)
    ax.grid(color="#E5E7EB")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    return savefig(fig, "radar", "best_model_radar_profile.png")


def plot_metric_correlation(df):
    metric_cols = [metric for metric in METRICS if metric in df.columns]
    corr = df[metric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_xticks(np.arange(len(metric_cols)), [DISPLAY_NAMES[col] for col in metric_cols], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metric_cols)), [DISPLAY_NAMES[col] for col in metric_cols])
    ax.set_title("Metric Correlation Heatmap", loc="left", fontsize=14, fontweight="bold", pad=12)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#111827")

    fig.tight_layout()
    return savefig(fig, "heatmaps", "metric_correlation_heatmap.png")


def main():
    set_thesis_style()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    df = load_results()
    df.to_csv(SAVE_PATH / "model_comparison_data.csv", index=False)

    generated = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        generated.append(plot_grouped_bar(df, metric))
        generated.append(plot_heatmap(df, metric))
        generated.append(plot_resolution_line(df, metric))
        generated.append(plot_metric_rank(df, metric))

    generated.append(plot_radar_for_best_resolution(df))
    generated.append(plot_metric_correlation(df))

    print(f"Saved {len(generated)} comparison charts to {SAVE_PATH}")


if __name__ == "__main__":
    main()
