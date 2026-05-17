import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.style import set_thesis_style


MODELS = ["resnet50", "mobilenetv2", "efficientnetb0"]
RESOLUTIONS = [224, 128, 64, 32]
RUNS = 1
SAVE_PATH = Path("plots/training_curves")

TRAIN_COLOR = "#1F6FEB"
VAL_COLOR = "#D73A49"
TOP2_COLOR = "#2DA44E"
TOP3_COLOR = "#8250DF"
LOSS_COLOR = "#FB8500"
GRID_COLOR = "#E5E7EB"


def read_training_log(log_path):
    df = pd.read_csv(log_path)
    df = df.dropna(how="all")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype(int)
    return df


def decorate_axis(ax, title, ylabel, ylim=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mark_best_epoch(ax, df):
    best_index = df["val_accuracy"].astype(float).idxmax()
    best_epoch = int(df.loc[best_index, "epoch"])
    best_value = float(df.loc[best_index, "val_accuracy"])
    ax.axvline(best_epoch, color="#6B7280", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.scatter([best_epoch], [best_value], s=58, color=VAL_COLOR, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(
        f"Best val acc: {best_value:.3f}\nEpoch {best_epoch}",
        xy=(best_epoch, best_value),
        xytext=(10, -34),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D0D7DE", "alpha": 0.95},
    )


def plot_single_experiment(df, model_name, resolution, run):
    title = f"{model_name.upper()} - {resolution}x{resolution} - Run {run}"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.985)

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["accuracy"], color=TRAIN_COLOR, linewidth=2.4, label="Training")
    ax.plot(df["epoch"], df["val_accuracy"], color=VAL_COLOR, linewidth=2.4, label="Validation")
    mark_best_epoch(ax, df)
    decorate_axis(ax, "Accuracy", "Accuracy", (0, 1.03))
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["loss"], color=TRAIN_COLOR, linewidth=2.4, label="Training")
    ax.plot(df["epoch"], df["val_loss"], color=VAL_COLOR, linewidth=2.4, label="Validation")
    decorate_axis(ax, "Loss", "Loss")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    if {"top_2_accuracy", "val_top_2_accuracy", "top_3_accuracy", "val_top_3_accuracy"}.issubset(df.columns):
        ax.plot(df["epoch"], df["val_top_2_accuracy"], color=TOP2_COLOR, linewidth=2.2, label="Validation top-2")
        ax.plot(df["epoch"], df["val_top_3_accuracy"], color=TOP3_COLOR, linewidth=2.2, label="Validation top-3")
        ax.plot(df["epoch"], df["val_accuracy"], color=VAL_COLOR, linewidth=2.0, label="Validation top-1")
    decorate_axis(ax, "Top-K Validation Accuracy", "Accuracy", (0, 1.03))
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    if "learning_rate" in df.columns:
        ax.plot(df["epoch"], df["learning_rate"], color=LOSS_COLOR, linewidth=2.4)
        ax.set_yscale("log")
    decorate_axis(ax, "Learning Rate Schedule", "Learning rate")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    output_path = SAVE_PATH / f"{model_name}_{resolution}_run{run}_training_dashboard.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_only(df, model_name, resolution, run):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df["epoch"], df["accuracy"], color=TRAIN_COLOR, linewidth=2.7, label="Training")
    ax.plot(df["epoch"], df["val_accuracy"], color=VAL_COLOR, linewidth=2.7, label="Validation")
    mark_best_epoch(ax, df)
    decorate_axis(ax, f"Accuracy Curve - {model_name} {resolution}x{resolution}", "Accuracy", (0, 1.03))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SAVE_PATH / f"{model_name}_{resolution}_run{run}_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_only(df, model_name, resolution, run):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df["epoch"], df["loss"], color=TRAIN_COLOR, linewidth=2.7, label="Training")
    ax.plot(df["epoch"], df["val_loss"], color=VAL_COLOR, linewidth=2.7, label="Validation")
    decorate_axis(ax, f"Loss Curve - {model_name} {resolution}x{resolution}", "Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(SAVE_PATH / f"{model_name}_{resolution}_run{run}_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_thesis_style()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    generated = 0
    for model_name in MODELS:
        for resolution in RESOLUTIONS:
            for run in range(1, RUNS + 1):
                log_path = Path("experiments") / model_name / str(resolution) / f"run_{run}" / "training_log.csv"

                if not log_path.exists():
                    print(f"Missing log: {log_path}")
                    continue

                df = read_training_log(log_path)
                plot_single_experiment(df, model_name, resolution, run)
                plot_accuracy_only(df, model_name, resolution, run)
                plot_loss_only(df, model_name, resolution, run)
                generated += 3

    print(f"Saved {generated} training curve PNG files to {SAVE_PATH}")


if __name__ == "__main__":
    main()
