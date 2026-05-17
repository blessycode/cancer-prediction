import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dataset import load_datasets
from src.utils.model_io import load_keras_model
from src.visualization.style import CLASS_PALETTE, set_thesis_style


MODELS = ["resnet50", "mobilenetv2", "efficientnetb0"]
RESOLUTIONS = [224, 128, 64, 32]
RUNS = 1
NUM_CLASSES = 8
CLASS_NAMES = ["A", "AGC", "ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SC"]
SAVE_DIR = Path("plots/roc_curves")
RESULTS_DIR = Path("results/roc_auc")


def collect_predictions(model, test_ds):
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_prob.extend(predictions)

    return np.asarray(y_true), np.asarray(y_prob)


def compute_roc_curves(y_true, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    curves = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        curves[class_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr),
        }

    micro_fpr, micro_tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    curves["micro_average"] = {
        "fpr": micro_fpr,
        "tpr": micro_tpr,
        "auc": auc(micro_fpr, micro_tpr),
    }

    all_fpr = np.unique(np.concatenate([curves[class_name]["fpr"] for class_name in CLASS_NAMES]))
    mean_tpr = np.zeros_like(all_fpr)

    for class_name in CLASS_NAMES:
        mean_tpr += np.interp(all_fpr, curves[class_name]["fpr"], curves[class_name]["tpr"])

    mean_tpr /= NUM_CLASSES
    curves["macro_average"] = {
        "fpr": all_fpr,
        "tpr": mean_tpr,
        "auc": auc(all_fpr, mean_tpr),
    }

    return y_true_bin, curves


def decorate_roc_axis(ax, title):
    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1.2, label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_per_class_roc(curves, model_name, resolution, run):
    fig, ax = plt.subplots(figsize=(10, 8))

    for index, class_name in enumerate(CLASS_NAMES):
        curve = curves[class_name]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=CLASS_PALETTE[index % len(CLASS_PALETTE)],
            linewidth=2.2,
            label=f"{class_name} AUC={curve['auc']:.3f}",
        )

    decorate_roc_axis(ax, f"Per-Class ROC Curves - {model_name} {resolution}x{resolution} Run {run}")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    fig.tight_layout()

    output_path = SAVE_DIR / f"{model_name}_{resolution}_run{run}_per_class_roc.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overall_roc(curves, model_name, resolution, run):
    fig, ax = plt.subplots(figsize=(9, 7))

    macro_curve = curves["macro_average"]
    micro_curve = curves["micro_average"]

    ax.plot(
        macro_curve["fpr"],
        macro_curve["tpr"],
        color="#D73A49",
        linewidth=3,
        label=f"Macro-average AUC={macro_curve['auc']:.3f}",
    )
    ax.plot(
        micro_curve["fpr"],
        micro_curve["tpr"],
        color="#1F6FEB",
        linewidth=3,
        label=f"Micro-average AUC={micro_curve['auc']:.3f}",
    )

    decorate_roc_axis(ax, f"Overall ROC Curves - {model_name} {resolution}x{resolution} Run {run}")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    output_path = SAVE_DIR / f"{model_name}_{resolution}_run{run}_overall_roc.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_combined_roc(curves, model_name, resolution, run):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    for index, class_name in enumerate(CLASS_NAMES):
        curve = curves[class_name]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=CLASS_PALETTE[index % len(CLASS_PALETTE)],
            linewidth=1.8,
            alpha=0.78,
            label=f"{class_name}={curve['auc']:.3f}",
        )

    ax.plot(
        curves["macro_average"]["fpr"],
        curves["macro_average"]["tpr"],
        color="#111827",
        linewidth=3.2,
        label=f"Macro={curves['macro_average']['auc']:.3f}",
    )
    ax.plot(
        curves["micro_average"]["fpr"],
        curves["micro_average"]["tpr"],
        color="#D73A49",
        linewidth=3.0,
        linestyle="-.",
        label=f"Micro={curves['micro_average']['auc']:.3f}",
    )

    decorate_roc_axis(ax, f"Per-Class and Overall ROC Curves - {model_name} {resolution}x{resolution} Run {run}")
    ax.legend(fontsize=8.2, loc="lower right", ncol=2)
    fig.tight_layout()

    output_path = SAVE_DIR / f"{model_name}_{resolution}_run{run}_combined_roc.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_auc_records(model_name, resolution, run, y_true, y_prob, curves):
    records = []

    for class_name in CLASS_NAMES:
        records.append(
            {
                "model": model_name,
                "resolution": resolution,
                "run": run,
                "curve": class_name,
                "auc": curves[class_name]["auc"],
            }
        )

    records.append(
        {
            "model": model_name,
            "resolution": resolution,
            "run": run,
            "curve": "macro_average",
            "auc": curves["macro_average"]["auc"],
        }
    )
    records.append(
        {
            "model": model_name,
            "resolution": resolution,
            "run": run,
            "curve": "micro_average",
            "auc": curves["micro_average"]["auc"],
        }
    )
    records.append(
        {
            "model": model_name,
            "resolution": resolution,
            "run": run,
            "curve": "weighted_average",
            "auc": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"),
        }
    )

    return records


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_thesis_style()

    all_records = []
    generated = 0

    for model_name in MODELS:
        for resolution in RESOLUTIONS:
            for run in range(1, RUNS + 1):
                dataset_path = Path("data") / "processed" / f"{resolution}x{resolution}"
                model_path = Path("experiments") / model_name / str(resolution) / f"run_{run}" / "best_model.keras"

                if not model_path.exists():
                    print(f"Missing model: {model_path}")
                    continue

                print(f"Generating ROC curves: {model_name} {resolution}x{resolution} run {run}")
                _, _, test_ds = load_datasets(str(dataset_path), (resolution, resolution))
                model = load_keras_model(model_path)
                y_true, y_prob = collect_predictions(model, test_ds)
                _, curves = compute_roc_curves(y_true, y_prob)

                plot_per_class_roc(curves, model_name, resolution, run)
                plot_overall_roc(curves, model_name, resolution, run)
                plot_combined_roc(curves, model_name, resolution, run)
                all_records.extend(build_auc_records(model_name, resolution, run, y_true, y_prob, curves))
                generated += 3

    if all_records:
        auc_df = pd.DataFrame(all_records)
        auc_df.to_csv(RESULTS_DIR / "roc_auc_summary.csv", index=False)
        auc_df.pivot_table(
            index=["model", "resolution", "run"],
            columns="curve",
            values="auc",
        ).reset_index().to_csv(RESULTS_DIR / "roc_auc_summary_wide.csv", index=False)

    print(f"\nSaved {generated} ROC curve PNG files to {SAVE_DIR}")
    print(f"Saved AUC summary tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
