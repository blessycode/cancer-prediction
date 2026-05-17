import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dataset import load_datasets
from src.utils.model_io import load_keras_model
from src.visualization.style import set_thesis_style


MODELS = ["resnet50", "mobilenetv2", "efficientnetb0"]
RESOLUTIONS = [224, 128, 64, 32]
RUNS = 1
CLASS_NAMES = ["A", "AGC", "ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SC"]
SAVE_DIR = Path("plots/confusion_matrices")


def collect_predictions(model, test_ds):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    return np.asarray(y_true), np.asarray(y_pred)


def plot_confusion_matrix(cm, model_name, resolution, run):
    cm_percent = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row percentage")

    ax.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix - {model_name} {resolution}x{resolution} Run {run}", pad=14)

    threshold = max(cm_percent.max() / 2, 35)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_percent[i, j] > threshold else "#1F2933"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_percent[i, j]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    output_path = SAVE_DIR / f"{model_name}_{resolution}_run{run}_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    set_thesis_style()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    for model_name in MODELS:
        for resolution in RESOLUTIONS:
            for run in range(1, RUNS + 1):
                dataset_path = Path("data") / "processed" / f"{resolution}x{resolution}"
                model_path = Path("experiments") / model_name / str(resolution) / f"run_{run}" / "best_model.keras"

                if not model_path.exists():
                    print(f"Missing model: {model_path}")
                    continue

                print(f"Generating confusion matrix: {model_name} {resolution}x{resolution} run {run}")
                _, _, test_ds = load_datasets(str(dataset_path), (resolution, resolution))
                model = load_keras_model(model_path)
                y_true, y_pred = collect_predictions(model, test_ds)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
                output_path = plot_confusion_matrix(cm, model_name, resolution, run)
                print(f"Saved: {output_path}")
                generated += 1

    print(f"\nSaved {generated} confusion matrix PNG files to {SAVE_DIR}")


if __name__ == "__main__":
    main()
