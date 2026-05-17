from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.style import CLASS_PALETTE, polish_axes, set_thesis_style


DATA_ROOT = Path("data/processed/224x224")
METADATA_PATH = Path("metadata.csv")
SAVE_DIR = Path("plots/eda")
CLASS_NAMES = ["A", "AGC", "ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SC"]


def collect_processed_counts(data_root):
    records = []
    for split_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            records.append(
                {
                    "split": split_dir.name,
                    "class_label": class_dir.name,
                    "count": len(list(class_dir.glob("*"))),
                }
            )
    return pd.DataFrame(records)


def plot_class_distribution(counts):
    totals = counts.groupby("class_label", as_index=False)["count"].sum()
    totals["class_label"] = pd.Categorical(totals["class_label"], categories=CLASS_NAMES, ordered=True)
    totals = totals.sort_values("class_label")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(totals["class_label"], totals["count"], color=CLASS_PALETTE)
    polish_axes(ax, "Class label", "Number of images", "Overall Class Distribution")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", (bar.get_x() + bar.get_width() / 2, height), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(SAVE_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(counts):
    pivot = counts.pivot_table(index="class_label", columns="split", values="count", fill_value=0)
    pivot = pivot.reindex(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#2F80ED", "#27AE60", "#F2994A"])
    polish_axes(ax, "Class label", "Number of images", "Train/Validation/Test Split by Class")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "split_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_image_dimensions(metadata_path):
    if not metadata_path.exists():
        return

    df = pd.read_csv(metadata_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["width"], df["height"], c="#2F80ED", alpha=0.55, edgecolors="white", linewidths=0.4)
    polish_axes(ax, "Width (pixels)", "Height (pixels)", "Original Image Dimensions")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "image_dimensions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(data_root):
    sample_paths = []
    for class_name in CLASS_NAMES:
        class_files = sorted((data_root / "train" / class_name).glob("*"))
        if class_files:
            sample_paths.append((class_name, class_files[0]))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (class_name, image_path) in zip(axes.flat, sample_paths):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(class_name, fontsize=11, fontweight="bold")
        ax.axis("off")

    for ax in axes.flat[len(sample_paths):]:
        ax.axis("off")

    fig.suptitle("Representative Processed Images by Class", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "sample_images_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_tables(counts):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(SAVE_DIR / "split_class_counts.csv", index=False)

    totals = counts.groupby("class_label", as_index=False)["count"].sum()
    totals["percentage"] = totals["count"] / totals["count"].sum() * 100
    totals.to_csv(SAVE_DIR / "overall_class_counts.csv", index=False)


def main():
    set_thesis_style()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    counts = collect_processed_counts(DATA_ROOT)
    save_summary_tables(counts)
    plot_class_distribution(counts)
    plot_split_distribution(counts)
    plot_image_dimensions(METADATA_PATH)
    plot_sample_grid(DATA_ROOT)

    print(f"EDA plots and tables saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
