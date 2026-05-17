import argparse
import os
import sys
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dataset import load_datasets
from src.models.model_factory import get_model
from src.training.trainer import train_model


MODELS = ["resnet50", "mobilenetv2", "efficientnetb0"]
RESOLUTIONS = [224, 128, 64, 32]
NUM_CLASSES = 8

FINE_TUNE_SETTINGS = {
    "resnet50": {
        "learning_rate": 7e-4,
        "fine_tune_learning_rate": 1e-5,
        "unfreeze_last_n": 45,
    },
    "mobilenetv2": {
        "learning_rate": 7e-4,
        "fine_tune_learning_rate": 8e-6,
        "unfreeze_last_n": 40,
    },
    "efficientnetb0": {
        "learning_rate": 7e-4,
        "fine_tune_learning_rate": 6e-6,
        "unfreeze_last_n": 50,
    },
}


def compute_class_weights(train_dir, max_weight=8.0):
    class_names = sorted(path.name for path in Path(train_dir).iterdir() if path.is_dir())
    counts = Counter()

    for index, class_name in enumerate(class_names):
        class_dir = Path(train_dir) / class_name
        counts[index] = len([path for path in class_dir.iterdir() if path.is_file()])

    total = sum(counts.values())
    num_classes = len(class_names)

    weights = {}
    for index in range(num_classes):
        if counts[index] == 0:
            weights[index] = 1.0
        else:
            balanced_weight = total / (num_classes * counts[index])
            weights[index] = float(min(balanced_weight, max_weight))

    print("Class counts:", {class_names[index]: counts[index] for index in range(num_classes)})
    print("Class weights:", {class_names[index]: round(weights[index], 3) for index in range(num_classes)})

    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train and fine-tune cervical cancer classifiers.")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS, choices=RESOLUTIONS)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--no-fine-tune", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def run_experiment(model_name, resolution, run, args):
    print("\n" + "=" * 60)
    print(f"Training {model_name} at {resolution}x{resolution} - run {run}")
    print("=" * 60)

    dataset_path = os.path.join("data", "processed", f"{resolution}x{resolution}")
    train_ds, val_ds, _ = load_datasets(dataset_path, (resolution, resolution))
    class_weight = None
    if not args.no_class_weights:
        class_weight = compute_class_weights(Path(dataset_path) / "train")

    model = get_model(
        model_name=model_name,
        input_shape=(resolution, resolution, 3),
        num_classes=NUM_CLASSES,
    )

    save_path = os.path.join("experiments", model_name, str(resolution), f"run_{run}")
    settings = FINE_TUNE_SETTINGS[model_name]

    train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        save_path=save_path,
        epochs=args.epochs,
        learning_rate=settings["learning_rate"],
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_learning_rate=settings["fine_tune_learning_rate"],
        unfreeze_last_n=settings["unfreeze_last_n"],
        class_weight=class_weight,
        monitor="val_accuracy",
    )


def main():
    args = parse_args()

    for model_name in args.models:
        for resolution in args.resolutions:
            for run in range(1, args.runs + 1):
                run_experiment(model_name, resolution, run, args)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
