import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

RESOLUTIONS = [224, 128, 64, 32]
TRAIN_SIZE = 0.70
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="Resize and split raw cervical image dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--clean-output", action="store_true", help="Remove the selected processed resolution folders before rebuilding them.")
    return parser.parse_args()


def create_directory(path):
    path.mkdir(parents=True, exist_ok=True)


def get_classes(raw_dataset_path):
    if not raw_dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {raw_dataset_path}")

    classes = sorted(path.name for path in raw_dataset_path.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in: {raw_dataset_path}")

    return classes


def split_files(image_files, seed):
    if len(image_files) < 3:
        raise ValueError("Each class needs at least 3 images for train/val/test splitting.")

    train_files, temp_files = train_test_split(
        image_files,
        test_size=(1 - TRAIN_SIZE),
        random_state=seed,
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=seed,
    )

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def resize_and_save(src_path, dst_path, resolution):
    with Image.open(src_path).convert("RGB") as image:
        image = ImageOps.pad(
            image,
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )
        image.save(dst_path)


def process_resolution(raw_dataset_path, output_path, classes, resolution, seed, clean_output=False):
    print(f"\nProcessing Resolution: {resolution}x{resolution}", flush=True)
    resolution_path = output_path / f"{resolution}x{resolution}"

    if clean_output and resolution_path.exists():
        shutil.rmtree(resolution_path)

    for split in ["train", "val", "test"]:
        for class_name in classes:
            create_directory(resolution_path / split / class_name)

    for class_name in classes:
        print(f"Processing Class: {class_name}", flush=True)
        class_path = raw_dataset_path / class_name

        image_files = [
            path.name
            for path in class_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_files = sorted(image_files)
        random.Random(seed).shuffle(image_files)

        split_data = split_files(image_files, seed)

        for split_name, files in split_data.items():
            for file_name in files:
                src_path = class_path / file_name
                dst_path = resolution_path / split_name / class_name / file_name

                try:
                    resize_and_save(src_path, dst_path, resolution)
                except Exception as exc:
                    print(f"Error processing {src_path}: {exc}")


def main():
    args = parse_args()
    raw_dataset_path = args.raw_dir.resolve()
    output_path = args.output_dir.resolve()

    classes = get_classes(raw_dataset_path)

    print("Raw dataset path:")
    print(raw_dataset_path)
    print("\nOutput path:")
    print(output_path)
    print("\nClasses Found:")
    print(classes)

    for resolution in args.resolutions:
        process_resolution(raw_dataset_path, output_path, classes, resolution, args.seed, args.clean_output)

    print("\nDataset preprocessing completed successfully.")


if __name__ == "__main__":
    main()
