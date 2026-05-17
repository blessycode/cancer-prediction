import csv
import struct
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "images"
METADATA_PATH = BASE_DIR / "metadata.csv"
FIELDNAMES = ["filepath", "class_label", "width", "height"]
JPEG_START_OF_FRAME_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def get_image_size(image_path):
    with image_path.open("rb") as image_file:
        header = image_file.read(24)

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return width, height

        if header[:2] != b"\xff\xd8":
            raise ValueError("unsupported image format")

        image_file.seek(2)
        while True:
            marker_start = image_file.read(1)
            while marker_start and marker_start != b"\xff":
                marker_start = image_file.read(1)

            marker = image_file.read(1)
            while marker == b"\xff":
                marker = image_file.read(1)

            if not marker:
                break

            marker_value = marker[0]
            if marker_value in (0xD8, 0xD9):
                continue

            segment_length_bytes = image_file.read(2)
            if len(segment_length_bytes) != 2:
                break

            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")

            if marker_value in JPEG_START_OF_FRAME_MARKERS:
                data = image_file.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return width, height

            image_file.seek(segment_length - 2, 1)

    raise ValueError("could not read image dimensions")


def build_metadata(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    rows = []

    for class_folder in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_folder.iterdir() if path.is_file()):
            try:
                width, height = get_image_size(image_path)

                rows.append(
                    {
                        "filepath": str(image_path.relative_to(BASE_DIR)),
                        "class_label": class_folder.name,
                        "width": width,
                        "height": height,
                    }
                )
            except Exception as exc:
                print(f"Skipping unreadable image: {image_path} ({exc})")

    if not rows:
        raise ValueError(f"No readable images were found in: {dataset_path}")

    return rows


def save_metadata(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_metadata(DATASET_PATH)
    save_metadata(rows, METADATA_PATH)

    print("Metadata CSV created successfully!")
    print(f"Total images: {len(rows)}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
