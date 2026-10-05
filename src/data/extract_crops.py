"""Extract beans from the labeled Specific-Defect contact sheets."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "raw" / "Specific-Defect"
OUTPUT_DIR = ROOT / "data" / "processed_crops"
NORMAL_IMAGE = "Normales.jpg"

def extract_from_image(image_path: Path, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    class_name = "sano" if image_path.name.lower() == NORMAL_IMAGE.lower() else "defecto"
    destination = output_dir / class_name
    destination.mkdir(parents=True, exist_ok=True)
    # A changed detector result must not leave obsolete crops from this sheet.
    for old_crop in destination.glob(f"{image_path.stem}_*.jpg"):
        old_crop.unlink()
    boxes = sorted((box for box in map(cv2.boundingRect, contours) if box[2] >= 20 and box[3] >= 20), key=lambda box: (box[1], box[0]))
    records = []
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        path = destination / f"{image_path.stem}_{index:04d}.jpg"
        if not cv2.imwrite(str(path), image[y:y + height, x:x + width]):
            raise OSError(f"Could not write crop: {path}")
        records.append({"path": path, "bbox": (x, y, width, height), "class": class_name})
    return records

def extract_all(source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Specific-Defect directory not found: {source_dir}")
    files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    if not files:
        raise FileNotFoundError(f"No JPG images found in {source_dir}")
    results = []
    for path in files:
        crops = extract_from_image(path, output_dir)
        results.extend(crops)
        print(f"{path.name}: {len(crops)} crops")
    print(f"Total crops: {len(results)}")
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    extract_all(args.source, args.output)

if __name__ == "__main__":
    main()
