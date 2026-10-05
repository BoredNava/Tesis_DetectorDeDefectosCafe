"""Create positive (1) and negative (0) image pairs within each split only."""
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "data" / "siamese_pairs"
CLASSES = ("sano", "defecto")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def generate_for_split(split_dir: Path, output_path: Path, seed: int = 42) -> int:
    rng = random.Random(seed)
    by_class = {label: sorted(p.resolve() for p in (split_dir / label).rglob("*")
                              if p.is_file() and p.suffix.lower() in SUFFIXES) for label in CLASSES}
    positive_rows = []
    for images in by_class.values():
        shuffled = images[:]
        rng.shuffle(shuffled)
        positive_rows.extend((str(a), str(b), 1) for a, b in zip(shuffled[::2], shuffled[1::2]))
    if not positive_rows or not by_class["sano"] or not by_class["defecto"]:
        raise ValueError(f"Se necesitan suficientes imágenes de ambas clases en {split_dir} para formar pares")
    negative_rows = [(str(rng.choice(by_class["sano"])), str(rng.choice(by_class["defecto"])), 0)
                     for _ in range(len(positive_rows))]
    rows = positive_rows + negative_rows
    rng.shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["image1", "image2", "label"]); writer.writerows(rows)
    return len(rows)

def generate_pairs(splits_dir: Path = SPLITS_DIR, output_dir: Path = OUTPUT_DIR, seed: int = 42) -> dict:
    counts = {}
    for offset, split in enumerate(("train", "val", "test")):
        split_dir = splits_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}; run build_splits.py first")
        counts[split] = generate_for_split(split_dir, output_dir / f"{split}_pairs.csv", seed + offset)
        print(f"{split}: {counts[split]} pairs")
    return counts

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_pairs(args.splits_dir, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
