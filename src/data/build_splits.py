"""Create leakage-safe train/validation/test splits using only USK-Coffee/train."""
from __future__ import annotations
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "dataset_config.yaml"
RAW_TRAIN_DIR = ROOT / "data" / "raw" / "USK-Coffee" / "train"
SPLITS_DIR = ROOT / "data" / "splits"
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_images(raw_train_dir: Path = RAW_TRAIN_DIR) -> list[dict]:
    """Read images only from train/<configured source class>/ directories."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records = []
    for label, folders in config["usk_coffee"].items():
        for folder in folders:
            class_dir = raw_train_dir / folder
            if not class_dir.is_dir():
                continue
            for image in sorted(class_dir.rglob("*")):
                if image.is_file() and image.suffix.lower() in SUFFIXES:
                    source = image.resolve()
                    records.append({
                        "source": source,
                        "class": label,
                        "source_group": "USK-Coffee-train",
                        # Each coffee-bean photo is an independent source sample.
                        "group_id": f"USK-Coffee-train:{source}",
                    })
    return records


def split_records(records: list[dict], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    if len(records) < 3:
        raise ValueError(f"Se necesitan al menos 3 imágenes de USK-Coffee/train; se encontraron {len(records)}")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios debe tener tres valores positivos que sumen 1; recibido: {ratios}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record.get("group_id", str(record["source"]))].append(record)
    group_ids = sorted(groups)
    if len(group_ids) < 6:
        raise ValueError(f"Se necesitan al menos 6 imágenes independientes para train/val/test; se encontraron {len(group_ids)}")

    labels = [groups[group_id][0]["class"] for group_id in group_ids]
    stratify = labels if min(labels.count(label) for label in set(labels)) >= 3 else None
    holdout_ratio = ratios[1] + ratios[2]
    train_ids, holdout_ids = train_test_split(
        group_ids, test_size=holdout_ratio, random_state=seed, stratify=stratify
    )
    holdout_labels = [groups[group_id][0]["class"] for group_id in holdout_ids]
    holdout_stratify = holdout_labels if min(holdout_labels.count(label) for label in set(holdout_labels)) >= 2 else None
    test_fraction_of_holdout = ratios[2] / holdout_ratio
    val_ids, test_ids = train_test_split(
        holdout_ids, test_size=test_fraction_of_holdout,
        random_state=seed, stratify=holdout_stratify
    )
    result = {
        name: [record for group_id in ids for record in groups[group_id]]
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    expected_classes = set(labels)
    for name, items in result.items():
        missing = expected_classes - {item["class"] for item in items}
        if missing:
            raise ValueError(
                f"La partición {name} no tendría la(s) clase(s) {sorted(missing)}. "
                "Se necesitan más imágenes de cada clase dentro de USK-Coffee/train."
            )
    return result


def build_splits(raw_train_dir: Path = RAW_TRAIN_DIR, output_dir: Path = SPLITS_DIR,
                 seed: int | None = None) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records = gather_images(raw_train_dir)
    if not records:
        raise FileNotFoundError(
            f"No se encontraron imágenes. Se buscaron en {raw_train_dir}/<clase>/; "
            "revisa los nombres de carpetas en config/dataset_config.yaml."
        )
    splits = split_records(records, config["seed"] if seed is None else seed,
                           tuple(config["split_ratios"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_index = output_dir / "index.csv"
    if previous_index.exists():
        with previous_index.open(newline="", encoding="utf-8") as file:
            for old in csv.DictReader(file):
                previous_file = output_dir / old["split_path"]
                if previous_file.is_file():
                    previous_file.unlink()

    audit = []
    for split, items in splits.items():
        for label in config["classes"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(items, 1):
            source = item["source"]
            dest = output_dir / split / item["class"] / f"USK-Coffee-train_{source.stem}_{index:05d}{source.suffix.lower()}"
            shutil.copy2(source, dest)
            audit.append({"split": split, "class": item["class"], "source_group": item["source_group"],
                          "group_id": item["group_id"], "source_path": str(source),
                          "split_path": str(dest.relative_to(output_dir))})

    fields = ["split", "class", "source_group", "group_id", "source_path", "split_path"]
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    for split, items in splits.items():
        counts = {label: sum(item["class"] == label for item in items) for label in config["classes"]}
        print(f"{split}: {len(items)} imágenes ({counts})")
    print(f"Fuente usada: {raw_train_dir}")
    print(f"Índice: {output_dir / 'index.csv'}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-train-dir", type=Path, default=RAW_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    build_splits(args.raw_train_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
