"""Pair-image datasets for metric-learning experiments."""
from __future__ import annotations
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.data.classification import image_transforms

class SiamesePairDataset(Dataset):
    def __init__(self, manifest: Path, image_size: int, training: bool = False):
        self.manifest=manifest
        with manifest.open(newline="",encoding="utf-8") as f:
            self.rows=list(csv.DictReader(f))
        if not self.rows: raise ValueError(f"Manifiesto sin pares: {manifest}")
        self.transform=image_transforms(image_size,training)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row["image1"]) as im: image1=self.transform(im.convert("RGB"))
        with Image.open(row["image2"]) as im: image2=self.transform(im.convert("RGB"))
        return image1,image2,int(row["label"])
