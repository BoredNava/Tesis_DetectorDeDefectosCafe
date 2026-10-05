"""Shared ImageFolder loading helpers with the project's configured label IDs."""
from __future__ import annotations
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]

def image_transforms(image_size: int, training: bool = False):
    ops=[transforms.Resize((image_size, image_size))]
    if training:
        aug=yaml.safe_load((ROOT / "config/augmentation.yaml").read_text(encoding="utf-8"))
        if aug["enabled"]:
            ops += [
                transforms.RandomHorizontalFlip(p=aug["horizontal_flip"]),
                transforms.RandomVerticalFlip(p=aug["vertical_flip"]),
                transforms.RandomRotation(degrees=aug["rotation_degrees"]),
                transforms.RandomAffine(degrees=0, translate=(aug["translation"],aug["translation"]),
                                        scale=(aug["scale_min"],aug["scale_max"])),
                transforms.RandomApply([transforms.ColorJitter(brightness=aug["brightness"],
                    contrast=aug["contrast"], saturation=aug["saturation"], hue=aug["hue"])],
                    p=aug["color_probability"]),
            ]
    ops += [transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    return transforms.Compose(ops)

def make_imagefolder(data_dir: Path, split: str, image_size: int, training: bool = False):
    dataset=datasets.ImageFolder(data_dir / split, transform=image_transforms(image_size, training))
    labels=yaml.safe_load((ROOT / "config/dataset_config.yaml").read_text(encoding="utf-8"))["classes"]
    missing=set(dataset.classes)-set(labels)
    if missing:
        raise ValueError(f"No hay IDs de clase configurados para: {sorted(missing)}")
    mapping={name:labels[name] for name in dataset.classes}
    dataset.samples=[(path,mapping[dataset.classes[target]]) for path,target in dataset.samples]
    dataset.targets=[mapping[dataset.classes[target]] for target in dataset.targets]
    dataset.class_to_idx=mapping
    dataset.classes=[name for name,_ in sorted(mapping.items(),key=lambda item:item[1])]
    return dataset

def make_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int = 0):
    data={split:make_imagefolder(data_dir,split,image_size,split=="train") for split in ("train","val")}
    for split,dataset in data.items():
        if len(dataset)==0: raise ValueError(f"La partición {split} está vacía: {data_dir/split}")
    if data["train"].class_to_idx != data["val"].class_to_idx:
        raise ValueError("Train y val deben tener las mismas clases")
    loaders={split:DataLoader(ds,batch_size=batch_size,shuffle=split=="train",num_workers=workers,
        pin_memory=torch.cuda.is_available(),persistent_workers=workers>0)
        for split,ds in data.items()}
    return loaders,data["train"].class_to_idx
