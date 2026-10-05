"""Validate box annotations and prepare the YAML manifest used by YOLO detection."""
from __future__ import annotations
from pathlib import Path
import yaml

IMAGE_SUFFIXES={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
ROOT=Path(__file__).resolve().parents[2]

def prepare_detection_dataset(data_dir:Path,classes:dict[str,int]):
    names=[name for name,index in sorted(classes.items(),key=lambda item:item[1])]
    if sorted(classes.values())!=list(range(len(classes))):
        raise ValueError("Los IDs de clase deben ser consecutivos desde 0")
    seen=set()
    for split in ("train","val","test"):
        image_dir=data_dir/"images"/split; label_dir=data_dir/"labels"/split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Faltan carpetas para {split}: se esperan {image_dir} y {label_dir}")
        images=[p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not images: raise ValueError(f"No hay imágenes en {image_dir}")
        split_seen=set()
        for image in images:
            label=label_dir/(image.stem+".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Falta etiqueta YOLO para {image}: {label} (usa archivo vacío solo si no hay objetos)")
            for line_number,line in enumerate(label.read_text(encoding="utf-8").splitlines(),1):
                if not line.strip(): continue
                parts=line.split()
                if len(parts)!=5: raise ValueError(f"{label}:{line_number}: cada caja debe tener class_id cx cy width height")
                try: class_id=int(parts[0]); coords=[float(x) for x in parts[1:]]
                except ValueError as exc: raise ValueError(f"ID de clase o coordenadas no válidas en {label}:{line_number}") from exc
                if class_id not in range(len(names)):
                    raise ValueError(f"ID de clase inválido en {label}:{line_number}")
                cx,cy,width,height=coords
                if not all(0<=x<=1 for x in (cx,cy,width,height)) or width<=0 or height<=0 or cx-width/2<0 or cx+width/2>1 or cy-height/2<0 or cy+height/2>1:
                    raise ValueError(f"Caja debe estar normalizada dentro de la imagen: {label}:{line_number}")
                split_seen.add(int(class_id))
        if not split_seen: raise ValueError(f"No hay cajas anotadas en {split}; revisa sus archivos labels/*.txt")
        missing_classes=set(range(len(names)))-split_seen
        if missing_classes: raise ValueError(f"Faltan clases {sorted(missing_classes)} en las cajas de {split}")
        seen.update(split_seen)
    if seen!=set(range(len(names))):
        raise ValueError(f"Clases sin cajas en el conjunto: {sorted(set(range(len(names)))-seen)}")
    manifest={"path":str(data_dir.resolve()),"train":"images/train","val":"images/val","test":"images/test","names":{i:name for i,name in enumerate(names)}}
    yaml_path=data_dir/"dataset.yaml";yaml_path.write_text(yaml.safe_dump(manifest,sort_keys=False,allow_unicode=True),encoding="utf-8")
    return yaml_path
