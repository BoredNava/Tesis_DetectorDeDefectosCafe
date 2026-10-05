"""Train YOLOv8 in classification mode from class-folder image splits."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"config/yolo.yaml"
def train(data_dir:Path, epochs:int, batch:int, image_size:int, workers:int, device:str|None, model_name:str, pretrained:bool, seed:int, run_name:str, patience:int, resume:Path|None):
    for split in ("train","val"):
        if not (data_dir/split).is_dir(): raise FileNotFoundError(f"Falta {data_dir/split}; ejecuta el notebook de preparación")
    if not any(p.is_file() for p in (data_dir/"train").rglob("*")):
        raise FileNotFoundError(f"No hay imágenes en {data_dir/'train'}")
    # A .pt initializes from pretrained classification weights; .yaml starts from scratch.
    if resume is not None:
        if not resume.is_file(): raise FileNotFoundError(f"No existe checkpoint para reanudar: {resume}")
        model=YOLO(str(resume))
    else:
        model=YOLO(model_name if pretrained else model_name.removesuffix(".pt")+".yaml")
    augment=yaml.safe_load((ROOT/"config/augmentation.yaml").read_text(encoding="utf-8"))
    aug_args={"fliplr":augment["horizontal_flip"],"flipud":augment["vertical_flip"],
              "scale":max(1-augment["scale_min"],augment["scale_max"]-1),
              "hsv_h":augment["hue"],"hsv_s":augment["saturation"],"hsv_v":augment["brightness"]} if augment["enabled"] else {
              "fliplr":0.0,"flipud":0.0,"scale":0.0,"hsv_h":0.0,"hsv_s":0.0,"hsv_v":0.0}
    train_args={"data":str(data_dir.resolve()),"epochs":epochs,"batch":batch,"imgsz":image_size,
                "workers":workers,"device":device or (0 if __import__("torch").cuda.is_available() else "cpu"),
                "project":str(ROOT/"checkpoints"),"name":run_name,"exist_ok":False,
                "seed":seed,"plots":True,"patience":patience,
                "auto_augment":None,"erasing":0.0,**aug_args}
    if resume is not None:
        # Ultralytics restores optimizer, scheduler and epoch from last.pt.
        model.train(resume=True,device=train_args["device"])
    else:
        model.train(**train_args)
    save_dir=Path(model.trainer.save_dir)
    print(f"Mejor checkpoint YOLOv8-cls: {save_dir/'weights'/'best.pt'}")
    print(f"Último checkpoint/reanudación: {save_dir/'weights'/'last.pt'}")
if __name__=="__main__":
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8")); p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/splits"); p.add_argument("--epochs",type=int,default=cfg["epochs"]); p.add_argument("--batch",type=int,default=cfg["batch"])
    p.add_argument("--image-size",type=int,default=cfg["img_size"]); p.add_argument("--workers",type=int,default=cfg["workers"]); p.add_argument("--device",default=None)
    p.add_argument("--model",default=cfg["model"]); p.add_argument("--pretrained",action=argparse.BooleanOptionalAction,default=cfg["pretrained"]); p.add_argument("--seed",type=int,default=cfg["seed"]); p.add_argument("--run-name",default="yolov8_cls")
    p.add_argument("--patience",type=int,default=cfg["patience"]); p.add_argument("--resume",type=Path,default=None,help="Reanuda desde checkpoints/.../weights/last.pt")
    a=p.parse_args(); train(a.data,a.epochs,a.batch,a.image_size,a.workers,a.device,a.model,a.pretrained,a.seed,a.run_name,a.patience,a.resume)
