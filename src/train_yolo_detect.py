"""Train YOLOv8 detection on manually annotated images with multiple beans."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch,yaml
from ultralytics import YOLO
from src.data.yolo_detection import prepare_detection_dataset
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/"config/yolo_detect.yaml"
def train(data_dir:Path,epochs:int,batch:int,image_size:int,workers:int,device:str|None,seed:int,patience:int,resume:Path|None):
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));data_yaml=prepare_detection_dataset(data_dir,cfg["classes"])
    if resume is not None:
        if not resume.is_file(): raise FileNotFoundError(f"No existe checkpoint para reanudar: {resume}")
        model=YOLO(str(resume))
        model.train(resume=True,device=device or (0 if torch.cuda.is_available() else "cpu"))
    else:
        model=YOLO(cfg["model"])
        model.train(data=str(data_yaml),epochs=epochs,batch=batch,imgsz=image_size,workers=workers,
            device=device or (0 if torch.cuda.is_available() else "cpu"),project=str(ROOT/"checkpoints"),
            name="yolov8_detect",exist_ok=False,seed=seed,plots=True,patience=patience,
            **cfg["augmentation"])
    save_dir=Path(model.trainer.save_dir)
    print(f"Mejor checkpoint YOLOv8-detect: {save_dir/'weights'/'best.pt'}")
    print(f"Último checkpoint/reanudación: {save_dir/'weights'/'last.pt'}")
if __name__=="__main__":
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/detection");p.add_argument("--epochs",type=int,default=cfg["epochs"]);p.add_argument("--batch",type=int,default=cfg["batch"])
    p.add_argument("--image-size",type=int,default=cfg["img_size"]);p.add_argument("--workers",type=int,default=cfg["workers"]);p.add_argument("--device",default=None);p.add_argument("--seed",type=int,default=cfg["seed"])
    p.add_argument("--patience",type=int,default=cfg["patience"]);p.add_argument("--resume",type=Path,default=None,help="Reanuda desde weights/last.pt")
    a=p.parse_args();train(a.data,a.epochs,a.batch,a.image_size,a.workers,a.device,a.seed,a.patience,a.resume)
