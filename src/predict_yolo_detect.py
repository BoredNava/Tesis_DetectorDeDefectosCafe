"""Run a trained YOLOv8 detector on a photo containing multiple beans."""
from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def predict(source:Path,checkpoint:Path,classes:list[int]|None,confidence:float,image_size:int):
    if not source.exists(): raise FileNotFoundError(f"No se encontró la imagen: {source}")
    if not checkpoint.is_file(): raise FileNotFoundError(f"No se encontró el detector entrenado: {checkpoint}")
    model=YOLO(str(checkpoint))
    results=model.predict(source=str(source),classes=classes,conf=confidence,imgsz=image_size,save=True,
        project=str(ROOT/"reports"/"predictions"),name="yolov8_detect",exist_ok=True)
    print(f"Imagen anotada guardada bajo {ROOT/'reports'/'predictions'/'yolov8_detect'}")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/yolov8_detect/weights/best.pt")
    p.add_argument("--classes",type=int,nargs="*",default=None,help="Sin este argumento muestra sano y defecto; usa --classes 1 para solo defectos")
    p.add_argument("--confidence",type=float,default=.25);p.add_argument("--image-size",type=int,default=640);a=p.parse_args();predict(a.source,a.checkpoint,a.classes,a.confidence,a.image_size)
