"""Evaluate a trained multi-bean YOLO detector on its held-out test scenes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import yaml
from ultralytics import YOLO
from src.data.yolo_detection import prepare_detection_dataset
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/"config/yolo_detect.yaml"
def evaluate(data:Path,checkpoint:Path,output:Path):
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));manifest=prepare_detection_dataset(data,cfg["classes"])
    metrics=YOLO(str(checkpoint)).val(data=str(manifest),split="test",plots=True,project=str(output.parent),name="yolov8_detect_test",exist_ok=True)
    values={key:float(value) for key,value in metrics.results_dict.items() if isinstance(value,(float,int))}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(values,indent=2),encoding="utf-8");print(json.dumps(values,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,default=ROOT/"data/detection");p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/yolov8_detect/weights/best.pt");p.add_argument("--output",type=Path,default=ROOT/"reports/yolov8_detect_test.json");a=p.parse_args();evaluate(a.data,a.checkpoint,a.output)
