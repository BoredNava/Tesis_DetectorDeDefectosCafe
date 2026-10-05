"""Evaluate a trained YOLOv8 classifier on the prepared test split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
def evaluate(data:Path,checkpoint:Path,project:Path):
    test_dir=data/"test"
    if not test_dir.is_dir():raise FileNotFoundError(f"No existe {test_dir}")
    model=YOLO(str(checkpoint));metrics=model.val(data=str(data.resolve()),split="test",project=str(project),name="yolov8_cls_test",plots=True)
    values={key:float(value) for key,value in metrics.results_dict.items() if isinstance(value,(float,int))}
    project.mkdir(parents=True,exist_ok=True);(project/"yolov8_cls_test_metrics.json").write_text(json.dumps(values,indent=2),encoding="utf-8")
    print(json.dumps(values,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,default=ROOT/"data/splits");p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/yolov8_cls/weights/best.pt");p.add_argument("--output-dir",type=Path,default=ROOT/"reports");a=p.parse_args();evaluate(a.data,a.checkpoint,a.output_dir)
