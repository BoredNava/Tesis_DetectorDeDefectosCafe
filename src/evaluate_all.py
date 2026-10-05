"""Dispatch test evaluation for YOLOv8-cls, Slim-CNN, or Siamese ResNet18."""
from __future__ import annotations
import argparse,subprocess,sys
MODULES={"yolov8_cls":"src.evaluate_yolo","yolov8_detect":"src.evaluate_yolo_detect","slim_cnn":"src.evaluate_slim_cnn","siamese_resnet18":"src.evaluate_siamese"}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--model",choices=MODULES,required=True)
    args,remaining=parser.parse_known_args()
    subprocess.run([sys.executable,"-m",MODULES[args.model],*remaining],check=True)
if __name__=="__main__":main()
