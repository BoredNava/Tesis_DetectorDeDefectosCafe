"""Dispatch training to one of the three thesis model pipelines."""
from __future__ import annotations
import argparse,subprocess,sys
MODULES={"yolov8_cls":"src.train_yolo","yolov8_detect":"src.train_yolo_detect","slim_cnn":"src.train_slim_cnn","siamese_resnet18":"src.train_siamese"}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--model",choices=MODULES,required=True,help="Choose an architecture explicitly")
    args,remaining=parser.parse_known_args()
    subprocess.run([sys.executable,"-m",MODULES[args.model],*remaining],check=True)
if __name__=="__main__":main()
