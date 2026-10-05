"""Segment a multi-bean scene with OpenCV, classify each crop, and draw boxes."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from src.inference.multi_bean import MultiBeanSlimCNN,analyze_multi_bean_image,draw_predictions
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True,help="Foto con varios granos sobre fondo claro")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/slim_cnn/best.pt")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/multi_bean_prediction.jpg")
    parser.add_argument("--only",choices=["sano","defecto"],default=None,help="Dibuja solo las cajas de una clase")
    parser.add_argument("--min-area-fraction",type=float,default=.0001)
    parser.add_argument("--max-area-fraction",type=float,default=.30)
    parser.add_argument("--min-confidence",type=float,default=0.0)
    args=parser.parse_args()
    image=cv2.imread(str(args.source))
    if image is None: raise FileNotFoundError(f"No pude abrir la imagen: {args.source}")
    model=MultiBeanSlimCNN(args.checkpoint)
    results,mask=analyze_multi_bean_image(image,model,args.min_area_fraction,args.max_area_fraction,args.min_confidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    annotated=draw_predictions(image,results,args.only)
    if not cv2.imwrite(str(args.output),annotated): raise OSError(f"No pude guardar {args.output}")
    print(f"Granos segmentados/clasificados: {len(results)}")
    print(f"Resultado: {args.output}")
    for i,result in enumerate(results,1): print(f"{i}: {result.label}, confianza={result.confidence:.3f}, caja={result.bbox}")
if __name__=="__main__":main()
