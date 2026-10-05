"""Evaluate Slim-CNN on the held-out internal test split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from src.data.classification import ROOT,make_imagefolder
from src.models.slim_cnn import SlimCNN

def evaluate(data_dir:Path,checkpoint:Path,output:Path):
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    ds=make_imagefolder(data_dir,"test",state["image_size"])
    names=[name for name,_ in sorted(state["class_to_idx"].items(),key=lambda item:item[1])]
    if ds.class_to_idx!=state["class_to_idx"]: raise ValueError("Las clases del test no coinciden con el checkpoint")
    loader=DataLoader(ds,batch_size=64,shuffle=False); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SlimCNN(len(names),state["channels"],state["dropout"]); model.load_state_dict(state["model_state"]); model.to(device).eval()
    truth=[];pred=[]
    with torch.no_grad():
        for images,labels in loader: truth.extend(labels.tolist());pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
    result={"accuracy":accuracy_score(truth,pred),"report":classification_report(truth,pred,labels=list(range(len(names))),target_names=names,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(truth,pred,labels=list(range(len(names)))).tolist(),"classes":names,"n_test":len(ds)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"accuracy":result["accuracy"],"n_test":result["n_test"],"report":result["report"]},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,default=ROOT/"data/splits");p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/slim_cnn/best.pt");p.add_argument("--output",type=Path,default=ROOT/"reports/slim_cnn_test.json");a=p.parse_args();evaluate(a.data,a.checkpoint,a.output)
