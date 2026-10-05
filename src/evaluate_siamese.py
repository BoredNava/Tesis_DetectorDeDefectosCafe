"""Evaluate Siamese ResNet18 pair verification on held-out test pairs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from src.data.siamese_dataset import SiamesePairDataset
from src.models.siamese_resnet18 import SiameseResNet18
ROOT=Path(__file__).resolve().parents[1]
def evaluate(pair_dir:Path,checkpoint:Path,output:Path):
    state=torch.load(checkpoint,map_location="cpu",weights_only=False); dataset=SiamesePairDataset(pair_dir/"test_pairs.csv",state["image_size"])
    loader=DataLoader(dataset,batch_size=64,shuffle=False);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SiameseResNet18(state["embedding_dim"],False);model.load_state_dict(state["model_state"]);model.to(device).eval();truth=[];pred=[]
    with torch.no_grad():
        for image1,image2,label in loader:
            z1,z2=model(image1.to(device),image2.to(device));distance=torch.nn.functional.pairwise_distance(z1,z2)
            pred.extend((distance<state["threshold"]).int().cpu().tolist());truth.extend(label.tolist())
    result={"accuracy":accuracy_score(truth,pred),"report":classification_report(truth,pred,labels=[0,1],target_names=["different","same"],output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(truth,pred,labels=[0,1]).tolist(),"threshold":state["threshold"],"n_test_pairs":len(dataset)}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--pairs",type=Path,default=ROOT/"data/siamese_pairs");p.add_argument("--checkpoint",type=Path,default=ROOT/"checkpoints/siamese_resnet18/best.pt");p.add_argument("--output",type=Path,default=ROOT/"reports/siamese_resnet18_test.json");a=p.parse_args();evaluate(a.pairs,a.checkpoint,a.output)
