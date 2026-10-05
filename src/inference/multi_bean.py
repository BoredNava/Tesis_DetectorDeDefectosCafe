"""Hybrid multi-bean pipeline: OpenCV segments beans, SlimCNN labels each crop."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from src.models.slim_cnn import SlimCNN

@dataclass
class BeanPrediction:
    bbox: tuple[int,int,int,int]
    label: str
    confidence: float
    crop: np.ndarray
    contour_area: float

def segment_beans(image_bgr: np.ndarray, min_area_fraction: float = 0.0001,
                  max_area_fraction: float = 0.30, padding_fraction: float = 0.15):
    """Return bean-like components and a foreground mask for a light, even background.

    The method estimates background color from image borders, thresholds Lab color
    distance, cleans the mask morphologically and returns external contours.
    Touching/overlapping beans may be returned as one component.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Se esperaba una imagen BGR a color cargada por OpenCV")
    height,width=image_bgr.shape[:2]; image_area=height*width
    if min(height,width)<32: raise ValueError("La imagen es demasiado pequeña para segmentar granos")
    lab=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2LAB)
    border=np.concatenate((lab[0,:,:],lab[-1,:,:],lab[:,0,:],lab[:,-1,:]),axis=0)
    background=np.median(border,axis=0).astype(np.float32)
    distance=np.linalg.norm(lab.astype(np.float32)-background[None,None,:],axis=2)
    distance=cv2.normalize(distance,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    _,mask=cv2.threshold(distance,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k=max(3,int(round(min(height,width)*0.006))|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    for contour in contours:
        area=float(cv2.contourArea(contour)); fraction=area/image_area
        if not min_area_fraction<=fraction<=max_area_fraction: continue
        x,y,w,h=cv2.boundingRect(contour)
        if w<5 or h<5: continue
        pad=max(2,int(round(max(w,h)*padding_fraction)))
        x0=max(0,x-pad);y0=max(0,y-pad);x1=min(width,x+w+pad);y1=min(height,y+h+pad)
        found.append((contour,(x0,y0,x1-x0,y1-y0),area))
    found.sort(key=lambda item:(item[1][1],item[1][0]))
    return found,mask

class MultiBeanSlimCNN:
    """Load a trained Slim-CNN checkpoint and classify OpenCV bean crops."""
    def __init__(self,checkpoint:Path|str,device:str|None=None):
        path=Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"No existe el checkpoint Slim-CNN: {path}. Entrénalo primero en 04_entrenamiento_modelos.ipynb.")
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state.get("architecture")!="slim_cnn": raise ValueError("El checkpoint debe pertenecer a Slim-CNN")
        self.classes={int(idx):name for name,idx in state["class_to_idx"].items()}
        self.image_size=int(state["image_size"])
        self.model=SlimCNN(len(self.classes),state["channels"],state["dropout"])
        self.model.load_state_dict(state["model_state"]);self.model.to(self.device).eval()
        self.transform=transforms.Compose([transforms.Resize((self.image_size,self.image_size)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def classify(self,crop_bgr:np.ndarray)->tuple[str,float]:
        rgb=cv2.cvtColor(crop_bgr,cv2.COLOR_BGR2RGB)
        tensor=self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities=self.model(tensor).softmax(dim=1)[0]
        index=int(probabilities.argmax().item())
        return self.classes[index],float(probabilities[index].item())

def _square_crop(image_bgr:np.ndarray,bbox:tuple[int,int,int,int])->np.ndarray:
    x,y,w,h=bbox; patch=image_bgr[y:y+h,x:x+w]
    side=max(w,h); border=np.concatenate((image_bgr[0,:,:],image_bgr[-1,:,:],image_bgr[:,0,:],image_bgr[:,-1,:]),axis=0)
    bg=tuple(int(v) for v in np.median(border,axis=0))
    canvas=np.full((side,side,3),bg,dtype=np.uint8)
    dx=(side-w)//2;dy=(side-h)//2;canvas[dy:dy+h,dx:dx+w]=patch
    return canvas

def analyze_multi_bean_image(image_bgr:np.ndarray,classifier:MultiBeanSlimCNN,
                             min_area_fraction:float=0.0001,max_area_fraction:float=0.30,
                             min_confidence:float=0.0):
    components,mask=segment_beans(image_bgr,min_area_fraction,max_area_fraction)
    predictions=[]
    for _,bbox,area in components:
        crop=_square_crop(image_bgr,bbox);label,confidence=classifier.classify(crop)
        if confidence>=min_confidence:
            predictions.append(BeanPrediction(bbox,label,confidence,crop,area))
    return predictions,mask

def draw_candidate_boxes(image_bgr:np.ndarray,components):
    """Visualize OpenCV-only region proposals before a classifier is available."""
    output=image_bgr.copy()
    for index,(_,bbox,_) in enumerate(components,1):
        x,y,w,h=bbox
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,165,255),2)
        cv2.putText(output,str(index),(x,max(18,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,165,255),2,cv2.LINE_AA)
    return output

def draw_predictions(image_bgr:np.ndarray,predictions:list[BeanPrediction],only_class:str|None=None):
    output=image_bgr.copy()
    for item in predictions:
        if only_class and item.label!=only_class: continue
        x,y,w,h=item.bbox;color=(0,170,0) if item.label=="sano" else (0,0,230)
        cv2.rectangle(output,(x,y),(x+w,y+h),color,2)
        text=f"{item.label} {item.confidence:.2f}"
        cv2.putText(output,text,(x,max(18,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    return output
