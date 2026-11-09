
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging, cv2
from src.common.model_registry import load_model_registry, require_model_path
from .detector import TrafficSignDetector
from .classifier import TrafficSignClassifier
logger=logging.getLogger(__name__)

@dataclass(frozen=True)
class TrafficSignPipelineConfig:
    det_model_path: Optional[str|Path]=None
    cls_model_path: Optional[str|Path]=None
    registry_path: Optional[str|Path]="models/registry.json"
    det_conf: float=.25
    cls_conf: float=.50
    iou: float=.45
    imgsz: int=640
    device: Optional[str]=None
    bbox_expand: float=.15
    skip_frames: int=1
    cache_conf: float=.90
    max_track_history: int=30
    display_size: tuple[int,int]=(960,540)

class TrafficSignPipeline:
    def __init__(self, config: TrafficSignPipelineConfig|None=None):
        self.config=config or TrafficSignPipelineConfig()
        registry=load_model_registry(self.config.registry_path)
        det=self.config.det_model_path
        cls=self.config.cls_model_path
        if det is None:
            try: det=require_model_path(registry,"traffic_sign_detector")
            except FileNotFoundError: det="models/traffic-sign-yolo.pt"
        if cls is None:
            try: cls=require_model_path(registry,"traffic_sign_classifier")
            except FileNotFoundError: cls="models/traffic-sign-classifier.pt"
        if not Path(det).exists():
            raise FileNotFoundError(f"Detector weights missing: {det}. See README to add your trained model.")
        if not Path(cls).exists():
            raise FileNotFoundError(f"Classifier weights missing: {cls}. See README to add your trained model.")
        self.detector=TrafficSignDetector(str(det),self.config.det_conf,self.config.iou,
                                          self.config.imgsz,self.config.device,True)
        self.classifier=TrafficSignClassifier(str(cls),self.config.device)
        self.frame_counter=0; self.track_history={}; self.cached_signs={}; self.hud_events=[]
    def process_frame(self, frame):
        if frame is None or frame.size==0: return []
        self.frame_counter+=1
        h,w=frame.shape[:2]; results=[]
        for det in self.detector.detect(frame):
            x1,y1,x2,y2=det["box"]; tid=det.get("track_id",-1); dc=det["conf"]
            if tid!=-1:
                cx=(x1+x2)//2; cy=(y1+y2)//2
                self.track_history.setdefault(tid,[]).append((cx,cy))
                self.track_history[tid]=self.track_history[tid][-self.config.max_track_history:]
            cached=self.cached_signs.get(tid) if tid!=-1 else None
            if cached and cached["top_conf"]>=self.config.cache_conf:
                result={"box":(x1,y1,x2,y2),"det_conf":dc,"track_id":tid,**cached}
                results.append(result); continue
            bw=x2-x1; bh=y2-y1; e=self.config.bbox_expand
            nx1=max(0,int(x1-bw*e)); ny1=max(0,int(y1-bh*e))
            nx2=min(w,int(x2+bw*e)); ny2=min(h,int(y2+bh*e))
            crop=frame[ny1:ny2,nx1:nx2].copy()
            cls_results=self.classifier.classify(crop,top_k=3)
            if cls_results and cls_results[0][1]>=self.config.cls_conf:
                top_class,top_conf=cls_results[0]
                vn_name=top_class.replace("_"," ").title()
            else:
                top_class="Unknown"; top_conf=cls_results[0][1] if cls_results else 0.0; vn_name="Unknown"
            payload={"sign_crop":crop,"cls_results":cls_results,"top_class":top_class,
                     "top_conf":top_conf,"vn_name":vn_name}
            result={"box":(x1,y1,x2,y2),"det_conf":dc,"track_id":tid,**payload}
            results.append(result)
            if tid!=-1 and top_class!="Unknown":
                old=self.cached_signs.get(tid)
                if old is None or top_conf>old["top_conf"]: self.cached_signs[tid]=payload
        self.rebuild_hud()
        return results
    def rebuild_hud(self):
        best={}
        for tid,c in self.cached_signs.items():
            name=c["top_class"]
            if name=="Unknown": continue
            if name not in best or c["top_conf"]>best[name]["cls_conf"]:
                best[name]={"class_name":name,"vn_name":c["vn_name"],"cls_conf":c["top_conf"],"track_id":tid}
        self.hud_events=sorted(best.values(),key=lambda x:x["cls_conf"],reverse=True)[:10]
