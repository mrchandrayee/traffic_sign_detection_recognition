
from __future__ import annotations
import logging
from pathlib import Path
import cv2
logger=logging.getLogger(__name__)
CLS_CONF_THRESHOLD=0.50

def get_sign_color(label: str):
    # OpenCV BGR colors
    s=label.lower()
    if "stop" in s or "no_entry" in s: return (0,0,255)
    if "warning" in s or "yield" in s: return (0,165,255)
    return (0,220,0)

class TrafficSignDetector:
    """YOLO-based traffic-sign detector. Supply a traffic-sign-trained .pt model."""
    def __init__(self, model_path: str, conf: float=.25, iou: float=.45,
                 imgsz: int=640, device: str|None=None, use_tracking: bool=True):
        self.model_path=model_path; self.conf=conf; self.iou=iou
        self.imgsz=imgsz; self.device=device; self.use_tracking=use_tracking
        self.model=None
        try:
            from ultralytics import YOLO
            self.model=YOLO(model_path)
        except Exception as e:
            logger.warning("YOLO model could not be loaded: %s",e)
            raise RuntimeError("Install requirements and provide a valid YOLO traffic-sign model.") from e
    def detect(self, frame):
        if frame is None or frame.size==0: return []
        kwargs=dict(source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                    verbose=False, device=self.device)
        # track() persists IDs across frames; predict() is safer for single-frame calls.
        if self.use_tracking:
            try: pred=self.model.track(persist=True, tracker="bytetrack.yaml", **kwargs)
            except Exception: pred=self.model.predict(**kwargs)
        else: pred=self.model.predict(**kwargs)
        out=[]
        for r in pred:
            if r.boxes is None: continue
            for b in r.boxes:
                xyxy=b.xyxy[0].cpu().numpy().astype(int).tolist()
                out.append({"box":tuple(xyxy),"conf":float(b.conf[0].cpu()),
                            "track_id":int(b.id[0].cpu()) if b.id is not None else -1})
        return out
