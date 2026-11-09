
from __future__ import annotations
import logging
logger=logging.getLogger(__name__)
class TrafficSignClassifier:
    """Image classifier returning [(class_name, confidence), ...]."""
    def __init__(self, weights_path: str|None=None, device: str|None=None):
        self.model=None; self.device=device
        if weights_path:
            try:
                from ultralytics import YOLO
                self.model=YOLO(weights_path)
            except Exception as e:
                raise RuntimeError(f"Could not load classifier weights {weights_path}: {e}") from e
    def classify(self, crop, top_k: int=3):
        if crop is None or crop.size==0 or self.model is None: return []
        results=self.model.predict(source=crop, verbose=False, device=self.device)
        if not results or results[0].probs is None: return []
        probs=results[0].probs
        names=results[0].names
        idx=probs.top5[:top_k].tolist()
        conf=probs.top5conf[:top_k].tolist()
        return [(str(names[int(i)]),float(c)) for i,c in zip(idx,conf)]
