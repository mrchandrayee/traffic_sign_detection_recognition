
from __future__ import annotations
import argparse, logging
from pathlib import Path
import cv2
from src.traffic_sign.pipeline import TrafficSignPipeline, TrafficSignPipelineConfig
from src.traffic_sign.video import write_video, show_frame, draw_on_frame

def parse_args():
    p=argparse.ArgumentParser(description="Traffic Sign Detection and Recognition")
    p.add_argument("--source", default="0", help="webcam index or video/image path")
    p.add_argument("--output", default="outputs/traffic_sign_output.mp4")
    p.add_argument("--det-model", default=None)
    p.add_argument("--cls-model", default=None)
    p.add_argument("--registry", default="models/registry.json")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--cls-conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--skip-frames", type=int, default=1)
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a=parse_args()
    source=int(a.source) if a.source.isdigit() else a.source
    cfg=TrafficSignPipelineConfig(det_model_path=a.det_model, cls_model_path=a.cls_model,
        registry_path=a.registry, det_conf=a.conf, cls_conf=a.cls_conf, iou=a.iou,
        imgsz=a.imgsz, device=a.device, skip_frames=max(1,a.skip_frames))
    pipe=TrafficSignPipeline(config=cfg)
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
        frame=cv2.imread(source)
        if frame is None: raise SystemExit(f"Cannot read image: {source}")
        results=pipe.process_frame(frame)
        out=draw_on_frame(frame, results, pipe.track_history, pipe.hud_events)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(a.output).with_suffix(".jpg")),out)
        if not a.no_display: show_frame("Traffic Sign Detection",out)
    else:
        write_video(pipe,source,a.output,max_frames=a.max_frames,imshow=not a.no_display)
if __name__=="__main__": main()
