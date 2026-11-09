
from __future__ import annotations
import os, time, cv2
from pathlib import Path
import numpy as np
from .visualizer import TrafficSignVisualizer

def show_frame(window_name, frame, wait_ms=1, display_size=None):
    shown=cv2.resize(frame,display_size) if display_size else frame
    cv2.imshow(window_name,shown)
    key=cv2.waitKey(wait_ms)&0xFF
    if key in (ord("p"),): cv2.waitKey(0)
    return key not in (27,ord("q"))

def draw_on_frame(frame,pipeline_results,track_history=None,hud_events=None):
    detections=[]; classifications=[]
    for r in pipeline_results:
        detections.append((r["box"],r["det_conf"]))
        classifications.append(r.get("cls_results",[]))
    vis=TrafficSignVisualizer.draw_frame(frame,detections,classifications,
             [r.get("track_id",-1) for r in pipeline_results])
    if track_history:
        for r in pipeline_results:
            tid=r.get("track_id",-1)
            if tid!=-1 and tid in track_history:
                pts=track_history[tid]
                if len(pts)>1:
                    color=(0,220,255)
                    for i in range(1,len(pts)):
                        ratio=i/float(len(pts)); thickness=max(1,int(1+3*ratio))
                        cv2.line(vis,pts[i-1],pts[i],color,thickness,cv2.LINE_AA)
                    cv2.circle(vis,pts[-1],5,(255,255,255),-1,cv2.LINE_AA)
    if hud_events: vis=TrafficSignVisualizer.draw_hud_overlay(vis,hud_events)
    return vis

def write_video(pipeline,source_path,target_path,max_frames=None,progress=True,
                progress_interval=30,imshow=False,window_name="Traffic Sign Detection",
                wait_ms=1,display_size=(960,540)):
    cap=cv2.VideoCapture(source_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video source: {source_path}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None: total=min(total,max_frames) if total else max_frames
    Path(target_path).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(target_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened(): cap.release(); raise RuntimeError(f"Cannot create output video: {target_path}")
    idx=0; t0=time.time()
    try:
        while True:
            ok,frame=cap.read()
            if not ok or (max_frames is not None and idx>=max_frames): break
            results=pipeline.process_frame(frame)
            vis=draw_on_frame(frame,results,pipeline.track_history,pipeline.hud_events)
            writer.write(vis)
            if imshow and not show_frame(window_name,vis,wait_ms,display_size): break
            idx+=1
            if progress and idx%progress_interval==0:
                print(f"Processed {idx}/{total or '?'} frames | {idx/max(time.time()-t0,1e-6):.1f} FPS")
    finally:
        cap.release(); writer.release()
        if imshow: cv2.destroyAllWindows()
    print(f"Saved {idx} frames to {target_path}")
