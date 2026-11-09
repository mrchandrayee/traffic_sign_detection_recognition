
from __future__ import annotations
import cv2
from .detector import get_sign_color
class TrafficSignVisualizer:
    @staticmethod
    def draw_frame(frame, detections, classifications, track_ids=None):
        vis=frame.copy()
        for i,(box,conf) in enumerate(detections):
            x1,y1,x2,y2=map(int,box); tid=track_ids[i] if track_ids and i<len(track_ids) else -1
            cls=classifications[i] if i<len(classifications) else []
            label=cls[0][0] if cls else "Traffic sign"
            score=cls[0][1] if cls else conf
            color=get_sign_color(label)
            cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            cv2.putText(vis,f"{label} {score:.2f} ID:{tid}",(x1,max(18,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,color,2,cv2.LINE_AA)
        return vis
    @staticmethod
    def draw_hud_overlay(frame, events):
        h,w=frame.shape[:2]; panel_w=min(245,w//2); x0=w-panel_w
        overlay=frame.copy()
        cv2.rectangle(overlay,(x0,0),(w,h),(12,16,24),-1)
        cv2.addWeighted(overlay,.78,frame,.22,0,frame)
        cv2.putText(frame,"TRAFFIC SIGN HUD",(x0+10,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        for i,e in enumerate(events[:10]):
            y=48+i*48
            cv2.rectangle(frame,(x0+7,y),(w-7,y+39),(25,25,25),-1)
            cv2.putText(frame,str(e.get("class_name","Unknown"))[:25],(x0+13,y+16),
                        cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            cv2.putText(frame,f"{e.get('vn_name','')}  {e.get('cls_conf',0):.2f}",(x0+13,y+31),
                        cv2.FONT_HERSHEY_SIMPLEX,.30,(0,210,255),1)
        return frame
