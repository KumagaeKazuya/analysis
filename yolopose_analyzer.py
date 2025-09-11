# YOLO11-Pose + ID追跡

from ultralytics import YOLO
import cv2
import os
import pandas as pd
from utils.visualization import draw_detections
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

def results_to_xyxy(results):
    """YOLO結果をByteTrack用の (x1, y1, x2, y2, score) に変換"""
    dets = []
    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            x1, y1, x2, y2 = map(float, box.tolist())
            dets.append([x1, y1, x2, y2, float(conf)])
    return np.array(dets)

def analyze_frames_with_tracking(frame_dir, result_dir, model_path="models/yolo11n-pose.pt"):
    os.makedirs(result_dir, exist_ok=True)
    model = YOLO(model_path)
    tracker = BYTETracker(track_thresh=0.5)

    all_detections = []

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    for f_idx, f in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, f)
        frame = cv2.imread(frame_path)

        results = model(frame)
        dets = results_to_xyxy(results)  # ByteTrack用

        online_targets = tracker.update(dets, [frame.shape[0], frame.shape[1]], exp=None)

        # CSV用に保存
        for t in online_targets:
            x1, y1, x2, y2 = t.tlbr
            track_id = t.track_id
            conf = t.score
            all_detections.append([f, track_id, x1, y1, x2, y2, conf, "person"])

        # 可視化
        vis_frame = draw_detections(frame, results, online_targets)
        cv2.imwrite(os.path.join(result_dir, f), vis_frame)

    df = pd.DataFrame(all_detections,
                    columns=["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"])
    df.to_csv("outputs/logs/detections_id.csv", index=False)
    print(f"✅ ID付き検出ログを outputs/logs/detections_id.csv に保存")
