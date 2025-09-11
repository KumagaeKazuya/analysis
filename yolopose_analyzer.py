# YOLO11-Pose + ID追跡 (Ultralytics組み込みトラッカー使用)

from ultralytics import YOLO
import cv2
import os
import pandas as pd
from utils.visualization import draw_detections_ultralytics
import numpy as np

def analyze_frames_with_tracking(frame_dir, result_dir, model_path="models/yolo11n-pose.pt"):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    model = YOLO(model_path)
    all_detections = []

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    for f_idx, f in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, f)

        # Ultralyticsの組み込みトラッカーを使用
        results = model.track(frame_path, persist=True, tracker="bytetrack.yaml")

        # トラッキング結果の処理
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy()

                # CSV用に保存
                for box, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = box
                    all_detections.append([f, track_id, x1, y1, x2, y2, conf, "person"])

        # 可視化
        frame = cv2.imread(frame_path)
        vis_frame = draw_detections_ultralytics(frame, results)
        cv2.imwrite(os.path.join(result_dir, f), vis_frame)

    # CSV保存
    if all_detections:
        df = pd.DataFrame(all_detections,
                        columns=["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"])
        df.to_csv("outputs/logs/detections_id.csv", index=False)
        print(f"✅ ID付き検出ログを outputs/logs/detections_id.csv に保存")
    else:
        print("⚠️ トラッキング結果がありませんでした")