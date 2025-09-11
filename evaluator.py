# 疑似ラベル評価 / 正規 mAP

from ultralytics import YOLO
from sklearn.metrics import average_precision_score
import cv2
import os

def pseudo_label_evaluation(video_path, det_model_path="models/yolo11n.pt", pose_model_path="models/yolo11n-pose.pt"):
    det_model = YOLO(det_model_path)
    pose_model = YOLO(pose_model_path)

    cap = cv2.VideoCapture(video_path)
    y_true, y_pred = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_results = det_model(frame, classes=[0], verbose=False)
        pose_results = pose_model(frame, verbose=False)

        y_true.append(1 if len(det_results[0].boxes) > 0 else 0)
        y_pred.append(1 if len(pose_results[0].boxes) > 0 else 0)

    cap.release()
    ap = average_precision_score(y_true, y_pred)
    print(f"[{os.path.basename(video_path)}] AP (pseudo): {ap:.3f}")
    return ap

def official_map_evaluation(model_path="models/yolo11n-pose.pt",
                            data_yaml="annotations/coco_val.yaml",
                            imgsz=640):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz)
    print("===== mAP / OKS 評価結果 =====")
    print(results)
    return results
