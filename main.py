import os
from evaluator import pseudo_label_evaluation, official_map_evaluation
from yolopose_analyzer import analyze_frames_with_tracking
from frame_sampler import sample_frames

video_dir = "videos"
data_yaml = "annotations/coco_val.yaml"
MODEL_DIR = "models"

DET_MODEL = f"{MODEL_DIR}/yolo11n.pt"
POSE_MODEL = f"{MODEL_DIR}/yolo11n-pose.pt"

# 出力ディレクトリ
FRAME_DIR = "outputs/frames"
RESULT_DIR = "outputs/results"

if os.path.exists(data_yaml):
    print("✅ GTあり：正規 mAP / OKS 評価を実行")
    official_map_evaluation(
        model_path=POSE_MODEL,
        data_yaml=data_yaml
    )
else:
    print("⚠️ GTなし：疑似ラベル評価 + ID付き検出を実行")
    for f in os.listdir(video_dir):
        if not f.endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, f)
        # フレーム抽出
        sample_frames(video_path, FRAME_DIR, interval_sec=2)
        # 推論 + ID付きログ
        analyze_frames_with_tracking(FRAME_DIR, RESULT_DIR, model_path=POSE_MODEL)
        # 疑似ラベル評価
        pseudo_label_evaluation(video_path, det_model_path=DET_MODEL, pose_model_path=POSE_MODEL)
