# YOLO11-Pose + ID追跡 (Ultralytics組み込みトラッカー使用)
# 修正版: エラーハンドリングとログ出力を強化

from ultralytics import YOLO
import cv2
import os
import pandas as pd
from utils.visualization import draw_detections_ultralytics
import numpy as np
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_frames_with_tracking(frame_dir, result_dir, model_path="models/yolo11n-pose.pt", config=None):
    """
    フレーム解析とトラッキング実行

    Args:
        frame_dir: フレームディレクトリパス
        result_dir: 結果出力ディレクトリパス
        model_path: モデルファイルパス
        config: 設定辞書（新規追加）

    Returns:
        dict: 処理結果の詳細情報
    """
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # デフォルト設定
    if config is None:
        config = {
            "confidence_threshold": 0.3,
            "tracking_config": "bytetrack.yaml",
            "save_visualizations": True
        }

    try:
        model = YOLO(model_path)
        logger.info(f"モデル読み込み完了: {model_path}")
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return {"error": f"model_load_failed: {e}"}

    all_detections = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    if not frame_files:
        logger.warning(f"フレームファイルが見つかりません: {frame_dir}")
        return {"error": "no_frames_found", "frame_dir": frame_dir}

    logger.info(f"処理対象フレーム数: {len(frame_files)}")

    # 処理統計
    processing_stats = {
        "total_frames": len(frame_files),
        "successful_frames": 0,
        "failed_frames": 0,
        "total_detections": 0,
        "unique_ids": set()
    }

    for f_idx, f in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, f)

        try:
            # Ultralyticsの組み込みトラッカーを使用
            results = model.track(
                frame_path,
                persist=True,
                tracker=config["tracking_config"],
                conf=config["confidence_threshold"],
                verbose=False  # ログの冗長性を抑制
            )

            # トラッキング結果の処理
            frame_detections = 0
            for r in results:
                if r.boxes is not None and r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    track_ids = r.boxes.id.cpu().numpy().astype(int)
                    confidences = r.boxes.conf.cpu().numpy()

                    # CSV用に保存
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box
                        all_detections.append([f, track_id, x1, y1, x2, y2, conf, "person"])
                        frame_detections += 1
                        processing_stats["unique_ids"].add(track_id)

            processing_stats["total_detections"] += frame_detections
            processing_stats["successful_frames"] += 1

            # 可視化（設定で制御）
            if config["save_visualizations"]:
                try:
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        vis_frame = draw_detections_ultralytics(frame, results)
                        output_path = os.path.join(result_dir, f)
                        cv2.imwrite(output_path, vis_frame)
                except Exception as vis_error:
                    logger.warning(f"可視化エラー {f}: {vis_error}")

            # 進捗表示
            if (f_idx + 1) % 50 == 0:
                logger.info(f"処理進捗: {f_idx + 1}/{len(frame_files)} ({(f_idx + 1)/len(frame_files)*100:.1f}%)")

        except Exception as e:
            logger.error(f"フレーム処理エラー {f}: {e}")
            processing_stats["failed_frames"] += 1
            continue

    # 処理統計の最終化
    processing_stats["unique_ids"] = len(processing_stats["unique_ids"])
    processing_stats["success_rate"] = processing_stats["successful_frames"] / processing_stats["total_frames"]

    # CSV保存
    csv_path = "outputs/logs/detections_id.csv"
    if all_detections:
        df = pd.DataFrame(all_detections,
                        columns=["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"])
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ ID付き検出ログを {csv_path} に保存")

        # 追加統計情報をログに出力
        logger.info(f"📊 処理統計:")
        logger.info(f"  - 成功フレーム: {processing_stats['successful_frames']}/{processing_stats['total_frames']}")
        logger.info(f"  - 総検出数: {processing_stats['total_detections']}")
        logger.info(f"  - ユニークID数: {processing_stats['unique_ids']}")
        logger.info(f"  - 平均信頼度: {df['conf'].mean():.3f}")

    else:
        logger.warning("⚠️ トラッキング結果がありませんでした")

    # 戻り値を詳細化
    return {
        "csv_path": csv_path if all_detections else None,
        "processing_stats": processing_stats,
        "config_used": config,
        "model_path": model_path,
        "result_dir": result_dir
    }


# 🆕 新規追加: バッチ処理関数
def analyze_multiple_videos_with_tracking(video_list, output_base_dir, model_path="models/yolo11n-pose.pt", config=None):
    """
    複数動画の一括処理

    Args:
        video_list: 動画パスのリスト
        output_base_dir: 出力ベースディレクトリ
        model_path: モデルパス
        config: 設定辞書

    Returns:
        dict: 全動画の処理結果
    """
    results = {
        "total_videos": len(video_list),
        "successful_videos": 0,
        "failed_videos": 0,
        "video_results": []
    }

    for video_path in video_list:
        video_name = Path(video_path).stem
        logger.info(f"🎬 処理開始: {video_name}")

        try:
            # フレーム抽出（既存のframe_sampler.pyを使用）
            from frame_sampler import sample_frames
            frame_dir = os.path.join(output_base_dir, "frames", video_name)
            sample_frames(video_path, frame_dir, interval_sec=config.get("frame_interval", 2))

            # 解析実行
            result_dir = os.path.join(output_base_dir, "results", video_name)
            analysis_result = analyze_frames_with_tracking(frame_dir, result_dir, model_path, config)

            analysis_result["video_name"] = video_name
            analysis_result["video_path"] = str(video_path)
            results["video_results"].append(analysis_result)
            results["successful_videos"] += 1

            logger.info(f"✅ 完了: {video_name}")

        except Exception as e:
            logger.error(f"❌ 動画処理失敗 {video_name}: {e}")
            results["failed_videos"] += 1
            results["video_results"].append({
                "video_name": video_name,
                "video_path": str(video_path),
                "error": str(e)
            })

    logger.info(f"📊 全体処理完了: 成功{results['successful_videos']}/{results['total_videos']}")
    return results