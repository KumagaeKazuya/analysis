"""
メイン分析関数モジュール（XLargeモデル確実使用版・実際使用モデル完全ログ対応版・キーポイント処理修正版）
"""

import os
import cv2
import csv
import time
import numpy as np
import logging
import gc
import torch
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from utils.camera_calibration import undistort_with_json

# 🔧 統一エラーハンドラーからインポート
from utils.error_handler import (
    ModelInitializationError,
    ResourceExhaustionError,
    VideoProcessingError,
    ResponseBuilder,
    handle_errors,
    ErrorContext,
    ErrorCategory
)

from .system import safe_model_initialization
from .validation import validate_frame_directory
from .memory import MemoryEfficientProcessor
from .visualization import draw_detections_ultralytics

logger = logging.getLogger(__name__)


def log_actual_model_usage(model, requested_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """実際に使用されているモデルの詳細をログに記録"""
    try:
        model_info = {
            "requested_path": requested_path,
            "actual_model_file": None,
            "model_size_mb": 0,
            "parameter_count": 0,
            "estimated_type": "UNKNOWN",
            "verification_passed": False,
            "file_exists": False
        }
        
        # 1. モデルファイルの特定
        actual_file = None
        if hasattr(model, 'ckpt_path') and model.ckpt_path:
            actual_file = str(model.ckpt_path)
        elif hasattr(model, 'model_path') and model.model_path:
            actual_file = str(model.model_path)
        elif hasattr(model, 'cfg') and hasattr(model.cfg, 'model_path'):
            actual_file = str(model.cfg.model_path)
        else:
            # フォールバック: 要求されたパスで確認
            actual_file = requested_path
        
        model_info["actual_model_file"] = actual_file
        
        # 2. ファイル存在確認とサイズ
        if actual_file and Path(actual_file).exists():
            model_info["file_exists"] = True
            size_bytes = Path(actual_file).stat().st_size
            model_info["model_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        else:
            logger.warning(f"⚠️ モデルファイルが見つかりません: {actual_file}")
        
        # 3. パラメータ数確認（最も確実な方法）
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                total_params = sum(p.numel() for p in model.model.parameters())
                model_info["parameter_count"] = total_params
                
                # パラメータ数による実際のモデルタイプ判定（YOLO11の実際の値）
                if total_params < 3_200_000:  # ~3.2M
                    model_info["estimated_type"] = "NANO"
                elif total_params < 12_000_000:  # ~11M
                    model_info["estimated_type"] = "SMALL"
                elif total_params < 26_000_000:  # ~25M
                    model_info["estimated_type"] = "MEDIUM"
                elif total_params < 44_000_000:  # ~43M
                    model_info["estimated_type"] = "LARGE"
                else:  # ~57M+
                    model_info["estimated_type"] = "XLARGE"
        except Exception as param_error:
            logger.warning(f"⚠️ パラメータ数取得エラー: {param_error}")
        
        # 4. 要求vs実際の検証
        requested_type = "UNKNOWN"
        if "11x" in requested_path or "yolo11x" in requested_path:
            requested_type = "XLARGE"
        elif "11l" in requested_path:
            requested_type = "LARGE"
        elif "11m" in requested_path:
            requested_type = "MEDIUM"
        elif "11s" in requested_path:
            requested_type = "SMALL"
        elif "11n" in requested_path:
            requested_type = "NANO"
        
        model_info["requested_type"] = requested_type
        model_info["verification_passed"] = (requested_type == model_info["estimated_type"])
        
        # 5. 詳細ログ出力
        logger.info("🔍 ========== 実際使用モデル検証結果 ==========")
        logger.info(f"📝 要求モデルパス: {requested_path}")
        logger.info(f"📂 実際モデルファイル: {model_info['actual_model_file']}")
        logger.info(f"📊 ファイルサイズ: {model_info['model_size_mb']}MB")
        logger.info(f"🔢 パラメータ数: {model_info['parameter_count']:,}")
        logger.info(f"🎯 要求モデルタイプ: {requested_type}")
        logger.info(f"🎯 実際モデルタイプ: {model_info['estimated_type']}")
        logger.info(f"📁 ファイル存在: {'✅' if model_info['file_exists'] else '❌'}")
        
        if model_info["verification_passed"]:
            logger.info("✅ モデル検証: 要求通りのモデルが使用されています")
        else:
            logger.error("❌ モデル不一致: 要求と異なるモデルが使用されています!")
            logger.error(f"   ⚠️  期待: {requested_type} → 実際: {model_info['estimated_type']}")
            
            # フォールバックが発生したことを明確にログ
            if requested_type == "XLARGE" and model_info['estimated_type'] == "NANO":
                logger.error("🔴 重大: XLARGEを要求したがNANOが使用されています!")
                logger.error("🔴 これにより検出精度が大幅に低下している可能性があります")
        
        logger.info("🔍 ============================================")
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ モデル使用ログ記録エラー: {e}")
        return {
            "requested_path": requested_path,
            "error": str(e),
            "verification_passed": False
        }


def load_model_with_verification(model_path: str, force_exact_model: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """確実なモデルロード + 使用モデル検証"""
    logger.info(f"🚀 モデルロード開始: {model_path}")
    logger.info(f"🎯 厳密モード: {'有効' if force_exact_model else '無効'}")
    
    # 1. モデルファイル存在確認
    if not Path(model_path).exists():
        if force_exact_model:
            error_msg = f"指定モデルが存在しません: {model_path}"
            logger.error(f"❌ {error_msg}")
            
            # 利用可能モデルをログ出力
            models_dir = Path(model_path).parent
            if models_dir.exists():
                available_models = list(models_dir.glob("*.pt"))
                if available_models:
                    logger.info(f"📂 利用可能モデル:")
                    for model_file in available_models:
                        size_mb = model_file.stat().st_size / (1024 * 1024)
                        logger.info(f"  {model_file.name} ({size_mb:.1f}MB)")
                else:
                    logger.error(f"❌ {models_dir} にモデルファイルが見つかりません")
            
            raise FileNotFoundError(error_msg)
        else:
            logger.warning(f"⚠️ 指定モデルが見つかりません: {model_path}")
            logger.info("🔄 フォールバック処理を開始...")
    
    # 2. モデルロード実行
    try:
        from ultralytics import YOLO
        
        start_time = time.time()
        
        # システムメモリ確認
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024 ** 3)
        logger.info(f"🧠 利用可能メモリ: {available_memory_gb:.1f}GB")
        
        # XLargeモデルには最低4GB必要
        if "11x" in model_path and available_memory_gb < 4.0:
            logger.warning(f"⚠️ XLargeモデルには4GB以上推奨（現在: {available_memory_gb:.1f}GB）")
        
        logger.info(f"📥 モデルロード中: {model_path}")
        model = YOLO(model_path)
        load_time = time.time() - start_time
        
        logger.info(f"✅ モデルロード完了: {load_time:.2f}秒")
        
        # 3. 実際使用モデルの検証とログ記録
        verification_result = log_actual_model_usage(model, model_path, logger)
        verification_result["load_time_seconds"] = load_time
        verification_result["available_memory_gb"] = available_memory_gb
        
        return model, verification_result
        
    except Exception as e:
        logger.error(f"❌ モデルロードエラー: {e}")
        if force_exact_model:
            raise
        else:
            logger.warning("🔄 緊急フォールバックを試行...")
            # 最後の手段: NANOモデル
            fallback_models = [
                "models/yolo/yolo11n-pose.pt",
                "models/yolo11n-pose.pt", 
                "yolo11n-pose.pt"
            ]
            
            for fallback_path in fallback_models:
                try:
                    logger.warning(f"📥 緊急フォールバック試行: {fallback_path}")
                    from ultralytics import YOLO
                    model = YOLO(fallback_path)
                    verification_result = log_actual_model_usage(model, fallback_path, logger)
                    verification_result["emergency_fallback"] = True
                    verification_result["original_requested"] = model_path
                    return model, verification_result
                except Exception as fallback_error:
                    logger.warning(f"⚠️ フォールバック失敗: {fallback_path} - {fallback_error}")
                    continue
            
            # 全てのフォールバックが失敗
            raise ModelInitializationError(f"全モデルロードに失敗: {e}")


def create_detection_visualization(frame, results, output_path: str, frame_file: str, config: Dict[str, Any]) -> bool:
    """検出枠付き画像生成（描画専用）"""
    try:
        if frame is None or results is None:
            logger.warning(f"フレームまたは結果が無効: {frame_file}")
            return False

        vis_frame = frame.copy()
        detection_count = 0
        
        # 検出結果を描画
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                
                # クラス情報
                if r.boxes.cls is not None:
                    classes = r.boxes.cls.cpu().numpy()
                else:
                    classes = [0] * len(boxes)
                
                # トラッキングID
                if r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))
                
                # 🎯 キーポイント描画（データ抽出ではなく描画のみ）
                if r.keypoints is not None:
                    keypoints = r.keypoints.data.cpu().numpy()
                    # キーポイントを画像に描画
                    for i, kpts in enumerate(keypoints):
                        if i < len(boxes) and confidences[i] >= config.get("confidence_threshold", 0.3):
                            # 17点キーポイントを描画
                            for j, (x, y, conf) in enumerate(kpts):
                                if conf > 0.5:  # 可視性閾値
                                    cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                
                # バウンディングボックス描画
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
                    if conf < config.get("confidence_threshold", 0.3):
                        continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    track_id = track_ids[i] if i < len(track_ids) else i
                    
                    # 検出枠描画
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # ラベル作成
                    class_name = config.get("class_names", {}).get(int(cls_id), "person")
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    
                    # ラベル描画
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    detection_count += 1
        
        # フレーム情報描画
        frame_info = f"Frame: {frame_file} | Detections: {detection_count}"
        cv2.putText(vis_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 🚨 重要: 検出枠付き画像保存時のログ
        success = cv2.imwrite(output_path, vis_frame)
        if success:
            logger.debug(f"✅ 検出枠付き画像保存: {output_path}")
        else:
            logger.error(f"❌ 検出枠付き画像保存失敗: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 可視化生成エラー {frame_file}: {e}")
        return False


@handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING, suppress_exceptions=False)
def analyze_frames_with_tracking_memory_efficient(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo/yolo11x-pose.pt",
    config: Optional[Dict[str, Any]] = None,
    pre_loaded_model: Optional[Any] = None,
    model_verification: Optional[Dict[str, Any]] = None,
    force_exact_model: bool = True
) -> Dict[str, Any]:
    """メモリ効率的なフレーム解析（歪み補正一貫適用版）"""
    from utils.camera_calibration import undistort_with_json

    with ErrorContext("XLargeモデル確実使用フレーム解析処理", logger=logger, raise_on_error=True) as ctx:
        # モデルパス・設定初期化（既存コード）
        logger.info("🎯 ========== モデル使用開始 ==========")
        logger.info(f"📝 要求モデルパス: {model_path}")
        logger.info(f"🔧 厳密モード: {'有効' if force_exact_model else '無効'}")
        logger.info("🎯 ====================================")

        if config is None:
            config = {
                "confidence_threshold": 0.3,
                "tracking_config": "bytetrack.yaml",
                "save_visualizations": True,
                "save_detection_frames": True,
                "batch_size": 16,
                "max_memory_gb": 6.0,
                "streaming_output": True,
                "device": "auto",
                "class_names": {0: "person"},
                "force_pose_task": True,
                "keypoint_processing_enabled": True
            }
        else:
            config = config.copy()
            config.setdefault("tracking_config", "bytetrack.yaml")
            config.setdefault("save_visualizations", True)
            config.setdefault("save_detection_frames", True)
            config.setdefault("class_names", {0: "person"})
            config.setdefault("force_pose_task", True)
            config.setdefault("keypoint_processing_enabled", True)
            if "11x" in model_path:
                config.setdefault("batch_size", 16)
                config.setdefault("max_memory_gb", 6.0)

        if not config.get("tracking_config") or config.get("tracking_config") == "":
            config["tracking_config"] = "bytetrack.yaml"
            logger.info("🔧 tracker設定をデフォルトに修正")

        os.makedirs(result_dir, exist_ok=True)
        vis_dir = os.path.join(result_dir, "visualized_frames")
        os.makedirs(vis_dir, exist_ok=True)
        processor = MemoryEfficientProcessor(config)

        ctx.add_info("result_dir", result_dir)
        ctx.add_info("vis_dir", vis_dir)
        ctx.add_info("batch_size", config.get("batch_size", 16))
        ctx.add_info("save_visualizations", config.get("save_visualizations", True))
        ctx.add_info("force_exact_model", force_exact_model)

        try:
            # モデル初期化
            if pre_loaded_model:
                logger.info("✅ 事前ロード済みモデルを使用")
                model = pre_loaded_model
                verification_info = model_verification or {}
            else:
                logger.info("🔄 新規モデルロード")
                model, verification_info = load_model_with_verification(model_path, force_exact_model)

            logger.info("🎯 ========== 最終使用モデル確認 ==========")
            if verification_info.get("verification_passed"):
                logger.info(f"✅ 要求通りのモデルで処理実行")
                logger.info(f"   モデルタイプ: {verification_info.get('estimated_type', 'UNKNOWN')}")
                logger.info(f"   パラメータ数: {verification_info.get('parameter_count', 0):,}")
                logger.info(f"   ファイルサイズ: {verification_info.get('model_size_mb', 0)}MB")
            else:
                logger.error(f"❌ フォールバックモデルで処理実行")
                logger.error(f"   要求タイプ: {verification_info.get('requested_type', 'UNKNOWN')}")
                logger.error(f"   実際タイプ: {verification_info.get('estimated_type', 'UNKNOWN')}")
                if verification_info.get("emergency_fallback"):
                    logger.error("🚨 緊急フォールバックが発生しました!")
                    logger.error(f"   元要求: {verification_info.get('original_requested', '不明')}")
            logger.info("🎯 ==========================================")
            ctx.add_info("model_verification", verification_info)

            # フレームディレクトリ検証
            frame_validation = validate_frame_directory(frame_dir)
            if not frame_validation.get("success", False):
                raise VideoProcessingError(
                    "フレームディレクトリの検証に失敗しました",
                    details=frame_validation.get("error", {})
                )

            frame_data = frame_validation["data"]
            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            total_frames = len(frame_files)
            ctx.add_info("total_frames", total_frames)

            logger.info(f"📊 処理対象: {total_frames}フレーム ({frame_data['total_size_mb']:.1f}MB)")
            logger.info(f"🎨 可視化保存: {config.get('save_visualizations', True)}")
            logger.info(f"📁 可視化ディレクトリ: {vis_dir}")
            logger.info(f"🎯 使用モデル: {verification_info.get('estimated_type', 'UNKNOWN')}")

            # CSV準備
            csv_path = os.path.join(result_dir, "detections_streaming.csv")
            base_headers = ["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"]
            coco_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
            keypoint_headers = []
            for name in coco_names:
                keypoint_headers.extend([f'{name}_x', f'{name}_y', f'{name}_conf'])
            full_headers = base_headers + keypoint_headers

            stats = {
                "total_frames": total_frames,
                "processed_frames": 0,
                "successful_frames": 0,
                "failed_frames": 0,
                "total_detections": 0,
                "unique_ids": set(),
                "memory_peaks": [],
                "batch_times": [],
                "visualization_stats": {
                    "generated": 0,
                    "failed": 0,
                    "skipped": 0
                },
                "keypoint_stats": {
                    "frames_with_keypoints": 0,
                    "total_keypoints_detected": 0,
                    "keypoints_per_person": []
                },
                "model_verification": verification_info
            }

            batch_size = config.get("batch_size", 16)
            if verification_info.get('estimated_type') == 'XLARGE':
                logger.info("🚀 XLARGEモデルでの高精度処理を開始します")
                logger.info(f"   予想処理時間: 通常の2-3倍")
                logger.info(f"   予想メモリ使用量: 4-8GB")

            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(full_headers)
                logger.info(f"📋 CSV出力準備完了:")
                logger.info(f"   基本列: {len(base_headers)}個")
                logger.info(f"   キーポイント列: {len(keypoint_headers)}個")
                logger.info(f"   総列数: {len(full_headers)}個")

                try:
                    for batch_start in range(0, total_frames, batch_size):
                        batch_end = min(batch_start + batch_size, total_frames)
                        batch_files = frame_files[batch_start:batch_end]
                        batch_start_time = time.time()
                        batch_detections = []

                        logger.info(f"📦 バッチ処理 {batch_start//batch_size + 1}/{(total_frames-1)//batch_size + 1}: "
                                    f"{len(batch_files)}フレーム (モデル: {verification_info.get('estimated_type', 'UNKNOWN')})")

                        for frame_file in batch_files:
                            frame_path = os.path.join(frame_dir, frame_file)
                            try:
                                if processor.check_memory_threshold():
                                    logger.warning("⚠️ メモリ使用量が閾値を超過。クリーンアップを実行...")
                                    processor.force_memory_cleanup()

                                tracker_config = config.get("tracking_config")
                                if not tracker_config or tracker_config == "":
                                    tracker_config = "bytetrack.yaml"
                                    logger.debug(f"🔧 tracker設定をデフォルトに修正: {tracker_config}")

                                # 🎯 歪み補正を推論前に適用
                                frame = cv2.imread(frame_path)
                                if frame is None:
                                    logger.warning(f"⚠️ フレーム読み込み失敗: {frame_file}")
                                    stats["visualization_stats"]["failed"] += 1
                                    continue
                                frame = undistort_with_json(frame, calib_path="configs/camera_params.json")

                                # 🎯 推論実行（画像データを直接渡す）
                                inference_params = {
                                    "source": frame,
                                    "persist": True,
                                    "tracker": tracker_config,
                                    "conf": config.get("confidence_threshold", 0.3),
                                    "task": "pose",
                                    "verbose": False,
                                    "save": False,
                                    "show": False
                                }
                                if config.get("force_pose_task", True):
                                    inference_params["task"] = "pose"

                                logger.info(f"🎯 推論パラメータ: tracker={tracker_config}, task=pose")
                                results = model.track(**inference_params)

                                frame_detections = 0
                                frame_has_keypoints = False

                                for r in results:
                                    if r.boxes is not None:
                                        boxes = r.boxes.xyxy.cpu().numpy()
                                        confidences = r.boxes.conf.cpu().numpy()
                                        if r.boxes.id is not None:
                                            track_ids = r.boxes.id.cpu().numpy().astype(int)
                                        else:
                                            track_ids = list(range(len(boxes)))
                                        if r.keypoints is not None:
                                            try:
                                                keypoints = r.keypoints.data.cpu().numpy()
                                                frame_has_keypoints = True
                                                logger.debug(f"🦴 フレーム {frame_file}: キーポイント検出 {keypoints.shape}")
                                                for i, (box, conf, kpts) in enumerate(zip(boxes, confidences, keypoints)):
                                                    if conf < config.get("confidence_threshold", 0.3):
                                                        continue
                                                    track_id = track_ids[i] if i < len(track_ids) else i
                                                    x1, y1, x2, y2 = box
                                                    detection_row = [
                                                        frame_file, int(track_id),
                                                        float(x1), float(y1), float(x2), float(y2),
                                                        float(conf), "person"
                                                    ]
                                                    valid_keypoints = 0
                                                    for j, name in enumerate(coco_names):
                                                        if j < len(kpts):
                                                            kpt_x, kpt_y, kpt_conf = kpts[j]
                                                            if np.isnan(kpt_x) or np.isinf(kpt_x):
                                                                kpt_x = 0.0
                                                            if np.isnan(kpt_y) or np.isinf(kpt_y):
                                                                kpt_y = 0.0
                                                            if np.isnan(kpt_conf) or np.isinf(kpt_conf):
                                                                kpt_conf = 0.0
                                                            detection_row.extend([
                                                                float(kpt_x),
                                                                float(kpt_y),
                                                                float(kpt_conf)
                                                            ])
                                                            if kpt_conf > 0.5:
                                                                valid_keypoints += 1
                                                        else:
                                                            detection_row.extend([0.0, 0.0, 0.0])
                                                    if len(detection_row) != len(full_headers):
                                                        logger.error(f"❌ 列数不一致: 期待{len(full_headers)}, 実際{len(detection_row)}")
                                                        logger.error(f"   フレーム: {frame_file}")
                                                        logger.error(f"   検出データ: {detection_row[:10]}...")
                                                        continue
                                                    batch_detections.append(detection_row)
                                                    frame_detections += 1
                                                    stats["unique_ids"].add(track_id)
                                                    stats["keypoint_stats"]["total_keypoints_detected"] += valid_keypoints
                                                    stats["keypoint_stats"]["keypoints_per_person"].append(valid_keypoints)
                                            except Exception as keypoint_error:
                                                logger.error(f"❌ キーポイント処理エラー {frame_file}: {keypoint_error}")
                                                frame_has_keypoints = False
                                        else:
                                            logger.warning(f"⚠️ フレーム {frame_file}: キーポイント未検出")
                                            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                                                if conf < config.get("confidence_threshold", 0.3):
                                                    continue
                                                track_id = track_ids[i] if i < len(track_ids) else i
                                                x1, y1, x2, y2 = box
                                                detection_row = [
                                                    frame_file, int(track_id),
                                                    float(x1), float(y1), float(x2), float(y2),
                                                    float(conf), "person"
                                                ]
                                                for _ in range(len(keypoint_headers)):
                                                    detection_row.append(0.0)
                                                if len(detection_row) != len(full_headers):
                                                    logger.error(f"❌ ゼロパディング列数不一致: 期待{len(full_headers)}, 実際{len(detection_row)}")
                                                    continue
                                                batch_detections.append(detection_row)
                                                frame_detections += 1
                                                stats["unique_ids"].add(track_id)

                                if frame_has_keypoints:
                                    stats["keypoint_stats"]["frames_with_keypoints"] += 1
                                stats["total_detections"] += frame_detections
                                stats["successful_frames"] += 1

                                # 🎨 検出枠付き画像生成
                                if config.get("save_visualizations", True):
                                    try:
                                        # 補正済みframeをそのまま可視化に渡す
                                        vis_filename = f"vis_{frame_file}"
                                        vis_output_path = os.path.join(vis_dir, vis_filename)
                                        success = create_detection_visualization(
                                            frame, results, vis_output_path, frame_file, config
                                        )
                                        if success:
                                            stats["visualization_stats"]["generated"] += 1
                                            logger.debug(f"✅ 可視化生成: {vis_filename}")
                                        else:
                                            stats["visualization_stats"]["failed"] += 1
                                    except Exception as vis_error:
                                        logger.warning(f"❌ 可視化エラー {frame_file}: {vis_error}")
                                        stats["visualization_stats"]["failed"] += 1
                                else:
                                    stats["visualization_stats"]["skipped"] += 1

                                del results

                            except Exception as frame_error:
                                logger.error(f"❌ フレーム処理エラー {frame_file}: {frame_error}", exc_info=True)
                                stats["failed_frames"] += 1
                                continue

                            stats["processed_frames"] += 1

                        if batch_detections:
                            try:
                                csv_writer.writerows(batch_detections)
                                csv_file.flush()
                                batch_keypoint_count = sum(1 for row in batch_detections
                                                           if any(row[8+i] != 0.0 for i in range(0, len(keypoint_headers), 3)))
                                logger.debug(f"📊 バッチCSV書き込み完了: 検出{len(batch_detections)}個, キーポイント付き{batch_keypoint_count}個")
                            except Exception as csv_error:
                                logger.error(f"❌ CSV書き込みエラー: {csv_error}")
                                raise

                        del batch_detections
                        processor.force_memory_cleanup()
                        batch_time = time.time() - batch_start_time
                        current_memory = processor.get_memory_usage()
                        stats["batch_times"].append(batch_time)
                        stats["memory_peaks"].append(current_memory)
                        progress = (batch_end / total_frames) * 100
                        vis_progress = stats["visualization_stats"]["generated"]
                        keypoint_frames = stats["keypoint_stats"]["frames_with_keypoints"]
                        logger.info(f"📊 進捗: {progress:.1f}% (メモリ: {current_memory:.2f}GB, "
                                    f"バッチ時間: {batch_time:.1f}s, 可視化: {vis_progress}個, "
                                    f"モデル: {verification_info.get('estimated_type', 'UNKNOWN')})")

                except Exception as e:
                    logger.error(f"❌ バッチ処理エラー: {e}", exc_info=True)
                    raise VideoProcessingError(f"バッチ処理に失敗しました: {e}", original_exception=e)

            stats["unique_ids"] = len(stats["unique_ids"])
            stats["success_rate"] = stats["successful_frames"] / total_frames if total_frames > 0 else 0
            stats["avg_batch_time"] = np.mean(stats["batch_times"]) if stats["batch_times"] else 0
            stats["peak_memory_gb"] = max(stats["memory_peaks"]) if stats["memory_peaks"] else 0
            keypoint_stats = stats["keypoint_stats"]
            keypoint_frame_rate = keypoint_stats["frames_with_keypoints"] / total_frames if total_frames > 0 else 0
            avg_keypoints_per_person = np.mean(keypoint_stats["keypoints_per_person"]) if keypoint_stats["keypoints_per_person"] else 0
            vis_stats = stats["visualization_stats"]
            vis_success_rate = vis_stats["generated"] / total_frames if total_frames > 0 else 0

            ctx.add_info("total_detections", stats["total_detections"])
            ctx.add_info("success_rate", stats["success_rate"])
            ctx.add_info("peak_memory_gb", stats["peak_memory_gb"])
            ctx.add_info("visualization_generated", vis_stats["generated"])
            ctx.add_info("visualization_success_rate", vis_success_rate)
            ctx.add_info("keypoint_frame_rate", keypoint_frame_rate)
            ctx.add_info("avg_keypoints_per_person", avg_keypoints_per_person)
            ctx.add_info("model_type_used", verification_info.get('estimated_type', 'UNKNOWN'))

            logger.info("🎯 ========== 処理完了サマリー ==========")
            logger.info(f"📊 処理完了統計:")
            logger.info(f"  ✅ 成功率: {stats['success_rate']:.1%}")
            logger.info(f"  🎯 総検出数: {stats['total_detections']}")
            logger.info(f"  👥 ユニークID: {stats['unique_ids']}")
            logger.info(f"  🧠 ピークメモリ: {stats['peak_memory_gb']:.2f}GB")
            logger.info(f"  ⏱️  平均バッチ時間: {stats['avg_batch_time']:.1f}s")
            logger.info(f"  🎨 可視化生成: {vis_stats['generated']}個 (成功率: {vis_success_rate:.1%})")
            logger.info(f"  📁 可視化保存先: {vis_dir}")
            logger.info(f"🦴 キーポイント統計:")
            logger.info(f"  📊 キーポイント付きフレーム: {keypoint_stats['frames_with_keypoints']} ({keypoint_frame_rate:.1%})")
            logger.info(f"  🎯 総キーポイント検出数: {keypoint_stats['total_keypoints_detected']}")
            logger.info(f"  👤 平均キーポイント/人: {avg_keypoints_per_person:.1f}")
            logger.info(f"  📋 出力CSV列数: {len(full_headers)} (基本: {len(base_headers)}, キーポイント: {len(keypoint_headers)})")
            logger.info(f"🎯 使用モデル最終確認:")
            logger.info(f"  📝 要求: {verification_info.get('requested_type', 'UNKNOWN')}")
            logger.info(f"  ✅ 実際: {verification_info.get('estimated_type', 'UNKNOWN')}")
            logger.info(f"  🔧 検証結果: {'成功' if verification_info.get('verification_passed') else 'フォールバック'}")
            logger.info("🎯 ====================================")

            return ResponseBuilder.success(
                data={
                    "csv_path": csv_path,
                    "visualization_dir": vis_dir,
                    "processing_stats": stats,
                    "config_used": config,
                    "model_path": model_path,
                    "model_verification": verification_info,
                    "result_dir": result_dir,
                    "memory_efficient": True,
                    "visualization_enabled": True,
                    "keypoint_processing_enabled": True,
                    "csv_columns": len(full_headers),
                    "keypoint_columns": len(keypoint_headers),
                    "xlarge_model_verified": verification_info.get('verification_passed', False)
                },
                message=f"フレーム解析完了（{verification_info.get('estimated_type', 'UNKNOWN')}モデル使用・キーポイント処理付き・検出枠付き画像生成付き）"
            )

        except ModelInitializationError as e:
            logger.error(f"❌ モデル初期化失敗: {e}")
            return ResponseBuilder.error(
                message="モデル初期化に失敗しました",
                details={"error": str(e), "model_path": model_path}
            )

        except ResourceExhaustionError as e:
            logger.error(f"❌ リソース不足: {e}")
            return ResponseBuilder.error(
                message="システムリソースが不足しています",
                details={"error": str(e)}
            )

        except VideoProcessingError as e:
            logger.error(f"❌ 動画処理エラー: {e}")
            return ResponseBuilder.error(
                message="動画処理中にエラーが発生しました",
                details={"error": str(e)}
            )

        except Exception as e:
            logger.error(f"❌ 予期しないエラー: {e}", exc_info=True)
            return ResponseBuilder.error(
                message="予期しないエラーが発生しました",
                details={"error": str(e)}
            )


@handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
def analyze_frames_with_tracking_enhanced(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo/yolo11x-pose.pt",
    config: Optional[Dict[str, Any]] = None,
    force_exact_model: bool = True
) -> Dict[str, Any]:
    """拡張版フレーム解析（完全修正版）"""
    with ErrorContext("XLarge拡張フレーム解析", logger=logger) as ctx:
        ctx.add_info("frame_dir", frame_dir)
        ctx.add_info("model_path", model_path)
        ctx.add_info("force_exact_model", force_exact_model)

        # 設定からモデルパス取得
        final_model_path = model_path
        if config:
            config_model_path = config.get("models", {}).get("pose")
            if config_model_path:
                final_model_path = config_model_path
                logger.info(f"📋 設定ファイルからモデルパス取得: {final_model_path}")
        
        logger.info(f"🎯 最終決定モデルパス: {final_model_path}")
        
        # タイル推論の判定
        if config and config.get("tile_inference", {}).get("enabled", False):
            logger.info("🔲 タイル推論モードで実行")
            try:
                from .tile_inference import analyze_frames_with_tile_inference
                return analyze_frames_with_tile_inference(
                    frame_dir, result_dir, final_model_path, config, force_exact_model
                )
            except ImportError:
                logger.warning("⚠️ タイル推論モジュールが見つかりません。通常推論に切り替えます")
        
        # 確実なモデルロード
        try:
            model, verification_result = load_model_with_verification(
                final_model_path, force_exact_model
            )
            
            ctx.add_info("model_verification", verification_result)
            
            # 通常推論実行
            return analyze_frames_with_tracking_memory_efficient(
                frame_dir, result_dir, final_model_path, config, 
                pre_loaded_model=model,
                model_verification=verification_result,
                force_exact_model=force_exact_model
            )
            
        except Exception as e:
            logger.error(f"❌ モデルロードまたは処理エラー: {e}")
            if force_exact_model:
                return ResponseBuilder.error(
                    message=f"指定モデル（{final_model_path}）の処理に失敗しました",
                    details={"error": str(e), "model_path": final_model_path}
                )
            else:
                logger.warning("🔄 フォールバックモードで再試行...")
                return analyze_frames_with_tracking_memory_efficient(
                    frame_dir, result_dir, final_model_path, config,
                    force_exact_model=False
                )