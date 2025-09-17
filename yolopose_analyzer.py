from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import traceback
import torch
from typing import Dict, Any, Optional, List
import time
import psutil

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInitializationError(Exception):
    """モデル初期化エラー"""
    pass

class VideoProcessingError(Exception):
    """動画処理エラー"""
    pass

class ResourceExhaustionError(Exception):
    """リソース不足エラー"""
    pass

def check_system_resources() -> Dict[str, Any]:
    """システムリソースの確認"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_free_gb": disk.free / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        logger.warning(f"システムリソース確認エラー: {e}")
        return {"error": str(e)}

def validate_model_file(model_path: str) -> Dict[str, Any]:
    """モデルファイルの詳細検証"""
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }

    # ファイル存在確認
    if not os.path.exists(model_path):
        validation_result["errors"].append(f"モデルファイルが存在しません: {model_path}")

        # ダウンロード提案
        model_name = os.path.basename(model_path)
        if model_name.startswith("yolo11"):
            validation_result["suggestions"].append(
                f"以下のコマンドでモデルをダウンロードできます:\n"
                f"wget https://github.com/ultralytics/assets/releases/download/v8.0.0/{model_name}\n"
                f"または Python で: from ultralytics import YOLO; YOLO('{model_name}')"
            )
        return validation_result

    # ファイルサイズ確認
    try:
        file_size = os.path.getsize(model_path)
        if file_size < 1024:  # 1KB未満は異常
            validation_result["errors"].append(f"モデルファイルのサイズが異常に小さいです: {file_size} bytes")
            return validation_result
        elif file_size < 1024*1024:  # 1MB未満は警告
            validation_result["warnings"].append(f"モデルファイルが小さすぎる可能性があります: {file_size/1024:.1f} KB")
    except Exception as e:
        validation_result["errors"].append(f"ファイルサイズ確認エラー: {e}")
        return validation_result

    # 読み込みテスト
    try:
        test_model = YOLO(model_path)
        validation_result["valid"] = True
        validation_result["warnings"].append("モデル検証完了")
    except Exception as e:
        validation_result["errors"].append(f"モデル読み込みテストエラー: {e}")
        validation_result["suggestions"].append(
            "モデルファイルが破損している可能性があります。再ダウンロードを試してください。"
        )

    return validation_result

def safe_model_initialization(model_path: str, config: Dict[str, Any]) -> YOLO:
    """安全なモデル初期化"""
    # システムリソース確認
    resources = check_system_resources()
    if "error" not in resources:
        logger.info(f"システムリソース: メモリ {resources['memory_available_gb']:.1f}GB 利用可能")

        if resources["memory_available_gb"] < 2.0:
            logger.warning("利用可能メモリが2GB未満です。処理が遅くなる可能性があります。")

        if resources["gpu_available"]:
            logger.info(f"GPU利用可能: {resources['gpu_count']}個のデバイス")
        else:
            logger.info("GPU利用不可。CPUで処理します。")

    # モデルファイル検証
    validation = validate_model_file(model_path)
    if not validation["valid"]:
        error_msg = "\n".join(validation["errors"])
        suggestions = "\n".join(validation["suggestions"])
        raise ModelInitializationError(f"{error_msg}\n\n推奨対応:\n{suggestions}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.warning(warning)

    try:
        # モデル初期化
        model = YOLO(model_path)

        # デバイス設定
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)
        logger.info(f"モデル初期化完了: {model_path} on {device}")

        # GPU使用時の追加設定
        if device == "cuda":
            try:
                # GPU メモリ確認
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU メモリ: {gpu_memory:.1f}GB")

                # 半精度演算設定
                if config.get("use_half_precision", True):
                    model.half()
                    logger.info("半精度演算を有効化")

            except Exception as e:
                logger.warning(f"GPU設定警告: {e}")

        return model

    except Exception as e:
        logger.error(f"モデル初期化エラー: {e}")
        logger.error(f"詳細: {traceback.format_exc()}")
        raise ModelInitializationError(f"モデル初期化に失敗しました: {e}")

def validate_frame_directory(frame_dir: str) -> Dict[str, Any]:
    """フレームディレクトリの検証"""
    validation_result = {
        "valid": False,
        "frame_count": 0,
        "total_size_mb": 0,
        "errors": [],
        "warnings": []
    }

    if not os.path.exists(frame_dir):
        validation_result["errors"].append(f"フレームディレクトリが存在しません: {frame_dir}")
        return validation_result

    try:
        frame_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        validation_result["frame_count"] = len(frame_files)

        if len(frame_files) == 0:
            validation_result["errors"].append("フレームファイルが見つかりません")
            return validation_result

        # ファイルサイズ確認
        total_size = 0
        corrupted_files = []

        for frame_file in frame_files[:10]:  # 最初の10ファイルをサンプル確認
            file_path = os.path.join(frame_dir, frame_file)
            try:
                size = os.path.getsize(file_path)
                total_size += size

                # OpenCVで読み込みテスト
                img = cv2.imread(file_path)
                if img is None:
                    corrupted_files.append(frame_file)

            except Exception as e:
                corrupted_files.append(f"{frame_file} (エラー: {e})")

        # 全体サイズ推定
        avg_size = total_size / min(10, len(frame_files))
        estimated_total_mb = (avg_size * len(frame_files)) / (1024*1024)
        validation_result["total_size_mb"] = estimated_total_mb

        if corrupted_files:
            validation_result["warnings"].append(f"破損ファイル: {corrupted_files}")

        if estimated_total_mb > 1000:  # 1GB以上
            validation_result["warnings"].append(f"大量のフレーム ({estimated_total_mb:.1f}MB)")

        validation_result["valid"] = True

    except Exception as e:
        validation_result["errors"].append(f"ディレクトリ検証エラー: {e}")

    return validation_result

import csv
import gc
from collections import deque

class MemoryEfficientProcessor:
    """メモリ効率を考慮した処理クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # メモリ使用量制限
        self.max_memory_gb = config.get("max_memory_gb", 4.0)
        self.batch_size = config.get("batch_size", 32)
        self.streaming_output = config.get("streaming_output", True)

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（GB）"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)

    def check_memory_threshold(self) -> bool:
        """メモリ使用量が閾値を超えているかチェック"""
        current_memory = self.get_memory_usage()
        return current_memory > self.max_memory_gb

    def force_memory_cleanup(self):
        """強制的なメモリクリーンアップ"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # メモリ使用量をログ出力
        memory_after = self.get_memory_usage()
        self.logger.info(f"メモリクリーンアップ実行後: {memory_after:.2f}GB")

def analyze_frames_with_tracking_memory_efficient(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    メモリ効率を考慮したフレーム解析
    ストリーミング処理でメモリ使用量を制御
    """

    # デフォルト設定
    if config is None:
        config = {
            "confidence_threshold": 0.3,
            "tracking_config": "bytetrack.yaml",
            "save_visualizations": True,
            "batch_size": 32,
            "max_memory_gb": 4.0,
            "streaming_output": True
        }

    os.makedirs(result_dir, exist_ok=True)

    # メモリ効率プロセッサを初期化
    processor = MemoryEfficientProcessor(config)

    try:
        # モデル初期化（エラーハンドリング強化版）
        model = safe_model_initialization(model_path, config)

        # フレームディレクトリ検証
        frame_validation = validate_frame_directory(frame_dir)
        if not frame_validation["valid"]:
            return {
                "error": "frame_validation_failed",
                "details": frame_validation["errors"],
                "frame_dir": frame_dir
            }

        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        total_frames = len(frame_files)
        logger.info(f"処理対象: {total_frames}フレーム ({frame_validation['total_size_mb']:.1f}MB)")

        # ストリーミング出力用のCSVファイルを開く
        csv_path = os.path.join(result_dir, "detections_streaming.csv")
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"])

        # 処理統計
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "total_detections": 0,
            "unique_ids": set(),
            "memory_peaks": [],
            "batch_times": []
        }

        # バッチ処理でメモリ効率化
        batch_size = config["batch_size"]

        try:
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_files = frame_files[batch_start:batch_end]

                batch_start_time = time.time()
                batch_detections = []

                logger.info(f"バッチ処理 {batch_start//batch_size + 1}/{(total_frames-1)//batch_size + 1}: "
                        f"{len(batch_files)}フレーム")

                # バッチ内のフレーム処理
                for frame_file in batch_files:
                    frame_path = os.path.join(frame_dir, frame_file)

                    try:
                        # メモリ使用量チェック
                        if processor.check_memory_threshold():
                            logger.warning("メモリ使用量が閾値を超過。クリーンアップを実行...")
                            processor.force_memory_cleanup()

                        # 推論実行
                        results = model.track(
                            frame_path,
                            persist=True,
                            tracker=config["tracking_config"],
                            conf=config["confidence_threshold"],
                            verbose=False
                        )

                        # 結果処理
                        frame_detections = 0
                        for r in results:
                            if r.boxes is not None and r.boxes.id is not None:
                                boxes = r.boxes.xyxy.cpu().numpy()
                                track_ids = r.boxes.id.cpu().numpy().astype(int)
                                confidences = r.boxes.conf.cpu().numpy()

                                for box, track_id, conf in zip(boxes, track_ids, confidences):
                                    x1, y1, x2, y2 = box
                                    detection_row = [frame_file, track_id, x1, y1, x2, y2, conf, "person"]
                                    batch_detections.append(detection_row)
                                    frame_detections += 1
                                    stats["unique_ids"].add(track_id)

                        stats["total_detections"] += frame_detections
                        stats["successful_frames"] += 1

                        # 可視化（メモリ効率考慮）
                        if config["save_visualizations"] and frame_detections > 0:
                            try:
                                # 画像を読み込み、処理後即座に解放
                                frame = cv2.imread(frame_path)
                                if frame is not None:
                                    from utils.visualization import draw_detections_ultralytics
                                    vis_frame = draw_detections_ultralytics(frame, results)
                                    output_path = os.path.join(result_dir, frame_file)
                                    cv2.imwrite(output_path, vis_frame)
                                    del frame, vis_frame  # 明示的な削除
                            except Exception as vis_error:
                                logger.warning(f"可視化エラー {frame_file}: {vis_error}")

                        # 結果オブジェクトを解放
                        del results

                    except Exception as frame_error:
                        logger.error(f"フレーム処理エラー {frame_file}: {frame_error}")
                        stats["failed_frames"] += 1
                        continue

                    stats["processed_frames"] += 1

                # バッチの検出結果をCSVに書き込み（ストリーミング）
                if batch_detections:
                    csv_writer.writerows(batch_detections)
                    csv_file.flush()  # 即座にディスクに書き込み

                # バッチ処理完了後のクリーンアップ
                del batch_detections
                processor.force_memory_cleanup()

                # 統計更新
                batch_time = time.time() - batch_start_time
                current_memory = processor.get_memory_usage()
                stats["batch_times"].append(batch_time)
                stats["memory_peaks"].append(current_memory)

                # 進捗報告
                progress = (batch_end / total_frames) * 100
                logger.info(f"進捗: {progress:.1f}% (メモリ: {current_memory:.2f}GB, "
                        f"バッチ時間: {batch_time:.1f}s)")

        finally:
            # CSVファイルを確実に閉じる
            csv_file.close()

        # 最終統計の計算
        stats["unique_ids"] = len(stats["unique_ids"])
        stats["success_rate"] = stats["successful_frames"] / total_frames if total_frames > 0 else 0
        stats["avg_batch_time"] = np.mean(stats["batch_times"]) if stats["batch_times"] else 0
        stats["peak_memory_gb"] = max(stats["memory_peaks"]) if stats["memory_peaks"] else 0

        logger.info(f"✅ 処理完了統計:")
        logger.info(f"  成功率: {stats['success_rate']:.1%}")
        logger.info(f"  総検出数: {stats['total_detections']}")
        logger.info(f"  ユニークID: {stats['unique_ids']}")
        logger.info(f"  ピークメモリ: {stats['peak_memory_gb']:.2f}GB")
        logger.info(f"  平均バッチ時間: {stats['avg_batch_time']:.1f}s")

        return {
            "csv_path": csv_path,
            "processing_stats": stats,
            "config_used": config,
            "model_path": model_path,
            "result_dir": result_dir,
            "memory_efficient": True
        }

    except ModelInitializationError as e:
        logger.error(f"モデル初期化失敗: {e}")
        return {"error": "model_initialization_failed", "details": str(e)}

    except ResourceExhaustionError as e:
        logger.error(f"リソース不足: {e}")
        return {"error": "resource_exhaustion", "details": str(e)}

    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(f"詳細: {traceback.format_exc()}")
        return {"error": "unexpected_error", "details": str(e)}