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
import csv
import gc
from collections import deque

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🆕 タイル推論クラスのインポート（エラーハンドリング強化）
try:
    from processors.tile_processor import TileProcessor, TileConfig, AdaptiveTileProcessor
    TILE_INFERENCE_AVAILABLE = True
    logger.info("✅ タイル推論モジュール読み込み成功")
except ImportError as e:
    TILE_INFERENCE_AVAILABLE = False
    logger.warning(f"⚠️ タイル推論モジュールが見つかりません: {e}")
    logger.info("通常推論のみ利用可能です。processors/tile_processor.py を作成してください。")

    # フォールバック用のダミークラス
    class TileProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("TileProcessor が利用できません")

    class TileConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TileConfig が利用できません")

    class AdaptiveTileProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("AdaptiveTileProcessor が利用できません")

# エラークラス定義
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

        # ディスク使用量確認（Windowsでもクロスプラットフォーム対応）
        try:
            if os.name == 'nt':  # Windows
                disk = psutil.disk_usage('C:')
            else:  # Unix/Linux/Mac
                disk = psutil.disk_usage('/')
        except:
            disk = None

        # ✅ GPU検出を改善（CUDA + MPS対応）
        gpu_available = False
        gpu_type = "none"
        gpu_count = 0
        gpu_name = "N/A"

        # CUDA (NVIDIA GPU) チェック
        if torch.cuda.is_available():
            gpu_available = True
            gpu_type = "cuda"
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
        # MPS (Apple Silicon GPU) チェック
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_available = True
            gpu_type = "mps"
            gpu_count = 1  # MPSは常に1デバイス
            gpu_name = "Apple Silicon GPU (MPS)"

        result = {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "cpu_count": psutil.cpu_count(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name
        }

        if disk:
            result["disk_free_gb"] = disk.free / (1024**3)

        return result
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
                f"以下の方法でモデルをダウンロードできます:\n"
                f"1. コマンド: python -c \"from ultralytics import YOLO; YOLO('{model_name}')\"\n"
                f"2. または公式サイトからダウンロード"
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
        # PyTorch 2.8対策: weights_onlyを強制的にFalseに
        import torch
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

        test_model = YOLO(model_path)

        # 元に戻す
        torch.load = _original_torch_load

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

        # ✅ GPU情報の出力を改善
        if resources["gpu_available"]:
            gpu_type = resources.get("gpu_type", "unknown")
            gpu_name = resources.get("gpu_name", "N/A")
            gpu_count = resources.get("gpu_count", 0)

            if gpu_type == "mps":
                logger.info(f"🍎 Apple Silicon GPU (MPS) 利用可能")
            elif gpu_type == "cuda":
                logger.info(f"🚀 NVIDIA GPU (CUDA) 利用可能: {gpu_count}個のデバイス")
                if gpu_name != "N/A":
                    logger.info(f"   GPU名: {gpu_name}")
            else:
                logger.info(f"GPU利用可能: {gpu_type.upper()}")
        else:
            logger.info("💻 GPU利用不可。CPUで処理します。")

    # モデルファイル検証
    validation = validate_model_file(model_path)
    if not validation["valid"]:
        error_msg = "\n".join(validation["errors"])
        suggestions = "\n".join(validation["suggestions"])
        raise ModelInitializationError(f"{error_msg}\n\n推奨対応:\n{suggestions}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.info(warning)


        # PyTorch 2.6以降のweights_only対策
        import torch.serialization
        try:
            # Ultralyticsのクラスをsafe_globalsに追加
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.PoseModel',
                'ultralytics.nn.tasks.DetectionModel'
            ])
        except AttributeError:
            # 古いPyTorchバージョンでは無視
            pass

    try:
        # PyTorch 2.8対策: weights_onlyを強制的にFalseに
        import torch
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

        # モデル初期化
        model = YOLO(model_path)

        # torch.loadを元に戻す
        torch.load = _original_torch_load

        # ✅ デバイス設定を改善
        device = config.get("device", "auto")
        
        if device == "auto":
            # 自動検出
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        try:
            model.to(device)
        except Exception as e:
            logger.warning(f"デバイス設定エラー: {e}. CPUを使用します。")
            device = "cpu"
            model.to(device)

        # ✅ デバイス情報の出力を改善
        if device == "mps":
            logger.info(f"モデル初期化完了: {model_path} on 🍎 {device.upper()}")
        elif device == "cuda":
            logger.info(f"モデル初期化完了: {model_path} on 🚀 {device.upper()}")
        else:
            logger.info(f"モデル初期化完了: {model_path} on 💻 {device.upper()}")

        # GPU使用時の追加設定
        if device == "cuda" and torch.cuda.is_available():
            try:
                # GPU メモリ確認
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU メモリ: {gpu_memory:.1f}GB")

                # 半精度演算設定（オプション）
                if config.get("use_half_precision", False):
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
            validation_result["suggestions"] = [
                "フレーム抽出が正常に実行されているか確認してください",
                "対応形式: .jpg, .jpeg, .png"
            ]
            return validation_result

        # ファイルサイズ確認
        total_size = 0
        corrupted_files = []

        sample_size = min(10, len(frame_files))
        for frame_file in frame_files[:sample_size]:
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
        if sample_size > 0:
            avg_size = total_size / sample_size
            estimated_total_mb = (avg_size * len(frame_files)) / (1024*1024)
            validation_result["total_size_mb"] = estimated_total_mb

        if corrupted_files:
            validation_result["warnings"].append(f"破損ファイル: {corrupted_files[:3]}")

        if validation_result["total_size_mb"] > 1000:  # 1GB以上
            validation_result["warnings"].append(f"大量のフレーム ({validation_result['total_size_mb']:.1f}MB)")

        validation_result["valid"] = True

    except Exception as e:
        validation_result["errors"].append(f"ディレクトリ検証エラー: {e}")

    return validation_result

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
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception as e:
            self.logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def check_memory_threshold(self) -> bool:
        """メモリ使用量が閾値を超えているかチェック"""
        current_memory = self.get_memory_usage()
        return current_memory > self.max_memory_gb

    def force_memory_cleanup(self):
        """強制的なメモリクリーンアップ"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"メモリクリーンアップエラー: {e}")

        # メモリ使用量をログ出力
        memory_after = self.get_memory_usage()
        self.logger.info(f"メモリクリーンアップ実行後: {memory_after:.2f}GB")

# 🆕 タイル推論統合メイン関数
def analyze_frames_with_tile_inference(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    タイル推論を使用したフレーム解析
    """

    # タイル推論が利用不可の場合は既存関数にフォールバック
    if not TILE_INFERENCE_AVAILABLE:
        logger.warning("タイル推論が利用できません。通常推論を実行します。")
        return analyze_frames_with_tracking_memory_efficient(
            frame_dir, result_dir, model_path, config
        )

    # デフォルト設定（タイル推論対応）
    if config is None:
        config = {
            "confidence_threshold": 0.3,
            "tracking_config": "bytetrack.yaml",
            "save_visualizations": True,
            "tile_inference": {
                "enabled": True,
                "tile_size": (640, 640),
                "overlap_ratio": 0.2,
                "use_adaptive": False,
                "max_tiles_per_frame": 16
            }
        }

    os.makedirs(result_dir, exist_ok=True)

    try:
        # モデル初期化
        model = safe_model_initialization(model_path, config)
        logger.info(f"モデル読み込み完了: {model_path}")

        # タイル推論の設定確認
        tile_config_data = config.get("tile_inference", {})
        tile_enabled = tile_config_data.get("enabled", False)

        if tile_enabled:
            # タイル推論プロセッサを初期化
            tile_config = TileConfig(
                tile_size=tuple(tile_config_data.get("tile_size", (640, 640))),
                overlap_ratio=tile_config_data.get("overlap_ratio", 0.2),
                min_confidence=config.get("confidence_threshold", 0.3),
                max_tiles_per_frame=tile_config_data.get("max_tiles_per_frame", 16)
            )

            use_adaptive = tile_config_data.get("use_adaptive", False)
            if use_adaptive:
                tile_processor = AdaptiveTileProcessor(model, tile_config)
                logger.info("🔲 適応的タイル推論を使用")
            else:
                tile_processor = TileProcessor(model, tile_config)
                logger.info("🔲 標準タイル推論を使用")
        else:
            tile_processor = None
            logger.info("📋 通常推論を使用")

        # フレームディレクトリ検証
        frame_validation = validate_frame_directory(frame_dir)
        if not frame_validation["valid"]:
            return {
                "error": "frame_validation_failed",
                "details": frame_validation["errors"],
                "suggestions": frame_validation.get("suggestions", []),
                "frame_dir": frame_dir
            }

        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        total_frames = len(frame_files)
        logger.info(f"処理対象: {total_frames}フレーム")

        # 結果保存用CSV準備
        csv_path = os.path.join(result_dir, "detections_enhanced.csv")
        csv_columns = ["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"]

        # タイル推論の場合は追加情報も保存
        if tile_processor:
            csv_columns.extend(["tile_source", "tile_count", "nms_reduction"])

        # 処理統計
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "total_detections": 0,
            "unique_ids": set(),
            "tile_stats": {
                "total_tiles_processed": 0,
                "avg_tiles_per_frame": 0,
                "avg_nms_reduction": 0
            } if tile_processor else None
        }

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_columns)

            for frame_idx, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)

                try:
                    # フレーム読み込み
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        logger.warning(f"フレーム読み込み失敗: {frame_file}")
                        stats["failed_frames"] += 1
                        continue

                    # 🔲 タイル推論 または 📋 通常推論
                    if tile_processor:
                        # タイル推論実行
                        try:
                            if hasattr(tile_processor, 'process_frame_with_adaptive_tiles'):
                                detection_result = tile_processor.process_frame_with_adaptive_tiles(frame, frame_idx)
                            else:
                                detection_result = tile_processor.process_frame_with_tiles(frame)

                            boxes = detection_result["boxes"]
                            confidences = detection_result["confidences"]
                            tile_sources = detection_result.get("tile_sources", [])
                            tile_count = detection_result.get("processing_stats", {}).get("num_tiles", 0)
                            nms_reduction = detection_result.get("nms_reduction_rate", 0)

                            # 統計更新
                            if stats["tile_stats"]:
                                stats["tile_stats"]["total_tiles_processed"] += tile_count
                                stats["tile_stats"]["avg_nms_reduction"] += nms_reduction

                        except Exception as e:
                            logger.error(f"タイル推論エラー {frame_file}: {e}")
                            stats["failed_frames"] += 1
                            continue

                    else:
                        # 通常推論（トラッキング付き）
                        try:
                            results = model.track(
                                frame,
                                persist=True,
                                tracker=config.get("tracking_config", "bytetrack.yaml"),
                                conf=config.get("confidence_threshold", 0.3),
                                verbose=False
                            )

                            # 結果変換
                            if results and results[0].boxes is not None:
                                boxes = results[0].boxes.xyxy.cpu().numpy()
                                confidences = results[0].boxes.conf.cpu().numpy()
                                # トラッキングIDがあれば使用
                                if results[0].boxes.id is not None:
                                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                                else:
                                    track_ids = list(range(len(boxes)))
                            else:
                                boxes = np.array([]).reshape(0, 4)
                                confidences = np.array([])
                                track_ids = []

                            tile_sources = [0] * len(boxes)  # 通常推論では全て0
                            tile_count = 1
                            nms_reduction = 0

                        except Exception as e:
                            logger.error(f"通常推論エラー {frame_file}: {e}")
                            stats["failed_frames"] += 1
                            continue

                    # 検出結果をCSVに保存
                    if len(boxes) > 0:
                        for i, (box, conf) in enumerate(zip(boxes, confidences)):
                            if tile_processor:
                                # タイル推論の場合はシンプルなID
                                person_id = f"tile_{frame_idx}_{i}"
                            else:
                                # 通常推論の場合はトラッキングID
                                person_id = track_ids[i] if i < len(track_ids) else f"det_{frame_idx}_{i}"

                            row_data = [
                                frame_file, person_id,
                                float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                                float(conf), "person"
                            ]

                            # タイル推論の場合は追加情報
                            if tile_processor:
                                tile_src = tile_sources[i] if i < len(tile_sources) else -1
                                row_data.extend([int(tile_src), int(tile_count), float(nms_reduction)])

                            csv_writer.writerow(row_data)
                            stats["unique_ids"].add(person_id)

                    # 可視化保存
                    if config.get("save_visualizations", False) and len(boxes) > 0:
                        try:
                            vis_frame = _draw_detections_enhanced(
                                frame, boxes, confidences, tile_sources if tile_processor else None
                            )
                            output_path = os.path.join(result_dir, f"vis_{frame_file}")
                            cv2.imwrite(output_path, vis_frame)
                        except Exception as e:
                            logger.warning(f"可視化保存エラー {frame_file}: {e}")

                    stats["total_detections"] += len(boxes)
                    stats["processed_frames"] += 1
                    stats["successful_frames"] += 1

                    # 進捗表示
                    if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == total_frames:
                        progress = (frame_idx + 1) / total_frames * 100
                        logger.info(f"進捗: {progress:.1f}% ({frame_idx + 1}/{total_frames})")

                except Exception as e:
                    logger.error(f"フレーム処理エラー {frame_file}: {e}")
                    stats["failed_frames"] += 1
                    continue

        # 最終統計計算
        stats["unique_ids"] = len(stats["unique_ids"])

        if stats["tile_stats"] and stats["processed_frames"] > 0:
            stats["tile_stats"]["avg_tiles_per_frame"] = (
                stats["tile_stats"]["total_tiles_processed"] / stats["processed_frames"]
            )
            stats["tile_stats"]["avg_nms_reduction"] = (
                stats["tile_stats"]["avg_nms_reduction"] / stats["processed_frames"]
            )

        # パフォーマンス統計
        performance_stats = {}
        if tile_processor:
            try:
                performance_stats = tile_processor.get_performance_stats()
            except Exception as e:
                logger.warning(f"パフォーマンス統計取得エラー: {e}")

        logger.info("✅ 処理完了")
        logger.info(f"  総検出数: {stats['total_detections']}")
        logger.info(f"  ユニークID: {stats['unique_ids']}")
        logger.info(f"  成功率: {stats['successful_frames']}/{stats['total_frames']} ({stats['successful_frames']/stats['total_frames']*100:.1f}%)")

        if stats["tile_stats"]:
            logger.info(f"  🔲 平均タイル数/フレーム: {stats['tile_stats']['avg_tiles_per_frame']:.1f}")
            logger.info(f"  🔲 平均NMS削減率: {stats['tile_stats']['avg_nms_reduction']:.1%}")

        return {
            "csv_path": csv_path,
            "processing_stats": stats,
            "performance_stats": performance_stats,
            "config_used": config,
            "model_path": model_path,
            "result_dir": result_dir,
            "tile_inference_enabled": tile_processor is not None,
            "success": True
        }

    except Exception as e:
        logger.error(f"処理エラー: {e}")
        logger.error(f"詳細: {traceback.format_exc()}")
        return {"error": "processing_failed", "details": str(e), "success": False}

def _draw_detections_enhanced(frame: np.ndarray,
                            boxes: np.ndarray,
                            confidences: np.ndarray,
                            tile_sources: Optional[List[int]] = None) -> np.ndarray:
    """
    拡張版検出結果描画（タイル情報付き）
    """
    vis_frame = frame.copy()

    # タイル別の色定義
    tile_colors = [
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青
        (0, 0, 255),    # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄色
        (128, 0, 255),  # 紫
        (255, 128, 0),  # オレンジ
    ]

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        try:
            x1, y1, x2, y2 = map(int, box)

            # タイルソースに応じて色を選択
            if tile_sources and i < len(tile_sources):
                tile_idx = tile_sources[i]
                color = tile_colors[tile_idx % len(tile_colors)]
            else:
                color = (0, 255, 0)  # デフォルト緑

            # バウンディングボックス描画
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # 信頼度表示
            conf_text = f"{conf:.2f}"
            cv2.putText(vis_frame, conf_text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # タイルインデックス表示（タイル推論の場合）
            if tile_sources and i < len(tile_sources):
                tile_text = f"T{tile_sources[i]}"
                cv2.putText(vis_frame, tile_text, (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            logger.warning(f"描画エラー (box {i}): {e}")
            continue

    return vis_frame

# 🆕 タイル推論統合版の拡張関数
def analyze_frames_with_tracking_enhanced(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    既存のanalyze_frames_with_tracking_memory_efficientの改良版
    タイル推論オプション付き
    """

    # 設定にタイル推論が含まれているかチェック
    if config and config.get("tile_inference", {}).get("enabled", False):
        return analyze_frames_with_tile_inference(frame_dir, result_dir, model_path, config)
    else:
        return analyze_frames_with_tracking_memory_efficient(frame_dir, result_dir, model_path, config)

# 🆕 比較実験用関数
def compare_tile_vs_normal_inference(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    sample_frames: int = 10
) -> Dict[str, Any]:
    """
    タイル推論と通常推論の比較実験
    """
    if not TILE_INFERENCE_AVAILABLE:
        logger.error("タイル推論モジュールが利用できません")
        return {"error": "tile_inference_not_available", "success": False}

    os.makedirs(result_dir, exist_ok=True)

    # フレームファイル取得
    try:
        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:sample_frames]

        if not frame_files:
            return {"error": "no_frames_found", "success": False}
    except Exception as e:
        return {"error": f"frame_directory_error: {e}", "success": False}


    try:
        # モデル初期化
        model = safe_model_initialization(model_path, {})

        # タイル推論設定
        tile_config = TileConfig(
            tile_size=(640, 640),
            overlap_ratio=0.2,
            min_confidence=0.3
        )
        tile_processor = TileProcessor(model, tile_config)

        comparison_results = {
            "normal_inference": {"total_detections": 0, "processing_times": []},
            "tile_inference": {"total_detections": 0, "processing_times": []},
            "frame_comparisons": []
        }

        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                logger.warning(f"フレーム読み込み失敗: {frame_file}")
                continue

            frame_comparison = {"frame": frame_file}

            # 1. 通常推論
            try:
                start_time = time.time()
                normal_results = model(frame, conf=0.3, verbose=False)
                normal_time = time.time() - start_time

                normal_boxes = []
                if normal_results and normal_results[0].boxes is not None:
                    normal_boxes = normal_results[0].boxes.xyxy.cpu().numpy()

                frame_comparison["normal"] = {
                    "detections": len(normal_boxes),
                    "processing_time": normal_time
                }
                comparison_results["normal_inference"]["total_detections"] += len(normal_boxes)
                comparison_results["normal_inference"]["processing_times"].append(normal_time)
            except Exception as e:
                logger.error(f"通常推論エラー {frame_file}: {e}")
                frame_comparison["normal"] = {
                    "detections": 0,
                    "processing_time": 0,
                    "error": str(e)
                }

            # 2. タイル推論
            try:
                start_time = time.time()
                tile_results = tile_processor.process_frame_with_tiles(frame)
                tile_time = time.time() - start_time

                tile_boxes = tile_results["boxes"]

                frame_comparison["tile"] = {
                    "detections": len(tile_boxes),
                    "processing_time": tile_time,
                    "num_tiles": tile_results.get("processing_stats", {}).get("num_tiles", 0),
                    "nms_reduction": tile_results.get("nms_reduction_rate", 0)
                }
                comparison_results["tile_inference"]["total_detections"] += len(tile_boxes)
                comparison_results["tile_inference"]["processing_times"].append(tile_time)
            except Exception as e:
                logger.error(f"タイル推論エラー {frame_file}: {e}")
                frame_comparison["tile"] = {
                    "detections": 0,
                    "processing_time": 0,
                    "num_tiles": 0,
                    "nms_reduction": 0,
                    "error": str(e)
                }

            # 3. 検出数改善率計算
            normal_det = frame_comparison["normal"]["detections"]
            tile_det = frame_comparison["tile"]["detections"]

            improvement = tile_det - normal_det
            improvement_rate = improvement / normal_det if normal_det > 0 else 0

            frame_comparison["improvement"] = {
                "detection_difference": improvement,
                "improvement_rate": improvement_rate,
                "time_overhead": frame_comparison["tile"]["processing_time"] - frame_comparison["normal"]["processing_time"]
            }

            comparison_results["frame_comparisons"].append(frame_comparison)

            # 比較可視化保存
            try:
                if normal_det > 0 or tile_det > 0:
                    normal_boxes = normal_boxes if 'normal_boxes' in locals() else np.array([]).reshape(0, 4)
                    tile_boxes_array = tile_boxes if isinstance(tile_boxes, np.ndarray) else np.array([]).reshape(0, 4)

                    _save_comparison_visualization(
                        frame, normal_boxes, tile_boxes_array,
                        os.path.join(result_dir, f"comparison_{frame_file}")
                    )
            except Exception as e:
                logger.warning(f"比較可視化保存エラー {frame_file}: {e}")

        # 全体統計計算
        if comparison_results["frame_comparisons"]:
            frame_comparisons = comparison_results["frame_comparisons"]

            # エラーがないフレームのみで統計計算
            valid_normal = [fc["normal"] for fc in frame_comparisons if "error" not in fc["normal"]]
            valid_tile = [fc["tile"] for fc in frame_comparisons if "error" not in fc["tile"]]

            comparison_results["summary"] = {
                "total_frames": len(frame_comparisons),
                "valid_normal_frames": len(valid_normal),
                "valid_tile_frames": len(valid_tile),
                "avg_normal_detections": np.mean([f["detections"] for f in valid_normal]) if valid_normal else 0,
                "avg_tile_detections": np.mean([f["detections"] for f in valid_tile]) if valid_tile else 0,
                "avg_normal_time": np.mean([f["processing_time"] for f in valid_normal]) if valid_normal else 0,
                "avg_tile_time": np.mean([f["processing_time"] for f in valid_tile]) if valid_tile else 0,
                "overall_detection_improvement": (
                    comparison_results["tile_inference"]["total_detections"] - 
                    comparison_results["normal_inference"]["total_detections"]
                ),
                "overall_improvement_rate": (
                    (comparison_results["tile_inference"]["total_detections"] - 
                    comparison_results["normal_inference"]["total_detections"]) /
                    comparison_results["normal_inference"]["total_detections"]
                    if comparison_results["normal_inference"]["total_detections"] > 0 else 0
                )
            }
        else:
            comparison_results["summary"] = {
                "error": "no_valid_comparisons"
            }

        # 結果保存
        import json
        with open(os.path.join(result_dir, "tile_comparison.json"), 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        logger.info("比較実験完了")
        if "error" not in comparison_results["summary"]:
            logger.info(f"検出数改善: {comparison_results['summary']['overall_improvement_rate']:.1%}")
            logger.info(f"処理時間オーバーヘッド: {comparison_results['summary']['avg_tile_time'] - comparison_results['summary']['avg_normal_time']:.2f}秒")

        comparison_results["success"] = True
        return comparison_results

    except Exception as e:
        logger.error(f"比較実験エラー: {e}")
        return {"error": f"comparison_experiment_failed: {e}", "success": False}

def _save_comparison_visualization(frame: np.ndarray,
                                normal_boxes: np.ndarray,
                                tile_boxes: np.ndarray,
                                output_path: str):
    """比較可視化の保存"""
    try:
        # 2つの結果を左右に並べて表示
        height, width = frame.shape[:2]
        comparison_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)

        # 左側：通常推論結果
        left_frame = frame.copy()
        if len(normal_boxes) > 0:
            for box in normal_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        comparison_frame[:, :width] = left_frame

        # 右側：タイル推論結果
        right_frame = frame.copy()
        if len(tile_boxes) > 0:
            for box in tile_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        comparison_frame[:, width:] = right_frame

        # ラベル追加
        cv2.putText(comparison_frame, f"Normal ({len(normal_boxes)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_frame, f"Tile ({len(tile_boxes)})",
                    (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(output_path, comparison_frame)
    except Exception as e:
        logger.warning(f"比較可視化保存エラー: {e}")

# ========== 既存の関数群（エラー修正版） ==========

def analyze_frames_with_tracking_memory_efficient(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    メモリ効率を考慮したフレーム解析（既存関数・エラー修正版）
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
                "suggestions": frame_validation.get("suggestions", []),
                "frame_dir": frame_dir,
                "success": False
            }

        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        total_frames = len(frame_files)
        logger.info(f"処理対象: {total_frames}フレーム ({frame_validation['total_size_mb']:.1f}MB)")

        # ストリーミング出力用のCSVファイルを開く
        csv_path = os.path.join(result_dir, "detections_streaming.csv")

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
        batch_size = config.get("batch_size", 32)

        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame", "person_id", "x1", "y1", "x2", "y2", "conf", "class_name"])

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
                                tracker=config.get("tracking_config", "bytetrack.yaml"),
                                conf=config.get("confidence_threshold", 0.3),
                                verbose=False
                            )

                            # 結果処理
                            frame_detections = 0
                            for r in results:
                                if r.boxes is not None:
                                    boxes = r.boxes.xyxy.cpu().numpy()
                                    confidences = r.boxes.conf.cpu().numpy()

                                    # トラッキングIDの処理
                                    if r.boxes.id is not None:
                                        track_ids = r.boxes.id.cpu().numpy().astype(int)
                                    else:
                                        track_ids = list(range(len(boxes)))

                                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                                        track_id = track_ids[i] if i < len(track_ids) else i
                                        x1, y1, x2, y2 = box
                                        detection_row = [frame_file, track_id, float(x1), float(y1), float(x2), float(y2), float(conf), "person"]
                                        batch_detections.append(detection_row)
                                        frame_detections += 1
                                        stats["unique_ids"].add(track_id)

                            stats["total_detections"] += frame_detections
                            stats["successful_frames"] += 1

                            # 可視化（メモリ効率考慮）
                            if config.get("save_visualizations", False) and frame_detections > 0:
                                try:
                                    frame = cv2.imread(frame_path)
                                    if frame is not None:
                                        vis_frame = draw_detections_ultralytics(frame, results)
                                        # ✅ vis_プレフィックスを追加
                                        vis_filename = f"vis_{frame_file}"
                                        output_path = os.path.join(result_dir, vis_filename)
                                        cv2.imwrite(output_path, vis_frame)
                                        logger.debug(f"可視化保存: {output_path}")
                                        del frame, vis_frame
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

            except Exception as e:
                logger.error(f"バッチ処理エラー: {e}")
                return {"error": f"batch_processing_failed: {e}", "success": False}

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
            "memory_efficient": True,
            "success": True
        }

    except ModelInitializationError as e:
        logger.error(f"モデル初期化失敗: {e}")
        return {"error": "model_initialization_failed", "details": str(e), "success": False}

    except ResourceExhaustionError as e:
        logger.error(f"リソース不足: {e}")
        return {"error": "resource_exhaustion", "details": str(e), "success": False}

    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(f"詳細: {traceback.format_exc()}")
        return {"error": "unexpected_error", "details": str(e), "success": False}

# ========== 既存のutils関数群（修正版） ==========

def draw_detections(frame, results, online_targets=None):
    """既存の描画関数（互換性維持）"""
    try:
        # バウンディングボックス描画
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ID表示
        if online_targets:
            for t in online_targets:
                x1, y1, x2, y2 = map(int, t.tlbr)
                track_id = t.track_id
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    except Exception as e:
        logger.warning(f"描画エラー: {e}")

    return frame

def draw_detections_ultralytics(frame, results):
    """Ultralytics組み込みトラッカー用の描画関数"""
    try:
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                # トラックIDがある場合
                if r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)

                    for box, conf, track_id in zip(boxes, confidences, track_ids):
                        x1, y1, x2, y2 = map(int, box)

                        # バウンディングボックス描画
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 信頼度表示
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # トラックID表示
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1-25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # トラックIDがない場合（通常の検出結果）
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # キーポイントがある場合の描画
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()
                for kpts in keypoints:
                    for x, y in kpts:
                        if x > 0 and y > 0:  # 有効なキーポイントのみ
                            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
    except Exception as e:
        logger.warning(f"Ultralytics描画エラー: {e}")

    return frame

# ========== メイン実行部（テスト用） ==========

if __name__ == "__main__":
    # コマンドライン引数解析
    import argparse

    parser = argparse.ArgumentParser(description='YOLO11 フレーム解析（タイル推論対応）')
    parser.add_argument('--frame-dir', required=True, help='フレームディレクトリ')
    parser.add_argument('--output-dir', required=True, help='出力ディレクトリ')
    parser.add_argument('--model', default='models/yolo11n-pose.pt', help='モデルパス')
    parser.add_argument('--tile', action='store_true', help='タイル推論を有効化')
    parser.add_argument('--adaptive', action='store_true', help='適応的タイル推論を使用')
    parser.add_argument('--compare', action='store_true', help='比較実験実行')
    parser.add_argument('--tile-size', type=int, nargs=2, default=[640, 640], 
                    help='タイルサイズ [width height]')
    parser.add_argument('--overlap', type=float, default=0.2, help='重複率')
    parser.add_argument('--confidence', type=float, default=0.3, help='信頼度閾値')

    args = parser.parse_args()

    # 結果ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.compare:
            # 比較実験
            print("🔬 タイル推論 vs 通常推論 比較実験")
            results = compare_tile_vs_normal_inference(
                args.frame_dir,
                args.output_dir,
                args.model,
                sample_frames=20
            )

            if results.get("success", False):
                print("✅ 比較実験完了")
                summary = results.get("summary", {})
                if "error" not in summary:
                    print(f"📊 検出数改善率: {summary.get('overall_improvement_rate', 0):.1%}")
                    print(f"⏱️ 処理時間オーバーヘッド: {summary.get('avg_tile_time', 0) - summary.get('avg_normal_time', 0):.2f}秒")
                else:
                    print(f"⚠️ 比較実験で問題発生: {summary['error']}")
            else:
                print(f"❌ 比較実験エラー: {results.get('error', 'unknown_error')}")

        elif args.tile:
            # タイル推論実行
            print("🔲 タイル推論実行")

            config = {
                "tile_inference": {
                    "enabled": True,
                    "tile_size": tuple(args.tile_size),
                    "overlap_ratio": args.overlap,
                    "use_adaptive": args.adaptive
                },
                "save_visualizations": True,
                "confidence_threshold": args.confidence
            }

            results = analyze_frames_with_tile_inference(
                args.frame_dir,
                args.output_dir,
                args.model,
                config
            )

            if results.get("success", False):
                print("✅ タイル推論完了")
                stats = results.get("processing_stats", {})
                print(f"📊 総検出数: {stats.get('total_detections', 0)}")
                print(f"👥 ユニークID: {stats.get('unique_ids', 0)}")
                tile_stats = stats.get("tile_stats")
                if tile_stats:
                    print(f"🔲 平均タイル数/フレーム: {tile_stats.get('avg_tiles_per_frame', 0):.1f}")
            else:
                print(f"❌ タイル推論エラー: {results.get('error', 'unknown_error')}")

        else:
            # 通常推論実行
            print("📋 通常推論実行")

            config = {
                "save_visualizations": True,
                "confidence_threshold": args.confidence
            }

            results = analyze_frames_with_tracking_memory_efficient(
                args.frame_dir,
                args.output_dir,
                args.model,
                config
            )

            if results.get("success", False):
                print("✅ 通常推論完了")
                stats = results.get("processing_stats", {})
                print(f"📊 総検出数: {stats.get('total_detections', 0)}")
                print(f"👥 ユニークID: {stats.get('unique_ids', 0)}")
                print(f"📈 成功率: {stats.get('success_rate', 0):.1%}")
            else:
                print(f"❌ 通常推論エラー: {results.get('error', 'unknown_error')}")

    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        logger.error(f"メイン実行エラー: {traceback.format_exc()}")
