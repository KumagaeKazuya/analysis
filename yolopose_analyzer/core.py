"""
メイン分析関数モジュール（統一エラーハンドリング対応版）
"""

import os
import cv2
import csv
import time
import numpy as np
import logging
import gc
from pathlib import Path
from typing import Dict, Any, Optional

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


@handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING, suppress_exceptions=False)
def analyze_frames_with_tracking_memory_efficient(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    メモリ効率的なフレーム解析（統一エラーハンドリング対応版）

    Args:
        frame_dir: フレームディレクトリ
        result_dir: 結果出力ディレクトリ
        model_path: YOLOモデルパス
        config: 処理設定

    Returns:
        ResponseBuilder形式の処理結果
    """
    with ErrorContext("フレーム解析処理", logger=logger, raise_on_error=True) as ctx:
        # デフォルト設定
        if config is None:
            config = {
                "confidence_threshold": 0.3,
                "tracking_config": "bytetrack.yaml",
                "save_visualizations": True,
                "batch_size": 32,
                "max_memory_gb": 4.0,
                "streaming_output": True,
                "device": "auto"
            }

        os.makedirs(result_dir, exist_ok=True)
        processor = MemoryEfficientProcessor(config)

        ctx.add_info("result_dir", result_dir)
        ctx.add_info("batch_size", config.get("batch_size", 32))

        try:
            # モデル初期化
            model = safe_model_initialization(model_path, config)

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

            logger.info(f"処理対象: {total_frames}フレーム ({frame_data['total_size_mb']:.1f}MB)")

            # CSV準備
            csv_path = os.path.join(result_dir, "detections_streaming.csv")

            # 統計初期化
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

            batch_size = config.get("batch_size", 32)

            # バッチ処理
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

                        for frame_file in batch_files:
                            frame_path = os.path.join(frame_dir, frame_file)

                            try:
                                # メモリチェック
                                if processor.check_memory_threshold():
                                    logger.warning("メモリ使用量が閾値を超過。クリーンアップを実行...")
                                    processor.force_memory_cleanup()

                                # トラッカー設定の安全性チェック
                                tracker_config = config.get("tracking_config")
                                if tracker_config is None or not tracker_config:
                                    tracker_config = "bytetrack.yaml"
                                    logger.debug(f"tracker設定が空だったためデフォルト値を使用: {tracker_config}")

                                # 推論実行
                                results = model.track(
                                    frame_path,
                                    persist=True,
                                    tracker=tracker_config,
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
                                            detection_row = [
                                                frame_file, track_id,
                                                float(x1), float(y1), float(x2), float(y2),
                                                float(conf), "person"
                                            ]
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
                                            # vis_プレフィックスを追加
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
                                logger.error(f"フレーム処理エラー {frame_file}: {frame_error}", exc_info=True)
                                stats["failed_frames"] += 1
                                continue

                            stats["processed_frames"] += 1

                        # バッチの検出結果をCSVに書き込み
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
                    logger.error(f"バッチ処理エラー: {e}", exc_info=True)
                    raise VideoProcessingError(f"バッチ処理に失敗しました: {e}", original_exception=e)

            # 最終統計の計算
            stats["unique_ids"] = len(stats["unique_ids"])
            stats["success_rate"] = stats["successful_frames"] / total_frames if total_frames > 0 else 0
            stats["avg_batch_time"] = np.mean(stats["batch_times"]) if stats["batch_times"] else 0
            stats["peak_memory_gb"] = max(stats["memory_peaks"]) if stats["memory_peaks"] else 0

            ctx.add_info("total_detections", stats["total_detections"])
            ctx.add_info("success_rate", stats["success_rate"])
            ctx.add_info("peak_memory_gb", stats["peak_memory_gb"])

            # 結果サマリーのログ出力
            logger.info(f"✅ 処理完了統計:")
            logger.info(f"  成功率: {stats['success_rate']:.1%}")
            logger.info(f"  総検出数: {stats['total_detections']}")
            logger.info(f"  ユニークID: {stats['unique_ids']}")
            logger.info(f"  ピークメモリ: {stats['peak_memory_gb']:.2f}GB")
            logger.info(f"  平均バッチ時間: {stats['avg_batch_time']:.1f}s")

            return ResponseBuilder.success(
                data={
                    "csv_path": csv_path,
                    "processing_stats": stats,
                    "config_used": config,
                    "model_path": model_path,
                    "result_dir": result_dir,
                    "memory_efficient": True
                },
                message="フレーム解析完了"
            )

        except ModelInitializationError as e:
            logger.error(f"モデル初期化失敗: {e}")
            return ResponseBuilder.error(
                message="モデル初期化に失敗しました",
                details={"error": str(e), "model_path": model_path}
            )

        except ResourceExhaustionError as e:
            logger.error(f"リソース不足: {e}")
            return ResponseBuilder.error(
                message="システムリソースが不足しています",
                details={"error": str(e)}
            )

        except VideoProcessingError as e:
            logger.error(f"動画処理エラー: {e}")
            return ResponseBuilder.error(
                message="動画処理中にエラーが発生しました",
                details={"error": str(e)}
            )

        except Exception as e:
            logger.error(f"予期しないエラー: {e}", exc_info=True)
            return ResponseBuilder.error(
                message="予期しないエラーが発生しました",
                details={"error": str(e)}
            )


@handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
def analyze_frames_with_tracking_enhanced(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    拡張版フレーム解析（タイル推論オプション付き）

    設定でtile_inference.enabled=Trueの場合、タイル推論を使用。
    それ以外は通常のメモリ効率版を使用。

    Args:
        frame_dir: フレームディレクトリ
        result_dir: 結果出力ディレクトリ
        model_path: YOLOモデルパス
        config: 処理設定（tile_inference設定を含む可能性あり）

    Returns:
        ResponseBuilder形式の処理結果
    """
    with ErrorContext("拡張フレーム解析", logger=logger) as ctx:
        ctx.add_info("frame_dir", frame_dir)
        ctx.add_info("model_path", model_path)

        if config and config.get("tile_inference", {}).get("enabled", False):
            # タイル推論モードへ
            logger.info("🔲 タイル推論モードで実行")
            try:
                from .tile_inference import analyze_frames_with_tile_inference
                return analyze_frames_with_tile_inference(frame_dir, result_dir, model_path, config)
            except ImportError:
                logger.warning("タイル推論モジュールが見つかりません。通常推論に切り替えます")
                return analyze_frames_with_tracking_memory_efficient(
                    frame_dir, result_dir, model_path, config
                )
        else:
            # 通常推論（メモリ効率版）
            logger.info("💻 通常推論（メモリ効率版）で実行")
            return analyze_frames_with_tracking_memory_efficient(
                frame_dir, result_dir, model_path, config
            )