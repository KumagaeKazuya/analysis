import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

class VideoProcessor:
    """動画処理を統括するクラス - パス修正版"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # タイル推論が有効かチェック
        self.tile_enabled = config.get('processing.tile_inference.enabled', False)
        if self.tile_enabled:
            self.logger.info("🔲 タイル推論モードで初期化")
        else:
            self.logger.info("📋 通常推論モードで初期化")

    def extract_frames(self, video_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """動画からフレームを抽出"""
        output_dir.mkdir(parents=True, exist_ok=True)
        interval_sec = kwargs.get('interval_sec', self.config.get('processing.frame_sampling.interval_sec', 2))

        try:
            self.logger.info(f"フレーム抽出開始: {video_path.name}")

            # 既存の実装を使用
            try:
                from frame_sampler import sample_frames
                sample_frames(str(video_path), str(output_dir), interval_sec=interval_sec)
            except ImportError:
                self.logger.warning("frame_sampler モジュールが見つかりません。内蔵実装を使用します。")
                self._sample_frames_builtin(str(video_path), str(output_dir), interval_sec)

            frame_files = list(output_dir.glob("*.jpg"))

            result = {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "frame_count": len(frame_files),
                "interval_sec": interval_sec,
                "success": True
            }

            self.logger.info(f"フレーム抽出完了: {len(frame_files)}フレーム")
            return result

        except Exception as e:
            self.logger.error(f"フレーム抽出エラー {video_path.name}: {e}")
            return {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "error": str(e),
                "success": False
            }

    def _sample_frames_builtin(self, video_path: str, save_dir: str, interval_sec: int = 2):
        """内蔵フレーム抽出実装"""
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_sec) if fps > 0 else 30
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                filename = os.path.join(save_dir, f"{os.path.basename(video_path)}_frame{frame_count:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        self.logger.info(f"内蔵実装でフレーム抽出完了: {saved_count}フレーム")

    def run_detection_tracking(self, frame_dir: Path, video_name: str, **kwargs) -> Dict[str, Any]:
        """検出・追跡処理を実行（パス修正版）"""
        try:
            self.logger.info(f"🔍 検出・追跡開始: {video_name}")

            # ✅ 修正: 結果ディレクトリのパスを明確に指定
            # frame_dir の親ディレクトリ（video_name）の下に results を作成
            video_base_dir = frame_dir.parent  # .../test
            result_dir = video_base_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"📂 結果保存先: {result_dir}")

            # 処理設定を構築
            processing_config = self._build_processing_config(kwargs)

            # タイル推論が有効な場合
            if self.tile_enabled:
                result = self._run_tile_inference(frame_dir, result_dir, video_name, processing_config)
            else:
                result = self._run_normal_inference(frame_dir, result_dir, video_name, processing_config)

            # ✅ 修正: CSVパスをログ出力
            if result.get("success") and result.get("csv_path"):
                self.logger.info(f"✅ CSV保存完了: {result['csv_path']}")

            return result

        except Exception as e:
            self.logger.error(f"検出・追跡エラー {video_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "video_name": video_name,
                "frame_dir": str(frame_dir),
                "error": str(e),
                "success": False
            }

    def _build_processing_config(self, kwargs: Dict) -> Dict[str, Any]:
        """処理設定の構築"""
        config = {
            "confidence_threshold": self.config.get('processing.detection.confidence_threshold', 0.3),
            "tracking_config": self.config.get('models.tracking_config', 'bytetrack.yaml'),
            "save_visualizations": kwargs.get('save_visualizations', True),
            "model_path": self.config.pose_model,
            "device": self.config.get('processing.device', 'cpu'),
            "batch_size": self.config.get('processing.batch_size', 8),
            "max_memory_gb": self.config.get('processing.max_memory_gb', 3.0),
            "streaming_output": self.config.get('processing.streaming_output', True)
        }

        # タイル推論設定を追加
        if self.tile_enabled:
            tile_config = self.config.get('processing.tile_inference', {})
            config["tile_inference"] = {
                "enabled": True,
                "tile_size": tuple(tile_config.get('tile_size', [640, 640])),
                "overlap_ratio": tile_config.get('overlap_ratio', 0.2),
                "use_adaptive": tile_config.get('use_adaptive', False),
                "max_tiles_per_frame": tile_config.get('max_tiles_per_frame', 16),
                "nms_threshold": tile_config.get('nms_threshold', 0.5)
            }

        return config

    def _run_tile_inference(self, frame_dir: Path, result_dir: Path, video_name: str, config: Dict) -> Dict[str, Any]:
        """タイル推論実行"""
        try:
            from yolopose_analyzer import analyze_frames_with_tile_inference

            result = analyze_frames_with_tile_inference(
                str(frame_dir),
                str(result_dir),
                config["model_path"],
                config
            )

            if "error" not in result:
                result.update({
                    "video_name": video_name,
                    "inference_type": "tile",
                    "success": True
                })

            return result

        except ImportError as e:
            self.logger.error(f"タイル推論関数のインポートエラー: {e}")
            return {"error": f"tile_inference_import_failed: {e}", "success": False}
        except Exception as e:
            self.logger.error(f"タイル推論エラー: {e}")
            return {"error": f"tile_inference_failed: {e}", "success": False}

    def _run_normal_inference(self, frame_dir: Path, result_dir: Path, video_name: str, config: Dict) -> Dict[str, Any]:
        """通常推論実行"""
        try:
            from yolopose_analyzer import analyze_frames_with_tracking_memory_efficient

            result = analyze_frames_with_tracking_memory_efficient(
                str(frame_dir),
                str(result_dir),
                model_path=config["model_path"],
                config=config
            )

            if "error" not in result:
                result.update({
                    "video_name": video_name,
                    "inference_type": "normal",
                    "success": True
                })

            return result

        except ImportError as e:
            self.logger.error(f"yolopose_analyzer モジュールのインポートエラー: {e}")
            return {
                "video_name": video_name,
                "frame_dir": str(frame_dir),
                "error": f"analyzer_import_failed: {e}",
                "success": False,
                "suggestion": "yolopose_analyzer.py が存在することを確認してください"
            }
        except Exception as e:
            self.logger.error(f"通常推論エラー {video_name}: {e}")
            return {
                "video_name": video_name,
                "frame_dir": str(frame_dir),
                "error": str(e),
                "success": False
            }