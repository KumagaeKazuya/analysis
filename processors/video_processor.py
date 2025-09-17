import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

class VideoProcessor:
    """動画処理を統括するクラス - 循環インポート修正版"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_frames(self, video_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        動画からフレームを抽出
        循環インポートを避けるため、関数内で遅延インポート
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        interval_sec = kwargs.get('interval_sec', self.config.get('processing.frame_sampling.interval_sec', 2))

        try:
            self.logger.info(f"フレーム抽出開始: {video_path.name}")

            # 遅延インポートで循環参照を回避
            try:
                from frame_sampler import sample_frames
            except ImportError:
                # フォールバック: 直接実装
                self.logger.warning("frame_sampler モジュールが見つかりません。内蔵実装を使用します。")
                self._sample_frames_builtin(str(video_path), str(output_dir), interval_sec)
            else:
                sample_frames(str(video_path), str(output_dir), interval_sec=interval_sec)

            # 抽出されたフレーム数を確認
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
        """
        フォールバック用の内蔵フレーム抽出実装
        """
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
        """
        検出・追跡処理を実行
        循環インポートを避けるため、関数内で遅延インポート
        """
        try:
            self.logger.info(f"検出・追跡開始: {video_name}")

            # 結果出力ディレクトリ
            result_dir = Path(self.config.output_dir) / "results" / video_name
            result_dir.mkdir(parents=True, exist_ok=True)

            # 処理設定を構築
            processing_config = {
                "confidence_threshold": self.config.get('processing.detection.confidence_threshold', 0.3),
                "tracking_config": self.config.get('models.tracking_config', 'bytetrack.yaml'),
                "save_visualizations": kwargs.get('save_visualizations', True)
            }

            # 遅延インポートで循環参照を回避
            try:
                from yolopose_analyzer import analyze_frames_with_tracking
            except ImportError as e:
                self.logger.error(f"yolopose_analyzer モジュールのインポートエラー: {e}")
                return {
                    "video_name": video_name,
                    "frame_dir": str(frame_dir),
                    "error": f"analyzer_import_failed: {e}",
                    "success": False,
                    "suggestion": "yolopose_analyzer.py が存在することを確認してください"
                }

            # 解析実行
            analysis_result = analyze_frames_with_tracking(
                str(frame_dir),
                str(result_dir),
                model_path=self.config.pose_model,
                config=processing_config
            )

            self.logger.info(f"検出・追跡完了: {video_name}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"検出・追跡エラー {video_name}: {e}")
            return {
                "video_name": video_name,
                "frame_dir": str(frame_dir),
                "error": str(e),
                "success": False
            }