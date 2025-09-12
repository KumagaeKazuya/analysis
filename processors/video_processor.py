# 動画処理クラス

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from frame_sampler import sample_frames  # 既存のモジュールを活用
from yolopose_analyzer import analyze_frames_with_tracking  # 既存のモジュールを活用

class VideoProcessor:
    """動画処理を統括するクラス"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_frames(self, video_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        動画からフレームを抽出

        Args:
            video_path: 動画ファイルパス
            output_dir: 出力ディレクトリ
            **kwargs: 追加パラメータ

        Returns:
            dict: 抽出結果
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 設定から間隔を取得
        interval_sec = kwargs.get('interval_sec', self.config.get('processing.frame_sampling.interval_sec', 2))

        try:
            self.logger.info(f"フレーム抽出開始: {video_path.name}")

            # 既存のsample_frames関数を使用
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

    def run_detection_tracking(self, frame_dir: Path, video_name: str, **kwargs) -> Dict[str, Any]:
        """
        検出・追跡処理を実行

        Args:
            frame_dir: フレームディレクトリ
            video_name: 動画名
            **kwargs: 追加パラメータ

        Returns:
            dict: 処理結果
        """
        try:
            self.logger.info(f"検出・追跡開始: {video_name}")

            # 結果出力ディレクトリ
            result_dir = Path(self.config.output_dir) / "results" / video_name

            # 処理設定を構築
            processing_config = {
                "confidence_threshold": self.config.get('processing.detection.confidence_threshold', 0.3),
                "tracking_config": self.config.get('models.tracking_config', 'bytetrack.yaml'),
                "save_visualizations": kwargs.get('save_visualizations', True)
            }

            # 既存の解析関数を使用（修正版）
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

    def process_video_batch(self, video_paths: List[Path], output_base_dir: Path) -> Dict[str, Any]:
        """
        複数動画の一括処理

        Args:
            video_paths: 動画パスのリスト
            output_base_dir: 出力ベースディレクトリ

        Returns:
            dict: 一括処理結果
        """
        batch_results = {
            "total_videos": len(video_paths),
            "successful_videos": 0,
            "failed_videos": 0,
            "results": []
        }

        self.logger.info(f"一括処理開始: {len(video_paths)}本の動画")

        for i, video_path in enumerate(video_paths, 1):
            self.logger.info(f"[{i}/{len(video_paths)}] 処理中: {video_path.name}")

            try:
                video_name = video_path.stem

                # 1. フレーム抽出
                frame_dir = output_base_dir / "frames" / video_name
                frame_result = self.extract_frames(video_path, frame_dir)

                if not frame_result["success"]:
                    raise Exception(f"フレーム抽出失敗: {frame_result.get('error', 'unknown error')}")

                # 2. 検出・追跡
                detection_result = self.run_detection_tracking(frame_dir, video_name)

                if "error" in detection_result:
                    raise Exception(f"検出・追跡失敗: {detection_result['error']}")

                # 結果をまとめる
                video_result = {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "frame_extraction": frame_result,
                    "detection_tracking": detection_result,
                    "success": True
                }

                batch_results["results"].append(video_result)
                batch_results["successful_videos"] += 1

                self.logger.info(f"✅ 完了: {video_name}")

            except Exception as e:
                self.logger.error(f"❌ 動画処理失敗 {video_path.name}: {e}")

                error_result = {
                    "video_name": video_path.stem,
                    "video_path": str(video_path),
                    "error": str(e),
                    "success": False
                }

                batch_results["results"].append(error_result)
                batch_results["failed_videos"] += 1

        success_rate = batch_results["successful_videos"] / batch_results["total_videos"]
        self.logger.info(
            f"一括処理完了: {batch_results['successful_videos']}/{batch_results['total_videos']} "
            f"({success_rate:.1%}) 成功"
        )

        return batch_results

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        動画の基本情報を取得

        Args:
            video_path: 動画ファイルパス

        Returns:
            dict: 動画情報
        """
        try:
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                return {"error": "動画ファイルを開けませんでした"}

            info = {
                "filename": video_path.name,
                "path": str(video_path),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_sec": None,
                "file_size_mb": video_path.stat().st_size / (1024 * 1024)
            }

            # 動画の長さを計算
            if info["fps"] > 0:
                info["duration_sec"] = info["frame_count"] / info["fps"]

            cap.release()
            return info

        except Exception as e:
            return {"error": f"動画情報取得エラー: {e}"}

    def validate_video_files(self, video_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        動画ファイルの検証

        Args:
            video_paths: 検証する動画パスのリスト

        Returns:
            dict: 有効/無効な動画ファイルのリスト
        """
        valid_videos = []
        invalid_videos = []

        for video_path in video_paths:
            if not video_path.exists():
                self.logger.warning(f"ファイルが見つかりません: {video_path}")
                invalid_videos.append(video_path)
                continue

            # 動画情報を取得して有効性をチェック
            video_info = self.get_video_info(video_path)

            if "error" in video_info:
                self.logger.warning(f"無効な動画ファイル {video_path.name}: {video_info['error']}")
                invalid_videos.append(video_path)
            else:
                valid_videos.append(video_path)

        self.logger.info(f"動画ファイル検証完了: 有効 {len(valid_videos)}, 無効 {len(invalid_videos)}")

        return {
            "valid": valid_videos,
            "invalid": invalid_videos
        }