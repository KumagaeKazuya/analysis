"""
動画処理モジュール（統一エラーハンドリング完全対応版）

🔧 主な改善点:
1. 全メソッドに統一エラーハンドラーデコレータ適用
2. ResponseBuilder形式のレスポンス統一
3. ErrorContextによる詳細なエラートラッキング
4. カスタムエラークラスの使用
5. スタックトレース自動記録
"""

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# 🔧 統一エラーハンドラーからインポート
from utils.error_handler import (
    VideoProcessingError,
    FileIOError,
    ConfigurationError,
    ResponseBuilder,
    handle_errors,
    validate_inputs,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """動画処理クラス（統一エラーハンドリング完全対応版）"""

    @handle_errors(logger=logger, error_category=ErrorCategory.INITIALIZATION)
    def __init__(self, config):
        """
        初期化（統一エラーハンドリング対応版）
        
        Args:
            config: 設定オブジェクト
        """
        with ErrorContext("VideoProcessor初期化", logger=logger) as ctx:
            if not config:
                raise ConfigurationError(
                    "設定オブジェクトがNullです",
                    details={"config": str(config)}
                )
                
            self.config = config
            self.logger = logging.getLogger(__name__)
            
            # タイル推論の有効性チェック
            self.tile_enabled = config.get('processing.tile_inference.enabled', False)
            
            # 必要な設定項目の存在確認
            required_configs = ['video_dir', 'model_dir', 'output_dir']
            missing_configs = []
            
            for req_config in required_configs:
                if not hasattr(config, req_config) or not getattr(config, req_config):
                    missing_configs.append(req_config)
            
            if missing_configs:
                raise ConfigurationError(
                    f"必要な設定が不足しています: {missing_configs}",
                    details={"missing_configs": missing_configs}
                )
            
            ctx.add_info("tile_enabled", self.tile_enabled)
            ctx.add_info("video_dir", getattr(config, 'video_dir', 'N/A'))
            
            if self.tile_enabled:
                self.logger.info("🔲 タイル推論モードで初期化")
            else:
                self.logger.info("📋 通常推論モードで初期化")

    @validate_inputs(
        video_path=lambda x: isinstance(x, (str, Path)),
        output_dir=lambda x: isinstance(x, (str, Path))
    )
    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def extract_frames(
        self, 
        video_path: Path, 
        output_dir: Path, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        動画からフレームを抽出（統一エラーハンドリング対応版）
        
        Args:
            video_path: 動画ファイルのパス
            output_dir: フレーム出力ディレクトリ
            **kwargs: interval_sec などの追加設定
            
        Returns:
            ResponseBuilder形式の結果辞書
        """
        with ErrorContext(f"フレーム抽出: {Path(video_path).name}", logger=self.logger) as ctx:
            # 出力ディレクトリ作成
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            interval_sec = kwargs.get(
                'interval_sec', 
                self.config.get('processing.frame_sampling.interval_sec', 2)
            )

            ctx.add_info("video_path", str(video_path))
            ctx.add_info("output_dir", str(output_dir))
            ctx.add_info("interval_sec", interval_sec)

            # 動画ファイルの存在確認
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileIOError(
                    f"動画ファイルが見つかりません: {video_path}",
                    details={"video_path": str(video_path)},
                    suggestions=["ファイルパスが正しいか確認してください"]
                )

            # ファイルサイズチェック
            file_size = video_path.stat().st_size
            if file_size < 1024:  # 1KB未満
                raise VideoProcessingError(
                    f"動画ファイルのサイズが異常に小さいです: {file_size} bytes",
                    details={"video_path": str(video_path), "file_size": file_size}
                )

            ctx.add_info("file_size_mb", file_size / (1024*1024))
            self.logger.info(f"フレーム抽出開始: {video_path.name} ({file_size/(1024*1024):.1f}MB)")

            # フレーム抽出実行
            try:
                # 既存のframe_samplerを使用
                from frame_sampler import sample_frames
                sample_frames(str(video_path), str(output_dir), interval_sec=interval_sec)

            except ImportError:
                self.logger.warning("frame_samplerモジュールが見つかりません。内蔵実装を使用")
                self._sample_frames_builtin(str(video_path), str(output_dir), interval_sec)

            # 抽出されたフレーム確認
            frame_files = list(output_dir.glob("*.jpg"))

            if not frame_files:
                raise VideoProcessingError(
                    "フレーム抽出に失敗しました（フレームが0件）",
                    details={
                        "video_path": str(video_path),
                        "output_dir": str(output_dir),
                        "interval_sec": interval_sec
                    },
                    severity=ErrorSeverity.ERROR,
                    suggestions=[
                        "動画ファイルが破損していないか確認してください",
                        "interval_secの値を確認してください",
                        "出力ディレクトリの書き込み権限を確認してください"
                    ]
                )

            ctx.add_info("frame_count", len(frame_files))
            self.logger.info(f"フレーム抽出完了: {len(frame_files)}フレーム")

            return ResponseBuilder.success(
                data={
                    "video_path": str(video_path),
                    "output_dir": str(output_dir),
                    "frame_count": len(frame_files),
                    "interval_sec": interval_sec,
                    "frame_files": [f.name for f in frame_files[:5]],  # サンプル表示
                    "file_size_mb": file_size / (1024*1024)
                },
                message=f"フレーム抽出完了: {len(frame_files)}フレーム"
            )

    @handle_errors(logger=logger, error_category=ErrorCategory.IO, suppress_exceptions=False)
    def _sample_frames_builtin(self, video_path: str, save_dir: str, interval_sec: int = 2):
        """
        内蔵フレーム抽出実装（統一エラーハンドリング対応版）

        Args:
            video_path: 動画ファイルパス
            save_dir: 保存ディレクトリ
            interval_sec: フレーム抽出間隔（秒）
        """
        with ErrorContext("内蔵フレーム抽出", logger=self.logger) as ctx:
            os.makedirs(save_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise VideoProcessingError(
                    f"動画ファイルを開けません: {video_path}",
                    details={"video_path": video_path},
                    suggestions=[
                        "ファイルが破損していないか確認してください",
                        "対応形式(.mp4, .avi, .mov等)か確認してください"
                    ]
                )

            # 動画情報取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frame_interval = int(fps * interval_sec) if fps > 0 else 30
            frame_count = 0
            saved_count = 0

            ctx.add_info("fps", fps)
            ctx.add_info("total_frames", total_frames)
            ctx.add_info("duration_sec", duration)
            ctx.add_info("frame_interval", frame_interval)

            self.logger.info(f"動画情報: FPS={fps:.1f}, 総フレーム数={total_frames}, 長さ={duration:.1f}秒")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    filename = os.path.join(
                        save_dir,
                        f"{os.path.basename(video_path)}_frame{frame_count:06d}.jpg"
                    )
                    success = cv2.imwrite(filename, frame)

                    if not success:
                        self.logger.warning(f"フレーム保存失敗: {filename}")
                    else:
                        saved_count += 1

                frame_count += 1

            cap.release()

            ctx.add_info("total_frames_processed", frame_count)
            ctx.add_info("saved_frames", saved_count)

            if saved_count == 0:
                raise VideoProcessingError(
                    "フレーム保存に完全に失敗しました",
                    details={
                        "processed_frames": frame_count,
                        "saved_frames": saved_count,
                        "save_dir": save_dir
                    }
                )

            self.logger.info(f"内蔵実装でフレーム抽出完了: {saved_count}フレーム")

    @validate_inputs(
        frame_dir=lambda x: isinstance(x, (str, Path)),
        video_name=lambda x: isinstance(x, str) and len(x) > 0
    )
    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def run_detection_tracking(
        self,
        frame_dir: Path,
        video_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        検出・追跡処理を実行（統一エラーハンドリング対応版）

        Args:
            frame_dir: フレームディレクトリ
            video_name: 動画名
            **kwargs: 追加設定

        Returns:
            ResponseBuilder形式の処理結果
        """
        with ErrorContext(f"検出・追跡: {video_name}", logger=self.logger) as ctx:
            self.logger.info(f"🔍 検出・追跡開始: {video_name}")

            # フレームディレクトリの存在確認
            frame_dir = Path(frame_dir)
            if not frame_dir.exists():
                raise FileIOError(
                    f"フレームディレクトリが見つかりません: {frame_dir}",
                    details={"frame_dir": str(frame_dir)},
                    suggestions=["フレーム抽出が正常に完了しているか確認してください"]
                )

            # フレームファイルの確認
            frame_files = list(frame_dir.glob("*.jpg"))
            if not frame_files:
                raise VideoProcessingError(
                    f"処理対象のフレームが見つかりません: {frame_dir}",
                    details={
                        "frame_dir": str(frame_dir),
                        "available_files": list(frame_dir.glob("*"))[:10]  # 他のファイルも表示
                    },
                    suggestions=[
                        "フレーム抽出が正常に完了しているか確認してください",
                        "ファイル拡張子が.jpgか確認してください"
                    ]
                )

            # 結果ディレクトリのパスを明確に指定
            video_base_dir = frame_dir.parent
            result_dir = video_base_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)

            ctx.add_info("frame_dir", str(frame_dir))
            ctx.add_info("result_dir", str(result_dir))
            ctx.add_info("frame_count", len(frame_files))

            self.logger.info(f"📂 結果保存先: {result_dir}")
            self.logger.info(f"📊 処理対象: {len(frame_files)}フレーム")

            # 処理設定を構築
            processing_config = self._build_processing_config(kwargs)
            ctx.add_info("tile_enabled", self.tile_enabled)
            ctx.add_info("config_keys", list(processing_config.keys()))

            # タイル推論が有効な場合
            if self.tile_enabled:
                result = self._run_tile_inference(
                    frame_dir, result_dir, video_name, processing_config
                )
            else:
                result = self._run_normal_inference(
                    frame_dir, result_dir, video_name, processing_config
                )

            # 結果の検証
            if not result.get("success", False):
                error_info = result.get("error", {})
                error_message = error_info.get('message', 'unknown error') if isinstance(error_info, dict) else str(error_info)

                raise VideoProcessingError(
                    f"検出・追跡処理が失敗しました: {error_message}",
                    details=result,
                    suggestions=[
                        "モデルファイルが存在し、読み込み可能か確認してください",
                        "フレームディレクトリにアクセス可能か確認してください",
                        "メモリ使用量を確認してください"
                    ]
                )

            # CSVパスをログ出力
            csv_path = result.get("data", {}).get("csv_path") or result.get("csv_path")
            if csv_path and Path(csv_path).exists():
                csv_size = Path(csv_path).stat().st_size
                self.logger.info(f"✅ CSV保存完了: {csv_path} ({csv_size} bytes)")
            else:
                self.logger.warning("⚠️ CSV生成されませんでした")

            return result

    @handle_errors(logger=logger, error_category=ErrorCategory.VALIDATION)
    def _build_processing_config(self, kwargs: Dict) -> Dict[str, Any]:
        """
        処理設定の構築（統一エラーハンドリング対応版）

        🔧 修正ポイント: None値を確実にフォールバック

        Args:
            kwargs: 追加設定

        Returns:
            完全な処理設定の辞書
        """
        with ErrorContext("処理設定構築", logger=self.logger) as ctx:
            # モデルパスの確認
            model_path = getattr(self.config, 'pose_model', None)
            if not model_path:
                raise ConfigurationError(
                    "pose_modelが設定されていません",
                    details={"config_keys": list(vars(self.config).keys())}
                )

            # 基本設定
            config = {
                # 検出設定
                "confidence_threshold": self.config.get(
                    'processing.detection.confidence_threshold', 0.3
                ),

                # tracking_configのNone対策
                "tracking_config": self.config.get('models.tracking_config') or 'bytetrack.yaml',

                # 可視化設定
                "save_visualizations": kwargs.get('save_visualizations', True),

                # モデルパス
                "model_path": model_path,

                # deviceのNone対策
                "device": self.config.get('processing.device') or 'cpu',

                # メモリ・バッチ設定
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

            # 設定値の検証
            if config["confidence_threshold"] < 0 or config["confidence_threshold"] > 1:
                self.logger.warning(f"confidence_threshold値が異常: {config['confidence_threshold']}")
                config["confidence_threshold"] = 0.3

            if config["batch_size"] <= 0:
                self.logger.warning(f"batch_size値が異常: {config['batch_size']}")
                config["batch_size"] = 8

            ctx.add_info("config_keys", list(config.keys()))
            ctx.add_info("confidence_threshold", config["confidence_threshold"])
            ctx.add_info("batch_size", config["batch_size"])

            return config

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def _run_tile_inference(
        self,
        frame_dir: Path,
        result_dir: Path,
        video_name: str,
        config: Dict
    ) -> Dict[str, Any]:
        """
        タイル推論実行（統一エラーハンドリング対応版）

        Args:
            frame_dir: フレームディレクトリ
            result_dir: 結果出力ディレクトリ
            video_name: 動画名
            config: 処理設定

        Returns:
            ResponseBuilder形式の処理結果
        """
        with ErrorContext("タイル推論", logger=self.logger) as ctx:
            try:
                from yolopose_analyzer import analyze_frames_with_tile_inference

                ctx.add_info("inference_type", "tile")

                result = analyze_frames_with_tile_inference(
                    str(frame_dir),
                    str(result_dir),
                    config["model_path"],
                    config
                )

                # ResponseBuilder形式への変換
                if isinstance(result, dict):
                    if result.get("success", False):
                        # 成功時
                        result_data = result.get("data", result)
                        result_data["video_name"] = video_name
                        result_data["inference_type"] = "tile"

                        return ResponseBuilder.success(
                            data=result_data,
                            message="タイル推論完了"
                        )
                    elif "error" in result:
                        # エラー時
                        return result  # すでにResponseBuilder形式
                    else:
                        # 従来形式（success/errorキーなし）
                        result["video_name"] = video_name
                        result["inference_type"] = "tile"

                        return ResponseBuilder.success(
                            data=result,
                            message="タイル推論完了"
                        )
                else:
                    raise VideoProcessingError(
                        "タイル推論の結果が不正な形式です",
                        details={"result_type": type(result).__name__}
                    )

            except ImportError as e:
                raise VideoProcessingError(
                    "タイル推論関数のインポートに失敗しました",
                    details={
                        "error": str(e),
                        "module": "yolopose_analyzer",
                        "function": "analyze_frames_with_tile_inference"
                    },
                    suggestions=[
                        "yolopose_analyzer.py が存在し、analyze_frames_with_tile_inference関数が定義されているか確認してください",
                        "通常推論に切り替えることを検討してください"
                    ],
                    original_exception=e
                )

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def _run_normal_inference(
        self,
        frame_dir: Path,
        result_dir: Path,
        video_name: str,
        config: Dict
    ) -> Dict[str, Any]:
        """
        通常推論実行（統一エラーハンドリング対応版）

        Args:
            frame_dir: フレームディレクトリ
            result_dir: 結果出力ディレクトリ
            video_name: 動画名
            config: 処理設定

        Returns:
            ResponseBuilder形式の処理結果
        """
        with ErrorContext("通常推論", logger=self.logger) as ctx:
            try:
                from yolopose_analyzer import analyze_frames_with_tracking_memory_efficient

                ctx.add_info("inference_type", "normal")

                result = analyze_frames_with_tracking_memory_efficient(
                    str(frame_dir),
                    str(result_dir),
                    model_path=config["model_path"],
                    config=config
                )

                # ResponseBuilder形式への変換
                if isinstance(result, dict):
                    if result.get("success", False):
                        # 成功時
                        result_data = result.get("data", result)
                        result_data["video_name"] = video_name
                        result_data["inference_type"] = "normal"

                        return ResponseBuilder.success(
                            data=result_data,
                            message="通常推論完了"
                        )
                    elif "error" in result:
                        # エラー時（すでにResponseBuilder形式）
                        return result
                    else:
                        # 従来形式（success/errorキーなし）
                        result["video_name"] = video_name
                        result["inference_type"] = "normal"

                        return ResponseBuilder.success(
                            data=result,
                            message="通常推論完了"
                        )
                else:
                    raise VideoProcessingError(
                        "通常推論の結果が不正な形式です",
                        details={"result_type": type(result).__name__}
                    )

            except ImportError as e:
                raise VideoProcessingError(
                    "yolopose_analyzerモジュールのインポートに失敗しました",
                    details={
                        "error": str(e),
                        "module": "yolopose_analyzer",
                        "function": "analyze_frames_with_tracking_memory_efficient"
                    },
                    suggestions=[
                        "yolopose_analyzer.py が存在することを確認してください",
                        "必要な依存関係がインストールされているか確認してください"
                    ],
                    original_exception=e
                )

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)  # INFO → PROCESSING に変更
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        処理統計を取得（統一エラーハンドリング対応版）
        
        Returns:
            統計情報の辞書
        """
        with ErrorContext("処理統計取得", logger=self.logger) as ctx:
            stats = {
                "tile_enabled": self.tile_enabled,
                "config_summary": {
                    "video_dir": getattr(self.config, 'video_dir', 'N/A'),
                    "model_dir": getattr(self.config, 'model_dir', 'N/A'),
                    "output_dir": getattr(self.config, 'output_dir', 'N/A'),
                    "pose_model": getattr(self.config, 'pose_model', 'N/A')
                },
                "processing_capabilities": {
                    "frame_sampling": True,
                    "detection_tracking": True,
                    "tile_inference": self.tile_enabled,
                    "memory_efficient": True
                }
            }

            ctx.add_info("stats_collected", True)
            return stats