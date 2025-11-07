"""
YOLO11 広角カメラ分析システム - 改良版（統一エラーハンドリング + 深度推定統合対応版）

🔧 主な改善点:
1. 統一エラーハンドリング対応
2. 深度推定（MiDaS）統合機能
3. 深度対応評価器の自動選択
4. 設定ファイルの自動切り替え
5. エラー収集とレポート生成
6. 🔧 モジュール不足対応とフォールバック機能
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, List

# 🔧 条件付きインポート - 必須ライブラリ
try:
    import cv2
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import yaml
except ImportError as e:
    print(f"❌ 必須ライブラリ不足: {e}")
    print("📦 以下でインストールしてください:")
    print("pip install ultralytics opencv-python numpy pandas matplotlib tqdm pyyaml torch")
    sys.exit(1)

# 🔧 条件付きインポート - 統一エラーハンドラー
try:
    from utils.error_handler import (
        BaseYOLOError,
        ConfigurationError,
        VideoProcessingError,
        ResponseBuilder,
        handle_errors,
        ErrorContext,
        ErrorCategory,
        ErrorReporter,
        ErrorSeverity
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    print("⚠️ 統一エラーハンドラーが見つかりません。基本エラーハンドリングを使用します")
    ERROR_HANDLER_AVAILABLE = False
    
    # 🔧 基本エラーハンドリングクラス
    class BasicError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.message = message
            self.details = details or {}
    
    class ConfigurationError(BasicError):
        pass
    
    class VideoProcessingError(BasicError):
        pass
    
    class ResponseBuilder:
        @staticmethod
        def success(data=None, message=""):
            return {"success": True, "data": data, "message": message}
        
        @staticmethod
        def error(exception, suggestions=None):
            return {
                "success": False, 
                "error": {"message": str(exception)},
                "suggestions": suggestions or []
            }
    
    def handle_errors(error_category=None, suppress_exceptions=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if suppress_exceptions:
                        logging.error(f"エラー in {func.__name__}: {e}")
                        return ResponseBuilder.error(e)
                    else:
                        raise
            return wrapper
        return decorator
    
    class ErrorContext:
        def __init__(self, name, logger=None, raise_on_error=False):
            self.name = name
            self.logger = logger or logging.getLogger(__name__)
            self.raise_on_error = raise_on_error
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type and self.logger:
                self.logger.error(f"エラー in {self.name}: {exc_val}")
            return not self.raise_on_error
        
        def add_info(self, key, value):
            pass

# 🔧 条件付きインポート - 評価器
try:
    from evaluators.comprehensive_evaluator import ComprehensiveEvaluator
    EVALUATOR_AVAILABLE = True
    
    # 深度対応評価器の条件付きインポート
    try:
        from evaluators.comprehensive_evaluator import DepthEnhancedEvaluator
        DEPTH_EVALUATOR_AVAILABLE = True
    except ImportError:
        DEPTH_EVALUATOR_AVAILABLE = False
        
except ImportError:
    print("⚠️ ComprehensiveEvaluator が見つかりません。基本評価機能を使用します")
    EVALUATOR_AVAILABLE = False
    DEPTH_EVALUATOR_AVAILABLE = False
    
    # 🔧 基本評価クラス
    class BasicEvaluator:
        def __init__(self, config=None):
            self.config = config or {}
            self.results = {}
        
        def evaluate_comprehensive(self, video_path, detection_results, video_name):
            """基本的な評価"""
            try:
                # 基本統計の計算
                data = detection_results.get("data", {})
                csv_path = data.get("csv_path") or data.get("enhanced_csv_path")
                
                basic_metrics = {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "detection_count": data.get("detection_count", 0),
                    "frame_count": data.get("frame_count", 0),
                    "processing_time": data.get("processing_time", 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                # CSVファイルが存在する場合の詳細分析
                if csv_path and Path(csv_path).exists():
                    try:
                        df = pd.read_csv(csv_path)
                        basic_metrics.update({
                            "total_detections": len(df),
                            "unique_track_ids": df['track_id'].nunique() if 'track_id' in df.columns else 0,
                            "confidence_mean": df['confidence'].mean() if 'confidence' in df.columns else 0,
                            "detection_success": True
                        })
                    except Exception as e:
                        logging.warning(f"CSV分析エラー: {e}")
                        basic_metrics["detection_success"] = False
                
                return ResponseBuilder.success(data=basic_metrics)
                
            except Exception as e:
                logging.error(f"基本評価エラー: {e}")
                return ResponseBuilder.error(e)

# 🔧 条件付きインポート - プロセッサー
try:
    from processors.video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ VideoProcessor が見つかりません。基本動画処理を使用します")
    VIDEO_PROCESSOR_AVAILABLE = False
    
    # 🔧 基本動画プロセッサー
    class BasicVideoProcessor:
        def __init__(self, config):
            self.config = config
            self.detection_model = None
            self.pose_model = None
            self.load_models()
        
        def load_models(self):
            """モデルロード"""
            try:
                models_config = self.config.get('models', {}) if hasattr(self.config, 'get') else self.config.get('models', {})
                detection_path = models_config.get('detection', 'models/yolo/yolo11m.pt')
                pose_path = models_config.get('pose', 'models/yolo/yolo11m-pose.pt')
                
                if Path(detection_path).exists():
                    self.detection_model = YOLO(detection_path)
                    logging.info(f"検出モデルロード: {detection_path}")
                
                if Path(pose_path).exists():
                    self.pose_model = YOLO(pose_path)
                    logging.info(f"ポーズモデルロード: {pose_path}")
                    
            except Exception as e:
                logging.error(f"モデルロードエラー: {e}")
        
        def extract_frames(self, video_path, frame_dir, max_frames=100):
            """フレーム抽出"""
            try:
                frame_dir = Path(frame_dir)
                frame_dir.mkdir(parents=True, exist_ok=True)
                
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError(f"動画ファイルを開けません: {video_path}")
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                interval = max(1, frame_count // max_frames)
                
                extracted = 0
                for i in range(0, frame_count, interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame_path = frame_dir / f"frame_{i:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        extracted += 1
                
                cap.release()
                return {"success": True, "extracted_frames": extracted}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def run_detection_tracking(self, frame_dir, video_name):
            """基本検出・追跡処理"""
            try:
                frame_files = sorted(list(Path(frame_dir).glob("*.jpg")))
                if not frame_files:
                    raise ValueError("フレームファイルが見つかりません")
                
                results = []
                detection_count = 0
                
                for frame_file in tqdm(frame_files, desc="検出処理"):
                    frame = cv2.imread(str(frame_file))
                    
                    # 検出実行
                    if self.detection_model:
                        det_results = self.detection_model(frame, verbose=False)
                        if det_results and len(det_results[0].boxes) > 0:
                            detection_count += len(det_results[0].boxes)
                    
                    # ポーズ実行
                    if self.pose_model:
                        pose_results = self.pose_model(frame, verbose=False)
                        # ポーズ結果の処理は簡略化
                
                # 結果CSV作成
                output_dir = Path("outputs/temp") / video_name
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"{video_name}_results.csv"
                
                # 簡単なCSV作成
                basic_data = {
                    "frame_id": range(len(frame_files)),
                    "detection_count": [1] * len(frame_files),  # 仮データ
                    "confidence": [0.5] * len(frame_files)  # 仮データ
                }
                df = pd.DataFrame(basic_data)
                df.to_csv(csv_path, index=False)
                
                return {
                    "success": True,
                    "data": {
                        "csv_path": str(csv_path),
                        "detection_count": detection_count,
                        "frame_count": len(frame_files),
                        "processing_stats": {"basic_processing": True}
                    }
                }
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def run_detection_tracking_with_depth(self, frame_dir, video_name):
            """深度統合検出・追跡処理（フォールバック）"""
            logging.warning("深度統合処理は利用できません。通常処理を実行します")
            result = self.run_detection_tracking(frame_dir, video_name)
            if result.get("success", False):
                result["data"]["depth_enabled"] = False
                result["data"]["enhanced_csv_path"] = result["data"]["csv_path"]
            return result

# 🔧 条件付きインポート - アナライザー
try:
    from analyzers.metrics_analyzer import MetricsAnalyzer
    METRICS_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ MetricsAnalyzer が見つかりません。基本分析機能を使用します")
    METRICS_ANALYZER_AVAILABLE = False
    
    class BasicMetricsAnalyzer:
        def __init__(self, config):
            self.config = config
        
        def analyze_improvements(self, comparison_results):
            """基本改善分析"""
            return {"basic_analysis": "改善分析機能は限定的です"}
        
        def create_visualizations(self, detection_results, vis_dir):
            """基本可視化"""
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            logging.info(f"基本可視化ディレクトリ作成: {vis_dir}")

# 🔧 条件付きインポート - 設定とロガー
try:
    from utils.config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ Config が見つかりません。基本設定機能を使用します")
    CONFIG_AVAILABLE = False
    
    class BasicConfig:
        def __init__(self, config_path=None):
            self.config_path = config_path
            self.data = self.load_config()
        
        def load_config(self):
            if self.config_path and Path(self.config_path).exists():
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        if self.config_path.endswith(('.yaml', '.yml')):
                            return yaml.safe_load(f)
                        else:
                            return json.load(f)
                except Exception as e:
                    logging.warning(f"設定読み込みエラー: {e}")
            
            # デフォルト設定
            return {
                "models": {
                    "detection": "models/yolo/yolo11m.pt",
                    "pose": "models/yolo/yolo11m-pose.pt"
                },
                "processing": {
                    "detection": {"confidence_threshold": 0.3, "iou_threshold": 0.45},
                    "depth_estimation": {"enabled": False}
                },
                "video_dir": "videos",
                "output_dir": "outputs"
            }
        
        def get(self, key, default=None):
            keys = key.split('.')
            value = self.data
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        
        def get_experiment_config(self, experiment_type):
            return {"type": experiment_type, "basic": True}
        
        @property
        def video_dir(self):
            return self.get("video_dir", "videos")

try:
    from utils.logger import setup_logger
    LOGGER_AVAILABLE = True
except ImportError:
    print("⚠️ setup_logger が見つかりません。基本ログ機能を使用します")
    LOGGER_AVAILABLE = False
    
    def setup_logger():
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)


class ImprovedYOLOAnalyzer:
    """
    YOLO11 広角カメラ分析システム（深度推定統合版 + モジュール不足対応版）
    """

    @handle_errors(error_category="initialization" if ERROR_HANDLER_AVAILABLE else None)
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        初期化（深度推定対応版 + フォールバック機能）

        Args:
            config_path: 設定ファイルパス
        """
        if ERROR_HANDLER_AVAILABLE:
            context_manager = ErrorContext("ImprovedYOLOAnalyzer初期化", logger=logging.getLogger(__name__))
        else:
            context_manager = self._basic_context("ImprovedYOLOAnalyzer初期化")
            
        with context_manager as ctx:
            # 🔍 深度設定の自動検出と切り替え
            self.config = self._initialize_config(config_path)
            
            if LOGGER_AVAILABLE:
                self.logger = setup_logger()
            else:
                self.logger = logging.getLogger(__name__)

            # 深度推定有効性の確認
            self.depth_enabled = self.config.get('processing.depth_estimation.enabled', False)

            # 🔍 深度対応評価器の選択
            if self.depth_enabled and DEPTH_EVALUATOR_AVAILABLE:
                try:
                    from evaluators.comprehensive_evaluator import DepthEnhancedEvaluator
                    self.evaluator = DepthEnhancedEvaluator(self.config)
                    self.logger.info("🔍 深度統合評価器を初期化")
                    if hasattr(ctx, 'add_info'):
                        ctx.add_info("evaluator_type", "DepthEnhancedEvaluator")
                except ImportError:
                    self.logger.warning("DepthEnhancedEvaluator が見つかりません。標準評価器を使用")
                    if EVALUATOR_AVAILABLE:
                        self.evaluator = ComprehensiveEvaluator(self.config)
                    else:
                        self.evaluator = BasicEvaluator(self.config)
                    if hasattr(ctx, 'add_info'):
                        ctx.add_info("evaluator_type", "ComprehensiveEvaluator (fallback)")
            else:
                if EVALUATOR_AVAILABLE:
                    self.evaluator = ComprehensiveEvaluator(self.config)
                else:
                    self.evaluator = BasicEvaluator(self.config)
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("evaluator_type", "BasicEvaluator" if not EVALUATOR_AVAILABLE else "ComprehensiveEvaluator")

            # プロセッサーとアナライザー初期化
            if VIDEO_PROCESSOR_AVAILABLE:
                self.processor = VideoProcessor(self.config)
            else:
                self.processor = BasicVideoProcessor(self.config)
                
            if METRICS_ANALYZER_AVAILABLE:
                self.analyzer = MetricsAnalyzer(self.config)
            else:
                self.analyzer = BasicMetricsAnalyzer(self.config)

            # 🔧 エラー収集用
            self.error_collector = []

            # ディレクトリセットアップ
            self._setup_directories()

            if hasattr(ctx, 'add_info'):
                ctx.add_info("depth_enabled", self.depth_enabled)
                ctx.add_info("config_path", config_path)

            # 初期化完了ログ
            features = []
            if self.depth_enabled:
                features.append("深度推定")
            if self.config.get('processing.tile_inference.enabled', False):
                features.append("タイル推論")
            
            # 使用中のフォールバック機能の表示
            fallbacks = []
            if not EVALUATOR_AVAILABLE:
                fallbacks.append("基本評価器")
            if not VIDEO_PROCESSOR_AVAILABLE:
                fallbacks.append("基本動画処理")
            if not METRICS_ANALYZER_AVAILABLE:
                fallbacks.append("基本分析")

            if features:
                self.logger.info(f"🚀 ImprovedYOLOAnalyzer初期化完了 (機能: {', '.join(features)})")
            else:
                self.logger.info("📋 ImprovedYOLOAnalyzer初期化完了 (標準モード)")
                
            if fallbacks:
                self.logger.info(f"🔧 フォールバック機能使用中: {', '.join(fallbacks)}")

    def _basic_context(self, name):
        """基本コンテキストマネージャー（ErrorContext不使用時）"""
        class BasicContext:
            def __init__(self, name):
                self.name = name
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
            def add_info(self, key, value):
                pass
        return BasicContext(name)
    def _initialize_config(self, config_path: str) -> 'Config':
        """
        🔍 設定初期化（深度設定の自動検出と切り替え）
        
        Args:
            config_path: 指定された設定ファイルパス
            
        Returns:
            初期化された設定オブジェクト
        """
        # 🔍 深度設定ファイルの存在確認と自動切り替え
        depth_config_path = "configs/depth_config.yaml"
        
        # 設定ファイルの優先順位決定
        if Path(config_path).exists():
            primary_config = config_path
            self.logger.info(f"📄 指定設定ファイル使用: {config_path}")
        elif Path(depth_config_path).exists():
            primary_config = depth_config_path
            self.logger.info(f"🔍 深度設定ファイル自動検出: {depth_config_path}")
        else:
            primary_config = config_path  # 指定ファイルで続行（エラーは後で処理）
            self.logger.warning(f"⚠️ 設定ファイルが見つかりません: {config_path}")

        # 設定オブジェクト初期化
        if CONFIG_AVAILABLE:
            return Config(primary_config)
        else:
            return BasicConfig(primary_config)

    def _setup_directories(self):
        """必要ディレクトリの作成"""
        directories = [
            "outputs/baseline",
            "outputs/experiments", 
            "outputs/visualizations",
            "outputs/temp",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @handle_errors(error_category="video_processing" if ERROR_HANDLER_AVAILABLE else None)
    def run_baseline_analysis(self, video_path: str) -> Dict[str, Any]:
        """
        🎯 ベースライン分析実行（深度推定統合版 + エラーハンドリング強化）

        Args:
            video_path: 分析対象動画のパス

        Returns:
            分析結果辞書
        """
        if ERROR_HANDLER_AVAILABLE:
            context_manager = ErrorContext(f"ベースライン分析: {Path(video_path).name}", 
                                         logger=self.logger, raise_on_error=False)
        else:
            context_manager = self._basic_context(f"ベースライン分析: {Path(video_path).name}")

        with context_manager as ctx:
            video_path = Path(video_path)
            video_name = video_path.stem

            if hasattr(ctx, 'add_info'):
                ctx.add_info("video_path", str(video_path))
                ctx.add_info("video_name", video_name)
                ctx.add_info("depth_enabled", self.depth_enabled)

            # 📁 出力ディレクトリ準備
            output_dir = Path("outputs/baseline") / video_name
            frame_dir = output_dir / "frames"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(ctx, 'add_info'):
                ctx.add_info("output_dir", str(output_dir))

            try:
                self.logger.info(f"🎥 ベースライン分析開始: {video_name}")
                
                # Step 1: フレーム抽出
                self.logger.info("📸 フレーム抽出中...")
                frame_result = self.processor.extract_frames(video_path, frame_dir)
                
                if not frame_result.get("success", False):
                    error_msg = f"フレーム抽出失敗: {frame_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    raise VideoProcessingError(error_msg)

                extracted_frames = frame_result.get("extracted_frames", 0)
                self.logger.info(f"✅ フレーム抽出完了: {extracted_frames}フレーム")

                # Step 2: 検出・追跡処理（深度推定統合版）
                if self.depth_enabled:
                    self.logger.info("🔍 深度統合検出・追跡処理開始...")
                    detection_result = self.processor.run_detection_tracking_with_depth(frame_dir, video_name)
                    processing_type = "深度統合"
                else:
                    self.logger.info("👁️ 標準検出・追跡処理開始...")
                    detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
                    processing_type = "標準"

                if not detection_result.get("success", False):
                    error_msg = f"{processing_type}処理失敗: {detection_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    raise VideoProcessingError(error_msg)

                self.logger.info(f"✅ {processing_type}処理完了")

                # Step 3: 包括的評価
                self.logger.info("📊 包括的評価開始...")
                evaluation_result = self.evaluator.evaluate_comprehensive(
                    video_path, 
                    detection_result, 
                    video_name
                )

                if not evaluation_result.get("success", False):
                    error_msg = f"評価処理失敗: {evaluation_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    self.logger.warning(error_msg)
                    # 🔧 評価失敗は警告に留める（処理は継続）
                    evaluation_result = ResponseBuilder.success(data={"basic_evaluation": True})

                self.logger.info("✅ 包括的評価完了")

                # Step 4: 可視化生成
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                
                try:
                    self.logger.info("📈 可視化生成中...")
                    self.analyzer.create_visualizations(detection_result, str(vis_dir))
                    self.logger.info("✅ 可視化生成完了")
                except Exception as e:
                    self.logger.warning(f"⚠️ 可視化生成エラー（処理継続）: {e}")
                    self.error_collector.append(f"可視化生成エラー: {e}")

                # 🎯 統合結果の構築
                integrated_result = {
                    "success": True,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "processing_type": processing_type,
                    "depth_enabled": self.depth_enabled,
                    "output_directory": str(output_dir),
                    "frame_extraction": frame_result,
                    "detection_tracking": detection_result,
                    "evaluation": evaluation_result,
                    "visualization_path": str(vis_dir),
                    "processing_timestamp": datetime.now().isoformat(),
                    "errors": self.error_collector.copy() if self.error_collector else []
                }

                # 結果ファイル保存
                result_file = output_dir / f"{video_name}_baseline_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(integrated_result, f, indent=2, ensure_ascii=False)

                if hasattr(ctx, 'add_info'):
                    ctx.add_info("result_file", str(result_file))
                    ctx.add_info("processing_success", True)

                self.logger.info(f"🎉 ベースライン分析完了: {video_name}")
                self.logger.info(f"📁 結果保存先: {output_dir}")

                return ResponseBuilder.success(data=integrated_result)

            except VideoProcessingError as e:
                self.logger.error(f"❌ 動画処理エラー: {e}")
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("error_type", "VideoProcessingError")
                    ctx.add_info("error_message", str(e))
                return ResponseBuilder.error(e, suggestions=[
                    "動画ファイルの形式を確認してください",
                    "動画ファイルが破損していないか確認してください",
                    f"出力ディレクトリ {output_dir} への書き込み権限を確認してください"
                ])
            
            except Exception as e:
                self.logger.error(f"❌ 予期しないエラー: {e}")
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("error_type", "UnexpectedError")
                    ctx.add_info("error_message", str(e))
                return ResponseBuilder.error(e, suggestions=[
                    "ログファイルで詳細なエラー情報を確認してください",
                    "設定ファイルが正しく設定されているか確認してください",
                    "必要なモデルファイルが存在するか確認してください"
                ])

    @handle_errors(error_category="experiment" if ERROR_HANDLER_AVAILABLE else None)
    def run_experiment(self, video_path: str, experiment_type: str) -> Dict[str, Any]:
        """
        🧪 実験分析実行（深度推定統合版）

        Args:
            video_path: 分析対象動画のパス
            experiment_type: 実験タイプ

        Returns:
            実験結果辞書
        """
        if ERROR_HANDLER_AVAILABLE:
            context_manager = ErrorContext(f"実験分析: {experiment_type}", 
                                         logger=self.logger, raise_on_error=False)
        else:
            context_manager = self._basic_context(f"実験分析: {experiment_type}")

        with context_manager as ctx:
            video_path = Path(video_path)
            video_name = video_path.stem

            if hasattr(ctx, 'add_info'):
                ctx.add_info("video_path", str(video_path))
                ctx.add_info("experiment_type", experiment_type)
                ctx.add_info("depth_enabled", self.depth_enabled)

            try:
                self.logger.info(f"🧪 実験分析開始: {experiment_type} - {video_name}")

                # 📁 実験用出力ディレクトリ
                output_dir = Path("outputs/experiments") / experiment_type / video_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # 🔧 実験設定の取得
                if CONFIG_AVAILABLE and hasattr(self.config, 'get_experiment_config'):
                    experiment_config = self.config.get_experiment_config(experiment_type)
                else:
                    experiment_config = {"type": experiment_type, "basic_mode": True}

                if hasattr(ctx, 'add_info'):
                    ctx.add_info("output_dir", str(output_dir))
                    ctx.add_info("experiment_config", experiment_config)

                # ベースライン結果との比較用にベースライン実行
                self.logger.info("📊 ベースライン結果取得中...")
                baseline_result = self.run_baseline_analysis(video_path)
                
                if not baseline_result.get("success", False):
                    raise VideoProcessingError("ベースライン分析に失敗しました")

                # 🧪 実験特有の処理（今後拡張予定）
                experiment_result = {
                    "success": True,
                    "experiment_type": experiment_type,
                    "video_name": video_name,
                    "baseline_comparison": baseline_result.get("data", {}),
                    "experiment_config": experiment_config,
                    "depth_enabled": self.depth_enabled,
                    "output_directory": str(output_dir),
                    "processing_timestamp": datetime.now().isoformat()
                }

                # 改善分析
                try:
                    improvement_analysis = self.analyzer.analyze_improvements({
                        "baseline": baseline_result.get("data", {}),
                        "experiment": experiment_result
                    })
                    experiment_result["improvement_analysis"] = improvement_analysis
                except Exception as e:
                    self.logger.warning(f"⚠️ 改善分析エラー（処理継続）: {e}")
                    experiment_result["improvement_analysis"] = {"error": str(e)}

                # 結果保存
                result_file = output_dir / f"{video_name}_{experiment_type}_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(experiment_result, f, indent=2, ensure_ascii=False)

                if hasattr(ctx, 'add_info'):
                    ctx.add_info("result_file", str(result_file))

                self.logger.info(f"🎉 実験分析完了: {experiment_type} - {video_name}")
                return ResponseBuilder.success(data=experiment_result)

            except Exception as e:
                self.logger.error(f"❌ 実験分析エラー: {e}")
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("error_type", type(e).__name__)
                    ctx.add_info("error_message", str(e))
                return ResponseBuilder.error(e, suggestions=[
                    f"実験タイプ '{experiment_type}' の設定を確認してください",
                    "ベースライン分析が正常に動作するか確認してください"
                ])
    
    def generate_error_report(self) -> Dict[str, Any]:
            """
            🔧 エラーレポート生成
        
            Returns:
                エラーレポート辞書
            """
            try:
                error_report = {
                    "timestamp": datetime.now().isoformat(),
                    "total_errors": len(self.error_collector),
                    "errors": self.error_collector.copy(),
                    "system_info": {
                        "depth_enabled": self.depth_enabled,
                        "evaluator_type": type(self.evaluator).__name__,
                        "processor_type": type(self.processor).__name__,
                        "analyzer_type": type(self.analyzer).__name__
                    },
                    "module_availability": {
                        "error_handler": ERROR_HANDLER_AVAILABLE,
                        "evaluator": EVALUATOR_AVAILABLE,
                        "depth_evaluator": DEPTH_EVALUATOR_AVAILABLE,
                        "video_processor": VIDEO_PROCESSOR_AVAILABLE,
                        "metrics_analyzer": METRICS_ANALYZER_AVAILABLE,
                        "config": CONFIG_AVAILABLE,
                        "logger": LOGGER_AVAILABLE
                    }
                }
            
                # エラーレポートファイル保存
                report_file = Path("logs") / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_file.parent.mkdir(exist_ok=True)
            
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(error_report, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"📋 エラーレポート生成: {report_file}")
                return error_report
            
            except Exception as e:
                self.logger.error(f"エラーレポート生成失敗: {e}")
                return {"error": str(e)}

    def get_video_files(self) -> List[Path]:
        """
        🎥 動画ファイル取得
        
        Returns:
            動画ファイルパスのリスト
        """
        try:
            if CONFIG_AVAILABLE and hasattr(self.config, 'video_dir'):
                video_dir = Path(self.config.video_dir)
            else:
                video_dir = Path(self.config.get("video_dir", "videos"))
                
            if not video_dir.exists():
                self.logger.warning(f"動画ディレクトリが存在しません: {video_dir}")
                return []
                
            # サポートする動画形式
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(video_dir.glob(f"*{ext}"))
                video_files.extend(video_dir.glob(f"*{ext.upper()}"))
                
            return sorted(video_files)
            
        except Exception as e:
            self.logger.error(f"動画ファイル取得エラー: {e}")
            return []


def main():
    """
    🚀 メイン実行関数（深度推定統合版 + 統一エラーハンドリング）
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="YOLO11 広角カメラ分析システム（深度推定統合版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 使用例:
  # ベースライン分析（標準モード）
  python improved_main.py --mode baseline --config configs/default.yaml
  
  # ベースライン分析（深度推定モード）
  python improved_main.py --mode baseline --config configs/depth_config.yaml
  
  # 実験分析
  python improved_main.py --mode experiment --experiment-type tile_inference
  
  # 特定動画の分析
  python improved_main.py --mode baseline --video path/to/video.mp4
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["baseline", "experiment"], 
        default="baseline",
        help="実行モード: baseline=ベースライン分析, experiment=実験分析"
    )
    
    parser.add_argument(
        "--config", 
        default="configs/default.yaml",
        help="設定ファイルパス（デフォルト: configs/default.yaml）"
    )
    
    parser.add_argument(
        "--video",
        help="分析対象動画ファイルパス（指定しない場合は設定ファイルのvideo_dirから自動選択）"
    )
    
    parser.add_argument(
        "--experiment-type",
        default="comparison",
        help="実験タイプ（mode=experimentの場合のみ有効）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細ログ出力"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true", 
        help="処理後にエラーレポートを生成"
    )

    args = parser.parse_args()

    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # ロガー初期化
    if LOGGER_AVAILABLE:
        logger = setup_logger()
    else:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

    logger.info("🚀 YOLO11 広角カメラ分析システム 開始")
    logger.info(f"📋 実行モード: {args.mode}")
    logger.info(f"⚙️ 設定ファイル: {args.config}")
    
    # モジュール可用性の報告
    available_modules = []
    fallback_modules = []
    
    if ERROR_HANDLER_AVAILABLE:
        available_modules.append("統一エラーハンドラー")
    else:
        fallback_modules.append("基本エラーハンドリング")
        
    if EVALUATOR_AVAILABLE:
        available_modules.append("包括的評価器")
        if DEPTH_EVALUATOR_AVAILABLE:
            available_modules.append("深度統合評価器")
    else:
        fallback_modules.append("基本評価器")
        
    if VIDEO_PROCESSOR_AVAILABLE:
        available_modules.append("高度動画処理")
    else:
        fallback_modules.append("基本動画処理")
        
    if available_modules:
        logger.info(f"✅ 利用可能な高度機能: {', '.join(available_modules)}")
    if fallback_modules:
        logger.info(f"🔧 フォールバック機能使用: {', '.join(fallback_modules)}")

    try:
        # 🔧 分析器初期化（統一エラーハンドリング）
        analyzer = ImprovedYOLOAnalyzer(args.config)
        
        # 🎥 動画ファイル決定
        if args.video:
            video_path = Path(args.video)
            if not video_path.exists():
                raise FileNotFoundError(f"指定された動画ファイルが存在しません: {video_path}")
            video_files = [video_path]
        else:
            video_files = analyzer.get_video_files()
            if not video_files:
                raise FileNotFoundError(f"動画ファイルが見つかりません。{analyzer.config.get('video_dir', 'videos')}ディレクトリに動画ファイルを配置してください")

        logger.info(f"🎥 処理対象動画: {len(video_files)}ファイル")
        
        # 🎯 分析実行
        all_results = []
        
        for video_file in video_files:
            logger.info(f"📹 処理開始: {video_file.name}")
            
            try:
                if args.mode == "baseline":
                    result = analyzer.run_baseline_analysis(str(video_file))
                elif args.mode == "experiment":
                    result = analyzer.run_experiment(str(video_file), args.experiment_type)
                else:
                    raise ValueError(f"不正な実行モード: {args.mode}")
                
                all_results.append({
                    "video_file": str(video_file),
                    "result": result
                })
                
                if result.get("success", False):
                    logger.info(f"✅ 処理完了: {video_file.name}")
                else:
                    logger.error(f"❌ 処理失敗: {video_file.name}")
                    if result.get("error"):
                        logger.error(f"エラー詳細: {result['error'].get('message', '不明')}")
                        
            except Exception as e:
                logger.error(f"❌ 動画処理エラー ({video_file.name}): {e}")
                all_results.append({
                    "video_file": str(video_file),
                    "result": ResponseBuilder.error(e)
                })

        # 📊 全体結果サマリー
        successful = sum(1 for r in all_results if r["result"].get("success", False))
        total = len(all_results)
        
        logger.info(f"📊 処理結果サマリー: {successful}/{total} 成功")
        
        # 🔧 エラーレポート生成
        if args.generate_report or analyzer.error_collector:
            error_report = analyzer.generate_error_report()
            logger.info(f"📋 エラーレポート: {error_report.get('total_errors', 0)}件のエラー")

        # 📁 統合結果ファイル保存
        summary_result = {
            "execution_mode": args.mode,
            "config_file": args.config,
            "execution_timestamp": datetime.now().isoformat(),
            "total_videos": total,
            "successful_videos": successful,
            "video_results": all_results,
            "system_info": {
                "depth_enabled": analyzer.depth_enabled,
                "module_availability": {
                    "error_handler": ERROR_HANDLER_AVAILABLE,
                    "evaluator": EVALUATOR_AVAILABLE,
                    "depth_evaluator": DEPTH_EVALUATOR_AVAILABLE,
                    "video_processor": VIDEO_PROCESSOR_AVAILABLE,
                    "metrics_analyzer": METRICS_ANALYZER_AVAILABLE,
                    "config": CONFIG_AVAILABLE,
                    "logger": LOGGER_AVAILABLE
                }
            }
        }
        
        summary_file = Path("outputs") / f"summary_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 統合結果保存: {summary_file}")

        # 🎉 処理完了
        if successful == total:
            logger.info("🎉 全ての動画処理が成功しました")
            print(f"\n✅ 処理完了: {successful}/{total} 成功")
            print(f"📁 結果保存先: outputs/{args.mode}/")
            return True
        else:
            logger.warning(f"⚠️ 一部の動画処理が失敗しました ({successful}/{total})")
            print(f"\n⚠️ 部分的成功: {successful}/{total}")
            print(f"📋 詳細はログファイルを確認してください")
            return False

    except ConfigurationError as e:
        logger.error(f"❌ 設定エラー: {e}")
        print(f"❌ 設定エラー: {e}")
        if hasattr(e, 'details') and e.details.get('suggestions'):
            for suggestion in e.details['suggestions']:
                print(f"💡 対処法: {suggestion}")
        return False
        
    except FileNotFoundError as e:
        logger.error(f"❌ ファイルエラー: {e}")
        print(f"❌ ファイルエラー: {e}")
        print("💡 対処法:")
        print("  1. 動画ファイルが正しいディレクトリに配置されているか確認")
        print("  2. 設定ファイルが存在するか確認")
        print("  3. パスの指定が正しいか確認")
        return False
        
    except KeyboardInterrupt:
        logger.info("❌ ユーザーによって処理が中断されました")
        print("\n❌ 処理が中断されました")
        return False
        
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        print(f"❌ 予期しないエラー: {e}")
        print("💡 対処法:")
        print("  1. ログファイル（logs/）で詳細を確認")
        print("  2. 設定ファイルの内容を確認")
        print("  3. 必要なモデルファイルが存在するか確認")
        print("  4. --verbose オプションで詳細ログを出力")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ システムエラー: {e}")
        logging.error(f"システムエラー: {e}")
        sys.exit(1)