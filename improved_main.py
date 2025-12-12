"""
YOLO11 広角カメラ分析システム - 改良版（統一エラーハンドリング + 深度推定統合対応版）

🔧 主な改善点:
1. 統一エラーハンドリング対応
2. 深度推定（MiDaS）統合機能
3. 深度対応評価器の自動選択
4. 設定ファイルの自動切り替え
5. エラー収集とレポート生成
6. 🔧 モジュール不足対応とフォールバック機能
7. 🔧 ErrorCategory.EVALUATION 対応（Stage 5修正）
8. 🔧 条件付きインポートの強化（Stage 6修正）
9. 🔧 直接インポート削除とフォールバック完全統合
"""
import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, List
import time
from datetime import datetime  # 🔧 追加
import traceback
import platform
from utils.camera_calibration import undistort_with_json
import numpy as np

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
    print("✅ 必須ライブラリのインポート成功")
except ImportError as e:
    print(f"❌ 必須ライブラリ不足: {e}")
    print("📦 以下でインストールしてください:")
    print("pip install ultralytics opencv-python numpy pandas matplotlib tqdm pyyaml torch")
    sys.exit(1)

# 🔧 条件付きインポート - 統一エラーハンドラー（完全統合版）
ERROR_HANDLER_AVAILABLE = False
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
    print("✅ 統一エラーハンドラーが利用可能です")
except ImportError as e:
    print(f"⚠️ 統一エラーハンドラーが見つかりません: {e}")
    print("🔧 基本エラーハンドリングを使用します")
    ERROR_HANDLER_AVAILABLE = False

# 🔧 条件付きインポート - yolopose_analyzer（XLargeモデル確実使用版）
YOLOPOSE_ANALYZER_AVAILABLE = False

try:
    from yolopose_analyzer import analyze_frames_with_tracking_enhanced
    YOLOPOSE_ANALYZER_AVAILABLE = True
    print("✅ yolopose_analyzer が利用可能です")
except ImportError as e:
    print(f"⚠️ yolopose_analyzer が見つかりません: {e}")
    YOLOPOSE_ANALYZER_AVAILABLE = False
    
    # 🔧 基本エラーハンドリングクラス（完全版）
    class BaseYOLOError(Exception):
        def __init__(self, message, details=None):
            super().__init__(message)
            self.message = message
            self.details = details or {}
    
    class ConfigurationError(BaseYOLOError):
        pass
    
    class VideoProcessingError(BaseYOLOError):
        pass
    
    # 🔧 ErrorCategory の完全実装
    class ErrorCategory:
        INITIALIZATION = "initialization"
        VIDEO_PROCESSING = "video_processing"
        EVALUATION = "evaluation"  # 🔧 Stage 5エラー解決用
        EXPERIMENT = "experiment"
        CONFIGURATION = "configuration"
        MODEL_LOADING = "model_loading"
        MODEL = "model"  # 🔧 追加
        DEPTH_PROCESSING = "depth_processing"
        PROCESSING = "processing"
        IO_OPERATIONS = "io_operations"
    
    # 🔧 ErrorSeverity の完全実装
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    # Line 89-250付近のResponseBuilderクラスを以下で完全置換:

    class ResponseBuilder:
        """統一レスポンスビルダー（完全後方互換版）"""

        @staticmethod
        def success(data=None, message=""):
            """成功レスポンス"""
            return {"success": True, "data": data, "message": message}

        @staticmethod
        def error(
            error=None,
            include_traceback: bool = True,
            suggestions=None,
            message=None,                         # 🔧 yolopose_analyzer用
            details=None,                         # 🔧 yolopose_analyzer用
            exception=None,                       # 🔧 後方互換性
            **kwargs                              # 🔧 完全互換のため
        ):
            """エラーレスポンス（完全互換API）"""
    
            # 引数の正規化（複数のパターンに対応）
            target_error = error or exception
    
            if message:
                # messageが直接指定された場合（yolopose_analyzer用）
                response = {
                    "success": False,
                    "error": {
                        "error_type": "CustomError",
                        "message": message,
                        "category": "unknown",
                        "severity": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            elif hasattr(target_error, 'to_dict') and callable(getattr(target_error, 'to_dict')):
                # BaseYOLOErrorの場合
                response = {
                    "success": False,
                    "error": target_error.to_dict()
                }
            elif isinstance(target_error, Exception):
                # 標準Exceptionの場合
                response = {
                    "success": False,
                    "error": {
                        "error_type": type(target_error).__name__,
                        "message": str(target_error),
                        "category": "unknown",
                        "severity": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
                if include_traceback:
                    import traceback
                    response["error"]["traceback"] = traceback.format_exc()
            elif isinstance(target_error, str):
                # 文字列エラーの場合
                response = {
                    "success": False,
                    "error": {
                        "error_type": "StringError",
                        "message": target_error,
                        "category": "unknown", 
                        "severity": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                # フォールバック
                response = {
                    "success": False,
                    "error": {
                        "error_type": "UnknownError",
                        "message": "不明なエラーが発生しました",
                        "category": "unknown",
                        "severity": "error",
                        "timestamp": datetime.now().isoformat()
                    }
                }
    
            # suggestions追加
            if suggestions:
                response["suggestions"] = suggestions
        
            # details追加（重要！）
            if details:
                response["details"] = details
        
            # その他のkwargs対応
            for key, value in kwargs.items():
                if key not in response:
                    response[key] = value
        
            return response

        @staticmethod
        def validation_error(field=None, message=None, details=None, **kwargs):
            """バリデーションエラー（完全後方互換）"""
            if message:
                error_message = message
            elif field:
                error_message = f"バリデーションエラー: {field}"
            else:
                error_message = "バリデーションエラー"
            
            response = {
                "success": False,
                "error": {
                    "error_type": "ValidationError",
                    "message": error_message,
                    "field": field,
                    "category": "validation",
                    "severity": "error",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
            # details引数のサポート（重要！）
            if details:
                response["details"] = details
            
            # その他のkwargs対応
            for key, value in kwargs.items():
                if key not in response:
                    response[key] = value
            
            return response

        @staticmethod
        def processing_error(step=None, message=None, details=None, **kwargs):
            """処理エラー（完全互換）"""
            error_message = message or f"処理エラー: {step}"
            result = {
                "success": False,
                "error": {
                    "error_type": "ProcessingError", 
                    "message": error_message,
                    "step": step,
                    "category": "processing",
                    "severity": "error",
                    "timestamp": datetime.now().isoformat()
               }
            }
        
            if details:
                result["details"] = details
            
            # その他のkwargs対応
            for key, value in kwargs.items():
                if key not in result:
                    result[key] = value
            
            return result

        @staticmethod
        def configuration_error(config_key=None, message=None, details=None, **kwargs):
            """設定エラー（完全互換）"""
            error_message = message or f"設定エラー: {config_key}"
            result = {
                "success": False,
                "error": {
                    "error_type": "ConfigurationError",
                    "message": error_message,
                    "config_key": config_key,
                    "category": "configuration",
                    "severity": "error",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
            if details:
                result["details"] = details
            
            # その他のkwargs対応
            for key, value in kwargs.items():
                if key not in result:
                    result[key] = value
            
            return result

        @staticmethod
        def model_error(model_path=None, message=None, details=None, **kwargs):
            """モデルエラー（完全互換）"""
            error_message = message or f"モデルエラー: {model_path}"
            result = {
                "success": False,
                "error": {
                    "error_type": "ModelError",
                    "message": error_message,
                    "model_path": model_path,
                    "category": "model",
                    "severity": "error",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
            if details:
                result["details"] = details
            
            # その他のkwargs対応
            for key, value in kwargs.items():
                if key not in result:
                    result[key] = value
            
            return result
    
    class ErrorContext:
        def __init__(self, name, logger=None, raise_on_error=False):
            self.name = name
            self.logger = logger or logging.getLogger(__name__)
            self.raise_on_error = raise_on_error
            
        def __enter__(self):
            self.logger.debug(f"🔍 エラーコンテキスト開始: {self.name}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type and self.logger:
                self.logger.error(f"❌ エラー in {self.name}: {exc_val}")
            elif self.logger:
                self.logger.debug(f"✅ エラーコンテキスト正常終了: {self.name}")
            return not self.raise_on_error
        
        def add_info(self, key, value):
            self.logger.debug(f"📝 {self.name} - {key}: {value}")

    class ErrorReporter:
        def __init__(self):
            self.errors = []
        
        def add_error(self, error, context=None):
            self.errors.append({"error": str(error), "context": context, "timestamp": datetime.now().isoformat()})

# 🔧 条件付きインポート - 評価器（完全統合版）
EVALUATOR_AVAILABLE = False
DEPTH_EVALUATOR_AVAILABLE = False
COMPREHENSIVE_EVALUATOR_AVAILABLE = False
ComprehensiveEvaluator = None
DepthEnhancedEvaluator = None

try:
    # evaluatorsモジュールの段階的チェック
    import evaluators
    print("✅ evaluators モジュールが見つかりました")
    
    try:
        from evaluators.comprehensive_evaluator import ComprehensiveEvaluator
        COMPREHENSIVE_EVALUATOR_AVAILABLE = True
        EVALUATOR_AVAILABLE = True
        print("✅ ComprehensiveEvaluator が利用可能です")
        
        try:
            from evaluators.comprehensive_evaluator import DepthEnhancedEvaluator
            DEPTH_EVALUATOR_AVAILABLE = True
            print("✅ DepthEnhancedEvaluator が利用可能です")
        except (ImportError, AttributeError) as e:
            print(f"⚠️ DepthEnhancedEvaluator が利用できません: {e}")
            DEPTH_EVALUATOR_AVAILABLE = False
            
    except ImportError as e:
        print(f"⚠️ ComprehensiveEvaluator のインポートに失敗: {e}")
        COMPREHENSIVE_EVALUATOR_AVAILABLE = False
        EVALUATOR_AVAILABLE = False
        
except ImportError as e:
    print(f"⚠️ evaluators モジュールが見つかりません: {e}")
    EVALUATOR_AVAILABLE = False
    COMPREHENSIVE_EVALUATOR_AVAILABLE = False
    DEPTH_EVALUATOR_AVAILABLE = False

# 🔧 条件付きインポート - 設定管理（完全統合版）
CONFIG_AVAILABLE = False
Config = None

try:
    from utils.config import Config
    CONFIG_AVAILABLE = True
    print("✅ Config が利用可能です")
except ImportError as e:
    print(f"⚠️ Config が見つかりません: {e}")
    print("🔧 基本設定機能を使用します")
    CONFIG_AVAILABLE = False

# 🔧 BasicEvaluator（フォールバック用・完全版）
if not EVALUATOR_AVAILABLE:
    print("🔧 基本評価機能を使用します")
    
    class BasicEvaluator:
        def __init__(self, config=None):
            self.config = config or {}
            self.results = {}
            self.logger = logging.getLogger(__name__)
        
        def evaluate_comprehensive(self, video_path, detection_results, video_name):
            """基本的な評価（完全版）"""
            try:
                self.logger.info(f"🔍 基本評価開始: {video_name}")
                
                # データ抽出の改善
                if isinstance(detection_results, dict):
                    if detection_results.get("success", False):
                        data = detection_results.get("data", {})
                    else:
                        self.logger.warning("detection_results が失敗状態です")
                        data = {}
                else:
                    data = {}
                
                csv_path = data.get("csv_path") or data.get("enhanced_csv_path")
                
                basic_metrics = {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "detection_count": data.get("detection_count", 0),
                    "frame_count": data.get("frame_count", 0),
                    "processing_time": data.get("processing_time", 0),
                    "timestamp": datetime.now().isoformat(),
                    "evaluator_type": "BasicEvaluator",
                    "depth_enabled": data.get("depth_enabled", False),
                    "processing_stats": data.get("processing_stats", {})
                }
                
                # CSVファイル分析の改善
                if csv_path and Path(csv_path).exists():
                    try:
                        df = pd.read_csv(csv_path)
                        self.logger.info(f"📊 CSV分析: {len(df)}行のデータ")
                        
                        csv_metrics = {
                            "total_detections": len(df),
                            "detection_success": True,
                            "csv_path": str(csv_path)
                        }
                        
                        # カラムベースの詳細分析
                        available_columns = df.columns.tolist()
                        csv_metrics["available_columns"] = available_columns
                        
                        if 'track_id' in df.columns:
                            csv_metrics["unique_track_ids"] = df['track_id'].nunique()
                        
                        if 'confidence' in df.columns:
                            csv_metrics["confidence_mean"] = float(df['confidence'].mean())
                            csv_metrics["confidence_std"] = float(df['confidence'].std())
                            csv_metrics["confidence_min"] = float(df['confidence'].min())
                            csv_metrics["confidence_max"] = float(df['confidence'].max())
                        
                        # 深度関連カラムの詳細確認
                        depth_columns = [col for col in df.columns if 'depth' in col.lower()]
                        if depth_columns:
                            csv_metrics["depth_columns"] = depth_columns
                            csv_metrics["depth_analysis_available"] = True
                            # 深度データの統計
                            for depth_col in depth_columns:
                                if df[depth_col].dtype in ['float64', 'int64']:
                                    csv_metrics[f"{depth_col}_mean"] = float(df[depth_col].mean())
                        
                        basic_metrics.update(csv_metrics)
                        
                    except Exception as e:
                        self.logger.warning(f"CSV分析エラー: {e}")
                        basic_metrics.update({
                            "detection_success": False,
                            "csv_error": str(e),
                            "csv_path": str(csv_path) if csv_path else None
                        })
                else:
                    self.logger.warning(f"CSV ファイルが見つかりません: {csv_path}")
                    basic_metrics["csv_available"] = False
                
                self.logger.info(f"✅ 基本評価完了: {video_name}")
                return ResponseBuilder.success(data=basic_metrics)
                
            except Exception as e:
                self.logger.error(f"❌ 基本評価エラー: {e}")
                return ResponseBuilder.error(e, suggestions=[
                    "評価データの形式を確認してください",
                    "CSVファイルが正しく生成されているか確認してください",
                    "入力データの型を確認してください"
                ])
        
        def evaluate_with_depth(self, video_path, detection_results, video_name):
            """深度対応評価（BasicEvaluator版）"""
            self.logger.info(f"🔍 深度対応基本評価: {video_name}")
            result = self.evaluate_comprehensive(video_path, detection_results, video_name)
            
            if result.get("success", False):
                result["data"]["depth_evaluator_type"] = "BasicEvaluator"
                result["data"]["depth_support"] = "limited"
            
            return result

# 🔧 条件付きインポート - プロセッサー（完全統合版）
VIDEO_PROCESSOR_AVAILABLE = False
VideoProcessor = None

try:
    from processors.video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
    print("✅ VideoProcessor が利用可能です")
except ImportError as e:
    print(f"⚠️ VideoProcessor が見つかりません: {e}")
    print("🔧 基本動画処理を使用します")
    VIDEO_PROCESSOR_AVAILABLE = False

# 🔧 BasicVideoProcessor（フォールバック用・完全版）
if not VIDEO_PROCESSOR_AVAILABLE:
    class BasicVideoProcessor:
        def __init__(self, config):
            """基本動画プロセッサー初期化（完全版）"""
            self.config = config
            self.logger = logging.getLogger(__name__)
            self.processing_stats = {}  # 🔧 統計情報辞書を初期化
    
            if hasattr(config, 'get'):
                self.output_dir = Path(config.get('output_dir', 'outputs'))
                self.max_frames = config.get('processing.max_frames', 100)
            else:
                self.output_dir = Path('outputs')
                self.max_frames = 100
    
            self.output_dir.mkdir(exist_ok=True)
            self.logger.info(f"🎬 基本動画プロセッサー初期化完了")
        
        def load_models(self):
            """モデルロード（パス重複修正版）"""
            try:
                if hasattr(self.config, 'get'):
                    models_config = self.config.get('models', {})
                elif isinstance(self.config, dict):
                    models_config = self.config.get('models', {})
                else:
                    models_config = {}
                
                # ⚡ パス重複を防ぐ修正
                detection_path = models_config.get('detection', 'models/yolo11x.pt')
                pose_path = models_config.get('pose', 'models/yolo11x-pose.pt')
                
                # パス重複チェック
                if detection_path.startswith('models/models/'):
                    detection_path = detection_path.replace('models/models/', 'models/')
                if pose_path.startswith('models/models/'):
                    pose_path = pose_path.replace('models/models/', 'models/')
                
                self.logger.info(f"🔍 モデルロード開始")
                self.logger.info(f"📊 検出モデルパス: {detection_path}")
                self.logger.info(f"📊 ポーズモデルパス: {pose_path}")
                
                # 検出モデル
                if Path(detection_path).exists():
                    self.detection_model = YOLO(detection_path)
                    self.logger.info(f"✅ 検出モデル: {detection_path}")
                else:
                    self.logger.warning(f"⚠️ 検出モデル未発見: {detection_path}")
                    # フォールバック: 自動ダウンロード
                    self.detection_model = YOLO('yolo11x.pt')
                    self.logger.info("✅ 検出モデル: 自動ダウンロード")

                # ポーズモデル
                if Path(pose_path).exists():
                    self.pose_model = YOLO(pose_path)
                    self.logger.info(f"✅ ポーズモデル: {pose_path}")
                else:
                    self.logger.warning(f"⚠️ ポーズモデル未発見: {pose_path}")
                    # フォールバック: 自動ダウンロード
                    self.pose_model = YOLO('yolo11x-pose.pt')
                    self.logger.info("✅ ポーズモデル: 自動ダウンロード")
                
                self.logger.info("✅ 全モデルロード完了")
                
            except Exception as e:
                self.logger.error(f"❌ モデルロードエラー: {e}")
                raise
        
        # BasicVideoProcessor の extract_frames メソッドを修正:

        def extract_frames(video_path, frame_dir, max_frames=1000):
            """
            フレーム抽出（タイムスタンプ付きディレクトリ対応・既存処理を踏襲）
            """
            import cv2
            from pathlib import Path

            logger = logging.getLogger(__name__)
            logger.info(f"📸 フレーム抽出開始: {video_path}")
            frame_dir = Path(frame_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)

            # 動画ファイルの存在確認
            if not Path(video_path).exists():
                logger.error(f"❌ 動画ファイルが存在しません: {video_path}")
                return {"success": False, "error": f"動画ファイルが存在しません: {video_path}"}

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"❌ 動画ファイルを開けません: {video_path}")
                return {"success": False, "error": f"動画ファイルを開けません: {video_path}"}

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            logger.info(f"📹 動画情報: {width}x{height}, {frame_count}フレーム, {fps:.1f}FPS, {duration:.1f}秒")

            # 抽出間隔計算
            interval = max(1, frame_count // max_frames)
            logger.info(f"🔢 抽出間隔: {interval} (最大{max_frames}フレーム)")

            extracted = 0
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % interval == 0:
                    frame_path = frame_dir / f"frame_{frame_number:06d}.jpg"
                    success = cv2.imwrite(str(frame_path), frame)
                    if success:
                        extracted += 1
                        if extracted >= max_frames:
                            break
                    else:
                        logger.warning(f"⚠️ フレーム保存失敗: {frame_path}")

                frame_number += 1

            cap.release()

            saved_frames = len(list(frame_dir.glob("frame_*.jpg")))
            logger.info(f"📊 抽出: {extracted}個, 実際に保存: {saved_frames}個")

            final_extracted = max(extracted, saved_frames)

            logger.info(f"✅ フレーム抽出完了: {final_extracted}フレーム")

            if final_extracted == 0:
                logger.error("❌ フレーム抽出に失敗しました")
                return {"success": False, "error": "フレーム抽出に失敗しました"}

            return {
                "success": True,
                "extracted_frames": final_extracted,
                "video_info": {
                    "total_frames": frame_count,
                    "fps": fps,
                    "duration": duration,
                    "resolution": [width, height],
                    "extraction_interval": interval
                }
            }
        
        def run_detection_tracking(self, frame_dir, video_name, output_dir=None):
            """基本検出・追跡処理（タイムスタンプ付きディレクトリ対応・機能維持版）"""
            try:
                self.logger.info(f"👁️ 基本検出・追跡処理開始: {video_name}")
                frame_files = sorted(list(Path(frame_dir).glob("*.jpg")))

                if not frame_files:
                    raise VideoProcessingError(f"フレームファイルが見つかりません: {frame_dir}")

                self.logger.info(f"📸 処理対象フレーム: {len(frame_files)}個")

                # モデルの事前ロード確認
                if not hasattr(self, 'detection_model') and not hasattr(self, 'pose_model'):
                    self.load_models()

                detection_count = 0
                frame_stats = []

                # 信頼度しきい値（より低く設定して検出率向上）
                conf_threshold = 0.25

                # 🔧 簡略化された処理ループ
                for i, frame_file in enumerate(frame_files):
                    try:
                        # フレーム読み込み
                        frame = cv2.imread(str(frame_file))
                        if frame is None:
                            self.logger.warning(f"⚠️ フレーム読み込み失敗: {frame_file}")
                            continue

                        frame = undistort_with_json(frame, calib_path="configs/camera_params.json")

                        frame_detections = 0

                        # 🔧 ポーズモデル優先（より確実）
                        if hasattr(self, 'pose_model') and self.pose_model:
                            try:
                                results = self.pose_model(frame, verbose=False, conf=conf_threshold)
                                if results and len(results[0].boxes) > 0:
                                    frame_detections = len(results[0].boxes)
                                    detection_count += frame_detections
                            except Exception as e:
                                self.logger.debug(f"フレーム{i}ポーズ検出エラー: {e}")

                        # フレーム統計記録（簡略化）
                        frame_stats.append({
                            "frame_id": i,
                            "frame_file": frame_file.name,
                            "detections": frame_detections,
                            "conf": conf_threshold,  # 固定値
                            "track_id": i,  # 簡易ID
                            "timestamp": datetime.now().isoformat()
                        })

                    except Exception as e:
                        self.logger.warning(f"フレーム{i}処理エラー（続行）: {e}")
                        continue

                # --- タイムスタンプ付きディレクトリ対応 ---
                if output_dir is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path("outputs/temp") / f"{video_name}_{timestamp}"
                else:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"{video_name}_results.csv"

                if frame_stats:
                    df = pd.DataFrame(frame_stats)
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"📊 結果CSV保存: {csv_path}")
                else:
                    # 🔧 空の場合でもCSVを作成
                    empty_df = pd.DataFrame(columns=["frame_id", "frame_file", "detections", "conf", "track_id", "timestamp"])
                    empty_df.to_csv(csv_path, index=False)
                    self.logger.warning("⚠️ 検出結果なし - 空のCSVを作成")

                # 統計情報
                self.processing_stats = {
                    "detection_tracking": {
                        "total_frames": len(frame_files),
                        "processed_frames": len(frame_stats),
                        "total_detections": detection_count,
                        "success_rate": len(frame_stats) / len(frame_files) if frame_files else 0
                    }
                }

                self.logger.info(f"✅ 基本検出・追跡完了: {detection_count}個検出 / {len(frame_stats)}フレーム処理")

                return {
                    "success": True,
                    "data": {
                        "csv_path": str(csv_path),
                        "detection_count": detection_count,
                        "frame_count": len(frame_files),
                        "processed_frames": len(frame_stats),
                        "processing_stats": self.processing_stats["detection_tracking"]
                    }
                }

            except Exception as e:
                self.logger.error(f"❌ 基本検出・追跡エラー: {e}")
                return {"success": False, "error": str(e)}
        
        def run_detection_tracking_with_depth(self, frame_dir, video_name):
            """深度統合検出・追跡処理（フォールバック版）"""
            self.logger.warning("🔧 深度統合処理は利用できません。通常処理にフォールバックします")
            result = self.run_detection_tracking(frame_dir, video_name)
            
            if result.get("success", False):
                result["data"]["depth_enabled"] = False
                result["data"]["depth_fallback"] = True
                result["data"]["enhanced_csv_path"] = result["data"]["csv_path"]
                self.logger.info("✅ フォールバック処理完了（深度機能無効）")
            
            return result

# 🔧 条件付きインポート - アナライザー（完全統合版）
METRICS_ANALYZER_AVAILABLE = False
MetricsAnalyzer = None

try:
    from analyzers.metrics_analyzer import MetricsAnalyzer
    METRICS_ANALYZER_AVAILABLE = True
    print("✅ MetricsAnalyzer が利用可能です")
except ImportError as e:
    print(f"⚠️ MetricsAnalyzer が見つかりません: {e}")
    print("🔧 基本分析機能を使用します")
    METRICS_ANALYZER_AVAILABLE = False

# 🔧 BasicMetricsAnalyzer（フォールバック用・完全版）
if not METRICS_ANALYZER_AVAILABLE:
    class BasicMetricsAnalyzer:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)
        
        def analyze_improvements(self, comparison_results):
            """基本改善分析（完全版）"""
            self.logger.info("📊 基本改善分析開始")
            
            try:
                baseline = comparison_results.get("baseline", {})
                experiment = comparison_results.get("experiment", {})
                
                analysis = {
                    "analyzer_type": "BasicMetricsAnalyzer",
                    "comparison_available": bool(baseline and experiment),
                    "timestamp": datetime.now().isoformat()
                }
                
                if baseline and experiment:
                    # 処理時間比較
                    baseline_time = baseline.get("processing_time", 0)
                    experiment_time = experiment.get("processing_time", 0)
                    
                    if baseline_time > 0:
                        time_improvement = ((baseline_time - experiment_time) / baseline_time) * 100
                        analysis["time_improvement_percent"] = time_improvement
                        analysis["time_comparison"] = {
                            "baseline_time": baseline_time,
                            "experiment_time": experiment_time,
                            "improvement": time_improvement
                        }
                    
                    # 検出数比較
                    baseline_detections = baseline.get("detection_count", 0)
                    experiment_detections = experiment.get("detection_count", 0)
                    
                    analysis["detection_comparison"] = {
                        "baseline": baseline_detections,
                        "experiment": experiment_detections,
                        "difference": experiment_detections - baseline_detections,
                        "improvement_rate": ((experiment_detections - baseline_detections) / baseline_detections * 100) if baseline_detections > 0 else 0
                    }
                    
                    # 品質比較
                    analysis["quality_comparison"] = {
                        "baseline_success": baseline.get("success", False),
                        "experiment_success": experiment.get("success", False)
                    }
                
                self.logger.info("✅ 基本改善分析完了")
                return analysis
                
            except Exception as e:
                self.logger.error(f"❌ 改善分析エラー: {e}")
                return {"basic_analysis": f"エラー: {e}", "error": True}
        
        def create_visualizations(self, detection_results, vis_dir):
            """
            可視化生成（タイムスタンプ付きディレクトリ対応・既存機能完全維持版）
            detection_results: run_detection_tracking等の出力(dict)
            vis_dir: 可視化画像の保存先ディレクトリ（タイムスタンプ付き）
            """
            import json
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from pathlib import Path
            from datetime import datetime

            self.logger.info(f"📈 基本可視化生成: {vis_dir}")

            # 初期化
            result = {
                "success": False,
                "error": "初期化エラー",
                "basic_stats_file": None,
                "graphs_generated": 0,
                "total_files": 0
            }

            try:
                vis_path = Path(str(vis_dir))
                vis_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 可視化ディレクトリ作成: {vis_path}")

                # detection_results の詳細ログ
                self.logger.info(f"🔧 detection_results type: {type(detection_results)}")
                self.logger.info(f"🔧 detection_results content: {detection_results}")

                # CSVパス抽出
                csv_path = None
                data = {}

                if isinstance(detection_results, dict):
                    if detection_results.get("success", False):
                        data = detection_results.get("data", {})
                        csv_path = data.get("csv_path")
                        # ネスト構造対応
                        if not csv_path and "detection_result" in data:
                            nested_data = data["detection_result"].get("data", {})
                            csv_path = nested_data.get("csv_path")
                self.logger.info(f"🔧 検出されたCSVパス: {csv_path}")

                # 基本統計ファイル作成（必ず作成）
                stats_file = vis_path / "basic_stats.json"
                basic_stats = {
                    "visualization_type": "BasicVisualization",
                    "detection_count": data.get("detection_count", 0),
                    "frame_count": data.get("frame_count", 0),
                    "processing_time": data.get("processing_time", 0),
                    "processing_stats": data.get("processing_stats", {}),
                    "timestamp": datetime.now().isoformat(),
                    "csv_path": str(csv_path) if csv_path else None,
                    "success": detection_results.get("success", False) if isinstance(detection_results, dict) else False
                }

                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(basic_stats, f, indent=2, ensure_ascii=False)
                self.logger.info(f"✅ 基本統計保存: {stats_file}")

                # 戻り値更新
                result.update({
                    "success": True,
                    "error": None,
                    "basic_stats_file": str(stats_file),
                    "total_files": 1,
                    "graphs_generated": 0
                })

                graphs_generated = 0
                graph_files = []

                try:
                    # フォント設定
                    try:
                        plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
                    except Exception:
                        plt.rcParams['font.family'] = 'DejaVu Sans'

                    # CSV ファイルの処理
                    if csv_path and Path(csv_path).exists():
                        self.logger.info(f"📊 CSVファイル読み込み: {csv_path}")
                        df = pd.read_csv(csv_path)
                        self.logger.info(f"📊 データ読み込み: {len(df)}行, カラム: {list(df.columns)}")

                        if not df.empty:
                            # 1. フレーム別検出数グラフ
                            if 'frame' in df.columns or 'frame_id' in df.columns:
                                try:
                                    frame_col = 'frame' if 'frame' in df.columns else 'frame_id'
                                    plt.figure(figsize=(12, 6))
                                    frame_counts = df[frame_col].value_counts().sort_index()
                                    plt.plot(frame_counts.index, frame_counts.values, 
                                    marker='o', linewidth=2, markersize=4, color='blue')
                                    plt.title('Detection Count by Frame', fontsize=16, pad=20)
                                    plt.xlabel('Frame Number', fontsize=12)
                                    plt.ylabel('Detection Count', fontsize=12)
                                    plt.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    timeline_path = vis_path / "detection_timeline.png"
                                    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
                                    plt.close()
                                    graphs_generated += 1
                                    graph_files.append(str(timeline_path))
                                    self.logger.info(f"✅ 時系列グラフ生成: {timeline_path}")
                                except Exception as e:
                                    self.logger.error(f"❌ 時系列グラフエラー: {e}")

                            # 2. 信頼度分布グラフ
                            if 'conf' in df.columns or 'confidence' in df.columns:
                                try:
                                    conf_col = 'conf' if 'conf' in df.columns else 'confidence'
                                    plt.figure(figsize=(10, 6))
                                    conf_data = df[conf_col].dropna()
                                    plt.hist(conf_data, bins=30, alpha=0.7, color='green', edgecolor='black')
                                    plt.axvline(conf_data.mean(), color='red', linestyle='--', 
                                                label=f'Average: {conf_data.mean():.3f}')
                                    plt.title('Confidence Distribution', fontsize=16, pad=20)
                                    plt.xlabel('Confidence', fontsize=12)
                                    plt.ylabel('Frequency', fontsize=12)
                                    plt.legend()
                                    plt.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    conf_path = vis_path / "confidence_distribution.png"
                                    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
                                    plt.close()
                                    graphs_generated += 1
                                    graph_files.append(str(conf_path))
                                    self.logger.info(f"✅ 信頼度分布グラフ生成: {conf_path}")
                                except Exception as e:
                                    self.logger.error(f"❌ 信頼度分布グラフエラー: {e}")

                            # 3. クラス分布グラフ
                            if 'class_name' in df.columns:
                                try:
                                    plt.figure(figsize=(12, 8))
                                    class_counts = df['class_name'].value_counts()
                                    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
                                    plt.title('Class Distribution', fontsize=16, pad=20)
                                    plt.xlabel('Class Name', fontsize=12)
                                    plt.ylabel('Detection Count', fontsize=12)
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    class_path = vis_path / "class_distribution.png"
                                    plt.savefig(class_path, dpi=300, bbox_inches='tight')
                                    plt.close()
                                    graphs_generated += 1
                                    graph_files.append(str(class_path))
                                    self.logger.info(f"✅ クラス分布グラフ生成: {class_path}")
                                except Exception as e:
                                    self.logger.error(f"❌ クラス分布グラフエラー: {e}")

                            # 4. 4点キーポイント可視化（もし4点CSVがあれば）
                            if 'filtered_csv_path' in data and data['filtered_csv_path'] and Path(data['filtered_csv_path']).exists():
                                try:
                                    filtered_csv = data['filtered_csv_path']
                                    self.logger.info(f"🎨 6点キーポイント可視化: {filtered_csv}")
                                    vis_4pt_result = self.create_6point_visualization(filtered_csv, data.get('video_path', ''), vis_path)
                                    if vis_4pt_result.get("success"):
                                        graphs_generated += 1
                                        graph_files.append(vis_4pt_result.get("output_dir"))
                                        self.logger.info(f"✅ 4点キーポイント可視化生成: {vis_4pt_result.get('output_dir')}")
                                except Exception as e:
                                    self.logger.error(f"❌ 4点キーポイント可視化エラー: {e}")

                        else:
                            self.logger.warning("⚠️ CSVデータが空です")
                    else:
                        self.logger.warning(f"⚠️ CSVファイルが見つからない: {csv_path}")
                
                except ImportError as e:
                    self.logger.warning(f"⚠️ matplotlib/pandasインポートエラー: {e}")
                except Exception as plot_error:
                    self.logger.error(f"❌ グラフ生成エラー: {plot_error}", exc_info=True)

                # 最終結果更新
                total_files = 1 + graphs_generated
                result.update({
                    "success": True,
                    "error": None,
                    "graphs_generated": graphs_generated,
                    "total_files": total_files,
                    "graph_files": graph_files
                })

                self.logger.info(f"🎨 可視化生成完了: 基本統計1個 + グラフ{graphs_generated}個 = 合計{total_files}個")

                return result

            except Exception as e:
                self.logger.error(f"❌ 可視化生成全体エラー: {e}", exc_info=True)
                result.update({
                    "success": False,
                    "error": str(e),
                    "graphs_generated": 0,
                    "total_files": 0
                })
                return result
                
        def _create_detection_charts(self, data, vis_path):
            """検出結果のチャート生成"""
            try:
                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns
                
                # 統計データの確認
                processing_stats = data.get("processing_stats", {})
                detection_count = data.get("detection_count", 0)
                
                # 1. 基本統計グラフ
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # 検出数グラフ
                ax1.bar(['検出数'], [detection_count], color='skyblue')
                ax1.set_title('総検出数')
                ax1.set_ylabel('件数')
                
                # 処理統計
                if processing_stats:
                    stats_keys = list(processing_stats.keys())[:5]  # 最大5項目
                    stats_values = [processing_stats[k] for k in stats_keys]
                    ax2.barh(stats_keys, stats_values, color='lightcoral')
                    ax2.set_title('処理統計')
                    ax2.set_xlabel('値')
                
                # フレーム数
                frame_count = data.get("frame_count", 0)
                ax3.pie([frame_count, max(1, 120 - frame_count)], 
                        labels=['処理済み', '未処理'], autopct='%1.1f%%',
                        colors=['lightgreen', 'lightgray'])
                ax3.set_title('フレーム処理状況')
                
                # 処理時間
                processing_time = data.get("processing_time", 0)
                ax4.bar(['処理時間'], [processing_time], color='gold')
                ax4.set_title('処理時間 (秒)')
                ax4.set_ylabel('秒')
                
                plt.tight_layout()
                plt.savefig(vis_path / "detection_summary.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"✅ サマリーチャート生成: detection_summary.png")
                
            except ImportError:
                self.logger.warning("⚠️ matplotlib未インストール - 基本統計のみ保存")
            except Exception as e:
                self.logger.error(f"❌ チャート生成エラー: {e}")

# 🔧 BasicConfig（フォールバック用・完全版）
if not CONFIG_AVAILABLE:
    class BasicConfig:
        def __init__(self, config_path=None):
            self.config_path = config_path
            self.data = self.load_config()
            self.logger = logging.getLogger(__name__)
        
        def load_config(self):
            """設定ロード（完全版）"""
            logger = logging.getLogger(__name__)
            logger.info(f"⚙️ 設定ファイル読み込み: {self.config_path}")
            
            if self.config_path and Path(self.config_path).exists():
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        if self.config_path.endswith(('.yaml', '.yml')):
                            config_data = yaml.safe_load(f)
                            logger.info("✅ YAML設定ファイル読み込み成功")
                        else:
                            config_data = json.load(f)
                            logger.info("✅ JSON設定ファイル読み込み成功")
                    
                    return config_data
                    
                except Exception as e:
                    logger.warning(f"⚠️ 設定読み込みエラー: {e}")
                    logger.info("🔧 デフォルト設定を使用します")
            else:
                logger.warning(f"⚠️ 設定ファイルが見つかりません: {self.config_path}")
                logger.info("🔧 デフォルト設定を使用します")
            
            # デフォルト設定（完全版）
            default_config = {
                "models": {
                    "detection": "models/yolo/yolo11x.pt",
                    "pose": "models/yolo/yolo11x-pose.pt"
                },
                "processing": {
                    "detection": {
                        "confidence_threshold": 0.3, 
                        "iou_threshold": 0.45,
                        "max_detections": 1000
                    },
                    "depth_estimation": {
                        "enabled": False,
                        "model": "midas_v21_small_256",
                        "model_path": "models/depth/midas_v21_small_256.pt"
                    },
                    "tile_inference": {
                        "enabled": False,
                        "tile_size": [640, 640],
                        "overlap": 0.1
                    }
                },
                "video_dir": "videos",
                "output_dir": "outputs",
                "logging": {
                    "level": "INFO",
                    "file": "logs/analysis.log"
                },
                "experiments": {
                    "comparison": {"type": "comparison", "description": "基本比較実験"},
                    "model_test": {"type": "model_test", "description": "モデル性能テスト"},
                    "tile_inference": {"type": "tile_inference", "description": "タイル推論テスト"}
                }
            }
            
            logger.info("✅ デフォルト設定を適用しました")
            return default_config
        
        def get(self, key, default=None):
            """設定値取得（ドット記法対応・完全版）"""
            keys = key.split('.')
            value = self.data
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        
        def get_experiment_config(self, experiment_type):
            """実験設定取得（完全版）"""
            experiment_configs = self.get("experiments", {})
            if experiment_type in experiment_configs:
                return experiment_configs[experiment_type]
            
            # デフォルト実験設定
            return {
                "type": experiment_type,
                "basic_mode": True,
                "description": f"基本実験: {experiment_type}",
                "enabled": True
            }
        
        @property
        def video_dir(self):
            return self.get("video_dir", "videos")

LOGGER_AVAILABLE = False
setup_logger = None

try:
    from utils.logger import setup_logger
    LOGGER_AVAILABLE = True
    print("✅ setup_logger が利用可能です")
except ImportError as e:
    print(f"⚠️ setup_logger が見つかりません: {e}")
    print("🔧 基本ログ機能を使用します")
    LOGGER_AVAILABLE = False

# 🔧 基本ログ設定（フォールバック用・完全版）
if not LOGGER_AVAILABLE:
    def setup_logger():
        """基本ログ設定（完全版）"""
        # ログディレクトリ作成
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # ログファイル名
        log_file = log_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 既存のハンドラーをクリア
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"🔧 基本ログ機能を初期化しました: {log_file}")
        return logger

# ✅ ここでクラス定義開始（if文の外、インデント0）
class ImprovedYOLOAnalyzer:
    """
    YOLO11 広角カメラ分析システム（完全統合版）
    - 深度推定統合対応
    - モジュール不足完全対応
    - Stage 5/6修正完了
    - フォールバック機能完全統合
    """

    # Line 834-854を修正:

    # Line 1272-1310の__init__メソッドを以下で置き換え:

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        初期化（完全統合版）

        Args:
            config_path: 設定ファイルパス
        """
        # 🔧 ロガーを最初に初期化
        self.logger = setup_logger()

        # エラーコンテキスト設定
        if ERROR_HANDLER_AVAILABLE:
            context_manager = ErrorContext("ImprovedYOLOAnalyzer初期化", logger=self.logger)
        else:
            context_manager = self._basic_context("ImprovedYOLOAnalyzer初期化")

        with context_manager as ctx:
            # 設定初期化
            self.config = self._initialize_config(config_path)

            # 深度推定有効性の確認
            self.depth_enabled = self.config.get('processing.depth_estimation.enabled', False)
            self.logger.info(f"🔍 深度推定: {'有効' if self.depth_enabled else '無効'}")

            # 🔧 厳密モデル使用フラグ追加
            self.force_exact_model = True
            self.model_verification_results = {}

            # 評価器の選択と初期化
            self._initialize_evaluator(ctx)

            # プロセッサーとアナライザー初期化
            self._initialize_processor_analyzer(ctx)

            # 🔧 analyzer の明示的初期化を追加
            self._initialize_analyzer(ctx)

            # エラー収集用
            self.error_collector = []

            # ディレクトリセットアップ
            self._setup_directories()

            # 初期化完了報告
            self._report_initialization(ctx)

    def _basic_context(self, name):
        """基本コンテキストマネージャー"""
        class BasicContext:
            def __init__(self, name):
                self.name = name
                self.logger = logging.getLogger(__name__)
            def __enter__(self):
                self.logger.debug(f"🔍 処理開始: {self.name}")
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type:
                    self.logger.error(f"❌ エラー in {self.name}: {exc_val}")
                else:
                    self.logger.debug(f"✅ 処理完了: {self.name}")
                return False
            def add_info(self, key, value):
                self.logger.debug(f"📝 {self.name} - {key}: {value}")
        return BasicContext(name)

    # Line 857の_initialize_configメソッドを修正:

    def _initialize_config(self, config_path: str):
        """設定初期化（完全版）"""
        depth_config_path = "configs/depth_config.yaml"
    
        # ⚠️ ここで self.logger を使う前に、ロガーを初期化する必要がある
        logger = logging.getLogger(__name__)  # 🔧 一時的なロガー使用
        logger.info(f"⚙️ 設定初期化開始: {config_path}")
    
        # 設定ファイルの優先順位決定
        if Path(config_path).exists():
            primary_config = config_path
            logger.info(f"📄 指定設定ファイル使用: {config_path}")
        elif Path(depth_config_path).exists():
            primary_config = depth_config_path
            logger.info(f"🔍 深度設定ファイル自動検出: {depth_config_path}")
        else:
            primary_config = config_path
            logger.warning(f"⚠️ 設定ファイルが見つかりません: {config_path}")

        # 設定オブジェクト初期化
        if CONFIG_AVAILABLE and Config:
            return Config(primary_config)
        else:
            return BasicConfig(primary_config)

    def _initialize_evaluator(self, ctx):
        """評価器初期化（完全版）"""
        if self.depth_enabled and DEPTH_EVALUATOR_AVAILABLE and DepthEnhancedEvaluator:
            try:
                self.evaluator = DepthEnhancedEvaluator(self.config)
                self.logger.info("🔍 深度統合評価器を初期化")
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("evaluator_type", "DepthEnhancedEvaluator")
            except Exception as e:
                self.logger.warning(f"DepthEnhancedEvaluator 初期化失敗: {e}")
                self._fallback_to_basic_evaluator(ctx)
        elif COMPREHENSIVE_EVALUATOR_AVAILABLE and ComprehensiveEvaluator:
            try:
                self.evaluator = ComprehensiveEvaluator(self.config)
                self.logger.info("📊 標準評価器を初期化")
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("evaluator_type", "ComprehensiveEvaluator")
            except Exception as e:
                self.logger.warning(f"ComprehensiveEvaluator 初期化失敗: {e}")
                self._fallback_to_basic_evaluator(ctx)
        else:
            self._fallback_to_basic_evaluator(ctx)

    def _fallback_to_basic_evaluator(self, ctx):
        """基本評価器へのフォールバック"""
        self.evaluator = BasicEvaluator(self.config)
        self.logger.info("🔧 基本評価器を初期化")
        if hasattr(ctx, 'add_info'):
            ctx.add_info("evaluator_type", "BasicEvaluator")

    def _initialize_processor_analyzer(self, ctx):
        """プロセッサー・アナライザー初期化（完全版）"""
        try:
            # Video Processor 初期化
            if VIDEO_PROCESSOR_AVAILABLE:
                self.logger.info("🎥 高度動画プロセッサーを初期化")
                self.processor = VideoProcessor(self.config)
            else:
                self.logger.info("🔄 BasicVideoProcessor を初期化")
                self.processor = BasicVideoProcessor(self.config)

            # 🔧 analyzer の初期化は別メソッドで行う（重複を避けるため）
            # self._initialize_analyzer(ctx) は __init__ で呼び出し済み
        
            ctx.add_info("processor_type", type(self.processor).__name__)
        
        except Exception as e:
            self.logger.error(f"❌ プロセッサー・アナライザー初期化エラー: {e}", exc_info=True)
            self._fallback_processor_analyzer(ctx)

    def _fallback_processor_analyzer(self, ctx):
        """プロセッサー・アナライザー用フォールバック"""
        try:
            self.logger.warning("🔄 基本プロセッサー・アナライザーにフォールバック")
            self.processor = BasicVideoProcessor(self.config)
        
            # analyzer が未初期化の場合は初期化
            if not hasattr(self, 'analyzer') or self.analyzer is None:
                self._create_fallback_analyzer(ctx)
            
            ctx.add_info("fallback_applied", True)
        
        except Exception as e:
            self.logger.error(f"❌ フォールバック失敗: {e}", exc_info=True)
            raise

    def _initialize_analyzer(self, ctx):
        """メトリクス分析器初期化（完全版）"""
        try:
            self.logger.info("📊 高度メトリクス分析器を初期化")
        
            # 🔧 METRICS_ANALYZER_AVAILABLE の確認
            if METRICS_ANALYZER_AVAILABLE:
                try:
                    # 高度分析器の初期化を試行
                    self.analyzer = MetricsAnalyzer(self.config)
                    self.logger.info("✅ MetricsAnalyzer初期化成功")
                    ctx.add_info("analyzer_type", "MetricsAnalyzer")
                    return
                except Exception as e:
                    self.logger.warning(f"MetricsAnalyzer初期化失敗: {e}")
        
            # 🔧 フォールバック: BasicMetricsAnalyzer
            self.logger.info("🔄 BasicMetricsAnalyzerにフォールバック")
            self.analyzer = BasicMetricsAnalyzer(self.config)
            self.logger.info("✅ BasicMetricsAnalyzer初期化成功")
            ctx.add_info("analyzer_type", "BasicMetricsAnalyzer")
        
            # 🔧 create_visualizations メソッドの存在確認
            if hasattr(self.analyzer, 'create_visualizations'):
                self.logger.info("✅ create_visualizations メソッド確認")
            else:
                self.logger.error("❌ create_visualizations メソッドが存在しません")
                # 🔧 メソッドを動的に追加
                self._add_fallback_visualization_method()
            
        except Exception as e:
            self.logger.error(f"❌ 分析器初期化エラー: {e}", exc_info=True)
            # 🔧 最終フォールバック
            self._create_fallback_analyzer(ctx)

    def _add_fallback_visualization_method(self):
        """フォールバック可視化メソッドの動的追加"""
        def fallback_create_visualizations(detection_results, vis_dir):
            """フォールバック可視化生成"""
            try:
                self.logger.info(f"🔧 フォールバック可視化生成: {vis_dir}")
            
                from pathlib import Path
                import json
                from datetime import datetime
            
                # ディレクトリ作成
                vis_path = Path(str(vis_dir))
                vis_path.mkdir(parents=True, exist_ok=True)
            
                # 基本統計ファイル作成
                stats_file = vis_path / "basic_stats.json"
                basic_stats = {
                    "visualization_type": "FallbackVisualization",
                    "timestamp": datetime.now().isoformat(),
                    "detection_results_type": str(type(detection_results)),
                    "success": True
                }
            
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(basic_stats, f, indent=2, ensure_ascii=False)
            
                self.logger.info(f"✅ フォールバック可視化完了: {stats_file}")
            
                return {
                    "success": True,
                    "basic_stats_file": str(stats_file),
                    "total_files": 1,
                    "graphs_generated": 0,
                    "fallback": True
                }
            
            except Exception as e:
                self.logger.error(f"❌ フォールバック可視化エラー: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "total_files": 0,
                    "graphs_generated": 0,
                    "fallback": True
                }
    
        # 🔧 メソッドを動的にバインド
        import types
        self.analyzer.create_visualizations = types.MethodType(fallback_create_visualizations, self.analyzer)
        self.logger.info("🔧 フォールバック可視化メソッド追加完了")

    def _create_fallback_analyzer(self, ctx):
        """最終フォールバック分析器の作成"""
        class FallbackAnalyzer:
            def __init__(self, config):
                self.config = config
                self.logger = logging.getLogger(__name__)
        
            def create_visualizations(self, detection_results, vis_dir):
                """最終フォールバック可視化"""
                try:
                    from pathlib import Path
                    import json
                    from datetime import datetime
                
                    vis_path = Path(str(vis_dir))
                    vis_path.mkdir(parents=True, exist_ok=True)
                
                    fallback_file = vis_path / "fallback_analysis.json"
                    fallback_data = {
                        "analyzer_type": "FallbackAnalyzer",
                        "timestamp": datetime.now().isoformat(),
                        "note": "分析器初期化失敗のため最終フォールバックを使用",
                        "detection_results_available": detection_results is not None
                    }
                
                    with open(fallback_file, 'w', encoding='utf-8') as f:
                        json.dump(fallback_data, f, indent=2, ensure_ascii=False)
                
                    return {
                        "success": True,
                        "total_files": 1,
                        "graphs_generated": 0,
                        "fallback": True,
                        "analyzer_type": "FallbackAnalyzer"
                    }
                
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "total_files": 0,
                        "graphs_generated": 0,
                        "fallback": True
                    }
    
        self.analyzer = FallbackAnalyzer(self.config)
        self.logger.warning("⚠️ 最終フォールバック分析器を作成")
        ctx.add_info("analyzer_type", "FallbackAnalyzer")

    def _setup_directories(self):
        """ディレクトリセットアップ（完全版）"""
        directories = [
            "outputs/baseline",
            "outputs/experiments",
            "outputs/visualizations",
            "outputs/temp",
            "logs",
            "models/yolo",
            "models/depth"
        ]
        
        self.logger.info("📁 ディレクトリセットアップ開始")
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"📁 作成/確認: {directory}")
        
        self.logger.info("✅ ディレクトリセットアップ完了")

    def _report_initialization(self, ctx):
        """初期化完了報告（完全版）"""
        features = []
        if self.depth_enabled:
            features.append("深度推定")
        if self.config.get('processing.tile_inference.enabled', False):
            features.append("タイル推論")
        
        # 使用中のフォールバック機能の表示
        fallbacks = []
        if not COMPREHENSIVE_EVALUATOR_AVAILABLE:
            fallbacks.append("基本評価器")
        if not VIDEO_PROCESSOR_AVAILABLE:
            fallbacks.append("基本動画処理")
        if not METRICS_ANALYZER_AVAILABLE:
            fallbacks.append("基本分析")
        if not CONFIG_AVAILABLE:
            fallbacks.append("基本設定")
        if not LOGGER_AVAILABLE:
            fallbacks.append("基本ログ")

        if features:
            self.logger.info(f"🚀 ImprovedYOLOAnalyzer初期化完了 (機能: {', '.join(features)})")
        else:
            self.logger.info("📋 ImprovedYOLOAnalyzer初期化完了 (標準モード)")
            
        if fallbacks:
            self.logger.info(f"🔧 フォールバック機能使用中: {', '.join(fallbacks)}")

        if hasattr(ctx, 'add_info'):
            ctx.add_info("depth_enabled", self.depth_enabled)
            ctx.add_info("fallback_count", len(fallbacks))

    @handle_errors(error_category=ErrorCategory.VIDEO_PROCESSING)
    def run_baseline_analysis(self, video_path: str, output_dir=None) -> Dict[str, Any]:
        """
        ベースライン分析実行（完全統合版）

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

            self.logger.info(f"🎯 ベースライン分析開始: {video_name}")

            if hasattr(ctx, 'add_info'):
                ctx.add_info("video_path", str(video_path))
                ctx.add_info("video_name", video_name)
                ctx.add_info("depth_enabled", self.depth_enabled)

            # 出力ディレクトリ準備
            if output_dir is None:
                output_dir = Path("outputs/baseline") / video_name
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_dir = output_dir / "frames"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            frame_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"📁 出力ディレクトリ: {output_dir}")

            try:
                # Step 1: フレーム抽出
                self.logger.info("📸 Step 1: フレーム抽出開始")
                
                # フレーム抽出実行
                frame_result = self.processor.extract_frames(video_path, frame_dir)
                
                # 🔧 基本的な成功/失敗チェック
                if not frame_result.get("success", False):
                    error_msg = f"フレーム抽出失敗: {frame_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise VideoProcessingError(error_msg)

                # 🔧 フレーム数の多重確認システム
                
                # 方法1: APIから返却された値
                api_extracted_frames = frame_result.get("extracted_frames", 0)
                self.logger.debug(f"📊 API返却フレーム数: {api_extracted_frames}")
                
                # 方法2: フレームディレクトリの直接確認
                frame_files_jpg = list(frame_dir.glob("frame_*.jpg"))
                frame_files_jpeg = list(frame_dir.glob("frame_*.jpeg"))
                frame_files_png = list(frame_dir.glob("frame_*.png"))
                
                # すべての画像ファイルを統合
                all_frame_files = frame_files_jpg + frame_files_jpeg + frame_files_png
                actual_frame_count = len(all_frame_files)
                
                self.logger.debug(f"📊 ディレクトリ内ファイル数:")
                self.logger.debug(f"  - JPGファイル: {len(frame_files_jpg)}個")
                self.logger.debug(f"  - JPEGファイル: {len(frame_files_jpeg)}個") 
                self.logger.debug(f"  - PNGファイル: {len(frame_files_png)}個")
                self.logger.debug(f"  - 合計: {actual_frame_count}個")
                
                # 方法3: processing_statsからの取得
                stats_frames = 0
                if hasattr(self.processor, 'processing_stats') and self.processor.processing_stats:
                    frame_extraction_stats = self.processor.processing_stats.get("frame_extraction", {})
                    stats_frames = frame_extraction_stats.get("extracted_frames", 0)
                
                self.logger.debug(f"📊 統計情報フレーム数: {stats_frames}")
                
                # 🔧 最も信頼できる値を採用
                frame_counts = [api_extracted_frames, actual_frame_count, stats_frames]
                valid_counts = [count for count in frame_counts if count > 0]
                
                if valid_counts:
                    # 有効な値がある場合は最大値を採用
                    final_frame_count = max(valid_counts)
                    self.logger.info(f"📊 フレーム数確定: {final_frame_count}個（候補: {frame_counts}）")
                else:
                    # すべて0の場合は詳細調査
                    self.logger.warning("⚠️ 全ての方法でフレーム数が0です。詳細調査を実行...")
                    
                    # 方法4: より広範囲なファイル確認
                    all_files = list(frame_dir.glob("*"))
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                    image_files = []
                    
                    for file_path in all_files:
                        if file_path.suffix.lower() in image_extensions:
                            image_files.append(file_path)
                    
                    final_frame_count = len(image_files)
                    
                    if final_frame_count > 0:
                        self.logger.info(f"🔍 広範囲確認で発見: {final_frame_count}個の画像ファイル")
                        # ファイル名の例を表示
                        sample_files = [f.name for f in image_files[:3]]
                        self.logger.debug(f"  サンプル: {sample_files}")
                    else:
                        # 方法5: 最後の手段 - ディレクトリ内容の完全チェック
                        self.logger.error("🔍 最終確認 - ディレクトリ内容:")
                        self.logger.error(f"  - パス: {frame_dir}")
                        self.logger.error(f"  - 存在確認: {frame_dir.exists()}")
                        self.logger.error(f"  - アクセス権限: {frame_dir.is_dir() if frame_dir.exists() else 'N/A'}")
                        
                        if frame_dir.exists():
                            all_content = list(frame_dir.glob("*"))
                            self.logger.error(f"  - 全ファイル({len(all_content)}個): {[f.name for f in all_content[:10]]}")
                            
                            # ファイルサイズ確認
                            for file_path in all_content[:5]:
                                if file_path.is_file():
                                    size_mb = file_path.stat().st_size / (1024 * 1024)
                                    self.logger.error(f"    {file_path.name}: {size_mb:.2f}MB")
                
                # 🔧 結果の詳細ログ出力
                self.logger.info(f"✅ Step 1完了: {final_frame_count}フレーム抽出")
                
                if final_frame_count > 0:
                    # 成功時の統計情報
                    if actual_frame_count > 0 and len(all_frame_files) > 0:
                        # ファイルサンプルの表示
                        sample_count = min(3, len(all_frame_files))
                        sample_files = [f.name for f in all_frame_files[:sample_count]]
                        self.logger.info(f"📁 保存ファイル例: {sample_files}")
                        
                        # ファイルサイズ統計
                        total_size = sum(f.stat().st_size for f in all_frame_files[:10])  # 最初の10ファイル
                        avg_size_kb = (total_size / min(10, len(all_frame_files))) / 1024 if all_frame_files else 0
                        self.logger.debug(f"📊 平均ファイルサイズ: {avg_size_kb:.1f}KB")
                    
                    # フレーム抽出率の計算
                    if hasattr(self.processor, 'processing_stats') and self.processor.processing_stats:
                        extraction_stats = self.processor.processing_stats.get("frame_extraction", {})
                        total_frames = extraction_stats.get("total_frames", 0)
                        if total_frames > 0:
                            extraction_rate = (final_frame_count / total_frames) * 100
                            self.logger.info(f"📊 抽出率: {extraction_rate:.1f}% ({final_frame_count}/{total_frames})")
                
                # 🔧 ゼロフレームの場合のエラー処理
                if final_frame_count == 0:
                    error_msg = "フレーム抽出数が0です。動画ファイルと処理を確認してください"
                    self.error_collector.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    
                    # 🔧 詳細なデバッグ情報を出力
                    self.logger.error(f"🔍 デバッグ情報詳細:")
                    self.logger.error(f"  動画ファイル:")
                    self.logger.error(f"    - パス: {video_path}")
                    self.logger.error(f"    - 存在: {Path(video_path).exists()}")
                    if Path(video_path).exists():
                        video_size = Path(video_path).stat().st_size / (1024 * 1024)
                        self.logger.error(f"    - サイズ: {video_size:.1f}MB")
                    
                    self.logger.error(f"  出力ディレクトリ:")
                    self.logger.error(f"    - パス: {frame_dir}")
                    self.logger.error(f"    - 存在: {frame_dir.exists()}")
                    self.logger.error(f"    - 権限: {oct(frame_dir.stat().st_mode)[-3:] if frame_dir.exists() else 'N/A'}")
                    
                    self.logger.error(f"  プロセッサ情報:")
                    self.logger.error(f"    - タイプ: {type(self.processor).__name__}")
                    self.logger.error(f"    - 設定: {getattr(self.processor, 'config', 'N/A')}")
                    
                    # frame_resultの詳細
                    self.logger.error(f"  API応答:")
                    self.logger.error(f"    - frame_result: {frame_result}")
                    
                    raise VideoProcessingError(error_msg)
                
                # ✅ 処理継続のためのフレーム数記録
                # 後続処理で使用するため、確定したフレーム数を保存
                if not hasattr(self, 'current_analysis_stats'):
                    self.current_analysis_stats = {}
                self.current_analysis_stats['extracted_frame_count'] = final_frame_count
                self.current_analysis_stats['frame_directory'] = str(frame_dir)
                
                self.logger.debug(f"📝 現在の解析統計を更新: {self.current_analysis_stats}")

                # Step 2: 検出・追跡処理（キーポイント確実取得版）
                self.logger.info("🎯 Step 2: YOLOポーズモデル確実使用処理開始")

                # 🔧 ポーズモデルパスの確実な取得
                models_config = self.config.get('models', {}) if hasattr(self.config, 'get') else {}
                pose_model_path = models_config.get('pose', 'models/yolo/yolo11x-pose.pt')

                # 🔧 修正: ポーズモデルの確実な確認
                if not Path(pose_model_path).exists():
                    self.logger.error(f"🚨 ポーズモデルが存在しません: {pose_model_path}")
    
                    # 代替ポーズモデルを探索
                    alternative_paths = [
                        "models/yolo/yolo11x-pose.pt",
                        "models/yolo11x-pose.pt", 
                        "yolo11x-pose.pt",
                        "models/yolo/yolo11l-pose.pt",
                        "models/yolo11l-pose.pt",
                        "models/yolo/yolo11m-pose.pt",
                        "models/yolo11m-pose.pt"
                    ]
    
                    found_model = None
                    for alt_path in alternative_paths:
                        if Path(alt_path).exists():
                            found_model = alt_path
                            self.logger.info(f"🔧 代替ポーズモデル発見: {alt_path}")
                            break
    
                    if found_model:
                        pose_model_path = found_model
                        self.logger.info(f"✅ ポーズモデルパス更新: {pose_model_path}")
                    else:
                        self.logger.error("🚨 利用可能なポーズモデルが見つかりません")
                        self.logger.error("🔧 以下のパスを確認してください:")
                        for path in [pose_model_path] + alternative_paths:
                            self.logger.error(f"  - {path}")
                        return ResponseBuilder.error(
                            message="ポーズモデルが見つかりません",
                            details={
                                "original_path": pose_model_path,
                                "searched_paths": alternative_paths,
                                "suggestion": "setup.py を実行してモデルをダウンロードしてください"
                            }
                        )

                # 🔧 修正: ポーズモデル確実使用の設定
                detection_config = {
                    "confidence_threshold": 0.3,
                    "tracking_config": "bytetrack.yaml",  # 🔧 確実に設定
                    "save_visualizations": True,
                    "save_detection_frames": True,
                    "force_pose_task": True,  # 🔧 ポーズタスク強制
                    "model_verification_required": True,  # 🔧 モデル検証必須
                    "keypoint_processing_enabled": True  # 🔧 キーポイント処理確実有効
                }

                self.logger.info(f"🎯 検出設定:")
                self.logger.info(f"  ポーズモデル: {pose_model_path}")
                self.logger.info(f"  tracker: {detection_config['tracking_config']}")
                self.logger.info(f"  ポーズタスク強制: {detection_config['force_pose_task']}")

                # 🚀 yolopose_analyzer での確実なキーポイント検出実行
                try:
                    if not YOLOPOSE_ANALYZER_AVAILABLE:
                        raise ImportError("yolopose_analyzer が利用できません")
    
                    # 🔧 キーポイント検出を確実にする設定
                    enhanced_config = {
                        "models": {
                            "pose": pose_model_path
                        },
                        "processing": {
                            "confidence_threshold": 0.3,
                            "save_keypoints": True,
                            "keypoint_format": "coco",
                            "force_keypoint_detection": True,
                            "model_policy": {
                                "verify_pose_model": True,
                                "require_keypoints": True,
                                "use_pose_model": True
                            }
                        },
                        "tracking": {
                            "tracker_type": "bytetrack",
                            "track_thresh": 0.6,
                            "track_buffer": 60,
                            "match_thresh": 0.8
                        },
                        "output": {
                            "save_visualizations": True,
                            "save_csv": True,
                            "csv_include_keypoints": True
                        },
                        "inference": {
                            "batch_size": 16,
                            "device": "auto",
                            "task": "pose"  # 🔧 configの中でタスク指定
                        }
                    }
    
                    # 既存設定とマージ
                    base_config = {}
                    if hasattr(self.config, '__dict__'):
                        base_config = self.config.__dict__
                    elif hasattr(self.config, 'data'):
                        base_config = self.config.data
    
                    for key, value in base_config.items():
                        if key not in enhanced_config:
                            enhanced_config[key] = value
    
                    self.logger.info("🚀 確実キーポイント検出を実行")
    
                    # yolopose_analyzer実行（キーポイント重視設定）
                    detection_result = analyze_frames_with_tracking_enhanced(
                        frame_dir=str(frame_dir),
                        result_dir=str(output_dir),
                        model_path=pose_model_path,
                        config=enhanced_config,
                        force_exact_model=True  # 🔧 確実なモデル使用フラグ
                    )
    
                    processing_type = "キーポイント統合"
    
                    # 🔍 キーポイント検出結果の詳細検証
                    if detection_result.get("success", False):
                        data = detection_result.get("data", {})
                        csv_path = data.get("csv_path")
        
                        if csv_path and Path(csv_path).exists():
                            # CSV内容の詳細確認
                            import pandas as pd
                            df = pd.read_csv(csv_path)
            
                            self.logger.info("🔍 ========== キーポイント検出結果検証 ==========")
                            self.logger.info(f"📊 検出データ形状: {df.shape}")
                            self.logger.info(f"📋 全列名: {df.columns.tolist()}")
            
                            # キーポイント列の確認
                            keypoint_cols = [col for col in df.columns if 'keypoint' in col.lower() or 'kpt' in col.lower()]
                            self.logger.info(f"🦴 キーポイント関連列: {len(keypoint_cols)}個")
            
                            if keypoint_cols:
                                self.logger.info(f"✅ キーポイント検出成功: {keypoint_cols[:10]}...")
                
                                # 4点キーポイント（COCO形式）の特別確認
                                target_keypoints = [3, 4, 5, 6]  # left_ear, right_ear, left_shoulder, right_shoulder
                                found_targets = []
                
                                for kpt_idx in target_keypoints:
                                    x_cols = [col for col in df.columns if f'keypoint_{kpt_idx}_x' in col or f'kpt_{kpt_idx}_x' in col]
                                    y_cols = [col for col in df.columns if f'keypoint_{kpt_idx}_y' in col or f'kpt_{kpt_idx}_y' in col]
                    
                                    if x_cols and y_cols:
                                        found_targets.append(f"COCO#{kpt_idx}")
                                        self.logger.info(f"✅ COCO#{kpt_idx}キーポイント発見: {x_cols[0]}, {y_cols[0]}")
                
                                if len(found_targets) >= 3:
                                    self.logger.info(f"🎯 4点キーポイント検出状況: {len(found_targets)}/4点発見")
                                    self.logger.info(f"  発見: {found_targets}")
                                else:
                                    self.logger.warning(f"⚠️ 4点キーポイント不完全: {len(found_targets)}/4点のみ")
                                    self.logger.warning(f"  発見済み: {found_targets}")
                            else:
                                self.logger.error("❌ キーポイント列が一切検出されていません！")
                                self.logger.error("🔧 原因: ポーズモデルが正しく動作していない可能性")
                
                                # サンプルデータの表示
                                if not df.empty:
                                    self.logger.error("📋 検出されたデータサンプル:")
                                    for col in df.columns[:10]:
                                        sample_value = df.iloc[0][col] if len(df) > 0 else "N/A"
                                        self.logger.error(f"  {col}: {sample_value}")
                        else:
                            self.logger.error(f"❌ 検出結果CSVが見つかりません: {csv_path}")
                    else:
                        error_msg = detection_result.get("error", "不明なエラー")
                        self.logger.error(f"❌ キーポイント検出失敗: {error_msg}")
                        self.error_collector.append(f"キーポイント検出失敗: {error_msg}")
                        raise VideoProcessingError(error_msg)

                except ImportError as e:
                    self.logger.error(f"❌ yolopose_analyzer インポートエラー: {e}")
                    self.logger.warning("🔄 BasicVideoProcessor にフォールバック（キーポイント機能制限）")
    
                    # フォールバック処理
                    if self.depth_enabled and hasattr(self.processor, 'run_detection_tracking_with_depth'):
                        detection_result = self.processor.run_detection_tracking_with_depth(frame_dir, video_name)
                    else:
                        detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
    
                    processing_type = "基本検出（フォールバック）"
    
                    # フォールバック時の警告
                    self.logger.warning("⚠️ yolopose_analyzerが利用できないため、キーポイント機能が制限されます")
                    self.logger.warning("💡 解決策: pip install yolopose-analyzer でインストールしてください")

                except Exception as e:
                    self.logger.error(f"❌ キーポイント検出処理エラー: {e}")
                    import traceback
                    self.logger.error(f"🔧 詳細トレースバック: {traceback.format_exc()}")
                    self.error_collector.append(f"キーポイント検出エラー: {e}")
                    raise VideoProcessingError(f"キーポイント検出に失敗: {e}")

                # 🔧 Step 2結果の最終確認
                if not detection_result.get("success", False):
                    error_msg = detection_result.get("error", "不明なエラー")
                    self.logger.error(f"❌ {processing_type}処理エラー: {error_msg}")
                    self.error_collector.append(f"{processing_type}処理失敗: {error_msg}")
                    raise VideoProcessingError(error_msg)

                self.logger.info(f"✅ Step 2完了: {processing_type}処理")

                # 🔧 検出統計の表示
                if detection_result.get("success", False):
                    data = detection_result.get("data", {})
                    detection_count = data.get("detection_count", 0)
                    frame_count = data.get("frame_count", 0)
    
                    self.logger.info(f"📊 検出統計:")
                    self.logger.info(f"  - 総検出数: {detection_count}")
                    self.logger.info(f"  - 処理フレーム数: {frame_count}")
    
                    if frame_count > 0:
                        detection_rate = (detection_count / frame_count)
                        self.logger.info(f"  - フレーム当たり検出数: {detection_rate:.2f}")

                # 🎯 Step 2.5: 4点キーポイント処理（オプション）
                try:
                    original_csv = detection_result["data"]["csv_path"]
                    filtered_result = self.filter_keypoints_to_6points(original_csv, output_dir)
                    if isinstance(filtered_result, dict) and filtered_result.get("success"):
                        sixpoint_csv = filtered_result.get("sixpoint_csv")
                        metrics_csv = filtered_result.get("metrics_csv")
                        detection_result["data"]["filtered_csv_path"] = sixpoint_csv
                        detection_result["data"]["metrics_csv_path"] = metrics_csv
                        detection_result["data"]["keypoint_mode"] = "6_points"
                        if sixpoint_csv and Path(sixpoint_csv).exists():
                            self.logger.info(f"🎨 6点可視化生成: {sixpoint_csv}")
                            frame_dir = Path(output_dir) / "frames"
                            vis_result = self.create_6point_visualization(output_dir, pd.read_csv(sixpoint_csv), frame_dir)
                        else:
                            self.logger.error(f"❌ 6点CSVファイルが見つかりません: {sixpoint_csv}")
                            vis_result = {"success": False, "error": "6点CSVファイルが見つかりません"}
                    else:
                        self.logger.error(f"❌ 6点フィルタリング失敗: {filtered_result}")
                        vis_result = {"success": False, "error": "6点フィルタリング失敗"}
                
                except Exception as e:
                    self.logger.error(f"❌ 4点処理エラー: {e}")
                    vis_result = {"success": False, "error": str(e)}

                # Step 3: 包括的評価
                self.logger.info("📊 Step 3: 包括的評価開始")
                
                # 評価メソッドの選択
                if hasattr(self.evaluator, 'evaluate_with_depth') and self.depth_enabled:
                    evaluation_result = self.evaluator.evaluate_with_depth(
                        video_path, detection_result, video_name
                    )
                else:
                    evaluation_result = self.evaluator.evaluate_comprehensive(
                        video_path, detection_result, video_name
                    )

                if not evaluation_result.get("success", False):
                    error_msg = f"評価処理失敗: {evaluation_result.get('error', {}).get('message', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    self.logger.warning(f"⚠️ {error_msg}")
                    # 評価失敗は警告に留める
                    evaluation_result = ResponseBuilder.success(data={
                        "basic_evaluation": True, 
                        "fallback": True,
                        "evaluator_type": type(self.evaluator).__name__
                    })

                self.logger.info("✅ Step 3完了: 包括的評価")

                # Step 4: 可視化生成
                self.logger.info("📈 Step 4: 可視化生成開始")
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)

                try:
                    # 🔧 戻り値を受け取って詳細ログ出力
                    vis_result = self.analyzer.create_visualizations(detection_result, vis_dir)
    
                    # 🔧 None チェックを追加
                    if vis_result is None:
                        self.logger.warning("⚠️ Step 4警告: 可視化メソッドがNoneを返しました")
                        vis_result = {"success": False, "error": "可視化メソッドがNoneを返しました"}
                    elif not isinstance(vis_result, dict):
                        self.logger.warning(f"⚠️ Step 4警告: 予期しない戻り値型: {type(vis_result)}")
                        vis_result = {"success": False, "error": f"予期しない戻り値型: {type(vis_result)}"}
    
                    # 🔧 安全な成功判定
                    if vis_result.get("success", False):
                        total_files = vis_result.get("total_files", 0)
                        graphs_count = vis_result.get("graphs_generated", 0)
                        self.logger.info(f"✅ Step 4完了: 可視化生成 ({total_files}個のファイル, {graphs_count}個のグラフ)")
                    else:
                        error_msg = vis_result.get("error", "不明なエラー")
                        self.logger.warning(f"⚠️ Step 4警告: 可視化生成エラー（処理継続）: {error_msg}")
                        self.error_collector.append(f"可視化生成エラー: {error_msg}")
        
                except Exception as e:
                    self.logger.warning(f"⚠️ Step 4警告: 可視化生成エラー（処理継続）: {e}")
                    self.logger.error(f"🔧 Step 4詳細エラー: {e}", exc_info=True)
                    self.error_collector.append(f"可視化生成エラー: {e}")
                    # 🔧 フォールバック用のダミー結果
                    vis_result = {"success": False, "error": str(e)}

                # 統合結果の構築
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
                    "errors": self.error_collector.copy() if self.error_collector else [],
                    "system_info": {
                        "evaluator_type": type(self.evaluator).__name__,
                        "processor_type": type(self.processor).__name__,
                        "analyzer_type": type(self.analyzer).__name__,
                        "config_type": type(self.config).__name__,
                        "module_availability": {
                            "error_handler": ERROR_HANDLER_AVAILABLE,
                            "comprehensive_evaluator": COMPREHENSIVE_EVALUATOR_AVAILABLE,
                            "depth_evaluator": DEPTH_EVALUATOR_AVAILABLE,
                            "video_processor": VIDEO_PROCESSOR_AVAILABLE,
                            "metrics_analyzer": METRICS_ANALYZER_AVAILABLE,
                            "config": CONFIG_AVAILABLE,
                            "logger": LOGGER_AVAILABLE
                        }
                    },
                    # --- ここから summary を追加 ---
                    "summary": {
                        "total_frames": detection_result.get("data", {}).get("frame_count", 0),
                        "total_detections": detection_result.get("data", {}).get("detection_count", 0),
                        "unique_ids": 0,  # 下で補完
                        "csv_path": detection_result.get("data", {}).get("csv_path", None),
                        "errors": self.error_collector.copy() if self.error_collector else [],
                    }
                }
                # ユニークID数をCSVから取得
                csv_path = integrated_result["summary"]["csv_path"]
                if csv_path and Path(csv_path).exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        if 'person_id' in df.columns:
                            integrated_result["summary"]["unique_ids"] = df['person_id'].nunique()
                    except Exception as e:
                        self.logger.warning(f"サマリー用CSV読込エラー: {e}")

                # 結果ファイル保存
                result_file = output_dir / f"{video_name}_baseline_result.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(integrated_result, f, indent=2, ensure_ascii=False)

                if hasattr(ctx, 'add_info'):
                    ctx.add_info("result_file", str(result_file))
                    ctx.add_info("processing_success", True)

                self.logger.info(f"🎉 ベースライン分析完了: {video_name}")
                self.logger.info(f"📁 結果保存先: {output_dir}")
                self.logger.info(f"📄 結果ファイル: {result_file}")

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
    
    # Line 1100付近（run_baseline_analysisメソッドの直後）に追加:

    # 完全置換: Line 2184-2296
    def filter_keypoints_to_6points(self, csv_path, output_dir):
        import pandas as pd
        import os

        self.logger.info("🎯 6点キーポイントフィルタリング開始")
        self.logger.info(f"📂 入力CSV: {csv_path}")

        if not Path(csv_path).exists():
            self.logger.error(f"❌ CSVファイルが存在しません: {csv_path}")
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        df = pd.read_csv(csv_path)
        self.logger.info(f"📋 検出された全列: {list(df.columns)}")

        required = [
            "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
            "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y"
        ]
        filtered = df.dropna(subset=required, how='any').copy()

        confidence_threshold = 0.2
        for kpt in ["left_ear", "right_ear", "left_shoulder", "right_shoulder"]:
            conf_col = f"{kpt}_conf"
            if conf_col in filtered.columns:
                filtered = filtered[filtered[conf_col] >= confidence_threshold]

        # head_center, shoulder_midを計算（filteredが空でも必ずカラムを追加）
        filtered["head_center_x"] = (filtered["left_ear_x"] + filtered["right_ear_x"]) / 2
        filtered["head_center_y"] = (filtered["left_ear_y"] + filtered["right_ear_y"]) / 2
        filtered["shoulder_mid_x"] = (filtered["left_shoulder_x"] + filtered["right_shoulder_x"]) / 2
        filtered["shoulder_mid_y"] = (filtered["left_shoulder_y"] + filtered["right_shoulder_y"]) / 2

        os.makedirs(output_dir, exist_ok=True)
        sixpoint_csv_path = os.path.join(output_dir, "6point_keypoints.csv")

        # 空でも必ずカラムだけのDataFrameを出力
        if len(filtered) == 0:
            filtered = pd.DataFrame(columns=[
                "frame", "person_id",
                "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
                "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
                "head_center_x", "head_center_y", "shoulder_mid_x", "shoulder_mid_y"
            ])
        filtered.to_csv(sixpoint_csv_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"📁 6点データ保存: {sixpoint_csv_path}（{len(filtered)}件）")

        # メトリクスも同様に
        if len(filtered) > 0:
            metrics_df = self._add_6point_metrics(filtered)
        else:
            metrics_df = filtered.copy()
            metrics_df["shoulder_width"] = []
            metrics_df["pose_angle"] = []
            metrics_df["keypoint_completeness"] = []
            metrics_df["pose_confidence"] = []
        metrics_csv_path = os.path.join(output_dir, "6point_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"📁 メトリクス保存: {metrics_csv_path}")

        return {
            "success": True,
            "sixpoint_csv": sixpoint_csv_path,
            "metrics_csv": metrics_csv_path,
            "valid_detections": len(filtered),
            "total_detections": len(df),
            "filter_rate": len(filtered) / len(df) if len(df) > 0 else 0
        }

    def _add_6point_metrics(self, df):
        """
        6点キーポイント専用メトリクス計算（shoulder_head_angle追加・省略なし完全版）
        """
        import numpy as np

        self.logger.info("📊 6点メトリクス計算開始")

        metrics_df = df.copy()

        # メトリクス初期化
        metrics_df['shoulder_width'] = 0.0
        metrics_df['head_center_x'] = 0.0
        metrics_df['head_center_y'] = 0.0
        metrics_df['shoulder_mid_x'] = 0.0
        metrics_df['shoulder_mid_y'] = 0.0
        metrics_df['pose_angle'] = 0.0
        metrics_df['keypoint_completeness'] = 0.0
        metrics_df['pose_confidence'] = 0.0
        metrics_df['shoulder_head_angle'] = 0.0  # ★なす角

        calculated_count = 0
        shoulder_width_count = 0
        head_position_count = 0
        pose_angle_count = 0
        angle_count = 0

        for idx, row in metrics_df.iterrows():
            try:
                # 肩幅
                if ('left_shoulder_x' in row and 'right_shoulder_x' in row and
                    'left_shoulder_y' in row and 'right_shoulder_y' in row):
                    left_x, left_y = float(row['left_shoulder_x']), float(row['left_shoulder_y'])
                    right_x, right_y = float(row['right_shoulder_x']), float(row['right_shoulder_y'])
                    if left_x > 0 and left_y > 0 and right_x > 0 and right_y > 0:
                        shoulder_width = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
                        metrics_df.at[idx, 'shoulder_width'] = shoulder_width
                        shoulder_width_count += 1

                # head_center
                if ('left_ear_x' in row and 'right_ear_x' in row and
                    'left_ear_y' in row and 'right_ear_y' in row):
                    left_ear_x, left_ear_y = float(row['left_ear_x']), float(row['left_ear_y'])
                    right_ear_x, right_ear_y = float(row['right_ear_x']), float(row['right_ear_y'])
                    if left_ear_x > 0 and left_ear_y > 0 and right_ear_x > 0 and right_ear_y > 0:
                        head_center_x = (left_ear_x + right_ear_x) / 2
                        head_center_y = (left_ear_y + right_ear_y) / 2
                        metrics_df.at[idx, 'head_center_x'] = head_center_x
                        metrics_df.at[idx, 'head_center_y'] = head_center_y
                        head_position_count += 1

                # 両肩の中点
                if ('left_shoulder_x' in row and 'right_shoulder_x' in row and
                    'left_shoulder_y' in row and 'right_shoulder_y' in row):
                    left_x, left_y = float(row['left_shoulder_x']), float(row['left_shoulder_y'])
                    right_x, right_y = float(row['right_shoulder_x']), float(row['right_shoulder_y'])
                    if left_x > 0 and left_y > 0 and right_x > 0 and right_y > 0:
                        shoulder_mid_x = (left_x + right_x) / 2
                        shoulder_mid_y = (left_y + right_y) / 2
                        metrics_df.at[idx, 'shoulder_mid_x'] = shoulder_mid_x
                        metrics_df.at[idx, 'shoulder_mid_y'] = shoulder_mid_y

                # 姿勢角度（肩ライン）
                if (metrics_df.at[idx, 'shoulder_width'] > 0 and
                    'left_shoulder_x' in row and 'right_shoulder_x' in row and
                    'left_shoulder_y' in row and 'right_shoulder_y' in row):
                    left_x, left_y = float(row['left_shoulder_x']), float(row['left_shoulder_y'])
                    right_x, right_y = float(row['right_shoulder_x']), float(row['right_shoulder_y'])
                    if left_x > 0 and right_x > 0:
                        angle_rad = np.arctan2(right_y - left_y, right_x - left_x)
                        angle_deg = np.degrees(angle_rad)
                        metrics_df.at[idx, 'pose_angle'] = angle_deg
                        pose_angle_count += 1

                # ★肩の中点とhead_centerのなす角
                sx, sy = metrics_df.at[idx, 'shoulder_mid_x'], metrics_df.at[idx, 'shoulder_mid_y']
                hx, hy = metrics_df.at[idx, 'head_center_x'], metrics_df.at[idx, 'head_center_y']
                if sx > 0 and sy > 0 and hx > 0 and hy > 0:
                    dx = hx - sx
                    dy = sy - hy  # y軸反転考慮
                    theta = np.degrees(np.arctan2(dy, dx))  # 水平右向き0度、上向き正
                    metrics_df.at[idx, 'shoulder_head_angle'] = theta
                    angle_count += 1

                # キーポイント完全性スコア
                available_keypoints = [
                    'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                    'head_center', 'shoulder_mid'
                ]
                valid_keypoints = 0
                total_keypoints = len(available_keypoints)
                for kpt in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']:
                    x_col, y_col = f"{kpt}_x", f"{kpt}_y"
                    if (x_col in row and y_col in row):
                        if float(row[x_col]) > 0 and float(row[y_col]) > 0:
                            valid_keypoints += 1
                if ('head_center_x' in row and 'head_center_y' in row):
                    if float(row['head_center_x']) > 0 and float(row['head_center_y']) > 0:
                        valid_keypoints += 1
                if ('shoulder_mid_x' in row and 'shoulder_mid_y' in row):
                    if float(row['shoulder_mid_x']) > 0 and float(row['shoulder_mid_y']) > 0:
                        valid_keypoints += 1

                completeness = valid_keypoints / total_keypoints
                metrics_df.at[idx, 'keypoint_completeness'] = completeness

                # ポーズ信頼度
                pose_confidence = float(row['conf']) * completeness if 'conf' in row else completeness
                metrics_df.at[idx, 'pose_confidence'] = pose_confidence

                calculated_count += 1
    
            except Exception as row_error:
                self.logger.debug(f"行 {idx} の6点メトリクス計算エラー: {row_error}")
                continue

        # 統計ログ
        total_rows = len(metrics_df)
        self.logger.info(f"📊 6点メトリクス計算完了:")
        self.logger.info(f"  処理行数: {calculated_count}/{total_rows}")
        self.logger.info(f"  肩幅計算: {shoulder_width_count}行")
        self.logger.info(f"  頭部位置: {head_position_count}行")
        self.logger.info(f"  姿勢角度: {pose_angle_count}行")
        self.logger.info(f"  なす角計算: {angle_count}行")

        if calculated_count > 0:
            avg_shoulder_width = metrics_df[metrics_df['shoulder_width'] > 0]['shoulder_width'].mean()
            avg_completeness = metrics_df['keypoint_completeness'].mean()
            avg_pose_conf = metrics_df['pose_confidence'].mean()
            avg_angle = metrics_df['shoulder_head_angle'].mean()
            self.logger.info(f"📊 メトリクス統計:")
            self.logger.info(f"  平均肩幅: {avg_shoulder_width:.1f}px")
            self.logger.info(f"  平均完全性: {avg_completeness:.2f}")
            self.logger.info(f"  平均ポーズ信頼度: {avg_pose_conf:.2f}")
            self.logger.info(f"  平均なす角: {avg_angle:.2f}度")

        return metrics_df

    def create_6point_visualization(self, output_dir, keypoints_df, frame_dir, log_path=None, apply_undistort=True):
        import cv2
        from pathlib import Path
        # from utils.camera_calibration import undistort_with_json

        vis_dir = Path(output_dir) / "visualized_frames_6points"
        vis_dir.mkdir(parents=True, exist_ok=True)

        if keypoints_df.empty:
            self.logger.warning("⚠️ キーポイントデータが空です。可視化画像は生成されません。")
            return {"success": False, "output_dir": str(vis_dir), "saved_count": 0}

        saved_count = 0

        for frame_name in keypoints_df["frame"].unique():
            frame_path = Path(frame_dir) / frame_name
            if not frame_path.exists():
                self.logger.warning(f"⚠️ フレーム画像が見つかりません: {frame_path}")
                continue

            frame = cv2.imread(str(frame_path))
            # --- 修正: apply_undistortフラグで制御 ---
            if apply_undistort:
                from utils.camera_calibration import undistort_with_json
                frame = undistort_with_json(frame, calib_path="configs/camera_params.json")

            rows = keypoints_df[keypoints_df['frame'] == frame_name]
            for _, row in rows.iterrows():
                keypoints = {
                    "left_ear": (row["left_ear_x"], row["left_ear_y"], row.get("left_ear_conf", 1.0)),
                    "right_ear": (row["right_ear_x"], row["right_ear_y"], row.get("right_ear_conf", 1.0)),
                    "left_shoulder": (row["left_shoulder_x"], row["left_shoulder_y"], row.get("left_shoulder_conf", 1.0)),
                    "right_shoulder": (row["right_shoulder_x"], row["right_shoulder_y"], row.get("right_shoulder_conf", 1.0)),
                }
                frame = self.draw_6point_keypoints(frame, keypoints, row)

            output_filename = f"6pt_{frame_name}"
            output_path = vis_dir / output_filename
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        self.logger.info(f"✅ 6点可視化画像を{saved_count}枚保存しました（{vis_dir}）")
        return {"success": True, "output_dir": str(vis_dir), "saved_count": saved_count}
    
    def draw_6point_keypoints(self, frame, keypoints, row, log_path=None):
        """
        6点キーポイント（両肩・両耳・head_center・両肩の中点）をシンプルな色で描画し、
        検出枠とIDも画像上に表示する
        """
        import cv2
        import json

        # シンプルな色設定
        kpt_color = (255, 0, 0)      # 青（全キーポイント共通）
        bbox_color = (0, 255, 0)     # 緑（検出枠）
        id_color = (0, 0, 255)       # 赤（ID）

        # 両耳・両肩の座標取得
        left_ear = keypoints.get('left_ear', None)
        right_ear = keypoints.get('right_ear', None)
        left_shoulder = keypoints.get('left_shoulder', None)
        right_shoulder = keypoints.get('right_shoulder', None)

        # head_center
        head_center_x = row.get('head_center_x')
        head_center_y = row.get('head_center_y')
        head_center = None
        if head_center_x is not None and head_center_y is not None:
            head_center = (int(head_center_x), int(head_center_y))

        # 両肩の中点
        shoulder_midpoint = None
        if left_shoulder and right_shoulder:
            shoulder_midpoint = (
                int((left_shoulder[0] + right_shoulder[0]) / 2),
                int((left_shoulder[1] + right_shoulder[1]) / 2)
            )

        # 4点＋head_center＋両肩中点を描画（全て同じ色・シンプル）
        for kpt_name, (x, y, conf) in keypoints.items():
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 6, kpt_color, -1)

        if head_center:
            cv2.circle(frame, head_center, 8, kpt_color, -1)
        if shoulder_midpoint:
            cv2.circle(frame, shoulder_midpoint, 8, kpt_color, -1)

        # 検出枠描画
        if all(k in row for k in ["x1", "y1", "x2", "y2"]):
            try:
                x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            except Exception:
                pass

        # ID表示
        if "person_id" in row and row["person_id"] is not None:
            pid = str(row["person_id"])
            # 枠の左上 or キーポイントの近くに表示
            if all(k in row for k in ["x1", "y1"]):
                pos = (int(row["x1"]), max(0, int(row["y1"]) - 10))
            elif left_shoulder:
                pos = (int(left_shoulder[0]), int(left_shoulder[1]) - 10)
            else:
                pos = (10, 30)
            cv2.putText(frame, f"ID:{pid}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, id_color, 2)

        # ログ保存（必要なら）
        log_data = {
            "frame": row.get("frame"),
            "person_id": row.get("person_id"),
            "left_ear": left_ear[:2] if left_ear else None,
            "right_ear": right_ear[:2] if right_ear else None,
            "left_shoulder": left_shoulder[:2] if left_shoulder else None,
            "right_shoulder": right_shoulder[:2] if right_shoulder else None,
            "head_center": head_center,
            "shoulder_midpoint": shoulder_midpoint,
        }
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        return frame

    def draw_4point_keypoints_dynamic(self, frame, keypoint_data, row):
        """動的4点キーポイント描画"""
        try:
            import cv2
            import numpy as np
        
            # ⚡ より目立つ色とサイズ
            ear_color = (0, 255, 0)       # 緑（耳）
            shoulder_color = (0, 100, 255) # オレンジ（肩）
            line_color = (0, 255, 255)     # 黄（線）
            text_color = (255, 255, 255)   # 白（テキスト）
        
            # ⚡ キーポイント描画
            ear_points = []
            shoulder_points = []
        
            for name, (x, y, conf) in keypoint_data.items():
                color = ear_color if 'ear' in name else shoulder_color
            
                # ⚡ 大きな円で描画
                cv2.circle(frame, (x, y), 8, color, -1)  
                cv2.circle(frame, (x, y), 12, text_color, 2)
            
                # ⚡ キーポイント名と信頼度
                cv2.putText(frame, f"{name.split('_')[0]}:{conf:.2f}", 
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
                # 点を分類
                if 'ear' in name:
                    ear_points.append((x, y))
                elif 'shoulder' in name:
                    shoulder_points.append((x, y))
        
            # ⚡ 肩ライン
            if len(shoulder_points) == 2:
                cv2.line(frame, shoulder_points[0], shoulder_points[1], line_color, 4)
        
            # ⚡ 頭部中心
            if len(ear_points) == 2:
                head_x = (ear_points[0][0] + ear_points[1][0]) // 2
                head_y = (ear_points[0][1] + ear_points[1][1]) // 2
                cv2.circle(frame, (head_x, head_y), 6, line_color, -1)
        
            # 人物ID表示
            person_id = row.get('person_id', -1)
            if person_id != -1 and keypoint_data:
                all_points = list(keypoint_data.values())
                center_x = int(np.mean([p[0] for p in all_points]))
                center_y = int(np.mean([p[1] for p in all_points])) - 30
            
                cv2.putText(frame, f"ID:{person_id}", (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
            return frame
        
        except Exception as e:
            self.logger.warning(f"動的描画エラー: {e}")
            return frame

    @handle_errors(error_category=ErrorCategory.EXPERIMENT)
    def run_experiment(self, video_path: str, experiment_type: str) -> Dict[str, Any]:
        """
        実験分析実行（完全統合版）

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

            self.logger.info(f"🧪 実験分析開始: {experiment_type} - {video_name}")

            if hasattr(ctx, 'add_info'):
                ctx.add_info("video_path", str(video_path))
                ctx.add_info("experiment_type", experiment_type)
                ctx.add_info("depth_enabled", self.depth_enabled)

            try:
                # 実験用出力ディレクトリ
                output_dir = Path("outputs/experiments") / experiment_type / video_name
                output_dir.mkdir(parents=True, exist_ok=True)

                self.logger.info(f"📁 実験出力ディレクトリ: {output_dir}")

                # 実験設定の取得
                if hasattr(self.config, 'get_experiment_config'):
                    experiment_config = self.config.get_experiment_config(experiment_type)
                else:
                    experiment_config = {"type": experiment_type, "basic_mode": True}

                self.logger.info(f"⚙️ 実験設定: {experiment_config}")

                # ベースライン結果との比較用にベースライン実行
                self.logger.info("📊 ベースライン結果取得中...")
                baseline_result = self.run_baseline_analysis(video_path)
                
                if not baseline_result.get("success", False):
                    raise VideoProcessingError("ベースライン分析に失敗しました")

                self.logger.info("✅ ベースライン結果取得完了")

                # 実験特有の処理
                experiment_result = {
                    "success": True,
                    "experiment_type": experiment_type,
                    "video_name": video_name,
                    "baseline_comparison": baseline_result.get("data", {}),
                    "experiment_config": experiment_config,
                    "depth_enabled": self.depth_enabled,
                    "output_directory": str(output_dir),
                    "processing_timestamp": datetime.now().isoformat(),
                    "system_info": {
                        "evaluator_type": type(self.evaluator).__name__,
                        "processor_type": type(self.processor).__name__,
                        "analyzer_type": type(self.analyzer).__name__
                    }
                }

                # 改善分析
                try:
                    self.logger.info("📈 改善分析開始...")
                    improvement_analysis = self.analyzer.analyze_improvements({
                        "baseline": baseline_result.get("data", {}),
                        "experiment": experiment_result
                    })
                    experiment_result["improvement_analysis"] = improvement_analysis
                    self.logger.info("✅ 改善分析完了")
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
                self.logger.info(f"📄 結果ファイル: {result_file}")
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
        """エラーレポート生成（完全版）"""
        try:
            self.logger.info("📋 エラーレポート生成開始")
            
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "total_errors": len(self.error_collector),
                "errors": self.error_collector.copy(),
                "system_info": {
                    "depth_enabled": self.depth_enabled,
                    "evaluator_type": type(self.evaluator).__name__,
                    "processor_type": type(self.processor).__name__,
                    "analyzer_type": type(self.analyzer).__name__,
                    "config_type": type(self.config).__name__
                },
                "module_availability": {
                    "error_handler": ERROR_HANDLER_AVAILABLE,
                    "evaluator": EVALUATOR_AVAILABLE,
                    "comprehensive_evaluator": COMPREHENSIVE_EVALUATOR_AVAILABLE,
                    "depth_evaluator": DEPTH_EVALUATOR_AVAILABLE,
                    "video_processor": VIDEO_PROCESSOR_AVAILABLE,
                    "metrics_analyzer": METRICS_ANALYZER_AVAILABLE,
                    "config": CONFIG_AVAILABLE,
                    "logger": LOGGER_AVAILABLE
                },
                "performance_info": {
                    "processing_stats": getattr(self.processor, 'processing_stats', {}),
                    "error_count_by_type": self._categorize_errors()
                }
            }
            
            # エラーレポートファイル保存
            report_file = Path("logs") / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"✅ エラーレポート生成完了: {report_file}")
            return error_report
            
        except Exception as e:
            self.logger.error(f"❌ エラーレポート生成失敗: {e}")
            return {"error": str(e)}

    def _categorize_errors(self) -> Dict[str, int]:
        """エラーの分類（完全版）"""
        categories = {
            "video_processing": 0,
            "model_loading": 0,
            "evaluation": 0,
            "configuration": 0,
            "depth_processing": 0,
            "io_operations": 0,
            "other": 0
        }
        
        for error in self.error_collector:
            error_lower = error.lower()
            if any(keyword in error_lower for keyword in ["video", "frame", "opencv", "mp4", "avi"]):
                categories["video_processing"] += 1
            elif any(keyword in error_lower for keyword in ["model", "yolo", "loading", "pt", "weights"]):
                categories["model_loading"] += 1
            elif any(keyword in error_lower for keyword in ["evaluation", "csv", "analysis", "metrics"]):
                categories["evaluation"] += 1
            elif any(keyword in error_lower for keyword in ["config", "setting", "yaml", "json"]):
                categories["configuration"] += 1
            elif any(keyword in error_lower for keyword in ["depth", "midas", "disparity"]):
                categories["depth_processing"] += 1
            elif any(keyword in error_lower for keyword in ["file", "directory", "path", "permission"]):
                categories["io_operations"] += 1
            else:
                categories["other"] += 1
        
        return categories

    def get_video_files(self) -> List[Path]:
        """動画ファイル取得（修正版・基本推論優先）"""
        try:
            # 🔧 --video オプションが指定された場合は、そのファイルのみ処理
            if hasattr(sys, 'argv') and '--video' in ' '.join(sys.argv):
                # コマンドライン引数から動画ファイルを直接取得する場合は
                # メイン関数で処理されるので、ここでは空リストを返す
                return []
        
            # 通常の動画ディレクトリ検索
            if hasattr(self.config, 'video_dir'):
                video_dir = Path(self.config.video_dir)
            else:
                video_dir = Path(self.config.get("video_dir", "videos"))
            
            self.logger.info(f"🎥 動画ディレクトリ検索: {video_dir}")
            
            if not video_dir.exists():
                self.logger.warning(f"⚠️ 動画ディレクトリが存在しません: {video_dir}")
                return []
            
            # サポートする動画形式
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = []
        
            for ext in video_extensions:
                found_files = list(video_dir.glob(f"*{ext}"))
                found_files.extend(video_dir.glob(f"*{ext.upper()}"))
                video_files.extend(found_files)
            
            video_files = sorted(set(video_files))
        
            if not video_files:
                self.logger.warning(f"⚠️ 動画ファイルが見つかりません: {video_dir}")
                return []
        
            self.logger.info(f"✅ 動画ファイル発見: {len(video_files)}個")
            for video_file in video_files:
                file_size_mb = video_file.stat().st_size / 1024 / 1024
                self.logger.debug(f"  📹 {video_file.name} ({file_size_mb:.1f}MB)")
            
            return video_files
        
        except Exception as e:
            self.logger.error(f"❌ 動画ファイル取得エラー: {e}")
            return []

def main():
    """
    メイン実行関数（タイムスタンプ付きディレクトリで毎回結果を保存し、過去の結果を残す仕様）
    既存の処理・設定・ログ・サマリー出力などはそのまま維持しつつ、全成果物を
    outputs/baseline/動画名_タイムスタンプ/
    に保存するように修正
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                     🎯 YOLO11 姿勢分析システム v2.1                    ║
    ║                        キーポイント検出・追跡・解析                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    parser = argparse.ArgumentParser(
        description="🎯 YOLO11姿勢分析システム - 動画から人物の姿勢を分析します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python improved_main.py input.mp4
    python improved_main.py input.mp4 --use-4points --keypoint-threshold 0.5
    python improved_main.py input.mp4 --enable-depth --depth-model dpt_hybrid
    python improved_main.py input.mp4 --config custom_config.yaml
    python improved_main.py input.mp4 --resolution 1920x1080 --quality high
    python improved_main.py --csv step6_xxxx/results.csv --frames-dir step6_xxxx/frames
        """
    )

    parser.add_argument('video_path', type=str, nargs='?', help='🎬 分析対象の動画ファイルパス')
    parser.add_argument('--config', type=str, default=None, help='⚙️ 設定ファイルパス（YAML/JSON形式）')
    parser.add_argument('--output-dir', type=str, default=None, help='📁 出力ディレクトリ')
    parser.add_argument('--use-4points', action='store_true', help='🦴 4点キーポイントモードを有効化')
    parser.add_argument('--keypoint-threshold', type=float, default=0.3, help='🎯 キーポイント信頼度閾値')
    parser.add_argument('--disable-shoulder-metrics', action='store_true', help='🚫 肩幅メトリクスを無効化')
    parser.add_argument('--disable-head-tracking', action='store_true', help='🚫 頭部追跡機能を無効化')
    parser.add_argument('--enable-depth', action='store_true', help='🌊 深度推定機能を有効化')
    parser.add_argument('--depth-model', type=str, default='dpt_hybrid', choices=['dpt_hybrid', 'midas', 'dpt_large'], help='🧠 深度推定モデルの選択')
    parser.add_argument('--resolution', type=str, default=None, help='📐 処理解像度（例: 1920x1080, 1280x720）')
    parser.add_argument('--fps', type=int, default=None, help='🎬 処理FPS')
    parser.add_argument('--quality', type=str, default='medium', choices=['low', 'medium', 'high', 'ultra'], help='🎨 処理品質レベル')
    parser.add_argument('--skip-frames', type=int, default=0, help='⏭️ スキップフレーム数')
    parser.add_argument('--debug', action='store_true', help='🐛 デバッグモードを有効化')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='📝 ログレベル')
    parser.add_argument('--save-intermediate', action='store_true', help='💾 中間ファイルを保存')
    parser.add_argument('--model-size', type=str, default='x', choices=['n', 's', 'm', 'l', 'x'], help='🎯 YOLOモデルサイズ')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='🎯 検出信頼度閾値')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='📐 IoU閾値')
    parser.add_argument('--disable-visualization', action='store_true', help='🚫 可視化出力を無効化')
    parser.add_argument('--output-format', type=str, default='csv', choices=['csv', 'json', 'both'], help='📊 出力データ形式')
    parser.add_argument('--csv', type=str, default=None, help='📊 既存検出結果CSV（動画推論せず可視化・メトリクスのみ実行）')
    parser.add_argument('--frames-dir', type=str, default=None, help='🖼️ フレーム画像ディレクトリ（CSVと合わせて指定）')

    args = parser.parse_args()

    # --- ここから正規化処理の分岐を追加 ---
    print("正規化処理を使いますか？(y/n): ", end="")
    use_normalization = input().strip().lower() == "y"
    normalization_params = {}
    normalization_input_csv = None

    if use_normalization:
        print("パラメータjson（linear/exp）があるフォルダを指定してください: ", end="")
        json_dir = input().strip()
        linear_json = os.path.join(json_dir, "function_parameters_linear.json")
        exp_json = os.path.join(json_dir, "function_parameters_exp.json")
        if os.path.exists(linear_json):
            normalization_params["linear"] = linear_json
        if os.path.exists(exp_json):
            normalization_params["exp"] = exp_json
        if not normalization_params:
            print("❌ function_parameters_linear.json/function_parameters_exp.jsonが見つかりません。正規化処理をスキップします。")
            use_normalization = False

        # ★ ここで元CSVファイルのパスも聞く
        print("正規化対象の6点メトリクスCSVファイル（例: 6point_metrics_with_column.csv）のパスを指定してください: ", end="")
        normalization_input_csv = input().strip()
        if not os.path.exists(normalization_input_csv):
            print(f"❌ 指定されたCSVが見つかりません: {normalization_input_csv}")
            use_normalization = False

    # ログレベル設定
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('yolo_pose_analysis.log', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 デバッグモードが有効化されました")

    try:
        # --- 既存CSVとフレーム画像から可視化・メトリクスのみ実行する場合 ---
        if args.csv and args.frames_dir:
            csv_path = Path(args.csv)
            frame_dir = Path(args.frames_dir)
            output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent

            logger.info(f"📊 既存CSVから6点抽出・可視化・メトリクス処理を開始します")
            analyzer = ImprovedYOLOAnalyzer(config_path=args.config or "configs/default.yaml")

            # 6点キーポイント抽出＆メトリクス計算
            filter_result = analyzer.filter_keypoints_to_6points(str(csv_path), str(output_dir))
            sixpoint_csv = filter_result["sixpoint_csv"]

            # 可視化画像生成（歪み補正をかけない！）
            import pandas as pd
            keypoints_df = pd.read_csv(sixpoint_csv)
            analyzer.create_6point_visualization(str(output_dir), keypoints_df, frame_dir, apply_undistort=False)

            logger.info("✅ 既存CSVからの6点可視化・メトリクス処理が完了しました")
            return 0

        # --- 通常の動画推論処理 ---
        if not args.video_path or not Path(args.video_path).exists():
            logger.error(f"❌ 動画ファイルが見つかりません: {args.video_path}")
            return 1

        video_path = Path(args.video_path)
        logger.info(f"🎬 動画ファイル: {video_path}")

        # タイムスタンプ付きディレクトリ名を生成（outputs/baseline/動画名_タイムスタンプ）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = video_path.stem
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("outputs/baseline") / f"{video_name}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 出力ディレクトリ: {output_dir}")

        # アナライザー初期化
        try:
            analyzer = ImprovedYOLOAnalyzer(config_path=args.config or "configs/default.yaml")
            if args.enable_depth:
                analyzer.depth_enabled = True
                logger.info("🔍 深度推定機能を有効化")
            logger.info("✅ アナライザー初期化完了")
        except Exception as e:
            logger.error(f"❌ アナライザー初期化エラー: {e}")
            import traceback
            logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
            return 1

        # 4点キーポイント設定
        if args.use_4points:
            try:
                if hasattr(analyzer, 'config') and hasattr(analyzer.config, 'data') and isinstance(analyzer.config.data, dict):
                    analyzer.config.data.setdefault('processing', {})
                    analyzer.config.data['processing']['use_4point_keypoints'] = True
                    analyzer.config.data['processing']['keypoint_confidence_threshold'] = args.keypoint_threshold
                    analyzer.config.data['processing']['force_pose_model'] = True
                    analyzer.config.data['processing']['verify_keypoint_columns'] = True
                    analyzer.config.data['processing'].setdefault('tracking', {})
                    analyzer.config.data['processing']['tracking']['config'] = 'bytetrack.yaml'
                    analyzer.config.data['processing']['enable_shoulder_metrics'] = not args.disable_shoulder_metrics
                    analyzer.config.data['processing']['enable_head_tracking'] = not args.disable_head_tracking
                    logger.info("🔧 設定ファイルを4点キーポイントモード用に更新")
                else:
                    logger.error("❌ 設定オブジェクトが不正です")
            except Exception as config_error:
                logger.error(f"❌ 4点モード設定エラー: {config_error}")
                logger.warning("⚠️ デフォルト設定で処理を続行します")

        # 品質設定
        quality_configs = {
            'low': {'resolution': '640x480', 'skip_frames': 2},
            'medium': {'resolution': '1280x720', 'skip_frames': 1},
            'high': {'resolution': '1920x1080', 'skip_frames': 0},
            'ultra': {'resolution': '1920x1080', 'skip_frames': 0}
        }
        if args.quality in quality_configs:
            quality_config = quality_configs[args.quality]
            if not args.resolution:
                args.resolution = quality_config['resolution']
            if args.skip_frames == 0:
                args.skip_frames = quality_config['skip_frames']

        # 解像度設定
        if args.resolution:
            try:
                width, height = map(int, args.resolution.split('x'))
                if hasattr(analyzer.config, 'data'):
                    analyzer.config.data.setdefault('processing', {})
                    analyzer.config.data['processing']['target_width'] = width
                    analyzer.config.data['processing']['target_height'] = height
                logger.info(f"📐 解像度設定: {width}x{height}")
            except ValueError:
                logger.warning(f"⚠️ 不正な解像度形式: {args.resolution}")

        # その他の処理設定
        if hasattr(analyzer.config, 'data') and analyzer.config.data:
            processing_config = analyzer.config.data.setdefault('processing', {})
            processing_config['confidence_threshold'] = args.confidence_threshold
            processing_config['iou_threshold'] = args.iou_threshold
            if args.fps:
                processing_config['target_fps'] = args.fps
            processing_config['skip_frames'] = args.skip_frames
            processing_config['save_intermediate'] = args.save_intermediate
            processing_config['enable_visualization'] = not args.disable_visualization
            model_size_map = {'n': 'nano', 's': 'small', 'm': 'medium', 'l': 'large', 'x': 'xlarge'}
            processing_config['model_size'] = model_size_map.get(args.model_size, 'xlarge')

        logger.info("🚀 ========== 姿勢分析処理開始 ==========")
        import time
        start_time = time.time()

        try:
            # ベースライン分析実行（タイムスタンプ付きoutput_dirを渡す）
            result = analyzer.run_baseline_analysis(str(video_path), output_dir=output_dir)
            if not result.get("success", False):
                error_msg = result.get("error", "不明なエラー")
                logger.error(f"❌ 分析処理失敗: {error_msg}")
                return 1

            processing_time = time.time() - start_time
            logger.info(f"⏱️ 総処理時間: {processing_time:.2f}秒")

            # サマリー取得（summary優先）
            data = result.get("data", {})
            summary = data.get("summary", {})
            total_frames = summary.get("total_frames", 0)
            total_detections = summary.get("total_detections", 0)
            unique_ids = summary.get("unique_ids", 0)
            errors = summary.get("errors", [])

            # サマリーがなければCSVから再取得
            if (not total_frames or not total_detections) and summary.get("csv_path"):
                csv_path = summary.get("csv_path")
                if csv_path and Path(csv_path).exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    total_detections = len(df)
                    total_frames = len(df['frame'].unique()) if 'frame' in df.columns else 0
                    unique_ids = len(df['person_id'].unique()) if 'person_id' in df.columns else 0

            logger.info("📊 ========== 処理結果サマリー ==========")
            logger.info(f"🎬 総フレーム数: {total_frames}")
            logger.info(f"🎯 総検出数: {total_detections}")
            logger.info(f"👥 ユニーク人物ID: {unique_ids}")

            if total_frames > 0:
                detection_rate = total_detections / total_frames
                logger.info(f"📈 フレーム当たり検出数: {detection_rate:.2f}")

            # エラー報告
            if errors:
                logger.warning(f"⚠️ 処理中のエラー: {len(errors)}件")
                for i, error in enumerate(errors[:5], 1):
                    logger.warning(f"  {i}. {error}")
                if len(errors) > 5:
                    logger.warning(f"  ... 他 {len(errors) - 5}件")

            # パフォーマンス統計
            fps = total_frames / processing_time if processing_time > 0 else 0
            logger.info(f"⚡ 処理性能: {fps:.2f} FPS")

            # サマリーJSONもタイムスタンプ付きoutput_dirに保存
            summary_file = output_dir / f"{video_name}_{timestamp}_summary.json"
            import json
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 サマリー保存: {summary_file}")

            # --- 正規化のみ実行する場合 ---
            if use_normalization and normalization_input_csv:
                import pandas as pd
                df = pd.read_csv(normalization_input_csv)

                output_dir = os.path.dirname(normalization_input_csv)

                # 直線近似
                if "linear" in normalization_params:
                    from analysis.normalization_preparation import load_linear_params, normalize_value_by_linear
                    a_l, b_l, c_l = load_linear_params(os.path.dirname(normalization_params["linear"]))
                    df_linear = df[df["column_position"].notnull()].copy()
                    if "shoulder_width" in df_linear.columns and "column_position" in df_linear.columns:
                        df_linear["shoulder_width_normalized_linear"] = df_linear.apply(
                            lambda row: normalize_value_by_linear(
                                row["shoulder_width"],
                                row["column_position"],
                                a_l, b_l, c_l,
                                reference_distance=1
                            ),
                            axis=1
                        )
                    out_csv_linear = os.path.join(output_dir, "6point_metrics_normalized_linear.csv")
                    df_linear.to_csv(out_csv_linear, index=False, encoding="utf-8-sig")
                    print(f"✅ 直線近似で正規化済みCSVを保存しました: {out_csv_linear}")

                # 指数近似
                if "exp" in normalization_params:
                    from analysis.normalization_preparation import load_exponential_params, normalize_value_by_decay
                    a_e, b_e, c_e = load_exponential_params(os.path.dirname(normalization_params["exp"]))
                    df_exp = df[df["column_position"].notnull()].copy()
                    if "shoulder_width" in df_exp.columns and "column_position" in df_exp.columns:
                        df_exp["shoulder_width_normalized_exp"] = df_exp.apply(
                            lambda row: normalize_value_by_decay(
                                row["shoulder_width"],
                                row["column_position"],
                                a_e, b_e, c_e,
                                reference_distance=1
                            ),
                            axis=1
                        )
                    out_csv_exp = os.path.join(output_dir, "6point_metrics_normalized_exp.csv")
                    df_exp.to_csv(out_csv_exp, index=False, encoding="utf-8-sig")
                    print(f"✅ 指数近似で正規化済みCSVを保存しました: {out_csv_exp}")

                return 0

        except Exception as e:
            logger.error(f"❌ 分析処理中にエラーが発生: {e}")
            import traceback
            logger.error(f"🔧 トレースバック:\n{traceback.format_exc()}")
            return 1

    except KeyboardInterrupt:
        logger.warning("⏸️ ユーザーによる処理中断")
        return 130

    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        import traceback
        logger.error(f"🔧 トレースバック:\n{traceback.format_exc()}")
        return 1



if __name__ == "__main__":
    # システム情報出力
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {platform.platform()}")
    print(f"🧠 CPU Count: {os.cpu_count()}")
    
    # GPU情報確認
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA: {torch.version.cuda}")
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 GPU: CUDA利用不可（CPU処理）")
    except ImportError:
        print("⚠️ PyTorch未インストール")
    
    print()
    
    # メイン処理実行
    sys.exit(main())