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

        def extract_frames(self, video_path, frame_dir, max_frames=1000):
            """フレーム抽出（統計修正版）"""
            try:
                # 🔧 processing_statsの確実な初期化
                if not hasattr(self, 'processing_stats'):
                    self.processing_stats = {}
            
                self.logger.info(f"📸 フレーム抽出開始: {video_path}")
                frame_dir = Path(frame_dir)
                frame_dir.mkdir(parents=True, exist_ok=True)

                # 動画ファイルの存在確認
                if not Path(video_path).exists():
                    self.logger.error(f"❌ 動画ファイルが存在しません: {video_path}")
                    return {"success": False, "error": f"動画ファイルが存在しません: {video_path}"}

                # ファイルサイズチェック
                file_size = Path(video_path).stat().st_size
                if file_size == 0:
                    self.logger.error(f"❌ 動画ファイルが空です: {video_path}")
                    return {"success": False, "error": f"動画ファイルが空です: {video_path}"}

                self.logger.info(f"📹 動画ファイルサイズ: {file_size / (1024*1024):.1f}MB")

                # 🔧 OpenCVでフレーム抽出
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"❌ 動画ファイルを開けません: {video_path}")
                    return {"success": False, "error": f"動画ファイルを開けません: {video_path}"}

                # 動画情報取得
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0

                self.logger.info(f"📹 動画情報: {width}x{height}, {frame_count}フレーム, {fps:.1f}FPS, {duration:.1f}秒")

                # フレーム数が0の場合のエラーハンドリング
                if frame_count <= 0:
                    cap.release()
                    self.logger.error(f"❌ 有効なフレームが見つかりません: {video_path}")
                    return {"success": False, "error": "有効なフレームが見つかりません"}

                # 抽出間隔計算
                interval = max(1, frame_count // max_frames)
                self.logger.info(f"🔢 抽出間隔: {interval} (最大{max_frames}フレーム)")

                # フレーム抽出ループ
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
                            self.logger.warning(f"⚠️ フレーム保存失敗: {frame_path}")
            
                    frame_number += 1

                cap.release()

                # 🔧 実際に保存されたフレーム数を再確認（重要！）
                saved_frames = len(list(frame_dir.glob("frame_*.jpg")))
                self.logger.info(f"📊 OpenCVで抽出: {extracted}個")
                self.logger.info(f"📊 実際に保存: {saved_frames}個")

                # 🔧 最大値を採用（確実にフレーム数を取得）
                final_extracted = max(extracted, saved_frames)

                # 統計情報の更新
                self.processing_stats["frame_extraction"] = {
                    "total_frames": frame_count,
                    "extracted_frames": final_extracted,  # ← 確実な値
                    "video_fps": fps,
                    "video_duration": duration,
                    "resolution": [width, height],
                    "extraction_interval": interval
                }

                self.logger.info(f"✅ フレーム抽出完了: {final_extracted}フレーム")

                if final_extracted == 0:
                    self.logger.error("❌ フレーム抽出に失敗しました")
                    return {"success": False, "error": "フレーム抽出に失敗しました"}

                # 🔧 確実にextracted_framesを返す
                return {
                    "success": True, 
                    "extracted_frames": final_extracted,  # ← 重要：この値が0になってはいけない
                    "video_info": self.processing_stats["frame_extraction"]
                }

            except Exception as e:
                self.logger.error(f"❌ フレーム抽出エラー: {e}")
                return {"success": False, "error": str(e)}
        
        def run_detection_tracking(self, frame_dir, video_name):
            """基本検出・追跡処理（安定化版）"""
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
        
                # 結果CSV作成
                output_dir = Path("outputs/temp") / video_name
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
        
        # Line 845付近の create_visualizations メソッドを完全置換:

        # Line 874付近のcreate_visualizationsメソッドを以下で完全置換:

        def create_visualizations(self, detection_results, vis_dir):
            """基本可視化（日時付きディレクトリ対応版）"""
            self.logger.info(f"📈 基本可視化生成: {vis_dir}")

            # 🔧 必ず戻り値を返すようにする（初期化）
            result = {
                "success": False,
                "error": "初期化エラー",
                "basic_stats_file": None,
                "graphs_generated": 0,
                "total_files": 0
            }

            try:
                from pathlib import Path
                import json
                from datetime import datetime

                # 🔧 修正: vis_dir がすでにタイムスタンプ付きの場合はそのまま使用
                vis_path = Path(str(vis_dir))
        
                # 🔧 追加: タイムスタンプが含まれていない場合のみ追加
                if not any(char.isdigit() for char in vis_path.name[-15:]):  # 末尾15文字に数字が含まれているかチェック
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vis_path = vis_path.parent / f"{vis_path.name}_{timestamp}"
        
                vis_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 日時付き可視化ディレクトリ作成: {vis_path}")

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

                # 戻り値更新（重要！）
                result.update({
                    "success": True,
                    "error": None,
                    "basic_stats_file": str(stats_file),
                    "total_files": 1,
                    "graphs_generated": 0
                })

                # 統計グラフ生成（既存のコード）
                graphs_generated = 0

                try:
                    # matplotlib/pandas のインポート
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import pandas as pd
    
                    # 簡易フォント設定
                    try:
                        plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
                    except:
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
                                    self.logger.info(f"✅ クラス分布グラフ生成: {class_path}")
                                except Exception as e:
                                    self.logger.error(f"❌ クラス分布グラフエラー: {e}")
        
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
                    "total_files": total_files
                })

                self.logger.info(f"🎨 可視化生成完了: 基本統計1個 + グラフ{graphs_generated}個 = 合計{total_files}個")

                # 🔧 必ず辞書を返す（確実性のため）
                return result

            except Exception as e:
                self.logger.error(f"❌ 可視化生成全体エラー: {e}", exc_info=True)
                # 🔧 エラー時も辞書を返す
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


    def run_baseline_analysis(self, video_path: str) -> Dict[str, Any]:
        """
        ベースライン分析実行（例外時も必ず辞書型で返す完全修正版）

        Args:
            video_path: 分析対象動画のパス

        Returns:
            分析結果辞書（必ずdict型、"success": True/False を含む）
        """
        try:
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("outputs/baseline") / f"{video_name}_{timestamp}"
                frame_dir = output_dir / "frames"
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 日時付き出力ディレクトリ: {output_dir}")

                try:
                    start_time = time.time()
                    detection_result = None
                    evaluation_result = None
                    vis_result = None
                    final_frame_count = 0

                    # Step 1: フレーム抽出
                    self.logger.info("📸 Step 1: フレーム抽出開始")
                    frame_result = self.processor.extract_frames(video_path, frame_dir)
                    if not frame_result.get("success", False):
                        error_msg = f"フレーム抽出失敗: {frame_result.get('error', '不明なエラー')}"
                        self.error_collector.append(error_msg)
                        self.logger.error(f"❌ {error_msg}")
                        raise VideoProcessingError(error_msg)

                    api_extracted_frames = frame_result.get("extracted_frames", 0)
                    frame_files_jpg = list(frame_dir.glob("frame_*.jpg"))
                    frame_files_jpeg = list(frame_dir.glob("frame_*.jpeg"))
                    frame_files_png = list(frame_dir.glob("frame_*.png"))
                    all_frame_files = [f for f in frame_dir.glob("frame_*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                    actual_frame_count = len(all_frame_files)
                    self.logger.info(f"検出されたフレームファイル: {[f.name for f in all_frame_files]}")
                    stats_frames = 0
                    if hasattr(self.processor, 'processing_stats') and self.processor.processing_stats:
                        frame_extraction_stats = self.processor.processing_stats.get("frame_extraction", {})
                        stats_frames = frame_extraction_stats.get("extracted_frames", 0)
                    frame_counts = [api_extracted_frames,                     actual_frame_count, stats_frames]
                    valid_counts = [count for count in frame_counts if count > 0]
                    if valid_counts:
                        final_frame_count = max(valid_counts)
                        self.logger.info(f"📊 フレーム数確定: {final_frame_count}個（候補: {frame_counts}）")
                    else:
                        self.logger.error("❌ 全ての方法でフレーム数が0です")
                        raise VideoProcessingError("フレーム抽出に失敗しました")
                    self.logger.info(f"✅ Step 1完了: {final_frame_count}フレーム抽出")

                    # Step 2: 検出・追跡処理
                    self.logger.info("🎯 Step 2: YOLOポーズモデル確実使用処理開始")
                    models_config = self.config.get('models', {}) if hasattr(self.config, 'get') else {}
                    pose_model_path = models_config.get('pose', 'models/yolo/yolo11x-pose.pt')
                    if not Path(pose_model_path).exists():
                        self.logger.error(f"🚨 ポーズモデルが存在しません: {pose_model_path}")
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
                            raise VideoProcessingError("ポーズモデルが見つかりません")
                    try:
                        if YOLOPOSE_ANALYZER_AVAILABLE:
                            enhanced_config = {
                                "models": {"pose": pose_model_path},
                                "processing": {
                                    "confidence_threshold": 0.3,
                                    "save_keypoints": True,
                                    "keypoint_format": "coco",
                                    "force_keypoint_detection": True
                                },
                                "tracking": {"tracker_type": "bytetrack"},
                                "output": {"save_csv": True, "csv_include_keypoints": True},
                                "inference": {"task": "pose"}
                            }
                            self.logger.info("🚀 確実キーポイント検出を実行")
                            detection_result = analyze_frames_with_tracking_enhanced(
                                frame_dir=str(frame_dir),
                                result_dir=str(output_dir),
                                model_path=pose_model_path,
                                config=enhanced_config,
                                force_exact_model=True
                            )
                            processing_type = "キーポイント統合"
                        else:
                            raise ImportError("yolopose_analyzer が利用できません")
                    except ImportError as e:
                        self.logger.error(f"❌ yolopose_analyzer インポートエラー: {e}")
                        self.logger.warning("🔄 BasicVideoProcessor にフォールバック")
                        if self.depth_enabled and hasattr(self.processor, 'run_detection_tracking_with_depth'):
                            detection_result = self.processor.run_detection_tracking_with_depth(frame_dir, video_name)
                        else:
                            detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
                        processing_type = "基本検出（フォールバック）"
                    if not detection_result.get("success", False):
                        error_msg = detection_result.get("error", "不明なエラー")
                        self.logger.error(f"❌ {processing_type}処理エラー: {error_msg}")
                        self.error_collector.append(f"{processing_type}処理失敗: {error_msg}")
                        raise VideoProcessingError(error_msg)
                    self.logger.info(f"✅ Step 2完了: {processing_type}処理")

                    # Step 2.5: 4点キーポイント処理
                    try:
                        original_csv = detection_result["data"]["csv_path"]
                        four_point_result = self.filter_4point_keypoints(original_csv, output_dir)
                        if four_point_result.get("success", False):
                            self.logger.info(f"✅ 4点フィルタリング成功: {four_point_result['valid_detections']}/{four_point_result['total_detections']}")
                            metrics_csv = four_point_result.get("metrics_csv_path")
                            if metrics_csv and Path(metrics_csv).exists():
                                self.logger.info(f"📁 メトリクスCSV確認: {metrics_csv}")
                                try:
                                    six_point_vis_result = self.create_6point_visualization(metrics_csv, output_dir)
                                    if six_point_vis_result.get("success", False):
                                        saved_frames = six_point_vis_result.get('frames_saved', 0)
                                        total_det = six_point_vis_result.get('total_detections', 0)
                                        success_rate = six_point_vis_result.get('frame_success_rate', 0)
                                        self.logger.info(f"✅ 6点可視化完了:")
                                        self.logger.info(f"  - 保存フレーム: {saved_frames}")
                                        self.logger.info(f"  - 総検出数: {total_det}")
                                        self.logger.info(f"  - 成功率: {success_rate:.1%}")
                                        four_point_result["six_point_visualization"] = six_point_vis_result
                                        vis_output_dir = six_point_vis_result.get('output_dir')
                                        if vis_output_dir:
                                            self.logger.info(f"📁 6点可視化ディレクトリ: {vis_output_dir}")
                                    else:
                                        error_msg = six_point_vis_result.get('error', '不明なエラー')
                                        self.logger.warning(f"⚠️ 6点可視化失敗: {error_msg}")
                                        four_point_result["six_point_visualization"] = six_point_vis_result
                                except Exception as vis_error:
                                    self.logger.error(f"❌ 6点可視化呼び出しエラー: {vis_error}")
                                    import traceback
                                    self.logger.error(f"🔧 詳細: {traceback.format_exc()}")
                                    four_point_result["six_point_visualization"] = {
                                        "success": False,
                                        "error": str(vis_error),
                                        "call_error": True
                                    }
                                try:
                                    four_point_vis_result = self.create_4point_visualization(metrics_csv, None, output_dir)
                                    if four_point_vis_result.get("success", False):
                                        saved_4pt = four_point_vis_result.get('frames_saved', 0)
                                        self.logger.info(f"✅ 4点可視化も完了: {saved_4pt}フレーム")
                                        four_point_result["four_point_visualization"] = four_point_vis_result
                                    else:
                                        self.logger.warning(f"⚠️ 4点可視化失敗: {four_point_vis_result.get('error', '不明')}")
                                except Exception as vis4_error:
                                    self.logger.warning(f"⚠️ 4点可視化エラー: {vis4_error}")
                            else:
                                self.logger.error(f"❌ メトリクスCSVが見つかりません: {metrics_csv}")
                                self.logger.error(f"🔧 4点結果の内容: {four_point_result}")
                                self.logger.debug("🔧 4点結果のキー:")
                            for key, value in four_point_result.items():
                                    self.logger.debug(f"  {key}: {value}")
                            detection_result["data"]["four_point_analysis"] = four_point_result
                            self.logger.info("✅ Step 2.5完了: 4点キーポイント処理成功")
                        else:
                            error_msg = four_point_result.get("error", "不明なエラー")
                            self.logger.error(f"❌ 4点フィルタリング失敗: {error_msg}")
                            detection_result["data"]["four_point_analysis"] = four_point_result
                            self.error_collector.append(f"4点キーポイント処理失敗: {error_msg}")
                    except Exception as e:
                        self.logger.error(f"❌ Step 2.5エラー: 4点キーポイント処理失敗: {e}")
                        four_point_result = {"success": False, "error": str(e)}
                        detection_result["data"]["four_point_analysis"] = four_point_result
                        self.error_collector.append(f"4点キーポイント処理エラー: {e}")

                    # Step 3: 包括的評価
                    self.logger.info("📊 Step 3: 包括的評価開始")
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
                        evaluation_result = ResponseBuilder.success(data={
                            "basic_evaluation": True, 
                            "fallback": True,
                            "evaluator_type": type(self.evaluator).__name__
                        })
                    self.logger.info("✅ Step 3完了: 包括的評価")

                    # Step 4: 基本可視化（スキップ）
                    self.logger.info("📈 Step 4: 基本可視化スキップ（6点可視化のみ使用）")
                    vis_result = {
                        "success": True, 
                        "message": "基本可視化はスキップされました（6点可視化を使用）",
                        "total_files": 0,
                        "graphs_generated": 0,
                        "skipped": True,
                        "reason": "6点可視化が優先されるため基本可視化を無効化"
                    }
                    self.logger.info("✅ Step 4完了: 基本可視化スキップ（6点可視化のみ使用）")

                    # 最終結果の構築
                    final_result = {
                        "video_name": video_name,
                        "processing_time": time.time() - start_time,
                        "detection_result": detection_result,
                        "evaluation_result": evaluation_result,
                        "visualization_result": vis_result,
                        "output_directory": str(output_dir),
                        "timestamp": timestamp,
                        "depth_enabled": self.depth_enabled,
                        "error_count": len(self.error_collector),
                        "errors": self.error_collector.copy()
                    }
                    return ResponseBuilder.success(data=final_result)

                except VideoProcessingError as step_error:
                    self.logger.error(f"❌ Step処理中エラー: {step_error}")
                    self.error_collector.append(f"Step処理エラー: {step_error}")
                    raise step_error

                except Exception as step_error:
                    self.logger.error(f"❌ 予期しないStep内エラー: {step_error}")
                    import traceback
                    self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
                    self.error_collector.append(f"予期しないStep内エラー: {step_error}")
                    raise VideoProcessingError(f"処理ステップでエラー: {step_error}")

        except VideoProcessingError as e:
            self.logger.error(f"❌ 動画処理エラー: {e}")
            self.error_collector.append(f"動画処理エラー: {e}")
            return ResponseBuilder.error(e, suggestions=[
                "動画ファイルの形式を確認してください",
                "動画ファイルの破損を確認してください",
                "モデルファイルが正しくダウンロードされているか確認してください"
            ])

        except Exception as e:
            self.logger.error(f"❌ 予期しないエラー: {e}")
            import traceback
            self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
            self.error_collector.append(f"予期しないエラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # 完全置換: Line 2184-2296
    def filter_4point_keypoints(self, csv_path, output_dir):
        """
        🎯 4点キーポイントフィルタリング（完全修正版）
    
        Args:
            csv_path: 入力CSVファイルパス
            output_dir: 出力ディレクトリ
        
        Returns:
            dict: 統一された戻り値形式
        """
        try:
            self.logger.info("🎯 4点キーポイントフィルタリング開始")
            self.logger.info(f"📂 入力CSV: {csv_path}")
            self.logger.info(f"📁 出力ディレクトリ: {output_dir}")
        
            # 🔧 入力検証
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
            # 🔧 出力ディレクトリ確認
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
            # CSVデータ読み込み
            df = pd.read_csv(csv_path)
            original_rows = len(df)
            self.logger.info(f"📊 元データ: {original_rows}行, {len(df.columns)}列")
            self.logger.debug(f"📋 列名: {list(df.columns)}")
        
            # 🎯 4点キーポイントの確認
            target_keypoints = {
                'left_ear': 3,      # COCO: 3番
                'right_ear': 4,     # COCO: 4番  
                'left_shoulder': 5, # COCO: 5番
                'right_shoulder': 6 # COCO: 6番
            }
        
            available_keypoints = {}
            missing_keypoints = []
        
            for kpt_name, coco_idx in target_keypoints.items():
                x_col = f"{kpt_name}_x"
                y_col = f"{kpt_name}_y"
                conf_col = f"{kpt_name}_conf"
            
                if all(col in df.columns for col in [x_col, y_col, conf_col]):
                    available_keypoints[kpt_name] = {
                        'x': x_col, 'y': y_col, 'conf': conf_col, 'coco_idx': coco_idx
                    }
                    self.logger.debug(f"✅ キーポイント利用可能: {kpt_name}")
                else:
                    missing_keypoints.append(kpt_name)
                    missing_cols = [col for col in [x_col, y_col, conf_col] if col not in df.columns]
                    self.logger.warning(f"⚠️ キーポイント不足: {kpt_name} - 欠損列: {missing_cols}")
            
            # 🔧 最小要件チェック
            if len(available_keypoints) < 2:
                raise ValueError(f"4点フィルタリングに必要なキーポイントが不足: 利用可能{len(available_keypoints)}/4点")
        
            self.logger.info(f"🎯 使用キーポイント: {list(available_keypoints.keys())} ({len(available_keypoints)}/4点)")
        
            # 🎯 フィルタリング処理
            confidence_threshold = 0.3
            if hasattr(self, 'config') and self.config:
                confidence_threshold = self.config.get('processing', {}).get('keypoint_confidence_threshold', 0.3)
        
            self.logger.info(f"🎯 信頼度閾値: {confidence_threshold}")
        
            filtered_data = []
            valid_detections = 0
        
            for idx, row in df.iterrows():
                # 基本情報の保持
                filtered_row = {}
            
                # 基本列のコピー
                basic_columns = ['frame', 'person_id', 'x1', 'y1', 'x2', 'y2', 'conf', 'class_name']
                for col in basic_columns:
                    if col in df.columns:
                        filtered_row[col] = row[col]
            
                # 4点キーポイント情報の追加
                valid_keypoints_count = 0
            
                for kpt_name, kpt_info in available_keypoints.items():
                    x_val = row[kpt_info['x']]
                    y_val = row[kpt_info['y']]
                    conf_val = row[kpt_info['conf']]
                
                    # データ保存
                    filtered_row[f"{kpt_name}_x"] = x_val
                    filtered_row[f"{kpt_name}_y"] = y_val
                    filtered_row[f"{kpt_name}_conf"] = conf_val
                
                    # 有効性チェック
                    if conf_val >= confidence_threshold and x_val > 0 and y_val > 0:
                        valid_keypoints_count += 1
            
                # 不足キーポイントはゼロ埋め
                for missing_kpt in missing_keypoints:
                    filtered_row[f"{missing_kpt}_x"] = 0.0
                    filtered_row[f"{missing_kpt}_y"] = 0.0
                    filtered_row[f"{missing_kpt}_conf"] = 0.0
            
                # 最低要件を満たす場合のみ保持
                min_valid_keypoints = max(1, len(available_keypoints) // 2)
                if valid_keypoints_count >= min_valid_keypoints:
                    filtered_data.append(filtered_row)
                    valid_detections += 1
        
            # 結果検証
            if not filtered_data:
                raise ValueError(f"フィルタリング後のデータが空です。閾値 {confidence_threshold} を確認してください。")
        
            self.logger.info(f"📊 フィルタリング結果: {valid_detections}/{original_rows} ({valid_detections/original_rows*100:.1f}%)")
        
            # 🎯 データフレーム作成
            filtered_df = pd.DataFrame(filtered_data)
        
            # 🎯 メトリクス計算
            self.logger.info("📊 4点メトリクス計算開始")
            metrics_df = self.calculate_4point_metrics(filtered_df.copy())
        
            # 🔧 ファイル保存（統一されたファイル名）
            filtered_csv_path = output_dir / "four_point_keypoints.csv"
            metrics_csv_path = output_dir / "four_point_keypoints_with_metrics.csv"
        
            # CSVファイル保存
            filtered_df.to_csv(filtered_csv_path, index=False)
            metrics_df.to_csv(metrics_csv_path, index=False)
        
            self.logger.info(f"✅ 4点フィルタリング完了")
            self.logger.info(f"📁 フィルタCSV: {filtered_csv_path}")
            self.logger.info(f"📁 メトリクスCSV: {metrics_csv_path}")
        
            # 🔧 修正: 統一された戻り値（呼び出し側の期待に合わせる）
            return {
                "success": True,
                "filtered_csv_path": str(filtered_csv_path),      # ← 統一
                "metrics_csv_path": str(metrics_csv_path),        # ← 重要！この名前
                "original_rows": original_rows,
                "filtered_rows": valid_detections,
                "valid_detections": valid_detections,
                "total_detections": original_rows,
                "filter_rate": valid_detections / original_rows,
                "available_keypoints": list(available_keypoints.keys()),
                "missing_keypoints": missing_keypoints,
                "confidence_threshold": confidence_threshold,
            
                # 追加統計情報
                "statistics": {
                    "filter_success_rate": valid_detections / original_rows,
                    "keypoints_coverage": len(available_keypoints) / 4.0,
                    "avg_valid_keypoints": sum(
                        sum(1 for kpt in available_keypoints.keys() 
                            if row.get(f"{kpt}_conf", 0) >= confidence_threshold)
                        for row in filtered_data
                    ) / len(filtered_data) if filtered_data else 0
                }
            }
        
        except Exception as e:
            self.logger.error(f"❌ 4点キーポイントフィルタリングエラー: {e}")
            import traceback
            self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
        
            # 🔧 エラー時も統一された戻り値
            return {
                "success": False,
                "error": str(e),
                "filtered_csv_path": None,
                "metrics_csv_path": None,       # ← 重要！
                "original_rows": 0,
                "filtered_rows": 0,
                "valid_detections": 0,
                "total_detections": 0,
                "filter_rate": 0.0,
                "available_keypoints": [],
                "missing_keypoints": [],
                "confidence_threshold": 0.3
            }

    def calculate_4point_metrics(self, df):
        """
        4点専用メトリクス計算（中点座標保存確実版）
        """
        import numpy as np
        import pandas as pd
    
        self.logger.info("📊 4点メトリクス計算開始")
    
        # 結果保存用の列を事前に初期化
        metrics_columns = [
            'shoulder_width', 'head_center_x', 'head_center_y', 
            'shoulder_center_x', 'shoulder_center_y', 'pose_angle',
            'pose_completeness', 'pose_confidence', 'head_shoulder_distance'
        ]
    
        for col in metrics_columns:
            if col not in df.columns:
                df[col] = np.nan
    
        # 統計カウンター
        stats = {
            'total_rows': len(df),
            'shoulder_width_calculated': 0,
            'head_center_calculated': 0,
            'shoulder_center_calculated': 0,
            'pose_angle_calculated': 0
        }
    
        # 各行のメトリクス計算
        for idx, row in df.iterrows():
            try:
                # 🎯 中点計算（必ず実行）
                # 頭中点（耳の中点）
                left_ear_x = row.get('left_ear_x')
                left_ear_y = row.get('left_ear_y')
                right_ear_x = row.get('right_ear_x')
                right_ear_y = row.get('right_ear_y')
                left_ear_conf = row.get('left_ear_conf', 0)
                right_ear_conf = row.get('right_ear_conf', 0)
            
                if (pd.notna(left_ear_x) and pd.notna(left_ear_y) and 
                    pd.notna(right_ear_x) and pd.notna(right_ear_y) and
                    left_ear_conf > 0.3 and right_ear_conf > 0.3):
                
                    head_center_x = (left_ear_x + right_ear_x) / 2
                    head_center_y = (left_ear_y + right_ear_y) / 2
                    df.at[idx, 'head_center_x'] = head_center_x
                    df.at[idx, 'head_center_y'] = head_center_y
                    stats['head_center_calculated'] += 1
            
                # 肩中点
                left_shoulder_x = row.get('left_shoulder_x')
                left_shoulder_y = row.get('left_shoulder_y')
                right_shoulder_x = row.get('right_shoulder_x')
                right_shoulder_y = row.get('right_shoulder_y')
                left_shoulder_conf = row.get('left_shoulder_conf', 0)
                right_shoulder_conf = row.get('right_shoulder_conf', 0)
            
                if (pd.notna(left_shoulder_x) and pd.notna(left_shoulder_y) and 
                    pd.notna(right_shoulder_x) and pd.notna(right_shoulder_y) and
                    left_shoulder_conf > 0.3 and right_shoulder_conf > 0.3):
                
                    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
                    shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
                    df.at[idx, 'shoulder_center_x'] = shoulder_center_x
                    df.at[idx, 'shoulder_center_y'] = shoulder_center_y
                    stats['shoulder_center_calculated'] += 1
                
                    # 肩幅計算
                    shoulder_width = np.sqrt((right_shoulder_x - left_shoulder_x)**2 + 
                                        (right_shoulder_y - left_shoulder_y)**2)
                    df.at[idx, 'shoulder_width'] = shoulder_width
                    stats['shoulder_width_calculated'] += 1
                
                    # 姿勢角度計算
                    angle = np.arctan2(right_shoulder_y - left_shoulder_y, 
                                    right_shoulder_x - left_shoulder_x)
                    angle_degrees = np.degrees(angle)
                    df.at[idx, 'pose_angle'] = angle_degrees
                    stats['pose_angle_calculated'] += 1
            
                # 頭-肩の距離
                if (pd.notna(df.at[idx, 'head_center_x']) and 
                    pd.notna(df.at[idx, 'shoulder_center_x'])):
                    head_shoulder_dist = np.sqrt(
                        (df.at[idx, 'head_center_x'] - df.at[idx, 'shoulder_center_x'])**2 +
                        (df.at[idx, 'head_center_y'] - df.at[idx, 'shoulder_center_y'])**2
                    )
                    df.at[idx, 'head_shoulder_distance'] = head_shoulder_dist
            
                # 完全性スコア計算
                valid_keypoints = sum([
                    1 for point in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
                    if row.get(f'{point}_conf', 0) > 0.3
                ])
                df.at[idx, 'pose_completeness'] = valid_keypoints / 4.0
            
                # ポーズ信頼度（平均）
                confidences = [row.get(f'{point}_conf', 0) 
                            for point in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
                            if pd.notna(row.get(f'{point}_conf'))]
                if confidences:
                    df.at[idx, 'pose_confidence'] = np.mean(confidences)
            
            except Exception as e:
                self.logger.warning(f"⚠️ 行 {idx} でメトリクス計算エラー: {e}")
                continue
    
        # 統計出力
        self.logger.info("📊 メトリクス計算完了:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
    
        # 🔧 中点座標の存在確認
        head_center_count = df['head_center_x'].notna().sum()
        shoulder_center_count = df['shoulder_center_x'].notna().sum()
    
        self.logger.info(f"✅ 中点座標確認:")
        self.logger.info(f"  頭中点座標数: {head_center_count}")
        self.logger.info(f"  肩中点座標数: {shoulder_center_count}")
    
        # 統計サマリー
        if stats['shoulder_width_calculated'] > 0:
            valid_widths = df['shoulder_width'].dropna()
            avg_width = valid_widths.mean()
            self.logger.info(f"📊 メトリクス統計:")
            self.logger.info(f"  平均肩幅: {avg_width:.1f}px")
        
            if stats['pose_completeness'] > 0:
                avg_completeness = df['pose_completeness'].mean()
                avg_confidence = df['pose_confidence'].mean()
                self.logger.info(f"  平均完全性: {avg_completeness:.2f}")
                self.logger.info(f"  平均ポーズ信頼度: {avg_confidence:.2f}")
    
        return df

    def create_4point_visualization(self, csv_path, video_path, output_dir):
        """4点キーポイント専用可視化生成（日時付きフォルダ対応版）"""
        try:
            import cv2
            import pandas as pd
            from pathlib import Path
            from datetime import datetime

            self.logger.info("🎨 4点可視化生成開始（日時付きフォルダ対応）")

            # 🔧 日時付き可視化ディレクトリ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = Path(output_dir) / f"visualized_frames_4points_{timestamp}"
            vis_dir.mkdir(exist_ok=True)

            self.logger.info(f"📁 4点可視化ディレクトリ: {vis_dir}")

            # CSV読み込み
            df = pd.read_csv(csv_path)

            if df.empty:
                self.logger.warning("⚠️ 4点CSVデータが空です")
                return {"success": False, "error": "Empty CSV data"}

            self.logger.info(f"📋 CSV列名: {df.columns.tolist()}")
            self.logger.info(f"📋 CSVデータ形状: {df.shape}")

            # フレームディレクトリの確認
            frames_dir = Path(output_dir) / "frames"
            if not frames_dir.exists():
                self.logger.error(f"❌ フレームディレクトリが存在しません: {frames_dir}")
                return {"success": False, "error": "Frames directory not found"}

            frame_files = sorted(frames_dir.glob("*.jpg"))
            if not frame_files:
                self.logger.error("❌ フレームファイルが見つかりません")
                return {"success": False, "error": "No frame files found"}

            self.logger.info(f"📁 フレームファイル数: {len(frame_files)}")
            self.logger.info(f"📁 フレームファイル例: {[f.name for f in frame_files[:3]]}")

            # フレーム番号の対応テーブル作成
            frame_mapping = {}
            for i, frame_file in enumerate(frame_files):
                frame_num_from_file = i
                frame_identifier = frame_file.name
                frame_mapping[frame_identifier] = frame_num_from_file
                frame_mapping[frame_num_from_file] = frame_identifier

            self.logger.info(f"📋 フレーム対応例: {list(frame_mapping.items())[:5]}")

            # キーポイント列の確認
            keypoint_columns = {
                'left_ear': {'x': 'left_ear_x', 'y': 'left_ear_y', 'conf': 'left_ear_conf'},
                'right_ear': {'x': 'right_ear_x', 'y': 'right_ear_y', 'conf': 'right_ear_conf'},
                'left_shoulder': {'x': 'left_shoulder_x', 'y': 'left_shoulder_y', 'conf': 'left_shoulder_conf'},
                'right_shoulder': {'x': 'right_shoulder_x', 'y': 'right_shoulder_y', 'conf': 'right_shoulder_conf'}
            }

            # 列の存在確認
            missing_columns = []
            for kpt_name, cols in keypoint_columns.items():
                for col_type, col_name in cols.items():
                    if col_name not in df.columns:
                        missing_columns.append(col_name)

            if missing_columns:
                self.logger.warning(f"⚠️ 不足列: {missing_columns}")

            saved_count = 0
            total_detections = 0
            processed_frames = 0
            debug_info = []

            # 各フレームに対する処理
            for frame_file in frame_files:
                processed_frames += 1
                frame_identifier = frame_file.name
    
                # 複数の方法でCSVデータを検索
                frame_data = None
    
                # 方法1: 完全なファイル名でマッチ
                frame_data = df[df['frame'] == frame_identifier]
    
                # 方法2: フレーム番号でマッチ（0から始まる連番）
                if frame_data.empty:
                    frame_index = processed_frames - 1
                    numeric_frame_data = df[df['frame'] == frame_index]
                    if not numeric_frame_data.empty:
                        frame_data = numeric_frame_data
    
                # 方法3: インデックス順序でマッチ
                if frame_data.empty and processed_frames <= len(df):
                    frame_data = df.iloc[[processed_frames - 1]]
    
                if not frame_data.empty:
                    # フレーム画像読み込み
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        self.logger.warning(f"⚠️ フレーム読み込み失敗: {frame_file}")
                        continue
        
                    frame_height, frame_width = frame.shape[:2]
                    temp_frame = frame.copy()
                    frame_detections = 0
        
                    for idx, row in frame_data.iterrows():
                        # キーポイントデータの抽出と検証
                        keypoints = {}
                        valid_keypoint_count = 0
            
                        for kpt_name, cols in keypoint_columns.items():
                            try:
                                x = float(row.get(cols['x'], 0))
                                y = float(row.get(cols['y'], 0))
                                conf = float(row.get(cols['conf'], 1.0))
                    
                                # 座標の有効性チェック（緩い条件）
                                if (0 <= x <= frame_width and 
                                   0 <= y <= frame_height and 
                                    conf > 0.1):  # 信頼度閾値を0.3から0.1に緩和
                                    keypoints[kpt_name] = (int(x), int(y), conf)
                                    valid_keypoint_count += 1
                            except (ValueError, TypeError) as e:
                                continue
            
                        # デバッグ情報記録
                        if processed_frames <= 3:  # 最初の3フレームのデバッグ
                            debug_info.append({
                                'frame': frame_identifier,
                                'valid_keypoints': valid_keypoint_count,
                                'keypoints': keypoints,
                                'row_data': {k: row.get(k) for k in ['left_ear_x', 'left_ear_y', 'left_ear_conf']}
                            })
            
                        # 1点でも有効なキーポイントがあれば描画
                        if valid_keypoint_count >= 1:  # 4から1に条件緩和
                            temp_frame = self.draw_4point_keypoints_robust(temp_frame, keypoints, row)
                            frame_detections += 1
        
                    # 1つでも検出があれば保存
                    if frame_detections > 0:
                        output_filename = f"4pt_{frame_file.name}"
                        output_path = vis_dir / output_filename
                        success = cv2.imwrite(str(output_path), temp_frame)
            
                        if success:
                            saved_count += 1
                            total_detections += frame_detections
                
                            # 最初の5枚の保存成功をログ
                            if saved_count <= 5:
                                self.logger.info(f"✅ 4点画像保存成功: {output_filename} (検出: {frame_detections})")
                        else:
                            self.logger.warning(f"❌ 画像保存失敗: {output_path}")
    
                # 進捗表示（頻度を下げる）
                if processed_frames % 100 == 0:
                    self.logger.info(f"🎨 4点可視化進捗: {processed_frames}フレーム (保存済み: {saved_count})")

            # デバッグ情報出力
            if debug_info:
                self.logger.info("🔧 デバッグ情報（最初の3フレーム）:")
                for info in debug_info:
                    self.logger.info(f"  フレーム: {info['frame']}, 有効キーポイント: {info['valid_keypoints']}")
                    self.logger.info(f"  サンプルデータ: {info['row_data']}")

            self.logger.info(f"✅ 4点可視化完了: {saved_count}フレーム保存 (検出数: {total_detections})")
            self.logger.info(f"📊 処理統計: {processed_frames}フレーム処理, 成功率: {(saved_count/processed_frames)*100:.1f}%")
    
            return {
                "success": True, 
                "frames_saved": saved_count, 
                "total_detections": total_detections,
                "processed_frames": processed_frames,
                "output_dir": str(vis_dir),
                "timestamp": timestamp,  # 🔧 追加: タイムスタンプ
                "debug_info": debug_info
                }
    
        except Exception as e:
            self.logger.error(f"❌ 4点可視化エラー: {e}")
            import traceback
            self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def create_6point_visualization(self, csv_path, output_dir):
        """
        6点キーポイント専用可視化生成（引数修正・フレーム検索強化版）
    
        Args:
            csv_path: メトリクス付きCSVファイルのパス
            output_dir: 出力ディレクトリのパス
        
        Returns:
            dict: 可視化結果
        """
        try:
            import cv2
            import pandas as pd
            from pathlib import Path
            from datetime import datetime

            self.logger.info("🎨 6点可視化生成開始（引数修正版）")
        
            # 🔧 入力検証の強化
            csv_path = Path(csv_path)
            output_dir = Path(output_dir)
        
            if not csv_path.exists():
                raise FileNotFoundError(f"メトリクスCSVファイルが見つかりません: {csv_path}")
            
            self.logger.info(f"📂 入力CSV: {csv_path}")
            self.logger.info(f"📁 出力ディレクトリ: {output_dir}")

            # 日時付き可視化ディレクトリ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = output_dir / "6point_visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"📁 6点可視化ディレクトリ: {vis_dir}")

            # CSV読み込み（メトリクス付きデータ）
            df = pd.read_csv(csv_path)
        
            if df.empty:
                self.logger.warning("⚠️ メトリクスCSVデータが空です")
                return {"success": False, "error": "Empty metrics CSV data"}

            # 🔧 フレームディレクトリの複数候補検索
            frame_search_dirs = [
                output_dir / "frames",           # 標準パス
                output_dir.parent / "frames",    # 親ディレクトリ
                Path(output_dir).resolve() / "frames",  # 絶対パス
                Path(csv_path).parent / "frames", # CSVと同階層
            ]
        
            frame_dir = None
            frame_files = []
        
            for search_dir in frame_search_dirs:
                if search_dir.exists():
                    potential_frames = sorted(search_dir.glob("*.jpg"))
                    if potential_frames:
                        frame_dir = search_dir
                        frame_files = potential_frames
                        self.logger.info(f"✅ フレームディレクトリ発見: {frame_dir}")
                        break
                    else:
                        self.logger.debug(f"🔍 フレームなし: {search_dir}")
                else:
                    self.logger.debug(f"🔍 ディレクトリなし: {search_dir}")

            if not frame_dir or not frame_files:
                error_msg = f"フレームファイルが見つかりません。検索パス: {[str(d) for d in frame_search_dirs]}"
                self.logger.error(f"❌ {error_msg}")
                return {"success": False, "error": error_msg}

            self.logger.info(f"📁 使用フレームディレクトリ: {frame_dir}")
            self.logger.info(f"📁 フレームファイル数: {len(frame_files)}")

            # 🔧 必要な列の確認（より柔軟）
            required_basic_cols = [
                'left_ear_x', 'left_ear_y', 'left_ear_conf',
                'right_ear_x', 'right_ear_y', 'right_ear_conf',
                'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_conf',
                'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_conf'
            ]
        
            center_cols = ['head_center_x', 'head_center_y', 'shoulder_center_x', 'shoulder_center_y']
        
            missing_basic = [col for col in required_basic_cols if col not in df.columns]
            missing_centers = [col for col in center_cols if col not in df.columns]
        
            if missing_basic:
                self.logger.error(f"❌ 基本キーポイント列が不足: {missing_basic}")
                return {"success": False, "error": f"Missing basic columns: {missing_basic}"}
            
            if missing_centers:
                self.logger.warning(f"⚠️ 中点列が不足（動的計算します）: {missing_centers}")

            self.logger.info(f"📋 CSVデータ形状: {df.shape}")
            self.logger.info(f"📋 利用可能な基本列: {len(required_basic_cols) - len(missing_basic)}/12")
            self.logger.info(f"📋 利用可能な中点列: {len(center_cols) - len(missing_centers)}/4")

            # 🔧 フレームマッピングの改善
            frame_mapping = {}
        
            # 方法1: ファイル名ベース
            for i, frame_file in enumerate(frame_files):
                frame_name = frame_file.name
                frame_mapping[frame_name] = i
                frame_mapping[i] = frame_name
            
                # 拡張子なしの名前も追加
                frame_stem = frame_file.stem
                frame_mapping[frame_stem] = i

            self.logger.debug(f"🔧 フレームマッピング例: {list(frame_mapping.items())[:10]}")

            # 処理統計
            processed_frames = 0
            saved_count = 0
            total_detections = 0
            skipped_no_data = 0
            debug_info = []

            # 🔧 フレーム別処理（改良版）
            for frame_file in frame_files:
                processed_frames += 1
                frame_identifier = frame_file.name
                frame_stem = frame_file.stem

                # 🔧 複数の方法でCSVデータを検索
                frame_data = pd.DataFrame()
            
                # 方法1: 完全なファイル名でマッチ
                if 'frame' in df.columns:
                    frame_data = df[df['frame'] == frame_identifier]
                    if frame_data.empty:
                        frame_data = df[df['frame'] == frame_stem]
            
                # 方法2: フレーム番号でマッチ（0から始まる連番）
                if frame_data.empty:
                    frame_index = processed_frames - 1
                    if 'frame' in df.columns:
                        frame_data = df[df['frame'] == frame_index]
                    
                    # フレーム番号（1から始まる）も試行
                    if frame_data.empty:
                        frame_data = df[df['frame'] == processed_frames]
            
                # 方法3: インデックス順序でマッチ（最後の手段）
                if frame_data.empty and processed_frames <= len(df):
                    try:
                        frame_data = df.iloc[[processed_frames - 1]]
                    except IndexError:
                        pass

                if frame_data.empty:
                    skipped_no_data += 1
                    continue

                # フレーム読み込み
                temp_frame = cv2.imread(str(frame_file))
                if temp_frame is None:
                    self.logger.warning(f"⚠️ フレーム読み込み失敗: {frame_file}")
                    continue

                frame_detections = 0
            
                # 🔧 各人物の6点描画
                for idx, row in frame_data.iterrows():
                    # キーポイント抽出
                    keypoints = {}
                    for point in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']:
                        for coord in ['x', 'y', 'conf']:
                            col_name = f"{point}_{coord}"
                            if col_name in df.columns and col_name in row.index:
                                keypoints[col_name] = row[col_name]

                    # 有効キーポイント数チェック
                    valid_count = sum(1 for point in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
                                    if keypoints.get(f"{point}_conf", 0) > 0.3)

                    if valid_count >= 2:  # 最低2点あれば描画
                        # 🎯 6点描画（4点 + 2中点）
                        temp_frame = self.draw_6point_keypoints_with_centers(temp_frame, keypoints, row)
                        frame_detections += 1

                    # デバッグ情報（最初の3フレーム）
                    if processed_frames <= 3:
                        debug_info.append({
                            'frame_file': frame_identifier,
                            'frame_index': processed_frames - 1,
                            'csv_frame_value': row.get('frame', 'N/A'),
                            'person_id': row.get('person_id', 'unknown'),
                            'valid_keypoints': valid_count,
                            'has_head_center': 'head_center_x' in row and pd.notna(row.get('head_center_x')),
                            'has_shoulder_center': 'shoulder_center_x' in row and pd.notna(row.get('shoulder_center_x')),
                            'matching_method': 'successful'
                        })

                # フレーム保存
                if frame_detections > 0:
                    output_filename = f"6point_frame_{processed_frames:06d}.jpg"
                    output_path = vis_dir / output_filename
                    success = cv2.imwrite(str(output_path), temp_frame)

                    if success:
                        saved_count += 1
                        total_detections += frame_detections
                    
                        # 最初の5枚の保存成功をログ
                        if saved_count <= 5:
                            self.logger.info(f"✅ 6点画像保存成功: {output_filename} (検出: {frame_detections})")
                    else:
                        self.logger.warning(f"❌ 画像保存失敗: {output_path}")

                # 進捗表示（100フレームごと）
                if processed_frames % 100 == 0:
                    self.logger.info(f"🎨 6点可視化進捗: {processed_frames}/{len(frame_files)} (保存済み: {saved_count})")

            # デバッグ情報出力
            if debug_info:
                self.logger.info("🔧 フレームマッチングデバッグ情報（最初の3フレーム）:")
                for info in debug_info[:9]:  # 最大9個
                    self.logger.info(f"  フレーム: {info['frame_file']} -> CSV: {info['csv_frame_value']}")
                    self.logger.info(f"  人物ID: {info['person_id']}, 有効キーポイント: {info['valid_keypoints']}")
                    self.logger.info(f"  頭中点: {info['has_head_center']}, 肩中点: {info['has_shoulder_center']}")

            self.logger.info(f"✅ 6点可視化完了: {saved_count}フレーム保存 (検出数: {total_detections})")
            self.logger.info(f"📊 処理統計:")
            self.logger.info(f"  - 総フレーム処理: {processed_frames}")
            self.logger.info(f"  - 画像保存成功: {saved_count}")
            self.logger.info(f"  - データなしスキップ: {skipped_no_data}")
            self.logger.info(f"  - 成功率: {(saved_count/processed_frames)*100:.1f}%")

            return {
                "success": True,
                "frames_saved": saved_count,
                "total_detections": total_detections,
                "processed_frames": processed_frames,
                "skipped_frames": skipped_no_data,
                "output_dir": str(vis_dir),
                "timestamp": timestamp,
                "debug_info": debug_info,
                "frame_success_rate": saved_count / processed_frames if processed_frames > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"❌ 6点可視化エラー: {e}")
            import traceback
            self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
            return {
                "success": False, 
                "error": str(e),
                "frames_saved": 0,
                "total_detections": 0,
                "processed_frames": 0
            }

    def draw_4point_keypoints_robust(self, frame, keypoints, row):
        """4点キーポイント描画（検出枠＋ID表示付き、文字ラベルなし）"""
        try:
            import cv2

            # 🎨 シンプル2色設定
            ear_color = (100, 180, 100)         # 落ち着いたグリーン（耳）
            shoulder_color = (100, 100, 180)    # 落ち着いたレッド（肩）
            center_color = (255, 200, 0)        # ゴールド（中点）
            line_color = (0, 255, 255)          # シアン（接続線）
        
            # 描画設定
            point_radius = 5         # キーポイントのサイズ
            center_radius = 8        # 中点のサイズ（少し大きく）
            outer_radius = 7         # 白い外枠
            line_thickness = 2       # 接続線の太さ
        
            drawn_points = 0

            # 🔲 検出枠の描画
            try:
                if hasattr(row, 'get'):
                    x1 = int(row.get('x1', 0))
                    y1 = int(row.get('y1', 0))
                    x2 = int(row.get('x2', 0))
                    y2 = int(row.get('y2', 0))
                    person_id = row.get('person_id', '?')
                    conf = float(row.get('conf', 0))
                
                    if x1 > 0 and y1 > 0 and x2 > x1 and y2 > y1:
                        # 検出枠の描画（緑色）
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                        # 🏷️ ID＋信頼度表示（背景付き）
                        id_text = f"ID:{person_id} ({conf:.2f})"
                        text_size = 0.6
                        text_thickness = 1
                    
                        # テキストサイズ計算
                        (text_w, text_h), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
                    
                        # 背景矩形
                        bg_x1 = x1
                        bg_y1 = y1 - text_h - 10
                        bg_x2 = x1 + text_w + 10
                        bg_y2 = y1
                    
                        # 背景描画（半透明黒）
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                        # テキスト描画（白）
                        cv2.putText(frame, id_text, (x1 + 5, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness)
            except Exception as e:
                self.logger.debug(f"検出枠描画エラー: {e}")

            # 🎯 各キーポイントの描画
            ear_points = []
            shoulder_points = []

            for kpt_name, (x, y, conf) in keypoints.items():
                # 肩と耳で色分け
                if 'ear' in kpt_name:
                    color = ear_color
                elif 'shoulder' in kpt_name:
                    color = shoulder_color
                else:
                    color = (128, 128, 128)  # デフォルトグレー
            
                try:
                    # メインの点
                    cv2.circle(frame, (x, y), point_radius, color, -1)
                
                    # 白い外枠（見やすさのため）
                    cv2.circle(frame, (x, y), outer_radius, (255, 255, 255), 1)
                
                    drawn_points += 1
                
                except Exception as e:
                    self.logger.debug(f"キーポイント描画スキップ: {kpt_name} - {e}")
                    continue

            # 🔗 接続線の描画
            try:
                # 肩のライン（肩の色で）
                if len(shoulder_points) == 2:
                    cv2.line(frame, shoulder_points[0], shoulder_points[1], 
                            shoulder_color, line_thickness)
            
                # 耳のライン（耳の色で、細め）
                if len(ear_points) == 2:
                    cv2.line(frame, ear_points[0], ear_points[1], 
                            ear_color, 1)  # より細い線
            except:
                pass

            # 🔧 中点の計算と描画（head_center統一版）
            try:
                # 🎯 head_center（両耳中点）の描画
                if len(ear_points) == 2:
                    head_center_x = (ear_points[0][0] + ear_points[1][0]) // 2
                    head_center_y = (ear_points[0][1] + ear_points[1][1]) // 2
                
                    # 中点描画（ゴールド色、やや大きめ）
                    cv2.circle(frame, (head_center_x, head_center_y), center_radius, center_color, -1)
                    cv2.circle(frame, (head_center_x, head_center_y), center_radius + 2, (255, 255, 255), 1)
                
                    # 🔧 ラベル修正: H-C（Head Center）
                    cv2.putText(frame, "H-C", (head_center_x + 10, head_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, center_color, 1)
            
                # 🎯 肩中点の描画
                if len(shoulder_points) == 2:
                    shoulder_center_x = (shoulder_points[0][0] + shoulder_points[1][0]) // 2
                    shoulder_center_y = (shoulder_points[0][1] + shoulder_points[1][1]) // 2
                
                    # 中点描画（ゴールド色、やや大きめ）
                    cv2.circle(frame, (shoulder_center_x, shoulder_center_y), center_radius, center_color, -1)
                    cv2.circle(frame, (shoulder_center_x, shoulder_center_y), center_radius + 2, (255, 255, 255), 1)
                
                    # ラベル描画
                    cv2.putText(frame, "S-C", (shoulder_center_x + 10, shoulder_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, center_color, 1)
                
                # 🔧 head_centerと肩中点を結ぶ線（体軸の可視化）
                if len(ear_points) == 2 and len(shoulder_points) == 2:
                    head_center = ((ear_points[0][0] + ear_points[1][0]) // 2, 
                                  (ear_points[0][1] + ear_points[1][1]) // 2)
                    shoulder_center = ((shoulder_points[0][0] + shoulder_points[1][0]) // 2,
                                     (shoulder_points[0][1] + shoulder_points[1][1]) // 2)
                
                    # 体軸線の描画（破線風）
                    cv2.line(frame, head_center, shoulder_center, line_color, 2)
                
            except Exception as e:
                self.logger.debug(f"中点描画エラー: {e}")

            return frame

        except Exception as e:
            self.logger.error(f"❌ キーポイント描画エラー: {e}")
            return frame
        
    def draw_6point_keypoints_with_centers(self, frame, keypoints, row_data):
        """
        6点キーポイント描画（4点 + 2つの中点）- 安全性強化版
        """
        try:
            import cv2
            import numpy as np
            import pandas as pd
    
            # 🎨 色定義（より見やすく）
            colors = {
                'left_ear': (0, 255, 0),          # 緑色 - 左耳
                'right_ear': (0, 200, 0),         # 濃い緑 - 右耳
                'left_shoulder': (0, 0, 255),     # 赤色 - 左肩
                'right_shoulder': (0, 0, 200),    # 濃い赤 - 右肩
                'head_center': (255, 0, 255),     # マゼンタ - 頭中点
                'shoulder_center': (0, 255, 255), # シアン - 肩中点
                'connection': (255, 255, 255),    # 白色 - 接続線
                'axis': (255, 255, 0)             # 黄色 - 体軸線
            }
    
            # 🔧 キーポイント座標抽出（安全性強化）
            valid_points = {}
            center_points = {}
    
            # 4つの基本キーポイント
            points_config = [
                ('left_ear', 'left_ear'),
                ('right_ear', 'right_ear'), 
                ('left_shoulder', 'left_shoulder'),
                ('right_shoulder', 'right_shoulder')
            ]
    
            for point_name, color_key in points_config:
                try:
                    x = keypoints.get(f"{point_name}_x")
                    y = keypoints.get(f"{point_name}_y") 
                    conf = keypoints.get(f"{point_name}_conf", 0)
            
                    # 🔧 安全な型変換
                    if x is not None and y is not None:
                        x_val = float(x) if not pd.isna(x) else None
                        y_val = float(y) if not pd.isna(y) else None
                        conf_val = float(conf) if not pd.isna(conf) else 0.0
                
                        if x_val is not None and y_val is not None and conf_val > 0.3:
                            # フレーム範囲チェック
                            frame_h, frame_w = frame.shape[:2]
                            if 0 <= x_val <= frame_w and 0 <= y_val <= frame_h:
                                valid_points[point_name] = {
                                    'pos': (int(x_val), int(y_val)),
                                    'conf': conf_val,
                                    'color': colors[color_key]
                                }
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"キーポイント{point_name}変換エラー: {e}")
                    continue
    
            # 🔧 中点計算（row_dataから取得または計算）
            # 頭中点（耳の中点）
            try:
                if hasattr(row_data, 'get'):
                    head_center_x = row_data.get('head_center_x')
                    head_center_y = row_data.get('head_center_y')
                else:
                    head_center_x = getattr(row_data, 'head_center_x', None)
                    head_center_y = getattr(row_data, 'head_center_y', None)
                
                if (head_center_x is not None and head_center_y is not None and 
                    not pd.isna(head_center_x) and not pd.isna(head_center_y)):
                    center_points['head_center'] = (int(float(head_center_x)), int(float(head_center_y)))
                elif 'left_ear' in valid_points and 'right_ear' in valid_points:
                    # 動的計算
                    left_ear = valid_points['left_ear']['pos']
                    right_ear = valid_points['right_ear']['pos']
                    center_points['head_center'] = (
                        int((left_ear[0] + right_ear[0]) / 2),
                        int((left_ear[1] + right_ear[1]) / 2)
                    )
            except Exception as e:
                self.logger.debug(f"頭中点計算エラー: {e}")
    
            # 肩中点
            try:
                if hasattr(row_data, 'get'):
                    shoulder_center_x = row_data.get('shoulder_center_x')
                    shoulder_center_y = row_data.get('shoulder_center_y')
                else:
                    shoulder_center_x = getattr(row_data, 'shoulder_center_x', None)
                    shoulder_center_y = getattr(row_data, 'shoulder_center_y', None)
                
                if (shoulder_center_x is not None and shoulder_center_y is not None and 
                    not pd.isna(shoulder_center_x) and not pd.isna(shoulder_center_y)):
                    center_points['shoulder_center'] = (int(float(shoulder_center_x)), int(float(shoulder_center_y)))
                elif 'left_shoulder' in valid_points and 'right_shoulder' in valid_points:
                    # 動的計算
                    left_shoulder = valid_points['left_shoulder']['pos']
                    right_shoulder = valid_points['right_shoulder']['pos']
                    center_points['shoulder_center'] = (
                        int((left_shoulder[0] + right_shoulder[0]) / 2),
                        int((left_shoulder[1] + right_shoulder[1]) / 2)
                    )
            except Exception as e:
                self.logger.debug(f"肩中点計算エラー: {e}")
    
            # 🎨 4つの基本キーポイント描画
            for point_name, point_data in valid_points.items():
                try:
                    pos = point_data['pos']
                    color = point_data['color']
                    conf = point_data['conf']
            
                    # キーポイントの円描画
                    cv2.circle(frame, pos, 6, color, -1)
                    cv2.circle(frame, pos, 8, (255, 255, 255), 2)  # 白い外枠
            
                    # ラベル描画
                    label = point_name.replace('_', ' ').title()
                    label_pos = (pos[0] + 10, pos[1] - 10)
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, color, 1)
                except Exception as e:
                    self.logger.debug(f"キーポイント{point_name}描画エラー: {e}")
    
            # 🎯 中点描画（特別なマーク）
            for center_name, center_pos in center_points.items():
                try:
                    color = colors[center_name]
                
                    # 中点の特別な描画（大きめの円 + クロスマーク）
                    cv2.circle(frame, center_pos, 8, color, -1)
                    cv2.circle(frame, center_pos, 10, (0, 0, 0), 2)  # 黒い外枠
            
                    # クロスマーク描画
                    cross_size = 6
                    cv2.line(frame, 
                            (center_pos[0] - cross_size, center_pos[1] - cross_size),
                            (center_pos[0] + cross_size, center_pos[1] + cross_size),
                            (0, 0, 0), 2)
                    cv2.line(frame, 
                            (center_pos[0] - cross_size, center_pos[1] + cross_size),
                            (center_pos[0] + cross_size, center_pos[1] - cross_size),
                            (0, 0, 0), 2)
            
                    # 中点ラベル
                    if center_name == 'head_center':
                        label = "HEAD"
                    elif center_name == 'shoulder_center':
                        label = "SHOULDER"
                    else:
                        label = center_name.upper()
            
                    label_pos = (center_pos[0] + 15, center_pos[1] + 5)
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
                except Exception as e:
                    self.logger.debug(f"中点{center_name}描画エラー: {e}")
    
            # 📏 接続線描画
            try:
                # 耳間の線
                if 'left_ear' in valid_points and 'right_ear' in valid_points:
                    cv2.line(frame, 
                            valid_points['left_ear']['pos'], 
                            valid_points['right_ear']['pos'],
                            colors['connection'], 2)
        
                # 肩間の線  
                if 'left_shoulder' in valid_points and 'right_shoulder' in valid_points:
                    cv2.line(frame, 
                            valid_points['left_shoulder']['pos'], 
                            valid_points['right_shoulder']['pos'],
                            colors['connection'], 3)
        
                # 体軸線（頭中点-肩中点）
                if 'head_center' in center_points and 'shoulder_center' in center_points:
                    cv2.line(frame, 
                            center_points['head_center'], 
                            center_points['shoulder_center'],
                            colors['axis'], 3)
            except Exception as e:
                self.logger.debug(f"接続線描画エラー: {e}")
    
            # 📊 統計情報をフレームに描画
            try:
                info_y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
        
                # 6点情報表示
                info_text = f"6-Point: {len(valid_points)} keypoints + {len(center_points)} centers"
                cv2.putText(frame, info_text, (10, info_y), font, 0.6, (255, 255, 255), 2)
        
                # 個別点の信頼度
                info_y += 20
                for i, (point_name, point_data) in enumerate(valid_points.items()):
                    if i >= 2:  # 最大2個まで表示
                        break
                    conf = point_data['conf']
                    info_text = f"{point_name}: {conf:.3f}"
                    cv2.putText(frame, info_text, (10, info_y), font, 0.4, (200, 200, 200), 1)
                    info_y += 15
            except Exception as e:
                self.logger.debug(f"統計情報描画エラー: {e}")
    
            return frame
    
        except Exception as e:
            self.logger.error(f"❌ 6点描画エラー: {e}")
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

    def run_experiment(self, video_path: str, experiment_type: str) -> Dict[str, Any]:
        """
        実験分析実行（完全統合版・エラーハンドリング内蔵）

        Args:
            video_path: 分析対象動画のパス
            experiment_type: 実験タイプ

        Returns:
            実験結果辞書
        """
        # 🔧 内蔵エラーハンドリング
        try:
            if ERROR_HANDLER_AVAILABLE:
                context_manager = ErrorContext(f"実験分析: {experiment_type}", 
                                            logger=self.logger, raise_on_error=False)
            else:
                context_manager = self._basic_context(f"実験分析: {experiment_type}")

            with context_manager as ctx:
                self.logger.info(f"🧪 実験分析開始: {experiment_type}")
            
                if hasattr(ctx, 'add_info'):
                    ctx.add_info("experiment_type", experiment_type)
                    ctx.add_info("video_path", str(video_path))

                # 実験タイプ別処理
                if experiment_type == "4point_keypoints":
                    # 4点キーポイント実験
                    experiment_result = self._run_4point_experiment(video_path)
                elif experiment_type == "depth_analysis":
                    # 深度分析実験
                    experiment_result = self._run_depth_experiment(video_path)
                elif experiment_type == "comparative_analysis":
                    # 比較分析実験
                    experiment_result = self._run_comparative_experiment(video_path)
                else:
                    # デフォルト: ベースライン分析を実行
                    self.logger.warning(f"⚠️ 不明な実験タイプ: {experiment_type}, ベースライン分析を実行")
                    experiment_result = self.run_baseline_analysis(video_path)
            
                return experiment_result

        except Exception as e:
            self.logger.error(f"❌ 実験分析エラー: {e}")
            import traceback
            self.logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
        
            # コンテキスト情報の追加（安全に）
            try:
                if 'ctx' in locals() and hasattr(ctx, 'add_info'):
                    ctx.add_info("error_type", type(e).__name__)
                    ctx.add_info("error_message", str(e))
            except:
                pass  # コンテキスト追加に失敗しても処理を続行
            
            return ResponseBuilder.error(e, suggestions=[
                f"実験タイプ '{experiment_type}' の設定を確認してください",
                "ベースライン分析が正常に動作するか確認してください",
                "必要なモデルファイルがダウンロードされているか確認してください",
                "ログファイルで詳細エラーを確認してください"
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
    メイン実行関数（4点キーポイント対応完全修正版）
    """
    # アスキーアートとバージョン情報
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                     🎯 YOLO11 姿勢分析システム v2.1                    ║
    ║                        キーポイント検出・追跡・解析                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # コマンドライン引数パーサーの設定
    parser = argparse.ArgumentParser(
        description="🎯 YOLO11姿勢分析システム - 動画から人物の姿勢を分析します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本的な姿勢分析
    python improved_main.py input.mp4

    # 4点キーポイント分析（高精度）
    python improved_main.py input.mp4 --use-4points --keypoint-threshold 0.5

    # 深度推定付き分析
    python improved_main.py input.mp4 --enable-depth --depth-model dpt_hybrid

    # カスタム設定ファイル使用
    python improved_main.py input.mp4 --config custom_config.yaml

    # 高解像度処理
    python improved_main.py input.mp4 --resolution 1920x1080 --quality high
        """
    )

    # 必須引数
    parser.add_argument(
        'video_path',
        type=str,
        help='🎬 分析対象の動画ファイルパス'
    )

    # オプション引数
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='⚙️ 設定ファイルパス（YAML/JSON形式）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='📁 出力ディレクトリ（デフォルト: results/動画名_YYYYMMDD_HHMMSS）'
    )

    # 🎯 4点キーポイント関連オプション
    parser.add_argument(
        '--use-4points',
        action='store_true',
        help='🦴 4点キーポイントモードを有効化（耳2点 + 肩2点）'
    )

    parser.add_argument(
        '--keypoint-threshold',
        type=float,
        default=0.3,
        help='🎯 キーポイント信頼度閾値（デフォルト: 0.3）'
    )

    parser.add_argument(
        '--disable-shoulder-metrics',
        action='store_true',
        help='🚫 肩幅メトリクスを無効化'
    )

    parser.add_argument(
        '--disable-head-tracking',
        action='store_true',
        help='🚫 頭部追跡機能を無効化'
    )

    # 深度推定オプション
    parser.add_argument(
        '--enable-depth',
        action='store_true',
        help='🌊 深度推定機能を有効化'
    )

    parser.add_argument(
        '--depth-model',
        type=str,
        default='dpt_hybrid',
        choices=['dpt_hybrid', 'midas', 'dpt_large'],
        help='🧠 深度推定モデルの選択'
    )

    # 処理オプション
    parser.add_argument(
        '--resolution',
        type=str,
        default=None,
        help='📐 処理解像度（例: 1920x1080, 1280x720）'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='🎬 処理FPS（フレーム間引き用）'
    )

    parser.add_argument(
        '--quality',
        type=str,
        default='medium',
        choices=['low', 'medium', 'high', 'ultra'],
        help='🎨 処理品質レベル'
    )

    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help='⏭️ スキップフレーム数（処理高速化用）'
    )

    # デバッグオプション
    parser.add_argument(
        '--debug',
        action='store_true',
        help='🐛 デバッグモードを有効化'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='📝 ログレベル'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='💾 中間ファイルを保存'
    )

    # モデルオプション
    parser.add_argument(
        '--model-size',
        type=str,
        default='x',
        choices=['n', 's', 'm', 'l', 'x'],
        help='🎯 YOLOモデルサイズ（n=nano, s=small, m=medium, l=large, x=xlarge）'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='🎯 検出信頼度閾値'
    )

    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.45,
        help='📐 IoU閾値（重複検出除去用）'
    )

    # 出力オプション
    parser.add_argument(
        '--disable-visualization',
        action='store_true',
        help='🚫 可視化出力を無効化'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        default='csv',
        choices=['csv', 'json', 'both'],
        help='📊 出力データ形式'
    )

    # 引数解析
    args = parser.parse_args()

    # 🔧 ログレベル設定
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

    # デバッグモード設定
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("🐛 デバッグモードが有効化されました")

    try:
        # 🔧 入力検証
        if not Path(args.video_path).exists():
            logger.error(f"❌ 動画ファイルが見つかりません: {args.video_path}")
            return 1

        video_path = Path(args.video_path)
        logger.info(f"🎬 動画ファイル: {video_path}")

        # 🔧 出力ディレクトリ設定
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = video_path.stem
            output_dir = Path("results") / f"{video_name}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 出力ディレクトリ: {output_dir}")

        # 🎯 アナライザー初期化
        try:
            # 🔧 修正: 正しいクラス名と初期化パラメータ
            analyzer = ImprovedYOLOAnalyzer(
                config_path=args.config or "configs/default.yaml"
            )
            # 🔧 深度推定設定の適用
            if args.enable_depth:
                analyzer.depth_enabled = True
                logger.info("🔍 深度推定機能を有効化")
            
            logger.info("✅ アナライザー初期化完了")
        except Exception as e:
            logger.error(f"❌ アナライザー初期化エラー: {e}")
            import traceback
            logger.error(f"🔧 詳細エラー: {traceback.format_exc()}")
            return 1

        # 🎯 修正: 4点キーポイント設定の確実な適用
        if args.use_4points:
            try:
                # 🔧 修正: 設定オーバーライドを強化
                if hasattr(analyzer, 'config') and hasattr(analyzer.config, 'data') and isinstance(analyzer.config.data, dict):
                    analyzer.config.data.setdefault('processing', {})
                    
                    # 🔧 修正: キーポイント処理設定を確実に適用
                    analyzer.config.data['processing']['use_4point_keypoints'] = True
                    analyzer.config.data['processing']['keypoint_confidence_threshold'] = args.keypoint_threshold
                    analyzer.config.data['processing']['force_pose_model'] = True  # 🔧 追加
                    analyzer.config.data['processing']['verify_keypoint_columns'] = True  # 🔧 追加
                    
                    # 🔧 修正: tracker設定も確実に設定
                    analyzer.config.data['processing'].setdefault('tracking', {})
                    analyzer.config.data['processing']['tracking']['config'] = 'bytetrack.yaml'
                    
                    # 🔧 修正: 肩・頭部設定の適用
                    if args.disable_shoulder_metrics:
                        analyzer.config.data['processing']['enable_shoulder_metrics'] = False
                        logger.info("🔧 肩メトリクスを無効化")
                    else:
                        analyzer.config.data['processing']['enable_shoulder_metrics'] = True
                        
                    if args.disable_head_tracking:
                        analyzer.config.data['processing']['enable_head_tracking'] = False
                        logger.info("🔧 頭部追跡を無効化")
                    else:
                        analyzer.config.data['processing']['enable_head_tracking'] = True
                        
                    logger.info("🔧 設定ファイルを4点キーポイントモード用に確実に更新")
                    logger.info(f"   キーポイント信頼度閾値: {args.keypoint_threshold}")
                    logger.info(f"   肩メトリクス: {'無効' if args.disable_shoulder_metrics else '有効'}")
                    logger.info(f"   頭部追跡: {'無効' if args.disable_head_tracking else '有効'}")
                    
                else:
                    # 🔧 修正: 設定がない場合の処理を強化
                    logger.error("❌ 設定オブジェクトが不正です")
                    logger.error("🔧 デフォルト4点設定を直接適用します")
                    
                    # 直接設定を作成
                    fallback_config = {
                        'processing': {
                            'use_4point_keypoints': True,
                            'keypoint_confidence_threshold': args.keypoint_threshold,
                            'force_pose_model': True,
                            'verify_keypoint_columns': True,
                            'tracking': {'config': 'bytetrack.yaml'},
                            'enable_shoulder_metrics': not args.disable_shoulder_metrics,
                            'enable_head_tracking': not args.disable_head_tracking
                        }
                    }
                    
                    if hasattr(analyzer, 'config'):
                        analyzer.config.data = fallback_config
                        logger.info("✅ フォールバック設定を適用")
                    else:
                        logger.error("🚨 設定の適用に完全に失敗しました")
                        logger.error("🚨 4点モードでの処理が正常に動作しない可能性があります")
                        
            except Exception as config_error:
                logger.error(f"❌ 4点モード設定エラー: {config_error}")
                logger.warning("⚠️ デフォルト設定で処理を続行します")

        # 🔧 品質設定の適用
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

        # 🔧 解像度設定
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

        # 🔧 その他の処理設定
        if hasattr(analyzer.config, 'data') and analyzer.config.data:
            processing_config = analyzer.config.data.setdefault('processing', {})
            
            # コマンドライン引数から設定を更新
            processing_config['confidence_threshold'] = args.confidence_threshold
            processing_config['iou_threshold'] = args.iou_threshold
            
            if args.fps:
                processing_config['target_fps'] = args.fps
                
            processing_config['skip_frames'] = args.skip_frames
            processing_config['save_intermediate'] = args.save_intermediate
            processing_config['enable_visualization'] = not args.disable_visualization
            
            # モデルサイズ設定
            model_size_map = {
                'n': 'nano', 's': 'small', 'm': 'medium', 
                'l': 'large', 'x': 'xlarge'
            }
            processing_config['model_size'] = model_size_map.get(args.model_size, 'xlarge')

        # 🎯 メイン分析処理実行
        logger.info("🚀 ========== 姿勢分析処理開始 ==========")
        
        start_time = time.time()
        
        try:
            # ベースライン分析実行
            result = analyzer.run_baseline_analysis(str(video_path))
            
            if result is None:
                logger.error("❌ 分析処理が異常終了しました（戻り値がNone）")
                return 1

            if not isinstance(result, dict):
                logger.error(f"❌ 分析処理の戻り値が想定外の型です: {type(result)}")
                return 1

            if not result.get("success", False):
                error_msg = result.get("error", "不明なエラー")
                # errorがdict型の場合も考慮
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                logger.error(f"❌ 分析処理失敗: {error_msg}")
                return 1

            processing_time = time.time() - start_time
            logger.info(f"⏱️ 総処理時間: {processing_time:.2f}秒")
            
            # 🎯 結果のレポート生成
            data = result.get("data", {})

            # detection_resultの中を参照
            detection_result = data.get("detection_result", {})
            detection_data = detection_result.get("data", {}) if isinstance(detection_result, dict) else {}

            # CSVファイルパス取得
            csv_path = detection_data.get("csv_path") or data.get("csv_path")
            if csv_path and Path(csv_path).exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                total_detections = len(df)
                total_frames = len(df['frame'].unique()) if 'frame' in df.columns else 0
                unique_ids = len(df['person_id'].unique()) if 'person_id' in df.columns else 0
            else:
                # フォールバック: 基本統計から取得
                total_detections = detection_data.get("total_detections", 0) or data.get("total_detections", 0)
                total_frames = detection_data.get("total_frames", 0) or data.get("total_frames", 0)
                unique_ids = detection_data.get("unique_ids", 0) or data.get("unique_ids", 0)
        
            logger.info("📊 ========== 処理結果サマリー ==========")
            logger.info(f"🎬 総フレーム数: {total_frames}")
            logger.info(f"🎯 総検出数: {total_detections}")
            logger.info(f"👥 ユニーク人物ID: {unique_ids}")
        
            if total_frames > 0:
                detection_rate = total_detections / total_frames
                logger.info(f"📈 フレーム当たり検出数: {detection_rate:.2f}")
        
            # キーポイント統計（4点モードの場合）
            if args.use_4points:
                keypoint_stats = detection_data.get("keypoint_stats", {}) or data.get("keypoint_stats", {})
                if keypoint_stats:
                    keypoint_frames = keypoint_stats.get("frames_with_keypoints", 0)
                    keypoint_rate = keypoint_frames / total_frames if total_frames > 0 else 0
                
                    logger.info("🦴 キーポイント統計:")
                    logger.info(f"  キーポイント検出フレーム: {keypoint_frames} ({keypoint_rate:.1%})")
                    logger.info(f"  総キーポイント数: {keypoint_stats.get('total_keypoints', 0)}")
                
                    avg_keypoints = keypoint_stats.get('avg_keypoints_per_person', 0)
                    if avg_keypoints > 0:
                        logger.info(f"  平均キーポイント/人: {avg_keypoints:.1f}")
        
            # 出力ファイル一覧
            output_files = detection_data.get("output_files", []) or data.get("output_files", [])
            if output_files:
                logger.info("📁 生成ファイル:")
                for file_path in output_files:
                    if Path(file_path).exists():
                        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                        logger.info(f"  ✅ {file_path} ({size_mb:.2f}MB)")
                    else:
                        logger.warning(f"  ⚠️ {file_path} (ファイルが見つかりません)")
        
            # パフォーマンス統計
            fps = total_frames / processing_time if processing_time > 0 else 0
            logger.info(f"⚡ 処理性能: {fps:.2f} FPS")
            
            # エラー報告
            if hasattr(analyzer, 'error_collector') and analyzer.error_collector:
                logger.warning(f"⚠️ 処理中のエラー: {len(analyzer.error_collector)}件")
                for i, error in enumerate(analyzer.error_collector[:5], 1):
                    logger.warning(f"  {i}. {error}")
                if len(analyzer.error_collector) > 5:
                    logger.warning(f"  ... 他 {len(analyzer.error_collector) - 5}件")
            
            logger.info("🎯 ========== 処理完了 ==========")
            
            # 成功時の追加情報
            if args.use_4points:
                logger.info("💡 4点キーポイントデータを確認してください:")
                logger.info("   - 4point_keypoints.csv: フィルタリング済みデータ")
                logger.info("   - 4point_metrics.csv: 姿勢メトリクス付きデータ")
            
            if args.enable_depth:
                logger.info("💡 深度推定データを確認してください:")
                logger.info("   - depth_analysis/ ディレクトリ内の深度マップ")
                
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