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
            """モデルロード（改善版）"""
            try:
                if hasattr(self.config, 'get'):
                    models_config = self.config.get('models', {})
                elif isinstance(self.config, dict):
                    models_config = self.config.get('models', {})
                else:
                    models_config = {}
                
                detection_path = models_config.get('detection', 'models/yolo/yolo11x.pt')
                pose_path = models_config.get('pose', 'models/yolo/yolo11x-pose.pt')
                
                self.logger.info(f"🔍 モデルロード開始")
                
                # 検出モデル
                if Path(detection_path).exists():
                    self.detection_model = YOLO(detection_path)
                    self.logger.info(f"✅ 検出モデル: {detection_path}")
                else:
                    self.logger.warning(f"⚠️ 検出モデル未発見: {detection_path}")
                
                # ポーズモデル
                if Path(pose_path).exists():
                    self.pose_model = YOLO(pose_path)
                    self.logger.info(f"✅ ポーズモデル: {pose_path}")
                else:
                    self.logger.warning(f"⚠️ ポーズモデル未発見: {pose_path}")
                    
            except Exception as e:
                self.logger.error(f"❌ モデルロードエラー: {e}")
        
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

        def create_visualizations(self, detection_results, vis_dir):
            """基本可視化（完全版・確実な戻り値付き）"""
            self.logger.info(f"📈 基本可視化生成: {vis_dir}")
    
            # 🔧 必ず戻り値を返すようにする
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
        
                # ディレクトリ作成
                vis_path = Path(str(vis_dir))
                vis_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"📁 可視化ディレクトリ作成: {vis_path}")
        
                # detection_results の詳細ログ
                self.logger.info(f"🔧 detection_results type: {type(detection_results)}")
        
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
                result["basic_stats_file"] = str(stats_file)
                result["total_files"] = 1
        
                # 統計グラフ生成
                graphs_generated = 0
        
                try:
                    # matplotlib/pandas のインポート
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import pandas as pd
            
                    # 簡易フォント設定
                    try:
                        plt.rcParams['font.family'] = 'Hiragino Sans'
                    except:
                        plt.rcParams['font.family'] = 'DejaVu Sans'
            
                    # CSV ファイルの処理
                    if csv_path and Path(csv_path).exists():
                        self.logger.info(f"📊 CSVファイル読み込み: {csv_path}")
                        df = pd.read_csv(csv_path)
                        self.logger.info(f"📊 データ読み込み: {len(df)}行, カラム: {list(df.columns)}")
                
                        if not df.empty:
                            # 1. フレーム別検出数グラフ
                            if 'frame' in df.columns:
                                try:
                                    plt.figure(figsize=(12, 6))
                                    frame_counts = df['frame'].value_counts().sort_index()
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
                            if 'conf' in df.columns:
                                try:
                                    plt.figure(figsize=(10, 6))
                                    conf_data = df['conf'].dropna()
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
        
                # 🔧 必ず辞書を返す
                return result
        
            except Exception as e:
                self.logger.error(f"❌ 可視化生成全体エラー: {e}", exc_info=True)
                # 🔧 エラー時も辞書を返す
                result.update({
                    "success": False,
                    "error": str(e)
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

            # 評価器の選択と初期化
            self._initialize_evaluator(ctx)
        
            # プロセッサーとアナライザー初期化
            self._initialize_processor_analyzer(ctx)

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
        # プロセッサー初期化
        if VIDEO_PROCESSOR_AVAILABLE and VideoProcessor:
            try:
                self.processor = VideoProcessor(self.config)
                self.logger.info("🎥 高度動画プロセッサーを初期化")
            except Exception as e:
                self.logger.warning(f"VideoProcessor 初期化失敗: {e}")
                self.processor = BasicVideoProcessor(self.config)
                self.logger.info("🔧 基本動画プロセッサーを初期化")
        else:
            self.processor = BasicVideoProcessor(self.config)
            self.logger.info("🔧 基本動画プロセッサーを初期化")
            
        # アナライザー初期化
        if METRICS_ANALYZER_AVAILABLE and MetricsAnalyzer:
            try:
                self.analyzer = MetricsAnalyzer(self.config)
                self.logger.info("📊 高度メトリクス分析器を初期化")
            except Exception as e:
                self.logger.warning(f"MetricsAnalyzer 初期化失敗: {e}")
                self.analyzer = BasicMetricsAnalyzer(self.config)
                self.logger.info("🔧 基本メトリクス分析器を初期化")
        else:
            self.analyzer = BasicMetricsAnalyzer(self.config)
            self.logger.info("🔧 基本メトリクス分析器を初期化")

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
    def run_baseline_analysis(self, video_path: str) -> Dict[str, Any]:
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
            output_dir = Path("outputs/baseline") / video_name
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

                # Step 2: 検出・追跡処理（フォールバック対応）
                if self.depth_enabled:
                    self.logger.info("🔍 Step 2: 深度統合検出・追跡処理開始")
                    if hasattr(self.processor, 'run_detection_tracking_with_depth'):
                        detection_result = self.processor.run_detection_tracking_with_depth(frame_dir, video_name)
                    else:
                        self.logger.warning("深度統合処理が利用できません。標準処理にフォールバック")
                        detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
                    processing_type = "深度統合"
                else:
                    self.logger.info("👁️ Step 2: 標準検出・追跡処理開始")
                    detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
                    processing_type = "標準"

                # 🔧 検出処理が失敗した場合のフォールバック
                if not detection_result.get("success", False):
                    error_msg = detection_result.get("error", "不明なエラー")
                    self.logger.warning(f"⚠️ {processing_type}処理エラー: {error_msg}")
                    
                    # BasicVideoProcessorのフォールバック処理を試行
                    if VIDEO_PROCESSOR_AVAILABLE and not isinstance(self.processor, BasicVideoProcessor):
                        self.logger.info("🔄 BasicVideoProcessorにフォールバック")
                        fallback_processor = BasicVideoProcessor(self.config)
                        fallback_processor.load_models()
                        detection_result = fallback_processor.run_detection_tracking(frame_dir, video_name)
                        
                        if detection_result.get("success", False):
                            self.logger.info("✅ フォールバック処理成功")
                            processing_type = "フォールバック"
                        else:
                            self.error_collector.append(f"{processing_type}処理失敗: {error_msg}")
                            self.logger.error(f"❌ フォールバック処理も失敗")
                            raise VideoProcessingError(error_msg)
                    else:
                        self.error_collector.append(f"{processing_type}処理失敗: {error_msg}")
                        self.logger.error(f"❌ {error_msg}")
                        raise VideoProcessingError(error_msg)

                self.logger.info(f"✅ Step 2完了: {processing_type}処理")

                # 🎯 Step 2.5: 4点キーポイント処理（オプション）
                use_4point_keypoints = self.config.get('processing.use_4point_keypoints', False)
                
                if use_4point_keypoints and detection_result.get("success", False):
                    self.logger.info("🎯 Step 2.5: 4点キーポイント処理開始")
                    
                    try:
                        original_csv = detection_result["data"]["csv_path"]
                        filtered_csv = self.filter_keypoints_to_4points(original_csv, output_dir)
                        
                        # 結果に4点情報追加
                        detection_result["data"]["filtered_csv_path"] = filtered_csv
                        detection_result["data"]["keypoint_mode"] = "4_points"
                        
                        # 4点専用可視化
                        vis_result = self.create_4point_visualization(filtered_csv, video_path, output_dir)
                        if vis_result.get("success", False):
                            detection_result["data"]["visualization_4points"] = vis_result
                            self.logger.info("✅ 4点専用可視化完了")
                        
                        self.logger.info("✅ Step 2.5完了: 4点キーポイント処理")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Step 2.5警告: 4点処理エラー（処理継続）: {e}")
                        self.error_collector.append(f"4点キーポイント処理エラー: {e}")

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
                    }
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

    # 🎯 4点キーポイント機能の追加
    def filter_keypoints_to_4points(self, csv_path, output_dir):
        """
        17点キーポイントを4点（両耳・両肩）にフィルタリング
        
        Args:
            csv_path: 元の17点キーポイントCSV
            output_dir: 出力ディレクトリ
            
        Returns:
            filtered_csv_path: 4点フィルタリング済みCSV
        """
        try:
            import pandas as pd
            import numpy as np
            
            self.logger.info("🎯 4点キーポイントフィルタリング開始")
            
            # CSV読み込み
            df = pd.read_csv(csv_path)
            
            # 4点キーポイント（両耳・両肩）のインデックス
            target_keypoints = {
                "left_ear": 3,
                "right_ear": 4, 
                "left_shoulder": 5,
                "right_shoulder": 6
            }
            
            # 4点用の新しいカラム作成
            for name, idx in target_keypoints.items():
                x_col = f"keypoint_{idx}_x"
                y_col = f"keypoint_{idx}_y"
                conf_col = f"keypoint_{idx}_conf"
                
                if x_col in df.columns:
                    df[f"kpt4_{name}_x"] = df[x_col]
                    df[f"kpt4_{name}_y"] = df[y_col]
                    df[f"kpt4_{name}_conf"] = df[conf_col]
            
            # 元の17点キーポイントカラムを削除
            original_kpt_cols = [col for col in df.columns if 'keypoint_' in col and 'kpt4_' not in col]
            df = df.drop(columns=original_kpt_cols)
            
            # 4点専用メトリクス追加
            self.add_4point_metrics(df)
            
            # 保存
            filtered_csv = Path(output_dir) / "detections_4points.csv"
            df.to_csv(filtered_csv, index=False)
            
            self.logger.info(f"✅ 4点フィルタリング完了: {filtered_csv}")
            return str(filtered_csv)
            
        except Exception as e:
            self.logger.error(f"❌ 4点フィルタリングエラー: {e}")
            return csv_path

    def add_4point_metrics(self, df):
        """4点キーポイント専用メトリクス計算"""
        try:
            import numpy as np
            
            # 肩幅計算
            if ('kpt4_left_shoulder_x' in df.columns and 
                'kpt4_right_shoulder_x' in df.columns):
                
                df['shoulder_width'] = np.sqrt(
                    (df['kpt4_left_shoulder_x'] - df['kpt4_right_shoulder_x'])**2 + 
                    (df['kpt4_left_shoulder_y'] - df['kpt4_right_shoulder_y'])**2
                )
            
            # 頭部中心位置
            if ('kpt4_left_ear_x' in df.columns and 
                'kpt4_right_ear_x' in df.columns):
                
                df['head_center_x'] = (df['kpt4_left_ear_x'] + df['kpt4_right_ear_x']) / 2
                df['head_center_y'] = (df['kpt4_left_ear_y'] + df['kpt4_right_ear_y']) / 2
            
            # 肩角度計算
            if 'shoulder_width' in df.columns:
                df['shoulder_angle'] = np.arctan2(
                    df['kpt4_right_shoulder_y'] - df['kpt4_left_shoulder_y'],
                    df['kpt4_right_shoulder_x'] - df['kpt4_left_shoulder_x']
                ) * 180 / np.pi
            
            # 4点品質スコア
            conf_cols = [col for col in df.columns if 'kpt4_' in col and '_conf' in col]
            if conf_cols:
                df['avg_4point_confidence'] = df[conf_cols].mean(axis=1)
                df['valid_4point_count'] = (df[conf_cols] > 0.3).sum(axis=1)
                df['keypoint_quality_score'] = df['avg_4point_confidence'] * (df['valid_4point_count'] / 4)
            
            self.logger.info("✅ 4点メトリクス計算完了")
            
        except Exception as e:
            self.logger.error(f"❌ 4点メトリクス計算エラー: {e}")

    def create_4point_visualization(self, csv_path, video_path, output_dir):
        """4点キーポイント専用可視化生成"""
        try:
            import cv2
            import pandas as pd
            
            self.logger.info("🎨 4点可視化生成開始")
            
            # 出力ディレクトリ
            vis_dir = Path(output_dir) / "visualized_frames_4points"
            vis_dir.mkdir(exist_ok=True)
            
            # CSV読み込み
            df = pd.read_csv(csv_path)
            
            # 動画読み込み
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"動画を開けません: {video_path}")
            
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 該当フレームのデータ
                frame_data = df[df['frame'] == frame_count]
                
                if not frame_data.empty:
                    # 4点キーポイント描画
                    for _, row in frame_data.iterrows():
                        frame = self.draw_4point_keypoints(frame, row)
                    
                    # フレーム保存
                    frame_file = vis_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    saved_count += 1
                
                frame_count += 1
                
                # 進捗表示
                if frame_count % 50 == 0:
                    self.logger.info(f"🎨 4点可視化進捗: {frame_count}フレーム")
            
            cap.release()
            
            self.logger.info(f"✅ 4点可視化完了: {saved_count}フレーム保存")
            return {"success": True, "frames_saved": saved_count, "output_dir": str(vis_dir)}
            
        except Exception as e:
            self.logger.error(f"❌ 4点可視化エラー: {e}")
            return {"success": False, "error": str(e)}

    def draw_4point_keypoints(self, frame, row):
        """フレームに4点キーポイント描画"""
        try:
            import cv2
            
            # 色定義
            ear_color = (0, 255, 0)      # 緑（耳）
            shoulder_color = (255, 0, 0)  # 青（肩）
            line_color = (0, 255, 255)    # 黄（線）
            text_color = (255, 255, 255)  # 白（テキスト）
            
            # 4点キーポイント取得
            keypoints = {}
            for name in ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']:
                x = row.get(f'kpt4_{name}_x', 0)
                y = row.get(f'kpt4_{name}_y', 0)
                conf = row.get(f'kpt4_{name}_conf', 0)
                
                if conf > 0.3 and x > 0 and y > 0:
                    keypoints[name] = (int(x), int(y), conf)
            
            # キーポイント描画
            ear_points = []
            shoulder_points = []
            
            for name, (x, y, conf) in keypoints.items():
                color = ear_color if 'ear' in name else shoulder_color
                
                # キーポイント円
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 8, text_color, 2)
                
                # ラベル
                cv2.putText(frame, f"{name}:{conf:.2f}", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 点を分類
                if 'ear' in name:
                    ear_points.append((x, y))
                elif 'shoulder' in name:
                    shoulder_points.append((x, y))
            
            # 肩ライン描画
            if len(shoulder_points) == 2:
                cv2.line(frame, shoulder_points[0], shoulder_points[1], line_color, 3)
                
                # 肩幅表示
                if 'shoulder_width' in row and not pd.isna(row['shoulder_width']):
                    mid_x = (shoulder_points[0][0] + shoulder_points[1][0]) // 2
                    mid_y = (shoulder_points[0][1] + shoulder_points[1][1]) // 2
                    cv2.putText(frame, f"SW:{row['shoulder_width']:.1f}", 
                               (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
            
            # 頭部中心描画
            if len(ear_points) == 2:
                head_x = (ear_points[0][0] + ear_points[1][0]) // 2
                head_y = (ear_points[0][1] + ear_points[1][1]) // 2
                cv2.circle(frame, (head_x, head_y), 4, line_color, -1)
            
            # 人物ID表示
            person_id = row.get('person_id', -1)
            if person_id != -1 and keypoints:
                all_points = list(keypoints.values())
                center_x = int(np.mean([p[0] for p in all_points]))
                center_y = int(np.mean([p[1] for p in all_points])) - 30
                
                quality_score = row.get('keypoint_quality_score', 0)
                text = f"ID:{person_id} Q:{quality_score:.2f}"
                cv2.putText(frame, text, (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"フレーム描画エラー: {e}")
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
    メイン実行関数（完全統合版）
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description="YOLO11 広角カメラ分析システム（完全統合版・フォールバック対応）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 使用例:
  # ベースライン分析（標準モード）
  python improved_main.py --mode baseline --config configs/default.yaml
  
  # ベースライン分析（深度推定モード）
  python improved_main.py --mode baseline --config configs/depth_config.yaml
  
  # 🎯 4点キーポイントモード
  python improved_main.py --mode baseline --use-4points --keypoint-threshold 0.4
  
  # 実験分析
  python improved_main.py --mode experiment --experiment-type tile_inference
  
  # 特定動画の分析
  python improved_main.py --mode baseline --video path/to/video.mp4
  
  # 詳細ログ + エラーレポート
  python improved_main.py --mode baseline --verbose --generate-report
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
    
    # 🎯 4点キーポイント専用引数をここに追加
    parser.add_argument(
        "--use-4points",
        action="store_true",
        help="4点キーポイントモードを有効化（両耳・両肩のみ）"
    )
    
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="キーポイント信頼度閾値（0.0-1.0、デフォルト: 0.3）"
    )
    
    parser.add_argument(
        "--disable-shoulder-metrics",
        action="store_true",
        help="肩幅・姿勢解析を無効化"
    )
    
    parser.add_argument(
        "--disable-head-tracking",
        action="store_true",
        help="頭部位置追跡を無効化"
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

    # 🎯 4点キーポイント設定の動的更新をここに追加
    if args.use_4points:
        print("🎯 4点キーポイントモードを強制有効化")
        print(f"📊 信頼度閾値: {args.keypoint_threshold}")
        
        if args.disable_shoulder_metrics:
            print("📏 肩幅・姿勢解析を無効化")
        
        if args.disable_head_tracking:
            print("👤 頭部位置追跡を無効化")

    # ログレベル設定
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # ロガー初期化
    logger = setup_logger()

    logger.info("🚀 YOLO11 広角カメラ分析システム 開始（完全統合版）")
    logger.info(f"📋 実行モード: {args.mode}")
    logger.info(f"⚙️ 設定ファイル: {args.config}")
    logger.info(f"📊 詳細ログ: {'有効' if args.verbose else '無効'}")
    
    # 🎯 4点キーポイントモードのログ出力
    if args.use_4points:
        logger.info("🎯 4点キーポイントモード有効")
        logger.info(f"📊 キーポイント信頼度閾値: {args.keypoint_threshold}")
        if args.disable_shoulder_metrics:
            logger.info("📏 肩幅・姿勢解析: 無効")
        if args.disable_head_tracking:
            logger.info("👤 頭部位置追跡: 無効")
    
    # モジュール可用性の詳細報告
    available_modules = []
    fallback_modules = []
    
    module_status = {
        "統一エラーハンドラー": ERROR_HANDLER_AVAILABLE,
        "包括的評価器": COMPREHENSIVE_EVALUATOR_AVAILABLE,
        "深度統合評価器": DEPTH_EVALUATOR_AVAILABLE,
        "高度動画処理": VIDEO_PROCESSOR_AVAILABLE,
        "高度メトリクス分析": METRICS_ANALYZER_AVAILABLE,
        "高度設定管理": CONFIG_AVAILABLE,
        "高度ログ機能": LOGGER_AVAILABLE
    }
    
    for module_name, available in module_status.items():
        if available:
            available_modules.append(module_name)
        else:
            fallback_modules.append(module_name.replace("高度", "基本").replace("統一", "基本").replace("包括的", "基本").replace("深度統合", "基本"))
        
    if available_modules:
        logger.info(f"✅ 利用可能な高度機能: {', '.join(available_modules)}")
    if fallback_modules:
        logger.info(f"🔧 フォールバック機能使用: {', '.join(fallback_modules)}")

    try:
        # 分析器初期化
        logger.info("⚙️ 分析器初期化開始...")
        analyzer = ImprovedYOLOAnalyzer(args.config)
        logger.info("✅ 分析器初期化完了")
        
        # 🎯 4点キーポイント設定の強制適用
        if args.use_4points:
            # 設定オーバーライド
            if hasattr(analyzer.config, 'data') and isinstance(analyzer.config.data, dict):
                analyzer.config.data.setdefault('processing', {})
                analyzer.config.data['processing']['use_4point_keypoints'] = True
                analyzer.config.data['processing']['keypoint_confidence_threshold'] = args.keypoint_threshold
                
                if args.disable_shoulder_metrics:
                    analyzer.config.data['processing']['enable_shoulder_metrics'] = False
                    
                if args.disable_head_tracking:
                    analyzer.config.data['processing']['enable_head_tracking'] = False
                    
                logger.info("🔧 設定ファイルを4点キーポイントモード用に動的更新")
            else:
                logger.warning("⚠️ 設定ファイルの動的更新に失敗しました")
        
        # 動画ファイル決定
        if args.video:
            video_path = Path(args.video)
            if not video_path.exists():
                raise FileNotFoundError(f"指定された動画ファイルが存在しません: {video_path}")
            video_files = [video_path]
            logger.info(f"🎬 指定動画: {video_path.name}")
        else:
            video_files = analyzer.get_video_files()
            if not video_files:
                raise FileNotFoundError(f"動画ファイルが見つかりません。{analyzer.config.get('video_dir', 'videos')}ディレクトリに動画ファイルを配置してください")

        logger.info(f"🎥 処理対象動画: {len(video_files)}ファイル")
        
        # 4点キーポイント情報の詳細表示
        if args.use_4points:
            logger.info("🎯 4点キーポイント処理設定:")
            logger.info("   - 対象キーポイント: left_ear, right_ear, left_shoulder, right_shoulder")
            logger.info(f"   - 信頼度閾値: {args.keypoint_threshold}")
            logger.info(f"   - 肩幅解析: {'無効' if args.disable_shoulder_metrics else '有効'}")
            logger.info(f"   - 頭部追跡: {'無効' if args.disable_head_tracking else '有効'}")
        
        # 分析実行
        all_results = []
        successful_count = 0
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"📹 処理開始 ({i}/{len(video_files)}): {video_file.name}")
            
            # 🎯 4点モードの場合の特別表示
            if args.use_4points:
                logger.info(f"🎯 4点キーポイントモードで処理中: {video_file.name}")
            
            try:
                if args.mode == "baseline":
                    result = analyzer.run_baseline_analysis(str(video_file))
                elif args.mode == "experiment":
                    result = analyzer.run_experiment(str(video_file), args.experiment_type)
                else:
                    raise ValueError(f"不正な実行モード: {args.mode}")
                
                all_results.append({
                    "video_file": str(video_file),
                    "video_name": video_file.name,
                    "result": result,
                    "keypoint_mode": "4_points" if args.use_4points else "17_points"  # 🎯 追加
                })
                
                if result.get("success", False):
                    successful_count += 1
                    # 🎯 4点モード成功の特別表示
                    if args.use_4points:
                        logger.info(f"✅ 4点キーポイント処理完了 ({i}/{len(video_files)}): {video_file.name}")
                        
                        # 4点専用結果の表示
                        data = result.get("data", {})
                        if "filtered_csv_path" in data:
                            logger.info(f"📊 4点フィルタリング済みCSV: {Path(data['filtered_csv_path']).name}")
                        if "visualization_4points" in data:
                            vis_info = data["visualization_4points"]
                            logger.info(f"🎨 4点可視化: {vis_info.get('frames_saved', 0)}フレーム生成")
                    else:
                        logger.info(f"✅ 処理完了 ({i}/{len(video_files)}): {video_file.name}")
                else:
                    logger.error(f"❌ 処理失敗 ({i}/{len(video_files)}): {video_file.name}")
                    if result.get("error"):
                        logger.error(f"  エラー詳細: {result['error'].get('message', '不明')}")
                        
            except Exception as e:
                logger.error(f"❌ 動画処理エラー ({video_file.name}): {e}")
                all_results.append({
                    "video_file": str(video_file),
                    "video_name": video_file.name,
                    "result": ResponseBuilder.error(e),
                    "keypoint_mode": "4_points" if args.use_4points else "17_points"  # 🎯 追加
                })

        # 全体結果サマリー（4点モード対応）
        total = len(all_results)
        success_rate = (successful_count / total) * 100 if total > 0 else 0
        
        logger.info(f"📊 処理結果サマリー: {successful_count}/{total} 成功 ({success_rate:.1f}%)")
        
        # 🎯 4点キーポイントモード特有のサマリー
        if args.use_4points:
            logger.info("🎯 4点キーポイントモード処理サマリー:")
            fourpoint_success = 0
            for result_entry in all_results:
                result = result_entry["result"]
                if result.get("success", False) and result_entry.get("keypoint_mode") == "4_points":
                    fourpoint_success += 1
            
            logger.info(f"   - 4点処理成功: {fourpoint_success}/{total}")
            logger.info(f"   - 信頼度閾値: {args.keypoint_threshold}")
            logger.info(f"   - 有効機能: 肩幅解析={'○' if not args.disable_shoulder_metrics else '×'}, 頭部追跡={'○' if not args.disable_head_tracking else '×'}")

        # エラーレポート生成
        if args.generate_report or analyzer.error_collector:
            logger.info("📋 エラーレポート生成中...")
            error_report = analyzer.generate_error_report()
            logger.info(f"📋 エラーレポート: {error_report.get('total_errors', 0)}件のエラー")

        # 統合結果ファイル保存
        summary_result = {
            "execution_mode": args.mode,
            "config_file": args.config,
            "keypoint_mode": "4_points" if args.use_4points else "17_points",  # 🎯 追加
            "keypoint_settings": {  # 🎯 追加
                "use_4points": args.use_4points,
                "threshold": args.keypoint_threshold,
                "shoulder_metrics": not args.disable_shoulder_metrics,
                "head_tracking": not args.disable_head_tracking
            } if args.use_4points else None,
            "execution_timestamp": datetime.now().isoformat(),
            "total_videos": total,
            "successful_videos": successful_count,
            "success_rate": success_rate,
            "video_results": all_results,
            "system_info": {
                "depth_enabled": analyzer.depth_enabled,
                "module_availability": module_status,
                "fallback_count": len(fallback_modules),
                "evaluator_type": type(analyzer.evaluator).__name__,
                "processor_type": type(analyzer.processor).__name__,
                "analyzer_type": type(analyzer.analyzer).__name__
            },
            "command_line_args": vars(args)
        }
        
        # 🎯 4点モード用のファイル名
        mode_suffix = f"{args.mode}_4points" if args.use_4points else args.mode
        summary_file = Path("outputs") / f"summary_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 統合結果保存: {summary_file}")

        # 処理完了
        if successful_count == total:
            logger.info("🎉 全ての動画処理が成功しました")
            print(f"\n✅ 処理完了: {successful_count}/{total} 成功 (成功率: 100%)")
            print(f"📁 結果保存先: outputs/{args.mode}/")
            
            # 🎯 4点モード特有の完了メッセージ
            if args.use_4points:
                print("🎯 4点キーポイントモード処理完了!")
                print("   - 出力: 肩幅、頭部位置、姿勢角度データを含むCSV")
                print("   - 可視化: 4点専用の見やすい可視化フレーム")
                print(f"   - 信頼度: {args.keypoint_threshold}以上のキーポイントのみ使用")
            
            if fallback_modules:
                print(f"🔧 フォールバック機能使用: {len(fallback_modules)}個")
            return True
        elif successful_count > 0:
            logger.warning(f"⚠️ 一部の動画処理が失敗しました ({successful_count}/{total})")
            print(f"\n⚠️ 部分的成功: {successful_count}/{total} (成功率: {success_rate:.1f}%)")
            print(f"📋 詳細はログファイルを確認してください")
            print(f"📁 結果保存先: outputs/{args.mode}/")
            return True
        else:
            logger.error("❌ 全ての動画処理が失敗しました")
            print(f"\n❌ 全て失敗: 0/{total}")
            print(f"📋 詳細はログファイルとエラーレポートを確認してください")
            return False

    except ConfigurationError as e:
        logger.error(f"❌ 設定エラー: {e}")
        print(f"❌ 設定エラー: {e}")
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
        print("  5. --generate-report でエラーレポートを生成")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ システムエラー: {e}")
        logging.error(f"システムエラー: {e}")
        sys.exit(1)