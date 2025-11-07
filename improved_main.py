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
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, Optional, List
import time

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
    
    class ResponseBuilder:
        @staticmethod
        def success(data=None, message=""):
            return {"success": True, "data": data, "message": message}
        
        @staticmethod
        def error(exception, suggestions=None):
            return {
                "success": False, 
                "error": {"message": str(exception), "type": type(exception).__name__},
                "suggestions": suggestions or []
            }
    
    # 🔧 handle_errors デコレーター（完全版）
    def handle_errors(error_category=None, suppress_exceptions=False, logger=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_logger = logger or logging.getLogger(__name__)
                    error_logger.error(f"❌ エラー in {func.__name__} (カテゴリ: {error_category}): {e}")
                    
                    if suppress_exceptions:
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
            self.config = config
            self.detection_model = None
            self.pose_model = None
            self.logger = logging.getLogger(__name__)
            self.processing_stats = {}
            self.load_models()
        
        def load_models(self):
            """モデルロード（改善版）"""
            try:
                if hasattr(self.config, 'get'):
                    models_config = self.config.get('models', {})
                elif isinstance(self.config, dict):
                    models_config = self.config.get('models', {})
                else:
                    models_config = {}
                
                detection_path = models_config.get('detection', 'models/yolo/yolo11m.pt')
                pose_path = models_config.get('pose', 'models/yolo/yolo11m-pose.pt')
                
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
        
        def extract_frames(self, video_path, frame_dir, max_frames=100):
            """フレーム抽出（完全版）"""
            try:
                self.logger.info(f"📸 フレーム抽出開始: {video_path}")
                frame_dir = Path(frame_dir)
                frame_dir.mkdir(parents=True, exist_ok=True)
                
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise VideoProcessingError(f"動画ファイルを開けません: {video_path}")
                
                # 動画情報取得
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                self.logger.info(f"📹 動画情報: {width}x{height}, {frame_count}フレーム, {fps:.1f}FPS, {duration:.1f}秒")
                
                interval = max(1, frame_count // max_frames)
                self.logger.info(f"🔢 抽出間隔: {interval} (最大{max_frames}フレーム)")
                
                extracted = 0
                for i in tqdm(range(0, frame_count, interval), desc="フレーム抽出"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame_path = frame_dir / f"frame_{i:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        extracted += 1
                
                cap.release()
                
                self.processing_stats["frame_extraction"] = {
                    "total_frames": frame_count,
                    "extracted_frames": extracted,
                    "video_fps": fps,
                    "video_duration": duration,
                    "resolution": [width, height],
                    "extraction_interval": interval
                }
                
                self.logger.info(f"✅ フレーム抽出完了: {extracted}フレーム")
                return {
                    "success": True, 
                    "extracted_frames": extracted, 
                    "video_info": self.processing_stats["frame_extraction"]
                }
                
            except Exception as e:
                self.logger.error(f"❌ フレーム抽出エラー: {e}")
                return {"success": False, "error": str(e)}
        
        def run_detection_tracking(self, frame_dir, video_name):
            """基本検出・追跡処理（完全版）"""
            try:
                self.logger.info(f"👁️ 検出・追跡処理開始: {video_name}")
                frame_files = sorted(list(Path(frame_dir).glob("*.jpg")))
                if not frame_files:
                    raise VideoProcessingError("フレームファイルが見つかりません")
                
                detection_count = 0
                pose_count = 0
                processing_times = []
                frame_stats = []
                
                # 信頼度しきい値
                conf_threshold = 0.3
                
                for i, frame_file in enumerate(tqdm(frame_files, desc="検出処理")):
                    frame_start_time = time.time()
                    frame = cv2.imread(str(frame_file))
                    
                    frame_detections = 0
                    frame_poses = 0
                    detection_boxes = []
                    
                    # 検出実行
                    if self.detection_model:
                        try:
                            det_results = self.detection_model(frame, verbose=False, conf=conf_threshold)
                            if det_results and len(det_results[0].boxes) > 0:
                                frame_detections = len(det_results[0].boxes)
                                detection_count += frame_detections
                                
                                # ボックス情報保存
                                for box in det_results[0].boxes:
                                    detection_boxes.append({
                                        "confidence": float(box.conf[0]) if len(box.conf) > 0 else 0.0,
                                        "class": int(box.cls[0]) if len(box.cls) > 0 else 0,
                                        "bbox": box.xyxy[0].tolist() if len(box.xyxy) > 0 else []
                                    })
                        except Exception as e:
                            self.logger.warning(f"フレーム{i}検出エラー: {e}")
                    
                    # ポーズ実行
                    if self.pose_model:
                        try:
                            pose_results = self.pose_model(frame, verbose=False, conf=conf_threshold)
                            if pose_results and len(pose_results[0].boxes) > 0:
                                frame_poses = len(pose_results[0].boxes)
                                pose_count += frame_poses
                        except Exception as e:
                            self.logger.warning(f"フレーム{i}ポーズエラー: {e}")
                    
                    # フレーム統計記録
                    frame_end_time = time.time()
                    processing_time = frame_end_time - frame_start_time
                    processing_times.append(processing_time)
                    
                    frame_stats.append({
                        "frame_id": i,
                        "frame_file": frame_file.name,
                        "detections": frame_detections,
                        "poses": frame_poses,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": np.mean([box["confidence"] for box in detection_boxes]) if detection_boxes else 0.0,
                        "track_id": i  # 簡易トラッキングID
                    })
                
                # 結果CSV作成
                output_dir = Path("outputs/temp") / video_name
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"{video_name}_results.csv"
                
                df = pd.DataFrame(frame_stats)
                df.to_csv(csv_path, index=False)
                
                # 処理統計
                self.processing_stats["detection_tracking"] = {
                    "total_frames": len(frame_files),
                    "total_detections": detection_count,
                    "total_poses": pose_count,
                    "avg_processing_time": np.mean(processing_times) if processing_times else 0,
                    "total_processing_time": sum(processing_times),
                    "fps_effective": len(frame_files) / sum(processing_times) if sum(processing_times) > 0 else 0
                }
                
                self.logger.info(f"✅ 検出・追跡完了: 検出{detection_count}個, ポーズ{pose_count}個")
                
                return {
                    "success": True,
                    "data": {
                        "csv_path": str(csv_path),
                        "detection_count": detection_count,
                        "pose_count": pose_count,
                        "frame_count": len(frame_files),
                        "processing_time": sum(processing_times),
                        "processing_stats": self.processing_stats["detection_tracking"]
                    }
                }
                
            except Exception as e:
                self.logger.error(f"❌ 検出・追跡エラー: {e}")
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
            """基本可視化（完全版）"""
            self.logger.info(f"📈 基本可視化生成: {vis_dir}")
            
            try:
                vis_path = Path(vis_dir)
                vis_path.mkdir(parents=True, exist_ok=True)
                
                # データ抽出
                if isinstance(detection_results, dict):
                    data = detection_results.get("data", {}) if detection_results.get("success", False) else {}
                else:
                    data = {}
                
                # 基本統計ファイル作成
                stats_file = vis_path / "basic_stats.json"
                
                basic_stats = {
                    "visualization_type": "BasicVisualization",
                    "detection_count": data.get("detection_count", 0),
                    "frame_count": data.get("frame_count", 0),
                    "processing_time": data.get("processing_time", 0),
                    "processing_stats": data.get("processing_stats", {}),
                    "timestamp": datetime.now().isoformat(),
                    "csv_path": data.get("csv_path"),
                    "success": detection_results.get("success", False) if isinstance(detection_results, dict) else False
                }
                
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(basic_stats, f, indent=2, ensure_ascii=False)
                
                # 簡易グラフ作成（matplotlib使用）
                try:
                    if data.get("csv_path") and Path(data["csv_path"]).exists():
                        df = pd.read_csv(data["csv_path"])
                        
                        # 検出数の時系列グラフ
                        plt.figure(figsize=(10, 6))
                        plt.plot(df['frame_id'], df['detections'], marker='o', linestyle='-', alpha=0.7)
                        plt.title('Frame-wise Detection Count')
                        plt.xlabel('Frame ID')
                        plt.ylabel('Detection Count')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(vis_path / "detection_timeline.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # 処理時間のヒストグラム
                        if 'processing_time' in df.columns:
                            plt.figure(figsize=(8, 6))
                            plt.hist(df['processing_time'], bins=20, alpha=0.7, edgecolor='black')
                            plt.title('Processing Time Distribution')
                            plt.xlabel('Processing Time (seconds)')
                            plt.ylabel('Frequency')
                            plt.grid(True, alpha=0.3)
                            plt.savefig(vis_path / "processing_time_dist.png", dpi=150, bbox_inches='tight')
                            plt.close()
                        
                        self.logger.info("📊 基本グラフ生成完了")
                
                except Exception as plot_error:
                    self.logger.warning(f"📊 グラフ生成エラー（処理継続）: {plot_error}")
                
                self.logger.info(f"✅ 基本可視化完了: {stats_file}")
                
            except Exception as e:
                self.logger.error(f"❌ 可視化エラー: {e}")

# 🔧 条件付きインポート - 設定とロガー（完全統合版）
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
                    "detection": "models/yolo/yolo11m.pt",
                    "pose": "models/yolo/yolo11m-pose.pt"
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
    class ImprovedYOLOAnalyzer:
        """
        YOLO11 広角カメラ分析システム（完全統合版）
        - 深度推定統合対応
        - モジュール不足完全対応
        - Stage 5/6修正完了
        - フォールバック機能完全統合
        """

    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        初期化（完全統合版）

        Args:
            config_path: 設定ファイルパス
        """
        # エラーコンテキスト設定
        if ERROR_HANDLER_AVAILABLE:
            context_manager = ErrorContext("ImprovedYOLOAnalyzer初期化", logger=logging.getLogger(__name__))
        else:
            context_manager = self._basic_context("ImprovedYOLOAnalyzer初期化")
            
        with context_manager as ctx:
            # 設定初期化
            self.config = self._initialize_config(config_path)
            
            # ロガー初期化
            self.logger = setup_logger()

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

    def _initialize_config(self, config_path: str):
        """設定初期化（完全版）"""
        depth_config_path = "configs/depth_config.yaml"
        
        self.logger.info(f"⚙️ 設定初期化開始: {config_path}")
        
        # 設定ファイルの優先順位決定
        if Path(config_path).exists():
            primary_config = config_path
            self.logger.info(f"📄 指定設定ファイル使用: {config_path}")
        elif Path(depth_config_path).exists():
            primary_config = depth_config_path
            self.logger.info(f"🔍 深度設定ファイル自動検出: {depth_config_path}")
        else:
            primary_config = config_path
            self.logger.warning(f"⚠️ 設定ファイルが見つかりません: {config_path}")

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
                frame_result = self.processor.extract_frames(video_path, frame_dir)
                
                if not frame_result.get("success", False):
                    error_msg = f"フレーム抽出失敗: {frame_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise VideoProcessingError(error_msg)

                extracted_frames = frame_result.get("extracted_frames", 0)
                self.logger.info(f"✅ Step 1完了: {extracted_frames}フレーム抽出")

                # Step 2: 検出・追跡処理
                if self.depth_enabled:
                    self.logger.info("🔍 Step 2: 深度統合検出・追跡処理開始")
                    detection_result = self.processor.run_detection_tracking_with_depth(frame_dir, video_name)
                    processing_type = "深度統合"
                else:
                    self.logger.info("👁️ Step 2: 標準検出・追跡処理開始")
                    detection_result = self.processor.run_detection_tracking(frame_dir, video_name)
                    processing_type = "標準"

                if not detection_result.get("success", False):
                    error_msg = f"{processing_type}処理失敗: {detection_result.get('error', '不明なエラー')}"
                    self.error_collector.append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise VideoProcessingError(error_msg)

                self.logger.info(f"✅ Step 2完了: {processing_type}処理")

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
                    self.analyzer.create_visualizations(detection_result, str(vis_dir))
                    self.logger.info("✅ Step 4完了: 可視化生成")
                except Exception as e:
                    self.logger.warning(f"⚠️ Step 4警告: 可視化生成エラー（処理継続）: {e}")
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
        """動画ファイル取得（完全版）"""
        try:
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
            
            self.logger.info(f"✅ 動画ファイル発見: {len(video_files)}個")
            for video_file in video_files:
                self.logger.debug(f"  📹 {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f}MB)")
                
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
    logger = setup_logger()

    logger.info("🚀 YOLO11 広角カメラ分析システム 開始（完全統合版）")
    logger.info(f"📋 実行モード: {args.mode}")
    logger.info(f"⚙️ 設定ファイル: {args.config}")
    logger.info(f"📊 詳細ログ: {'有効' if args.verbose else '無効'}")
    
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
        
        # 分析実行
        all_results = []
        successful_count = 0
        
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"📹 処理開始 ({i}/{len(video_files)}): {video_file.name}")
            
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
                    "result": result
                })
                
                if result.get("success", False):
                    successful_count += 1
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
                    "result": ResponseBuilder.error(e)
                })

        # 全体結果サマリー
        total = len(all_results)
        success_rate = (successful_count / total) * 100 if total > 0 else 0
        
        logger.info(f"📊 処理結果サマリー: {successful_count}/{total} 成功 ({success_rate:.1f}%)")
        
        # エラーレポート生成
        if args.generate_report or analyzer.error_collector:
            logger.info("📋 エラーレポート生成中...")
            error_report = analyzer.generate_error_report()
            logger.info(f"📋 エラーレポート: {error_report.get('total_errors', 0)}件のエラー")

        # 統合結果ファイル保存
        summary_result = {
            "execution_mode": args.mode,
            "config_file": args.config,
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
        
        summary_file = Path("outputs") / f"summary_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 統合結果保存: {summary_file}")

        # 処理完了
        if successful_count == total:
            logger.info("🎉 全ての動画処理が成功しました")
            print(f"\n✅ 処理完了: {successful_count}/{total} 成功 (成功率: 100%)")
            print(f"📁 結果保存先: outputs/{args.mode}/")
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