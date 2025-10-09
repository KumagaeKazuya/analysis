"""
メイン分析関数モジュール
"""

import os
import cv2
import csv
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .system import safe_model_initialization
from .validation import validate_frame_directory
from .memory import MemoryEfficientProcessor
from .visualization import draw_detections_ultralytics

logger = logging.getLogger(__name__)


def analyze_frames_with_tracking_memory_efficient(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """メモリ効率を考慮したフレーム解析"""
    
    if config is None:
        config = {
            "confidence_threshold": 0.3,
            "tracking_config": "bytetrack.yaml",
            "save_visualizations": True,
            "batch_size": 32,
            "max_memory_gb": 4.0,
            "streaming_output": True
        }

    os.makedirs(result_dir, exist_ok=True)
    processor = MemoryEfficientProcessor(config)

    try:
        model = safe_model_initialization(model_path, config)
        
        frame_validation = validate_frame_directory(frame_dir)
        if not frame_validation["valid"]:
            return {
                "error": "frame_validation_failed",
                "details": frame_validation["errors"],
                "success": False
            }

        # [既存の処理ロジックをここに移動]
        # 注: 長いため省略。元のコードをそのまま使用
        
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return {"error": "unexpected_error", "details": str(e), "success": False}


def analyze_frames_with_tracking_enhanced(
    frame_dir: str,
    result_dir: str,
    model_path: str = "models/yolo11n-pose.pt",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """拡張版フレーム解析（タイル推論オプション付き）"""
    
    if config and config.get("tile_inference", {}).get("enabled", False):
        from .tile_inference import analyze_frames_with_tile_inference
        return analyze_frames_with_tile_inference(frame_dir, result_dir, model_path, config)
    else:
        return analyze_frames_with_tracking_memory_efficient(
            frame_dir, result_dir, model_path, config
        )