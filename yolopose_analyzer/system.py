"""
システムリソース管理モジュール
元の yolopose_analyzer.py から抽出
"""

import os
import torch
import psutil
import logging
import traceback
from typing import Dict, Any
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 🔧 統一エラーハンドラーからインポート
from utils.error_handler import (
    ModelInitializationError,
    ResourceExhaustionError,
    ResponseBuilder,
    handle_errors,
    ErrorContext,
    ErrorCategory
)


@handle_errors(logger=logger, error_category=ErrorCategory.SYSTEM)
def check_system_resources() -> Dict[str, Any]:
    """
    システムリソースの確認

    Returns:
        システムリソース情報の辞書（ResponseBuilder形式）
    """
    with ErrorContext("システムリソース確認", logger=logger) as ctx:
        memory = psutil.virtual_memory()

        # ディスク使用量確認
        try:
            disk = psutil.disk_usage('C:' if os.name == 'nt' else '/')
        except:
            disk = None

        # GPU検出
        gpu_available = False
        gpu_type = "none"
        gpu_count = 0
        gpu_name = "N/A"

        if torch.cuda.is_available():
            gpu_available = True
            gpu_type = "cuda"
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_available = True
            gpu_type = "mps"
            gpu_count = 1
            gpu_name = "Apple Silicon GPU (MPS)"

        data = {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "cpu_count": psutil.cpu_count(),
            "gpu_available": gpu_available,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name
        }

        if disk:
            data["disk_free_gb"] = disk.free / (1024**3)

        ctx.add_info("gpu_type", gpu_type)
        ctx.add_info("memory_available", data["memory_available_gb"])

        return ResponseBuilder.success(data=data, message="システムリソース確認完了")


@handle_errors(logger=logger, error_category=ErrorCategory.INITIALIZATION)
def safe_model_initialization(model_path: str, config: Dict[str, Any]) -> YOLO:
    """
    安全なモデル初期化（統一エラーハンドリング対応版）
    
    Args:
        model_path: YOLOモデルファイルのパス
        config: 設定辞書
        
    Returns:
        初期化済みYOLOモデル
        
    Raises:
        ModelInitializationError: モデル初期化に失敗した場合
        ResourceExhaustionError: リソース不足の場合
    """
    with ErrorContext("モデル初期化", logger=logger) as ctx:
        # システムリソース確認
        resources_response = check_system_resources()
        if not resources_response.get("success", False):
            raise ResourceExhaustionError(
                "システムリソースの確認に失敗しました",
                details=resources_response.get("error", {})
            )
        
        resources = resources_response["data"]
        ctx.add_info("available_memory_gb", resources["memory_available_gb"])
        
        if resources["memory_available_gb"] < 2.0:
            logger.warning("利用可能メモリが2GB未満です")
            
        # GPU情報ログ出力
        if resources["gpu_available"]:
            gpu_type = resources.get("gpu_type", "unknown")
            gpu_name = resources.get("gpu_name", "N/A")
            
            if gpu_type == "mps":
                logger.info("🍎 Apple Silicon GPU (MPS) 利用可能")
            elif gpu_type == "cuda":
                logger.info(f"🚀 NVIDIA GPU (CUDA) 利用可能: {gpu_name}")
            else:
                logger.info(f"GPU利用可能: {gpu_type.upper()}")
        else:
            logger.info("💻 GPU利用不可。CPUで処理します")
            
        # モデルファイル検証
        from yolopose_analyzer.validation import validate_model_file
        validation = validate_model_file(model_path)
        
        if not validation.get("success", False):
            error_details = validation.get("error", {})
            raise ModelInitializationError(
                f"モデルファイル検証失敗: {error_details.get('message', 'unknown')}",
                details=error_details
            )

        # PyTorch weights_only対策
        try:
            import torch.serialization
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.PoseModel',
                'ultralytics.nn.tasks.DetectionModel'
            ])
        except AttributeError:
            pass

        # モデル読み込み
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(
            *args, **{**kwargs, 'weights_only': False}
        )

        try:
            model = YOLO(model_path)
        finally:
            torch.load = _original_torch_load

        # デバイス設定
        device = config.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        try:
            model.to(device)
        except Exception as e:
            logger.warning(f"デバイス設定エラー: {e}. CPUを使用します")
            device = "cpu"
            model.to(device)

        ctx.add_info("model_path", model_path)
        ctx.add_info("device", device)
        
        # デバイス別のログ出力
        device_emoji = {"mps": "🍎", "cuda": "🚀", "cpu": "💻"}
        logger.info(f"モデル初期化完了: {model_path} on {device_emoji.get(device, '')} {device.upper()}")
        
        return model