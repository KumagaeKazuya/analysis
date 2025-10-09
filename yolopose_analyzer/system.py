"""
システムリソース管理モジュール
"""

import os
import torch
import psutil
import logging
from typing import Dict, Any
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def check_system_resources() -> Dict[str, Any]:
    """システムリソースの確認"""
    try:
        memory = psutil.virtual_memory()

        # ディスク使用量確認
        try:
            if os.name == 'nt':  # Windows
                disk = psutil.disk_usage('C:')
            else:  # Unix/Linux/Mac
                disk = psutil.disk_usage('/')
        except:
            disk = None

        # GPU検出（CUDA + MPS対応）
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

        result = {
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
            result["disk_free_gb"] = disk.free / (1024**3)

        return result
    except Exception as e:
        logger.warning(f"システムリソース確認エラー: {e}")
        return {"error": str(e)}


def safe_model_initialization(model_path: str, config: Dict[str, Any]) -> YOLO:
    """安全なモデル初期化"""
    from .validation import validate_model_file
    
    # システムリソース確認
    resources = check_system_resources()
    if "error" not in resources:
        logger.info(f"システムリソース: メモリ {resources['memory_available_gb']:.1f}GB 利用可能")

        if resources["memory_available_gb"] < 2.0:
            logger.warning("利用可能メモリが2GB未満です。処理が遅くなる可能性があります。")

        if resources["gpu_available"]:
            gpu_type = resources.get("gpu_type", "unknown")
            gpu_name = resources.get("gpu_name", "N/A")
            
            if gpu_type == "mps":
                logger.info(f"🍎 Apple Silicon GPU (MPS) 利用可能")
            elif gpu_type == "cuda":
                logger.info(f"🚀 NVIDIA GPU (CUDA) 利用可能: {resources['gpu_count']}個")
        else:
            logger.info("💻 GPU利用不可。CPUで処理します。")

    # モデルファイル検証
    validation = validate_model_file(model_path)
    if not validation["valid"]:
        error_msg = "\n".join(validation["errors"])
        suggestions = "\n".join(validation["suggestions"])
        raise ModelInitializationError(f"{error_msg}\n\n推奨対応:\n{suggestions}")

    try:
        # PyTorch 2.8対策
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

        model = YOLO(model_path)
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
            logger.warning(f"デバイス設定エラー: {e}. CPUを使用します。")
            device = "cpu"
            model.to(device)

        logger.info(f"モデル初期化完了: {model_path} on {device.upper()}")

        return model

    except Exception as e:
        logger.error(f"モデル初期化エラー: {e}")
        raise ModelInitializationError(f"モデル初期化に失敗しました: {e}")


class ModelInitializationError(Exception):
    """モデル初期化エラー"""
    pass


class ResourceExhaustionError(Exception):
    """リソース不足エラー"""
    pass