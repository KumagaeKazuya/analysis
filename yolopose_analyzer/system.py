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


class ModelInitializationError(Exception):
    """モデル初期化エラー"""
    pass


class ResourceExhaustionError(Exception):
    """リソース不足エラー"""
    pass


def check_system_resources() -> Dict[str, Any]:
    """
    システムリソースの確認
    
    Returns:
        システムリソース情報の辞書
    """
    try:
        memory = psutil.virtual_memory()

        # ディスク使用量確認（クロスプラットフォーム対応）
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

        # CUDA (NVIDIA GPU) チェック
        if torch.cuda.is_available():
            gpu_available = True
            gpu_type = "cuda"
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
        # MPS (Apple Silicon GPU) チェック
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
    """
    安全なモデル初期化
    
    Args:
        model_path: YOLOモデルファイルのパス
        config: 設定辞書（device等を含む）
        
    Returns:
        初期化済みYOLOモデル
        
    Raises:
        ModelInitializationError: モデル初期化に失敗した場合
    """
    # システムリソース確認
    resources = check_system_resources()
    if "error" not in resources:
        logger.info(f"システムリソース: メモリ {resources['memory_available_gb']:.1f}GB 利用可能")

        if resources["memory_available_gb"] < 2.0:
            logger.warning("利用可能メモリが2GB未満です。処理が遅くなる可能性があります。")

        # GPU情報の出力
        if resources["gpu_available"]:
            gpu_type = resources.get("gpu_type", "unknown")
            gpu_name = resources.get("gpu_name", "N/A")
            gpu_count = resources.get("gpu_count", 0)

            if gpu_type == "mps":
                logger.info(f"🍎 Apple Silicon GPU (MPS) 利用可能")
            elif gpu_type == "cuda":
                logger.info(f"🚀 NVIDIA GPU (CUDA) 利用可能: {gpu_count}個のデバイス")
                if gpu_name != "N/A":
                    logger.info(f"   GPU名: {gpu_name}")
            else:
                logger.info(f"GPU利用可能: {gpu_type.upper()}")
        else:
            logger.info("💻 GPU利用不可。CPUで処理します。")

    # モデルファイル検証
    from .validation import validate_model_file
    validation = validate_model_file(model_path)
    
    if not validation["valid"]:
        error_msg = "\n".join(validation["errors"])
        suggestions = "\n".join(validation["suggestions"])
        raise ModelInitializationError(f"{error_msg}\n\n推奨対応:\n{suggestions}")

    if validation["warnings"]:
        for warning in validation["warnings"]:
            logger.info(warning)

    # PyTorch 2.6以降のweights_only対策
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.PoseModel',
            'ultralytics.nn.tasks.DetectionModel'
        ])
    except AttributeError:
        pass

    try:
        # PyTorch 2.8対策: weights_onlyを強制的にFalseに
        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(
            *args, **{**kwargs, 'weights_only': False}
        )

        # モデル初期化
        model = YOLO(model_path)

        # torch.loadを元に戻す
        torch.load = _original_torch_load

        # デバイス設定
        device = config.get("device", "auto")
        
        if device == "auto":
            # 自動検出
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

        # デバイス情報の出力
        if device == "mps":
            logger.info(f"モデル初期化完了: {model_path} on 🍎 {device.upper()}")
        elif device == "cuda":
            logger.info(f"モデル初期化完了: {model_path} on 🚀 {device.upper()}")
        else:
            logger.info(f"モデル初期化完了: {model_path} on 💻 {device.upper()}")

        # GPU使用時の追加設定
        if device == "cuda" and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU メモリ: {gpu_memory:.1f}GB")

                # 半精度演算設定（オプション）
                if config.get("use_half_precision", False):
                    model.half()
                    logger.info("半精度演算を有効化")
            except Exception as e:
                logger.warning(f"GPU設定警告: {e}")

        return model

    except Exception as e:
        logger.error(f"モデル初期化エラー: {e}")
        logger.error(f"詳細: {traceback.format_exc()}")
        raise ModelInitializationError(f"モデル初期化に失敗しました: {e}")