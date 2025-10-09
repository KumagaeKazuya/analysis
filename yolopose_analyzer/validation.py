"""
検証機能モジュール
"""

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def validate_model_file(model_path: str) -> Dict[str, Any]:
    """モデルファイルの詳細検証"""
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }

    if not os.path.exists(model_path):
        validation_result["errors"].append(f"モデルファイルが存在しません: {model_path}")
        
        model_name = os.path.basename(model_path)
        if model_name.startswith("yolo11"):
            validation_result["suggestions"].append(
                f"モデルダウンロード方法:\n"
                f"python -c \"from ultralytics import YOLO; YOLO('{model_name}')\""
            )
        return validation_result

    try:
        file_size = os.path.getsize(model_path)
        if file_size < 1024:
            validation_result["errors"].append(f"ファイルサイズ異常: {file_size} bytes")
            return validation_result
        elif file_size < 1024*1024:
            validation_result["warnings"].append(f"ファイルサイズ小: {file_size/1024:.1f} KB")
    except Exception as e:
        validation_result["errors"].append(f"ファイルサイズ確認エラー: {e}")
        return validation_result

    try:
        import torch
        _original = torch.load
        torch.load = lambda *a, **k: _original(*a, **{**k, 'weights_only': False})
        
        test_model = YOLO(model_path)
        torch.load = _original
        
        validation_result["valid"] = True
        validation_result["warnings"].append("モデル検証完了")
    except Exception as e:
        validation_result["errors"].append(f"モデル読み込みエラー: {e}")
        validation_result["suggestions"].append("モデルファイルが破損している可能性があります")

    return validation_result


def validate_frame_directory(frame_dir: str) -> Dict[str, Any]:
    """フレームディレクトリの検証"""
    validation_result = {
        "valid": False,
        "frame_count": 0,
        "total_size_mb": 0,
        "errors": [],
        "warnings": []
    }

    if not os.path.exists(frame_dir):
        validation_result["errors"].append(f"ディレクトリが存在しません: {frame_dir}")
        return validation_result

    try:
        frame_files = [
            f for f in os.listdir(frame_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        validation_result["frame_count"] = len(frame_files)

        if len(frame_files) == 0:
            validation_result["errors"].append("フレームファイルが見つかりません")
            validation_result["suggestions"] = [
                "フレーム抽出が正常に実行されているか確認してください",
                "対応形式: .jpg, .jpeg, .png"
            ]
            return validation_result

        # サンプルファイルでサイズ確認
        total_size = 0
        corrupted_files = []
        sample_size = min(10, len(frame_files))
        
        for frame_file in frame_files[:sample_size]:
            file_path = os.path.join(frame_dir, frame_file)
            try:
                size = os.path.getsize(file_path)
                total_size += size

                img = cv2.imread(file_path)
                if img is None:
                    corrupted_files.append(frame_file)
            except Exception as e:
                corrupted_files.append(f"{frame_file} ({e})")

        if sample_size > 0:
            avg_size = total_size / sample_size
            estimated_total_mb = (avg_size * len(frame_files)) / (1024*1024)
            validation_result["total_size_mb"] = estimated_total_mb

        if corrupted_files:
            validation_result["warnings"].append(f"破損ファイル: {corrupted_files[:3]}")

        if validation_result["total_size_mb"] > 1000:
            validation_result["warnings"].append(
                f"大量のフレーム ({validation_result['total_size_mb']:.1f}MB)"
            )

        validation_result["valid"] = True

    except Exception as e:
        validation_result["errors"].append(f"ディレクトリ検証エラー: {e}")

    return validation_result