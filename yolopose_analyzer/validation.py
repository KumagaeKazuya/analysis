"""
検証機能モジュール（統一エラーハンドリング対応版）
"""

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO

# 🔧 統一エラーハンドラーからインポート
from utils.error_handler import (
    ValidationError,
    ResponseBuilder,
    handle_errors,
    validate_inputs,
    ErrorContext,
    ErrorCategory
)

logger = logging.getLogger(__name__)


@validate_inputs(model_path=lambda x: isinstance(x, str) and len(x) > 0)
@handle_errors(logger=logger, error_category=ErrorCategory.VALIDATION)
def validate_model_file(model_path: str) -> Dict[str, Any]:
    """
    モデルファイルの詳細検証（統一エラーハンドリング対応版）

    Args:
        model_path: YOLOモデルファイルのパス

    Returns:
        ResponseBuilder形式の検証結果
    """
    with ErrorContext("モデルファイル検証", logger=logger) as ctx:
        ctx.add_info("model_path", model_path)

        # ファイル存在確認
        if not os.path.exists(model_path):
            # ダウンロード提案を含める
            model_name = os.path.basename(model_path)
            suggestion = ""
            if model_name.startswith("yolo11"):
                suggestion = f"以下でダウンロード可能: python -c \"from ultralytics import YOLO; YOLO('{model_name}')\""

            return ResponseBuilder.validation_error(
                field="model_path",
                message=f"モデルファイルが存在しません: {model_path}",
                value=model_path,
                details={"suggestion": suggestion} if suggestion else None
            )

        # ファイルサイズ確認
        file_size = os.path.getsize(model_path)
        ctx.add_info("file_size_bytes", file_size)

        if file_size < 1024:  # 1KB未満
            return ResponseBuilder.validation_error(
                field="model_file_size",
                message=f"モデルファイルのサイズが異常に小さいです: {file_size} bytes",
                value=file_size,
                details={"suggestion": "モデルファイルが破損している可能性があります"}
            )

        if file_size < 1024*1024:  # 1MB未満は警告
            logger.warning(f"モデルファイルが小さい可能性: {file_size/1024:.1f} KB")

        # 読み込みテスト
        try:
            import torch
            _original = torch.load
            torch.load = lambda *a, **k: _original(*a, **{**k, 'weights_only': False})

            test_model = YOLO(model_path)
            torch.load = _original

            return ResponseBuilder.success(
                data={
                    "model_path": model_path,
                    "file_size_mb": file_size / (1024*1024),
                    "model_type": test_model.task if hasattr(test_model, 'task') else 'unknown'
                },
                message="モデルファイル検証完了"
            )

        except Exception as e:
            raise ValidationError(
                "モデル読み込みテストに失敗しました",
                details={
                    "model_path": model_path,
                    "error": str(e),
                    "suggestion": "モデルファイルが破損している可能性があります。再ダウンロードを試してください"
                },
                original_exception=e
            )


@validate_inputs(frame_dir=lambda x: isinstance(x, str))
@handle_errors(logger=logger, error_category=ErrorCategory.VALIDATION)
def validate_frame_directory(frame_dir: str) -> Dict[str, Any]:
    """
    フレームディレクトリの検証（統一エラーハンドリング対応版）

    Args:
        frame_dir: フレームディレクトリのパス

    Returns:
        ResponseBuilder形式の検証結果
    """
    with ErrorContext("フレームディレクトリ検証", logger=logger) as ctx:
        ctx.add_info("frame_dir", frame_dir)

        if not os.path.exists(frame_dir):
            return ResponseBuilder.validation_error(
                field="frame_dir",
                message=f"ディレクトリが存在しません: {frame_dir}",
                value=frame_dir,
                details={"suggestion": "フレーム抽出が正常に実行されているか確認してください"}
            )

        # 対応フォーマットのファイルを検索
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        frame_files = [
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(supported_formats)
        ]

        if len(frame_files) == 0:
            return ResponseBuilder.validation_error(
                field="frame_files",
                message="フレームファイルが見つかりません",
                value=frame_dir,
                details={
                    "supported_formats": list(supported_formats),
                    "suggestion": "対応形式でフレームが抽出されているか確認してください"
                }
            )

        # サイズ確認（サンプリング）
        total_size = 0
        corrupted_files = []
        sample_count = min(10, len(frame_files))

        for frame_file in frame_files[:sample_count]:
            file_path = os.path.join(frame_dir, frame_file)
            try:
                size = os.path.getsize(file_path)
                total_size += size

                # OpenCV読み込みテスト
                img = cv2.imread(file_path)
                if img is None:
                    corrupted_files.append(frame_file)
            except Exception as e:
                corrupted_files.append(f"{frame_file} ({str(e)[:50]})")

        # 全体サイズ推定
        if sample_count > 0:
            avg_size = total_size / sample_count
            estimated_total_mb = (avg_size * len(frame_files)) / (1024*1024)
        else:
            estimated_total_mb = 0

        ctx.add_info("frame_count", len(frame_files))
        ctx.add_info("estimated_size_mb", estimated_total_mb)

        # 警告ログ
        if corrupted_files:
            logger.warning(f"破損ファイル検出: {corrupted_files[:3]}{'...' if len(corrupted_files) > 3 else ''}")

        if estimated_total_mb > 1000:  # 1GB以上
            logger.warning(f"大量のフレームデータ: {estimated_total_mb:.1f}MB")

        return ResponseBuilder.success(
            data={
                "frame_count": len(frame_files),
                "total_size_mb": estimated_total_mb,
                "corrupted_files": corrupted_files,
                "sample_tested": sample_count,
                "supported_formats": list(supported_formats)
            },
            message="フレームディレクトリ検証完了"
        )


# 🔧 後方互換性のためのラッパー関数（必要に応じて）
def validate_model_file_legacy(model_path: str) -> Dict[str, Any]:
    """後方互換性のための従来形式レスポンス"""
    result = validate_model_file(model_path)

    # ResponseBuilder形式 → 従来形式に変換
    if result.get("success"):
        return {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data": result.get("data", {})
        }
    else:
        return {
            "valid": False,
            "errors": [result.get("message", "Unknown error")],
            "warnings": [],
            "data": {}
        }


def validate_frame_directory_legacy(frame_dir: str) -> Dict[str, Any]:
    """後方互換性のための従来形式レスポンス"""
    result = validate_frame_directory(frame_dir)

    if result.get("success"):
        data = result.get("data", {})
        return {
            "valid": True,
            "frame_count": data.get("frame_count", 0),
            "total_size_mb": data.get("total_size_mb", 0),
            "errors": [],
            "warnings": [f"破損ファイル: {data['corrupted_files'][:3]}"] if data.get("corrupted_files") else []
        }
    else:
        return {
            "valid": False,
            "frame_count": 0,
            "total_size_mb": 0,
            "errors": [result.get("message", "Unknown error")],
            "warnings": []
        }