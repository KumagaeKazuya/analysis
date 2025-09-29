# 初期環境構築

"""
YOLO11 広角カメラ分析システム セットアップスクリプト
必要なディレクトリとファイルを作成し、モデルをダウンロード
"""

import os
import sys
import subprocess
from pathlib import Path
import requests
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """必要なディレクトリ構造を作成"""
    directories = [
        "videos",
        "models",
        "outputs/frames",
        "outputs/results",
        "outputs/logs",
        "outputs/reports",
        "outputs/visualizations",
        "outputs/baseline",
        "outputs/experiments",
        "data/raw",
        "data/processed",
        "logs",
        "runs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ作成: {directory}")

    print("✅ ディレクトリ構造の作成完了")

def download_yolo_models():
    """YOLO11モデルのダウンロード"""
    try:
        from ultralytics import YOLO

        models = [
            "yolo11n.pt",          # 検出用
            "yolo11n-pose.pt",     # ポーズ推定用
            "yolo11s.pt",          # アンサンブル用
            "yolo11m.pt"           # アンサンブル用
        ]

        models_dir = Path("models")

        for model_name in models:
            model_path = models_dir / model_name

            if model_path.exists():
                logger.info(f"モデル既存: {model_name}")
                continue

            try:
                logger.info(f"モデルダウンロード中: {model_name}")
                model = YOLO(model_name)

                # modelsディレクトリに移動
                import shutil
                if Path(model_name).exists():
                    shutil.move(model_name, model_path)
                    logger.info(f"モデル配置完了: {model_path}")

            except Exception as e:
                logger.error(f"モデルダウンロードエラー {model_name}: {e}")
                continue

        print("✅ YOLOモデルのダウンロード完了")

    except ImportError:
        logger.error("ultralyticsライブラリがインストールされていません")
        return False
    except Exception as e:
        logger.error(f"モデルダウンロードエラー: {e}")
        return False

    return True

def create_sample_configs():
    """サンプル設定ファイルの作成"""

    # バイトトラック設定
    bytetrack_yaml = """
# ByteTrack設定ファイル
track_thresh: 0.5
track_buffer: 30
match_thresh: 0.8
mot20: False
"""

    bytetrack_path = Path("configs/bytetrack.yaml")
    bytetrack_path.parent.mkdir(exist_ok=True)

    with open(bytetrack_path, 'w') as f:
        f.write(bytetrack_yaml)

    logger.info(f"ByteTrack設定ファイル作成: {bytetrack_path}")

    # カメラパラメータ雛形
    camera_params = {
        "camera_matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
        "distortion_coefficients": [-0.2, 0.1, 0, 0, 0],
        "image_size": [640, 480],
        "calibration_date": "2024-01-01",
        "notes": "サンプルキャリブレーションファイル"
    }

    import json
    camera_path = Path("configs/camera_params.json")
    with open(camera_path, 'w') as f:
        json.dump(camera_params, f, indent=2)

    logger.info(f"カメラパラメータファイル作成: {camera_path}")

    print("✅ 設定ファイルの作成完了")

def install_dependencies():
    """依存関係のインストール確認"""
    try:
        import ultralytics
        import cv2
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import yaml
        import psutil
        import torch

        print("✅ 主要な依存関係が確認されました")

        # GPU利用可能性チェック
        if torch.cuda.is_available():
            print(f"🚀 GPU利用可能: {torch.cuda.device_count()}個のデバイス")
        else:
            print("💻 CPU処理モードで動作します")

        return True

    except ImportError as e:
        logger.error(f"依存関係不足: {e}")
        print("❌ requirements.txtからライブラリをインストールしてください:")
        print("pip install -r requirements.txt")
        return False

def create_sample_video_info():
    """サンプル動画配置の説明ファイル作成"""

    readme_content = """# 動画ファイルの配置について

このディレクトリ（videos/）に分析対象の動画ファイルを配置してください。

## 対応形式
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)

## 推奨仕様
- 解像度: 1280x720以上（広角カメラを想定）
- フレームレート: 30fps以下
- ファイルサイズ: 1GB以下（メモリ効率のため）

## サンプル動画
広角カメラで撮影された人物が含まれる動画を配置してください。
例：防犯カメラ映像、店舗監視カメラ、イベント会場の映像など

## ファイル名の注意
- 日本語ファイル名は避けることを推奨
- スペースを含む場合はアンダースコア（_）に置換

配置後、以下のコマンドで分析を開始できます：
python improved_main.py --mode baseline
"""

    videos_readme = Path("videos/README.md")
    with open(videos_readme, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    logger.info(f"動画配置説明ファイル作成: {videos_readme}")

def create_run_scripts():
    """実行用スクリプトの作成"""

    # Windows用バッチファイル
    windows_script = """@echo off
echo YOLO11 広角カメラ分析システム
echo.

if not exist "yolo11_env" (
    echo 仮想環境を作成中...
    python -m venv yolo11_env
)

echo 仮想環境を有効化中...
call yolo11_env\\Scripts\\activate

echo 依存関係をインストール中...
pip install -r requirements.txt

echo ベースライン分析を開始...
python improved_main.py --mode baseline --config configs/default.yaml

pause
"""

    with open("run_windows.bat", 'w', encoding='utf-8') as f:
        f.write(windows_script)

    # Linux/Mac用シェルスクリプト
    unix_script = """#!/bin/bash
echo "YOLO11 広角カメラ分析システム"
echo

if [ ! -d "yolo11_env" ]; then
    echo "仮想環境を作成中..."
    python3 -m venv yolo11_env
fi

echo "仮想環境を有効化中..."
source yolo11_env/bin/activate

echo "依存関係をインストール中..."
pip install -r requirements.txt

echo "ベースライン分析を開始..."
python improved_main.py --mode baseline --config configs/default.yaml
"""

    unix_path = Path("run_unix.sh")
    with open(unix_path, 'w', encoding='utf-8') as f:
        f.write(unix_script)

    # 実行権限付与
    try:
        import stat
        unix_path.chmod(unix_path.stat().st_mode | stat.S_IEXEC)
    except:
        pass

    logger.info("実行スクリプトを作成: run_windows.bat, run_unix.sh")

def main():
    """メインセットアップ処理"""
    print("🚀 YOLO11 広角カメラ分析システム セットアップ開始")
    print("=" * 50)

    # 1. ディレクトリ作成
    create_directory_structure()

    # 2. 依存関係チェック
    if not install_dependencies():
        print("❌ セットアップ中断: 依存関係をインストールしてください")
        return False

    # 3. モデルダウンロード
    if not download_yolo_models():
        print("⚠️ モデルダウンロードに問題がありました。手動で確認してください。")

    # 4. 設定ファイル作成
    create_sample_configs()

    # 5. 説明ファイル作成
    create_sample_video_info()

    # 6. 実行スクリプト作成
    create_run_scripts()

    print("=" * 50)
    print("✅ セットアップ完了!")
    print()
    print("次のステップ:")
    print("1. videos/ ディレクトリに分析対象の動画ファイルを配置")
    print("2. 以下のコマンドでベースライン分析を開始:")
    print("   python improved_main.py --mode baseline")
    print()
    print("タイル推論を試す場合:")
    print("   python yolopose_analyzer.py --frame-dir outputs/frames --output-dir outputs/results --tile")

    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ セットアップが中断されました")
    except Exception as e:
        print(f"❌ セットアップエラー: {e}")
        logger.error(f"詳細: {e}", exc_info=True)