"""
YOLO11 広角カメラ分析システム セットアップスクリプト（テスト必要モデル特化版）
テストに必要なモデルのみをダウンロードし、Mediumモデルをデフォルトに設定
"""

import os
import sys
import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """プロジェクトディレクトリ構造の作成"""
    directories = [
        # データディレクトリ
        "data/raw",
        "data/processed", 
        "data/annotations",
        
        # モデルディレクトリ
        "models/yolo",
        "models/depth",
        
        # 出力ディレクトリ
        "outputs/baseline",
        "outputs/experiments",
        "outputs/visualizations",
        
        # 設定ディレクトリ
        "configs",
        
        # ログディレクトリ
        "logs",
        
        # 動画ディレクトリ
        "videos",
        
        # キャッシュ・一時ディレクトリ
        "cache",
        "temp"
    ]
    
    print("📁 ディレクトリ構造作成中...")
    created_count = 0
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"ディレクトリ作成: {directory}")
            created_count += 1
        else:
            logger.info(f"ディレクトリ既存: {directory}")
    
    print(f"✅ ディレクトリ構造作成完了: {created_count}個作成")
    return True

def download_file_with_progress(url: str, file_path: Path, timeout: int = 300) -> bool:
    """進捗表示付きファイルダウンロード"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        elapsed_time = time.time() - start_time
                        speed = downloaded / (1024 * 1024) / elapsed_time if elapsed_time > 0 else 0
                        
                        print(f"\r   進捗: {progress:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f}MB) {speed:.1f}MB/s", end="")
        
        print()  # 改行
        return True
        
    except Exception as e:
        logger.error(f"ダウンロードエラー {url}: {e}")
        if file_path.exists():
            file_path.unlink()
        return False

def download_yolo_models():
    """YOLO11モデルのダウンロード（テスト必要モデルのみ）"""
    try:
        from ultralytics import YOLO

        # 🎯 テストに必要なモデルのみに限定
        models = {
            # テスト必要な基本検出モデル
            "yolo11m.pt": {
                "description": "YOLO11 Medium - 中型検出モデル（テスト用・デフォルト）",
                "size": "約50MB",
                "use_case": "高精度検出用・テスト対象・デフォルト使用"
            },
            "yolo11x.pt": {
                "description": "YOLO11 XLarge - 最大検出モデル（テスト用）",
                "size": "約137MB",
                "use_case": "最高精度検出用・テスト対象"
            },
            
            # テスト必要なポーズ推定モデル
            "yolo11m-pose.pt": {
                "description": "YOLO11 Medium Pose - 中型ポーズ推定（テスト用・デフォルト）",
                "size": "約51MB",
                "use_case": "高精度ポーズ推定・テスト対象・デフォルト使用"
            },
            "yolo11x-pose.pt": {
                "description": "YOLO11 XLarge Pose - 最大ポーズ推定（テスト用）",
                "size": "約140MB",
                "use_case": "最高精度ポーズ推定・テスト対象"
            },
            
            # 🔧 追加: 軽量モデル（比較用・フォールバック用）
            "yolo11n.pt": {
                "description": "YOLO11 Nano - 軽量検出モデル（フォールバック用）",
                "size": "約6MB",
                "use_case": "高速処理用・フォールバック"
            },
            "yolo11n-pose.pt": {
                "description": "YOLO11 Nano Pose - 軽量ポーズ推定（フォールバック用）",
                "size": "約7MB",
                "use_case": "高速ポーズ推定・フォールバック"
            }
        }

        models_dir = Path("models/yolo")
        success_count = 0
        total_count = len(models)

        print("🔽 YOLOモデルダウンロード開始（テスト必要分のみ）...")
        print(f"対象モデル数: {total_count}")
        print("\n📊 ダウンロード対象:")
        print("   🎯 テスト必要: yolo11m.pt, yolo11x.pt, yolo11m-pose.pt, yolo11x-pose.pt")
        print("   🔧 フォールバック用: yolo11n.pt, yolo11n-pose.pt")
        print("   🚀 デフォルト使用: yolo11m.pt, yolo11m-pose.pt (Medium)")

        for model_name, info in models.items():
            model_path = models_dir / model_name

            if model_path.exists():
                file_size = model_path.stat().st_size / (1024*1024)  # MB
                logger.info(f"モデル既存: {model_name} ({file_size:.1f}MB)")
                print(f"   ✅ 既存: {model_name} ({file_size:.1f}MB)")
                success_count += 1
                continue

            try:
                print(f"\n📥 ダウンロード中: {model_name}")
                print(f"   説明: {info['description']}")
                print(f"   サイズ: {info['size']}")
                print(f"   用途: {info['use_case']}")

                # YOLOモデルのダウンロード
                model = YOLO(model_name)

                # ダウンロードされたファイルをmodels/yoloに移動
                downloaded_path = Path(model_name)
                if downloaded_path.exists():
                    shutil.move(str(downloaded_path), str(model_path))
                    file_size = model_path.stat().st_size / (1024*1024)
                    print(f"   ✅ 配置完了: {model_path} ({file_size:.1f}MB)")
                    success_count += 1
                else:
                    # ホームディレクトリなどからの検索
                    possible_locations = [
                        Path.home() / ".ultralytics" / "models" / model_name,
                        Path.cwd() / model_name,
                        Path("~/.cache/ultralytics").expanduser() / model_name
                    ]

                    found = False
                    for location in possible_locations:
                        if location.exists():
                            shutil.copy2(str(location), str(model_path))
                            file_size = model_path.stat().st_size / (1024*1024)
                            print(f"   ✅ 発見・配置完了: {model_path} ({file_size:.1f}MB)")
                            success_count += 1
                            found = True
                            break

                    if not found:
                        print(f"   ⚠️ ファイル配置に失敗: {model_name}")

            except Exception as e:
                logger.error(f"モデルダウンロードエラー {model_name}: {e}")
                print(f"   ❌ エラー: {e}")
                continue

        print(f"\n✅ YOLOモデルダウンロード完了: {success_count}/{total_count}")
        
        # テスト成功に必要な最小限のモデルをチェック
        essential_for_test = [
            "models/yolo/yolo11m.pt",
            "models/yolo/yolo11m-pose.pt"
        ]
        
        essential_count = sum(1 for model in essential_for_test if Path(model).exists())
        
        if essential_count >= 2:
            print("🎯 テスト実行に必要な最小限のモデル（Medium）が揃いました")
            
            # モデル情報ファイル作成
            create_model_info(models_dir, models)
            
            return True
        else:
            print("⚠️ テスト実行に必要なモデルが不足しています")
            print("   必要: yolo11m.pt, yolo11m-pose.pt")
            return success_count > 0

    except ImportError:
        logger.error("ultralyticsライブラリがインストールされていません")
        print("❌ ultralyticsライブラリが必要です")
        print("インストール: pip install ultralytics")
        return False
    except Exception as e:
        logger.error(f"モデルダウンロードエラー: {e}")
        print(f"❌ YOLOモデルダウンロードエラー: {e}")
        return False

def create_model_info(models_dir: Path, models_config: Dict[str, Dict[str, str]]):
    """モデル情報ファイルの作成"""
    try:
        model_info = {
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {},
            "default_models": {
                "detection": "yolo11m.pt",
                "pose": "yolo11m-pose.pt"
            },
            "model_hierarchy": {
                "nano": ["yolo11n.pt", "yolo11n-pose.pt"],
                "medium": ["yolo11m.pt", "yolo11m-pose.pt"],
                "xlarge": ["yolo11x.pt", "yolo11x-pose.pt"]
            }
        }
        
        # 実際にダウンロードされたモデルの情報を追加
        for model_name, config in models_config.items():
            model_path = models_dir / model_name
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024*1024)
                model_info["models"][model_name] = {
                    "path": str(model_path),
                    "size_mb": round(file_size, 2),
                    "description": config["description"],
                    "use_case": config["use_case"],
                    "downloaded": True
                }
            else:
                model_info["models"][model_name] = {
                    "path": str(model_path),
                    "size_mb": 0,
                    "description": config["description"],
                    "use_case": config["use_case"],
                    "downloaded": False
                }
        
        info_path = models_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"モデル情報ファイル作成: {info_path}")
        print(f"   📄 モデル情報ファイル作成: {info_path}")
        
    except Exception as e:
        logger.error(f"モデル情報ファイル作成エラー: {e}")

def download_midas_models():
    """🔍 MiDaSモデルのダウンロード（軽量版のみ）"""
    print("\n🔍 深度推定モデル（MiDaS）ダウンロード開始...")
    print("📊 テスト用軽量版のみダウンロード")

    # 🔧 テスト用に軽量版のみに限定
    midas_models = {
        "midas_v21_small_256.pt": {
            "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
            "description": "MiDaS v2.1 Small - 軽量深度推定モデル（テスト用・デフォルト）",
            "size": "約23MB",
            "input_size": "256x256",
            "use_case": "リアルタイム処理用・テスト対象・デフォルト使用",
            "checksum": None
        }
        # 🗑️ 削除: 大容量モデルはテストに不要
        # "midas_v21_384.pt": {...},
        # "dpt_large_384.pt": {...}
    }

    models_dir = Path("models/depth")
    success_count = 0
    total_count = len(midas_models)

    for model_name, info in midas_models.items():
        model_path = models_dir / model_name

        if model_path.exists():
            file_size = model_path.stat().st_size / (1024*1024)  # MB
            logger.info(f"深度モデル既存: {model_name} ({file_size:.1f}MB)")
            print(f"   ✅ 既存: {model_name} ({file_size:.1f}MB)")
            success_count += 1
            continue

        try:
            print(f"\n📥 ダウンロード中: {model_name}")
            print(f"   説明: {info['description']}")
            print(f"   サイズ: {info['size']}")
            print(f"   入力サイズ: {info['input_size']}")
            print(f"   用途: {info['use_case']}")

            # ダウンロード実行
            success = download_file_with_progress(info['url'], model_path)

            if success:
                file_size = model_path.stat().st_size / (1024*1024)
                print(f"   ✅ ダウンロード完了: {file_size:.1f}MB")
                success_count += 1
            else:
                print(f"   ❌ ダウンロード失敗")

        except Exception as e:
            logger.error(f"深度モデルダウンロードエラー {model_name}: {e}")
            print(f"   ❌ エラー: {e}")
            continue

    print(f"\n✅ MiDaSモデルダウンロード完了: {success_count}/{total_count}")

    # 深度推定設定ファイルの生成
    if success_count > 0:
        create_depth_config(models_dir)

    return success_count > 0

def create_sample_configs():
    """サンプル設定ファイルの作成（Mediumモデル仕様）"""

    # バイトトラック設定
    bytetrack_yaml = """# ByteTrack設定ファイル
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

    # 🔧 基本設定ファイル（Mediumモデル仕様）
    default_config = """# YOLO11 基本設定ファイル（Mediumモデル仕様）
processing:
  depth_estimation:
    enabled: false
  detection:
    confidence_threshold: 0.3
    iou_threshold: 0.45
  tile_inference:
    enabled: false
    tile_size: [640, 640]
    overlap_ratio: 0.2

models:
  pose: "models/yolo/yolo11m-pose.pt"    # Medium ポーズモデル（デフォルト）
  detection: "models/yolo/yolo11m.pt"    # Medium 検出モデル（デフォルト）

  # 利用可能なモデル（コメントアウト）
  # Nano (高速):     yolo11n.pt, yolo11n-pose.pt
  # Small (バランス): yolo11s.pt, yolo11s-pose.pt  
  # Medium (高精度):  yolo11m.pt, yolo11m-pose.pt ← 現在使用中
  # Large (より高精度): yolo11l.pt, yolo11l-pose.pt
  # XLarge (最高精度): yolo11x.pt, yolo11x-pose.pt

video_dir: "videos"
output_dir: "outputs"
"""

    default_config_path = Path("configs/default.yaml")
    with open(default_config_path, 'w', encoding='utf-8') as f:
        f.write(default_config)

    logger.info(f"基本設定ファイル作成（Mediumモデル仕様）: {default_config_path}")
    print(f"   📄 基本設定ファイル作成（Mediumモデル仕様）: {default_config_path}")

    # カメラパラメータ雛形
    camera_params = {
        "camera_matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
        "distortion_coefficients": [-0.2, 0.1, 0, 0, 0],
        "image_size": [640, 480],
        "calibration_date": "2024-01-01",
        "notes": "サンプルキャリブレーションファイル - 実際の値に置き換えてください"
    }

    camera_path = Path("configs/camera_params.json")
    with open(camera_path, 'w') as f:
        json.dump(camera_params, f, indent=2)

    logger.info(f"カメラパラメータファイル作成: {camera_path}")

    print("✅ 設定ファイルの作成完了（Mediumモデル仕様）")

def create_depth_config(models_dir: Path):
    """🔍 深度推定設定ファイルの生成（Mediumモデル仕様）"""
    try:
        # 利用可能なモデルの確認
        available_models = []
        for model_file in models_dir.glob("*.pt"):
            model_info = {
                "name": model_file.stem,
                "path": str(model_file),
                "size_mb": model_file.stat().st_size / (1024*1024)
            }
            available_models.append(model_info)

        # 🔧 深度推定設定の生成（Mediumモデル仕様）
        depth_config = {
            "processing": {
                "depth_estimation": {
                    "enabled": True,
                    "model_type": "midas",
                    "model_path": str(models_dir / "midas_v21_small_256.pt") if (models_dir / "midas_v21_small_256.pt").exists() else "",
                    "device": "auto",
                    "input_size": [256, 256],
                    "classroom_mode": True,
                    "camera_height": 3.0,
                    "camera_angle": 15,
                    "save_depth_maps": False,
                    "distance_zones": {
                        "front": [0.0, 3.0],
                        "middle": [3.0, 6.0],
                        "back": [6.0, 10.0],
                        "far_back": [10.0, 20.0]
                    }
                },
                "detection": {
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.45
                },
                "tile_inference": {
                    "enabled": False
                }
            },
            "models": {
                "pose": "models/yolo/yolo11m-pose.pt",      # 🔧 Medium ポーズモデル
                "detection": "models/yolo/yolo11m.pt",       # 🔧 Medium 検出モデル
                "available_depth_models": available_models
            },
            "video_dir": "videos",
            "output_dir": "outputs"
        }

        # YAML形式で保存
        config_path = Path("configs/depth_config.yaml")

        try:
            import yaml
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(depth_config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"深度推定設定ファイル作成（Mediumモデル仕様）: {config_path}")
            print(f"   📄 深度推定設定ファイル作成（Mediumモデル仕様）: {config_path}")

        except ImportError:
            # yamlが利用できない場合はJSON形式
            config_path = config_path.with_suffix('.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(depth_config, f, indent=2, ensure_ascii=False)

            logger.info(f"深度推定設定ファイル作成(JSON・Mediumモデル仕様): {config_path}")
            print(f"   📄 深度推定設定ファイル作成(JSON・Mediumモデル仕様): {config_path}")

    except Exception as e:
        logger.error(f"深度推定設定ファイル作成エラー: {e}")

def create_requirements_file():
    """requirements.txtファイルの作成"""
    requirements = """# YOLO11 広角カメラ分析システム 必要ライブラリ（テスト最適化版）

# 🎯 コアライブラリ
ultralytics>=8.0.200    # YOLO11
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=9.5.0

# 📊 データ処理・可視化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# 🔍 深度推定（軽量版のみ）
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0

# 📄 設定・データ形式
pyyaml>=6.0
tqdm>=4.65.0
requests>=2.31.0

# 🧪 テスト・開発用
psutil>=5.9.0          # パフォーマンス測定用

# 🎥 動画処理（オプション）
# ffmpeg-python>=0.2.0  # 必要に応じてコメントアウト解除

# 📊 統計分析（オプション）
# scipy>=1.10.0         # 必要に応じてコメントアウト解除
# scikit-learn>=1.3.0   # 必要に応じてコメントアウト解除
"""

    requirements_path = Path("requirements.txt")
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    logger.info(f"requirements.txtファイル作成: {requirements_path}")
    print(f"   📄 requirements.txtファイル作成: {requirements_path}")

def main():
    """メインセットアップ処理"""
    print("🚀 YOLO11 広角カメラ分析システム セットアップ開始")
    print("🎯 テスト必要モデル特化版 - Mediumモデルをデフォルトに設定")
    print("=" * 80)
    
    success_count = 0
    total_steps = 5
    
    try:
        # Step 1: ディレクトリ構造作成
        print("\n📁 Step 1/5: ディレクトリ構造作成")
        if create_directory_structure():
            success_count += 1
            print("✅ ディレクトリ構造作成完了")
        else:
            print("❌ ディレクトリ構造作成失敗")
        
        # Step 2: YOLOモデルダウンロード
        print(f"\n🎯 Step 2/5: YOLOモデルダウンロード（テスト必要分のみ）")
        if download_yolo_models():
            success_count += 1
            print("✅ YOLOモデルダウンロード完了")
        else:
            print("❌ YOLOモデルダウンロード失敗")
        
        # Step 3: 深度推定モデルダウンロード
        print(f"\n🔍 Step 3/5: 深度推定モデルダウンロード（軽量版のみ）")
        if download_midas_models():
            success_count += 1
            print("✅ 深度推定モデルダウンロード完了")
        else:
            print("❌ 深度推定モデルダウンロード失敗")
        
        # Step 4: 設定ファイル作成
        print(f"\n📄 Step 4/5: 設定ファイル作成（Mediumモデル仕様）")
        try:
            create_sample_configs()
            success_count += 1
            print("✅ 設定ファイル作成完了")
        except Exception as e:
            logger.error(f"設定ファイル作成エラー: {e}")
            print("❌ 設定ファイル作成失敗")
        
        # Step 5: requirements.txt作成
        print(f"\n📦 Step 5/5: requirements.txt作成")
        try:
            create_requirements_file()
            success_count += 1
            print("✅ requirements.txt作成完了")
        except Exception as e:
            logger.error(f"requirements.txt作成エラー: {e}")
            print("❌ requirements.txt作成失敗")
        
        # 結果報告
        print("\n" + "=" * 80)
        print("🎊 セットアップ完了報告")
        print("=" * 80)
        print(f"✅ 成功したステップ: {success_count}/{total_steps}")
        
        if success_count == total_steps:
            print("🎉 全ステップが成功しました！")
            print("\n📋 次のステップ:")
            print("1. videos/ ディレクトリに動画ファイルを配置")
            print("2. python test_system.py でシステムテスト実行")
            print("3. python improved_main.py --mode baseline --config configs/default.yaml で分析実行")
            print("\n🎯 デフォルト設定:")
            print("- 検出モデル: yolo11m.pt (Medium)")
            print("- ポーズモデル: yolo11m-pose.pt (Medium)")
            print("- 深度モデル: midas_v21_small_256.pt (軽量版)")
            
        elif success_count >= 3:
            print("⚠️ 一部ステップが失敗しましたが、基本機能は利用可能です")
            print("失敗したステップを確認し、必要に応じて個別に実行してください")
        else:
            print("❌ 重要なステップが失敗しました")
            print("エラーログを確認し、依存関係を解決してから再実行してください")
        
        print(f"\n📝 ログファイル: logs/setup.log")
        return success_count == total_steps

    except KeyboardInterrupt:
        print("\n❌ セットアップが中断されました")
        return False
    except Exception as e:
        logger.error(f"セットアップエラー: {e}")
        print(f"❌ セットアップエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)