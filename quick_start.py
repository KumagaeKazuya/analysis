"""
YOLO11広角カメラ分析システム クイックスタート
初回実行用の簡単なスクリプト
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system():
    """システム要件チェック"""
    print("🔍 システム要件をチェック中...")

    # Python バージョンチェック
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8以上が必要です。現在: {python_version.major}.{python_version.minor}")
        return False

    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

    # 依存ライブラリチェック
    required_packages = [
        'ultralytics', 'opencv-python', 'torch', 'numpy',
        'pandas', 'matplotlib', 'pyyaml', 'psutil'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 以下のパッケージが不足しています: {', '.join(missing_packages)}")
        print("以下のコマンドでインストールしてください:")
        print("pip install -r requirements.txt")
        return False

    print("✅ 依存パッケージが揃っています")
    return True

def check_files():
    """必要ファイルの存在確認"""
    print("📁 必要ファイルをチェック中...")

    essential_files = [
        'improved_main.py',
        'yolopose_analyzer.py',
        'configs/default.yaml',
        'requirements.txt'
    ]

    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ 以下のファイルが不足しています: {missing_files}")
        return False

    print("✅ 必要ファイルが揃っています")
    return True

def setup_directories():
    """必要ディレクトリの作成"""
    print("📂 ディレクトリを作成中...")

    directories = [
        "videos", "models", "outputs/frames", "outputs/results", 
        "outputs/logs", "outputs/reports", "outputs/visualizations"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ ディレクトリ作成完了")

def check_videos():
    """動画ファイルの確認"""
    print("🎥 動画ファイルをチェック中...")

    video_dir = Path("videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))

    if not video_files:
        print("⚠️ videos/ ディレクトリに動画ファイルがありません")
        print("分析対象の動画ファイル（.mp4, .avi, .mov）を配置してください")
        return False

    print(f"✅ {len(video_files)}個の動画ファイルを発見")
    for video in video_files:
        print(f"  - {video.name}")

    return True

def download_models():
    """YOLOモデルのダウンロード"""
    print("📥 YOLOモデルをダウンロード中...")

    try:
        from ultralytics import YOLO

        models = ["yolo11n.pt", "yolo11n-pose.pt"]
        models_dir = Path("models")

        for model_name in models:
            model_path = models_dir / model_name

            if model_path.exists():
                print(f"✅ モデル既存: {model_name}")
                continue

            print(f"⬇️ ダウンロード中: {model_name}")
            model = YOLO(model_name)

            # modelsディレクトリに移動
            if Path(model_name).exists():
                Path(model_name).rename(model_path)
                print(f"✅ モデル配置: {model_path}")

        return True

    except Exception as e:
        print(f"❌ モデルダウンロードエラー: {e}")
        return False

def run_demo():
    """デモンストレーション実行"""
    print("🚀 デモンストレーション実行...")

    try:
        # 簡単なベースライン実行
        cmd = [sys.executable, "improved_main.py", "--mode", "baseline", "--config", "configs/default.yaml"]

        print("実行コマンド:", " ".join(cmd))
        print("=" * 50)

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print("=" * 50)
            print("✅ デモンストレーション完了!")
            return True
        else:
            print("❌ デモンストレーション実行エラー")
            return False

    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")
        return False

def show_next_steps():
    """次のステップを表示"""
    print("\n🎯 次のステップ:")
    print("1. outputs/reports/ でベースライン結果を確認")
    print("2. タイル推論を試す:")
    print("   python yolopose_analyzer.py --frame-dir outputs/frames --output-dir outputs/tile_results --tile")
    print("3. 改善実験を実行:")
    print("   python improved_main.py --mode experiment --experiment-type tile_comparison")
    print("\n📚 詳細な使用方法:")
    print("   README.md または configs/default.yaml の設定を確認してください")

def main():
    """メイン処理"""
    print("🚀 YOLO11 広角カメラ分析システム クイックスタート")
    print("=" * 60)

    # 1. システムチェック
    if not check_system():
        print("❌ システム要件を満たしていません")
        return False

    # 2. ファイルチェック
    if not check_files():
        print("❌ 必要ファイルが不足しています")
        return False

    # 3. ディレクトリ作成
    setup_directories()

    # 4. モデルダウンロード
    if not download_models():
        print("⚠️ モデルダウンロードに問題がありました。手動で確認してください。")

    # 5. 動画ファイル確認
    if not check_videos():
        print("\n⚠️ 動画ファイルを配置してから再実行してください")
        print("配置場所: videos/ ディレクトリ")
        print("対応形式: .mp4, .avi, .mov")
        return False

    # 6. デモ実行
    print("\n" + "=" * 60)
    user_input = input("デモンストレーションを実行しますか？ (y/n): ")

    if user_input.lower() in ['y', 'yes', 'はい']:
        if run_demo():
            show_next_steps()
        else:
            print("デモ実行に失敗しました。ログを確認してください。")
    else:
        print("✅ セットアップ完了! 準備ができたら以下を実行してください:")
        print("python improved_main.py --mode baseline")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 クイックスタート完了!")
        else:
            print("\n❌ セットアップに問題がありました")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ 中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        logger.error(f"クイックスタートエラー: {e}", exc_info=True)
        sys.exit(1)