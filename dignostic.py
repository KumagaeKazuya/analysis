"""
YOLO11広角カメラ分析システム 診断スクリプト
システムの問題を特定し、解決策を提示
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SystemDiagnostic:
    """システム診断クラス"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.suggestions = []

    def run_full_diagnostic(self):
        """完全診断を実行"""
        print("🔍 YOLO11システム診断開始")
        print("=" * 50)

        self.check_python_version()
        self.check_dependencies()
        self.check_file_structure()
        self.check_models()
        self.check_config_files()
        self.check_video_files()
        self.check_permissions()
        self.check_disk_space()
        self.check_gpu_availability()

        self.print_diagnostic_report()

    def check_python_version(self):
        """Python バージョンチェック"""
        print("📋 Python バージョンチェック...")

        version = sys.version_info
        if version < (3, 8):
            self.issues.append(f"Python バージョンが古すぎます: {version.major}.{version.minor} (3.8以上が必要)")
            self.suggestions.append("Python 3.8以上をインストールしてください")
        else:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

    def check_dependencies(self):
        """依存関係チェック"""
        print("📦 依存関係チェック...")

        critical_packages = {
            'ultralytics': 'pip install ultralytics',
            'cv2': 'pip install opencv-python',
            'torch': 'pip install torch',
            'numpy': 'pip install numpy',
            'pandas': 'pip install pandas',
            'matplotlib': 'pip install matplotlib',
            'yaml': 'pip install pyyaml',
            'psutil': 'pip install psutil'
        }

        missing_packages = []

        for package, install_cmd in critical_packages.items():
            try:
                importlib.import_module(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (不足)")
                missing_packages.append((package, install_cmd))

        if missing_packages:
            self.issues.append(f"{len(missing_packages)}個のパッケージが不足")
            for package, cmd in missing_packages:
                self.suggestions.append(f"{package}: {cmd}")

    def check_file_structure(self):
        """ファイル構造チェック"""
        print("📁 ファイル構造チェック...")

        required_files = [
            'improved_main.py',
            'yolopose_analyzer.py',
            'configs/default.yaml',
            'requirements.txt'
        ]

        required_dirs = [
            'videos', 'models', 'outputs', 'configs',
            'utils', 'evaluators', 'processors'
        ]

        missing_files = []
        missing_dirs = []

        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                print(f"❌ {file_path}")
            else:
                print(f"✅ {file_path}")

        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
                print(f"❌ {dir_path}/ (ディレクトリ)")
            else:
                print(f"✅ {dir_path}/")

        if missing_files:
            self.issues.append(f"必要ファイルが不足: {missing_files}")
        if missing_dirs:
            self.issues.append(f"必要ディレクトリが不足: {missing_dirs}")
            self.suggestions.append("python setup.py を実行してください")

    def check_models(self):
        """モデルファイルチェック"""
        print("🤖 モデルファイルチェック...")

        model_files = [
            'models/yolo11n.pt',
            'models/yolo11n-pose.pt'
        ]

        missing_models = []

        for model_path in model_files:
            if not Path(model_path).exists():
                missing_models.append(model_path)
                print(f"❌ {model_path}")
            else:
                # ファイルサイズチェック
                size_mb = Path(model_path).stat().st_size / (1024*1024)
                if size_mb < 1:
                    print(f"⚠️ {model_path} (サイズが小さい: {size_mb:.1f}MB)")
                    self.warnings.append(f"{model_path} のサイズが異常に小さい可能性があります")
                else:
                    print(f"✅ {model_path} ({size_mb:.1f}MB)")

        if missing_models:
            self.issues.append(f"モデルファイル不足: {missing_models}")
            self.suggestions.append("python setup.py でモデルをダウンロードしてください")

    def check_config_files(self):
        """設定ファイルチェック"""
        print("⚙️ 設定ファイルチェック...")

        config_files = [
            'configs/default.yaml',
            'configs/bytetrack.yaml'
        ]

        for config_path in config_files:
            if not Path(config_path).exists():
                print(f"❌ {config_path}")
                self.issues.append(f"設定ファイル不足: {config_path}")
            else:
                print(f"✅ {config_path}")

                # YAML構文チェック
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    print(f"✅ {config_path} (YAML構文正常)")
                except yaml.YAMLError as e:
                    print(f"⚠️ {config_path} (YAML構文エラー)")
                    self.warnings.append(f"{config_path} のYAML構文にエラー: {e}")
                except Exception as e:
                    print(f"⚠️ {config_path} (読み込みエラー)")
                    self.warnings.append(f"{config_path} の読み込みエラー: {e}")

    def check_video_files(self):
        """動画ファイルチェック"""
        print("🎥 動画ファイルチェック...")

        video_dir = Path("videos")
        if not video_dir.exists():
            print("❌ videos/ ディレクトリが存在しません")
            self.issues.append("videos/ ディレクトリが存在しません")
            return

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f"*{ext}")))

        if not video_files:
            print("⚠️ 動画ファイルが見つかりません")
            self.warnings.append("分析対象の動画ファイルを videos/ に配置してください")
        else:
            print(f"✅ {len(video_files)}個の動画ファイルを発見")

            # ファイルサイズチェック
            for video_file in video_files:
                try:
                    size_mb = video_file.stat().st_size / (1024*1024)
                    if size_mb > 1000:  # 1GB以上
                        print(f"⚠️ {video_file.name} (大容量: {size_mb:.0f}MB)")
                        self.warnings.append(f"{video_file.name} は大容量ファイルです。処理に時間がかかる可能性があります")
                    else:
                        print(f"✅ {video_file.name} ({size_mb:.0f}MB)")
                except Exception as e:
                    print(f"⚠️ {video_file.name} (ファイル情報取得エラー)")

    def check_permissions(self):
        """権限チェック"""
        print("🔒 権限チェック...")

        test_dirs = ['outputs', 'models', 'configs']

        for dir_path in test_dirs:
            try:
                test_file = Path(dir_path) / '.permission_test'
                test_file.touch()
                test_file.unlink()
                print(f"✅ {dir_path}/ (書き込み権限あり)")
            except Exception as e:
                print(f"❌ {dir_path}/ (書き込み権限なし)")
                self.issues.append(f"{dir_path}/ への書き込み権限がありません")

    def check_disk_space(self):
        """ディスク容量チェック"""
        print("💾 ディスク容量チェック...")

        try:
            import psutil
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)

            if free_gb < 1:
                print(f"❌ ディスク容量不足: {free_gb:.1f}GB")
                self.issues.append("ディスク容量が不足しています")
            elif free_gb < 5:
                print(f"⚠️ ディスク容量少: {free_gb:.1f}GB")
                self.warnings.append("ディスク容量が少なくなっています")
            else:
                print(f"✅ ディスク容量: {free_gb:.1f}GB利用可能")

        except ImportError:
            print("⚠️ psutil未インストール - ディスク容量チェックをスキップ")
        except Exception as e:
            print(f"⚠️ ディスク容量チェックエラー: {e}")

    def check_gpu_availability(self):
        """GPU利用可能性チェック"""
        print("🚀 GPU利用可能性チェック...")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✅ GPU利用可能: {gpu_count}個のデバイス")

                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

                    if gpu_memory < 2:
                        self.warnings.append(f"GPU {i} のメモリが少ない可能性があります")

            else:
                print("💻 GPU利用不可 - CPU処理で動作")
                self.warnings.append("GPU利用不可。処理が遅くなる可能性があります")

        except ImportError:
            print("⚠️ PyTorch未インストール - GPU チェックをスキップ")
        except Exception as e:
            print(f"⚠️ GPU チェックエラー: {e}")

    def check_import_modules(self):
        """主要モジュールのインポートテスト"""
        print("🔍 モジュールインポートテスト...")

        modules_to_test = [
            ('utils.config', 'Config'),
            ('evaluators.comprehensive_evaluator', 'ComprehensiveEvaluator'),
            ('processors.video_processor', 'VideoProcessor'),
            ('processors.tile_processor', 'TileProcessor')
        ]

        for module_name, class_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                getattr(module, class_name)
                print(f"✅ {module_name}.{class_name}")
            except ImportError as e:
                print(f"❌ {module_name}.{class_name} (インポートエラー)")
                self.issues.append(f"{module_name} のインポートに失敗: {e}")
            except AttributeError as e:
                print(f"❌ {module_name}.{class_name} (クラス不足)")
                self.issues.append(f"{module_name} に {class_name} クラスが見つかりません")
            except Exception as e:
                print(f"⚠️ {module_name}.{class_name} (その他エラー)")
                self.warnings.append(f"{module_name} テストエラー: {e}")

    def print_diagnostic_report(self):
        """診断レポートを出力"""
        print("\n" + "=" * 50)
        print("📋 診断レポート")
        print("=" * 50)

        if not self.issues and not self.warnings:
            print("🎉 問題は検出されませんでした!")
            print("システムは正常に動作する準備ができています。")
            print("\n次のステップ:")
            print("1. python quick_start.py でクイックスタート")
            print("2. python improved_main.py --mode baseline でベースライン分析")
            return

        if self.issues:
            print(f"❌ 重大な問題: {len(self.issues)}件")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.warnings:
            print(f"\n⚠️ 警告: {len(self.warnings)}件")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.suggestions:
            print(f"\n💡 推奨対応: {len(self.suggestions)}件")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")

        print("\n🔧 トラブルシューティング:")
        print("1. まず 'python setup.py' を実行")
        print("2. 'pip install -r requirements.txt' で依存関係をインストール")
        print("3. 問題が解決しない場合は個別に対応")

        if self.issues:
            print("\n重大な問題があるため、システムが正常に動作しない可能性があります。")
        else:
            print("\n軽微な問題のみです。システムは動作する可能性があります。")

def main():
    """メイン実行"""
    diagnostic = SystemDiagnostic()

    try:
        diagnostic.run_full_diagnostic()

        # 追加テスト: モジュールインポート
        print("\n" + "=" * 50)
        diagnostic.check_import_modules()

        # 最終レポート更新
        diagnostic.print_diagnostic_report()

    except KeyboardInterrupt:
        print("\n❌ 診断が中断されました")
    except Exception as e:
        print(f"\n❌ 診断エラー: {e}")
        logger.error(f"診断エラー詳細: {e}", exc_info=True)

if __name__ == "__main__":
    main()