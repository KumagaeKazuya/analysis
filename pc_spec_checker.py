"""
PC スペック診断 & YOLO11 モデル推奨ツール
このPCで使用可能なYOLOモデルサイズを判定します
"""

import sys
import platform
import subprocess

def check_system_specs():
    """システムスペックの詳細確認"""
    print("=" * 60)
    print("🖥️  PC スペック診断")
    print("=" * 60)

    # 基本情報
    print("\n【基本情報】")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"アーキテクチャ: {platform.machine()}")
    print(f"Python バージョン: {sys.version.split()[0]}")

    # CPU情報
    print("\n【CPU情報】")
    try:
        import psutil
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()

        print(f"物理コア数: {cpu_count_physical}")
        print(f"論理コア数: {cpu_count_logical}")
        if cpu_freq:
            print(f"クロック: {cpu_freq.current:.0f} MHz (最大: {cpu_freq.max:.0f} MHz)")

        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"現在のCPU使用率: {cpu_percent}%")
    except ImportError:
        print("⚠️ psutil未インストール - 詳細情報取得不可")
        print("インストール: pip install psutil")

    # メモリ情報
    print("\n【メモリ情報】")
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        print(f"総メモリ: {memory.total / (1024**3):.1f} GB")
        print(f"使用中: {memory.used / (1024**3):.1f} GB ({memory.percent}%)")
        print(f"利用可能: {memory.available / (1024**3):.1f} GB")
        
        # スワップメモリ
        swap = psutil.swap_memory()
        print(f"スワップ: {swap.total / (1024**3):.1f} GB (使用: {swap.percent}%)")
    except ImportError:
        print("⚠️ メモリ情報取得不可")

    # GPU情報
    print("\n【GPU情報】")
    gpu_available = False
    gpu_memory_gb = 0
    gpu_name = "N/A"

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA 利用可能")
            print(f"GPU数: {gpu_count}")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = props.total_memory / (1024**3)

                print(f"\nGPU {i}: {gpu_name}")
                print(f"  メモリ: {gpu_memory_gb:.1f} GB")
                print(f"  コンピュート能力: {props.major}.{props.minor}")
                print(f"  マルチプロセッサ数: {props.multi_processor_count}")

                # 現在のメモリ使用状況
                if torch.cuda.is_available():
                    try:
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(f"  使用中メモリ: {allocated:.2f} GB")
                        print(f"  予約済みメモリ: {reserved:.2f} GB")
                    except:
                        pass
        else:
            print("❌ CUDA 利用不可")
            print("CPUモードで動作します")

            # CUDAがインストールされているか確認
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"PyTorch CUDA バージョン: {cuda_version}")
                print("⚠️ GPU ドライバーの確認が必要です")

    except ImportError:
        print("⚠️ PyTorch未インストール - GPU情報取得不可")
        print("インストール: pip install torch")

    # ディスク情報
    print("\n【ディスク情報】")
    try:
        import psutil
        disk = psutil.disk_usage('.')

        print(f"総容量: {disk.total / (1024**3):.1f} GB")
        print(f"使用中: {disk.used / (1024**3):.1f} GB ({disk.percent}%)")
        print(f"空き容量: {disk.free / (1024**3):.1f} GB")
    except:
        print("⚠️ ディスク情報取得不可")

    return {
        "gpu_available": gpu_available,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_name": gpu_name,
        "ram_gb": memory.total / (1024**3) if 'memory' in locals() else 0,
        "ram_available_gb": memory.available / (1024**3) if 'memory' in locals() else 0,
        "cpu_cores": cpu_count_physical if 'cpu_count_physical' in locals() else 0
    }

def recommend_yolo_model(specs):
    """スペックに基づいてYOLOモデルを推奨"""
    print("\n" + "=" * 60)
    print("🎯 YOLO11 モデル推奨")
    print("=" * 60)

    gpu_available = specs["gpu_available"]
    gpu_memory_gb = specs["gpu_memory_gb"]
    ram_available_gb = specs["ram_available_gb"]

    print(f"\n判定条件:")
    print(f"  GPU: {'✅ あり' if gpu_available else '❌ なし'}")
    if gpu_available:
        print(f"  GPU メモリ: {gpu_memory_gb:.1f} GB")
    print(f"  利用可能RAM: {ram_available_gb:.1f} GB")

    # モデル情報
    models = {
        "yolo11n": {"size_mb": 2.6, "gpu_ram": 1, "system_ram": 2, "speed": "最速", "accuracy": "低"},
        "yolo11s": {"size_mb": 9.4, "gpu_ram": 2, "system_ram": 3, "speed": "速い", "accuracy": "中低"},
        "yolo11m": {"size_mb": 20.1, "gpu_ram": 4, "system_ram": 6, "speed": "中", "accuracy": "中高"},
        "yolo11l": {"size_mb": 25.3, "gpu_ram": 6, "system_ram": 8, "speed": "遅い", "accuracy": "高"},
        "yolo11x": {"size_mb": 56.9, "gpu_ram": 8, "system_ram": 12, "speed": "最遅", "accuracy": "最高"},
    }

    print("\n【モデル推奨結果】\n")

    recommended = []
    possible = []
    not_recommended = []

    for model_name, model_info in models.items():
        status = ""
        reason = []

        if gpu_available:
            # GPU使用時の判定
            if gpu_memory_gb >= model_info["gpu_ram"] * 1.5:  # 1.5倍の余裕
                status = "✅ 推奨"
                recommended.append(model_name)
            elif gpu_memory_gb >= model_info["gpu_ram"]:
                status = "⚠️ 可能（ギリギリ）"
                possible.append(model_name)
                reason.append("GPU メモリが少なめ")
            else:
                status = "❌ 非推奨"
                not_recommended.append(model_name)
                reason.append(f"GPU メモリ不足 (必要: {model_info['gpu_ram']}GB)")
        else:
            # CPU使用時の判定
            if ram_available_gb >= model_info["system_ram"] * 1.5:
                if model_name in ["yolo11n", "yolo11s"]:
                    status = "✅ 推奨（CPU）"
                    recommended.append(model_name)
                elif model_name == "yolo11m":
                    status = "⚠️ 可能（遅い）"
                    possible.append(model_name)
                    reason.append("CPUで処理が遅い")
                else:
                    status = "❌ 非推奨"
                    not_recommended.append(model_name)
                    reason.append("CPUでは非常に遅い")
            elif ram_available_gb >= model_info["system_ram"]:
                if model_name in ["yolo11n", "yolo11s"]:
                    status = "⚠️ 可能（ギリギリ）"
                    possible.append(model_name)
                    reason.append("メモリが少なめ")
                else:
                    status = "❌ 非推奨"
                    not_recommended.append(model_name)
                    reason.append("メモリ不足")
            else:
                status = "❌ 非推奨"
                not_recommended.append(model_name)
                reason.append("メモリ不足")

        # 結果表示
        print(f"{status} {model_name}.pt")
        print(f"    サイズ: {model_info['size_mb']} MB")
        print(f"    必要GPU RAM: {model_info['gpu_ram']} GB / システムRAM: {model_info['system_ram']} GB")
        print(f"    速度: {model_info['speed']} / 精度: {model_info['accuracy']}")
        if reason:
            print(f"    理由: {', '.join(reason)}")
        print()

    # サマリー
    print("=" * 60)
    print("📋 推奨サマリー")
    print("=" * 60)

    if recommended:
        print(f"\n✅ 推奨モデル: {', '.join(recommended)}")
        print(f"   → configs/default.yaml で設定: {recommended[-1]}.pt")

    if possible:
        print(f"\n⚠️ 使用可能（制限あり）: {', '.join(possible)}")

    if not_recommended:
        print(f"\n❌ 非推奨: {', '.join(not_recommended)}")

    # 具体的な推奨
    print("\n" + "=" * 60)
    print("💡 具体的な推奨設定")
    print("=" * 60)

    if gpu_available:
        if gpu_memory_gb >= 8:
            best_model = "yolo11x"
            print(f"\n🎯 最適: {best_model}.pt（最高精度）")
        elif gpu_memory_gb >= 6:
            best_model = "yolo11l"
            print(f"\n🎯 最適: {best_model}.pt（高精度）")
        elif gpu_memory_gb >= 4:
            best_model = "yolo11m"
            print(f"\n🎯 最適: {best_model}.pt（バランス型）")
        else:
            best_model = "yolo11s"
            print(f"\n🎯 最適: {best_model}.pt（軽量・高速）")
    else:
        if ram_available_gb >= 6:
            best_model = "yolo11s"
            print(f"\n🎯 最適: {best_model}.pt（CPU使用・バランス）")
        else:
            best_model = "yolo11n"
            print(f"\n🎯 最適: {best_model}.pt（CPU使用・最軽量）")

    print(f"\nconfigs/default.yaml に以下を設定:")
    print("```yaml")
    print("models:")
    print(f'  detection: "{best_model}.pt"')
    print(f'  pose: "{best_model}-pose.pt"')
    print("```")

    # パフォーマンス最適化のヒント
    print("\n" + "=" * 60)
    print("⚙️ パフォーマンス最適化のヒント")
    print("=" * 60)

    if gpu_available:
        print("\n✅ GPU利用可能 - 以下の最適化を推奨:")
        print("  1. 半精度演算（FP16）を有効化")
        print("     processing.gpu_settings.use_half_precision: true")
        print("  2. バッチサイズを調整")
        print(f"     processing.gpu_settings.batch_size: {max(4, int(gpu_memory_gb / 2))}")
    else:
        print("\n⚠️ CPU使用 - 以下で高速化:")
        print("  1. フレーム数を制限")
        print("     processing.frame_sampling.max_frames: 500")
        print("  2. フレーム間隔を広げる")
        print("     processing.frame_sampling.interval_sec: 5")
        print("  3. 信頼度閾値を上げる")
        print("     processing.detection.confidence_threshold: 0.5")

    if ram_available_gb < 8:
        print("\n⚠️ メモリ不足の可能性 - 以下を推奨:")
        print("  1. 可視化を無効化")
        print("     visualization.save_annotated_frames: false")
        print("  2. ストリーミング処理を有効化")
        print("     processing.streaming_output: true")

def main():
    """メイン実行"""
    try:
        specs = check_system_specs()
        recommend_yolo_model(specs)

        print("\n" + "=" * 60)
        print("✅ 診断完了")
        print("=" * 60)
        print("\n次のステップ:")
        print("1. 推奨モデルを configs/default.yaml に設定")
        print("2. python setup.py でモデルをダウンロード")
        print("3. python improved_main.py --mode baseline で実行")

    except Exception as e:
        print(f"\n❌ エラー: {e}")
        print("\n必要なパッケージをインストール:")
        print("pip install psutil torch")

if __name__ == "__main__":
    main()