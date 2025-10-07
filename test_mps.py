import torch
import sys

print("=" * 60)
print("🍎 Apple Silicon (MPS) GPU テスト")
print("=" * 60)

print(f"\nPython: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

# MPS (Metal Performance Shaders) チェック
print("\n【GPU確認】")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("\n🚀 Apple GPU (MPS) が使えます！")

    # 簡単な演算テスト
    try:
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        y = x * 2
        print(f"\n演算テスト:")
        print(f"  入力: {x.cpu().numpy()}")
        print(f"  結果: {y.cpu().numpy()}")
        print("\n✅ GPU加速が正常に動作します！")

        # メモリ情報
        print(f"\n【推奨設定】")
        print(f"configs/default.yaml に以下を追加:")
        print(f'```yaml')
        print(f'processing:')
        print(f'  device: "mps"  # Apple GPU使用')
        print(f'  batch_size: 8  # メモリに応じて調整')
        print(f'```')
        
    except Exception as e:
        print(f"\n⚠️ GPU演算エラー: {e}")
        print("CPUモードにフォールバックします")
        print(f"\n【推奨設定】")
        print(f"configs/default.yaml に以下を追加:")
        print(f'```yaml')
        print(f'processing:')
        print(f'  device: "cpu"  # CPU使用')
        print(f'```')
else:
    print("\n💻 MPSが利用できません - CPUモードで動作")

    # 理由の診断
    print("\n【原因分析】")
    print(f"PyTorchバージョン: {torch.__version__}")

    # バージョンチェック
    import platform
    print(f"システム: {platform.system()} {platform.machine()}")

    if platform.machine() != "arm64":
        print("⚠️ このMacはIntelプロセッサです（MPSはApple Siliconのみ）")
    else:
        print("⚠️ PyTorchがMPSに対応していない可能性があります")
        print("   対処法: pip install --upgrade torch torchvision")

    print(f"\n【推奨設定】")
    print(f"configs/default.yaml に以下を追加:")
    print(f'```yaml')
    print(f'processing:')
    print(f'  device: "cpu"  # CPU使用')
    print(f'  batch_size: 4  # CPUモードでは小さめに')
    print(f'```')

# CUDA確認（参考情報）
print("\n【参考】CUDA (NVIDIA GPU) 確認")
print(f"CUDA available: {torch.cuda.is_available()}")
print("(Apple SiliconではCUDAは常にFalseです)")

print("\n" + "=" * 60)
print("テスト完了")
print("=" * 60)