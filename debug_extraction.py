# debug_extraction_fixed.py として保存して実行
import cv2
import sys
from pathlib import Path
import logging

# improved_main.pyから必要なクラスをインポート
sys.path.append('.')

def test_improved_analyzer():
    """ImprovedYOLOAnalyzerの正しい初期化方法をテスト"""
    
    print("🔍 ImprovedYOLOAnalyzer初期化テスト")
    
    try:
        from improved_main import ImprovedYOLOAnalyzer
        
        # 🔧 正しい初期化方法を確認
        config_path = "configs/default.yaml"
        
        # 引数なしで初期化を試行
        print("🚀 引数なし初期化テスト...")
        try:
            analyzer = ImprovedYOLOAnalyzer(config_path)
            print("✅ 引数なし初期化成功")
        except Exception as e:
            print(f"❌ 引数なし初期化失敗: {e}")
            
            # 引数なし完全版で試行
            try:
                analyzer = ImprovedYOLOAnalyzer()
                print("✅ 完全引数なし初期化成功")
            except Exception as e2:
                print(f"❌ 完全引数なし初期化失敗: {e2}")
                return None
        
        # プロセッサを取得してテスト
        processor = analyzer.processor
        print(f"📦 プロセッサタイプ: {type(processor).__name__}")
        
        # フレーム抽出テスト
        test_output = Path("debug_extraction_test")
        test_output.mkdir(exist_ok=True)
        
        video_path = "videos/test.mp4"
        frame_dir = test_output / "frames"
        
        print(f"\n🚀 extract_frames実行...")
        result = processor.extract_frames(video_path, frame_dir)
        
        print(f"\n📊 extract_frames結果:")
        print(f"  success: {result.get('success', False)}")
        print(f"  extracted_frames: {result.get('extracted_frames', 0)}")
        print(f"  error: {result.get('error', 'なし')}")
        
        # 実際のファイル確認
        if frame_dir.exists():
            all_files = list(frame_dir.glob("*"))
            jpg_files = list(frame_dir.glob("*.jpg"))
            
            print(f"\n📁 実際のディレクトリ内容:")
            print(f"  全ファイル: {len(all_files)}個")
            print(f"  JPGファイル: {len(jpg_files)}個")
            
            if all_files:
                print(f"  ファイル名サンプル: {[f.name for f in all_files[:5]]}")
        
        return analyzer
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return None
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_baseline_analysis_direct():
    """run_baseline_analysisを直接テスト"""
    
    print("\n" + "="*50)
    print("🔍 run_baseline_analysis直接テスト")
    
    try:
        from improved_main import ImprovedYOLOAnalyzer
        
        # 正しい方法で初期化
        analyzer = ImprovedYOLOAnalyzer("configs/default.yaml")
        
        # ベースライン分析を直接実行
        video_path = "videos/test.mp4"
        print(f"🚀 ベースライン分析実行: {video_path}")
        
        result = analyzer.run_baseline_analysis(video_path)
        
        print(f"\n📊 ベースライン分析結果:")
        print(f"  success: {result.get('success', False)}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"  processing_info: {data.get('processing_info', {})}")
            print(f"  video_info: {data.get('video_info', {})}")
        else:
            print(f"  error: {result.get('error', 'なし')}")
            
    except Exception as e:
        print(f"❌ ベースライン分析エラー: {e}")
        import traceback
        traceback.print_exc()

def debug_step_by_step():
    """ステップバイステップデバッグ"""
    
    print("\n" + "="*50)
    print("🔍 ステップバイステップデバッグ")
    
    # Step 1: 動画ファイル確認
    video_path = Path("videos/test.mp4")
    print(f"📹 動画ファイル:")
    print(f"  パス: {video_path}")
    print(f"  存在: {video_path.exists()}")
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"  サイズ: {size_mb:.1f}MB")
    
    # Step 2: 設定ファイル確認
    config_path = Path("configs/default.yaml")
    print(f"\n⚙️ 設定ファイル:")
    print(f"  パス: {config_path}")
    print(f"  存在: {config_path.exists()}")
    
    # Step 3: 出力ディレクトリ確認
    output_base = Path("outputs/baseline/test")
    frame_dir = output_base / "frames"
    print(f"\n📁 出力ディレクトリ:")
    print(f"  ベース: {output_base}")
    print(f"  フレーム: {frame_dir}")
    print(f"  ベース存在: {output_base.exists()}")
    print(f"  フレーム存在: {frame_dir.exists()}")
    
    # 既存のフレームファイル確認
    if frame_dir.exists():
        existing_files = list(frame_dir.glob("*"))
        print(f"  既存ファイル: {len(existing_files)}個")
        if existing_files:
            print(f"  サンプル: {[f.name for f in existing_files[:5]]}")

if __name__ == "__main__":
    # 段階的にテスト実行
    debug_step_by_step()
    
    analyzer = test_improved_analyzer()
    
    if analyzer:
        test_baseline_analysis_direct()
    else:
        print("❌ 分析器の初期化に失敗したため、ベースライン分析をスキップ")