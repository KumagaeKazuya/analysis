"""
コマンドライン実行モジュール

元の yolopose_analyzer.py の if __name__ == "__main__" 部分を
独立したモジュールとして実装しています。

実行例:
    # 通常推論
    python -m yolopose_analyzer.cli --frame-dir outputs/frames --output-dir outputs/results
    
    # タイル推論
    python -m yolopose_analyzer.cli --frame-dir outputs/frames --output-dir outputs/results --tile
    
    # 比較実験
    python -m yolopose_analyzer.cli --frame-dir outputs/frames --output-dir outputs/results --compare
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    
    # コマンドライン引数解析
    parser = argparse.ArgumentParser(
        description='YOLO11 フレーム解析（タイル推論対応）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常推論
  python yolopose_analyzer.py --frame-dir outputs/frames/test --output-dir outputs/results/test
  
  # タイル推論
  python yolopose_analyzer.py --frame-dir outputs/frames/test --output-dir outputs/results/test_tile --tile
  
  # タイル推論（適応的）
  python yolopose_analyzer.py --frame-dir outputs/frames/test --output-dir outputs/results/test_adaptive --tile --adaptive
  
  # 比較実験（通常 vs タイル）
  python yolopose_analyzer.py --frame-dir outputs/frames/test --output-dir outputs/results/comparison --compare
        """
    )
    
    parser.add_argument('--frame-dir', required=True, 
                       help='フレームディレクトリのパス')
    parser.add_argument('--output-dir', required=True, 
                       help='出力ディレクトリのパス')
    parser.add_argument('--model', default='models/yolo11n-pose.pt', 
                       help='YOLOモデルのパス (デフォルト: models/yolo11n-pose.pt)')
    
    # 推論モード選択
    parser.add_argument('--tile', action='store_true', 
                       help='タイル推論を有効化')
    parser.add_argument('--adaptive', action='store_true', 
                       help='適応的タイル推論を使用（--tile と併用）')
    parser.add_argument('--compare', action='store_true', 
                       help='比較実験を実行（通常推論 vs タイル推論）')
    
    # タイル推論パラメータ
    parser.add_argument('--tile-size', type=int, nargs=2, default=[640, 640],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='タイルサイズ (デフォルト: 640 640)')
    parser.add_argument('--overlap', type=float, default=0.2,
                       help='タイル重複率 0.0-1.0 (デフォルト: 0.2)')
    parser.add_argument('--max-tiles', type=int, default=16,
                       help='フレームあたりの最大タイル数 (デフォルト: 16)')
    
    # 検出パラメータ
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='信頼度閾値 0.0-1.0 (デフォルト: 0.3)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='デバイス選択 (デフォルト: auto)')
    
    # その他
    parser.add_argument('--no-vis', action='store_true',
                       help='可視化画像の保存を無効化')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='バッチサイズ (デフォルト: 32)')
    parser.add_argument('--sample-frames', type=int, default=20,
                       help='比較実験時のサンプルフレーム数 (デフォルト: 20)')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # モジュールインポート
    try:
        from . import (
            analyze_frames_with_tracking_memory_efficient,
            analyze_frames_with_tile_inference,
            compare_tile_vs_normal_inference,
            TILE_INFERENCE_AVAILABLE
        )
    except ImportError:
        # 直接実行された場合のフォールバック
        from yolopose_analyzer import (
            analyze_frames_with_tracking_memory_efficient,
            analyze_frames_with_tile_inference,
            compare_tile_vs_normal_inference,
            TILE_INFERENCE_AVAILABLE
        )
    
    try:
        # ==========================================
        # モード1: 比較実験
        # ==========================================
        if args.compare:
            print("=" * 60)
            print("🔬 タイル推論 vs 通常推論 比較実験")
            print("=" * 60)
            
            if not TILE_INFERENCE_AVAILABLE:
                print("❌ タイル推論モジュールが利用できません")
                print("processors/tile_processor.py が存在することを確認してください")
                sys.exit(1)
            
            results = compare_tile_vs_normal_inference(
                frame_dir=args.frame_dir,
                result_dir=args.output_dir,
                model_path=args.model,
                sample_frames=args.sample_frames
            )
            
            if results.get("success", False):
                print("\n✅ 比較実験完了")
                summary = results.get("summary", {})
                
                if "error" not in summary:
                    print(f"\n📊 結果サマリー:")
                    print(f"  サンプルフレーム数: {summary.get('total_frames', 0)}")
                    print(f"  平均検出数（通常）: {summary.get('avg_normal_detections', 0):.1f}")
                    print(f"  平均検出数（タイル）: {summary.get('avg_tile_detections', 0):.1f}")
                    print(f"  検出数改善率: {summary.get('overall_improvement_rate', 0):.1%}")
                    print(f"  平均処理時間（通常）: {summary.get('avg_normal_time', 0):.2f}秒")
                    print(f"  平均処理時間（タイル）: {summary.get('avg_tile_time', 0):.2f}秒")
                    print(f"  時間オーバーヘッド: {summary.get('avg_tile_time', 0) - summary.get('avg_normal_time', 0):.2f}秒")
                    print(f"\n📁 結果保存先: {args.output_dir}/tile_comparison.json")
                else:
                    print(f"⚠️ 比較実験で問題発生: {summary['error']}")
            else:
                print(f"\n❌ 比較実験エラー: {results.get('error', 'unknown_error')}")
                if 'details' in results:
                    print(f"詳細: {results['details']}")
        
        # ==========================================
        # モード2: タイル推論
        # ==========================================
        elif args.tile:
            print("=" * 60)
            print("🔲 タイル推論実行")
            print("=" * 60)
            
            if not TILE_INFERENCE_AVAILABLE:
                print("❌ タイル推論モジュールが利用できません")
                print("通常推論を実行します...")
                args.tile = False
            else:
                print(f"タイルサイズ: {args.tile_size[0]}x{args.tile_size[1]}")
                print(f"重複率: {args.overlap}")
                print(f"最大タイル数: {args.max_tiles}")
                if args.adaptive:
                    print("モード: 適応的タイル推論")
                else:
                    print("モード: 標準タイル推論")
                print()
                
                config = {
                    "confidence_threshold": args.confidence,
                    "device": args.device,
                    "save_visualizations": not args.no_vis,
                    "batch_size": args.batch_size,
                    "tile_inference": {
                        "enabled": True,
                        "tile_size": tuple(args.tile_size),
                        "overlap_ratio": args.overlap,
                        "use_adaptive": args.adaptive,
                        "max_tiles_per_frame": args.max_tiles
                    }
                }
                
                results = analyze_frames_with_tile_inference(
                    frame_dir=args.frame_dir,
                    result_dir=args.output_dir,
                    model_path=args.model,
                    config=config
                )
                
                if results.get("success", False):
                    print("\n✅ タイル推論完了")
                    stats = results.get("processing_stats", {})
                    tile_stats = stats.get("tile_stats", {})
                    
                    print(f"\n📊 処理統計:")
                    print(f"  総フレーム数: {stats.get('total_frames', 0)}")
                    print(f"  成功フレーム: {stats.get('successful_frames', 0)}")
                    print(f"  総検出数: {stats.get('total_detections', 0)}")
                    print(f"  ユニークID: {stats.get('unique_ids', 0)}")
                    
                    if tile_stats:
                        print(f"\n🔲 タイル統計:")
                        print(f"  平均タイル数/フレーム: {tile_stats.get('avg_tiles_per_frame', 0):.1f}")
                        print(f"  平均NMS削減率: {tile_stats.get('avg_nms_reduction', 0):.1%}")
                    
                    print(f"\n📁 結果保存先:")
                    print(f"  CSV: {results.get('csv_path', 'N/A')}")
                    print(f"  可視化: {args.output_dir}/vis_*.jpg")
                else:
                    print(f"\n❌ タイル推論エラー: {results.get('error', 'unknown_error')}")
                    if 'details' in results:
                        print(f"詳細: {results['details']}")
        
        # ==========================================
        # モード3: 通常推論
        # ==========================================
        else:
            print("=" * 60)
            print("📋 通常推論実行")
            print("=" * 60)
            print(f"信頼度閾値: {args.confidence}")
            print(f"デバイス: {args.device}")
            print(f"バッチサイズ: {args.batch_size}")
            print()
            
            config = {
                "confidence_threshold": args.confidence,
                "device": args.device,
                "save_visualizations": not args.no_vis,
                "batch_size": args.batch_size,
                "max_memory_gb": 4.0,
                "streaming_output": True
            }
            
            results = analyze_frames_with_tracking_memory_efficient(
                frame_dir=args.frame_dir,
                result_dir=args.output_dir,
                model_path=args.model,
                config=config
            )
            
            if results.get("success", False):
                print("\n✅ 通常推論完了")
                stats = results.get("processing_stats", {})
                
                print(f"\n📊 処理統計:")
                print(f"  総フレーム数: {stats.get('total_frames', 0)}")
                print(f"  成功フレーム: {stats.get('successful_frames', 0)}")
                print(f"  総検出数: {stats.get('total_detections', 0)}")
                print(f"  ユニークID: {stats.get('unique_ids', 0)}")
                print(f"  成功率: {stats.get('success_rate', 0):.1%}")
                print(f"  ピークメモリ: {stats.get('peak_memory_gb', 0):.2f}GB")
                print(f"  平均バッチ時間: {stats.get('avg_batch_time', 0):.1f}秒")
                
                print(f"\n📁 結果保存先:")
                print(f"  CSV: {results.get('csv_path', 'N/A')}")
                print(f"  可視化: {args.output_dir}/vis_*.jpg")
            else:
                print(f"\n❌ 通常推論エラー: {results.get('error', 'unknown_error')}")
                if 'details' in results:
                    print(f"詳細: {results['details']}")
        
        print("\n" + "=" * 60)
        print("処理完了")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n❌ ユーザーによって中断されました")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        logger.error("エラー詳細", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()