# 新規作成: create_depth_visualizations.py
"""
深度可視化生成スクリプト
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from utils.depth_visualization import create_depth_enhanced_visualizations

def main():
    parser = argparse.ArgumentParser(description='深度分析可視化生成')
    parser.add_argument('--baseline-dir', type=str, help='ベースライン結果ディレクトリ')
    parser.add_argument('--video-name', type=str, help='特定動画の分析')
    
    args = parser.parse_args()
    
    # 最新のベースライン結果を探す
    if not args.baseline_dir:
        import glob
        baseline_dirs = sorted(glob.glob("outputs/baseline/baseline_*"))
        if not baseline_dirs:
            print("❌ ベースライン結果が見つかりません")
            print("先に以下を実行してください:")
            print("  python improved_main.py --mode baseline --config configs/depth_config.yaml")
            return
        args.baseline_dir = baseline_dirs[-1]
    
    baseline_path = Path(args.baseline_dir)
    
    # 実験結果JSONを読み込み
    json_path = baseline_path / "experiment_results.json"
    if not json_path.exists():
        print(f"❌ 実験結果JSONが見つかりません: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    videos = results.get('videos', [])
    
    if args.video_name:
        # 特定動画のみ処理
        videos = [v for v in videos if v.get('video_name') == args.video_name]
        if not videos:
            print(f"❌ 動画が見つかりません: {args.video_name}")
            return
    
    # 各動画の深度可視化を生成
    for video_result in videos:
        video_name = video_result.get('video_name', 'unknown')
        
        # 深度統合結果の確認
        detection_results = video_result.get('detection_results', {})
        if not detection_results.get('data', {}).get('depth_enabled', False):
            print(f"⚠️ {video_name}: 深度情報が有効ではありません")
            continue
        
        print(f"🎨 {video_name}: 深度可視化生成中...")
        
        # 可視化出力ディレクトリ
        viz_dir = baseline_path / video_name / "depth_visualizations"
        
        try:
            create_depth_enhanced_visualizations(
                detection_results, 
                viz_dir, 
                video_name
            )
            print(f"✅ {video_name}: 深度可視化完了")
            
        except Exception as e:
            print(f"❌ {video_name}: 深度可視化エラー - {e}")
    
    print(f"\n🎯 深度可視化生成完了")
    print(f"結果ディレクトリ: {baseline_path}")

if __name__ == "__main__":
    main()