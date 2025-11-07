"""
処理結果確認スクリプト（深度推定統合対応版）
outputs/baseline/ の結果を簡易分析

🔍 新機能:
- 深度推定結果の分析
- 深度統合CSV確認
- ゾーン別分析
- 深度可視化確認
- 比較実験結果表示
"""

import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_latest_baseline():
    """最新のベースライン結果ディレクトリを探す"""
    baseline_dirs = sorted(glob.glob("outputs/baseline/baseline_*"))
    if not baseline_dirs:
        return None
    return baseline_dirs[-1]

def find_experiment_results():
    """実験結果ディレクトリを探す"""
    experiment_dirs = sorted(glob.glob("outputs/experiments/*"))
    return experiment_dirs

def check_depth_integration(df: pd.DataFrame) -> Dict[str, Any]:
    """深度推定統合状況の確認"""
    depth_info = {
        "has_depth": False,
        "depth_success_rate": 0.0,
        "zone_distribution": {},
        "depth_stats": {},
        "depth_confidence_stats": {}
    }
    
    try:
        # 深度関連カラムの確認
        depth_columns = [col for col in df.columns if 'depth' in col.lower()]
        
        if depth_columns:
            depth_info["has_depth"] = True
            depth_info["depth_columns"] = depth_columns
            
            # 深度距離の成功率
            if 'depth_distance' in df.columns:
                valid_depth = df[df['depth_distance'] >= 0]
                depth_info["depth_success_rate"] = len(valid_depth) / len(df) if len(df) > 0 else 0
                
                if len(valid_depth) > 0:
                    depth_info["depth_stats"] = {
                        "mean": float(valid_depth['depth_distance'].mean()),
                        "std": float(valid_depth['depth_distance'].std()),
                        "min": float(valid_depth['depth_distance'].min()),
                        "max": float(valid_depth['depth_distance'].max()),
                        "median": float(valid_depth['depth_distance'].median())
                    }
            
            # ゾーン分布
            if 'depth_zone' in df.columns:
                zone_counts = df['depth_zone'].value_counts()
                total = len(df)
                depth_info["zone_distribution"] = {
                    zone: {"count": int(count), "percentage": float(count/total*100)}
                    for zone, count in zone_counts.items()
                }
            
            # 深度信頼度
            if 'depth_confidence' in df.columns:
                depth_conf = df['depth_confidence'].dropna()
                if len(depth_conf) > 0:
                    depth_info["depth_confidence_stats"] = {
                        "mean": float(depth_conf.mean()),
                        "std": float(depth_conf.std()),
                        "min": float(depth_conf.min()),
                        "max": float(depth_conf.max())
                    }
        
    except Exception as e:
        logger.warning(f"深度統合確認エラー: {e}")
    
    return depth_info

def display_depth_analysis(depth_info: Dict[str, Any]):
    """深度分析結果の表示"""
    if not depth_info["has_depth"]:
        print("  🔍 深度推定: 無効")
        return
    
    print(f"  🔍 深度推定: 有効")
    print(f"  深度成功率: {depth_info['depth_success_rate']:.1%}")
    
    # 深度統計
    if depth_info.get("depth_stats"):
        stats = depth_info["depth_stats"]
        print(f"  深度距離統計:")
        print(f"    平均: {stats['mean']:.2f}")
        print(f"    標準偏差: {stats['std']:.2f}")
        print(f"    範囲: {stats['min']:.2f} - {stats['max']:.2f}")
        print(f"    中央値: {stats['median']:.2f}")
    
    # ゾーン分布
    if depth_info.get("zone_distribution"):
        print(f"  ゾーン別分布:")
        for zone, data in depth_info["zone_distribution"].items():
            print(f"    {zone}: {data['count']}件 ({data['percentage']:.1f}%)")
    
    # 深度信頼度
    if depth_info.get("depth_confidence_stats"):
        conf_stats = depth_info["depth_confidence_stats"]
        print(f"  深度信頼度:")
        print(f"    平均: {conf_stats['mean']:.3f}")
        print(f"    範囲: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")

def check_depth_visualizations(video_dir: Path) -> Dict[str, Any]:
    """深度可視化ファイルの確認"""
    viz_info = {
        "depth_analysis_graphs": [],
        "depth_heatmaps": [],
        "depth_summaries": [],
        "depth_comparisons": []
    }
    
    try:
        # 深度分析グラフ
        depth_graphs = list(video_dir.glob("**/depth_analysis_*.png"))
        viz_info["depth_analysis_graphs"] = [str(p) for p in depth_graphs]
        
        # 深度ヒートマップ
        heatmaps = list(video_dir.glob("**/depth_heatmap_*.png"))
        viz_info["depth_heatmaps"] = [str(p) for p in heatmaps]
        
        # 深度サマリー
        summaries = list(video_dir.glob("**/depth_summary_*.txt"))
        viz_info["depth_summaries"] = [str(p) for p in summaries]
        
        # 深度比較グラフ
        comparisons = list(video_dir.glob("**/depth_comparison_*.png"))
        viz_info["depth_comparisons"] = [str(p) for p in comparisons]
        
    except Exception as e:
        logger.warning(f"深度可視化確認エラー: {e}")
    
    return viz_info

def analyze_model_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """モデル性能の詳細分析"""
    perf_info = {}
    
    try:
        # 検出品質分析
        if 'conf' in df.columns:
            # 信頼度別分布
            conf_ranges = {
                "excellent": len(df[df['conf'] > 0.9]),
                "good": len(df[(df['conf'] > 0.7) & (df['conf'] <= 0.9)]),
                "fair": len(df[(df['conf'] > 0.5) & (df['conf'] <= 0.7)]),
                "poor": len(df[df['conf'] <= 0.5])
            }
            
            total = len(df)
            perf_info["confidence_distribution"] = {
                level: {"count": count, "percentage": count/total*100}
                for level, count in conf_ranges.items()
            }
        
        # バウンディングボックス品質
        if all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2']):
            df['width'] = df['x2'] - df['x1']
            df['height'] = df['y2'] - df['y1']
            df['area'] = df['width'] * df['height']
            df['aspect_ratio'] = df['width'] / df['height']
            
            perf_info["bbox_stats"] = {
                "avg_area": float(df['area'].mean()),
                "avg_width": float(df['width'].mean()),
                "avg_height": float(df['height'].mean()),
                "avg_aspect_ratio": float(df['aspect_ratio'].mean()),
                "area_std": float(df['area'].std())
            }
        
        # 追跡品質（IDがある場合）
        if 'person_id' in df.columns and 'frame' in df.columns:
            id_stats = df.groupby('person_id').agg({
                'frame': ['count', 'min', 'max'],
                'conf': ['mean', 'std']
            }).round(3)
            
            track_lengths = df.groupby('person_id')['frame'].count()
            
            perf_info["tracking_stats"] = {
                "total_tracks": int(df['person_id'].nunique()),
                "avg_track_length": float(track_lengths.mean()),
                "max_track_length": int(track_lengths.max()),
                "min_track_length": int(track_lengths.min()),
                "long_tracks": int(len(track_lengths[track_lengths > 30])),  # 30フレーム以上
                "short_tracks": int(len(track_lengths[track_lengths < 10]))   # 10フレーム未満
            }
    
    except Exception as e:
        logger.warning(f"性能分析エラー: {e}")
    
    return perf_info

def display_performance_analysis(perf_info: Dict[str, Any]):
    """性能分析結果の表示"""
    
    # 信頼度分布
    if "confidence_distribution" in perf_info:
        print(f"\n  📊 検出品質分析:")
        conf_dist = perf_info["confidence_distribution"]
        print(f"    優秀 (>0.9): {conf_dist['excellent']['count']}件 ({conf_dist['excellent']['percentage']:.1f}%)")
        print(f"    良好 (0.7-0.9): {conf_dist['good']['count']}件 ({conf_dist['good']['percentage']:.1f}%)")
        print(f"    普通 (0.5-0.7): {conf_dist['fair']['count']}件 ({conf_dist['fair']['percentage']:.1f}%)")
        print(f"    低品質 (≤0.5): {conf_dist['poor']['count']}件 ({conf_dist['poor']['percentage']:.1f}%)")
    
    # バウンディングボックス統計
    if "bbox_stats" in perf_info:
        bbox = perf_info["bbox_stats"]
        print(f"\n  📦 バウンディングボックス統計:")
        print(f"    平均面積: {bbox['avg_area']:.0f} px²")
        print(f"    平均サイズ: {bbox['avg_width']:.0f} × {bbox['avg_height']:.0f} px")
        print(f"    平均アスペクト比: {bbox['avg_aspect_ratio']:.2f}")
        print(f"    面積標準偏差: {bbox['area_std']:.0f}")
    
    # 追跡統計
    if "tracking_stats" in perf_info:
        track = perf_info["tracking_stats"]
        print(f"\n  🎯 追跡品質分析:")
        print(f"    総追跡数: {track['total_tracks']}人")
        print(f"    平均追跡長: {track['avg_track_length']:.1f}フレーム")
        print(f"    最長追跡: {track['max_track_length']}フレーム")
        print(f"    長期追跡 (>30f): {track['long_tracks']}人")
        print(f"    短期追跡 (<10f): {track['short_tracks']}人")

def check_experiment_results():
    """実験結果の確認"""
    experiment_dirs = find_experiment_results()
    
    if not experiment_dirs:
        print("\n🧪 実験結果: なし")
        return
    
    print(f"\n🧪 実験結果: {len(experiment_dirs)}件")
    
    for exp_dir in experiment_dirs[-3:]:  # 最新3件のみ表示
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        print(f"\n  📋 実験: {exp_name}")
        
        # 実験結果JSONの確認
        result_json = exp_path / "experiment_results.json"
        if result_json.exists():
            try:
                with open(result_json, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                print(f"    実験タイプ: {results.get('experiment_type', 'N/A')}")
                print(f"    処理動画数: {len(results.get('videos', []))}")
                print(f"    実行時間: {results.get('total_processing_time', 'N/A')}")
                
                # 成功率
                if 'videos' in results:
                    successful = len([v for v in results['videos'] if v.get('success', False)])
                    success_rate = successful / len(results['videos']) * 100
                    print(f"    成功率: {success_rate:.1f}%")
                
                # 深度推定実験の特別な表示
                if results.get('experiment_type') == 'depth_analysis_comparison':
                    print(f"    🔍 深度推定比較実験")
                    if 'comparison_metrics' in results:
                        metrics = results['comparison_metrics']
                        print(f"    改善効果: {metrics.get('improvement_summary', 'N/A')}")
            
            except Exception as e:
                print(f"    ⚠️ 結果読み込みエラー: {e}")
        
        # 比較レポート
        comparison_files = list(exp_path.glob("*comparison*.html"))
        if comparison_files:
            print(f"    📊 比較レポート: {len(comparison_files)}件")

def check_model_files():
    """利用可能なモデルファイルの確認"""
    print("\n🔧 利用可能なモデル:")
    
    # YOLOモデル
    yolo_dir = Path("models/yolo")
    if yolo_dir.exists():
        yolo_models = list(yolo_dir.glob("*.pt"))
        print(f"  YOLO: {len(yolo_models)}件")
        
        # サイズ別表示
        sizes = ['n', 's', 'm', 'l', 'x']
        for size in sizes:
            size_models = [m for m in yolo_models if f'yolo11{size}' in m.name]
            if size_models:
                model_types = []
                for model in size_models:
                    if '-pose' in model.name:
                        model_types.append('ポーズ')
                    elif '-seg' in model.name:
                        model_types.append('セグメ')
                    else:
                        model_types.append('検出')
                print(f"    {size.upper()}サイズ: {', '.join(model_types)}")
    
    # 深度推定モデル
    depth_dir = Path("models/depth")
    if depth_dir.exists():
        depth_models = list(depth_dir.glob("*.pt"))
        print(f"  🔍 深度推定: {len(depth_models)}件")
        for model in depth_models:
            size_mb = model.stat().st_size / (1024*1024)
            print(f"    {model.name} ({size_mb:.1f}MB)")

def generate_quick_summary(baseline_dir: str):
    """クイックサマリーの生成"""
    try:
        # 全動画の結果を統合
        all_detections = []
        video_count = 0
        total_frames = 0
        
        for video_dir in Path(baseline_dir).iterdir():
            if not video_dir.is_dir() or video_dir.name in ['reports', 'visualizations']:
                continue
            
            video_count += 1
            
            # CSV検索
            csv_files = list(video_dir.glob("**/*detection*.csv"))
            if csv_files:
                try:
                    df = pd.read_csv(csv_files[0])
                    all_detections.append(df)
                    if 'frame' in df.columns:
                        total_frames += df['frame'].nunique()
                except Exception as e:
                    logger.warning(f"CSV読み込みエラー {video_dir.name}: {e}")
        
        if all_detections:
            combined_df = pd.concat(all_detections, ignore_index=True)
            
            print(f"\n📈 全体サマリー:")
            print(f"  処理動画数: {video_count}")
            print(f"  総フレーム数: {total_frames}")
            print(f"  総検出数: {len(combined_df)}")
            print(f"  ユニーク人数: {combined_df['person_id'].nunique()}")
            print(f"  平均信頼度: {combined_df['conf'].mean():.3f}")
            
            # 深度統合サマリー
            depth_info = check_depth_integration(combined_df)
            if depth_info["has_depth"]:
                print(f"  🔍 深度推定成功率: {depth_info['depth_success_rate']:.1%}")
                if depth_info.get("zone_distribution"):
                    main_zone = max(depth_info["zone_distribution"].items(), key=lambda x: x[1]['count'])
                    print(f"  主要ゾーン: {main_zone[0]} ({main_zone[1]['percentage']:.1f}%)")
    
    except Exception as e:
        logger.warning(f"サマリー生成エラー: {e}")

def check_results():
    """結果確認メイン処理（深度推定統合対応版）"""
    print("=" * 80)
    print("📊 YOLO11 広角カメラ分析システム - 処理結果確認（深度推定統合対応版）")
    print("=" * 80)

    # 最新の結果ディレクトリを探す
    baseline_dir = find_latest_baseline()

    if not baseline_dir:
        print("❌ 結果が見つかりません")
        print("先に以下を実行してください:")
        print("  python improved_main.py --mode baseline")
        print("  python improved_main.py --mode baseline --config configs/depth_config.yaml  # 深度推定版")
        return

    print(f"\n結果ディレクトリ: {baseline_dir}\n")

    # 利用可能なモデルの確認
    check_model_files()

    # 動画ディレクトリを探す
    video_dirs = [d for d in Path(baseline_dir).iterdir()
                if d.is_dir() and d.name not in ['reports', 'visualizations']]

    if not video_dirs:
        print("⚠️ 動画処理結果が見つかりません")
        return

    print(f"\n処理済み動画数: {len(video_dirs)}")

    # 各動画の結果を確認
    for i, video_dir in enumerate(video_dirs, 1):
        print("\n" + "━" * 80)
        print(f"📹 動画 {i}/{len(video_dirs)}: {video_dir.name}")
        print("━" * 80)

        # CSV確認（複数の可能性のある場所を探索）
        csv_path = None
        possible_csv_paths = [
            video_dir / "results" / "detections.csv",
            video_dir / "results" / "detections_streaming.csv",
            video_dir / "results" / "detections_enhanced.csv",  # 🔍 深度統合版
        ]

        for path in possible_csv_paths:
            if path.exists():
                csv_path = path
                break

        # それでも見つからない場合は再帰的に探す
        if not csv_path:
            csv_files = list(video_dir.glob("**/*detection*.csv"))
            if csv_files:
                csv_path = csv_files[0]

        if csv_path and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)

                print(f"\n✅ 検出結果CSV: {csv_path.name}")
                print(f"  総検出数: {len(df)}")
                print(f"  ユニークID: {df['person_id'].nunique()}")
                print(f"  フレーム数: {df['frame'].nunique()}")
                print(f"  平均信頼度: {df['conf'].mean():.3f}")
                print(f"  信頼度範囲: {df['conf'].min():.3f} - {df['conf'].max():.3f}")

                # 🔍 深度推定統合分析
                depth_info = check_depth_integration(df)
                display_depth_analysis(depth_info)

                # 詳細性能分析
                perf_info = analyze_model_performance(df)
                display_performance_analysis(perf_info)

                # ID別検出数（上位10に拡張）
                print(f"\n  ID別検出数（上位10）:")
                id_counts = df['person_id'].value_counts().head(10)
                for pid, count in id_counts.items():
                    avg_conf = df[df['person_id'] == pid]['conf'].mean()
                    print(f"    ID {pid}: {count}回 (平均信頼度: {avg_conf:.3f})")

            except Exception as e:
                print(f"⚠️ CSV読み込みエラー: {e}")
        else:
            print(f"❌ 検出結果CSVが見つかりません")
            print(f"   探索したパス:")
            for path in possible_csv_paths:
                print(f"     - {path}")

        # フレーム数確認
        frames_dir = video_dir / "frames"
        if frames_dir.exists():
            frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
            print(f"\n✅ 抽出フレーム: {len(frame_files)}枚")
        else:
            # 別の場所にある可能性
            frame_dirs = list(video_dir.glob("**/frames"))
            if frame_dirs:
                frame_files = list(frame_dirs[0].glob("*.jpg")) + list(frame_dirs[0].glob("*.png"))
                print(f"\n✅ 抽出フレーム: {len(frame_files)}枚")

        # 可視化画像確認
        vis_files = list(video_dir.glob("**/vis_*.jpg"))
        if vis_files:
            print(f"✅ 可視化画像: {len(vis_files)}枚")

        # 🔍 深度可視化確認
        depth_viz = check_depth_visualizations(video_dir)
        if any(depth_viz.values()):
            print(f"\n🔍 深度可視化:")
            if depth_viz["depth_analysis_graphs"]:
                print(f"  分析グラフ: {len(depth_viz['depth_analysis_graphs'])}件")
            if depth_viz["depth_heatmaps"]:
                print(f"  ヒートマップ: {len(depth_viz['depth_heatmaps'])}件")
            if depth_viz["depth_summaries"]:
                print(f"  サマリー: {len(depth_viz['depth_summaries'])}件")
            if depth_viz["depth_comparisons"]:
                print(f"  比較グラフ: {len(depth_viz['depth_comparisons'])}件")

        # 統計グラフ確認
        viz_dir = video_dir / "visualizations"
        if viz_dir.exists():
            graph_files = list(viz_dir.glob("*.png"))
            if graph_files:
                print(f"\n✅ 統計グラフ: {len(graph_files)}枚")
                for graph in graph_files[:5]:  # 最初の5件のみ表示
                    print(f"    - {graph.name}")
                if len(graph_files) > 5:
                    print(f"    ... 他 {len(graph_files)-5} 件")

    # 全体サマリー
    generate_quick_summary(baseline_dir)

    # レポート確認
    print("\n" + "=" * 80)
    print("📄 レポート")
    print("=" * 80)

    reports_dir = Path(baseline_dir) / "reports"
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*"))
        if report_files:
            print(f"\n✅ 生成されたレポート:")
            for report in report_files:
                size_kb = report.stat().st_size / 1024
                print(f"  - {report.name} ({size_kb:.1f} KB)")

            # HTMLレポートのパス表示
            html_reports = list(reports_dir.glob("*.html"))
            if html_reports:
                print(f"\n💡 HTMLレポートを開く:")
                for html in html_reports:
                    print(f"  {html.absolute()}")
                    print(f"\n  コマンド:")
                    print(f"  open {html.absolute()}")  # Mac
                    print(f"  # または")
                    print(f"  xdg-open {html.absolute()}")  # Linux
    else:
        print("\n⚠️ レポートディレクトリが見つかりません")

    # 🧪 実験結果確認
    check_experiment_results()

    # 実験結果JSON確認
    json_path = Path(baseline_dir) / "experiment_results.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            print(f"\n✅ 実験結果JSON:")
            print(f"  実験名: {results.get('experiment_name', 'N/A')}")
            print(f"  フェーズ: {results.get('phase', 'N/A')}")
            print(f"  処理動画数: {len(results.get('videos', []))}")
            
            # 深度推定関連の結果
            if 'depth_estimation' in results:
                depth_results = results['depth_estimation']
                print(f"  🔍 深度推定成功: {depth_results.get('enabled', False)}")
                if depth_results.get('success_rate'):
                    print(f"  深度成功率: {depth_results['success_rate']:.1%}")
                    
        except Exception as e:
            print(f"⚠️ JSON読み込みエラー: {e}")

    # 総括と推奨事項
    print("\n" + "=" * 80)
    print("✅ 結果確認完了")
    print("=" * 80)
    print("\n📋 次のステップ:")
    print("1. HTMLレポートをブラウザで開く")
    print("2. 可視化画像を確認:")
    print(f"   ls {baseline_dir}/*/results/vis_*.jpg")
    print("3. 🔍 深度分析グラフを確認:")
    print(f"   ls {baseline_dir}/*/visualizations/depth_*.png")
    print("4. より大きいモデルで試す:")
    print("   - configs/default.yaml でモデルを yolo11m-pose.pt に変更")
    print("   - 再度 python improved_main.py --mode baseline")
    print("5. 深度推定を有効化:")
    print("   python improved_main.py --mode baseline --config configs/depth_config.yaml")
    print("6. タイル推論を試す:")
    print("   - configs/default.yaml で tile_inference.enabled: true")
    print("7. 🧪 深度推定比較実験:")
    print("   python improved_main.py --mode experiment --experiment-type depth_analysis_comparison")
    
    print("\n🎯 性能改善のヒント:")
    print("- 信頼度が低い場合: より大きなモデル(m/l/x)を試す")
    print("- 追跡が不安定な場合: ByteTrack設定を調整")
    print("- 🔍 深度推定精度向上: より大きな深度モデルを使用")
    print("- メモリ不足の場合: タイル推論を有効化")

def main():
    """メイン実行"""
    try:
        check_results()
    except KeyboardInterrupt:
        print("\n❌ 確認処理が中断されました")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 トラブルシューティング:")
        print(f"1. 結果ディレクトリの確認:")
        print(f"   ls -la outputs/baseline/")
        print(f"2. ログファイルの確認:")
        print(f"   cat logs/latest.log")
        print(f"3. 権限の確認:")
        print(f"   ls -la outputs/")

if __name__ == "__main__":
    main()