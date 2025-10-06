"""
処理結果確認スクリプト
outputs/baseline/ の結果を簡易分析
"""

import pandas as pd
import os
import json
from pathlib import Path
import glob

def find_latest_baseline():
    """最新のベースライン結果ディレクトリを探す"""
    baseline_dirs = sorted(glob.glob("outputs/baseline/baseline_*"))
    if not baseline_dirs:
        return None
    return baseline_dirs[-1]

def check_results():
    """結果確認"""
    print("=" * 60)
    print("📊 処理結果確認")
    print("=" * 60)

    # 最新の結果ディレクトリを探す
    baseline_dir = find_latest_baseline()

    if not baseline_dir:
        print("❌ 結果が見つかりません")
        print("先に以下を実行してください:")
        print("  python improved_main.py --mode baseline")
        return

    print(f"\n結果ディレクトリ: {baseline_dir}\n")

    # 動画ディレクトリを探す
    video_dirs = [d for d in Path(baseline_dir).iterdir()
                if d.is_dir() and d.name not in ['reports', 'visualizations']]

    if not video_dirs:
        print("⚠️ 動画処理結果が見つかりません")
        return

    print(f"処理済み動画数: {len(video_dirs)}\n")

    # 各動画の結果を確認
    for video_dir in video_dirs:
        print("━" * 60)
        print(f"📹 動画: {video_dir.name}")
        print("━" * 60)

        # CSV確認（複数の可能性のある場所を探索）
        csv_path = None
        possible_csv_paths = [
            video_dir / "results" / "detections.csv",
            video_dir / "results" / "detections_streaming.csv",
            video_dir / "results" / "detections_enhanced.csv",
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

                # ID別検出数
                print(f"\n  ID別検出数（上位5）:")
                id_counts = df['person_id'].value_counts().head(5)
                for pid, count in id_counts.items():
                    print(f"    ID {pid}: {count}回")

                # 信頼度分布
                high_conf = len(df[df['conf'] > 0.7])
                mid_conf = len(df[(df['conf'] >= 0.5) & (df['conf'] <= 0.7)])
                low_conf = len(df[df['conf'] < 0.5])

                print(f"\n  信頼度分布:")
                print(f"    高 (>0.7): {high_conf} ({high_conf/len(df)*100:.1f}%)")
                print(f"    中 (0.5-0.7): {mid_conf} ({mid_conf/len(df)*100:.1f}%)")
                print(f"    低 (<0.5): {low_conf} ({low_conf/len(df)*100:.1f}%)")

                # バウンディングボックスサイズ
                df['width'] = df['x2'] - df['x1']
                df['height'] = df['y2'] - df['y1']
                df['area'] = df['width'] * df['height']

                print(f"\n  検出ボックスサイズ:")
                print(f"    平均面積: {df['area'].mean():.0f} px²")
                print(f"    平均幅: {df['width'].mean():.0f} px")
                print(f"    平均高さ: {df['height'].mean():.0f} px")

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

        # 統計グラフ確認
        viz_dir = video_dir / "visualizations"
        if viz_dir.exists():
            graph_files = list(viz_dir.glob("*.png"))
            if graph_files:
                print(f"✅ 統計グラフ: {len(graph_files)}枚")
                for graph in graph_files:
                    print(f"    - {graph.name}")

    # レポート確認
    print("\n" + "=" * 60)
    print("📄 レポート")
    print("=" * 60)

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
        except Exception as e:
            print(f"⚠️ JSON読み込みエラー: {e}")

    # 総括
    print("\n" + "=" * 60)
    print("✅ 結果確認完了")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. HTMLレポートをブラウザで開く")
    print("2. 可視化画像を確認:")
    print(f"   ls {baseline_dir}/*/results/vis_*.jpg")
    print("3. より大きいモデルで試す:")
    print("   - configs/default.yaml でモデルを変更")
    print("   - 再度 python improved_main.py --mode baseline")
    print("4. タイル推論を試す:")
    print("   - configs/default.yaml で tile_inference.enabled: true")

if __name__ == "__main__":
    try:
        check_results()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()