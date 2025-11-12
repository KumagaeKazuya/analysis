"""
統計グラフ生成の独立テスト
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI不要モード
import matplotlib.pyplot as plt
from pathlib import Path

def test_csv_visualization():
    """CSVから統計グラフを生成するテスト"""
    
    # CSVファイルパス
    csv_path = "outputs/baseline/test/results/detections_streaming.csv"
    output_dir = Path("outputs/baseline/test/visualizations")
    
    print(f"📊 CSVファイルテスト: {csv_path}")
    
    # ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir}")
    
    try:
        # CSV読み込み
        if Path(csv_path).exists():
            print(f"✅ CSVファイル確認: {Path(csv_path).stat().st_size} bytes")
            
            df = pd.read_csv(csv_path)
            print(f"✅ CSV読み込み成功: {len(df)}行")
            print(f"📋 カラム名: {list(df.columns)}")
            print(f"📊 データサンプル:")
            print(df.head(3))
            print(f"📊 データ型:")
            print(df.dtypes)
            
            graphs_created = 0
            
            # 1. フレーム別検出数グラフ
            if 'frame' in df.columns:
                try:
                    plt.figure(figsize=(12, 6))
                    frame_counts = df['frame'].value_counts().sort_index()
                    print(f"📈 フレーム数統計: {len(frame_counts)}個のフレーム")
                    
                    plt.plot(frame_counts.index, frame_counts.values, 
                            marker='o', linewidth=2, markersize=4, color='blue')
                    plt.title('フレーム別検出数の推移', fontsize=14)
                    plt.xlabel('フレーム番号', fontsize=12)
                    plt.ylabel('検出数', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    timeline_path = output_dir / "detection_timeline.png"
                    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    graphs_created += 1
                    print(f"✅ 時系列グラフ生成: {timeline_path}")
                except Exception as e:
                    print(f"❌ 時系列グラフエラー: {e}")
            
            # 2. 信頼度分布グラフ
            if 'conf' in df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    conf_data = df['conf'].dropna()
                    print(f"📈 信頼度統計: 平均={conf_data.mean():.3f}, 範囲=[{conf_data.min():.3f}, {conf_data.max():.3f}]")
                    
                    plt.hist(conf_data, bins=30, alpha=0.7, color='green', edgecolor='black')
                    plt.axvline(conf_data.mean(), color='red', linestyle='--', 
                               label=f'平均: {conf_data.mean():.3f}')
                    plt.title('信頼度分布', fontsize=14)
                    plt.xlabel('信頼度', fontsize=12)
                    plt.ylabel('頻度', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    conf_path = output_dir / "confidence_distribution.png"
                    plt.savefig(conf_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    graphs_created += 1
                    print(f"✅ 信頼度分布グラフ生成: {conf_path}")
                except Exception as e:
                    print(f"❌ 信頼度分布グラフエラー: {e}")
            
            # 3. クラス分布グラフ
            if 'class_name' in df.columns:
                try:
                    class_counts = df['class_name'].value_counts()
                    print(f"📈 クラス統計: {list(class_counts.index)} (合計: {len(class_counts)}種類)")
                    
                    plt.figure(figsize=(12, 8))
                    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
                    plt.title('検出クラス別分布', fontsize=14)
                    plt.xlabel('クラス名', fontsize=12)
                    plt.ylabel('検出数', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    class_path = output_dir / "class_distribution.png"
                    plt.savefig(class_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    graphs_created += 1
                    print(f"✅ クラス分布グラフ生成: {class_path}")
                except Exception as e:
                    print(f"❌ クラス分布グラフエラー: {e}")
            
            # 基本統計JSON作成
            import json
            from datetime import datetime
            
            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_detections": len(df),
                "unique_frames": df['frame'].nunique() if 'frame' in df.columns else 0,
                "unique_persons": df['person_id'].nunique() if 'person_id' in df.columns else 0,
                "avg_confidence": float(df['conf'].mean()) if 'conf' in df.columns else 0.0,
                "class_distribution": df['class_name'].value_counts().to_dict() if 'class_name' in df.columns else {},
                "graphs_generated": graphs_created
            }
            
            stats_path = output_dir / "basic_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✅ 基本統計JSON生成: {stats_path}")
            
            print(f"\n🎉 テスト完了！生成ファイル数: {graphs_created + 1}")
            print(f"📁 確認コマンド: ls -la {output_dir}")
            
        else:
            print(f"❌ CSVファイルが見つかりません: {csv_path}")
            
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_visualization()