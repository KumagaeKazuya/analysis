import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib
import os
import argparse

# 日本語フォント設定（Macの場合はヒラギノ Sans）
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

def plot_detection_histogram(df, frame_col, pid, interval, out_dir):
    """
    指定person_idの検出数ヒストグラムを作成
    第3層フィルタリング用の視覚的確認に使用
    """
    df_pid = df[df['person_id'] == pid].copy()
    if df_pid.empty:
        print(f"person_id={pid}: データなし")
        return
    # frame列が "frame_000200.jpg" のような場合、数値部分だけ抽出
    df_pid[frame_col] = df_pid[frame_col].astype(str).str.extract(r'(\d+)')[0]
    frames = pd.to_numeric(df_pid[frame_col], errors='coerce').dropna().astype(int)
    if frames.empty:
        print(f"person_id={pid}: フレームデータなし")
        return
    frame_max = frames.max()
    # 区間を 40-1200, 1240-2400 ... の形に
    bins = np.arange(40, frame_max + interval, interval)
    bins = np.insert(bins, 0, 0)  # 先頭に0を追加
    counts, edges = np.histogram(frames, bins=bins)
    tick_label = []
    for i in range(len(counts)):
        if i == 0:
            tick_label.append(f"0-{int(edges[1]-1)}")
        else:
            tick_label.append(f"{int(edges[i])}-{int(edges[i+1]-1)}")
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(counts)), counts, tick_label=tick_label)
    plt.xlabel("フレーム区間")
    plt.ylabel("検出数")
    plt.title(f"person_id={pid} の{interval}フレーム毎の検出数")
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"detection_hist_pid{pid}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"person_id={pid} のヒストグラムを {out_path} に保存しました。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true', help='正規化ありで実行')
    parser.add_argument('--interval', type=int, default=1200, help='ヒストグラムの区間幅（フレーム数）')
    args = parser.parse_args()

    csv_file = input("CSVファイル名を入力してください（例: 6point_metrics.csv）: ")
    df = pd.read_csv(csv_file)
    frames_dir = input("フレーム画像フォルダを指定してください（例: ./frames）: ")
    id_col = 'person_id'
    frame_col = 'frame'
    left_ear_x_col = 'left_ear_x'
    right_ear_x_col = 'right_ear_x'
    left_ear_y_col = 'left_ear_y'
    right_ear_y_col = 'right_ear_y'

    # --- 正規化カラム名の自動判定 ---
    if args.normalize:
        if 'shoulder_width_normalized_linear' in df.columns:
            shoulder_col = 'shoulder_width_normalized_linear'
            shoulder_x_cols = ['left_shoulder_x_normalized_linear', 'right_shoulder_x_normalized_linear']
            shoulder_y_cols = ['left_shoulder_y_normalized_linear', 'right_shoulder_y_normalized_linear']
            ear_x_cols = ['left_ear_x_normalized_linear', 'right_ear_x_normalized_linear']
            ear_y_cols = ['left_ear_y_normalized_linear', 'right_ear_y_normalized_linear']
        elif 'shoulder_width_normalized_exp' in df.columns:
            shoulder_col = 'shoulder_width_normalized_exp'
            shoulder_x_cols = ['left_shoulder_x_normalized_exp', 'right_shoulder_x_normalized_exp']
            shoulder_y_cols = ['left_shoulder_y_normalized_exp', 'right_shoulder_y_normalized_exp']
            ear_x_cols = ['left_ear_x_normalized_exp', 'right_ear_x_normalized_exp']
            ear_y_cols = ['left_ear_y_normalized_exp', 'right_ear_y_normalized_exp']
        else:
            raise ValueError("正規化肩幅カラムが見つかりません")
    else:
        if 'shoulder_width' in df.columns:
            shoulder_col = 'shoulder_width'
            shoulder_x_cols = ['left_shoulder_x', 'right_shoulder_x']
            shoulder_y_cols = ['left_shoulder_y', 'right_shoulder_y']
            ear_x_cols = ['left_ear_x', 'right_ear_x']
            ear_y_cols = ['left_ear_y', 'right_ear_y']
        else:
            raise ValueError("肩幅カラムが見つかりません")

    if args.normalize:
        if 'ear_distance_normalized_linear' in df.columns:
            ear_dist_col = 'ear_distance_normalized_linear'
        elif 'ear_distance_normalized_exp' in df.columns:
            ear_dist_col = 'ear_distance_normalized_exp'
        else:
            ear_dist_col = None
    else:
        ear_dist_col = None

    # 出力フォルダ作成（タイムスタンプ付き）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"length_analysis_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # サブフォルダ作成
    subfolders = {
        "csv": os.path.join(out_dir, "csv"),
        "plots": os.path.join(out_dir, "plots"),
        "normalized": os.path.join(out_dir, "normalized_pose"),
        "raw": os.path.join(out_dir, "raw_pose"),
        "hist": os.path.join(out_dir, "hist")
    }
    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)

    # 元CSVの絶対パスを記録
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"元CSVファイル: {os.path.abspath(csv_file)}\n")
        f.write(f"作成日時: {timestamp}\n")
        f.write(f"フレーム画像フォルダ: {os.path.abspath(frames_dir)}\n")
        f.write(f"正規化: {'あり' if args.normalize else 'なし'}\n")

    total_frames = df[frame_col].nunique()
    summary_list = []
    all_persons_rows = []
    all_graph_rows = []

    # === 第1層フィルタリング: 信頼度閾値 0.5 ===
    conf_threshold = 0.50

    for pid in sorted(df[id_col].unique()):
        df_id = df[df[id_col] == pid].copy()

        # --- 第1層: 信頼度フィルタリング（両肩が閾値以上のみ残す） ---
        conf_cols = ['left_shoulder_conf', 'right_shoulder_conf']
        if all(col in df_id.columns for col in conf_cols):
            df_id = df_id[
                (df_id['left_shoulder_conf'] > conf_threshold) &
                (df_id['right_shoulder_conf'] > conf_threshold)
            ]

        if len(df_id) == 0:
            print(f"person_id={pid} は第1層フィルタで全て除外されました。スキップします。")
            continue

        # 両耳間距離（正規化後カラムなければユークリッド距離で計算）
        if ear_dist_col is None or ear_dist_col not in df_id.columns:
            if all(col in df_id.columns for col in [left_ear_x_col, right_ear_x_col, left_ear_y_col, right_ear_y_col]):
                df_id['ear_distance'] = np.sqrt(
                    (df_id[left_ear_x_col] - df_id[right_ear_x_col]) ** 2 +
                    (df_id[left_ear_y_col] - df_id[right_ear_y_col]) ** 2
                )
                ear_dist_col_local = 'ear_distance'
            else:
                df_id['ear_distance'] = np.nan
                ear_dist_col_local = 'ear_distance'
        else:
            ear_dist_col_local = ear_dist_col

        frames_with_id = df_id[frame_col].nunique()
        percentage = (frames_with_id / total_frames) * 100 if total_frames > 0 else 0

        # 肩幅中央値に最も近い行を抽出
        median_val = df_id[shoulder_col].median()
        median_idx = (df_id[shoulder_col] - median_val).abs().idxmin()
        median_row = df_id.loc[median_idx]

        # 両耳間距離中央値に最も近い行を抽出
        median_val_ear = df_id[ear_dist_col_local].median()
        median_idx_ear = (df_id[ear_dist_col_local] - median_val_ear).abs().idxmin()
        median_row_ear = df_id.loc[median_idx_ear]

        # all_persons.csv用に保存
        all_persons_rows.append(median_row)

        # データ保存（csvサブフォルダへ）
        save_cols = [frame_col, shoulder_col, ear_dist_col_local]
        save_file = os.path.join(subfolders["csv"], f"length_data_{pid}.csv")
        df_id[save_cols].to_csv(save_file, index=False)

        # --- グラフ化対象フレームをまとめる ---
        df_id['person_id'] = pid  # 念のためperson_idを明示
        all_graph_rows.append(df_id)

        # 2x2グラフ作成（plotsサブフォルダへ）
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"person_id: {pid} の肩幅・両耳間長さの時系列推移と分布", fontname='Hiragino Sans')

        axs[0, 0].plot(df_id[frame_col], df_id[shoulder_col], marker='o', color='b')
        axs[0, 0].set_ylabel('肩幅', fontname='Hiragino Sans')
        axs[0, 0].set_title('肩幅の推移', fontname='Hiragino Sans')

        axs[1, 0].plot(df_id[frame_col], df_id[ear_dist_col_local], marker='o', color='g')
        axs[1, 0].set_ylabel('両耳間長さ', fontname='Hiragino Sans')
        axs[1, 0].set_title('両耳間長さの推移', fontname='Hiragino Sans')
        axs[1, 0].set_xlabel('フレーム番号', fontname='Hiragino Sans')

        # x軸ラベルの間引き
        if len(df_id[frame_col]) > 20:
            xticks = df_id[frame_col].iloc[::10]
        else:
            xticks = df_id[frame_col]
        axs[1, 0].set_xticks(xticks)
        axs[1, 0].tick_params(axis='x', rotation=45)

        # --- 分布のx軸幅を10刻みに設定 ---
        # 肩幅
        min_shoulder = np.floor(df_id[shoulder_col].min() / 10) * 10
        max_shoulder = np.ceil(df_id[shoulder_col].max() / 10) * 10
        axs[0, 1].hist(df_id[shoulder_col], bins=15, color='b', alpha=0.7)
        axs[0, 1].set_xlabel('肩幅', fontname='Hiragino Sans')
        axs[0, 1].set_ylabel('数', fontname='Hiragino Sans')
        axs[0, 1].set_title('肩幅の分布', fontname='Hiragino Sans')
        axs[0, 1].set_xlim(min_shoulder, max_shoulder)
        axs[0, 1].set_xticks(np.arange(min_shoulder, max_shoulder + 1, 10))

        # 両耳間長さ
        min_ear = np.floor(df_id[ear_dist_col_local].min() / 10) * 10
        max_ear = np.ceil(df_id[ear_dist_col_local].max() / 10) * 10
        axs[1, 1].hist(df_id[ear_dist_col_local], bins=15, color='g', alpha=0.7)
        axs[1, 1].set_xlabel('両耳間長さ', fontname='Hiragino Sans')
        axs[1, 1].set_ylabel('検出数', fontname='Hiragino Sans')
        axs[1, 1].set_title('両耳間長さの分布', fontname='Hiragino Sans')
        axs[1, 1].set_xlim(min_ear, max_ear)
        axs[1, 1].set_xticks(np.arange(min_ear, max_ear + 1, 10))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(subfolders["plots"], f"length_plot_{pid}.png")
        plt.savefig(out_file)
        plt.close(fig)

        # --- 代表値フレーム画像への座標可視化 ---
        frame_num = median_row[frame_col]
        img_candidates = [f for f in os.listdir(frames_dir) if str(frame_num) in f]
        if img_candidates:
            img_path = os.path.join(frames_dir, img_candidates[0])
            from PIL import Image
            img = Image.open(img_path)
            plt.figure(figsize=(8,8))
            plt.imshow(img)
            # 肩・耳の座標を描画（正規化ありなら正規化後、なしなら元座標）
            for x_col, y_col in zip(shoulder_x_cols + ear_x_cols, shoulder_y_cols + ear_y_cols):
                if x_col in median_row and y_col in median_row:
                    plt.scatter(
                        median_row[x_col], median_row[y_col],
                        color='#FF0033',  # 赤
                        edgecolors='#FF0033',
                        s=10,
                        linewidths=1.0,
                        zorder=3
                    )
            plt.title(f'person_id={pid} frame={frame_num} {"正規化後" if args.normalize else "元"}座標')
            out_file_pose = os.path.join(subfolders["normalized" if args.normalize else "raw"], f'pose_pid{pid}_frame{frame_num}_{"normalized" if args.normalize else "raw"}.png')
            plt.savefig(out_file_pose)
            plt.close()
            print(f"person_id={pid} の代表値座標（{"正規化後" if args.normalize else "元"}）を {out_file_pose} に保存しました")
        else:
            print(f"person_id={pid} frame={frame_num} の画像が見つかりません")

        # summary情報
        summary_list.append({
            'person_id': pid,
            'frames_with_id': frames_with_id,
            'total_frames': total_frames,
            'percentage': percentage
        })

    # all_persons.csvに中央値該当行を保存（ルート直下）
    if all_persons_rows:
        all_persons_save = os.path.join(out_dir, "all_persons.csv")
        pd.DataFrame(all_persons_rows).to_csv(all_persons_save, index=False)

    # summary csv（ルート直下）
    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"全person_idの検出割合サマリーを {summary_csv} に保存しました。")
    print(f"全person_idのデータ・グラフ・csvが {out_dir} 以下のサブフォルダに保存されました。")
    print(f"元CSVファイル情報は {info_path} に記録されています。")

    # --- グラフ化対象フレームの全データを1つのcsvにまとめて保存 ---
    if all_graph_rows:
        all_graph_df = pd.concat(all_graph_rows, ignore_index=True)
        graph_csv_file = os.path.join(out_dir, "graph_frames_all.csv")
        all_graph_df.to_csv(graph_csv_file, index=False)
        print(f"第1層フィルタ通過後の全データを {graph_csv_file} に保存しました。")
    else:
        graph_csv_file = None

    # === 第2層フィルタリング: 50%以上の検出率 ===
    print("\n" + "="*60)
    print("第2層フィルタリング: 全体取得数の50%以上")
    print("="*60)
    
    summary_df['rate'] = summary_df['frames_with_id'] / summary_df['total_frames']
    valid_ids_layer2 = summary_df.loc[summary_df['rate'] >= 0.5, 'person_id'].astype(int).tolist()
    excluded_ids_layer2 = summary_df.loc[summary_df['rate'] < 0.5, 'person_id'].astype(int).tolist()
    
    print(f"第2層通過ID（50%以上）: {valid_ids_layer2}")
    print(f"第2層除外ID（50%未満）: {excluded_ids_layer2}")

    # === 第3層フィルタリング用のヒストグラム生成 ===
    print("\n" + "="*60)
    print("第3層フィルタリング用ヒストグラム生成")
    print("="*60)
    print("※第3層の判定は次節(正規化関数獲得)で手動指定します")
    
    if graph_csv_file and os.path.exists(graph_csv_file):
        graph_df = pd.read_csv(graph_csv_file)
        
        # 第2層通過IDのヒストグラム作成
        print(f"\nヒストグラム生成対象ID（第2層通過）: {valid_ids_layer2}")
        for pid in valid_ids_layer2:
            person_hist_dir = os.path.join(subfolders["hist"], f"person_{pid}")
            os.makedirs(person_hist_dir, exist_ok=True)
            plot_detection_histogram(graph_df, frame_col=frame_col, pid=pid, 
                                    interval=args.interval, out_dir=person_hist_dir)
        
        print(f"\n第2層通過ID {len(valid_ids_layer2)}件 のヒストグラムを {subfolders['hist']} 以下に保存しました。")
        print("これらのヒストグラムを目視確認し、時間窓で不安定なID（例: 2, 3, 19）を特定してください。")
    else:
        print(f"graph_frames_all.csvが見つからないため、ヒストグラムは出力されません。")

    # === 最終レポート ===
    print("\n" + "="*60)
    print("フィルタリング結果サマリー")
    print("="*60)
    print(f"第1層（信頼度0.5以上）通過: {len(summary_df)} ID")
    print(f"第2層（検出率50%以上）通過: {len(valid_ids_layer2)} ID → {valid_ids_layer2}")
    print(f"第2層除外: {len(excluded_ids_layer2)} ID → {excluded_ids_layer2}")
    print(f"\n次のステップ:")
    print(f"  1. {subfolders['hist']} 内のヒストグラムを確認")
    print(f"  2. 時間窓で検出が不安定なIDを特定")
    print(f"  3. 正規化関数獲得スクリプトで除外ID指定")

if __name__ == "__main__":
    main()