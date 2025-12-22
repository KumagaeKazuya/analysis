import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib
import os

# 日本語フォント設定（Macの場合はヒラギノ Sans）
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

def main():
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
    elif 'shoulder_width' in df.columns:
        shoulder_col = 'shoulder_width'
        shoulder_x_cols = ['left_shoulder_x', 'right_shoulder_x']
        shoulder_y_cols = ['left_shoulder_y', 'right_shoulder_y']
        ear_x_cols = ['left_ear_x', 'right_ear_x']
        ear_y_cols = ['left_ear_y', 'right_ear_y']
    else:
        raise ValueError("正規化肩幅カラムが見つかりません")

    if 'ear_distance_normalized_linear' in df.columns:
        ear_dist_col = 'ear_distance_normalized_linear'
    elif 'ear_distance_normalized_exp' in df.columns:
        ear_dist_col = 'ear_distance_normalized_exp'
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
    }
    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)

    # 元CSVの絶対パスを記録
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"元CSVファイル: {os.path.abspath(csv_file)}\n")
        f.write(f"作成日時: {timestamp}\n")
        f.write(f"フレーム画像フォルダ: {os.path.abspath(frames_dir)}\n")

    total_frames = df[frame_col].nunique()
    summary_list = []
    all_persons_rows = []

    for pid in sorted(df[id_col].unique()):
        df_id = df[df[id_col] == pid].copy()

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

        # IQR外れ値除去
        q1 = df_id[shoulder_col].quantile(0.25)
        q3 = df_id[shoulder_col].quantile(0.75)
        iqr_mask = (df_id[shoulder_col] >= q1) & (df_id[shoulder_col] <= q3)
        df_iqr = df_id[iqr_mask]
        if len(df_iqr) == 0:
            df_iqr = df_id

        # 肩幅中央値に最も近い行を抽出
        median_val = df_iqr[shoulder_col].median()
        median_idx = (df_iqr[shoulder_col] - median_val).abs().idxmin()
        median_row = df_iqr.loc[median_idx]

        # 両耳間距離IQR外れ値除去
        q1_ear = df_iqr[ear_dist_col_local].quantile(0.25)
        q3_ear = df_iqr[ear_dist_col_local].quantile(0.75)
        iqr_mask_ear = (df_iqr[ear_dist_col_local] >= q1_ear) & (df_iqr[ear_dist_col_local] <= q3_ear)
        df_iqr_ear = df_iqr[iqr_mask_ear]
        if len(df_iqr_ear) == 0:
            df_iqr_ear = df_iqr

        # 両耳間距離中央値に最も近い行を抽出
        median_val_ear = df_iqr_ear[ear_dist_col_local].median()
        median_idx_ear = (df_iqr_ear[ear_dist_col_local] - median_val_ear).abs().idxmin()
        median_row_ear = df_iqr_ear.loc[median_idx_ear]

        # all_persons.csv用に保存
        all_persons_rows.append(median_row)
        # if median_idx != median_idx_ear:
        #    all_persons_rows.append(median_row_ear)

        # データ保存（csvサブフォルダへ）
        save_cols = [frame_col, shoulder_col, ear_dist_col_local]
        save_file = os.path.join(subfolders["csv"], f"length_data_{pid}.csv")
        df_id[save_cols].to_csv(save_file, index=False)

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

        axs[0, 1].hist(df_id[shoulder_col], bins=15, color='b', alpha=0.7)
        axs[0, 1].set_xlabel('肩幅', fontname='Hiragino Sans')
        axs[0, 1].set_ylabel('数', fontname='Hiragino Sans')
        axs[0, 1].set_title('肩幅の分布', fontname='Hiragino Sans')

        axs[1, 1].hist(df_id[ear_dist_col_local], bins=15, color='g', alpha=0.7)
        axs[1, 1].set_xlabel('両耳間長さ', fontname='Hiragino Sans')
        axs[1, 1].set_ylabel('検出数', fontname='Hiragino Sans')
        axs[1, 1].set_title('両耳間長さの分布', fontname='Hiragino Sans')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(subfolders["plots"], f"length_plot_{pid}.png")
        plt.savefig(out_file)
        plt.close(fig)

        # --- 代表値フレーム画像への座標可視化（正規化後カラム） ---
        frame_num = median_row[frame_col]
        img_candidates = [f for f in os.listdir(frames_dir) if str(frame_num) in f]
        if img_candidates:
            img_path = os.path.join(frames_dir, img_candidates[0])
            from PIL import Image
            img = Image.open(img_path)
            plt.figure(figsize=(8,8))
            plt.imshow(img)
            # 肩・耳の正規化後座標を描画（色は黒、サイズはさらに小さく、枠線付きで見やすく）
            # 正規化後は青（枠線も青）、サイズはさらに小さく
            for x_col, y_col in zip(shoulder_x_cols + ear_x_cols, shoulder_y_cols + ear_y_cols):
                if x_col in median_row and y_col in median_row:
                    plt.scatter(
                        median_row[x_col], median_row[y_col],
                        color='#FF0033',  # 赤
                        edgecolors='#FF0033',  # 枠線も赤
                        s=10,  # さらに小さく
                        linewidths=1.0,
                        zorder=3
                    )
            plt.title(f'person_id={pid} frame={frame_num} 正規化後座標')
            out_file_pose = os.path.join(subfolders["normalized"], f'pose_pid{pid}_frame{frame_num}_normalized.png')
            plt.savefig(out_file_pose)
            plt.close()
            print(f"person_id={pid} の代表値座標（正規化後）を {out_file_pose} に保存しました")
        else:
            print(f"person_id={pid} frame={frame_num} の画像が見つかりません")

        # --- 代表値フレーム画像への座標可視化（正規化前カラム） ---
        if 'left_shoulder_x' in median_row and 'right_shoulder_x' in median_row:
            raw_shoulder_x_cols = ['left_shoulder_x', 'right_shoulder_x']
            raw_shoulder_y_cols = ['left_shoulder_y', 'right_shoulder_y']
            raw_ear_x_cols = ['left_ear_x', 'right_ear_x']
            raw_ear_y_cols = ['left_ear_y', 'right_ear_y']
            img_path = os.path.join(frames_dir, img_candidates[0])
            img = Image.open(img_path)
            plt.figure(figsize=(8,8))
            plt.imshow(img)
            for x_col, y_col in zip(raw_shoulder_x_cols + raw_ear_x_cols, raw_shoulder_y_cols + raw_ear_y_cols):
                if x_col in median_row and y_col in median_row:
                    plt.scatter(
                        median_row[x_col], median_row[y_col],
                        color='#FF0033',  # 赤
                        edgecolors='#FF0033',  # 枠線も赤
                        s=10,  # さらに小さく
                        linewidths=1.0,
                        zorder=3
                    )
            plt.title(f'person_id={pid} frame={frame_num} 正規化前座標')
            out_file_pose_raw = os.path.join(subfolders["raw"], f'pose_pid{pid}_frame{frame_num}_raw.png')
            plt.savefig(out_file_pose_raw)
            plt.close()
            print(f"person_id={pid} の代表値座標（正規化前）を {out_file_pose_raw} に保存しました")

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

if __name__ == "__main__":
    main()