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
    id_col = 'person_id'
    frame_col = 'frame'
    left_ear_x_col = 'left_ear_x'
    right_ear_x_col = 'right_ear_x'
    left_ear_y_col = 'left_ear_y'
    right_ear_y_col = 'right_ear_y'

    # --- 正規化カラム名の自動判定 ---
    if 'shoulder_width_normalized_linear' in df.columns:
        shoulder_col = 'shoulder_width_normalized_linear'
    elif 'shoulder_width_normalized_exp' in df.columns:
        shoulder_col = 'shoulder_width_normalized_exp'
    elif 'shoulder_width' in df.columns:
        shoulder_col = 'shoulder_width'
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

    # 元CSVの絶対パスを記録
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"元CSVファイル: {os.path.abspath(csv_file)}\n")
        f.write(f"作成日時: {timestamp}\n")

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

        # データ保存
        save_cols = [frame_col, shoulder_col, ear_dist_col_local]
        save_file = os.path.join(out_dir, f"length_data_{pid}.csv")
        df_id[save_cols].to_csv(save_file, index=False)

        # 2x2グラフ作成
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

        out_file = os.path.join(out_dir, f"length_plot_{pid}.png")
        plt.savefig(out_file)
        plt.close(fig)

        # summary情報
        summary_list.append({
            'person_id': pid,
            'frames_with_id': frames_with_id,
            'total_frames': total_frames,
            'percentage': percentage
        })

    # all_persons.csvに中央値該当行を保存
    if all_persons_rows:
        all_persons_save = os.path.join(out_dir, "all_persons.csv")
        pd.DataFrame(all_persons_rows).to_csv(all_persons_save, index=False)

    # summary csv
    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"全person_idの検出割合サマリーを {summary_csv} に保存しました。")
    print(f"全person_idのデータ・グラフ・csvが {out_dir} に保存されました。")
    print(f"元CSVファイル情報は {info_path} に記録されています。")

if __name__ == "__main__":
    main()