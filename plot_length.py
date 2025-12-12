import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'Hiragino Sans'

def analyze_all_persons(csv_file):
    df = pd.read_csv(csv_file)
    id_col = 'person_id'
    frame_col = 'frame'
    shoulder_col = 'shoulder_width'
    left_ear_x_col = 'left_ear_x'
    right_ear_x_col = 'right_ear_x'

    # 出力フォルダ作成（タイムスタンプ付き）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"length_analysis_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 元CSVの絶対パスを記録
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w", encoding="utf-8-sig") as f:
        f.write(f"元CSVファイル: {os.path.abspath(csv_file)}\n")
        f.write(f"作成日時: {timestamp}\n")

    # 検出数/推論フレーム割合を全体で計算
    total_frames = df[frame_col].nunique()
    summary_list = []

    # frame_idごとにサブフォルダ作成し、各person_idのcsvとグラフを格納
    frame_group = df.groupby(frame_col)
    for frame_id, frame_df in frame_group:
        frame_subdir = os.path.join(out_dir, f"frame_{frame_id}")
        os.makedirs(frame_subdir, exist_ok=True)
        # 全員分まとめcsv
        frame_save = os.path.join(frame_subdir, f"all_persons.csv")
        frame_df.to_csv(frame_save, index=False)

        for pid in sorted(frame_df[id_col].unique()):
            df_id = frame_df[frame_df[id_col] == pid]
            frames_with_id = df_id[frame_col].nunique()
            percentage = (frames_with_id / total_frames) * 100 if total_frames > 0 else 0

            # 両耳間長さ
            df_id = df_id.copy()
            df_id['ear_distance'] = np.abs(df_id[left_ear_x_col] - df_id[right_ear_x_col])

            # データ保存
            save_cols = [frame_col, shoulder_col, 'ear_distance']
            save_file = os.path.join(frame_subdir, f"length_data_{pid}.csv")
            df_id[save_cols].to_csv(save_file, index=False)

            # 2x2グラフ作成
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"frame: {frame_id} / person_id: {pid} の肩幅・両耳間長さの時系列推移と分布", fontname='Hiragino Sans')

            axs[0, 0].set_ylabel('肩幅 (pixel)', fontname='Hiragino Sans')
            axs[0, 1].set_xlabel('肩幅 (pixel)', fontname='Hiragino Sans')
            axs[0, 1].set_title('肩幅の分布', fontname='Hiragino Sans')

            axs[1, 0].set_ylabel('両耳間長さ (pixel)', fontname='Hiragino Sans')
            axs[1, 0].set_title('両耳間長さの推移', fontname='Hiragino Sans')
            axs[1, 1].set_xlabel('両耳間長さ (pixel)', fontname='Hiragino Sans')
            axs[1, 1].set_title('両耳間長さの分布', fontname='Hiragino Sans')

            if len(df_id[frame_col]) > 20:
                xticks = df_id[frame_col].iloc[::10]
            else:
                xticks = df_id[frame_col]
            axs[1, 0].set_xticks(xticks)
            axs[1, 0].tick_params(axis='x', rotation=45)

            axs[0, 1].hist(df_id[shoulder_col], bins=15, color='b', alpha=0.7)
            axs[0, 1].set_xlabel('肩幅 (mm)', fontname='Hiragino Sans')
            axs[0, 1].set_ylabel('数', fontname='Hiragino Sans')
            axs[0, 1].set_title('肩幅の分布', fontname='Hiragino Sans')

            axs[1, 1].hist(df_id['ear_distance'], bins=15, color='g', alpha=0.7)
            axs[1, 1].set_xlabel('両耳間長さ (mm)', fontname='Hiragino Sans')
            axs[1, 1].set_ylabel('検出数', fontname='Hiragino Sans')
            axs[1, 1].set_title('両耳間長さの分布', fontname='Hiragino Sans')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_file = os.path.join(frame_subdir, f"length_plot_{pid}.png")
            plt.savefig(out_file)
            plt.close(fig)

            # summary情報
            summary_list.append({
                'frame_id': frame_id,
                'person_id': pid,
                'frames_with_id': frames_with_id,
                'total_frames': total_frames,
                'percentage': percentage
            })

    # summary csv
    summary_df = pd.DataFrame(summary_list)
    summary_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"全person_idの検出割合サマリーを {summary_csv} に保存しました。")
    print(f"frame_idごとのサブフォルダに個別データ・グラフ・csvが保存されました。")
    print(f"元CSVファイル情報は {info_path} に記録されています。")

def main():
    csv_file = input("CSVファイル名を入力してください（例: 6point_metrics.csv）: ")
    analyze_all_persons(csv_file)

if __name__ == "__main__":
    main()