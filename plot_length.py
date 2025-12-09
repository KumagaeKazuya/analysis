import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib

# 日本語フォント設定（Macの場合はヒラギノ角ゴなど）
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

def main():
    csv_file = input("CSVファイル名を入力してください（例: 6point_metrics.csv）: ")
    df = pd.read_csv(csv_file)
    id_value = input("対象のperson_idを入力してください: ")

    id_col = 'person_id'
    frame_col = 'frame'
    shoulder_col = 'shoulder_width'
    left_ear_x_col = 'left_ear_x'
    right_ear_x_col = 'right_ear_x'

    # person_idがint型で格納されている場合に備えて型変換
    try:
        id_value_int = int(id_value)
        df_id = df[df[id_col] == id_value_int]
    except ValueError:
        df_id = df[df[id_col] == id_value]

    if df_id.empty:
        print(f"person_id {id_value} のデータが見つかりません。")
        return

    frames_with_id = df_id[frame_col].nunique()
    total_frames = df[frame_col].nunique()
    percentage = (frames_with_id / total_frames) * 100
    print(f"person_id '{id_value}' の抽出フレーム割合: {percentage:.2f}% （{frames_with_id}/{total_frames}）")

    # 両耳間長さ（y座標は同じとみなしてx座標のみで計算）
    df_id['ear_distance'] = np.abs(df_id[left_ear_x_col] - df_id[right_ear_x_col])

    # データ保存（肩幅・両耳間長さ・フレーム番号のみ抽出）
    save_cols = [frame_col, shoulder_col, 'ear_distance']
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_file = f"length_data_{id_value}_{timestamp}.csv"
    df_id[save_cols].to_csv(save_file, index=False)
    print(f"抽出データを {save_file} に保存しました。")

    # 2x2グラフ作成
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"person_id: {id_value} の肩幅・両耳間長さの時系列推移と分布", fontname='Hiragino Sans')

    # 肩幅の時系列
    axs[0, 0].plot(df_id[frame_col], df_id[shoulder_col], marker='o', color='b')
    axs[0, 0].set_ylabel('肩幅 (mm)', fontname='Hiragino Sans')
    axs[0, 0].set_title('肩幅の推移', fontname='Hiragino Sans')

    # 両耳間長さの時系列
    axs[1, 0].plot(df_id[frame_col], df_id['ear_distance'], marker='o', color='g')
    axs[1, 0].set_ylabel('両耳間長さ (mm)', fontname='Hiragino Sans')
    axs[1, 0].set_title('両耳間長さの推移', fontname='Hiragino Sans')
    axs[1, 0].set_xlabel('フレーム番号', fontname='Hiragino Sans')

    # x軸ラベルの間引き（例：10個ごとに表示）
    if len(df_id[frame_col]) > 20:
        xticks = df_id[frame_col].iloc[::10]
    else:
        xticks = df_id[frame_col]
    axs[1, 0].set_xticks(xticks)
    axs[1, 0].tick_params(axis='x', rotation=45)

    # 肩幅の分布（ヒストグラム）
    axs[0, 1].hist(df_id[shoulder_col], bins=15, color='b', alpha=0.7)
    axs[0, 1].set_xlabel('肩幅 (mm)', fontname='Hiragino Sans')
    axs[0, 1].set_ylabel('数', fontname='Hiragino Sans')
    axs[0, 1].set_title('肩幅の分布', fontname='Hiragino Sans')

    # 両耳間長さの分布（ヒストグラム）
    axs[1, 1].hist(df_id['ear_distance'], bins=15, color='g', alpha=0.7)
    axs[1, 1].set_xlabel('両耳間長さ (mm)', fontname='Hiragino Sans')
    axs[1, 1].set_ylabel('検出数', fontname='Hiragino Sans')
    axs[1, 1].set_title('両耳間長さの分布', fontname='Hiragino Sans')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = f"length_plot_{id_value}_{timestamp}.png"
    plt.savefig(out_file)
    print(f"グラフを {out_file} に保存しました。")

    plt.show()

if __name__ == "__main__":
    main()