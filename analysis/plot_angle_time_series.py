import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.rc('font', family='AppleGothic')  # Macの場合

# --- 対話形式で2つのCSVを指定 ---
print("正規化後のCSVファイルパスを入力してください（例: 6point_metrics_normalized_linear.csv）:")
normalized_csv_path = input().strip()
print("肩幅固定後の代表値CSVファイルパスを入力してください（例: all_persons.csv）:")
fixed_csv_path = input().strip()

# --- CSV読み込み ---
df_norm = pd.read_csv(normalized_csv_path)
df_fixed = pd.read_csv(fixed_csv_path)

# --- 肩幅固定CSVからperson_idごとに肩幅正規化後の肩座標を抽出（代表値フレームのもの） ---
shoulder_cols = [
    "person_id",
    "left_shoulder_x_normalized_exp", "left_shoulder_y_normalized_exp",
    "right_shoulder_x_normalized_exp", "right_shoulder_y_normalized_exp"
]
shoulder_fixed = df_fixed[shoulder_cols].drop_duplicates("person_id").set_index("person_id")

# --- 正規化後CSVから両耳座標とframeを抽出 ---
ear_cols = [
    "person_id", "frame",
    "left_ear_x_normalized_exp", "left_ear_y_normalized_exp",
    "right_ear_x_normalized_exp", "right_ear_y_normalized_exp"
]
ears_norm = df_norm[ear_cols]

# --- 肩座標を全時系列にマージ（person_idで結合） ---
df = ears_norm.merge(shoulder_fixed, left_on="person_id", right_index=True, how="left")

# --- なす角計算 ---
def calc_shoulder_ear_angle(row):
    # 固定肩座標（肩幅正規化後）
    lsh_x = row["left_shoulder_x_normalized_exp"]
    lsh_y = row["left_shoulder_y_normalized_exp"]
    rsh_x = row["right_shoulder_x_normalized_exp"]
    rsh_y = row["right_shoulder_y_normalized_exp"]
    # 時系列ごとの耳座標（正規化後）
    lea_x = row["left_ear_x_normalized_exp"]
    lea_y = row["left_ear_y_normalized_exp"]
    rea_x = row["right_ear_x_normalized_exp"]
    rea_y = row["right_ear_y_normalized_exp"]
    try:
        # 肩中点
        shoulder_cx = (lsh_x + rsh_x) / 2
        shoulder_cy = (lsh_y + rsh_y) / 2
        # 肩幅固定ベクトル（右肩→左肩）
        shoulder_vec = np.array([lsh_x - rsh_x, lsh_y - rsh_y])
        # 耳中点
        ear_cx = (lea_x + rea_x) / 2
        ear_cy = (lea_y + rea_y) / 2
        # 肩中点→耳中点ベクトル
        shoulder_to_ear_vec = np.array([ear_cx - shoulder_cx, ear_cy - shoulder_cy])
        # なす角
        dot = np.dot(shoulder_vec, shoulder_to_ear_vec)
        norm1 = np.linalg.norm(shoulder_vec)
        norm2 = np.linalg.norm(shoulder_to_ear_vec)
        if norm1 == 0 or norm2 == 0:
            return np.nan
        cos_theta = np.clip(dot / (norm1 * norm2), -1, 1)
        return np.degrees(np.arccos(cos_theta))
    except Exception:
        return np.nan

df['shoulder_ear_angle'] = df.apply(calc_shoulder_ear_angle, axis=1)

# --- グラフ描画UI ---
print("表示するグラフを選択してください：")
print("1: 全ID＋時系列平均")
print("2: 選択ID＋その時系列平均")
print("3: 単一ID＋全記録平均（水平線）")
print("4: 2x2グリッド（1,2,3をまとめて表示）")
graph_type = input("番号で選択: ").strip()

fig = None
axes = None

if graph_type == "1":
    plt.figure(figsize=(14, 7))
    for pid in df['person_id'].unique():
        person_df = df[df['person_id'] == pid]
        plt.scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                    label=f'ID {pid}', s=30, alpha=0.7)
    mean_df = df.groupby('frame')['shoulder_ear_angle'].mean().reset_index()
    plt.scatter(mean_df['frame'], mean_df['shoulder_ear_angle'],
                color='red', marker='D', s=80, label='平均')
    plt.xlabel('フレーム', fontsize=13)
    plt.ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
    plt.title('時系列ごとのなす角度（個人点＋平均）', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()

elif graph_type == "2":
    selected_ids_input = input("複数IDをカンマ区切りで入力（例: 1,2,3）: ").strip()
    selected_ids = [int(i) for i in selected_ids_input.split(",") if i.strip().isdigit()]
    plt.figure(figsize=(14, 7))
    for pid in selected_ids:
        person_df = df[df['person_id'] == pid]
        plt.scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                    label=f'ID {pid}', s=30, alpha=0.7)
    multi_mean_df = df[df['person_id'].isin(selected_ids)].groupby('frame')['shoulder_ear_angle'].mean().reset_index()
    plt.scatter(multi_mean_df['frame'], multi_mean_df['shoulder_ear_angle'],
                color='red', marker='D', s=80, label='選択ID時系列平均')
    plt.xlabel('フレーム', fontsize=13)
    plt.ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
    plt.title(f'選択ID({selected_ids})＋時系列平均', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()

elif graph_type == "3":
    single_id_input = input("単一IDを入力（例: 1）: ").strip()
    single_id = int(single_id_input) if single_id_input.isdigit() else None
    plt.figure(figsize=(14, 7))
    person_df = df[df['person_id'] == single_id]
    plt.scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                label=f'ID {single_id}', s=30, alpha=0.7)
    avg = person_df['shoulder_ear_angle'].mean()
    plt.axhline(avg, color='red', linestyle='--', linewidth=2, label='全記録平均')
    plt.xlabel('フレーム', fontsize=13)
    plt.ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
    plt.title(f'単一ID({single_id})＋全記録平均', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()

elif graph_type == "4":
    selected_ids_input = input("複数IDをカンマ区切りで入力（例: 1,2,3）: ").strip()
    selected_ids = [int(i) for i in selected_ids_input.split(",") if i.strip().isdigit()]
    single_id_input = input("単一IDを入力（例: 1）: ").strip()
    single_id = int(single_id_input) if single_id_input.isdigit() else None

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    # 1枚目：全ID＋時系列平均
    for pid in df['person_id'].unique():
        person_df = df[df['person_id'] == pid]
        axes[0].scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                        label=f'ID {pid}', s=30, alpha=0.7)
    mean_df = df.groupby('frame')['shoulder_ear_angle'].mean().reset_index()
    axes[0].scatter(mean_df['frame'], mean_df['shoulder_ear_angle'],
                    color='red', marker='D', s=80, label='平均')
    axes[0].set_xlabel('フレーム', fontsize=13)
    axes[0].set_ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
    axes[0].set_title('全ID＋時系列平均', fontsize=15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2枚目：選択ID＋その時系列平均
    for pid in selected_ids:
        person_df = df[df['person_id'] == pid]
        axes[1].scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                        label=f'ID {pid}', s=30, alpha=0.7)
    multi_mean_df = df[df['person_id'].isin(selected_ids)].groupby('frame')['shoulder_ear_angle'].mean().reset_index()
    axes[1].scatter(multi_mean_df['frame'], multi_mean_df['shoulder_ear_angle'],
                    color='red', marker='D', s=80, label='選択ID時系列平均')
    axes[1].set_xlabel('フレーム', fontsize=13)
    axes[1].set_ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
    axes[1].set_title(f'選択ID({selected_ids})＋時系列平均', fontsize=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3枚目：単一ID＋全記録平均（水平線）
    if single_id is not None:
        person_df = df[df['person_id'] == single_id]
        axes[2].scatter(person_df['frame'], person_df['shoulder_ear_angle'],
                        label=f'ID {single_id}', s=30, alpha=0.7)
        avg = person_df['shoulder_ear_angle'].mean()
        axes[2].axhline(avg, color='red', linestyle='--', linewidth=2, label='全記録平均')
        axes[2].set_xlabel('フレーム', fontsize=13)
        axes[2].set_ylabel('なす角度（肩幅固定ベクトル×肩中点→耳中点）', fontsize=13)
        axes[2].set_title(f'単一ID({single_id})＋全記録平均', fontsize=15)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')

    # 4枚目は空白
    axes[3].axis('off')

    plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("outputs", "angle_time_series", timestamp)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "angle_time_series_selected.png")

if fig is not None:
    fig.savefig(save_path)
else:
    plt.savefig(save_path)
print(f"グラフ画像を保存しました: {save_path}")
plt.show()