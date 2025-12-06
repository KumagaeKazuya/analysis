import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
import sys

import matplotlib
matplotlib.rc('font', family='AppleGothic')  # Macの場合

if len(sys.argv) < 2:
    print("使い方: python prot_angle_time_series.py <CSVファイルパス>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

def extract_frame_number(frame_name):
    match = re.search(r'frame[_\-]?(\d+)', str(frame_name))
    return int(match.group(1)) if match else None

df['frame_number'] = df['frame'].apply(extract_frame_number)

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
        plt.scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                    label=f'ID {pid}', s=30, alpha=0.7)
    mean_df = df.groupby('frame_number')['shoulder_head_angle'].mean().reset_index()
    plt.scatter(mean_df['frame_number'], mean_df['shoulder_head_angle'],
                color='red', marker='D', s=80, label='平均')
    plt.xlabel('フレーム番号', fontsize=13)
    plt.ylabel('正規化なす角度', fontsize=13)
    plt.title('時系列ごとの正規化なす角度（個人点＋平均）', fontsize=15)
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
        plt.scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                    label=f'ID {pid}', s=30, alpha=0.7)
    multi_mean_df = df[df['person_id'].isin(selected_ids)].groupby('frame_number')['shoulder_head_angle'].mean().reset_index()
    plt.scatter(multi_mean_df['frame_number'], multi_mean_df['shoulder_head_angle'],
                color='red', marker='D', s=80, label='選択ID時系列平均')
    plt.xlabel('フレーム番号', fontsize=13)
    plt.ylabel('正規化なす角度', fontsize=13)
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
    plt.scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                label=f'ID {single_id}', s=30, alpha=0.7)
    avg = person_df['shoulder_head_angle'].mean()
    plt.axhline(avg, color='red', linestyle='--', linewidth=2, label='全記録平均')
    plt.xlabel('フレーム番号', fontsize=13)
    plt.ylabel('正規化なす角度', fontsize=13)
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
        axes[0].scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                        label=f'ID {pid}', s=30, alpha=0.7)
    mean_df = df.groupby('frame_number')['shoulder_head_angle'].mean().reset_index()
    axes[0].scatter(mean_df['frame_number'], mean_df['shoulder_head_angle'],
                    color='red', marker='D', s=80, label='平均')
    axes[0].set_xlabel('フレーム番号', fontsize=13)
    axes[0].set_ylabel('正規化なす角度', fontsize=13)
    axes[0].set_title('全ID＋時系列平均', fontsize=15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2枚目：選択ID＋その時系列平均
    for pid in selected_ids:
        person_df = df[df['person_id'] == pid]
        axes[1].scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                        label=f'ID {pid}', s=30, alpha=0.7)
    multi_mean_df = df[df['person_id'].isin(selected_ids)].groupby('frame_number')['shoulder_head_angle'].mean().reset_index()
    axes[1].scatter(multi_mean_df['frame_number'], multi_mean_df['shoulder_head_angle'],
                    color='red', marker='D', s=80, label='選択ID時系列平均')
    axes[1].set_xlabel('フレーム番号', fontsize=13)
    axes[1].set_ylabel('正規化なす角度', fontsize=13)
    axes[1].set_title(f'選択ID({selected_ids})＋時系列平均', fontsize=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3枚目：単一ID＋全記録平均（水平線）
    if single_id is not None:
        person_df = df[df['person_id'] == single_id]
        axes[2].scatter(person_df['frame_number'], person_df['shoulder_head_angle'],
                        label=f'ID {single_id}', s=30, alpha=0.7)
        avg = person_df['shoulder_head_angle'].mean()
        axes[2].axhline(avg, color='red', linestyle='--', linewidth=2, label='全記録平均')
        axes[2].set_xlabel('フレーム番号', fontsize=13)
        axes[2].set_ylabel('正規化なす角度', fontsize=13)
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