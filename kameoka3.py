import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging
import argparse
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def find_file(base_dir, subdir, ext, name_contains=None):
    pattern = f"*{name_contains if name_contains else ''}.{ext}"
    search_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    files = glob.glob(os.path.join(search_dir, pattern))
    return files[0] if files else None

def parse_args():
    parser = argparse.ArgumentParser(description="閾値計算スクリプト（プロジェクトフォルダ指定）")
    parser.add_argument("--project-dir", type=str, required=True, help="kameoka1.pyで作成したプロジェクトフォルダ")
    return parser.parse_args()

def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def load_monitor_config(config_file):
    if not os.path.exists(config_file):
        logger.error(f"モニター設定ファイルが見つかりません: {config_file}")
        raise FileNotFoundError(f"{config_file} not found")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"モニター設定を読み込みました: {len(config['monitors'])}台")
    return config

def sample_monitor_brightness(video_path, monitors, num_samples=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = 40
    logger.info(f"明るさサンプリング中... (総フレーム: {total_frames}, 間隔: {sample_interval})")
    brightness_data = {i: [] for i in range(len(monitors))}
    frame_idx = 0
    sample_count = 0
    while frame_idx < total_frames and sample_count < num_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        for i, monitor in enumerate(monitors):
            bbox = monitor.get("display_bbox", monitor["bbox"])
            x1, y1, x2, y2 = normalize_bbox(bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    brightness_data[i].append(mean_brightness)
        sample_count += 1
        frame_idx += sample_interval
    cap.release()
    monitor_avg_brightness = {}
    for i, values in brightness_data.items():
        if values:
            monitor_avg_brightness[i] = np.mean(values)
        else:
            monitor_avg_brightness[i] = 0.0
    logger.info(f"サンプリング完了: {sample_count}フレーム")
    return monitor_avg_brightness, brightness_data

def calculate_threshold_methods(brightness_values: List[float]) -> Dict:
    brightness_array = np.array(brightness_values)
    methods = {}
    hist, bins = np.histogram(brightness_array, bins=256, range=(0, 256))
    otsu_threshold = 0
    max_variance = 0
    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        variance = w0 * w1 * (mu0 - mu1) ** 2
        if variance > max_variance:
            max_variance = variance
            otsu_threshold = t
    methods['otsu'] = int(otsu_threshold)
    methods['mean'] = int(np.mean(brightness_array))
    methods['median'] = int(np.median(brightness_array))
    methods['percentile_25'] = int(np.percentile(brightness_array, 25))
    methods['percentile_50'] = int(np.percentile(brightness_array, 50))
    methods['percentile_75'] = int(np.percentile(brightness_array, 75))
    sorted_brightness = np.sort(brightness_array)
    gap_threshold = 30
    max_gap = 0
    gap_position = len(sorted_brightness) // 2
    for i in range(1, len(sorted_brightness)):
        gap = sorted_brightness[i] - sorted_brightness[i-1]
        if gap > max_gap and gap > gap_threshold:
            max_gap = gap
            gap_position = i
    if max_gap > gap_threshold:
        methods['bimodal'] = int((sorted_brightness[gap_position-1] + sorted_brightness[gap_position]) / 2)
    else:
        methods['bimodal'] = methods['median']
    return methods

def plot_all_monitors_histogram(monitor_brightness: Dict, monitors: List[Dict], threshold_methods: Dict, output_path):
    plt.figure(figsize=(12, 6))
    brightness_values = list(monitor_brightness.values())
    plt.hist(brightness_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    colors = {'otsu': 'red', 'mean': 'green', 'median': 'blue', 'bimodal': 'orange'}
    for method, threshold in threshold_methods.items():
        if method in colors:
            plt.axvline(threshold, color=colors[method], linestyle='--', linewidth=2, 
                       label=f'{method.upper()}: {threshold}')
    plt.xlabel('Average Brightness', fontsize=12)
    plt.ylabel('Number of Monitors', fontsize=12)
    plt.title('All Monitors - Brightness Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"保存: {output_path}")
    plt.close()

def plot_group_histograms(monitor_brightness: Dict, monitors: List[Dict], output_path):
    groups = {}
    for i, brightness in monitor_brightness.items():
        group = monitors[i].get('group', 'unknown')
        if group not in groups:
            groups[group] = []
        groups[group].append(brightness)
    fig, axes = plt.subplots(1, len(groups), figsize=(15, 4))
    if len(groups) == 1:
        axes = [axes]
    colors = {'top': 'lightcoral', 'middle': 'lightgreen', 'bottom': 'lightblue'}
    for ax, (group, values) in zip(axes, groups.items()):
        ax.hist(values, bins=30, color=colors.get(group, 'gray'), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Brightness', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{group.capitalize()} ({len(values)} monitors)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"保存: {output_path}")
    plt.close()

def plot_threshold_simple(monitor_brightness: Dict, recommended_threshold: int, output_path):
    plt.figure(figsize=(12, 6))
    brightness_values = list(monitor_brightness.values())
    monitor_ids = list(monitor_brightness.keys())
    colors = ['green' if b > recommended_threshold else 'red' for b in brightness_values]
    plt.bar(monitor_ids, brightness_values, color=colors, edgecolor='black', alpha=0.7)
    plt.axhline(recommended_threshold, color='blue', linestyle='--', linewidth=2, 
               label=f'Threshold: {recommended_threshold}')
    plt.xlabel('Monitor ID', fontsize=12)
    plt.ylabel('Average Brightness', fontsize=12)
    plt.title('Monitor Brightness with Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"保存: {output_path}")
    plt.close()

def plot_threshold_calculation(brightness_values: List[float], threshold_methods: Dict, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1 = axes[0, 0]
    ax1.hist(brightness_values, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
    colors = {'otsu': 'red', 'mean': 'green', 'median': 'blue', 'bimodal': 'orange'}
    for method, threshold in threshold_methods.items():
        if method in colors:
            ax1.axvline(threshold, color=colors[method], linestyle='--', linewidth=2, 
                       label=f'{method}: {threshold}')
    ax1.set_xlabel('Brightness')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram with Multiple Thresholds')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    sorted_brightness = np.sort(brightness_values)
    cumulative = np.arange(1, len(sorted_brightness) + 1) / len(sorted_brightness) * 100
    ax2.plot(sorted_brightness, cumulative, linewidth=2, color='darkblue')
    ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax2.axhline(75, color='orange', linestyle='--', alpha=0.5, label='75%')
    ax2.set_xlabel('Brightness')
    ax2.set_ylabel('Cumulative %')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    ax3.boxplot([brightness_values], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax3.set_ylabel('Brightness')
    ax3.set_title('Box Plot')
    ax3.grid(True, alpha=0.3)
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = []
    for method, value in threshold_methods.items():
        on_count = sum(1 for b in brightness_values if b > value)
        off_count = len(brightness_values) - on_count
        table_data.append([method.upper(), value, on_count, off_count])
    table = ax4.table(cellText=table_data,
                     colLabels=['Method', 'Threshold', 'ON', 'OFF'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Threshold Comparison', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"保存: {output_path}")
    plt.close()

def select_best_threshold(threshold_methods: Dict, brightness_values: List[float]) -> Tuple[int, str]:
    std = np.std(brightness_values)
    brightness_range = max(brightness_values) - min(brightness_values)
    if brightness_range < 20:
        if np.mean(brightness_values) > 100:
            method = 'all_on'
            threshold = int(np.mean(brightness_values) - 10)
        else:
            method = 'all_off'
            threshold = int(np.mean(brightness_values) + 10)
    elif std < 15:
        method = 'few_on'
        threshold = threshold_methods['percentile_75']
    elif 'bimodal' in threshold_methods and threshold_methods['bimodal'] != threshold_methods['median']:
        method = 'bimodal'
        threshold = threshold_methods['bimodal']
    else:
        method = 'otsu'
        threshold = threshold_methods['otsu']
    logger.info(f"選択された閾値: {threshold} (方法: {method})")
    return threshold, method

def main():
    args = parse_args()
    base_dir = args.project_dir

    # 必要なファイル探索
    input_video = find_file(base_dir, "video", "mp4")
    monitor_config_json = find_file(base_dir, "json", "json", "monitor_config")
    threshold_config_json = os.path.join(base_dir, "json", "threshold_config.json")
    output_dir = os.path.join(base_dir, "threshold_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # 出力画像パス
    all_monitors_histogram = os.path.join(output_dir, "all_monitors_histogram.png")
    group_histograms = os.path.join(output_dir, "group_histograms.png")
    threshold_simple = os.path.join(output_dir, "threshold_simple.png")
    threshold_calculation = os.path.join(output_dir, "threshold_calculation.png")

    if not input_video or not os.path.exists(monitor_config_json):
        logger.error("必要なファイルが見つかりません")
        logger.error(f"input_video: {input_video}")
        logger.error(f"monitor_config_json: {monitor_config_json}")
        return

    # モニター設定読み込み
    monitor_config = load_monitor_config(monitor_config_json)
    monitors = monitor_config['monitors']

    # 明るさサンプリング
    monitor_brightness, brightness_time_series = sample_monitor_brightness(
        input_video,
        monitors,
        num_samples=30
    )

    brightness_values = list(monitor_brightness.values())

    # 閾値計算
    logger.info("\n閾値計算中...")
    threshold_methods = calculate_threshold_methods(brightness_values)
    for method, value in threshold_methods.items():
        logger.info(f"  {method}: {value}")

    # 最適閾値選択
    recommended_threshold, selected_method = select_best_threshold(threshold_methods, brightness_values)

    # ヒストグラム生成
    logger.info("\nヒストグラム生成中...")
    plot_all_monitors_histogram(monitor_brightness, monitors, threshold_methods, all_monitors_histogram)
    plot_group_histograms(monitor_brightness, monitors, group_histograms)
    plot_threshold_simple(monitor_brightness, recommended_threshold, threshold_simple)
    plot_threshold_calculation(brightness_values, threshold_methods, threshold_calculation)

    # ONモニターのリストを作成
    on_monitors = [
        i for i, brightness in monitor_brightness.items() 
        if brightness > recommended_threshold
    ]

    # 閾値設定を保存
    threshold_config = {
        "threshold": recommended_threshold,
        "method": selected_method,
        "all_methods": threshold_methods,
        "statistics": {
            "mean": float(np.mean(brightness_values)),
            "median": float(np.median(brightness_values)),
            "std": float(np.std(brightness_values)),
            "min": float(np.min(brightness_values)),
            "max": float(np.max(brightness_values))
        },
        "on_monitors": on_monitors
    }

    with open(threshold_config_json, 'w', encoding='utf-8') as f:
        json.dump(threshold_config, f, indent=2, ensure_ascii=False)

    logger.info(f"\n閾値設定を保存: {threshold_config_json}")

    # サマリー
    logger.info("\n" + "=" * 80)
    logger.info("完了！")
    logger.info("=" * 80)
    logger.info(f"推奨閾値: {recommended_threshold} (方法: {selected_method})")
    logger.info(f"ONモニター: {len(on_monitors)}台 / {len(monitors)}台")
    logger.info(f"ONモニターID: {on_monitors}")
    logger.info(f"\n生成された画像:")
    logger.info(f"  - {all_monitors_histogram}")
    logger.info(f"  - {group_histograms}")
    logger.info(f"  - {threshold_simple}")
    logger.info(f"  - {threshold_calculation}")

    logger.info("\n" + "=" * 80)
    logger.info("次のステップ:")
    logger.info("  python kameoka4.py --project-dir <プロジェクトフォルダ>")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()