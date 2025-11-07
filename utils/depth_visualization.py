"""
深度推定可視化モジュール
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_depth_enhanced_visualizations(
    detection_results: Dict[str, Any],
    output_dir: Path,
    video_name: str
) -> None:
    """
    深度情報統合可視化の生成

    Args:
        detection_results: 検出結果辞書
        output_dir: 出力ディレクトリ
        video_name: 動画名
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 深度統合CSVの確認
        result_data = detection_results.get("data", {})
        enhanced_csv_path = result_data.get("enhanced_csv_path")

        if not enhanced_csv_path or not Path(enhanced_csv_path).exists():
            logger.warning(f"深度統合CSVが見つかりません: {enhanced_csv_path}")
            return

        df = pd.read_csv(enhanced_csv_path)

        if df.empty:
            logger.warning("深度統合CSVが空です")
            return

        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 2x3レイアウトで深度分析グラフを生成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'深度分析レポート: {video_name}', fontsize=16)

        # 1. 距離ゾーン別検出数
        if 'depth_zone' in df.columns:
            zone_counts = df['depth_zone'].value_counts()
            if len(zone_counts) > 0:
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A'][:len(zone_counts)]
                bars = axes[0, 0].bar(zone_counts.index, zone_counts.values, color=colors)
                axes[0, 0].set_title('教室ゾーン別検出数')
                axes[0, 0].set_ylabel('検出数')
                axes[0, 0].tick_params(axis='x', rotation=45)

                # 数値ラベル追加
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom')

        # 2. 距離-信頼度散布図
        if 'depth_distance' in df.columns and 'conf' in df.columns:
            valid_depth = df[df['depth_distance'] >= 0]
            if len(valid_depth) > 0:
                scatter = axes[0, 1].scatter(
                    valid_depth['depth_distance'],
                    valid_depth['conf'],
                    alpha=0.6,
                    c=valid_depth.index,
                    cmap='viridis',
                    s=30
                )
                axes[0, 1].set_title('距離 vs 検出信頼度')
                axes[0, 1].set_xlabel('深度距離値')
                axes[0, 1].set_ylabel('信頼度')
                axes[0, 1].grid(True, alpha=0.3)

        # 3. ゾーン別信頼度ボックスプロット
        if 'depth_zone' in df.columns and 'conf' in df.columns:
            zones = df['depth_zone'].unique()
            zone_confs = [df[df['depth_zone'] == zone]['conf'].values for zone in zones]
            zone_confs = [data for data in zone_confs if len(data) > 0]  # 空のデータを除外

            if zone_confs:
                box_plot = axes[0, 2].boxplot(zone_confs, labels=zones[:len(zone_confs)])
                axes[0, 2].set_title('ゾーン別信頼度分布')
                axes[0, 2].set_ylabel('信頼度')
                axes[0, 2].tick_params(axis='x', rotation=45)

        # 4. 時系列での平均距離推移
        if 'frame' in df.columns and 'depth_distance' in df.columns:
            valid_depth_df = df[df['depth_distance'] >= 0]
            if len(valid_depth_df) > 0:
                frame_depth = valid_depth_df.groupby('frame')['depth_distance'].mean()
                axes[1, 0].plot(frame_depth.index, frame_depth.values,
                            color='green', linewidth=2, marker='o', markersize=3)
                axes[1, 0].set_title('フレーム別平均距離推移')
                axes[1, 0].set_xlabel('フレーム番号')
                axes[1, 0].set_ylabel('平均深度距離')
                axes[1, 0].grid(True, alpha=0.3)

        # 5. ID別距離範囲
        if 'person_id' in df.columns and 'depth_distance' in df.columns:
            valid_df = df[df['depth_distance'] >= 0]
            if len(valid_df) > 0:
                id_stats = valid_df.groupby('person_id')['depth_distance'].agg(['min', 'max', 'mean'])
                if len(id_stats) > 0:
                    axes[1, 1].scatter(id_stats['min'], id_stats['max'],
                                    alpha=0.7, s=50, c='blue')

                    # 対角線
                    min_val = min(id_stats['min'].min(), id_stats['max'].min())
                    max_val = max(id_stats['min'].max(), id_stats['max'].max())
                    axes[1, 1].plot([min_val, max_val], [min_val, max_val],
                                'r--', alpha=0.5, label='等距離線')

                    axes[1, 1].set_title('ID別距離範囲 (最小 vs 最大)')
                    axes[1, 1].set_xlabel('最小距離')
                    axes[1, 1].set_ylabel('最大距離')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

        # 6. 深度信頼度ヒストグラム
        if 'depth_confidence' in df.columns:
            depth_conf_data = df['depth_confidence'].dropna()
            if len(depth_conf_data) > 0:
                axes[1, 2].hist(depth_conf_data, bins=20, alpha=0.7,
                            color='purple', edgecolor='black')
                axes[1, 2].set_title('深度推定信頼度分布')
                axes[1, 2].set_xlabel('深度信頼度')
                axes[1, 2].set_ylabel('頻度')
                axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = output_dir / f"depth_analysis_{video_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ 深度分析グラフ保存: {output_path}")

        # 統計サマリーテキストファイル生成
        _create_depth_summary_text(df, output_dir, video_name)

    except Exception as e:
        logger.error(f"深度可視化生成エラー: {e}", exc_info=True)


def _create_depth_summary_text(df: pd.DataFrame, output_dir: Path, video_name: str) -> None:
    """深度統計サマリーテキストファイル生成"""
    try:
        summary_lines = [
            f"深度分析サマリー: {video_name}",
            "=" * 50,
            f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "基本統計:",
            f"  総検出数: {len(df)}",
        ]

        # 深度成功率
        if 'depth_distance' in df.columns:
            valid_depth = df[df['depth_distance'] >= 0]
            success_rate = len(valid_depth) / len(df) if len(df) > 0 else 0
            summary_lines.extend([
                f"  有効深度検出数: {len(valid_depth)}",
                f"  深度成功率: {success_rate:.1%}",
            ])

            if len(valid_depth) > 0:
                summary_lines.extend([
                    f"  平均距離: {valid_depth['depth_distance'].mean():.2f}",
                    f"  距離範囲: {valid_depth['depth_distance'].min():.2f} - {valid_depth['depth_distance'].max():.2f}",
                ])

        # ゾーン分布
        if 'depth_zone' in df.columns:
            summary_lines.extend(["", "ゾーン別分布:"])
            zone_counts = df['depth_zone'].value_counts()
            for zone, count in zone_counts.items():
                percentage = count / len(df) * 100
                summary_lines.append(f"  {zone}: {count}件 ({percentage:.1f}%)")

        # 信頼度統計
        if 'conf' in df.columns:
            summary_lines.extend([
                "",
                "検出信頼度統計:",
                f"  平均信頼度: {df['conf'].mean():.3f}",
                f"  信頼度範囲: {df['conf'].min():.3f} - {df['conf'].max():.3f}",
            ])

        # 深度信頼度統計
        if 'depth_confidence' in df.columns:
            depth_conf = df['depth_confidence'].dropna()
            if len(depth_conf) > 0:
                summary_lines.extend([
                    "",
                    "深度信頼度統計:",
                    f"  平均深度信頼度: {depth_conf.mean():.3f}",
                    f"  深度信頼度範囲: {depth_conf.min():.3f} - {depth_conf.max():.3f}",
                ])

        # ファイル保存
        summary_path = output_dir / f"depth_summary_{video_name}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        logger.info(f"✅ 深度サマリー保存: {summary_path}")

    except Exception as e:
        logger.error(f"深度サマリー生成エラー: {e}")


def create_depth_heatmap(depth_map: np.ndarray, output_path: Path) -> None:
    """
    深度ヒートマップの生成

    Args:
        depth_map: 深度マップ配列
        output_path: 出力パス
    """
    try:
        plt.figure(figsize=(12, 8))

        # カラーマップで深度を可視化
        plt.imshow(depth_map, cmap='plasma', aspect='auto')
        plt.colorbar(label='深度値')
        plt.title('教室深度マップ')
        plt.xlabel('X座標 (pixel)')
        plt.ylabel('Y座標 (pixel)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ 深度ヒートマップ保存: {output_path}")

    except Exception as e:
        logger.error(f"深度ヒートマップ生成エラー: {e}")


def create_depth_zone_visualization(
    frame_image: np.ndarray,
    detections: list,
    output_path: Path
) -> None:
    """
    深度ゾーン可視化（検出結果重畳表示）

    Args:
        frame_image: フレーム画像
        detections: 検出結果リスト（ゾーン情報含む）
        output_path: 出力パス
    """
    try:
        # 画像をコピー
        vis_image = frame_image.copy()

        # ゾーン別色設定
        zone_colors = {
            'front': (0, 255, 0),      # 緑
            'middle': (255, 255, 0),   # 黄
            'back': (255, 0, 0),       # 赤
            'far_back': (255, 0, 255), # マゼンタ
            'unknown': (128, 128, 128)  # グレー
        }

        # 検出結果を描画
        for detection in detections:
            bbox = detection.get('bbox', [])
            zone = detection.get('depth_zone', 'unknown')
            confidence = detection.get('conf', 0)

            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = zone_colors.get(zone, (128, 128, 128))

                # バウンディングボックス描画
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # ラベル描画
                label = f"{zone} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_image, (x1, y1-label_size[1]-10),
                            (x1+label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存
        cv2.imwrite(str(output_path), vis_image)
        logger.info(f"✅ 深度ゾーン可視化保存: {output_path}")

    except Exception as e:
        logger.error(f"深度ゾーン可視化エラー: {e}")


# 互換性のための追加関数
def generate_depth_report(
    detection_results: Dict[str, Any],
    output_dir: Path,
    video_name: str
) -> None:
    """
    深度レポート生成（メイン関数のエイリアス）
    """
    create_depth_enhanced_visualizations(detection_results, output_dir, video_name)


def create_comparative_depth_visualization(
    baseline_results: Dict[str, Any],
    improved_results: Dict[str, Any],
    output_dir: Path,
    experiment_name: str
) -> None:
    """
    深度推定改善前後の比較可視化

    Args:
        baseline_results: ベースライン結果
        improved_results: 改善後結果
        output_dir: 出力ディレクトリ
        experiment_name: 実験名
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 比較データの準備
        baseline_csv = baseline_results.get("data", {}).get("enhanced_csv_path")
        improved_csv = improved_results.get("data", {}).get("enhanced_csv_path")

        if not baseline_csv or not improved_csv:
            logger.warning("比較用CSVファイルが見つかりません")
            return

        baseline_df = pd.read_csv(baseline_csv)
        improved_df = pd.read_csv(improved_csv)

        # 比較グラフ生成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'深度推定改善比較: {experiment_name}', fontsize=16)

        # 1. 成功率比較
        baseline_success = len(baseline_df[baseline_df['depth_distance'] >= 0]) / len(baseline_df)
        improved_success = len(improved_df[improved_df['depth_distance'] >= 0]) / len(improved_df)

        axes[0, 0].bar(['ベースライン', '改善後'], [baseline_success, improved_success],
                    color=['lightcoral', 'lightblue'])
        axes[0, 0].set_title('深度推定成功率')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].set_ylim(0, 1)

        # 数値ラベル追加
        for i, v in enumerate([baseline_success, improved_success]):
            axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')

        # 2. 平均信頼度比較
        baseline_conf = baseline_df['conf'].mean()
        improved_conf = improved_df['conf'].mean()

        axes[0, 1].bar(['ベースライン', '改善後'], [baseline_conf, improved_conf],
                    color=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('平均検出信頼度')
        axes[0, 1].set_ylabel('信頼度')

        # 3. ゾーン分布比較
        if 'depth_zone' in baseline_df.columns and 'depth_zone' in improved_df.columns:
            baseline_zones = baseline_df['depth_zone'].value_counts(normalize=True)
            improved_zones = improved_df['depth_zone'].value_counts(normalize=True)

            zones = list(set(baseline_zones.index) | set(improved_zones.index))
            baseline_values = [baseline_zones.get(zone, 0) for zone in zones]
            improved_values = [improved_zones.get(zone, 0) for zone in zones]

            x = np.arange(len(zones))
            width = 0.35

            axes[1, 0].bar(x - width/2, baseline_values, width, label='ベースライン', color='lightcoral')
            axes[1, 0].bar(x + width/2, improved_values, width, label='改善後', color='lightblue')
            axes[1, 0].set_title('ゾーン別検出分布')
            axes[1, 0].set_ylabel('相対頻度')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(zones, rotation=45)
            axes[1, 0].legend()

        # 4. 改善効果サマリー
        improvement_metrics = {
            '成功率': improved_success - baseline_success,
            '平均信頼度': improved_conf - baseline_conf,
        }

        metric_names = list(improvement_metrics.keys())
        metric_values = list(improvement_metrics.values())
        colors = ['green' if v > 0 else 'red' for v in metric_values]

        bars = axes[1, 1].bar(metric_names, metric_values, color=colors)
        axes[1, 1].set_title('改善効果')
        axes[1, 1].set_ylabel('改善量')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 数値ラベル
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:+.3f}', ha='center',
                        va='bottom' if height >= 0 else 'top')

        plt.tight_layout()

        # 保存
        output_path = output_dir / f"depth_comparison_{experiment_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ 深度比較可視化保存: {output_path}")

    except Exception as e:
        logger.error(f"深度比較可視化エラー: {e}", exc_info=True)