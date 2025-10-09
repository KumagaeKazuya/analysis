"""
描画ユーティリティモジュール（完全版）
"""

import cv2
import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def draw_detections(frame, results, online_targets=None):
    """
    旧版ByteTracker用の描画関数（後方互換性）
    
    Args:
        frame: 入力フレーム
        results: YOLO推論結果
        online_targets: ByteTrackerの追跡ターゲット
        
    Returns:
        描画済みフレーム
    """
    try:
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if online_targets:
            for t in online_targets:
                x1, y1, x2, y2 = map(int, t.tlbr)
                track_id = t.track_id
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    except Exception as e:
        logger.warning(f"描画エラー: {e}")

    return frame


def draw_detections_ultralytics(frame, results):
    """
    Ultralytics組み込みトラッカー用の描画関数
    
    Args:
        frame: 入力フレーム
        results: YOLO推論結果（トラッキングID付き）
        
    Returns:
        描画済みフレーム
    """
    try:
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()

                if r.boxes.id is not None:
                    track_ids = r.boxes.id.cpu().numpy().astype(int)

                    for box, conf, track_id in zip(boxes, confidences, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"ID:{track_id}", (x1, y1-25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if hasattr(r, 'keypoints') and r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()
                for kpts in keypoints:
                    for x, y in kpts:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
    except Exception as e:
        logger.warning(f"Ultralytics描画エラー: {e}")

    return frame


def draw_detections_enhanced(
    frame: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    tile_sources: Optional[List[int]] = None
) -> np.ndarray:
    """
    拡張版検出結果描画（タイル情報付き）
    
    タイル推論結果を色分けして可視化します。
    
    Args:
        frame: 入力フレーム
        boxes: 検出ボックス (N, 4) [x1, y1, x2, y2]
        confidences: 信頼度スコア (N,)
        tile_sources: タイルソースインデックス (N,) - オプション
        
    Returns:
        描画済みフレーム
    """
    vis_frame = frame.copy()

    # タイル別の色定義（8色）
    tile_colors = [
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青
        (0, 0, 255),    # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄色
        (128, 0, 255),  # 紫
        (255, 128, 0),  # オレンジ
    ]

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        try:
            x1, y1, x2, y2 = map(int, box)

            # タイルソースに応じて色を選択
            if tile_sources and i < len(tile_sources):
                tile_idx = tile_sources[i]
                color = tile_colors[tile_idx % len(tile_colors)]
            else:
                color = (0, 255, 0)  # デフォルト緑

            # バウンディングボックス描画
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # 信頼度表示
            conf_text = f"{conf:.2f}"
            cv2.putText(vis_frame, conf_text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # タイルインデックス表示（タイル推論の場合）
            if tile_sources and i < len(tile_sources):
                tile_text = f"T{tile_sources[i]}"
                cv2.putText(vis_frame, tile_text, (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            logger.warning(f"描画エラー (box {i}): {e}")
            continue

    return vis_frame


def save_comparison_visualization(
    frame: np.ndarray,
    normal_boxes: np.ndarray,
    tile_boxes: np.ndarray,
    output_path: str
) -> None:
    """
    通常推論とタイル推論の比較可視化
    
    2つの結果を左右に並べて保存します。
    
    Args:
        frame: 元フレーム
        normal_boxes: 通常推論のボックス
        tile_boxes: タイル推論のボックス
        output_path: 保存先パス
    """
    try:
        height, width = frame.shape[:2]
        comparison_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)

        # 左側：通常推論結果（緑）
        left_frame = frame.copy()
        if len(normal_boxes) > 0:
            for box in normal_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        comparison_frame[:, :width] = left_frame

        # 右側：タイル推論結果（赤）
        right_frame = frame.copy()
        if len(tile_boxes) > 0:
            for box in tile_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        comparison_frame[:, width:] = right_frame

        # ラベル追加
        cv2.putText(comparison_frame, f"Normal ({len(normal_boxes)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_frame, f"Tile ({len(tile_boxes)})",
                    (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(output_path, comparison_frame)
        logger.info(f"比較画像保存: {output_path}")
    except Exception as e:
        logger.warning(f"比較可視化保存エラー: {e}")


# ========================================
# 統計グラフ生成関数（オプション）
# ========================================

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    def create_detection_statistics_plot(df: pd.DataFrame, output_path: str) -> None:
        """
        検出統計のグラフを生成
        
        Args:
            df: 検出結果のDataFrame（必須カラム: conf, person_id, frame, x1, y1, x2, y2）
            output_path: 保存先パス
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 信頼度分布
        axes[0, 0].hist(df['conf'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('信頼度分布')
        axes[0, 0].set_xlabel('信頼度')
        axes[0, 0].set_ylabel('頻度')
        
        # ID別検出数
        id_counts = df['person_id'].value_counts().head(10)
        axes[0, 1].bar(range(len(id_counts)), id_counts.values, color='green')
        axes[0, 1].set_title('ID別検出数 (上位10)')
        axes[0, 1].set_xlabel('Person ID')
        axes[0, 1].set_ylabel('検出数')
        
        # バウンディングボックスサイズ分布
        df['box_area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
        axes[1, 0].hist(df['box_area'], bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('検出ボックスサイズ分布')
        axes[1, 0].set_xlabel('面積')
        axes[1, 0].set_ylabel('頻度')
        
        # フレーム別検出数
        frame_counts = df.groupby('frame').size()
        axes[1, 1].plot(frame_counts.values, color='purple')
        axes[1, 1].set_title('フレーム別検出数')
        axes[1, 1].set_xlabel('フレーム番号')
        axes[1, 1].set_ylabel('検出数')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"統計グラフ保存: {output_path}")
    
    MATPLOTLIB_AVAILABLE = True

except ImportError:
    logger.info("matplotlib未インストール - グラフ生成機能は利用できません")
    MATPLOTLIB_AVAILABLE = False
    
    def create_detection_statistics_plot(df, output_path):
        logger.warning("matplotlib未インストールのため、グラフ生成をスキップします")


# ========================================
# エクスポート
# ========================================

__all__ = [
    'draw_detections',
    'draw_detections_ultralytics',
    'draw_detections_enhanced',
    'save_comparison_visualization',
    'create_detection_statistics_plot',
    'MATPLOTLIB_AVAILABLE'
]