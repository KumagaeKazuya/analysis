"""
描画ユーティリティモジュール
"""

import cv2
import numpy as np
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def draw_detections(frame, results, online_targets=None):
    """旧版ByteTracker用の描画関数（後方互換性）"""
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
    """Ultralytics組み込みトラッカー用の描画関数"""
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


def draw_detections_enhanced(frame: np.ndarray,
                            boxes: np.ndarray,
                            confidences: np.ndarray,
                            tile_sources: Optional[List[int]] = None) -> np.ndarray:
    """拡張版検出結果描画（タイル情報付き）"""
    vis_frame = frame.copy()

    tile_colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    ]

    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        try:
            x1, y1, x2, y2 = map(int, box)

            if tile_sources and i < len(tile_sources):
                tile_idx = tile_sources[i]
                color = tile_colors[tile_idx % len(tile_colors)]
            else:
                color = (0, 255, 0)

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if tile_sources and i < len(tile_sources):
                tile_text = f"T{tile_sources[i]}"
                cv2.putText(vis_frame, tile_text, (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            logger.warning(f"描画エラー (box {i}): {e}")
            continue

    return vis_frame