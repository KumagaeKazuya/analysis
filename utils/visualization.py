import cv2
import numpy as np

def draw_detections(frame, results, online_targets=None):
    """旧版のBYTETracker用の描画関数（後方互換性のため保持）"""
    # バウンディングボックス描画
    for r in results:
        if r.boxes is not None:
            for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ID表示
    if online_targets:
        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame

def draw_detections_ultralytics(frame, results):
    """Ultralytics組み込みトラッカー用の描画関数"""
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            # トラックIDがある場合
            if r.boxes.id is not None:
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for box, conf, track_id in zip(boxes, confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # バウンディングボックス描画
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 信頼度表示
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # トラックID表示
                    cv2.putText(frame, f"ID:{track_id}", (x1, y1-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # トラックIDがない場合（通常の検出結果）
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # キーポイントがある場合の描画
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            for kpts in keypoints:
                for x, y in kpts:
                    if x > 0 and y > 0:  # 有効なキーポイントのみ
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    return frame