import cv2

def draw_detections(frame, results, online_targets=None):
    # バウンディングボックス描画
    for r in results:
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
