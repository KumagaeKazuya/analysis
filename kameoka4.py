import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import json
import os
import argparse
import glob

# ========================================
# 手動設定の参照データ（全モニター位置）
# ========================================
REFERENCE_MONITORS = [
    # 上段
    {"bbox": (269, 289, 151, 372), "name": "Monitor_0", "group": "top", "display_bbox": (261, 290, 365, 351)},
    {"bbox": (414, 288, 283, 372), "name": "Monitor_1", "group": "top", "display_bbox": (368, 289, 466, 347)},
    {"bbox": (569, 287, 417, 409), "name": "Monitor_2", "group": "top", "display_bbox": (474, 288, 585, 350)},
    {"bbox": (761, 282, 629, 372), "name": "Monitor_3", "group": "top", "display_bbox": (683, 287, 790, 350)},
    {"bbox": (891, 276, 788, 375), "name": "Monitor_4", "group": "top", "display_bbox": (800, 285, 902, 348)},
    {"bbox": (996, 274, 896, 368), "name": "Monitor_5", "group": "top", "display_bbox": (915, 283, 1017, 341)},
    {"bbox": (1136, 270, 1038, 367), "name": "Monitor_6", "group": "top", "display_bbox": (1037, 281, 1138, 345)},
    {"bbox": (1273, 276, 1177, 364), "name": "Monitor_7", "group": "top", "display_bbox": (1164, 282, 1256, 344)},
    {"bbox": (1397, 268, 1293, 360), "name": "Monitor_8", "group": "top", "display_bbox": (1282, 282, 1376, 344)},
    {"bbox": (1502, 269, 1636, 428), "name": "Monitor_9", "group": "top", "display_bbox": (1511, 275, 1596, 336)},
    {"bbox": (1621, 259, 1799, 352), "name": "Monitor_10", "group": "top", "display_bbox": (1620, 273, 1698, 334)},
    {"bbox": (1788, 255, 1912, 348), "name": "Monitor_11", "group": "top", "display_bbox": (1736, 274, 1802, 330)},
    # 中段
    {"bbox": (384, 372, 144, 614), "name": "Monitor_12", "group": "middle", "display_bbox": (285, 376, 429, 458)},
    {"bbox": (677, 369, 488, 553), "name": "Monitor_13", "group": "middle", "display_bbox": (564, 378, 704, 460)},
    {"bbox": (854, 370, 653, 559), "name": "Monitor_14", "group": "middle", "display_bbox": (713, 375, 854, 457)},
    {"bbox": (1026, 372, 845, 561), "name": "Monitor_15", "group": "middle", "display_bbox": (875, 375, 1019, 459)},
    {"bbox": (1046, 377, 1214, 565), "name": "Monitor_16", "group": "middle", "display_bbox": (1040, 372, 1179, 453)},
    {"bbox": (1213, 374, 1396, 558), "name": "Monitor_17", "group": "middle", "display_bbox": (1212, 376, 1342, 452)},
    {"bbox": (1380, 370, 1608, 548), "name": "Monitor_18", "group": "middle", "display_bbox": (1375, 367, 1511, 456)},
    {"bbox": (1677, 363, 1901, 613), "name": "Monitor_19", "group": "middle", "display_bbox": (1677, 361, 1779, 441)},
    # 下段
    {"bbox": (434, 605, 140, 1023), "name": "Monitor_20", "group": "bottom", "display_bbox": (319, 558, 534, 676)},
    {"bbox": (754, 601, 418, 1015), "name": "Monitor_21", "group": "bottom", "display_bbox": (531, 566, 758, 686)},
    {"bbox": (1005, 634, 676, 1035), "name": "Monitor_22", "group": "bottom", "display_bbox": (781, 570, 1003, 686)},
    {"bbox": (1246, 594, 1001, 1007), "name": "Monitor_23", "group": "bottom", "display_bbox": (1019, 569, 1224, 687)},
    {"bbox": (1265, 594, 1570, 1008), "name": "Monitor_24", "group": "bottom", "display_bbox": (1295, 567, 1474, 677)},
    {"bbox": (1531, 587, 1907, 1018), "name": "Monitor_25", "group": "bottom", "display_bbox": (1545, 563, 1700, 665)},
]

def find_file(base_dir, subdir, ext, name_contains=None):
    pattern = f"*{name_contains if name_contains else ''}.{ext}"
    search_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    files = glob.glob(os.path.join(search_dir, pattern))
    return files[0] if files else None

def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def load_on_monitors(threshold_config_path):
    if not os.path.exists(threshold_config_path):
        print(f"\n【エラー】{threshold_config_path} が見つかりません。")
        print("先に kameoka3.py を実行してください。")
        raise FileNotFoundError(f"{threshold_config_path} not found. Run kameoka3.py first.")
    with open(threshold_config_path, 'r') as f:
        config = json.load(f)
    on_monitors = config.get("on_monitors", [])
    if not on_monitors:
        print("\n【警告】ONモニターが0台です。")
        print("threshold_config.jsonの内容を確認してください。")
        print("全モニターを検出対象にしますか? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            on_monitors = list(range(len(REFERENCE_MONITORS)))
            print(f"全{len(on_monitors)}台のモニターを対象にします。")
        else:
            raise ValueError("ONモニターが指定されていません。処理を中断します。")
    else:
        print(f"\n[情報] ONモニター: {len(on_monitors)}台")
        print(f"  対象モニターID: {on_monitors}")
    return on_monitors

def detect_people_from_video_with_roi(video_path, sample_interval_sec, yolo_model, reference_monitors, roi_margin, roi_confidence, threshold_config_path):
    print(f"\n[検出] 動画から人を検出中（ROI検出モード・ONモニターのみ）...")
    try:
        on_monitors = load_on_monitors(threshold_config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        raise
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    print(f"  動画情報: {duration_sec:.1f}秒, {fps}fps, {total_frames}フレーム")
    sample_interval_frames = int(sample_interval_sec * fps)
    print(f"  YOLOモデル読み込み中: {yolo_model}")
    model = YOLO(yolo_model)
    monitor_detections = {i: [] for i in range(len(reference_monitors))}
    frame_idx = 0
    sample_count = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width = frame.shape[:2]
        frame_detections_per_monitor = {i: [] for i in range(len(reference_monitors))}
        for ref_idx, ref_monitor in enumerate(reference_monitors):
            if ref_idx not in on_monitors:
                continue
            ref_bbox = normalize_bbox(ref_monitor["bbox"])
            x1, y1, x2, y2 = ref_bbox
            w = x2 - x1
            h = y2 - y1
            x1_roi = max(0, int(x1 - w * roi_margin / 2))
            y1_roi = max(0, int(y1 - h * roi_margin / 2))
            x2_roi = min(frame_width, int(x2 + w * roi_margin / 2))
            y2_roi = min(frame_height, int(y2 + h * roi_margin / 2))
            roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            if roi.size == 0:
                continue
            results = model(roi, verbose=False, conf=roi_confidence)
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                        abs_x1 = x1_roi + rx1
                        abs_y1 = y1_roi + ry1
                        abs_x2 = x1_roi + rx2
                        abs_y2 = y1_roi + ry2
                        cx = (abs_x1 + abs_x2) // 2
                        cy = (abs_y1 + abs_y2) // 2
                        width = abs_x2 - abs_x1
                        height = abs_y2 - abs_y1
                        conf = float(box.conf[0])
                        frame_detections_per_monitor[ref_idx].append({
                            "cx": cx,
                            "cy": cy,
                            "width": width,
                            "height": height,
                            "conf": conf,
                        })
        total_detected = 0
        for ref_idx, detections in frame_detections_per_monitor.items():
            if len(detections) > 0:
                best_detection = max(detections, key=lambda d: d["conf"])
                monitor_detections[ref_idx].append({
                    **best_detection,
                    "frame": frame_idx,
                    "ref_monitor_idx": ref_idx,
                    "ref_monitor_name": reference_monitors[ref_idx]["name"]
                })
                total_detected += 1
        sample_count += 1
        print(f"  サンプル {sample_count}: {frame_idx/fps:.1f}秒, 検出: {total_detected}人")
        frame_idx += sample_interval_frames
    cap.release()
    total_detections = sum(len(detections) for detections in monitor_detections.values())
    monitors_with_detection = sum(1 for detections in monitor_detections.values() if len(detections) > 0)
    print(f"  総検出数: {total_detections}件")
    print(f"  検出があったモニター: {monitors_with_detection}/{len(on_monitors)}台")
    return monitor_detections

def generate_monitor_config(video_path, sample_interval_sec, yolo_model, roi_margin, roi_confidence, min_detection_count, threshold_config_path):
    print("=" * 60)
    print("自動bbox生成（ROI検出・ONモニターのみ）")
    print("=" * 60)
    monitor_detections = detect_people_from_video_with_roi(
        video_path,
        sample_interval_sec,
        yolo_model,
        REFERENCE_MONITORS,
        roi_margin,
        roi_confidence,
        threshold_config_path
    )
    print(f"\n[Step 2] 各モニターの検出結果を集計...")
    detected_positions = {}
    for ref_idx, detections in monitor_detections.items():
        if len(detections) >= min_detection_count:
            avg_cx = int(np.mean([d["cx"] for d in detections]))
            avg_cy = int(np.mean([d["cy"] for d in detections]))
            avg_conf = np.mean([d["conf"] for d in detections])
            avg_width = np.mean([d["width"] for d in detections])
            avg_height = np.mean([d["height"] for d in detections])
            ref_bbox = normalize_bbox(REFERENCE_MONITORS[ref_idx]["bbox"])
            ref_cx = (ref_bbox[0] + ref_bbox[2]) // 2
            ref_cy = (ref_bbox[1] + ref_bbox[3]) // 2
            adjusted_cx = int(avg_cx * 0.7 + ref_cx * 0.3)
            adjusted_cy = int(avg_cy * 0.7 + ref_cy * 0.3)
            diff = np.sqrt((avg_cx - ref_cx)**2 + (avg_cy - ref_cy)**2)
            detected_positions[ref_idx] = {
                "center": [adjusted_cx, adjusted_cy],
                "detected_width": avg_width,
                "detected_height": avg_height,
                "detection_count": len(detections),
                "avg_confidence": round(avg_conf, 3),
                "matched_distance": 0.0,
                "has_person": True,
                "detection_method": "roi",
                "center_offset": round(diff, 1)
            }
            monitor_name = REFERENCE_MONITORS[ref_idx]["name"]
            offset_info = f", オフセット={diff:.0f}px" if diff > 80 else ""
            print(f"  {monitor_name}: 検出数={len(detections)}, サイズ({avg_width:.0f}x{avg_height:.0f}), 信頼度={avg_conf:.3f}{offset_info}")
        else:
            if len(detections) > 0:
                print(f"  {REFERENCE_MONITORS[ref_idx]['name']}: 検出数不足 ({len(detections)}回)")
    print(f"\n  人検出によるマッチング: {len(detected_positions)}台")
    print(f"\n[Step 3] 全参照モニターに番号を割り当て...")
    detected_monitors = []
    monitors_with_person = []
    monitors_without_person = []
    for ref_idx, ref_monitor in enumerate(REFERENCE_MONITORS):
        ref_number = int(ref_monitor["name"].split("_")[1])
        ref_bbox = normalize_bbox(ref_monitor["bbox"])
        ref_cx = (ref_bbox[0] + ref_bbox[2]) // 2
        ref_cy = (ref_bbox[1] + ref_bbox[3]) // 2
        if ref_idx in detected_positions:
            detection = detected_positions[ref_idx]
            rx1, ry1, rx2, ry2 = ref_bbox
            w = rx2 - rx1
            h = ry2 - ry1
            bbox_x1 = max(0, int(rx1 - w * roi_margin / 2))
            bbox_y1 = max(0, int(ry1 - h * roi_margin / 2))
            bbox_x2 = int(rx2 + w * roi_margin / 2)
            bbox_y2 = int(ry2 + h * roi_margin / 2)
            final_w = bbox_x2 - bbox_x1
            final_h = bbox_y2 - bbox_y1
            monitor = {
                "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                "display_bbox": list(ref_monitor["display_bbox"]),
                "name": f"Monitor_{ref_number}",
                "group": ref_monitor["group"],
                "center": detection["center"],
                "detection_count": detection["detection_count"],
                "avg_confidence": detection["avg_confidence"],
                "matched_distance": detection["matched_distance"],
                "reference_number": ref_number,
                "has_person": True,
                "detection_method": detection.get("detection_method", "unknown"),
                "yolo_detected_size": {
                    "width": round(detection["detected_width"], 1), 
                    "height": round(detection["detected_height"], 1)
                },
                "final_bbox_size": {"width": final_w, "height": final_h}
            }
            if "center_offset" in detection:
                monitor["center_offset"] = detection["center_offset"]
            monitors_with_person.append(ref_number)
            method = detection.get("detection_method", "unknown")
            print(f"  Monitor_{ref_number}: 人検出あり [{method}] (YOLO size={detection['detected_width']:.0f}x{detection['detected_height']:.0f} → 最終size={final_w}x{final_h})")
        else:
            monitor = {
                "bbox": list(ref_bbox),
                "display_bbox": list(ref_monitor["display_bbox"]),
                "name": f"Monitor_{ref_number}",
                "group": ref_monitor["group"],
                "center": [ref_cx, ref_cy],
                "detection_count": 0,
                "avg_confidence": 0.0,
                "matched_distance": 0.0,
                "reference_number": ref_number,
                "has_person": False
            }
            monitors_without_person.append(ref_number)
            print(f"  Monitor_{ref_number}: 人検出なし（参照位置使用）")
        detected_monitors.append(monitor)
    print(f"\n  人検出あり: {monitors_with_person}")
    print(f"  人検出なし: {monitors_without_person}")
    return {
        "monitors": detected_monitors,
        "monitors_with_person": monitors_with_person,
        "monitors_without_person": monitors_without_person,
        "total_monitors": len(REFERENCE_MONITORS),
        "detected_people": len(detected_positions),
        "config_used": {
            "sample_interval_sec": sample_interval_sec,
            "min_detection_count": min_detection_count,
            "method": "roi_detection_on_monitors_only",
            "roi_margin": roi_margin,
            "roi_confidence": roi_confidence,
        }
    }

def visualize_result(video_path, monitors, monitors_with_person, monitors_without_person, output_image, roi_margin):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("動画からフレームを取得できませんでした")
        return
    colors = {
        "top": (0, 255, 0),
        "middle": (0, 165, 255),
        "bottom": (0, 0, 255)
    }
    frame_height, frame_width = frame.shape[:2]
    for m in monitors:
        color = colors.get(m["group"], (255, 255, 255))
        if not m.get("has_person", False):
            color = tuple(int(c * 0.4) for c in color)
            line_thickness = 1
        else:
            line_thickness = 2
        for ref_monitor in REFERENCE_MONITORS:
            if ref_monitor["name"] == m["name"]:
                ref_bbox = normalize_bbox(ref_monitor["bbox"])
                rx1, ry1, rx2, ry2 = ref_bbox
                w = rx2 - rx1
                h = ry2 - ry1
                roi_x1 = max(0, int(rx1 - w * roi_margin / 2))
                roi_y1 = max(0, int(ry1 - h * roi_margin / 2))
                roi_x2 = min(frame_width, int(rx2 + w * roi_margin / 2))
                roi_y2 = min(frame_height, int(ry2 + h * roi_margin / 2))
                roi_color = tuple(int(c * 0.3) for c in color)
                dash_length = 10
                for x in range(roi_x1, roi_x2, dash_length * 2):
                    cv2.line(frame, (x, roi_y1), (min(x + dash_length, roi_x2), roi_y1), roi_color, 1)
                for x in range(roi_x1, roi_x2, dash_length * 2):
                    cv2.line(frame, (x, roi_y2), (min(x + dash_length, roi_x2), roi_y2), roi_color, 1)
                for y in range(roi_y1, roi_y2, dash_length * 2):
                    cv2.line(frame, (roi_x1, y), (roi_x1, min(y + dash_length, roi_y2)), roi_color, 1)
                for y in range(roi_y1, roi_y2, dash_length * 2):
                    cv2.line(frame, (roi_x2, y), (roi_x2, min(y + dash_length, roi_y2)), roi_color, 1)
                break
        x1, y1, x2, y2 = m["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
        if m["display_bbox"]:
            dx1, dy1, dx2, dy2 = m["display_bbox"]
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 1)
        cx, cy = m["center"]
        cv2.circle(frame, (cx, cy), 4, color, -1)
        label = m['name']
        if not m.get("has_person", False):
            label += " (empty)"
        else:
            if "yolo_detected_size" in m:
                yolo_w = m["yolo_detected_size"]["width"]
                yolo_h = m["yolo_detected_size"]["height"]
                method = m.get("detection_method", "?")[0].upper()
                label += f" [{yolo_w:.0f}x{yolo_h:.0f}]{method}"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, f"Total: {len(monitors)} monitors", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    y_offset = 60
    for group, color in colors.items():
        count_with = sum(1 for m in monitors if m["group"] == group and m.get("has_person", False))
        count_without = sum(1 for m in monitors if m["group"] == group and not m.get("has_person", False))
        cv2.putText(frame, f"{group}: {count_with}+{count_without} (people+empty)", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    cv2.putText(frame, f"With person: {len(monitors_with_person)}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 25
    cv2.putText(frame, f"Empty: {len(monitors_without_person)}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    y_offset += 25
    cv2.putText(frame, f"Method: ROI (ON monitors only)", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    y_offset += 25
    cv2.putText(frame, f"ROI margin: {roi_margin} (dashed lines)", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
    cv2.imwrite(output_image, frame)
    print(f"\n可視化画像保存: {output_image}")

def main():
    parser = argparse.ArgumentParser(description="自動bbox生成スクリプト（ROI検出・ONモニターのみ）")
    parser.add_argument("--project-dir", type=str, required=True, help="タイムスタンプ付きプロジェクトフォルダ")
    parser.add_argument("--yolo-model", type=str, default="yolo11m.pt", help="YOLOモデルファイルパス")
    parser.add_argument("--sample-interval-sec", type=int, default=15, help="サンプリング間隔（秒）")
    parser.add_argument("--min-detection-count", type=int, default=1, help="検出最低回数")
    parser.add_argument("--roi-margin", type=float, default=0.25, help="ROIマージン（割合）")
    parser.add_argument("--roi-confidence", type=float, default=0.1, help="YOLO検出信頼度閾値")
    args = parser.parse_args()

    base_dir = args.project_dir
    json_dir = os.path.join(base_dir, "json")
    img_dir = os.path.join(base_dir, "img")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    output_json = os.path.join(json_dir, "monitor_config.json")
    output_image = os.path.join(img_dir, "detected_monitors.png")

    # 必要なファイルを自動探索
    input_video = find_file(base_dir, "video", "mp4", "output_corrected")
    threshold_config = find_file(base_dir, "json", "json", "threshold_config")

    print("=" * 60)
    print("自動bbox生成スクリプト（ROI検出・ONモニターのみ）")
    print("=" * 60)

    if not input_video or not os.path.exists(threshold_config):
        print(f"\n【エラー】必要なファイルが見つかりません")
        print(f"input_video: {input_video}")
        print(f"threshold_config: {threshold_config}")
        return

    try:
        result = generate_monitor_config(
            input_video,
            args.sample_interval_sec,
            args.yolo_model,
            args.roi_margin,
            args.roi_confidence,
            args.min_detection_count,
            threshold_config
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n処理を中断しました: {e}")
        return

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nJSON保存: {output_json}")

    visualize_result(
        input_video, 
        result["monitors"], 
        result["monitors_with_person"],
        result["monitors_without_person"],
        output_image,
        args.roi_margin
    )

    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"総モニター数: {result['total_monitors']}台")
    print(f"人検出あり: {len(result['monitors_with_person'])}台")
    print(f"人検出なし: {len(result['monitors_without_person'])}台")

    print(f"使用方法: ROI検出（ONモニターのみ, 信頼度={args.roi_confidence}, マージン={args.roi_margin}）")

    for group in ["top", "middle", "bottom"]:
        count_with = sum(1 for m in result["monitors"] if m["group"] == group and m.get("has_person", False))
        count_without = sum(1 for m in result["monitors"] if m["group"] == group and not m.get("has_person", False))
        print(f"  {group}: {count_with}人 + {count_without}空席")

    print(f"\n人検出ありのモニター: {result['monitors_with_person']}")
    print(f"人検出なしのモニター: {result['monitors_without_person']}")

    yolo_sizes = []
    for m in result["monitors"]:
        if m.get("has_person") and "yolo_detected_size" in m:
            yolo_sizes.append(m["yolo_detected_size"])

    if yolo_sizes:
        print(f"\nYOLO検出サイズ統計:")
        widths = [s["width"] for s in yolo_sizes]
        heights = [s["height"] for s in yolo_sizes]
        print(f"  幅: 平均={np.mean(widths):.1f}px, 最小={np.min(widths):.1f}px, 最大={np.max(widths):.1f}px")
        print(f"  高さ: 平均={np.mean(heights):.1f}px, 最小={np.min(heights):.1f}px, 最大={np.max(heights):.1f}px")

    print("\n" + "=" * 60)
    print("完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()