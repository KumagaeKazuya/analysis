import cv2
import numpy as np
import json
import logging
import os
import argparse
import glob
from typing import List, Dict, Tuple
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def find_file(base_dir, subdir, ext, name_contains=None):
    pattern = f"*{name_contains if name_contains else ''}.{ext}"
    search_dir = os.path.join(base_dir, subdir) if subdir else base_dir
    files = glob.glob(os.path.join(search_dir, pattern))
    return files[0] if files else None

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOpose + モニターROI検出システム")
    parser.add_argument("--project-dir", type=str, required=True, help="kameoka1.pyで作成したプロジェクトフォルダ")
    parser.add_argument("--show-preview", action="store_true", help="処理中にプレビュー表示")
    parser.add_argument("--save-debug-rois", action="store_true", help="ROI画像を保存する")
    return parser.parse_args()

class MonitorPoseDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.monitor_states = self._load_monitor_states()
        self.model = YOLO(config["yolo_model"])
        self.frame_count = 0
        
        if config.get("save_debug_rois", False):
            os.makedirs(config.get("debug_roi_dir", "debug_roi"), exist_ok=True)
        
        logger.info(f"YOLOモデル読み込み: {config['yolo_model']}")
        logger.info(f"モニター状態データ: {len(self.monitor_states)}フレーム分")
        logger.info(f"信頼度閾値: {config.get('confidence_threshold', 0.25)}")
        logger.info(f"ROI padding: {config.get('roi_padding', 0)}px")
    
    def _load_monitor_states(self) -> Dict[int, List[Dict]]:
        states = {}
        with open(self.config["monitor_states_json"], 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                frame_num = data["frame"]
                states[frame_num] = data["monitors"]
        return states
    
    def get_active_monitors(self, frame_num: int) -> List[Dict]:
        if frame_num not in self.monitor_states:
            return []
        return [m for m in self.monitor_states[frame_num] if m["is_powered_on"]]
    
    def detect_poses_in_rois(self, frame: np.ndarray, frame_num: int) -> List[Dict]:
        active_monitors = self.get_active_monitors(frame_num)
        all_detections = []
        h, w = frame.shape[:2]
        padding = self.config.get("roi_padding", 0)
        small_padding = self.config.get("small_monitor_padding", 0)
        small_threshold = self.config.get("small_monitor_threshold", 100)
        for monitor in active_monitors:
            bbox = monitor["bbox"]
            x1, y1, x2, y2 = bbox
            roi_w = x2 - x1
            roi_h = y2 - y1
            current_padding = padding
            if roi_w < small_threshold or roi_h < small_threshold:
                current_padding = small_padding
            x1_pad = max(0, x1 - current_padding)
            y1_pad = max(0, y1 - current_padding)
            x2_pad = min(w, x2 + current_padding)
            y2_pad = min(h, y2 + current_padding)
            if x2_pad <= x1_pad or y2_pad <= y1_pad:
                continue
            roi = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            if roi.size == 0:
                continue
            conf = self.config.get("confidence_threshold", 0.25)
            results = self.model(roi, conf=conf, verbose=False)
            monitor_detections = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                if result.keypoints is not None and len(result.keypoints) > 0:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    boxes_data = result.boxes.data.cpu().numpy()
                    for keypoints, box in zip(keypoints_data, boxes_data):
                        detection = {
                            "monitor_id": monitor["id"],
                            "monitor_name": monitor["name"],
                            "bbox": bbox,
                            "person_box": [
                                float(box[0]) + x1_pad,
                                float(box[1]) + y1_pad,
                                float(box[2]) + x1_pad,
                                float(box[3]) + y1_pad
                            ],
                            "confidence": float(box[4]),
                            "keypoints": []
                        }
                        for kp in keypoints:
                            detection["keypoints"].append({
                                "x": float(kp[0]) + x1_pad,
                                "y": float(kp[1]) + y1_pad,
                                "confidence": float(kp[2])
                            })
                        monitor_detections.append(detection)
                else:
                    boxes_data = result.boxes.data.cpu().numpy()
                    for box in boxes_data:
                        detection = {
                            "monitor_id": monitor["id"],
                            "monitor_name": monitor["name"],
                            "bbox": bbox,
                            "person_box": [
                                float(box[0]) + x1_pad,
                                float(box[1]) + y1_pad,
                                float(box[2]) + x1_pad,
                                float(box[3]) + y1_pad
                            ],
                            "confidence": float(box[4]),
                            "keypoints": []
                        }
                        monitor_detections.append(detection)
            if len(monitor_detections) > 1:
                best_detection = None
                max_area = 0
                for det in monitor_detections:
                    x1, y1, x2, y2 = det["person_box"]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_detection = det
                if best_detection:
                    all_detections.append(best_detection)
            elif len(monitor_detections) == 1:
                all_detections.append(monitor_detections[0])
        return all_detections
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict], active_monitors: List[Dict]) -> np.ndarray:
        result_frame = frame.copy()
        if self.config["draw_monitor_boxes"]:
            for monitor in active_monitors:
                x1, y1, x2, y2 = monitor["bbox"]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_frame, f"M{monitor['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for detection in detections:
            x1, y1, x2, y2 = detection["person_box"]
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"M{detection['monitor_id']} {detection['confidence']:.2f}"
            cv2.putText(result_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            keypoints = detection["keypoints"]
            if len(keypoints) == 0:
                continue
            for kp in keypoints:
                if kp["confidence"] > 0.5:
                    cv2.circle(result_frame, (int(kp["x"]), int(kp["y"])), 3, (0, 255, 255), -1)
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            for connection in skeleton:
                idx1, idx2 = connection[0] - 1, connection[1] - 1
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    kp1, kp2 = keypoints[idx1], keypoints[idx2]
                    if kp1["confidence"] > 0.5 and kp2["confidence"] > 0.5:
                        cv2.line(result_frame, (int(kp1["x"]), int(kp1["y"])), (int(kp2["x"]), int(kp2["y"])), (0, 255, 0), 2)
        return result_frame
    
    def draw_info(self, frame: np.ndarray, detections: List[Dict], active_count: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y = w - 400, 30
        cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 100), (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"Active Monitors: {active_count}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(frame, f"People Detected: {len(detections)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y += 25
        cv2.putText(frame, f"Model: {self.config['yolo_model']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return frame
    
    def process_video(self):
        cap = cv2.VideoCapture(self.config["input_video"])
        if not cap.isOpened():
            logger.error(f"動画を開けません: {self.config['input_video']}")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"動画: {width}x{height}, {fps}FPS, {total_frames}フレーム")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.config["output_video"], fourcc, fps, (width, height))
        total_detections = 0
        detection_by_monitor = {}
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1
                detections = self.detect_poses_in_rois(frame, self.frame_count)
                active_monitors = self.get_active_monitors(self.frame_count)
                total_detections += len(detections)
                for det in detections:
                    monitor_name = det["monitor_name"]
                    if monitor_name not in detection_by_monitor:
                        detection_by_monitor[monitor_name] = 0
                    detection_by_monitor[monitor_name] += 1
                result = self.draw_results(frame, detections, active_monitors)
                result = self.draw_info(result, detections, len(active_monitors))
                out.write(result)
                if self.config["show_preview"]:
                    cv2.imshow('YOLOpose on Monitor ROIs', cv2.resize(result, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if self.frame_count % (fps * 10) == 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"進捗: {progress:.1f}% | 検出: {len(detections)}人 | 累計: {total_detections}人")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            self._print_statistics(total_detections, detection_by_monitor)
    
    def _print_statistics(self, total: int, by_monitor: Dict):
        logger.info("=" * 80)
        logger.info("検出統計")
        logger.info("=" * 80)
        logger.info(f"総検出数: {total}人")
        logger.info(f"平均検出数/フレーム: {total / self.frame_count:.2f}人")
        if by_monitor:
            logger.info("\nモニター別検出数:")
            for monitor_name, count in sorted(by_monitor.items()):
                logger.info(f"  {monitor_name}: {count}人")
        logger.info("=" * 80)
        logger.info(f"出力動画: {self.config['output_video']}")
        logger.info("=" * 80)

def main():
    args = parse_args()
    base_dir = args.project_dir
    input_video = find_file(base_dir, "video", "mp4", "output_corrected")
    monitor_states_json = find_file(base_dir, "json", "json", "monitor_states")
    output_video = os.path.join(base_dir, "video", "output_with_pose.mp4")
    debug_roi_dir = os.path.join(base_dir, "debug_roi")
    yolo_model = "yolo11m-pose.pt"
    config = {
        "input_video": input_video,
        "output_video": output_video,
        "monitor_states_json": monitor_states_json,
        "yolo_model": yolo_model,
        "confidence_threshold": 0.15,
        "roi_padding": 0,
        "small_monitor_padding": 0,
        "small_monitor_threshold": 100,
        "one_person_per_monitor": False,
        "show_preview": args.show_preview,
        "draw_monitor_boxes": True,
        "save_debug_rois": args.save_debug_rois,
        "debug_roi_dir": debug_roi_dir,
    }
    logger.info("=" * 80)
    logger.info("YOLOpose + モニターROI検出システム")
    logger.info("=" * 80)
    if not input_video or not os.path.exists(monitor_states_json):
        logger.error("必要なファイルが見つかりません")
        logger.error(f"input_video: {input_video}")
        logger.error(f"monitor_states_json: {monitor_states_json}")
        return
    try:
        detector = MonitorPoseDetector(config)
        detector.process_video()
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()