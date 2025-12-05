import cv2
import numpy as np
import logging
import json
import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque
import argparse

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
    parser = argparse.ArgumentParser(description="モニターROI抽出スクリプト（プロジェクトフォルダ指定）")
    parser.add_argument("--project-dir", type=str, required=True, help="タイムスタンプ付きプロジェクトフォルダ")
    parser.add_argument("--show-preview", action="store_true", help="処理中にプレビュー表示")
    parser.add_argument("--save-rois", action="store_true", help="ROI画像を保存する")
    parser.add_argument("--save-interval", type=int, default=40, help="ROI画像保存間隔（フレーム数）")
    return parser.parse_args()

POWER_DETECTION_CONFIG = {
    "smoothing_alpha": 0.5,
    "stability_window": 15,
    "on_threshold": 0.6,
    "confidence_distance": 50,
}

ROI_EXTRACTION_CONFIG = {
    "enabled": True,
    "padding": 20,
    "min_roi_size": (50, 50),
    "max_roi_size": (500, 500),
}

@dataclass
class MonitorROI:
    monitor_id: int
    monitor_name: str
    bbox: Tuple[int, int, int, int]
    roi_image: np.ndarray
    is_powered_on: bool
    confidence: float

@dataclass
class MonitorInfo:
    id: int
    name: str
    bbox: Tuple[int, int, int, int]
    group: str
    center: Tuple[int, int]
    display_bbox: Optional[Tuple[int, int, int, int]] = None
    mean_brightness: float = -1.0
    is_powered_on: bool = False
    confidence: float = 0.0
    power_history: Optional[deque] = None
    brightness_history: Optional[List[float]] = None

    def __post_init__(self):
        if self.power_history is None:
            self.power_history = deque(maxlen=50)
        if self.brightness_history is None:
            self.brightness_history = []

    def to_dict_for_json(self) -> Dict:
        return {
            "id": int(self.id),
            "name": str(self.name),
            "bbox": list(self.bbox),
            "group": str(self.group),
            "is_powered_on": bool(self.is_powered_on),
            "confidence": float(self.confidence),
            "mean_brightness": float(self.mean_brightness),
            "display_bbox": list(self.display_bbox) if self.display_bbox else None,
        }

def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def load_monitor_config(config_path):
    if not os.path.exists(config_path):
        logger.error(f"モニター設定ファイルが見つかりません: {config_path}")
        logger.error("先にauto_generate_bbox.pyを実行してください")
        raise FileNotFoundError(f"{config_path} が見つかりません")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"モニター設定を読み込みました: {config_path}")
    logger.info(f" モニター数: {len(config['monitors'])}")
    return config

def load_threshold_config(config_path):
    if not os.path.exists(config_path):
        logger.error(f"閾値設定ファイルが見つかりません: {config_path}")
        logger.error("先にcalculate_threshold.pyを実行してください")
        raise FileNotFoundError(f"{config_path} が見つかりません")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    threshold = config.get('threshold', 76)
    method = config.get('method', config.get('detection_method', 'unknown'))
    logger.info(f"閾値設定を読み込みました: {config_path}")
    logger.info(f" 閾値: {threshold}")
    logger.info(f" メソッド: {method}")
    return threshold, method

class MonitorDetectionSystem:
    def __init__(self, monitor_config: Dict, threshold: int, threshold_method: str):
        self.monitor_config = monitor_config
        self.detected_monitors: List[MonitorInfo] = []
        self.power_config = POWER_DETECTION_CONFIG
        self.roi_config = ROI_EXTRACTION_CONFIG
        self.frame_counter = 0
        self.threshold = threshold
        self.threshold_method = threshold_method

        self._initialize_monitors()
        logger.info(f"モニター登録: {len(self.detected_monitors)}台")
        logger.info(f"使用閾値: {threshold} (方法: {threshold_method})")

    def _initialize_monitors(self):
        for i, m_cfg in enumerate(self.monitor_config["monitors"]):
            bbox = normalize_bbox(tuple(m_cfg["bbox"]))
            display_bbox = m_cfg.get("display_bbox", None)
            if display_bbox:
                display_bbox = normalize_bbox(tuple(display_bbox))
            monitor = MonitorInfo(
                id=i,
                name=m_cfg["name"],
                bbox=bbox,
                group=m_cfg.get("group", "middle"),
                center=tuple(m_cfg.get("center", [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])),
                display_bbox=display_bbox,
            )
            self.detected_monitors.append(monitor)

    def calculate_metrics(self, frame: np.ndarray, monitor: MonitorInfo) -> Dict:
        bbox_to_use = monitor.display_bbox if monitor.display_bbox is not None else monitor.bbox
        if bbox_to_use is None:
            return {'mean_brightness': 0.0}
        x1, y1, x2, y2 = bbox_to_use
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return {'mean_brightness': 0.0}
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {'mean_brightness': 0.0}
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        monitor.brightness_history.append(mean_brightness)
        return {'mean_brightness': mean_brightness}

    def judge_power_state(self, monitor: MonitorInfo) -> Tuple[bool, float]:
        is_on = monitor.mean_brightness > self.threshold
        distance = abs(monitor.mean_brightness - self.threshold)
        confidence = min(distance / self.power_config["confidence_distance"], 1.0)
        return is_on, confidence

    def update_all_monitors(self, frame: np.ndarray):
        self.frame_counter += 1
        alpha = self.power_config["smoothing_alpha"]
        for monitor in self.detected_monitors:
            metrics = self.calculate_metrics(frame, monitor)
            is_first_frame = (monitor.mean_brightness < 0)
            if is_first_frame:
                monitor.mean_brightness = metrics['mean_brightness']
            else:
                monitor.mean_brightness = alpha * metrics['mean_brightness'] + (1 - alpha) * monitor.mean_brightness
            is_on, confidence = self.judge_power_state(monitor)
            monitor.confidence = confidence
            monitor.power_history.append(is_on)
            window = self.power_config["stability_window"]
            if len(monitor.power_history) >= window:
                recent = list(monitor.power_history)[-window:]
                on_ratio = sum(recent) / len(recent)
                threshold = self.power_config["on_threshold"]
                monitor.is_powered_on = on_ratio > threshold
            else:
                monitor.is_powered_on = is_on

    def get_powered_on_monitors(self) -> List[MonitorInfo]:
        return [m for m in self.detected_monitors if m.is_powered_on]

    def extract_monitor_rois(self, frame: np.ndarray) -> List[MonitorROI]:
        if not self.roi_config["enabled"]:
            return []
        rois = []
        padding = self.roi_config["padding"]
        min_w, min_h = self.roi_config["min_roi_size"]
        max_w, max_h = self.roi_config["max_roi_size"]
        h, w = frame.shape[:2]
        for monitor in self.get_powered_on_monitors():
            x1, y1, x2, y2 = monitor.bbox
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w, x2 + padding)
            y2_pad = min(h, y2 + padding)
            roi_w = x2_pad - x1_pad
            roi_h = y2_pad - y1_pad
            if roi_w < min_w or roi_h < min_h or roi_w > max_w or roi_h > max_h:
                continue
            roi_image = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            roi_info = MonitorROI(
                monitor_id=monitor.id,
                monitor_name=monitor.name,
                bbox=(x1_pad, y1_pad, x2_pad, y2_pad),
                roi_image=roi_image,
                is_powered_on=monitor.is_powered_on,
                confidence=monitor.confidence
            )
            rois.append(roi_info)
        return rois

    def get_monitor_states_for_json(self) -> List[Dict]:
        return [m.to_dict_for_json() for m in self.detected_monitors]

    def draw_monitors(self, frame: np.ndarray) -> np.ndarray:
        for monitor in self.detected_monitors:
            if monitor.is_powered_on:
                if monitor.display_bbox is not None:
                    dx1, dy1, dx2, dy2 = monitor.display_bbox
                else:
                    dx1, dy1, dx2, dy2 = monitor.bbox
                color = (0, 255, 0)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 3)
                monitor_text = f"{monitor.id}"
                cv2.putText(frame, monitor_text,
                            (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                brightness_text = f"{monitor.mean_brightness:.0f}"
                cv2.putText(frame, brightness_text,
                            (dx1, dy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

class VideoAnalysisSystem:
    def __init__(self, monitor_config: Dict, threshold: int, threshold_method: str, video_config, power_config, roi_config):
        self.monitor_detector = MonitorDetectionSystem(monitor_config, threshold, threshold_method)
        self.frame_counter = 0
        self.json_file = None
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.video_config = video_config
        self.power_config = power_config
        self.roi_config = roi_config

        if self.video_config["save_rois"]:
            os.makedirs(self.video_config["roi_output_dir"], exist_ok=True)
        self.json_file = open(self.video_config["monitor_states_json"], 'w', encoding='utf-8')
        logger.info("動画処理システム初期化完了")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_config["input_path"])
        if not cap.isOpened():
            logger.error(f"動画はありません: {self.video_config['input_path']}")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"動画情報: {width}x{height}, {fps}FPS, {total_frames}フレーム")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_config["output_path"], fourcc, fps, (width, height))
        roi_stats = {"total_extracted": 0, "by_monitor": {}}
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_counter += 1
                self.monitor_detector.update_all_monitors(frame)
                self._save_monitor_states()
                rois = self.monitor_detector.extract_monitor_rois(frame)
                if self.video_config["save_rois"] and self.frame_counter % self.video_config["save_interval"] == 0:
                    self._save_rois(rois, self.frame_counter)
                roi_stats["total_extracted"] += len(rois)
                for roi in rois:
                    if roi.monitor_name not in roi_stats["by_monitor"]:
                        roi_stats["by_monitor"][roi.monitor_name] = 0
                    roi_stats["by_monitor"][roi.monitor_name] += 1
                processed = self.monitor_detector.draw_monitors(frame.copy())
                processed = self._draw_info(processed, rois)
                out.write(processed)
                if self.video_config["show_preview"]:
                    cv2.imshow('モニター検出', cv2.resize(processed, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if self.frame_counter % (fps * 10) == 0:
                    progress = (self.frame_counter / total_frames) * 100
                    on_count = sum(1 for m in self.monitor_detector.detected_monitors if m.is_powered_on)
                    logger.info(f"進捗: {progress:.1f}% | ON: {on_count}/{len(self.monitor_detector.detected_monitors)} | ROI: {len(rois)}")
        finally:
            cap.release()
            out.release()
            if self.json_file:
                self.json_file.close()
            cv2.destroyAllWindows()
            self._print_statistics(roi_stats)

    def _save_monitor_states(self):
        state_data = {
            "frame": self.frame_counter,
            "monitors": self.monitor_detector.get_monitor_states_for_json()
        }
        self.json_file.write(json.dumps(state_data, ensure_ascii=False) + '\n')

    def _save_rois(self, rois: List[MonitorROI], frame_num: int):
        for roi in rois:
            filename = f"{self.video_config['roi_output_dir']}/frame{frame_num:06d}_{roi.monitor_name}.jpg"
            cv2.imwrite(filename, roi.roi_image)

    def _draw_info(self, frame: np.ndarray, rois: List[MonitorROI]) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y = w - 550, 30
        cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, 10), (w - 10, y + 160), (255, 255, 255), 1)
        cv2.putText(frame, f"フレーム: {self.frame_counter}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        total = len(self.monitor_detector.detected_monitors)
        on_count = sum(1 for m in self.monitor_detector.detected_monitors if m.is_powered_on)
        cv2.putText(frame, f"モニター: {total}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(frame, f"ON: {on_count} / OFF: {total - on_count}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        cv2.putText(frame, f"ROI: {len(rois)}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        method_short = {
            'all_off': 'すべてOFF', 'all_on': 'すべてON', 'few_on': '少数ON',
            'bimodal': 'バイモーダル', 'percentile': 'パーセンタイル', 'default': 'デフォルト'
        }
        method_text = method_short.get(self.threshold_method, '自動')
        cv2.putText(frame, f"しきい値: {self.threshold} ({method_text})", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 20
        cv2.putText(frame, f"JSONに保存しています...", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return frame

    def _print_statistics(self, roi_stats: Dict):
        logger.info("=" * 100)
        logger.info("処理完了統計")
        logger.info("=" * 100)
        monitors = self.monitor_detector.detected_monitors
        on_count = sum(1 for m in monitors if m.is_powered_on)
        logger.info(f"使用閾値: {self.threshold} (方法: {self.threshold_method})")
        logger.info(f"最終結果: ON={on_count}台 / OFF={len(monitors) - on_count}台")
        logger.info(f"総ROI抽出数: {roi_stats['total_extracted']}")
        logger.info(f"平均ROI数/フレーム: {roi_stats['total_extracted'] / self.frame_counter:.2f}")
        logger.info("\n明るさ統計:")
        on_brightness = []
        off_brightness = []
        for monitor in monitors:
            if len(monitor.brightness_history) > 0:
                avg_bright = np.mean(monitor.brightness_history)
                if monitor.is_powered_on:
                    on_brightness.append(avg_bright)
                else:
                    off_brightness.append(avg_bright)
        if len(on_brightness) > 0:
            logger.info(f" ONモニター: 平均={np.mean(on_brightness):.1f}, 範囲={np.min(on_brightness):.1f}～{np.max(on_brightness):.1f}")
        else:
            logger.info(f" ONモニター:なし")
        if len(off_brightness) > 0:
            logger.info(f" OFFモニター: 平均={np.mean(off_brightness):.1f}, 範囲={np.min(off_brightness):.1f}～{np.max(off_brightness):.1f}")
        else:
            logger.info(f" OFFモニター:なし")
        logger.info("\nモニター別ROI抽出回数:")
        if len(roi_stats['by_monitor']) > 0:
            for monitor_name, count in sorted(roi_stats['by_monitor'].items())[:10]:
                logger.info(f" {monitor_name}: {count}回")
            if len(roi_stats['by_monitor']) > 10:
                logger.info(f" ... 他{len(roi_stats['by_monitor'])-10}台")
        else:
            logger.info(f" なし（全モニターOFF）")
        logger.info("=" * 100)
        logger.info(f"出力ファイル:")
        logger.info(f" - 動画: {self.video_config['output_path']}")
        logger.info(f" - JSON: {self.video_config['monitor_states_json']}")
        logger.info(f" - ROI画像: {self.video_config['roi_output_dir']}/")
        logger.info("=" * 100)

def main():
    args = parse_args()
    base_dir = args.project_dir

    # 入力ファイル探索
    input_video = find_file(base_dir, "video", "mp4")
    monitor_config_json = find_file(base_dir, "json", "json", "monitor_config")
    threshold_config_json = find_file(base_dir, "json", "json", "threshold_config")
    output_video = os.path.join(base_dir, "video", "output_monitor_rois.mp4")
    roi_output_dir = os.path.join(base_dir, "roi")
    monitor_states_json = os.path.join(base_dir, "json", "monitor_states.json")

    # save_intervalを40に固定
    VIDEO_CONFIG = {
        "input_path": input_video,
        "output_path": output_video,
        "roi_output_dir": roi_output_dir,
        "monitor_states_json": monitor_states_json,
        "threshold_config_json": threshold_config_json,
        "monitor_config_json": monitor_config_json,
        "show_preview": args.show_preview,
        "save_rois": args.save_rois,
        "save_interval": 40,  # ここを40に固定
    }

    if not input_video or not os.path.exists(monitor_config_json) or not os.path.exists(threshold_config_json):
        logger.error("必要なファイルが見つかりません")
        logger.error(f"input_video: {input_video}")
        logger.error(f"monitor_config_json: {monitor_config_json}")
        logger.error(f"threshold_config_json: {threshold_config_json}")
        return

    try:
        monitor_config = load_monitor_config(VIDEO_CONFIG["monitor_config_json"])
        threshold, method = load_threshold_config(VIDEO_CONFIG["threshold_config_json"])
        system = VideoAnalysisSystem(
            monitor_config, threshold, method,
            VIDEO_CONFIG, POWER_DETECTION_CONFIG, ROI_EXTRACTION_CONFIG
        )
        system.process_video()
    except FileNotFoundError:
        logger.error("処理を中断しました")
        return
    except KeyboardInterrupt:
        logger.info("\n中断されました")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()