"""
深度推定プロセッサ（MiDaS統合版）
教室後方カメラでの単眼深度推定に特化
"""

import torch
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from utils.error_handler import (
    ModelInitializationError,
    VideoProcessingError,
    handle_errors,
    ErrorContext,
    ErrorCategory,
    ResponseBuilder
)

logger = logging.getLogger(__name__)

class ClassroomDepthProcessor:
    """教室環境特化の深度推定プロセッサー"""

    @handle_errors(logger=logger, error_category=ErrorCategory.INITIALIZATION)
    def __init__(self, config):
        """
        初期化

        Args:
            config: 設定オブジェクト
        """
        with ErrorContext("深度推定プロセッサー初期化", logger=logger) as ctx:
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.transform = None

            # 教室環境パラメータ
            depth_config = config.get('processing.depth_estimation', {})
            self.camera_height = depth_config.get('camera_height', 2.8)
            self.camera_angle = depth_config.get('camera_angle', 25)
            self.classroom_mode = depth_config.get('classroom_mode', True)

            ctx.add_info("device", str(self.device))
            ctx.add_info("classroom_mode", self.classroom_mode)
            ctx.add_info("camera_height", self.camera_height)

            self._initialize_model()

    @handle_errors(logger=logger, error_category=ErrorCategory.MODEL)
    def _initialize_model(self):
        """MiDaSモデルの初期化"""
        with ErrorContext("MiDaS模型初期化", logger=logger) as ctx:
            try:
                # MiDaSモデルのロード
                model_type = self.config.get('processing.depth_estimation.model_size', 'small')

                if model_type == 'small':
                    self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
                    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                    self.transform = midas_transforms.small_transform
                else:
                    self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
                    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                    self.transform = midas_transforms.default_transform

                self.model.to(self.device)
                self.model.eval()

                ctx.add_info("model_type", model_type)
                ctx.add_info("device", str(self.device))

                logger.info(f"✅ MiDaS {model_type}モデル初期化完了")

            except Exception as e:
                raise ModelInitializationError(
                    f"MiDaS初期化失敗: {e}",
                    details={
                        "model_type": model_type,
                        "device": str(self.device),
                        "torch_version": torch.__version__
                    },
                    suggestions=[
                        "インターネット接続を確認してください",
                        "torch hubのキャッシュをクリアしてください: torch.hub.set_dir('/tmp')"
                    ],
                    original_exception=e
                )

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        画像から深度マップを推定

        Args:
            image: 入力画像 (BGR)

        Returns:
            深度マップ (0-255の範囲で正規化済み)
        """
        with ErrorContext("深度推定処理", logger=logger) as ctx:
            if image is None or image.size == 0:
                raise VideoProcessingError("入力画像が無効です")

            ctx.add_info("image_shape", image.shape)

            # RGB変換とリサイズ
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 前処理
            input_batch = self.transform(rgb_image).to(self.device)

            # 推論
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # CPUに移動して後処理
            depth_map = prediction.cpu().numpy()

            # 教室環境での補正
            if self.classroom_mode:
                depth_map = self._apply_classroom_correction(depth_map)

            # 0-255範囲に正規化
            depth_map_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            ctx.add_info("depth_range", f"{depth_map.min():.2f} - {depth_map.max():.2f}")

            return depth_map_normalized

    def _apply_classroom_correction(self, depth_map: np.ndarray) -> np.ndarray:
        """教室環境での深度補正"""
        # 俯瞰角度補正
        angle_rad = np.radians(self.camera_angle)
        angle_correction = np.cos(angle_rad)

        # 高さ補正
        height_factor = self.camera_height / np.sin(angle_rad)

        # 線形補正適用
        corrected = depth_map * angle_correction

        return corrected

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def estimate_object_distance(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        バウンディングボックス内の距離推定

        Args:
            depth_map: 深度マップ
            bbox: (x1, y1, x2, y2)

        Returns:
            距離情報の辞書
        """
        with ErrorContext("物体距離推定", logger=logger) as ctx:
            x1, y1, x2, y2 = bbox

            # 境界チェック
            h, w = depth_map.shape
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                return {"distance": -1, "zone": "invalid", "confidence": 0.0}

            # ROI抽出
            roi_depth = depth_map[y1:y2, x1:x2]

            if roi_depth.size == 0:
                return {"distance": -1, "zone": "invalid", "confidence": 0.0}

            # 中央部分の深度値を使用（より安定）
            h_roi, w_roi = roi_depth.shape
            center_roi = roi_depth[h_roi//4:3*h_roi//4, w_roi//4:3*w_roi//4]

            if center_roi.size == 0:
                center_roi = roi_depth

            # 統計計算
            median_distance = float(np.median(center_roi))
            mean_distance = float(np.mean(center_roi))
            std_distance = float(np.std(center_roi))

            # 信頼度計算（標準偏差が小さいほど信頼度高）
            confidence = max(0.0, 1.0 - (std_distance / 255.0))

            # 教室内ゾーン分類
            zone = self._classify_classroom_zone(median_distance)

            ctx.add_info("bbox", bbox)
            ctx.add_info("median_distance", median_distance)
            ctx.add_info("zone", zone)

            return {
                "distance": median_distance,
                "mean_distance": mean_distance,
                "distance_std": std_distance,
                "zone": zone,
                "confidence": confidence
            }

    def _classify_classroom_zone(self, distance: float) -> str:
        """距離に基づく教室ゾーン分類"""
        if distance < 0:
            return "unknown"
        elif distance < 85:  # 前列（深度値での近距離）
            return "front_row"
        elif distance < 140:  # 中列
            return "middle_row"
        elif distance < 200:  # 後列
            return "back_row"
        else:  # 教壇・最前方
            return "teacher_area"

    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def analyze_classroom_depth(self, image: np.ndarray) -> Dict[str, Any]:
        """
        教室全体の深度分析

        Args:
            image: 入力画像

        Returns:
            教室深度分析結果
        """
        with ErrorContext("教室深度分析", logger=logger) as ctx:
            # 基本深度推定
            depth_map = self.estimate_depth(image)

            # 教室エリア分析
            seating_analysis = self._analyze_seating_areas(depth_map)

            # 統計情報
            depth_stats = {
                "mean_depth": float(np.mean(depth_map)),
                "std_depth": float(np.std(depth_map)),
                "min_depth": float(np.min(depth_map)),
                "max_depth": float(np.max(depth_map))
            }

            ctx.add_info("depth_stats", depth_stats)

            return ResponseBuilder.success(
                data={
                    "depth_map": depth_map,
                    "seating_analysis": seating_analysis,
                    "depth_statistics": depth_stats,
                    "classroom_zones": self._get_zone_distribution(depth_map)
                },
                message="教室深度分析完了"
            )

    def _analyze_seating_areas(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """座席エリアごとの深度分析"""
        h, w = depth_map.shape

        # 教室を縦方向に分割（前列・中列・後列）
        front_area = depth_map[:h//3, :]
        middle_area = depth_map[h//3:2*h//3, :]
        back_area = depth_map[2*h//3:, :]

        return {
            "front_avg_depth": float(np.mean(front_area)),
            "middle_avg_depth": float(np.mean(middle_area)),
            "back_avg_depth": float(np.mean(back_area)),
            "depth_gradient": float(np.mean(back_area) - np.mean(front_area)),
            "area_ratios": {
                "front": front_area.size / depth_map.size,
                "middle": middle_area.size / depth_map.size,
                "back": back_area.size / depth_map.size
            }
        }

    def _get_zone_distribution(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """教室ゾーン別分布計算"""
        zones = {"front_row": 0, "middle_row": 0, "back_row": 0, "teacher_area": 0, "unknown": 0}

        flat_depth = depth_map.flatten()
        for depth_val in flat_depth:
            zone = self._classify_classroom_zone(float(depth_val))
            zones[zone] += 1

        total_pixels = depth_map.size
        zone_ratios = {k: v/total_pixels for k, v in zones.items()}

        return {
            "zone_counts": zones,
            "zone_ratios": zone_ratios
        }