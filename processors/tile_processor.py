import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
from ultralytics import YOLO
import time
import psutil
from dataclasses import dataclass

@dataclass
class TileConfig:
    """タイル分割設定"""
    tile_size: Tuple[int, int] = (640, 640)  # タイルサイズ (width, height)
    overlap_ratio: float = 0.2               # 重複率 (0.0-1.0)
    min_confidence: float = 0.3              # 最小信頼度
    nms_threshold: float = 0.5               # NMS閾値
    enable_dynamic_tiling: bool = True       # 動的タイリング有効化
    max_tiles_per_frame: int = 16            # 1フレームあたりの最大タイル数

@dataclass
class TileResult:
    """タイル推論結果"""
    tile_index: int
    original_boxes: np.ndarray      # 元画像座標でのボックス
    confidences: np.ndarray         # 信頼度
    tile_position: Tuple[int, int]  # タイル位置 (x, y)
    tile_size: Tuple[int, int]      # タイル実サイズ

class TileProcessor:
    """タイル推論を行うプロセッサ"""

    def __init__(self, model: YOLO, config: TileConfig = None):
        self.model = model
        self.config = config or TileConfig()
        self.logger = logging.getLogger(__name__)

        # 統計情報
        self.stats = {
            "total_tiles_processed": 0,
            "total_detections": 0,
            "average_tile_time": 0,
            "memory_usage_mb": 0
        }

    def process_frame_with_tiles(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        フレームをタイル分割して推論実行

        Args:
            frame: 入力フレーム (numpy array)

        Returns:
            統合された検出結果
        """
        start_time = time.time()

        # フレーム情報
        frame_height, frame_width = frame.shape[:2]
        self.logger.debug(f"処理フレームサイズ: {frame_width}x{frame_height}")

        # タイル分割
        tiles_info = self._generate_tiles(frame_width, frame_height)
        self.logger.info(f"タイル分割: {len(tiles_info)}個のタイル生成")

        # 各タイルで推論実行
        tile_results = []
        for i, tile_info in enumerate(tiles_info):
            tile_result = self._process_single_tile(frame, tile_info, i)
            if tile_result is not None:
                tile_results.append(tile_result)

        # 結果統合
        integrated_result = self._integrate_tile_results(
            tile_results, (frame_width, frame_height)
        )

        # 統計更新
        processing_time = time.time() - start_time
        self.stats["total_tiles_processed"] += len(tiles_info)
        self.stats["average_tile_time"] = processing_time / len(tiles_info) if tiles_info else 0

        integrated_result["processing_stats"] = {
            "num_tiles": len(tiles_info),
            "processing_time": processing_time,
            "detections_per_tile": len(tile_results) / len(tiles_info) if tiles_info else 0
        }

        return integrated_result

    def _generate_tiles(self, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """タイル情報を生成"""
        tile_width, tile_height = self.config.tile_size
        overlap_w = int(tile_width * self.config.overlap_ratio)
        overlap_h = int(tile_height * self.config.overlap_ratio)

        stride_w = tile_width - overlap_w
        stride_h = tile_height - overlap_h

        tiles_info = []

        y = 0
        while y < frame_height:
            x = 0
            while x < frame_width:
                # タイルの実サイズを計算（画像端で調整）
                actual_tile_w = min(tile_width, frame_width - x)
                actual_tile_h = min(tile_height, frame_height - y)

                # 小さすぎるタイルはスキップ
                if actual_tile_w < tile_width * 0.5 or actual_tile_h < tile_height * 0.5:
                    x += stride_w
                    continue

                tiles_info.append({
                    "x": x,
                    "y": y,
                    "width": actual_tile_w,
                    "height": actual_tile_h,
                    "tile_index": len(tiles_info)
                })

                x += stride_w
                if x >= frame_width:
                    break

            y += stride_h
            if y >= frame_height:
                break

        # 最大タイル数制限
        if len(tiles_info) > self.config.max_tiles_per_frame:
            self.logger.warning(f"タイル数が制限を超過: {len(tiles_info)} > {self.config.max_tiles_per_frame}")
            # 中心部分を優先してタイルを選択
            tiles_info = self._select_priority_tiles(tiles_info, frame_width, frame_height)

        return tiles_info

    def _select_priority_tiles(self, tiles_info: List[Dict], frame_width: int, frame_height: int) -> List[Dict]:
        """優先度に基づいてタイルを選択"""
        center_x, center_y = frame_width // 2, frame_height // 2

        # 中心からの距離でソート
        def distance_from_center(tile):
            tile_center_x = tile["x"] + tile["width"] // 2
            tile_center_y = tile["y"] + tile["height"] // 2
            return ((tile_center_x - center_x) ** 2 + (tile_center_y - center_y) ** 2) ** 0.5

        tiles_info.sort(key=distance_from_center)
        selected_tiles = tiles_info[:self.config.max_tiles_per_frame]

        self.logger.info(f"優先タイル選択: {len(selected_tiles)}個を選択")
        return selected_tiles

    def _process_single_tile(self, frame: np.ndarray, tile_info: Dict, tile_index: int) -> Optional[TileResult]:
        """単一タイルの推論処理"""
        try:
            # タイル切り出し
            x, y = tile_info["x"], tile_info["y"]
            w, h = tile_info["width"], tile_info["height"]
            tile_frame = frame[y:y+h, x:x+w]

            # 推論実行
            results = self.model(
                tile_frame,
                conf=self.config.min_confidence,
                verbose=False
            )

            # 結果が空の場合
            if not results or results[0].boxes is None:
                return None

            boxes = results[0].boxes
            if len(boxes) == 0:
                return None

            # 座標を元画像座標に変換
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            # タイル座標から元画像座標への変換
            original_boxes = xyxy.copy()
            original_boxes[:, [0, 2]] += x  # x座標
            original_boxes[:, [1, 3]] += y  # y座標

            return TileResult(
                tile_index=tile_index,
                original_boxes=original_boxes,
                confidences=confidences,
                tile_position=(x, y),
                tile_size=(w, h)
            )

        except Exception as e:
            self.logger.error(f"タイル{tile_index}処理エラー: {e}")
            return None

    def _integrate_tile_results(self, tile_results: List[TileResult], frame_size: Tuple[int, int]) -> Dict[str, Any]:
        """タイル結果の統合とNMS適用"""
        if not tile_results:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "confidences": np.array([]),
                "detection_count": 0,
                "tile_sources": []
            }

        # 全てのボックスと信頼度を統合
        all_boxes = []
        all_confidences = []
        tile_sources = []

        for tile_result in tile_results:
            all_boxes.append(tile_result.original_boxes)
            all_confidences.append(tile_result.confidences)
            tile_sources.extend([tile_result.tile_index] * len(tile_result.original_boxes))

        if not all_boxes:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "confidences": np.array([]),
                "detection_count": 0,
                "tile_sources": []
            }

        integrated_boxes = np.vstack(all_boxes)
        integrated_confidences = np.hstack(all_confidences)

        # カスタムNMS適用
        keep_indices = self._apply_custom_nms(
            integrated_boxes,
            integrated_confidences,
            self.config.nms_threshold
        )

        final_boxes = integrated_boxes[keep_indices]
        final_confidences = integrated_confidences[keep_indices]
        final_tile_sources = [tile_sources[i] for i in keep_indices]

        self.logger.info(f"NMS前: {len(integrated_boxes)}個 → NMS後: {len(final_boxes)}個")

        return {
            "boxes": final_boxes,
            "confidences": final_confidences,
            "detection_count": len(final_boxes),
            "tile_sources": final_tile_sources,
            "nms_reduction_rate": 1 - (len(final_boxes) / len(integrated_boxes)) if integrated_boxes.size > 0 else 0
        }

    def _apply_custom_nms(self, boxes: np.ndarray, confidences: np.ndarray, threshold: float) -> List[int]:
        """カスタムNMS実装"""
        if len(boxes) == 0:
            return []

        # 信頼度でソート（降順）
        order = confidences.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # 残りのボックスとのIoU計算
            ious = self._calculate_iou_vectorized(boxes[i], boxes[order[1:]])

            # 閾値以下のもののみ残す
            order = order[1:][ious <= threshold]

        return keep

    def _calculate_iou_vectorized(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """ベクトル化されたIoU計算"""
        if len(boxes) == 0:
            return np.array([])

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-8)

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        return {
            **self.stats,
            "current_memory_mb": memory_mb,
            "tiles_per_second": 1 / self.stats["average_tile_time"] if self.stats["average_tile_time"] > 0 else 0
        }

class AdaptiveTileProcessor(TileProcessor):
    """適応的タイル推論プロセッサ（より高度な機能）"""

    def __init__(self, model: YOLO, config: TileConfig = None):
        super().__init__(model, config)
        self.frame_history = []  # フレーム履歴
        self.detection_density_map = None  # 検出密度マップ

    def process_frame_with_adaptive_tiles(self, frame: np.ndarray, frame_index: int = 0) -> Dict[str, Any]:
        """
        適応的タイル分割による推論
        検出密度に基づいてタイルサイズを動的調整
        """
        # 検出密度マップを更新
        self._update_detection_density_map(frame.shape[:2])

        # 密度に基づくタイル分割
        tiles_info = self._generate_adaptive_tiles(frame.shape[1], frame.shape[0])

        # 通常のタイル推論を実行
        result = self.process_frame_with_tiles(frame)
        result["adaptive_tiles_count"] = len(tiles_info)

        # フレーム履歴を更新
        self._update_frame_history(result, frame_index)

        return result

    def _generate_adaptive_tiles(self, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        """検出密度に基づく適応的タイル生成"""
        if self.detection_density_map is None:
            # 初回は通常のタイル分割
            return self._generate_tiles(frame_width, frame_height)

        # 高密度領域をより細かく分割
        adaptive_tiles = []

        # 簡単な実装例：密度マップに基づく分割
        base_tiles = self._generate_tiles(frame_width, frame_height)

        for tile_info in base_tiles:
            # この領域の検出密度を確認
            density = self._get_region_density(tile_info)

            if density > 0.5:  # 高密度の場合、さらに分割
                sub_tiles = self._subdivide_tile(tile_info, 2)
                adaptive_tiles.extend(sub_tiles)
            else:
                adaptive_tiles.append(tile_info)

        return adaptive_tiles[:self.config.max_tiles_per_frame]

    def _subdivide_tile(self, tile_info: Dict, subdivision: int) -> List[Dict]:
        """タイルをさらに細かく分割"""
        sub_tiles = []
        sub_width = tile_info["width"] // subdivision
        sub_height = tile_info["height"] // subdivision

        for i in range(subdivision):
            for j in range(subdivision):
                sub_tiles.append({
                    "x": tile_info["x"] + j * sub_width,
                    "y": tile_info["y"] + i * sub_height,
                    "width": sub_width,
                    "height": sub_height,
                    "tile_index": len(sub_tiles)
                })

        return sub_tiles

    def _update_detection_density_map(self, frame_shape: Tuple[int, int]):
        """検出密度マップの更新（簡略実装）"""
        if self.detection_density_map is None:
            self.detection_density_map = np.zeros((frame_shape[0] // 64, frame_shape[1] // 64))

        # 実際の実装では過去の検出結果に基づいて更新
        # ここでは簡略化

    def _get_region_density(self, tile_info: Dict) -> float:
        """指定領域の検出密度を取得（簡略実装）"""
        return 0.3  # 実際の実装では密度マップから計算

    def _update_frame_history(self, result: Dict, frame_index: int):
        """フレーム履歴の更新"""
        self.frame_history.append({
            "frame_index": frame_index,
            "detection_count": result["detection_count"],
            "processing_time": result.get("processing_stats", {}).get("processing_time", 0)
        })

        # 履歴サイズ制限
        if len(self.frame_history) > 100:
            self.frame_history.pop(0)