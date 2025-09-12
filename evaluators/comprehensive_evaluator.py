# 詳細評価システム

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from collections import defaultdict

class ComprehensiveEvaluator:
    """広角カメラ特有の課題を考慮した包括的評価システム"""

    def __init__(self, config):
        self.config = config
        self.metrics_history = []

    def evaluate_comprehensive(self, video_path, detection_results, video_name):
        """包括的な評価メトリクス計算"""

        metrics = {
            "basic_metrics": self._calculate_basic_metrics(detection_results),
            "spatial_metrics": self._calculate_spatial_metrics(detection_results),
            "temporal_metrics": self._calculate_temporal_metrics(detection_results),
            "quality_metrics": self._calculate_quality_metrics(detection_results),
            "wide_angle_metrics": self._calculate_wide_angle_metrics(detection_results),
        }

        # 統合スコア計算
        metrics["integrated_score"] = self._calculate_integrated_score(metrics)

        return metrics

    def _calculate_basic_metrics(self, detection_results):
        """基本的な検出メトリクス"""
        if "csv_path" not in detection_results:
            return {"error": "detection results not found"}

        df = pd.read_csv(detection_results["csv_path"])

        return {
            "total_detections": len(df),
            "unique_persons": df['person_id'].nunique(),
            "avg_confidence": df['conf'].mean(),
            "confidence_std": df['conf'].std(),
            "min_confidence": df['conf'].min(),
            "max_confidence": df['conf'].max(),
            "frames_with_detection": df['frame'].nunique(),
            "avg_detections_per_frame": len(df) / df['frame'].nunique() if df['frame'].nunique() > 0 else 0
        }

    def _calculate_spatial_metrics(self, detection_results):
        """空間的分布メトリクス（広角カメラ特有）"""
        df = pd.read_csv(detection_results["csv_path"])

        # バウンディングボックスサイズ分析
        df['width'] = df['x2'] - df['x1']
        df['height'] = df['y2'] - df['y1']
        df['area'] = df['width'] * df['height']
        df['aspect_ratio'] = df['width'] / df['height']

        # 画像領域分析（上部・中央・下部での性能差）
        frame_sample = cv2.imread(str(Path("outputs/frames").glob("*.jpg").__next__()))
        if frame_sample is not None:
            frame_height = frame_sample.shape[0]
            df['vertical_region'] = pd.cut(
                (df['y1'] + df['y2']) / 2,
                bins=[0, frame_height/3, 2*frame_height/3, frame_height],
                labels=['top', 'middle', 'bottom']
            )

        return {
            "avg_detection_size": df['area'].mean(),
            "size_variance": df['area'].std(),
            "size_distribution": df['area'].describe().to_dict(),
            "aspect_ratio_stats": df['aspect_ratio'].describe().to_dict(),
            "region_detection_counts": df['vertical_region'].value_counts().to_dict() if 'vertical_region' in df.columns else {},
            "edge_detection_ratio": self._calculate_edge_detection_ratio(df, frame_sample),
        }

    def _calculate_temporal_metrics(self, detection_results):
        """時系列的な追跡性能メトリクス"""
        df = pd.read_csv(detection_results["csv_path"])

        # ID継続性分析
        id_trajectories = df.groupby('person_id')['frame'].apply(list).to_dict()

        trajectory_lengths = [len(frames) for frames in id_trajectories.values()]

        # ID切り替え推定
        id_switches = self._estimate_id_switches(df)

        # 時間的一貫性
        temporal_consistency = self._calculate_temporal_consistency(df)

        return {
            "avg_trajectory_length": np.mean(trajectory_lengths),
            "max_trajectory_length": np.max(trajectory_lengths) if trajectory_lengths else 0,
            "min_trajectory_length": np.min(trajectory_lengths) if trajectory_lengths else 0,
            "trajectory_length_std": np.std(trajectory_lengths),
            "estimated_id_switches": id_switches,
            "temporal_consistency_score": temporal_consistency,
            "tracking_fragmentation": len(trajectory_lengths) / df['person_id'].nunique() if df['person_id'].nunique() > 0 else 0
        }

    def _calculate_quality_metrics(self, detection_results):
        """検出品質メトリクス"""
        df = pd.read_csv(detection_results["csv_path"])

        # 信頼度分布分析
        confidence_bins = pd.cut(df['conf'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
        confidence_distribution = confidence_bins.value_counts(normalize=True).to_dict()

        # 品質劣化指標
        quality_degradation = self._analyze_quality_degradation(df)

        return {
            "confidence_distribution": {str(k): v for k, v in confidence_distribution.items()},
            "high_confidence_ratio": len(df[df['conf'] > 0.7]) / len(df) if len(df) > 0 else 0,
            "low_confidence_ratio": len(df[df['conf'] < 0.3]) / len(df) if len(df) > 0 else 0,
            "quality_degradation_score": quality_degradation,
        }

    def _calculate_wide_angle_metrics(self, detection_results):
        """広角カメラ特有のメトリクス"""
        df = pd.read_csv(detection_results["csv_path"])

        # 歪み影響推定
        distortion_impact = self._estimate_distortion_impact(df)

        # エッジ領域での性能
        edge_performance = self._analyze_edge_performance(df)

        # スケール変動分析
        scale_variation = self._analyze_scale_variation(df)

        return {
            "distortion_impact_score": distortion_impact,
            "edge_performance_score": edge_performance,
            "scale_variation_score": scale_variation,
            "wide_angle_challenge_score": (distortion_impact + edge_performance + scale_variation) / 3
        }

    def _calculate_integrated_score(self, metrics):
        """統合評価スコア計算"""
        weights = {
            "detection_accuracy": 0.3,
            "tracking_stability": 0.25,
            "spatial_coverage": 0.2,
            "temporal_consistency": 0.15,
            "wide_angle_robustness": 0.1
        }

        # 各カテゴリのスコア正規化と重み付け合計
        scores = {}
        for category, weight in weights.items():
            scores[category] = self._normalize_category_score(metrics, category)

        integrated_score = sum(score * weights[cat] for cat, score in scores.items())

        return {
            "overall_score": integrated_score,
            "category_scores": scores,
            "weights": weights
        }

    def _estimate_id_switches(self, df):
        """ID切り替え数の推定"""
        # 簡易推定: 同じフレームで重複するバウンディングボックスを検出
        id_switches = 0

        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]

            # 重複するバウンディングボックスのペアを検出
            boxes = frame_data[['x1', 'y1', 'x2', 'y2']].values
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    iou = self._calculate_iou(boxes[i], boxes[j])
                    if iou > 0.5:  # 高いIoUは同一人物の可能性
                        id_switches += 1

        return id_switches

    def _calculate_iou(self, box1, box2):
        """IoU計算"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _calculate_temporal_consistency(self, df):
        """時間的一貫性スコア"""
        consistency_scores = []

        for person_id in df['person_id'].unique():
            person_data = df[df['person_id'] == person_id].sort_values('frame')

            if len(person_data) < 2:
                continue

            # 連続フレーム間のバウンディングボックス変動
            movements = []
            for i in range(1, len(person_data)):
                prev_center = ((person_data.iloc[i-1]['x1'] + person_data.iloc[i-1]['x2']) / 2,
                            (person_data.iloc[i-1]['y1'] + person_data.iloc[i-1]['y2']) / 2)
                curr_center = ((person_data.iloc[i]['x1'] + person_data.iloc[i]['x2']) / 2,
                            (person_data.iloc[i]['y1'] + person_data.iloc[i]['y2']) / 2)

                movement = np.sqrt((curr_center[0] - prev_center[0])**2 +
                                (curr_center[1] - prev_center[1])**2)
                movements.append(movement)

            if movements:
                consistency_scores.append(1.0 / (1.0 + np.std(movements)))

        return np.mean(consistency_scores) if consistency_scores else 0

    def _estimate_distortion_impact(self, df):
        """歪み影響の推定"""
        # エッジ領域での検出精度低下を歪み影響として推定
        if 'vertical_region' in df.columns:
            edge_detections = len(df[df['vertical_region'].isin(['top', 'bottom'])])
            center_detections = len(df[df['vertical_region'] == 'middle'])

            if center_detections > 0:
                return 1.0 - (edge_detections / (edge_detections + center_detections))

        return 0.5  # デフォルト値

    def _analyze_edge_performance(self, df):
        """エッジ領域での性能分析"""
        # 画像端での検出信頼度の分析
        frame_sample = cv2.imread(str(Path("outputs/frames").glob("*.jpg").__next__()))
        if frame_sample is None:
            return 0.5

        frame_width = frame_sample.shape[1]

        # 左右端での検出
        edge_threshold = frame_width * 0.1  # 端から10%の領域
        edge_detections = df[(df['x1'] < edge_threshold) | (df['x2'] > frame_width - edge_threshold)]
        center_detections = df[(df['x1'] >= edge_threshold) & (df['x2'] <= frame_width - edge_threshold)]

        if len(center_detections) > 0 and len(edge_detections) > 0:
            edge_confidence = edge_detections['conf'].mean()
            center_confidence = center_detections['conf'].mean()
            return edge_confidence / center_confidence

        return 0.5

    def _analyze_scale_variation(self, df):
        """スケール変動分析"""
        if len(df) == 0:
            return 0

        df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
        scale_variation = df['area'].std() / df['area'].mean() if df['area'].mean() > 0 else 0

        # 正規化 (低い変動が良いスコア)
        return 1.0 / (1.0 + scale_variation)

    def _normalize_category_score(self, metrics, category):
        """カテゴリスコアの正規化"""
        # カテゴリ別の正規化ロジック
        if category == "detection_accuracy":
            return metrics["basic_metrics"].get("avg_confidence", 0)
        elif category == "tracking_stability":
            return 1.0 - (metrics["temporal_metrics"].get("estimated_id_switches", 0) / 100)
        # 他のカテゴリも同様に実装

        return 0.5  # デフォルト値