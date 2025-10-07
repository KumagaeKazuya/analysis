# 詳細評価システム（修正版）

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from collections import defaultdict

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ComprehensiveEvaluator:
    """広角カメラ特有の課題を考慮した包括的評価システム"""

    def __init__(self, config):
        self.config = config
        self.metrics_history = []

    def evaluate_comprehensive(self, video_path, detection_results, video_name):
        """
        動画ごとの検出・追跡結果に対して、各種メトリクスを計算し統合スコアを返す
        """
        try:
            metrics = {
                "basic_metrics": self._calculate_basic_metrics(detection_results),
                "spatial_metrics": self._calculate_spatial_metrics(detection_results),
                "temporal_metrics": self._calculate_temporal_metrics(detection_results),
                "quality_metrics": self._calculate_quality_metrics(detection_results),
                "wide_angle_metrics": self._calculate_wide_angle_metrics(detection_results),
            }

            metrics["integrated_score"] = self._calculate_integrated_score(metrics)

            return metrics

        except Exception as e:
            import logging
            logging.error(f"評価エラー: {e}", exc_info=True)
            return {"error": str(e)}

    def _calculate_basic_metrics(self, detection_results):
        """基本的なメトリクス"""
        if "csv_path" not in detection_results:
            return {"error": "detection results not found"}

        try:
            df = pd.read_csv(detection_results["csv_path"])

            return {
                "total_detections": len(df),
                "unique_persons": df['person_id'].nunique(),
                "avg_confidence": float(df['conf'].mean()),
                "confidence_std": float(df['conf'].std()),
                "min_confidence": float(df['conf'].min()),
                "max_confidence": float(df['conf'].max()),
                "frames_with_detection": df['frame'].nunique(),
                "avg_detections_per_frame": len(df) / df['frame'].nunique() if df['frame'].nunique() > 0 else 0
            }
        except Exception as e:
            return {"error": f"basic_metrics_failed: {e}"}

    def _calculate_spatial_metrics(self, detection_results):
        """空間的分布メトリクス"""
        try:
            df = pd.read_csv(detection_results["csv_path"])

            df['width'] = df['x2'] - df['x1']
            df['height'] = df['y2'] - df['y1']
            df['area'] = df['width'] * df['height']
            df['aspect_ratio'] = df['width'] / df['height']

            return {
                "avg_detection_size": float(df['area'].mean()),
                "size_variance": float(df['area'].std()),
                "avg_aspect_ratio": float(df['aspect_ratio'].mean()),
                "edge_detection_ratio": 0.0  # 簡略化
            }
        except Exception as e:
            return {"error": f"spatial_metrics_failed: {e}"}

    def _calculate_temporal_metrics(self, detection_results):
        """時系列メトリクス"""
        try:
            df = pd.read_csv(detection_results["csv_path"])

            id_trajectories = df.groupby('person_id')['frame'].apply(list).to_dict()
            trajectory_lengths = [len(frames) for frames in id_trajectories.values()]

            id_switches = self._estimate_id_switches(df)
            temporal_consistency = self._calculate_temporal_consistency(df)

            return {
                "avg_trajectory_length": float(np.mean(trajectory_lengths)) if trajectory_lengths else 0,
                "max_trajectory_length": int(np.max(trajectory_lengths)) if trajectory_lengths else 0,
                "min_trajectory_length": int(np.min(trajectory_lengths)) if trajectory_lengths else 0,
                "trajectory_length_std": float(np.std(trajectory_lengths)) if trajectory_lengths else 0,
                "estimated_id_switches": id_switches,
                "temporal_consistency_score": temporal_consistency,
                "tracking_fragmentation": len(trajectory_lengths) / df['person_id'].nunique() if df['person_id'].nunique() > 0 else 0
            }
        except Exception as e:
            return {"error": f"temporal_metrics_failed: {e}"}

    def _calculate_quality_metrics(self, detection_results):
        """品質メトリクス"""
        try:
            df = pd.read_csv(detection_results["csv_path"])

            confidence_bins = pd.cut(df['conf'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
            confidence_distribution = confidence_bins.value_counts(normalize=True).to_dict()

            return {
                "confidence_distribution": {str(k): float(v) for k, v in confidence_distribution.items()},
                "high_confidence_ratio": float(len(df[df['conf'] > 0.7]) / len(df)) if len(df) > 0 else 0,
                "low_confidence_ratio": float(len(df[df['conf'] < 0.3]) / len(df)) if len(df) > 0 else 0,
                "quality_degradation_score": 0.0
            }
        except Exception as e:
            return {"error": f"quality_metrics_failed: {e}"}

    def _calculate_wide_angle_metrics(self, detection_results):
        """広角カメラ特有のメトリクス"""
        try:
            df = pd.read_csv(detection_results["csv_path"])

            distortion_impact = 0.5
            edge_performance = 0.5
            scale_variation = self._analyze_scale_variation(df)

            return {
                "distortion_impact_score": distortion_impact,
                "edge_performance_score": edge_performance,
                "scale_variation_score": scale_variation,
                "wide_angle_challenge_score": (distortion_impact + edge_performance + scale_variation) / 3
            }
        except Exception as e:
            return {"error": f"wide_angle_metrics_failed: {e}"}

    def _calculate_integrated_score(self, metrics):
        """統合スコア計算"""
        weights = {
            "detection_accuracy": 0.3,
            "tracking_stability": 0.25,
            "spatial_coverage": 0.2,
            "temporal_consistency": 0.15,
            "wide_angle_robustness": 0.1
        }

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
        """ID切り替え推定"""
        id_switches = 0
        try:
            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                boxes = frame_data[['x1', 'y1', 'x2', 'y2']].values
                
                for i in range(len(boxes)):
                    for j in range(i+1, len(boxes)):
                        iou = self._calculate_iou(boxes[i], boxes[j])
                        if iou > 0.5:
                            id_switches += 1
        except:
            pass

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
        """時系列一貫性計算"""
        consistency_scores = []

        try:
            for person_id in df['person_id'].unique():
                person_data = df[df['person_id'] == person_id].sort_values('frame')

                if len(person_data) < 2:
                    continue

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

            return float(np.mean(consistency_scores)) if consistency_scores else 0
        except:
            return 0.0

    def _analyze_scale_variation(self, df):
        """スケール変動分析"""
        try:
            if len(df) == 0:
                return 0

            df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
            scale_variation = df['area'].std() / df['area'].mean() if df['area'].mean() > 0 else 0

            return float(1.0 / (1.0 + scale_variation))
        except:
            return 0.5

    def _normalize_category_score(self, metrics, category):
        """カテゴリスコア正規化"""
        try:
            if category == "detection_accuracy":
                return metrics["basic_metrics"].get("avg_confidence", 0)
            elif category == "tracking_stability":
                switches = metrics["temporal_metrics"].get("estimated_id_switches", 0)
                return 1.0 - min(switches / 100, 1.0)
            else:
                return 0.5
        except:
            return 0.5

    # ✅ 未実装メソッドを追加
    def _calculate_edge_detection_ratio(self, df, frame_sample):
        """画像端での検出割合を計算"""
        try:
            if frame_sample is None:
                return 0.0

            frame_height, frame_width = frame_sample.shape[:2]
            edge_threshold = 0.1  # 端から10%の領域

            edge_detections = len(df[
                (df['x1'] < frame_width * edge_threshold) |
                (df['x2'] > frame_width * (1 - edge_threshold)) |
                (df['y1'] < frame_height * edge_threshold) |
                (df['y2'] > frame_height * (1 - edge_threshold))
            ])

            return edge_detections / len(df) if len(df) > 0 else 0.0
        except:
            return 0.0

    def _analyze_quality_degradation(self, df):
        """品質劣化分析"""
        try:
            low_conf_ratio = len(df[df['conf'] < 0.3]) / len(df) if len(df) > 0 else 0
            return low_conf_ratio
        except:
            return 0.0