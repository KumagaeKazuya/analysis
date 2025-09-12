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
        """
        動画ごとの検出・追跡結果に対して、各種メトリクスを計算し統合スコアを返す
        """

        metrics = {
            "basic_metrics": self._calculate_basic_metrics(detection_results),      # 検出数や信頼度など基本指標
            "spatial_metrics": self._calculate_spatial_metrics(detection_results),  # バウンディングボックスの空間分布
            "temporal_metrics": self._calculate_temporal_metrics(detection_results),# ID追跡の時系列的安定性
            "quality_metrics": self._calculate_quality_metrics(detection_results),  # 信頼度分布や品質劣化
            "wide_angle_metrics": self._calculate_wide_angle_metrics(detection_results), # 広角カメラ特有の指標
        }

        # 各カテゴリのスコアを統合した総合評価値を計算
        metrics["integrated_score"] = self._calculate_integrated_score(metrics)

        return metrics

    def _calculate_basic_metrics(self, detection_results):
        """
        検出結果の基本的なメトリクス（検出数、信頼度、フレーム数など）を計算
        """
        if "csv_path" not in detection_results:
            return {"error": "detection results not found"}

        df = pd.read_csv(detection_results["csv_path"])

        return {
            "total_detections": len(df),  # 総検出数
            "unique_persons": df['person_id'].nunique(),  # 一意な人物ID数
            "avg_confidence": df['conf'].mean(),          # 平均信頼度
            "confidence_std": df['conf'].std(),           # 信頼度の標準偏差
            "min_confidence": df['conf'].min(),           # 最小信頼度
            "max_confidence": df['conf'].max(),           # 最大信頼度
            "frames_with_detection": df['frame'].nunique(),# 検出があったフレーム数
            "avg_detections_per_frame": len(df) / df['frame'].nunique() if df['frame'].nunique() > 0 else 0 # 1フレームあたり平均検出数
        }

    def _calculate_spatial_metrics(self, detection_results):
        """空間的分布メトリクス（広角カメラ特有）"""
        df = pd.read_csv(detection_results["csv_path"])

        # バウンディングボックスのサイズ・アスペクト比を計算
        df['width'] = df['x2'] - df['x1']
        df['height'] = df['y2'] - df['y1']
        df['area'] = df['width'] * df['height']
        df['aspect_ratio'] = df['width'] / df['height']

        # 画像の上下・中央領域ごとに検出数を集計
        frame_sample = cv2.imread(str(Path("outputs/frames").glob("*.jpg").__next__()))
        if frame_sample is not None:
            frame_height = frame_sample.shape[0]
            df['vertical_region'] = pd.cut(
                (df['y1'] + df['y2']) / 2,
                bins=[0, frame_height/3, 2*frame_height/3, frame_height],
                labels=['top', 'middle', 'bottom']
            )

        return {
            "avg_detection_size": df['area'].mean(),  # 平均検出サイズ
            "size_variance": df['area'].std(),        # 検出サイズのばらつき
            "size_distribution": df['area'].describe().to_dict(), # サイズ分布統計
            "aspect_ratio_stats": df['aspect_ratio'].describe().to_dict(), # アスペクト比統計
            "region_detection_counts": df['vertical_region'].value_counts().to_dict() if 'vertical_region' in df.columns else {}, # 領域ごとの検出数
            "edge_detection_ratio": self._calculate_edge_detection_ratio(df, frame_sample), # 画像端での検出割合
        }

    def _calculate_temporal_metrics(self, detection_results):
        """
        ID追跡の時系列的な安定性や一貫性を評価
        """
        df = pd.read_csv(detection_results["csv_path"])

        # 各人物IDごとのフレーム軌跡を抽出
        id_trajectories = df.groupby('person_id')['frame'].apply(list).to_dict()

        trajectory_lengths = [len(frames) for frames in id_trajectories.values()]

        # ID切り替え（スイッチ）数を推定
        id_switches = self._estimate_id_switches(df)

        # 時間的一貫性スコアを計算
        temporal_consistency = self._calculate_temporal_consistency(df)

        return {
            "avg_trajectory_length": np.mean(trajectory_lengths), # 平均軌跡長
            "max_trajectory_length": np.max(trajectory_lengths) if trajectory_lengths else 0,
            "min_trajectory_length": np.min(trajectory_lengths) if trajectory_lengths else 0,
            "trajectory_length_std": np.std(trajectory_lengths),
            "estimated_id_switches": id_switches,                 # ID切り替え推定値
            "temporal_consistency_score": temporal_consistency,   # 時系列一貫性スコア
            "tracking_fragmentation": len(trajectory_lengths) / df['person_id'].nunique() if df['person_id'].nunique() > 0 else 0 # 追跡の分断度
        }

    def _calculate_quality_metrics(self, detection_results):
        """
        検出信頼度の分布や品質劣化指標を分析
        """
        df = pd.read_csv(detection_results["csv_path"])

        # 信頼度を区分けして分布を集計
        confidence_bins = pd.cut(df['conf'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
        confidence_distribution = confidence_bins.value_counts(normalize=True).to_dict()

        # 品質劣化指標（例: 低信頼度の割合など）
        quality_degradation = self._analyze_quality_degradation(df)

        return {
            "confidence_distribution": {str(k): v for k, v in confidence_distribution.items()}, # 信頼度分布
            "high_confidence_ratio": len(df[df['conf'] > 0.7]) / len(df) if len(df) > 0 else 0, # 高信頼度割合
            "low_confidence_ratio": len(df[df['conf'] < 0.3]) / len(df) if len(df) > 0 else 0,  # 低信頼度割合
            "quality_degradation_score": quality_degradation,                                   # 品質劣化スコア
        }

    def _calculate_wide_angle_metrics(self, detection_results):
        """
        広角カメラ特有の歪みや端領域での性能低下などを評価
        """
        df = pd.read_csv(detection_results["csv_path"])

        # 歪み影響推定
        distortion_impact = self._estimate_distortion_impact(df)

        # 画像端での検出性能
        edge_performance = self._analyze_edge_performance(df)

        # 検出サイズのスケール変動
        scale_variation = self._analyze_scale_variation(df)

        return {
            "distortion_impact_score": distortion_impact, # 歪み影響スコア
            "edge_performance_score": edge_performance,   # 端領域性能スコア
            "scale_variation_score": scale_variation,     # スケール変動スコア
            "wide_angle_challenge_score": (distortion_impact + edge_performance + scale_variation) / 3 # 総合広角課題スコア
        }

    def _calculate_integrated_score(self, metrics):
        """
        各カテゴリのスコアを重み付けして統合評価値を算出
        """
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
            "overall_score": integrated_score, # 総合スコア
            "category_scores": scores,         # 各カテゴリスコア
            "weights": weights                 # 重み
        }

    def _estimate_id_switches(self, df):
        """
        ID切り替え数の推定（同一フレーム内で重複するバウンディングボックスの数をカウント）
        """
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
        """
        2つのバウンディングボックスのIoU（重なり度）を計算
        """
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
        """
        追跡対象ごとにフレーム間のバウンディングボックス移動量のばらつきを計算し、
        一貫性スコアとして返す（動きが安定しているほど高スコア）
        """
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
                consistency_scores.append(1.0 / (1.0 + np.std(movements))) # ばらつきが小さいほど高スコア

        return np.mean(consistency_scores) if consistency_scores else 0

    def _estimate_distortion_impact(self, df):
        """
        画像上下端領域での検出数から歪み影響を推定
        """
        if 'vertical_region' in df.columns:
            edge_detections = len(df[df['vertical_region'].isin(['top', 'bottom'])])
            center_detections = len(df[df['vertical_region'] == 'middle'])

            if center_detections > 0:
                return 1.0 - (edge_detections / (edge_detections + center_detections))

        return 0.5  # デフォルト値

    def _analyze_edge_performance(self, df):
        """
        画像左右端領域での検出信頼度を分析し、中心領域と比較したスコアを返す
        """
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
        """
        各カテゴリのスコアを0～1に正規化
        """
        if category == "detection_accuracy":
            return metrics["basic_metrics"].get("avg_confidence", 0) # 平均信頼度
        elif category == "tracking_stability":
            return 1.0 - (metrics["temporal_metrics"].get("estimated_id_switches", 0) / 100) # ID切り替えが少ないほど高スコア
        # 他のカテゴリも同様に実装

        return 0.5  # デフォルト値