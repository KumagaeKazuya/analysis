"""
詳細評価システム

主な拡張点:
1. 深度情報を考慮した評価メトリクス
2. 教室ゾーン別分析機能
3. 距離別検出精度評価
4. 深度推定品質評価
5. 統合スコアへの深度要素追加
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

# 統一エラーハンドラーからインポート
from utils.error_handler import (
    ValidationError,
    FileIOError,
    handle_errors,
    ErrorContext,
    ErrorCategory,
    ResponseBuilder
)

logger = logging.getLogger(__name__)

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ComprehensiveEvaluator:
    """広角カメラ特有の課題を考慮した包括的評価システム"""

    def __init__(self, config):
        self.config = config
        self.metrics_history = []
        self.depth_enabled = config.get('processing.depth_estimation.enabled', False)

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def evaluate_comprehensive(self, video_path, detection_results, video_name):
        """
        動画ごとの検出・追跡結果に対して、各種メトリクスを計算し統合スコアを返す
        深度推定が有効な場合は深度関連メトリクスも含む
        """
        with ErrorContext(f"包括評価: {video_name}", logger=logger) as ctx:
            try:
                metrics = {
                    "basic_metrics": self._calculate_basic_metrics(detection_results),
                    "spatial_metrics": self._calculate_spatial_metrics(detection_results),
                    "temporal_metrics": self._calculate_temporal_metrics(detection_results),
                    "quality_metrics": self._calculate_quality_metrics(detection_results),
                    "wide_angle_metrics": self._calculate_wide_angle_metrics(detection_results),
                }

                # 🔍 深度推定が有効な場合は深度メトリクスを追加
                if self.depth_enabled and detection_results.get("data", {}).get("depth_enabled", False):
                    metrics["depth_metrics"] = self._calculate_depth_metrics(detection_results)
                    ctx.add_info("depth_evaluation_enabled", True)
                else:
                    ctx.add_info("depth_evaluation_enabled", False)

                metrics["integrated_score"] = self._calculate_integrated_score(metrics)

                ctx.add_info("metrics_calculated", list(metrics.keys()))
                return {
                    "success": True,
                    "metrics": metrics
                }

            except Exception as e:
                logger.error(f"評価エラー: {e}", exc_info=True)
                import traceback
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_basic_metrics(self, detection_results):
        """基本的なメトリクス"""
        with ErrorContext("基本メトリクス計算", logger=logger) as ctx:
            # 深度統合CSVがある場合はそれを優先使用
            csv_path = None
            if detection_results.get("data", {}).get("enhanced_csv_path"):
                csv_path = detection_results["data"]["enhanced_csv_path"]
                ctx.add_info("using_enhanced_csv", True)
            elif detection_results.get("data", {}).get("csv_path"):
                csv_path = detection_results["data"]["csv_path"]
                ctx.add_info("using_basic_csv", True)
            elif "csv_path" in detection_results:
                csv_path = detection_results["csv_path"]
                ctx.add_info("using_legacy_csv", True)

            if not csv_path:
                return {"error": "detection results not found"}

        try:
            df = pd.read_csv(csv_path)
            ctx.add_info("csv_rows", len(df))

            if df.empty:
                    return {
                        "total_detections": 0,
                        "unique_persons": 0,
                        "avg_confidence": 0.0,
                        "confidence_std": 0.0,
                        "min_confidence": 0.0,
                        "max_confidence": 0.0,
                        "frames_with_detection": 0,
                        "avg_detections_per_frame": 0.0
                    }

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

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_spatial_metrics(self, detection_results):
        """空間的分布メトリクス"""
        with ErrorContext("空間メトリクス計算", logger=logger) as ctx:
            try:
                csv_path = self._get_csv_path(detection_results)
                df = pd.read_csv(csv_path)

                if df.empty:
                    return {
                        "avg_detection_size": 0.0,
                        "size_variance": 0.0,
                        "avg_aspect_ratio": 0.0,
                        "edge_detection_ratio": 0.0,
                        "spatial_distribution_score": 0.0
                    }

                # バウンディングボックス分析
                df['width'] = df['x2'] - df['x1']
                df['height'] = df['y2'] - df['y1']
                df['area'] = df['width'] * df['height']
                df['aspect_ratio'] = df['width'] / df['height']
                df['center_x'] = (df['x1'] + df['x2']) / 2
                df['center_y'] = (df['y1'] + df['y2']) / 2

                # 画像端での検出分析（概算値）
                edge_detection_ratio = self._estimate_edge_detection_ratio(df)

                # 空間分布の均等性スコア
                spatial_distribution_score = self._calculate_spatial_distribution_score(df)

                ctx.add_info("analyzed_detections", len(df))

                return {
                    "avg_detection_size": float(df['area'].mean()),
                    "size_variance": float(df['area'].std()),
                    "avg_aspect_ratio": float(df['aspect_ratio'].mean()),
                    "edge_detection_ratio": edge_detection_ratio,
                    "spatial_distribution_score": spatial_distribution_score,
                    "center_concentration": self._calculate_center_concentration(df)
                }
            except Exception as e:
                return {"error": f"spatial_metrics_failed: {e}"}

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_temporal_metrics(self, detection_results):
        """時系列メトリクス"""
        with ErrorContext("時系列メトリクス計算", logger=logger) as ctx:
            try:
                csv_path = self._get_csv_path(detection_results)
                df = pd.read_csv(csv_path)

                if df.empty:
                    return {
                        "avg_trajectory_length": 0.0,
                        "max_trajectory_length": 0,
                        "min_trajectory_length": 0,
                        "trajectory_length_std": 0.0,
                        "estimated_id_switches": 0,
                        "temporal_consistency_score": 0.0,
                        "tracking_fragmentation": 0.0
                    }

                id_trajectories = df.groupby('person_id')['frame'].apply(list).to_dict()
                trajectory_lengths = [len(frames) for frames in id_trajectories.values()]

                id_switches = self._estimate_id_switches(df)
                temporal_consistency = self._calculate_temporal_consistency(df)

                ctx.add_info("unique_trajectories", len(trajectory_lengths))
                ctx.add_info("estimated_id_switches", id_switches)

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

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_quality_metrics(self, detection_results):
        """品質メトリクス"""
        with ErrorContext("品質メトリクス計算", logger=logger) as ctx:
            try:
                csv_path = self._get_csv_path(detection_results)
                df = pd.read_csv(csv_path)

                if df.empty:
                    return {
                        "confidence_distribution": {},
                        "high_confidence_ratio": 0.0,
                        "low_confidence_ratio": 0.0,
                        "quality_degradation_score": 1.0
                    }

                confidence_bins = pd.cut(df['conf'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
                confidence_distribution = confidence_bins.value_counts(normalize=True).to_dict()

                quality_degradation = self._analyze_quality_degradation(df)

                ctx.add_info("confidence_mean", float(df['conf'].mean()))

                return {
                    "confidence_distribution": {str(k): float(v) for k, v in confidence_distribution.items()},
                    "high_confidence_ratio": float(len(df[df['conf'] > 0.7]) / len(df)) if len(df) > 0 else 0,
                    "medium_confidence_ratio": float(len(df[(df['conf'] >= 0.5) & (df['conf'] <= 0.7)]) / len(df)) if len(df) > 0 else 0,
                    "low_confidence_ratio": float(len(df[df['conf'] < 0.3]) / len(df)) if len(df) > 0 else 0,
                    "quality_degradation_score": quality_degradation
                }
            except Exception as e:
                return {"error": f"quality_metrics_failed: {e}"}

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_wide_angle_metrics(self, detection_results):
        """広角カメラ特有のメトリクス"""
        with ErrorContext("広角メトリクス計算", logger=logger) as ctx:
            try:
                csv_path = self._get_csv_path(detection_results)
                df = pd.read_csv(csv_path)

                if df.empty:
                    return {
                        "distortion_impact_score": 0.5,
                        "edge_performance_score": 0.5,
                        "scale_variation_score": 0.5,
                        "wide_angle_challenge_score": 0.5
                    }

                # 歪み影響スコア（中央部vs周辺部の信頼度比較）
                distortion_impact = self._analyze_distortion_impact(df)

                # 端部性能スコア
                edge_performance = self._analyze_edge_performance(df)

                # スケール変動スコア
                scale_variation = self._analyze_scale_variation(df)

                wide_angle_challenge_score = (distortion_impact + edge_performance + scale_variation) / 3

                ctx.add_info("distortion_impact", distortion_impact)
                ctx.add_info("edge_performance", edge_performance)

                return {
                    "distortion_impact_score": distortion_impact,
                    "edge_performance_score": edge_performance,
                    "scale_variation_score": scale_variation,
                    "wide_angle_challenge_score": wide_angle_challenge_score
                }
            except Exception as e:
                return {"error": f"wide_angle_metrics_failed: {e}"}

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_depth_metrics(self, detection_results):
        """
        🔍 深度推定関連メトリクス計算（新規追加）
        """
        with ErrorContext("深度メトリクス計算", logger=logger) as ctx:
            enhanced_csv_path = detection_results.get("data", {}).get("enhanced_csv_path")
            if not enhanced_csv_path or not Path(enhanced_csv_path).exists():
                return {"error": "enhanced CSV not found"}

            try:
                df = pd.read_csv(enhanced_csv_path)

                if df.empty:
                    return {"error": "empty enhanced CSV"}

                # 深度情報の存在確認
                required_columns = ['depth_distance', 'depth_zone', 'depth_confidence']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    return {"error": f"missing depth columns: {missing_columns}"}

                # 有効な深度データのフィルタリング
                valid_depth_df = df[df['depth_distance'] >= 0]
                valid_ratio = len(valid_depth_df) / len(df) if len(df) > 0 else 0

                ctx.add_info("total_detections", len(df))
                ctx.add_info("valid_depth_detections", len(valid_depth_df))
                ctx.add_info("depth_success_rate", valid_ratio)

                # 教室ゾーン別分析
                zone_analysis = self._analyze_classroom_zones(df)

                # 距離別検出品質分析
                distance_quality = self._analyze_distance_quality(valid_depth_df)

                # 深度推定信頼度分析
                depth_confidence_analysis = self._analyze_depth_confidence(df)

                # 距離-信頼度相関分析
                distance_confidence_correlation = self._analyze_distance_confidence_correlation(valid_depth_df)

                return {
                    "depth_success_rate": valid_ratio,
                    "total_detections": len(df),
                    "valid_depth_detections": len(valid_depth_df),
                    "zone_analysis": zone_analysis,
                    "distance_quality": distance_quality,
                    "depth_confidence_analysis": depth_confidence_analysis,
                    "distance_confidence_correlation": distance_confidence_correlation,
                    "overall_depth_quality_score": self._calculate_depth_quality_score(df, valid_depth_df)
                }

            except Exception as e:
                ctx.add_info("error_details", str(e))
                return {"error": f"depth_metrics_failed: {str(e)}"}

    def _analyze_classroom_zones(self, df):
        """教室ゾーン別分析"""
        try:
            if 'depth_zone' not in df.columns:
                return {"error": "depth_zone column not found"}

            zone_stats = df.groupby('depth_zone').agg({
                'conf': ['mean', 'std', 'count'],
                'person_id': 'nunique',
                'depth_confidence': 'mean' if 'depth_confidence' in df.columns else lambda x: 0
            }).round(3)

            zone_distribution = df['depth_zone'].value_counts().to_dict()

            # 各ゾーンの検出密度（相対的）
            total_detections = len(df)
            zone_density = {zone: count/total_detections for zone, count in zone_distribution.items()}

            return {
                "zone_distribution": zone_distribution,
                "zone_density": zone_density,
                "zone_confidence_stats": zone_stats.to_dict() if not zone_stats.empty else {},
                "best_performing_zone": max(zone_density.items(), key=lambda x: x[1])[0] if zone_density else "unknown",
                "zone_balance_score": self._calculate_zone_balance_score(zone_density)
            }
        except Exception as e:
            return {"error": f"zone_analysis_failed: {str(e)}"}

    def _analyze_distance_quality(self, valid_depth_df):
        """距離別検出品質分析"""
        try:
            if valid_depth_df.empty or 'depth_distance' not in valid_depth_df.columns:
                return {"error": "no valid depth data"}

            # 距離をビンに分割
            distance_bins = pd.qcut(
                valid_depth_df['depth_distance'],
                q=5,
                labels=['very_near', 'near', 'medium', 'far', 'very_far'],
                duplicates='drop'
            )

            distance_quality_stats = valid_depth_df.groupby(distance_bins).agg({
                'conf': ['mean', 'std', 'count'],
                'depth_confidence': 'mean' if 'depth_confidence' in valid_depth_df.columns else lambda x: 0
            }).round(3)

            # 距離別検出効率
            distance_efficiency = {}
            for distance_range in distance_bins.cat.categories:
                range_data = valid_depth_df[distance_bins == distance_range]
                if len(range_data) > 0:
                    efficiency = (range_data['conf'].mean() * range_data['depth_confidence'].mean() 
                                if 'depth_confidence' in range_data.columns else range_data['conf'].mean())
                    distance_efficiency[str(distance_range)] = float(efficiency)

            return {
                "distance_quality_stats": distance_quality_stats.to_dict() if not distance_quality_stats.empty else {},
                "distance_efficiency": distance_efficiency,
                "optimal_detection_range": max(distance_efficiency.items(), key=lambda x: x[1])[0] if distance_efficiency else "unknown"
            }
        except Exception as e:
            return {"error": f"distance_quality_failed: {str(e)}"}

    def _analyze_depth_confidence(self, df):
        """深度推定信頼度分析"""
        try:
            if 'depth_confidence' not in df.columns:
                return {"error": "depth_confidence column not found"}

            depth_conf_data = df['depth_confidence'].dropna()

            if depth_conf_data.empty:
                return {"error": "no depth confidence data"}

            return {
                "mean_depth_confidence": float(depth_conf_data.mean()),
                "std_depth_confidence": float(depth_conf_data.std()),
                "min_depth_confidence": float(depth_conf_data.min()),
                "max_depth_confidence": float(depth_conf_data.max()),
                "high_depth_confidence_ratio": float(len(depth_conf_data[depth_conf_data > 0.7]) / len(depth_conf_data)),
                "low_depth_confidence_ratio": float(len(depth_conf_data[depth_conf_data < 0.3]) / len(depth_conf_data))
            }
        except Exception as e:
            return {"error": f"depth_confidence_failed: {str(e)}"}

    def _analyze_distance_confidence_correlation(self, valid_depth_df):
        """距離-信頼度相関分析"""
        try:
            if valid_depth_df.empty or 'depth_distance' not in valid_depth_df.columns:
                return {"error": "no valid depth data for correlation"}

            # 検出信頼度との相関
            detection_conf_corr = valid_depth_df['depth_distance'].corr(valid_depth_df['conf'])

            # 深度信頼度との相関（存在する場合）
            depth_conf_corr = None
            if 'depth_confidence' in valid_depth_df.columns:
                depth_conf_corr = valid_depth_df['depth_distance'].corr(valid_depth_df['depth_confidence'])

            # 距離による検出性能劣化の分析
            performance_degradation = self._analyze_distance_performance_degradation(valid_depth_df)

            return {
                "distance_detection_confidence_correlation": float(detection_conf_corr) if not pd.isna(detection_conf_corr) else 0.0,
                "distance_depth_confidence_correlation": float(depth_conf_corr) if depth_conf_corr and not pd.isna(depth_conf_corr) else None,
                "performance_degradation_with_distance": performance_degradation
            }
        except Exception as e:
            return {"error": f"correlation_analysis_failed: {str(e)}"}

    def _calculate_depth_quality_score(self, df, valid_depth_df):
        """深度推定品質の総合スコア"""
        try:
            # 成功率スコア (0-1)
            success_rate = len(valid_depth_df) / len(df) if len(df) > 0 else 0

            # 深度信頼度スコア (0-1)
            depth_conf_score = 0.5  # デフォルト
            if 'depth_confidence' in df.columns:
                depth_conf_data = df['depth_confidence'].dropna()
                if not depth_conf_data.empty:
                    depth_conf_score = float(depth_conf_data.mean())

            # ゾーンバランススコア (0-1)
            zone_balance_score = 0.5  # デフォルト
            if 'depth_zone' in df.columns:
                zone_dist = df['depth_zone'].value_counts(normalize=True)
                # エントロピーベースのバランススコア
                if len(zone_dist) > 1:
                    entropy = -sum(p * np.log(p + 1e-10) for p in zone_dist.values)
                    max_entropy = np.log(len(zone_dist))
                    zone_balance_score = entropy / max_entropy if max_entropy > 0 else 0.5

            # 重み付き総合スコア
            weights = [0.4, 0.35, 0.25]  # [成功率, 深度信頼度, ゾーンバランス]
            scores = [success_rate, depth_conf_score, zone_balance_score]

            overall_score = sum(w * s for w, s in zip(weights, scores))

            return {
                "overall_score": float(overall_score),
                "component_scores": {
                    "success_rate": float(success_rate),
                    "depth_confidence": float(depth_conf_score),
                    "zone_balance": float(zone_balance_score)
                },
                "weights": {
                    "success_rate": weights[0],
                    "depth_confidence": weights[1],
                    "zone_balance": weights[2]
                }
            }
        except Exception as e:
            return {"error": f"depth_quality_score_failed: {str(e)}"}

    @handle_errors(logger=logger, error_category=ErrorCategory.EVALUATION)
    def _calculate_integrated_score(self, metrics):
        """統合スコア計算（深度推定対応版）"""
        with ErrorContext("統合スコア計算", logger=logger) as ctx:
            # 深度推定が有効な場合は重みを調整
            if self.depth_enabled and "depth_metrics" in metrics:
                weights = {
                    "detection_accuracy": 0.25,
                    "tracking_stability": 0.20,
                    "spatial_coverage": 0.15,
                    "temporal_consistency": 0.15,
                    "wide_angle_robustness": 0.10,
                    "depth_quality": 0.15  # 新規追加
                }
            else:
                # 従来の重み
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

            ctx.add_info("calculated_categories", list(scores.keys()))
            ctx.add_info("integrated_score", integrated_score)

            return {
                "overall_score": integrated_score,
                "category_scores": scores,
                "weights": weights,
                "depth_enhanced": self.depth_enabled and "depth_metrics" in metrics
            }

    def _normalize_category_score(self, metrics, category):
        """カテゴリスコア正規化（深度対応版）"""
        try:
            if category == "detection_accuracy":
                return metrics["basic_metrics"].get("avg_confidence", 0)
            elif category == "tracking_stability":
                switches = metrics["temporal_metrics"].get("estimated_id_switches", 0)
                return max(0, 1.0 - min(switches / 100, 1.0))
            elif category == "spatial_coverage":
                return metrics["spatial_metrics"].get("spatial_distribution_score", 0.5)
            elif category == "temporal_consistency":
                return metrics["temporal_metrics"].get("temporal_consistency_score", 0.5)
            elif category == "wide_angle_robustness":
                return metrics["wide_angle_metrics"].get("wide_angle_challenge_score", 0.5)
            elif category == "depth_quality":
                # 🔍 深度品質スコア（新規追加）
                if "depth_metrics" in metrics:
                    depth_quality = metrics["depth_metrics"].get("overall_depth_quality_score", {})
                    if isinstance(depth_quality, dict):
                        return depth_quality.get("overall_score", 0.5)
                    else:
                        return 0.5
                else:
                    return 0.5
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"カテゴリスコア正規化エラー ({category}): {e}")
            return 0.5

    # ========================================
    # 🔧 ヘルパーメソッド群
    # ========================================

    def _get_csv_path(self, detection_results):
        """CSVパスを取得（深度統合CSV優先）"""
        # 深度統合CSVがある場合はそれを優先
        if detection_results.get("data", {}).get("enhanced_csv_path"):
            return detection_results["data"]["enhanced_csv_path"]
        elif detection_results.get("data", {}).get("csv_path"):
            return detection_results["data"]["csv_path"]
        elif "csv_path" in detection_results:
            return detection_results["csv_path"]
        else:
            raise ValidationError("CSV path not found in detection results")

    def _estimate_edge_detection_ratio(self, df):
        """画像端での検出割合を推定"""
        try:
            if df.empty:
                return 0.0

            # 画像サイズの推定（概算）
            frame_width = max(df['x2'].max(), 1920)  # デフォルト値
            frame_height = max(df['y2'].max(), 1080)
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

    def _calculate_spatial_distribution_score(self, df):
        """空間分布の均等性スコア"""
        try:
            if df.empty or 'center_x' not in df.columns:
                return 0.5

            # 画像を4象限に分割して検出数の均等性を評価
            frame_width = max(df['center_x'].max(), 1920)
            frame_height = max(df['center_y'].max(), 1080)

            quadrants = {
                'top_left': len(df[(df['center_x'] < frame_width/2) & (df['center_y'] < frame_height/2)]),
                'top_right': len(df[(df['center_x'] >= frame_width/2) & (df['center_y'] < frame_height/2)]),
                'bottom_left': len(df[(df['center_x'] < frame_width/2) & (df['center_y'] >= frame_height/2)]),
                'bottom_right': len(df[(df['center_x'] >= frame_width/2) & (df['center_y'] >= frame_height/2)])
            }

            counts = list(quadrants.values())
            if sum(counts) == 0:
                return 0.5

            # 均等性の計算（標準偏差の逆数）
            mean_count = np.mean(counts)
            std_count = np.std(counts)

            if std_count == 0:
                return 1.0  # 完全に均等
            else:
                return min(1.0, mean_count / (std_count + 1))

        except:
            return 0.5

    def _calculate_center_concentration(self, df):
        """中央部集中度の計算"""
        try:
            if df.empty or 'center_x' not in df.columns:
                return 0.5

            frame_width = max(df['center_x'].max(), 1920)
            frame_height = max(df['center_y'].max(), 1080)

            # 中央25%領域の定義
            center_x_min, center_x_max = frame_width * 0.375, frame_width * 0.625
            center_y_min, center_y_max = frame_height * 0.375, frame_height * 0.625

            center_detections = len(df[
                (df['center_x'] >= center_x_min) & (df['center_x'] <= center_x_max) &
                (df['center_y'] >= center_y_min) & (df['center_y'] <= center_y_max)
            ])

            return center_detections / len(df) if len(df) > 0 else 0.0
        except:
            return 0.5

    def _analyze_distortion_impact(self, df):
        """歪み影響の分析"""
        try:
            if df.empty:
                return 0.5

            # 中央部と周辺部の信頼度比較
            frame_width = max(df['x2'].max(), 1920)
            frame_height = max(df['y2'].max(), 1080)

            df['center_x'] = (df['x1'] + df['x2']) / 2
            df['center_y'] = (df['y1'] + df['y2']) / 2

            # 中央部（50%領域）
            center_region = (
                (df['center_x'] > frame_width * 0.25) & (df['center_x'] < frame_width * 0.75) &
                (df['center_y'] > frame_height * 0.25) & (df['center_y'] < frame_height * 0.75)
            )

            center_conf = df[center_region]['conf'].mean() if center_region.any() else 0
            edge_conf = df[~center_region]['conf'].mean() if (~center_region).any() else 0

            # 歪み影響スコア（中央部信頼度 / 周辺部信頼度の比率）
            if edge_conf > 0:
                distortion_score = min(1.0, center_conf / edge_conf)
            else:
                distortion_score = 0.5

            return distortion_score
        except:
            return 0.5

    def _analyze_edge_performance(self, df):
        """端部性能の分析"""
        try:
            edge_ratio = self._estimate_edge_detection_ratio(df)
            # 端部検出が適度にある場合は良い性能とみなす
            return min(1.0, edge_ratio * 3)  # 0.33で満点
        except:
            return 0.5

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

    def _analyze_quality_degradation(self, df):
        """品質劣化分析"""
        try:
            low_conf_ratio = len(df[df['conf'] < 0.3]) / len(df) if len(df) > 0 else 0
            return 1.0 - low_conf_ratio  # 低信頼度が少ないほど高スコア
        except:
            return 0.5

    def _calculate_zone_balance_score(self, zone_density):
        """ゾーンバランススコアの計算"""
        try:
            if not zone_density:
                return 0.5

            densities = list(zone_density.values())
            if len(densities) <= 1:
                return 0.5

            # エントロピーベースのバランス評価
            entropy = -sum(p * np.log(p + 1e-10) for p in densities if p > 0)
            max_entropy = np.log(len(densities))

            return entropy / max_entropy if max_entropy > 0 else 0.5
        except:
            return 0.5

    def _analyze_distance_performance_degradation(self, valid_depth_df):
        """距離による性能劣化の分析"""
        try:
            if valid_depth_df.empty:
                return {"error": "no valid depth data"}

            # 距離でソートして性能劣化を分析
            sorted_df = valid_depth_df.sort_values('depth_distance')

            # 近距離と遠距離の信頼度比較
            n = len(sorted_df)
            near_third = sorted_df.iloc[:n//3]['conf'].mean()
            far_third = sorted_df.iloc[2*n//3:]['conf'].mean()

            degradation_ratio = (near_third - far_third) / near_third if near_third > 0 else 0

            return {
                "near_distance_confidence": float(near_third),
                "far_distance_confidence": float(far_third),
                "performance_degradation_ratio": float(degradation_ratio)
            }
        except Exception as e:
            return {"error": f"degradation_analysis_failed: {str(e)}"}


# ========================================
# 🔍 深度統合評価器クラス（継承版）
# ========================================

class DepthEnhancedEvaluator(ComprehensiveEvaluator):
    """深度情報統合評価システム（継承版）"""

    def __init__(self, config):
        super().__init__(config)
        self.depth_enabled = True  # 常に深度推定有効
        logger.info("🔍 深度統合評価器初期化完了")

    def evaluate_comprehensive(self, video_path, detection_results, video_name):
        """深度情報を必ず考慮した包括評価"""
        # 親クラスの評価を実行（深度メトリクス含む）
        metrics = super().evaluate_comprehensive(video_path, detection_results, video_name)

        # 深度評価が含まれていない場合の追加チェック
        if "depth_metrics" not in metrics and detection_results.get("data", {}).get("depth_enabled", False):
            metrics["depth_metrics"] = self._calculate_depth_metrics(detection_results)
            metrics["integrated_score"] = self._calculate_integrated_score(metrics)

        return metrics