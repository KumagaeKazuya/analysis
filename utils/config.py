# 設定管理

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """設定管理クラス"""

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if not os.path.exists(self.config_path):
            # デフォルト設定を作成
            return self._create_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を作成"""
        default_config = {
            "project_name": "yolo11_wide_angle_analysis",
            "version": "1.0.0",
            "paths": {
                "video_dir": "videos",
                "model_dir": "models",
                "output_dir": "outputs"
            },
            "models": {
                "detection": "yolo11n.pt",
                "pose": "yolo11n-pose.pt",
                "tracking_config": "bytetrack.yaml"
            },
            "processing": {
                "frame_sampling": {
                    "interval_sec": 2,
                    "max_frames": 1000
                },
                "detection": {
                    "confidence_threshold": 0.3,
                    "iou_threshold": 0.5,
                    "classes": [0]
                },
                "tracking": {
                    "track_thresh": 0.5,
                    "track_buffer": 30,
                    "match_thresh": 0.8
                }
            },
            "evaluation": {
                "metrics": {
                    "basic_metrics": True,
                    "spatial_metrics": True,
                    "temporal_metrics": True,
                    "quality_metrics": True,
                    "wide_angle_metrics": True
                },
                "thresholds": {
                    "high_confidence": 0.7,
                    "low_confidence": 0.3,
                    "iou_overlap": 0.5,
                    "edge_region_ratio": 0.1
                }
            }
        }

        # デフォルト設定ファイルを保存
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        return default_config

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法対応）"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, key: str, value: Any) -> None:
        """設定値を更新"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """設定をファイルに保存"""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    # 便利なプロパティ
    @property
    def video_dir(self) -> str:
        return self.get('paths.video_dir', 'videos')

    @property
    def model_dir(self) -> str:
        return self.get('paths.model_dir', 'models')

    @property
    def output_dir(self) -> str:
        return self.get('paths.output_dir', 'outputs')

    @property
    def detection_model(self) -> str:
        return os.path.join(self.model_dir, self.get('models.detection', 'yolo11n.pt'))

    @property
    def pose_model(self) -> str:
        return os.path.join(self.model_dir, self.get('models.pose', 'yolo11n-pose.pt'))

    def get_experiment_config(self, experiment_type: str) -> Dict[str, Any]:
        """実験固有の設定を取得"""
        experiments_config = {
            "calibration": {
                "type": "camera_calibration",
                "parameters": {
                    "enable_undistortion": True,
                    "calibration_file": "camera_params.json"
                }
            },
            "ensemble": {
                "type": "model_ensemble",
                "parameters": {
                    "models": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                    "voting_strategy": "confidence_weighted"
                }
            },
            "augmentation": {
                "type": "data_augmentation",
                "parameters": {
                    "enable_tta": True,
                    "tta_scales": [0.8, 1.0, 1.2],
                    "tta_flips": [False, True]
                }
            }
        }

        return experiments_config.get(experiment_type, {})