# 設定管理

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic

T = TypeVar('T')

class ConfigValue(Generic[T]):
    """設定値のラッパークラス（型安全性向上）"""

    def __init__(self, value: T, key: str):
        self._value = value
        self._key = key

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"ConfigValue({self._key}={self._value})"

class Config:
    """設定管理クラス - 型安全版"""

    # デフォルト設定の定数化
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5
    DEFAULT_FRAME_INTERVAL = 2.0
    DEFAULT_DETECTION_CLASSES = [0]  # person

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        if not os.path.exists(self.config_path):
            print(f"設定ファイルが見つかりません: {self.config_path}")
            print("デフォルト設定を作成します...")
            return self._create_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if config is None:
                print("設定ファイルが空です。デフォルト設定を使用します。")
                return self._create_default_config()

            return config

        except yaml.YAMLError as e:
            print(f"YAML解析エラー: {e}")
            print("デフォルト設定を使用します。")
            return self._create_default_config()
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return self._create_default_config()

    def _validate_config(self):
        """設定値の型と妥当性をチェック"""
        validations = [
            ("processing.frame_sampling.interval_sec", (int, float), lambda x: x > 0),
            ("processing.detection.confidence_threshold", (int, float), lambda x: 0 <= x <= 1),
            ("processing.detection.iou_threshold", (int, float), lambda x: 0 <= x <= 1),
            ("processing.detection.classes", (list,), lambda x: all(isinstance(i, int) for i in x)),
        ]

        for key, expected_types, validator in validations:
            value = self.get(key)
            if value is not None:
                if not isinstance(value, expected_types):
                    print(f"警告: {key} の型が不正です。期待値: {expected_types}, 実際: {type(value)}")
                elif not validator(value):
                    print(f"警告: {key} の値が範囲外です: {value}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得（ドット記法対応 + 型安全）

        Args:
            key: 設定キー（例: "processing.detection.classes"）
            default: デフォルト値

        Returns:
            設定値（型変換済み）
        """
        keys = key.split('.')
        value = self._config

        # ネストした辞書を辿る
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        # 型変換とバリデーション
        return self._convert_and_validate(key, value, default)

    def _convert_and_validate(self, key: str, value: Any, default: Any) -> Any:
        """キー別の型変換とバリデーション"""
        # classes の特別処理（文字列→リスト変換）
        if key.endswith('.classes'):
            return self._convert_classes(value, default, key)  # keyを渡してログ改善

        # threshold の範囲チェック
        if 'threshold' in key.lower():
            return self._convert_threshold(value, default, key)

        # interval_sec の正数チェック
        if key.endswith('interval_sec'):
            return self._convert_interval(value, default, key)

        return value

    def _convert_classes(self, value: Any, default: Any, key: str = "") -> List[int]:
        """classes設定値の変換"""
        if isinstance(value, str):
            try:
                return [int(x.strip()) for x in value.split(',') if x.strip()]
            except ValueError:
                print(f"警告: {key} の文字列をリストに変換できません: {value}")
                return default if default is not None else self.DEFAULT_DETECTION_CLASSES
        elif isinstance(value, (int, float)):
            return [int(value)]
        elif isinstance(value, list):
            try:
                return [int(x) for x in value]
            except ValueError:
                print(f"警告: {key} のリスト要素を整数に変換できません: {value}")
                return default if default is not None else self.DEFAULT_DETECTION_CLASSES
        return default if default is not None else self.DEFAULT_DETECTION_CLASSES

    def _convert_threshold(self, value: Any, default: Any, key: str = "") -> float:
        """threshold設定値の変換"""
        if isinstance(value, (int, float)):
            value = float(value)
            if not 0 <= value <= 1:
                print(f"警告: {key} が範囲外です（0-1）: {value}")
                value = max(0, min(1, value))  # クランプ
            return value
        return default if default is not None else self.DEFAULT_CONFIDENCE_THRESHOLD

    def _convert_interval(self, value: Any, default: Any, key: str = "") -> float:
        """interval設定値の変換"""
        if isinstance(value, (int, float)):
            value = float(value)
            if value <= 0:
                print(f"警告: {key} は正数である必要があります: {value}")
                return default if default is not None else self.DEFAULT_FRAME_INTERVAL
            return value
        return default if default is not None else self.DEFAULT_FRAME_INTERVAL

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """リスト型の設定値を安全に取得"""
        value = self.get(key, default)
        if not isinstance(value, list):
            return default if default is not None else []
        return value

    def get_float(self, key: str, default: float = 0.0) -> float:
        """float型の設定値を安全に取得"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """int型の設定値を安全に取得"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """bool型の設定値を安全に取得"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def get_experiment_config(self, experiment_type: str) -> Dict[str, Any]:
        """
        実験タイプに応じた設定を返す
        improved_main.py で使用される
        """
        # 設定ファイルから実験設定を取得
        experiments_config = self.get('experiments', {})

        if experiment_type in experiments_config:
            return experiments_config[experiment_type]

        # デフォルト実験設定
        default_configs = {
            "calibration": {
                "type": "camera_calibration",
                "parameters": {
                    "enable_undistortion": True,
                    "calibration_file": "configs/camera_params.json"
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
            },

            "tile_comparison": {
                "type": "tile_inference_comparison",
                "parameters": {
                    "compare_with_baseline": True,
                    "sample_frames": 10,
                    "tile_sizes": [[640, 640], [800, 800]],
                    "overlap_ratios": [0.1, 0.2, 0.3]
                }
            }
        }

        return default_configs.get(experiment_type, {
            "type": experiment_type,
            "parameters": {}
        })

    def _create_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を作成（型を明示）"""
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
                "device": "auto",  # cpu/mps/cuda 自動選択
                "batch_size": 8,
                "max_memory_gb": 3.0,
                "streaming_output": True,
                "frame_sampling": {
                    "interval_sec": self.DEFAULT_FRAME_INTERVAL,
                    "max_frames": 1000
                },
                "detection": {
                    "confidence_threshold": self.DEFAULT_CONFIDENCE_THRESHOLD,
                    "iou_threshold": self.DEFAULT_IOU_THRESHOLD,
                    "classes": self.DEFAULT_DETECTION_CLASSES
                },
                "tracking": {
                    "track_thresh": 0.5,
                    "track_buffer": 30,
                    "match_thresh": 0.8
                },
                "tile_inference": {
                    "enabled": False,
                    "tile_size": [640, 640],
                    "overlap_ratio": 0.2,
                    "max_tiles_per_frame": 16,
                    "use_adaptive": False,
                    "nms_threshold": 0.5,
                    "memory_efficient_mode": True
                }
            },
            "evaluation": {
                "metrics": {
                    "basic_metrics": True,
                    "spatial_metrics": True,
                    "temporal_metrics": True,
                    "quality_metrics": True,
                    "wide_angle_metrics": True,
                    "tile_metrics": True  # 追加
                },
                "thresholds": {
                    "high_confidence": 0.7,
                    "low_confidence": 0.3,
                    "iou_overlap": 0.5,
                    "edge_region_ratio": 0.1
                }
            },
            "visualization": {
                "generate_plots": True,
                "save_annotated_frames": False,  # メモリ節約
                "create_summary_video": False,   # メモリ節約
                "plot_settings": {
                    "dpi": 300,
                    "figure_size": [12, 8],
                    "font_size": 12
                }
            },
            "experiments": {
                "calibration": {
                    "type": "camera_calibration",
                    "parameters": {
                        "enable_undistortion": True,
                        "calibration_file": "configs/camera_params.json"
                        }
                }
            }
        }

        # デフォルト設定ファイルを保存
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            print(f"デフォルト設定ファイルを作成しました: {self.config_path}")
        except Exception as e:
            print(f"設定ファイル作成エラー: {e}")

        return default_config

    # 便利なプロパティ（型安全版）
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
        model_file = self.get('models.detection', 'yolo11n.pt')
        return os.path.join(self.model_dir, model_file)

    @property
    def pose_model(self) -> str:
        model_file = self.get('models.pose', 'yolo11n-pose.pt')
        return os.path.join(self.model_dir, model_file)

    @property
    def detection_classes(self) -> List[int]:
        """検出対象クラスのリストを安全に取得"""
        return self.get_list('processing.detection.classes', [0])

    @property
    def confidence_threshold(self) -> float:
        """信頼度閾値をfloatで取得"""
        return self.get_float('processing.detection.confidence_threshold', 0.3)

    def print_config_summary(self):
        """設定内容の要約を表示（デバッグ用）"""
        print("=== 設定ファイル要約 ===")
        print(f"検出モデル: {self.detection_model}")
        print(f"ポーズモデル: {self.pose_model}")
        print(f"デバイス: {self.get('processing.device', 'cpu')}")
        print(f"信頼度閾値: {self.confidence_threshold}")
        print(f"検出クラス: {self.detection_classes}")
        print(f"フレーム間隔: {self.get_float('processing.frame_sampling.interval_sec')}秒")
        print("========================")
