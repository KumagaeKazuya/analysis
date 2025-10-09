"""
YOLO11 フレーム解析モジュール
"""

from .core import (
    analyze_frames_with_tracking_memory_efficient,
    analyze_frames_with_tracking_enhanced
)
from .tile_inference import (
    analyze_frames_with_tile_inference,
    compare_tile_vs_normal_inference
)
from .system import (
    check_system_resources,
    safe_model_initialization
)
from .validation import (
    validate_frame_directory,
    validate_model_file
)
from .memory import MemoryEfficientProcessor
from .visualization import (
    draw_detections,
    draw_detections_ultralytics
)

# 後方互換性のため、全ての主要関数をエクスポート
__all__ = [
    'analyze_frames_with_tracking_memory_efficient',
    'analyze_frames_with_tracking_enhanced',
    'analyze_frames_with_tile_inference',
    'compare_tile_vs_normal_inference',
    'check_system_resources',
    'safe_model_initialization',
    'validate_frame_directory',
    'validate_model_file',
    'MemoryEfficientProcessor',
    'draw_detections',
    'draw_detections_ultralytics',
]

__version__ = '1.0.0'