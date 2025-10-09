"""
YOLO11 フレーム解析モジュール

このモジュールは元の yolopose_analyzer.py をモジュール化したものです。
後方互換性を保ちながら、保守性と拡張性を向上させています。

使用例:
    # 通常推論
    from yolopose_analyzer import analyze_frames_with_tracking_memory_efficient
    results = analyze_frames_with_tracking_memory_efficient(
        frame_dir="outputs/frames/test",
        result_dir="outputs/results/test"
    )
    
    # タイル推論
    from yolopose_analyzer import analyze_frames_with_tile_inference
    config = {"tile_inference": {"enabled": True}}
    results = analyze_frames_with_tile_inference(
        frame_dir="outputs/frames/test",
        result_dir="outputs/results/test",
        config=config
    )
"""

import logging

# バージョン情報
__version__ = '1.0.0'
__author__ = 'YOLO11 Project'

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ========================================
# 公開API: 主要な分析関数
# ========================================

from .core import (
    analyze_frames_with_tracking_memory_efficient,
    analyze_frames_with_tracking_enhanced
)

from .tile_inference import (
    analyze_frames_with_tile_inference,
    compare_tile_vs_normal_inference
)

# ========================================
# 公開API: システム関連
# ========================================

from .system import (
    check_system_resources,
    safe_model_initialization,
    ModelInitializationError,
    ResourceExhaustionError
)

# ========================================
# 公開API: 検証関連
# ========================================

from .validation import (
    validate_frame_directory,
    validate_model_file
)

# ========================================
# 公開API: メモリ管理
# ========================================

from .memory import MemoryEfficientProcessor

# ========================================
# 公開API: 可視化
# ========================================

from .visualization import (
    draw_detections,
    draw_detections_ultralytics
)

# draw_detections_enhanced はオプション（タイル推論利用時）
try:
    from .visualization import draw_detections_enhanced
except ImportError:
    draw_detections_enhanced = None

# ========================================
# タイル推論の利用可能性フラグ
# ========================================

try:
    from processors.tile_processor import TileProcessor, TileConfig, AdaptiveTileProcessor
    TILE_INFERENCE_AVAILABLE = True
except ImportError:
    TILE_INFERENCE_AVAILABLE = False
    
    # ダミークラス（後方互換性のため）
    class TileProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("TileProcessor が利用できません")
    
    class TileConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TileConfig が利用できません")
    
    class AdaptiveTileProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("AdaptiveTileProcessor が利用できません")

# ========================================
# エラークラス（後方互換性）
# ========================================

class VideoProcessingError(Exception):
    """動画処理エラー"""
    pass

# ========================================
# 公開API一覧
# ========================================

__all__ = [
    # バージョン情報
    '__version__',
    '__author__',
    
    # 分析関数
    'analyze_frames_with_tracking_memory_efficient',
    'analyze_frames_with_tracking_enhanced',
    'analyze_frames_with_tile_inference',
    'compare_tile_vs_normal_inference',
    
    # システム関連
    'check_system_resources',
    'safe_model_initialization',
    
    # 検証関連
    'validate_frame_directory',
    'validate_model_file',
    
    # メモリ管理
    'MemoryEfficientProcessor',
    
    # 可視化
    'draw_detections',
    'draw_detections_ultralytics',
    'draw_detections_enhanced',
    
    # エラークラス
    'ModelInitializationError',
    'ResourceExhaustionError',
    'VideoProcessingError',
    
    # タイル推論
    'TileProcessor',
    'TileConfig',
    'AdaptiveTileProcessor',
    'TILE_INFERENCE_AVAILABLE',
]

# ========================================
# モジュール読み込み時の情報表示
# ========================================

logger = logging.getLogger(__name__)

if TILE_INFERENCE_AVAILABLE:
    logger.info("✅ タイル推論モジュール利用可能")
else:
    logger.info("📋 通常推論のみ利用可能（タイル推論モジュールなし）")