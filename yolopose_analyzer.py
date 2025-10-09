"""
YOLO11 フレーム解析 - 後方互換性ラッパー
新しいモジュール構造への移行をサポート

使用方法:
    # 従来通りの使用方法（後方互換）
    from yolopose_analyzer import analyze_frames_with_tracking_memory_efficient
    
    # または新しいモジュール構造（推奨）
    from yolopose_analyzer.core import analyze_frames_with_tracking_memory_efficient
    
    # CLI実行も従来通り
    python yolopose_analyzer.py --frame-dir outputs/frames --output-dir outputs/results
"""

import logging
import warnings

# ロギング設定（既存コードとの互換性）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# 新しいモジュール構造からすべてを再エクスポート
# ========================================

try:
    # 主要な分析関数
    from yolopose_analyzer.core import (
        analyze_frames_with_tracking_memory_efficient,
        analyze_frames_with_tracking_enhanced
    )
    
    # タイル推論関連
    from yolopose_analyzer.tile_inference import (
        analyze_frames_with_tile_inference,
        compare_tile_vs_normal_inference
    )
    
    # システム関連
    from yolopose_analyzer.system import (
        check_system_resources,
        safe_model_initialization,
        ModelInitializationError,
        ResourceExhaustionError
    )
    
    # 検証関連
    from yolopose_analyzer.validation import (
        validate_frame_directory,
        validate_model_file
    )
    
    # メモリ管理
    from yolopose_analyzer.memory import MemoryEfficientProcessor
    
    # 可視化
    from yolopose_analyzer.visualization import (
        draw_detections,
        draw_detections_ultralytics,
        draw_detections_enhanced
    )
    
    MODULAR_IMPORT_SUCCESS = True
    
except ImportError as e:
    # モジュール化されたコードが利用できない場合の警告
    warnings.warn(
        f"モジュール構造からのインポートに失敗しました: {e}\n"
        "yolopose_analyzer/ ディレクトリが正しく配置されているか確認してください。",
        ImportWarning
    )
    MODULAR_IMPORT_SUCCESS = False
    
    # フォールバック: エラーを発生させる
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "yolopose_analyzerモジュールが正しくインストールされていません。\n"
            "yolopose_analyzer/ ディレクトリの構造を確認してください。"
        )
    
    # すべての関数をダミー化
    analyze_frames_with_tracking_memory_efficient = _raise_import_error
    analyze_frames_with_tracking_enhanced = _raise_import_error
    analyze_frames_with_tile_inference = _raise_import_error
    compare_tile_vs_normal_inference = _raise_import_error
    check_system_resources = _raise_import_error
    safe_model_initialization = _raise_import_error
    validate_frame_directory = _raise_import_error
    validate_model_file = _raise_import_error
    
    class MemoryEfficientProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("MemoryEfficientProcessor が利用できません")
    
    draw_detections = _raise_import_error
    draw_detections_ultralytics = _raise_import_error
    
    class ModelInitializationError(Exception):
        pass
    
    class ResourceExhaustionError(Exception):
        pass

# ========================================
# 追加のエラークラス（元のコードとの互換性）
# ========================================

class VideoProcessingError(Exception):
    """動画処理エラー"""
    pass


# ========================================
# タイル推論の利用可能性フラグ
# ========================================

try:
    from processors.tile_processor import TileProcessor, TileConfig, AdaptiveTileProcessor
    TILE_INFERENCE_AVAILABLE = True
    logger.info("✅ タイル推論モジュール利用可能")
except ImportError:
    TILE_INFERENCE_AVAILABLE = False
    logger.warning("⚠️ タイル推論モジュールが見つかりません（通常推論のみ利用可能）")
    
    # ダミークラス
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
# __all__ で公開APIを明示
# ========================================

__all__ = [
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
# バージョン情報
# ========================================

__version__ = '1.0.0'
__author__ = 'YOLO11 Project'


# ========================================
# メイン実行部（CLI）
# ========================================

if __name__ == "__main__":
    if not MODULAR_IMPORT_SUCCESS:
        print("❌ モジュール構造のインポートに失敗しました")
        print("yolopose_analyzer/ ディレクトリが存在し、正しく構成されているか確認してください")
        import sys
        sys.exit(1)
    
    # CLI実行を新しいモジュールに委譲
    try:
        from yolopose_analyzer.cli import main
        main()
    except ImportError:
        # フォールバック: 最小限のCLI実装
        print("⚠️ CLI モジュールが見つかりません。基本機能のみ実行します。")
        
        import argparse
        import os
        
        parser = argparse.ArgumentParser(description='YOLO11 フレーム解析')
        parser.add_argument('--frame-dir', required=True, help='フレームディレクトリ')
        parser.add_argument('--output-dir', required=True, help='出力ディレクトリ')
        parser.add_argument('--model', default='models/yolo11n-pose.pt', help='モデルパス')
        parser.add_argument('--tile', action='store_true', help='タイル推論を有効化')
        parser.add_argument('--confidence', type=float, default=0.3, help='信頼度閾値')
        
        args = parser.parse_args()
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            config = {
                "confidence_threshold": args.confidence,
                "save_visualizations": True
            }
            
            if args.tile and TILE_INFERENCE_AVAILABLE:
                print("🔲 タイル推論実行")
                config["tile_inference"] = {"enabled": True}
                results = analyze_frames_with_tile_inference(
                    args.frame_dir, args.output_dir, args.model, config
                )
            else:
                print("📋 通常推論実行")
                results = analyze_frames_with_tracking_memory_efficient(
                    args.frame_dir, args.output_dir, args.model, config
                )
            
            if results.get("success", False):
                print("✅ 処理完了")
                stats = results.get("processing_stats", {})
                print(f"📊 総検出数: {stats.get('total_detections', 0)}")
                print(f"👥 ユニークID: {stats.get('unique_ids', 0)}")
            else:
                print(f"❌ エラー: {results.get('error', 'unknown_error')}")
        
        except Exception as e:
            print(f"❌ 実行エラー: {e}")
            logger.error("エラー詳細", exc_info=True)