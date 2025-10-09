"""
コマンドライン実行モジュール
"""

import argparse
import logging
from pathlib import Path

from .core import analyze_frames_with_tracking_memory_efficient
from .tile_inference import (
    analyze_frames_with_tile_inference,
    compare_tile_vs_normal_inference
)

logger = logging.getLogger(__name__)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='YOLO11 フレーム解析（タイル推論対応）')
    parser.add_argument('--frame-dir', required=True, help='フレームディレクトリ')
    parser.add_argument('--output-dir', required=True, help='出力ディレクトリ')
    parser.add_argument('--model', default='models/yolo11n-pose.pt', help='モデルパス')
    parser.add_argument('--tile', action='store_true', help='タイル推論を有効化')
    parser.add_argument('--adaptive', action='store_true', help='適応的タイル推論')
    parser.add_argument('--compare', action='store_true', help='比較実験実行')
    parser.add_argument('--tile-size', type=int, nargs=2, default=[640, 640])
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--confidence', type=float, default=0.3)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        if args.compare:
            print("🔬 タイル推論 vs 通常推論 比較実験")
            results = compare_tile_vs_normal_inference(
                args.frame_dir, args.output_dir, args.model, sample_frames=20
            )
            # [結果表示ロジック]

        elif args.tile:
            print("🔲 タイル推論実行")
            config = {
                "tile_inference": {
                    "enabled": True,
                    "tile_size": tuple(args.tile_size),
                    "overlap_ratio": args.overlap,
                    "use_adaptive": args.adaptive
                },
                "confidence_threshold": args.confidence
            }
            results = analyze_frames_with_tile_inference(
                args.frame_dir, args.output_dir, args.model, config
            )
            # [結果表示ロジック]

        else:
            print("📋 通常推論実行")
            config = {"confidence_threshold": args.confidence}
            results = analyze_frames_with_tracking_memory_efficient(
                args.frame_dir, args.output_dir, args.model, config
            )
            # [結果表示ロジック]

    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        logger.error(f"エラー詳細", exc_info=True)


if __name__ == "__main__":
    main()