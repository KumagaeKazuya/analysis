"""
メモリ効率処理モジュール
"""

import gc
import torch
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryEfficientProcessor:
    """メモリ効率を考慮した処理クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_memory_gb = config.get("max_memory_gb", 4.0)
        self.batch_size = config.get("batch_size", 32)
        self.streaming_output = config.get("streaming_output", True)

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（GB）"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception as e:
            self.logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def check_memory_threshold(self) -> bool:
        """メモリ使用量が閾値を超えているかチェック"""
        current_memory = self.get_memory_usage()
        return current_memory > self.max_memory_gb

    def force_memory_cleanup(self):
        """強制的なメモリクリーンアップ"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"メモリクリーンアップエラー: {e}")

        memory_after = self.get_memory_usage()
        self.logger.info(f"メモリクリーンアップ実行後: {memory_after:.2f}GB")