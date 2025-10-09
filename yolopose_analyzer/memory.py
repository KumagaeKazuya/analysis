"""
メモリ効率処理モジュール
元の yolopose_analyzer.py から抽出
"""

import gc
import torch
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MemoryEfficientProcessor:
    """
    メモリ効率を考慮した処理クラス
    
    バッチ処理とメモリモニタリングを提供します。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書（max_memory_gb, batch_size, streaming_output等）
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # メモリ使用量制限
        self.max_memory_gb = config.get("max_memory_gb", 4.0)
        self.batch_size = config.get("batch_size", 32)
        self.streaming_output = config.get("streaming_output", True)

    def get_memory_usage(self) -> float:
        """
        現在のメモリ使用量を取得（GB単位）
        
        Returns:
            メモリ使用量（GB）
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception as e:
            self.logger.warning(f"メモリ使用量取得エラー: {e}")
            return 0.0

    def check_memory_threshold(self) -> bool:
        """
        メモリ使用量が閾値を超えているかチェック
        
        Returns:
            閾値超過の場合True
        """
        current_memory = self.get_memory_usage()
        return current_memory > self.max_memory_gb

    def force_memory_cleanup(self):
        """
        強制的なメモリクリーンアップ
        
        ガベージコレクションとGPUキャッシュクリアを実行します。
        """
        try:
            # Python ガベージコレクション
            gc.collect()
            
            # CUDA キャッシュクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # MPS キャッシュクリア（PyTorch 2.0以降）
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # MPSにはempty_cache相当の機能がないため、gc.collectのみ
                    pass
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"メモリクリーンアップエラー: {e}")

        # メモリ使用量をログ出力
        memory_after = self.get_memory_usage()
        self.logger.info(f"メモリクリーンアップ実行後: {memory_after:.2f}GB")

    def get_optimal_batch_size(self, available_memory_gb: float) -> int:
        """
        利用可能メモリに基づいて最適なバッチサイズを計算
        
        Args:
            available_memory_gb: 利用可能メモリ（GB）
            
        Returns:
            推奨バッチサイズ
        """
        if available_memory_gb < 2:
            return 8
        elif available_memory_gb < 4:
            return 16
        elif available_memory_gb < 8:
            return 32
        else:
            return 64

    def monitor_memory(self) -> Dict[str, float]:
        """
        メモリ使用状況の詳細モニタリング
        
        Returns:
            メモリ統計情報の辞書
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            stats = {
                "process_memory_gb": memory_info.rss / (1024**3),
                "total_memory_gb": virtual_memory.total / (1024**3),
                "available_memory_gb": virtual_memory.available / (1024**3),
                "memory_percent": virtual_memory.percent,
            }
            
            # GPU メモリ情報（CUDA）
            if torch.cuda.is_available():
                stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            
            return stats
        except Exception as e:
            self.logger.warning(f"メモリモニタリングエラー: {e}")
            return {}