# ログ管理

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = "yolo11_analyzer",
                level: str = "INFO",
                log_file: bool = True,
                log_dir: str = "outputs/logs") -> logging.Logger:
    """
    ログシステムのセットアップ

    Args:
        name: ロガー名
        level: ログレベル
        log_file: ファイル出力の有無
        log_dir: ログファイル出力ディレクトリ

    Returns:
        logging.Logger: 設定済みロガー
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 既存のハンドラーをクリア
    logger.handlers.clear()

    # フォーマッター作成
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラー（オプション）
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = Path(log_dir) / f"yolo11_analyzer_{timestamp}.log"

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"ログファイル作成: {log_file_path}")

    return logger

class ProgressLogger:
    """進捗表示用のログクラス"""

    def __init__(self, logger: logging.Logger, total: int, name: str = "処理"):
        self.logger = logger
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = datetime.now()

    def update(self, increment: int = 1) -> None:
        """進捗を更新"""
        self.current += increment
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            elapsed = datetime.now() - self.start_time

            if self.current == self.total:
                self.logger.info(
                    f"✅ {self.name}完了: {self.current}/{self.total} "
                    f"(100.0%) - 所要時間: {elapsed}"
                )
            else:
                eta = elapsed * (self.total - self.current) / self.current
                self.logger.info(
                    f"⏳ {self.name}進捗: {self.current}/{self.total} "
                    f"({percentage:.1f}%) - 残り時間: {eta}"
                )

    def finish(self) -> None:
        """処理完了"""
        if self.current < self.total:
            self.current = self.total
            self.update(0)