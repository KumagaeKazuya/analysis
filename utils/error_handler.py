"""
統一エラーハンドリングシステム

プロジェクト全体で一貫したエラー処理を提供します。
"""

import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from datetime import datetime
from enum import Enum

# ========================================
# 1. エラーカテゴリの定義
# ========================================

class ErrorCategory(Enum):
    """エラーカテゴリ"""
    VALIDATION = "validation"
    INITIALIZATION = "initialization"
    PROCESSING = "processing"
    EVALUATION = "evaluation"
    MODEL = "model"  # 🔧 新規追加
    EXPERIMENT = "experiment"  # 🔧 新規追加
    DEPTH_PROCESSING = "depth_processing"  # 🔧 新規追加
    VIDEO_PROCESSING = "video_processing"  # 🔧 新規追加
    RESOURCE = "resource"
    IO = "io"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """エラー深刻度"""
    CRITICAL = "critical"  # システム停止レベル
    ERROR = "error"        # 処理失敗
    WARNING = "warning"    # 警告
    INFO = "info"          # 情報


# ========================================
# 2. カスタムエラークラス階層
# ========================================

class BaseYOLOError(Exception):
    """
    プロジェクト基底エラークラス
    
    全てのカスタムエラーはこれを継承します。
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """エラー情報を辞書形式で返す"""
        error_dict = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
        
        if self.original_exception:
            error_dict["original_error"] = str(self.original_exception)
            error_dict["traceback"] = traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            )
        
        return error_dict


# --- 具体的なエラークラス ---

class ValidationError(BaseYOLOError):
    """入力検証エラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ModelInitializationError(BaseYOLOError):
    """モデル初期化エラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INITIALIZATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class VideoProcessingError(BaseYOLOError):
    """動画処理エラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ResourceExhaustionError(BaseYOLOError):
    """リソース不足エラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class FileIOError(BaseYOLOError):
    """ファイルI/Oエラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.IO,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


class ConfigurationError(BaseYOLOError):
    """設定エラー"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            **kwargs
        )


# ========================================
# 3. 標準化されたレスポンス形式
# ========================================

class ResponseBuilder:
    """標準化されたレスポンスを生成"""
    
    @staticmethod
    def success(data: Any = None, message: str = "処理成功") -> Dict[str, Any]:
        """成功レスポンス"""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        return response
    
    @staticmethod
    def error(
        error: Union[BaseYOLOError, Exception],
        include_traceback: bool = True,
        suggestions: Optional[list] = None
    ) -> Dict[str, Any]:
        """エラーレスポンス"""
        
        if isinstance(error, BaseYOLOError):
            response = {
                "success": False,
                "error": error.to_dict()
            }
        else:
            # 標準的なExceptionの場合
            response = {
                "success": False,
                "error": {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "category": ErrorCategory.UNKNOWN.value,
                    "severity": ErrorSeverity.ERROR.value,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            if include_traceback:
                response["error"]["traceback"] = traceback.format_exc()
        
        if suggestions:
            response["suggestions"] = suggestions
        
        return response
    
    @staticmethod
    def validation_error(
        field: str,
        message: str,
        value: Any = None
    ) -> Dict[str, Any]:
        """検証エラー（特化版）"""
        error = ValidationError(
            message=f"検証エラー: {field}",
            details={
                "field": field,
                "message": message,
                "value": value
            }
        )
        return ResponseBuilder.error(error)


# ========================================
# 4. エラーハンドリングデコレータ
# ========================================

T = TypeVar('T')

def handle_errors(
    logger: Optional[logging.Logger] = None,
    default_return: Any = None,
    suppress_exceptions: bool = False,
    error_category: ErrorCategory = ErrorCategory.PROCESSING
):
    """
    エラーハンドリングデコレータ
    
    使用例:
        @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
        def process_video(video_path: str) -> Dict[str, Any]:
            # 処理
            return ResponseBuilder.success(data={"result": "ok"})
    
    Args:
        logger: ロガーインスタンス
        default_return: エラー時のデフォルト戻り値
        suppress_exceptions: 例外を抑制するか（Falseの場合は再送出）
        error_category: エラーカテゴリ
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            
            except BaseYOLOError as e:
                # カスタムエラーの場合
                func_logger.error(
                    f"{func.__name__}でエラー発生: {e.message}",
                    extra={"error_details": e.to_dict()}
                )
                
                if not suppress_exceptions:
                    raise
                
                return default_return or ResponseBuilder.error(e)
            
            except Exception as e:
                # 予期しないエラーの場合
                func_logger.error(
                    f"{func.__name__}で予期しないエラー: {e}",
                    exc_info=True
                )

                if not suppress_exceptions:
                    raise

                # 予期しないエラーをラップ
                wrapped_error = BaseYOLOError(
                    message=f"予期しないエラー in {func.__name__}: {e}",
                    category=error_category,
                    severity=ErrorSeverity.ERROR,
                    original_exception=e
                )

                return default_return or ResponseBuilder.error(wrapped_error)

        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    入力検証デコレータ

    使用例:
        @validate_inputs(
            video_path=lambda x: Path(x).exists(),
            confidence=lambda x: 0 <= x <= 1
        )
        def analyze_video(video_path: str, confidence: float):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logging.getLogger(func.__module__)

            # 引数名を取得
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 検証実行
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    try:
                        if not validator(value):
                            raise ValidationError(
                                f"パラメータ '{param_name}' の検証失敗",
                                details={
                                    "parameter": param_name,
                                    "value": value,
                                    "validator": validator.__name__ if hasattr(validator, '__name__') else str(validator)
                                }
                            )
                    except ValidationError:
                        raise
                    except Exception as e:
                        func_logger.warning(
                            f"検証関数自体がエラー: {param_name}",
                            exc_info=True
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


# ========================================
# 5. コンテキストマネージャー
# ========================================

class ErrorContext:
    """
    エラーコンテキスト管理

    使用例:
        with ErrorContext("動画処理", logger=logger) as ctx:
            # 処理
            ctx.add_info("frames_processed", 100)
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        raise_on_error: bool = True
    ):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error
        self.context_info = {}
        self.error: Optional[Exception] = None

    def __enter__(self):
        self.logger.info(f"開始: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"完了: {self.operation_name}")
            return True

        self.error = exc_val

        # エラー情報をログ出力
        self.logger.error(
            f"エラー in {self.operation_name}: {exc_val}",
            extra={"context_info": self.context_info},
            exc_info=True
        )

        # 例外を抑制するかどうか
        return not self.raise_on_error

    def add_info(self, key: str, value: Any):
        """コンテキスト情報を追加"""
        self.context_info[key] = value


# ========================================
# 6. エラーレポート生成
# ========================================

class ErrorReporter:
    """エラーレポート生成"""

    @staticmethod
    def generate_report(errors: list[BaseYOLOError]) -> str:
        """エラーサマリーレポート生成"""
        if not errors:
            return "エラーなし"

        report_lines = [
            "=" * 60,
            f"エラーレポート ({len(errors)}件)",
            "=" * 60,
            ""
        ]

        # カテゴリ別集計
        from collections import defaultdict
        by_category = defaultdict(list)
        by_severity = defaultdict(list)

        for error in errors:
            by_category[error.category].append(error)
            by_severity[error.severity].append(error)

        report_lines.append("【カテゴリ別】")
        for category, errs in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
            report_lines.append(f"  {category.value}: {len(errs)}件")

        report_lines.append("")
        report_lines.append("【深刻度別】")
        for severity, errs in sorted(by_severity.items(), key=lambda x: x[0].value):
            report_lines.append(f"  {severity.value}: {len(errs)}件")

        report_lines.append("")
        report_lines.append("【詳細】")
        for i, error in enumerate(errors[:10], 1):  # 最大10件
            report_lines.append(f"{i}. [{error.severity.value.upper()}] {error.message}")
            report_lines.append(f"   カテゴリ: {error.category.value}")
            report_lines.append(f"   時刻: {error.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

        if len(errors) > 10:
            report_lines.append(f"... 他 {len(errors) - 10}件")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


# ========================================
# 7. 使用例とテスト
# ========================================

if __name__ == "__main__":
    # ロガー設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # === 使用例1: デコレータでエラーハンドリング ===
    @handle_errors(logger=logger, error_category=ErrorCategory.PROCESSING)
    def process_video_example(video_path: str) -> Dict[str, Any]:
        """動画処理の例"""
        if not video_path.endswith('.mp4'):
            raise ValidationError(
                "無効な動画形式",
                details={"video_path": video_path, "expected": ".mp4"}
            )

        # 正常処理
        return ResponseBuilder.success(
            data={"frames": 100, "detections": 50},
            message="動画処理完了"
        )

    # === 使用例2: 入力検証デコレータ ===
    @validate_inputs(
        confidence=lambda x: 0 <= x <= 1,
        video_path=lambda x: isinstance(x, str) and len(x) > 0
    )
    @handle_errors(logger=logger)
    def analyze_with_validation(video_path: str, confidence: float):
        """検証付き分析"""
        return ResponseBuilder.success(
            data={"video": video_path, "conf": confidence}
        )

    # === 使用例3: コンテキストマネージャー ===
    def process_with_context():
        """コンテキストマネージャーの例"""
        with ErrorContext("バッチ処理", logger=logger) as ctx:
            ctx.add_info("batch_size", 32)
            ctx.add_info("total_frames", 1000)

            # 処理
            print("処理中...")

    # === テスト実行 ===
    print("\n=== テスト1: 正常系 ===")
    result = process_video_example("test.mp4")
    print(f"成功: {result['success']}")
    print(f"データ: {result.get('data')}")

    print("\n=== テスト2: エラー系 ===")
    result = process_video_example("test.avi")
    print(f"成功: {result['success']}")
    print(f"エラー: {result['error']['message']}")

    print("\n=== テスト3: 検証エラー ===")
    try:
        analyze_with_validation("test.mp4", confidence=1.5)
    except ValidationError as e:
        print(f"検証エラー発生: {e.message}")

    print("\n=== テスト4: コンテキスト ===")
    process_with_context()

    print("\n=== テスト5: エラーレポート ===")
    test_errors = [
        ValidationError("テストエラー1", details={"field": "video_path"}),
        ModelInitializationError("テストエラー2", details={"model": "yolo11n.pt"}),
        ResourceExhaustionError("テストエラー3", details={"memory": "8GB"})
    ]

    report = ErrorReporter.generate_report(test_errors)
    print(report)