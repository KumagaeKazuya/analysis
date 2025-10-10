"""
YOLO11 広角カメラ分析システム - 改良版（統一エラーハンドリング対応版）
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, Optional

# 🔧 統一エラーハンドラーからインポート
from utils.error_handler import (
    BaseYOLOError,
    ConfigurationError,
    VideoProcessingError,
    ResponseBuilder,
    handle_errors,
    ErrorContext,
    ErrorCategory,
    ErrorReporter,
    ErrorSeverity
)

# 既存モジュール
from evaluators.comprehensive_evaluator import ComprehensiveEvaluator
from processors.video_processor import VideoProcessor
from analyzers.metrics_analyzer import MetricsAnalyzer
from utils.config import Config
from utils.logger import setup_logger


class ImprovedYOLOAnalyzer:
    def __init__(self, config_path="configs/default.yaml"):
        self.config = Config(config_path)
        self.logger = setup_logger()
        self.evaluator = ComprehensiveEvaluator(self.config)
        self.processor = VideoProcessor(self.config)
        self.analyzer = MetricsAnalyzer(self.config)

        # 🔧 エラー収集用
        self.error_collector = []

        self._setup_directories()

    def _setup_directories(self):
        """必要なディレクトリを作成"""
        dirs = [
            "outputs/baseline", "outputs/experiments", "outputs/frames",
            "outputs/results", "outputs/logs", "outputs/reports",
            "outputs/visualizations"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @handle_errors(error_category=ErrorCategory.PROCESSING, suppress_exceptions=False)
    def run_baseline_establishment(self) -> Dict[str, Any]:
        """
        ベースライン確立処理（統一エラーハンドリング対応版）
        """
        with ErrorContext("ベースライン確立", logger=self.logger, raise_on_error=True) as ctx:
            self.logger.info("🚀 ベースライン確立を開始")

            # 実験ディレクトリ作成
            experiment_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir = Path("outputs/baseline") / experiment_name
            output_dir.mkdir(exist_ok=True)

            ctx.add_info("experiment_name", experiment_name)
            ctx.add_info("output_dir", str(output_dir))

            # 結果格納用
            results = {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "phase": "baseline",
                "videos": []
            }

            # 動画ファイル取得
            video_files = list(Path(self.config.video_dir).glob("*.mp4"))
            if not video_files:
                raise ConfigurationError(
                    f"動画ファイルが見つかりません: {self.config.video_dir}",
                    details={"video_dir": self.config.video_dir},
                    severity=ErrorSeverity.CRITICAL
                )

            ctx.add_info("video_count", len(video_files))

            # 各動画処理
            for video_path in video_files:
                self.logger.info(f"処理中: {video_path.name}")

                try:
                    video_result = self._process_single_video_baseline(video_path, output_dir)
                    results["videos"].append(video_result)

                except BaseYOLOError as e:
                    self.logger.error(f"動画処理エラー {video_path.name}: {e.message}")
                    self.error_collector.append(e)
                    
                    # エラー情報を結果に追加
                    results["videos"].append({
                        "video_name": video_path.stem,
                        "error": e.to_dict(),
                        "success": False
                    })
                    continue

            # 結果保存
            self._save_experiment_results(results, output_dir)
            self._generate_baseline_report(results, output_dir)

            # エラーレポート生成
            if self.error_collector:
                error_report = ErrorReporter.generate_report(self.error_collector)
                self.logger.warning(f"\n{error_report}")
                
                # エラーレポートをファイルに保存
                with open(output_dir / "error_report.txt", 'w', encoding='utf-8') as f:
                    f.write(error_report)

            self.logger.info("✅ ベースライン確立完了")
            
            return ResponseBuilder.success(
                data=results,
                message=f"ベースライン確立完了: {len(results['videos'])}動画処理"
            )

    @handle_errors(error_category=ErrorCategory.PROCESSING)
    def _process_single_video_baseline(self, video_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        単一動画のベースライン処理（統一エラーハンドリング対応版）
        """
        with ErrorContext(f"動画処理: {video_path.name}", logger=self.logger) as ctx:
            video_name = video_path.stem
            ctx.add_info("video_name", video_name)

            # 1. フレーム抽出
            frame_dir = output_dir / video_name / "frames"
            frame_result = self.processor.extract_frames(video_path, frame_dir)
            
            if not frame_result.get("success", False):
                raise VideoProcessingError(
                    "フレーム抽出に失敗しました",
                    details=frame_result.get("error", {})
                )

            # 2. 検出・追跡実行
            result_dir = output_dir / video_name / "results"
            detection_results = self.processor.run_detection_tracking(frame_dir, video_name)

            # 3. 結果チェック
            if not detection_results.get("success", False):
                error_msg = detection_results.get("error", "unknown_error")
                raise VideoProcessingError(
                    f"検出処理失敗: {error_msg}",
                    details=detection_results
                )

            # ✅ 修正: ResponseBuilder形式に対応したCSVパス取得
            result_data = detection_results.get("data", {})
            csv_path = result_data.get("csv_path")

            # 4. 評価
            if not csv_path or not Path(csv_path).exists():
                self.logger.warning(f"CSVファイル未生成: {csv_path}")
                metrics = ResponseBuilder.error(
                    Exception("CSV未生成"),
                    suggestions=["フレーム抽出と検出処理を確認してください"]
                )
            else:
                try:
                    metrics = self.evaluator.evaluate_comprehensive(
                        video_path, detection_results, video_name
                    )
                except Exception as e:
                    self.logger.error(f"評価エラー: {e}")
                    metrics = ResponseBuilder.error(e)

            # 5. 可視化
            vis_dir = output_dir / video_name / "visualizations"
            try:
                self.analyzer.create_visualizations(detection_results, vis_dir)
            except Exception as e:
                self.logger.warning(f"可視化エラー: {e}")

            # ✅ 修正: 統計情報も data から取得
            processing_stats = result_data.get("processing_stats", {})

            return {
                "video_name": video_name,
                "video_path": str(video_path),
                "metrics": metrics,
                "frame_count": len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0,
                "detection_file": csv_path or "",
                "processing_stats": detection_results.get("processing_stats", {}),
                "success": detection_results.get("success", False)
            }

    @handle_errors(error_category=ErrorCategory.PROCESSING, suppress_exceptions=False)
    def run_improvement_experiment(self, experiment_type: str) -> Dict[str, Any]:
        """
        改善実験処理（統一エラーハンドリング対応版）
        """
        with ErrorContext(f"改善実験: {experiment_type}", logger=self.logger) as ctx:
            self.logger.info(f"🔬 改善実験開始: {experiment_type}")

            try:
                # 実験設定読み込み
                exp_config = self.config.get_experiment_config(experiment_type)
                
                # 実験名と出力ディレクトリ作成
                experiment_name = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = Path("outputs/experiments") / experiment_name
                output_dir.mkdir(exist_ok=True)

                ctx.add_info("experiment_name", experiment_name)
                ctx.add_info("experiment_type", experiment_type)

                # ベースラインとの比較実験実行
                comparison_results = self._run_comparison_experiment(exp_config, output_dir)

                # 改善効果分析
                improvement_analysis = self.analyzer.analyze_improvements(comparison_results)

                # レポート生成
                self._generate_improvement_report(improvement_analysis, output_dir)

                self.logger.info("✅ 改善実験完了")
                
                return ResponseBuilder.success(
                    data=improvement_analysis,
                    message=f"改善実験完了: {experiment_type}"
                )

            except Exception as e:
                self.logger.error(f"改善実験エラー: {e}")
                return ResponseBuilder.error(e)

    def _save_experiment_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """実験結果をJSON保存"""
        try:
            results_file = output_dir / "experiment_results.json"

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"実験結果保存: {results_file}")

        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")

    def _generate_baseline_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """ベースラインレポートを生成"""
        try:
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)

            try:
                from reports.baseline_report_generator import BaselineReportGenerator

                report_generator = BaselineReportGenerator(results, self.config)
                report_generator.generate_html_report(reports_dir / "baseline_report.html")
                report_generator.generate_markdown_report(reports_dir / "baseline_report.md")

                self.logger.info(f"ベースラインレポート生成完了: {reports_dir}")

            except ImportError as e:
                self.logger.warning(f"レポート生成モジュールが見つかりません: {e}")
                self._generate_simple_baseline_report(results, reports_dir)

        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")

    def _generate_simple_baseline_report(self, results: Dict[str, Any], reports_dir: Path) -> None:
        """簡易ベースラインレポート生成"""
        try:
            # JSON形式で保存
            with open(reports_dir / "baseline_report.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "experiment_name": results.get("experiment_name", "baseline"),
                    "results": results
                }, f, indent=2, ensure_ascii=False)

            # 簡易HTML生成
            html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>ベースラインレポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>ベースラインレポート</h1>
    <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>実験結果</h2>
    <pre>{json.dumps(results, indent=2, ensure_ascii=False)}</pre>
</body>
</html>"""

            with open(reports_dir / "baseline_report.html", 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info("簡易ベースラインレポート生成完了")

        except Exception as e:
            self.logger.error(f"簡易レポート生成エラー: {e}")

    def _run_comparison_experiment(self, exp_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """
        ベースラインとの比較実験
        実験タイプごとに処理を分岐
        """
        try:
            from experiments.experiment_runner import ExperimentRunner
            experiment_runner = ExperimentRunner(self.config)

            experiment_type = exp_config["type"]

            if experiment_type == "camera_calibration":
                return experiment_runner.run_calibration_experiment(exp_config, output_dir)
            elif experiment_type == "model_ensemble":
                return experiment_runner.run_ensemble_experiment(exp_config, output_dir)
            elif experiment_type == "data_augmentation":
                return experiment_runner.run_augmentation_experiment(exp_config, output_dir)
            elif experiment_type == "tile_inference_comparison":
                return experiment_runner.run_tile_comparison_experiment(exp_config, output_dir)
            else:
                return ResponseBuilder.error(
                    Exception(f"未知の実験タイプ: {experiment_type}"),
                    suggestions=["対応実験タイプ: camera_calibration, model_ensemble, data_augmentation, tile_inference_comparison"]
                )

        except ImportError as e:
            self.logger.error(f"実験ランナーのインポートエラー: {e}")
            return ResponseBuilder.error(e, suggestions=["experiments.experiment_runnerモジュールを確認してください"])
        except Exception as e:
            self.logger.error(f"実験実行エラー: {e}")
            return ResponseBuilder.error(e)

    def _generate_improvement_report(self, improvement_analysis: Dict[str, Any], output_dir: Path) -> None:
        """改善実験レポート生成"""
        try:
            from reports.improvement_report_generator import ImprovementReportGenerator
            report_generator = ImprovementReportGenerator(improvement_analysis, self.config)
            report_generator.generate_html_report(output_dir / "improvement_report.html")
            report_generator.generate_markdown_report(output_dir / "improvement_report.md")

            self.logger.info(f"改善実験レポート生成完了: {output_dir}")

        except ImportError:
            self.logger.warning("ImprovementReportGenerator が見つかりません。簡易レポートを生成します。")
            self._generate_simple_improvement_report(improvement_analysis, output_dir)
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")

    def _generate_simple_improvement_report(self, improvement_analysis: Dict[str, Any], output_dir: Path) -> None:
        """簡易改善レポート生成（フォールバック）"""
        try:
            # JSON保存
            with open(output_dir / "improvement_report.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "improvement_analysis": improvement_analysis
                }, f, indent=2, ensure_ascii=False)

            # 簡易HTML生成
            html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>改善実験レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>改善実験レポート</h1>
    <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>改善分析結果</h2>
    <pre>{json.dumps(improvement_analysis, indent=2, ensure_ascii=False)}</pre>
</body>
</html>"""

            with open(output_dir / "improvement_report.html", 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info("簡易改善レポート生成完了")

        except Exception as e:
            self.logger.error(f"簡易改善レポート生成エラー: {e}")


def main():
    """メイン実行（統一エラーハンドリング対応版）"""
    parser = argparse.ArgumentParser(description='YOLO11 広角カメラ分析システム')
    parser.add_argument('--mode', choices=['baseline', 'experiment'],
                        default='baseline', help='実行モード')
    parser.add_argument('--experiment-type', type=str, help='実験タイプ')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='設定ファイル')

    args = parser.parse_args()

    try:
        analyzer = ImprovedYOLOAnalyzer(args.config)

        if args.mode == 'baseline':
            result = analyzer.run_baseline_establishment()

            if result.get("success", False):
                print("✅ ベースライン確立成功")
                # 詳細情報表示
                data = result.get("data", {})
                videos = data.get("videos", [])
                success_count = len([v for v in videos if v.get("success", False)])
                print(f"処理済み動画: {len(videos)}件 (成功: {success_count}件)")
            else:
                print(f"❌ エラー: {result.get('error', {}).get('message', 'unknown')}")
                sys.exit(1)

        elif args.mode == 'experiment':
            if not args.experiment_type:
                print("実験モードでは --experiment-type が必要です")
                print("利用可能な実験タイプ: camera_calibration, model_ensemble, data_augmentation, tile_inference_comparison")
                sys.exit(1)

            result = analyzer.run_improvement_experiment(args.experiment_type)

            if result.get("success", False):
                print(f"✅ 実験完了: {args.experiment_type}")
                # 改善結果の概要表示
                data = result.get("data", {})
                if "improvement" in data:
                    improvement = data["improvement"]
                    print(f"改善効果: {improvement}")
            else:
                print(f"❌ エラー: {result.get('error', {}).get('message', 'unknown')}")
                sys.exit(1)

    except BaseYOLOError as e:
        print(f"❌ エラー発生: {e.message}")
        print(f"カテゴリ: {e.category.value}")
        print(f"深刻度: {e.severity.value}")

        if e.details:
            print(f"詳細: {e.details}")

        if e.suggestions:
            print(f"解決策: {', '.join(e.suggestions)}")

        logging.error("実行エラー", exc_info=True)
        sys.exit(1)

    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        logging.error("予期しないエラー", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()