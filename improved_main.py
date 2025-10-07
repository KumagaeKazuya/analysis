# 改良版メインシステム

"""
YOLO11 広角カメラ分析システム - 改良版
ベースライン確立から改善実験まで対応
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from reports.baseline_report_generator import BaselineReportGenerator
import argparse

# 自作モジュール
from evaluators.comprehensive_evaluator import ComprehensiveEvaluator
from processors.video_processor import VideoProcessor
from analyzers.metrics_analyzer import MetricsAnalyzer
from utils.config import Config
from utils.logger import setup_logger

class ImprovedYOLOAnalyzer:
    def __init__(self, config_path="configs/default.yaml"):
        # 設定ファイル読み込みと各種モジュール初期化
        self.config = Config(config_path)
        self.logger = setup_logger()
        self.evaluator = ComprehensiveEvaluator(self.config)
        self.processor = VideoProcessor(self.config)
        self.analyzer = MetricsAnalyzer(self.config)

        # 出力ディレクトリ作成
        self._setup_directories()

    def _setup_directories(self):
        """必要なディレクトリを作成"""
        dirs = [
            "outputs/baseline",
            "outputs/experiments",
            "outputs/frames",
            "outputs/results",
            "outputs/logs",
            "outputs/reports",
            "outputs/visualizations"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def run_baseline_establishment(self):
        """
        ベースライン確立処理
        1. 動画ファイルを取得
        2. 各動画ごとにフレーム抽出・検出・追跡・評価・可視化
        3. 結果保存とレポート生成
        """
        self.logger.info("🚀 ベースライン確立を開始")

        # 実験名と出力ディレクトリ作成
        experiment_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("outputs/baseline") / experiment_name
        output_dir.mkdir(exist_ok=True)

        # 結果格納用辞書
        results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "phase": "baseline",
            "videos": []
        }

        # 動画ファイル一覧取得
        video_files = list(Path(self.config.video_dir).glob("*.mp4"))
        if not video_files:
            self.logger.error(f"動画ファイルが見つかりません: {self.config.video_dir}")
            return None

        # 各動画ごとに処理
        for video_path in video_files:
            self.logger.info(f"処理中: {video_path.name}")

            try:
                # 単一動画のベースライン処理
                video_result = self._process_single_video_baseline(video_path, output_dir)
                results["videos"].append(video_result)

            except Exception as e:
                self.logger.error(f"動画処理エラー {video_path.name}: {e}")
                continue

        # 結果保存とレポート生成
        self._save_experiment_results(results, output_dir)
        self._generate_baseline_report(results, output_dir)

        self.logger.info("✅ ベースライン確立完了")
        return results

def _process_single_video_baseline(self, video_path, output_dir):
    """
    単一動画のベースライン処理
    1. フレーム抽出
    2. 検出・追跡
    3. 評価
    4. 可視化
    """
    video_name = video_path.stem

    # 1. フレーム抽出
    frame_dir = output_dir / video_name / "frames"  # ← 修正: パスを調整
    self.processor.extract_frames(video_path, frame_dir)

    # 2. 検出・追跡実行
    result_dir = output_dir / video_name / "results"  # ← 修正: 結果ディレクトリを明示
    detection_results = self.processor.run_detection_tracking(frame_dir, video_name)

    # 3. エラーハンドリング強化
    if not detection_results.get("success", False):
        self.logger.error(f"検出処理失敗: {detection_results.get('error', 'unknown')}")
        return {
            "video_name": video_name,
            "video_path": str(video_path),
            "error": detection_results.get("error", "detection_failed"),
            "frame_count": len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0
        }

    # 4. 評価（CSVパスを安全に取得）
    csv_path = detection_results.get("csv_path")
    if not csv_path or not Path(csv_path).exists():
        self.logger.warning(f"CSVファイルが見つかりません: {csv_path}")
        # CSVがなくても処理を続行
        metrics = {"error": "csv_not_found"}
    else:
        try:
            metrics = self.evaluator.evaluate_comprehensive(
                video_path, detection_results, video_name
            )
        except Exception as e:
            self.logger.error(f"評価エラー: {e}")
            metrics = {"error": str(e)}

    # 5. 可視化
    vis_dir = output_dir / video_name / "visualizations"
    try:
        self.analyzer.create_visualizations(detection_results, vis_dir)
    except Exception as e:
        self.logger.warning(f"可視化エラー: {e}")

    # 結果を辞書で返す
    return {
        "video_name": video_name,
        "video_path": str(video_path),
        "metrics": metrics,
        "frame_count": len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0,
        "detection_file": csv_path or "",
        "processing_stats": detection_results.get("processing_stats", {}),
        "success": detection_results.get("success", False)
    }

def run_improvement_experiment(self, experiment_type):
    """
    改善実験処理
    1. 実験設定読み込み
    2. ベースラインとの比較実験
    3. 改善効果分析
    4. レポート生成
    """
    self.logger.info(f"🔬 改善実験開始: {experiment_type}")

    try:
        # 実験設定読み込み
        exp_config = self.config.get_experiment_config(experiment_type)

        # 実験名と出力ディレクトリ作成
        experiment_name = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("outputs/experiments") / experiment_name
        output_dir.mkdir(exist_ok=True)

        # ベースラインとの比較実験実行
        comparison_results = self._run_comparison_experiment(exp_config, output_dir)

        # 改善効果分析
        improvement_analysis = self.analyzer.analyze_improvements(comparison_results)

        # レポート生成
        self._generate_improvement_report(improvement_analysis, output_dir)

        self.logger.info("✅ 改善実験完了")
        return improvement_analysis

    except Exception as e:
        self.logger.error(f"改善実験エラー: {e}")
        return {"error": str(e), "success": False}

def _save_experiment_results(self, results, output_dir):
    """実験結果をJSONファイルに保存"""
    import json
    from datetime import datetime

    try:
        results_file = output_dir / "experiment_results.json"

        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=convert_datetime)

        self.logger.info(f"実験結果を保存: {results_file}")

    except Exception as e:
        self.logger.error(f"実験結果保存エラー: {e}")

def _generate_baseline_report(self, results, output_dir):
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

def _generate_simple_baseline_report(self, results, reports_dir):
    """簡易ベースラインレポート生成"""
    import json
    from datetime import datetime

    try:
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "experiment_name": results.get("experiment_name", "baseline"),
            "results": results
        }

        with open(reports_dir / "baseline_report.json", 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

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

def _run_comparison_experiment(self, exp_config, output_dir):
    """
    ベースラインとの比較実験
    実験タイプごとに処理を分岐
    """
    # 実験ランナーを動的インポート
    try:
        from experiments.experiment_runner import ExperimentRunner
        experiment_runner = ExperimentRunner(self.config)

        if exp_config["type"] == "camera_calibration":
            return experiment_runner.run_calibration_experiment(exp_config, output_dir)
        elif exp_config["type"] == "model_ensemble":
            return experiment_runner.run_ensemble_experiment(exp_config, output_dir) 
        elif exp_config["type"] == "data_augmentation":
            return experiment_runner.run_augmentation_experiment(exp_config, output_dir)
        elif exp_config["type"] == "tile_inference_comparison":
            return experiment_runner.run_tile_comparison_experiment(exp_config, output_dir)
        else:
            return {"error": f"unknown_experiment_type: {exp_config['type']}", "success": False}

    except ImportError as e:
        self.logger.error(f"実験ランナーのインポートエラー: {e}")
        return {"error": "experiment_runner_import_failed", "success": False}
    except Exception as e:
        self.logger.error(f"実験実行エラー: {e}")
        return {"error": f"experiment_execution_failed: {e}", "success": False}

def _generate_improvement_report(self, improvement_analysis, output_dir):
    """改善実験レポート生成"""
    try:
        from reports.improvement_report_generator import ImprovementReportGenerator
        report_generator = ImprovementReportGenerator(improvement_analysis, self.config)
        report_generator.generate_html_report(output_dir / "improvement_report.html")
        report_generator.generate_markdown_report(output_dir / "improvement_report.md")

        self.logger.info(f"改善実験レポート生成完了: {output_dir}")

    except ImportError:
        # フォールバック: 簡易レポート生成
        self.logger.warning("ImprovementReportGenerator が見つかりません。簡易レポートを生成します。")
        self._generate_simple_improvement_report(improvement_analysis, output_dir)
    except Exception as e:
        self.logger.error(f"レポート生成エラー: {e}")

def _generate_simple_improvement_report(self, improvement_analysis, output_dir):
    """簡易改善レポート生成（フォールバック）"""
    import json

    # JSON形式でレポート保存
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "improvement_analysis": improvement_analysis
    }

    with open(output_dir / "improvement_report.json", 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # 簡易HTMLレポート
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>改善実験レポート</title>
    </head>
    <body>
        <h1>改善実験レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <pre>{json.dumps(improvement_analysis, indent=2, ensure_ascii=False)}</pre>
    </body>
    </html>
    """

    with open(output_dir / "improvement_report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

    self.logger.info("簡易改善レポート生成完了")

def main():
    """
    コマンドライン引数を受け取り、実行モードに応じて処理を分岐
    --mode baseline: ベースライン確立
    --mode experiment: 改善実験
    """
    parser = argparse.ArgumentParser(description='YOLO11 広角カメラ分析システム')
    parser.add_argument('--mode', choices=['baseline', 'experiment'],
                    default='baseline', help='実行モード')
    parser.add_argument('--experiment-type', type=str,
                    help='実験タイプ (experiment mode時)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                    help='設定ファイルパス')

    args = parser.parse_args()

    try:
        # メインクラス初期化
        analyzer = ImprovedYOLOAnalyzer(args.config)

        # ベースライン確立モード
        if args.mode == 'baseline':
            results = analyzer.run_baseline_establishment()

        # 改善実験モード
        elif args.mode == 'experiment':
            if not args.experiment_type:
                print("実験モードでは --experiment-type が必要です")
                sys.exit(1)
            results = analyzer.run_improvement_experiment(args.experiment_type)

    except Exception as e:
        logging.error(f"実行エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()