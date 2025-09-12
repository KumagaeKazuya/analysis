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
import argparse

# 自作モジュール
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
        """9/11-19: ベースライン確立"""
        self.logger.info("🚀 ベースライン確立を開始")

        experiment_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("outputs/baseline") / experiment_name
        output_dir.mkdir(exist_ok=True)

        results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "phase": "baseline",
            "videos": []
        }

        # 動画ファイル処理
        video_files = list(Path(self.config.video_dir).glob("*.mp4"))
        if not video_files:
            self.logger.error(f"動画ファイルが見つかりません: {self.config.video_dir}")
            return None

        for video_path in video_files:
            self.logger.info(f"処理中: {video_path.name}")

            try:
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
        """単一動画のベースライン処理"""
        video_name = video_path.stem

        # 1. フレーム抽出
        frame_dir = output_dir / "frames" / video_name
        self.processor.extract_frames(video_path, frame_dir)

        # 2. 検出・追跡実行
        detection_results = self.processor.run_detection_tracking(frame_dir, video_name)

        # 3. 詳細評価
        metrics = self.evaluator.evaluate_comprehensive(
            video_path, detection_results, video_name
        )

        # 4. 可視化生成
        vis_dir = output_dir / "visualizations" / video_name
        self.analyzer.create_visualizations(detection_results, vis_dir)

        return {
            "video_name": video_name,
            "video_path": str(video_path),
            "metrics": metrics,
            "frame_count": len(list(frame_dir.glob("*.jpg"))),
            "detection_file": str(detection_results.get("csv_path", ""))
        }

    def run_improvement_experiment(self, experiment_type):
        """9/20以降: 改善実験"""
        self.logger.info(f"🔬 改善実験開始: {experiment_type}")

        # 実験設定読み込み
        exp_config = self.config.get_experiment_config(experiment_type)

        experiment_name = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path("outputs/experiments") / experiment_name
        output_dir.mkdir(exist_ok=True)

        # ベースラインとの比較実験実行
        comparison_results = self._run_comparison_experiment(exp_config, output_dir)

        # 改善効果分析
        improvement_analysis = self.analyzer.analyze_improvements(comparison_results)

        # レポート生成
        self._generate_improvement_report(improvement_analysis, output_dir)

        return improvement_analysis

    def _run_comparison_experiment(self, exp_config, output_dir):
        """ベースラインとの比較実験"""
        # 実装例: カメラキャリブレーション適用版
        if exp_config["type"] == "calibration":
            return self._run_calibration_experiment(exp_config, output_dir)
        elif exp_config["type"] == "ensemble":
            return self._run_ensemble_experiment(exp_config, output_dir)
        # 他の実験タイプも同様に追加

    def _save_experiment_results(self, results, output_dir):
        """実験結果の保存"""
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _generate_baseline_report(self, results, output_dir):
        """ベースラインレポート生成"""
        report_generator = BaselineReportGenerator(results, self.config)
        report_generator.generate_html_report(output_dir / "baseline_report.html")
        report_generator.generate_markdown_report(output_dir / "baseline_report.md")

def main():
    parser = argparse.ArgumentParser(description='YOLO11 広角カメラ分析システム')
    parser.add_argument('--mode', choices=['baseline', 'experiment'],
                    default='baseline', help='実行モード')
    parser.add_argument('--experiment-type', type=str,
                    help='実験タイプ (experiment mode時)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                    help='設定ファイルパス')

    args = parser.parse_args()

    try:
        analyzer = ImprovedYOLOAnalyzer(args.config)

        if args.mode == 'baseline':
            results = analyzer.run_baseline_establishment()

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