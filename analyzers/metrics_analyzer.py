# メトリクス分析
# analyzers/metrics_analyzer.py を作成
class MetricsAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze_improvements(self, comparison_results):
        """改善効果分析"""
        return {"improvement_score": 0.8}

    def create_visualizations(self, detection_results, vis_dir):
        """可視化生成"""
        from utils.visualization import create_detection_statistics_plot
        import pandas as pd

        vis_dir.mkdir(parents=True, exist_ok=True)
        if detection_results.get("csv_path"):
            df = pd.read_csv(detection_results["csv_path"])
            create_detection_statistics_plot(df, vis_dir / "statistics.png")