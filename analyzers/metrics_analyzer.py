class MetricsAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze_improvements(self, comparison_results):
        """改善効果分析"""
        return {"improvement_score": 0.8}

    def create_visualizations(self, detection_results, vis_dir):
        """可視化生成（csv_path探索を強化し例外時も必ず辞書を返す）"""
        try:
            from utils.visualization import create_detection_statistics_plot
            import pandas as pd

            vis_dir.mkdir(parents=True, exist_ok=True)
            # csv_pathを直下とdataの両方から探索
            csv_path = detection_results.get("csv_path")
            if not csv_path and isinstance(detection_results, dict):
                data = detection_results.get("data", {})
                csv_path = data.get("csv_path")
            if csv_path:
                df = pd.read_csv(csv_path)
                vis_path = vis_dir / "statistics.png"
                create_detection_statistics_plot(df, vis_path)
                visualizations = [str(vis_path)]
                return {
                    "success": True,
                    "visualizations": visualizations,
                    "total_files": len(visualizations),
                    "graphs_generated": len(visualizations)
                }
            else:
                return {
                    "success": False,
                    "error": "csv_pathが検出結果に含まれていません",
                    "total_files": 0,
                    "graphs_generated": 0
                }
        except Exception as e:
            import traceback
            if hasattr(self, "logger"):
                self.logger.error(f"❌ 可視化生成エラー: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "total_files": 0,
                "graphs_generated": 0
            }