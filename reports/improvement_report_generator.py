"""
改善実験レポート生成モジュール
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

class ImprovementReportGenerator:
    """改善実験レポート生成クラス"""

    def __init__(self, improvement_analysis: Dict[str, Any], config):
        self.improvement_analysis = improvement_analysis
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_html_report(self, output_path: Path):
        """HTML形式のレポートを生成"""
        try:
            html_content = self._create_html_template()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"HTMLレポート生成: {output_path}")

        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")

    def generate_markdown_report(self, output_path: Path):
        """Markdown形式のレポートを生成"""
        try:
            md_content = self._create_markdown_template()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            self.logger.info(f"Markdownレポート生成: {output_path}")

        except Exception as e:
            self.logger.error(f"Markdownレポート生成エラー: {e}")

    def _create_html_template(self) -> str:
        """HTMLテンプレートを作成"""

        # 改善指標の抽出
        improvement_score = self.improvement_analysis.get("improvement_score", 0)
        experiment_type = self._get_experiment_type()

        html_template = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>改善実験レポート - {experiment_type}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #e8f5e8; padding: 15px; border-left: 4px solid #4CAF50; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .improvement-positive {{ color: #4CAF50; font-weight: bold; }}
                .improvement-negative {{ color: #f44336; font-weight: bold; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .data-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>改善実験レポート</h1>
                <p><strong>実験タイプ:</strong> {experiment_type}</p>
                <p><strong>生成日時:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>総合改善スコア:</strong> <span class="{'improvement-positive' if improvement_score > 0 else 'improvement-negative'}">{improvement_score:.3f}</span></p>
            </div>

            <div class="summary">
                <h2>📊 実験サマリー</h2>
                {self._generate_summary_html()}
            </div>

            <h2>🔍 詳細メトリクス</h2>
            <div class="metrics">
                {self._generate_metrics_html()}
            </div>

            <h2>📈 改善分析</h2>
            {self._generate_improvement_analysis_html()}

            <h2>🗄️ 生データ</h2>
            <details>
                <summary>実験データ詳細（クリックして展開）</summary>
                <pre>{json.dumps(self.improvement_analysis, indent=2, ensure_ascii=False)}</pre>
            </details>

            <hr>
            <footer>
                <p>このレポートは YOLO11 広角カメラ分析システム により自動生成されました。</p>
            </footer>
        </body>
        </html>
        """

        return html_template

    def _create_markdown_template(self) -> str:
        """Markdownテンプレートを作成"""

        improvement_score = self.improvement_analysis.get("improvement_score", 0)
        experiment_type = self._get_experiment_type()

        md_template = f"""# 改善実験レポート

## 実験概要

- **実験タイプ**: {experiment_type}
- **実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **総合改善スコア**: {improvement_score:.3f}

## 📊 実験サマリー

{self._generate_summary_markdown()}

## 🔍 詳細結果

{self._generate_detailed_results_markdown()}

## 📈 改善分析

{self._generate_improvement_analysis_markdown()}

## 💡 推奨事項

{self._generate_recommendations_markdown()}

---

*このレポートは YOLO11 広角カメラ分析システム により自動生成されました。*
"""

        return md_template

    def _get_experiment_type(self) -> str:
        """実験タイプを取得"""
        return self.improvement_analysis.get("experiment_type", "未知の実験")

    def _generate_summary_html(self) -> str:
        """サマリーのHTML部分を生成"""
        try:
            experiment_type = self._get_experiment_type()

            if "tile" in experiment_type.lower():
                return self._generate_tile_summary_html()
            elif "ensemble" in experiment_type.lower():
                return self._generate_ensemble_summary_html()
            elif "calibration" in experiment_type.lower():
                return self._generate_calibration_summary_html()
            else:
                return "<p>実験結果の詳細な分析が利用できません。</p>"

        except Exception as e:
            return f"<p>サマリー生成エラー: {e}</p>"

    def _generate_tile_summary_html(self) -> str:
        """タイル推論実験のサマリー"""
        configurations_tested = self.improvement_analysis.get("configurations_tested", 0)
        best_config = self.improvement_analysis.get("best_configuration", {})

        html = f"""
        <ul>
            <li><strong>テスト設定数:</strong> {configurations_tested}種類</li>
            <li><strong>最適設定:</strong> {best_config.get('tile_config', 'N/A')}</li>
            <li><strong>最大改善率:</strong> {best_config.get('improvement_rate', 0):.1%}</li>
        </ul>
        """

        return html

    def _generate_ensemble_summary_html(self) -> str:
        """アンサンブル実験のサマリー"""
        models_used = self.improvement_analysis.get("models_used", [])
        voting_strategy = self.improvement_analysis.get("voting_strategy", "N/A")

        html = f"""
        <ul>
            <li><strong>使用モデル数:</strong> {len(models_used)}個</li>
            <li><strong>投票戦略:</strong> {voting_strategy}</li>
            <li><strong>モデル一覧:</strong> {', '.join(models_used)}</li>
        </ul>
        """

        return html

    def _generate_calibration_summary_html(self) -> str:
        """キャリブレーション実験のサマリー"""
        calibration_applied = self.improvement_analysis.get("calibration_applied", False)

        html = f"""
        <ul>
            <li><strong>歪み補正適用:</strong> {'はい' if calibration_applied else 'いいえ'}</li>
            <li><strong>精度改善推定:</strong> {self.improvement_analysis.get('improvement_metrics', {}).get('estimated_accuracy_improvement', 0):.1%}</li>
        </ul>
        """

        return html

    def _generate_metrics_html(self) -> str:
        """メトリクスカードのHTML生成"""
        improvement_metrics = self.improvement_analysis.get("improvement_metrics", {})

        metrics_html = ""
        for key, value in improvement_metrics.items():
            metrics_html += f"""
            <div class="metric-card">
                <h3>{self._format_metric_name(key)}</h3>
                <p><strong>{self._format_metric_value(value)}</strong></p>
            </div>
            """

        return metrics_html if metrics_html else "<p>詳細メトリクスが利用できません。</p>"

    def _format_metric_name(self, key: str) -> str:
        """メトリクス名を日本語表示用にフォーマット"""
        name_mapping = {
            "detection_accuracy_improvement": "検出精度改善",
            "false_positive_reduction": "誤検出削減", 
            "processing_time_overhead": "処理時間オーバーヘッド",
            "robustness_improvement": "堅牢性改善",
            "small_object_detection_boost": "小物体検出向上",
            "estimated_accuracy_improvement": "推定精度改善"
        }

        return name_mapping.get(key, key.replace("_", " ").title())

    def _format_metric_value(self, value) -> str:
        """メトリクス値のフォーマット"""
        if isinstance(value, float):
            if abs(value) < 1:
                return f"{value:.1%}"
            else:
                return f"{value:.2f}"
        return str(value)

    def _generate_improvement_analysis_html(self) -> str:
        """改善分析のHTML生成"""
        improvement_score = self.improvement_analysis.get("improvement_score", 0)

        if improvement_score > 0.1:
            analysis_class = "improvement-positive"
            analysis_text = "大幅な改善が確認されました"
        elif improvement_score > 0.05:
            analysis_class = "improvement-positive"
            analysis_text = "改善が確認されました"
        elif improvement_score > 0:
            analysis_class = "improvement-positive"
            analysis_text = "わずかな改善が確認されました"
        else:
            analysis_class = "improvement-negative"
            analysis_text = "改善効果は限定的でした"

        return f'<p class="{analysis_class}">📊 {analysis_text}（スコア: {improvement_score:.3f}）</p>'

    def _generate_summary_markdown(self) -> str:
        """MarkdownサマリーSTEP: 生成"""
        return "実験結果の詳細な分析を実行しました。"

    def _generate_detailed_results_markdown(self) -> str:
        """Markdown詳細結果生成"""
        return f"```json\n{json.dumps(self.improvement_analysis, indent=2, ensure_ascii=False)}\n```"

    def _generate_improvement_analysis_markdown(self) -> str:
        """Markdown改善分析生成"""
        improvement_score = self.improvement_analysis.get("improvement_score", 0)

        if improvement_score > 0.05:
            return "✅ 実験により性能改善が確認されました。"
        else:
            return "⚠️ 改善効果は限定的でした。パラメータの調整を検討してください。"

    def _generate_recommendations_markdown(self) -> str:
        """Markdown推奨事項生成"""
        experiment_type = self._get_experiment_type()

        recommendations = {
            "tile_comparison": [
                "最適なタイルサイズと重複率の組み合わせを本格運用に適用",
                "処理時間とのトレードオフを考慮したパラメータ調整",
                "より多くの動画での検証実験"
            ],
            "ensemble": [
                "最適なモデル組み合わせでの本格運用",
                "投票戦略の更なる最適化",
                "処理時間コストの検討"
            ],
            "calibration": [
                "キャリブレーションパラメータの精密化",
                "様々な環境での検証",
                "歪み補正効果の定量的評価"
            ]
        }

        recs = recommendations.get(experiment_type, ["追加の実験と検証を推奨"])
        return "\n".join([f"- {rec}" for rec in recs])