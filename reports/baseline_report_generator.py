import json

class BaselineReportGenerator:
    def __init__(self, results, config):
        self.results = results
        self.config = config

    def generate_html_report(self, output_path):
        # 簡易HTMLレポート生成例
        html = "<html><head><title>ベースラインレポート</title></head><body>"
        html += f"<h1>ベースラインレポート</h1><pre>{json.dumps(self.results, indent=2, ensure_ascii=False)}</pre>"
        html += "</body></html>"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def generate_markdown_report(self, output_path):
        # 簡易Markdownレポート生成例
        md = "# ベースラインレポート\n\n"
        md += "```\n" + json.dumps(self.results, indent=2, ensure_ascii=False) + "\n```"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)