"""Report generator — automated PDF/HTML performance reports.

Creates comprehensive model evaluation reports combining
metrics, plots, and statistical test results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML performance reports for model evaluation.

    Combines metrics tables, configuration summaries, and
    embedded visualizations into a single report.

    Args:
        output_dir: Directory for generated reports.
        title: Report title.

    Example:
        >>> gen = ReportGenerator(output_dir="reports/")
        >>> gen.add_section("Model Metrics", metrics_df)
        >>> gen.add_section("Configuration", config_dict)
        >>> gen.generate("model_report")
    """

    def __init__(
        self,
        output_dir: str = "reports",
        title: str = "ML Financial Forecaster — Performance Report",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._title = title
        self._sections: List[Dict[str, Any]] = []

    def add_section(
        self,
        title: str,
        content: Any,
        section_type: str = "auto",
    ) -> ReportGenerator:
        """Add a section to the report.

        Args:
            title: Section heading.
            content: Section content (DataFrame, dict, string, or HTML).
            section_type: 'table', 'json', 'text', 'html', or 'auto'.

        Returns:
            Self for method chaining.
        """
        if section_type == "auto":
            if isinstance(content, pd.DataFrame):
                section_type = "table"
            elif isinstance(content, dict):
                section_type = "json"
            else:
                section_type = "text"

        self._sections.append({
            "title": title,
            "content": content,
            "type": section_type,
        })
        return self

    def add_metrics_summary(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Metrics",
    ) -> ReportGenerator:
        """Add a metrics summary card.

        Args:
            metrics: Metric name → value.
            title: Section title.

        Returns:
            Self.
        """
        df = pd.DataFrame(
            [(k, f"{v:.6f}") for k, v in metrics.items()],
            columns=["Metric", "Value"],
        )
        return self.add_section(title, df)

    def add_image(self, path: str, title: str = "") -> ReportGenerator:
        """Add an image to the report.

        Args:
            path: Path to image file.
            title: Caption.

        Returns:
            Self.
        """
        self._sections.append({
            "title": title,
            "content": path,
            "type": "image",
        })
        return self

    def generate(self, filename: str = "report") -> Path:
        """Generate the HTML report.

        Args:
            filename: Output filename (without extension).

        Returns:
            Path to generated report.
        """
        html = self._build_html()
        output_path = self._output_dir / f"{filename}.html"
        output_path.write_text(html, encoding="utf-8")

        logger.info("Generated report: %s (%d sections)", output_path, len(self._sections))
        return output_path

    def _build_html(self) -> str:
        """Build the complete HTML document."""
        sections_html = "\n".join(
            self._render_section(s) for s in self._sections
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5; color: #333; line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        header {{
            background: linear-gradient(135deg, #1a237e, #0d47a1);
            color: white; padding: 2rem; margin-bottom: 2rem; border-radius: 8px;
        }}
        header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        header p {{ opacity: 0.8; font-size: 0.9rem; }}
        .section {{
            background: white; border-radius: 8px; padding: 1.5rem;
            margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            font-size: 1.3rem; color: #1a237e; margin-bottom: 1rem;
            padding-bottom: 0.5rem; border-bottom: 2px solid #e3f2fd;
        }}
        table {{
            width: 100%; border-collapse: collapse; font-size: 0.9rem;
        }}
        th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; color: #555; }}
        tr:hover {{ background: #fafafa; }}
        .json-block {{
            background: #263238; color: #eeffff; padding: 1rem;
            border-radius: 4px; font-family: monospace; font-size: 0.85rem;
            overflow-x: auto; white-space: pre-wrap;
        }}
        .text-block {{ white-space: pre-wrap; font-size: 0.95rem; }}
        img {{ max-width: 100%; border-radius: 4px; margin: 0.5rem 0; }}
        footer {{ text-align: center; padding: 1rem; color: #999; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self._title}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        {sections_html}
        <footer>ML Financial Forecaster — Automated Report</footer>
    </div>
</body>
</html>"""

    def _render_section(self, section: Dict[str, Any]) -> str:
        """Render a single section to HTML."""
        title = section["title"]
        content = section["content"]
        stype = section["type"]

        if stype == "table" and isinstance(content, pd.DataFrame):
            body = content.to_html(index=False, classes="metrics-table", border=0)
        elif stype == "json" and isinstance(content, dict):
            body = f'<div class="json-block">{json.dumps(content, indent=2, default=str)}</div>'
        elif stype == "image":
            body = f'<img src="{content}" alt="{title}">'
        elif stype == "html":
            body = str(content)
        else:
            body = f'<div class="text-block">{content}</div>'

        return f"""
        <div class="section">
            <h2>{title}</h2>
            {body}
        </div>"""
