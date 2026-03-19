"""
HTML Exporter with Tufte Styling
=================================

Export content to HTML with Tufte-inspired design.

Tufte principles:
- High data-ink ratio
- Margin notes for secondary content
- Minimal decoration
- Clear typography
"""

import logging
import re

from ..core.protocol import WritingResult

logger = logging.getLogger(__name__)


class HTMLExporter:
    """
    Export writing results to HTML with Tufte styling.

    Features:
    - Tufte CSS styling
    - Margin notes
    - Quality metrics sidebar
    - Responsive design
    """

    def __init__(
        self,
        include_quality_sidebar: bool = True,
        include_tufte_css: bool = True
    ):
        """
        Initialize HTML exporter.

        Args:
            include_quality_sidebar: Include quality metrics in sidebar
            include_tufte_css: Include Tufte CSS styling
        """
        self.include_quality_sidebar = include_quality_sidebar
        self.include_tufte_css = include_tufte_css
        self.logger = logger

    def export(self, result: WritingResult) -> str:
        """
        Export writing result to HTML.

        Args:
            result: Writing result to export

        Returns:
            HTML string with Tufte styling
        """
        # Convert markdown-style content to HTML
        html_content = self._markdown_to_html(result.content)

        # Build HTML document
        html = self._generate_html_document(
            content=html_content,
            result=result
        )

        return html

    def _generate_html_document(
        self,
        content: str,
        result: WritingResult
    ) -> str:
        """Generate complete HTML document."""
        parts = []

        # HTML header
        parts.append('<!DOCTYPE html>')
        parts.append('<html lang="en">')
        parts.append('<head>')
        parts.append('    <meta charset="UTF-8">')
        parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        parts.append(f'    <title>{result.mode.value.title()} Content</title>')

        # Add Tufte CSS if requested
        if self.include_tufte_css:
            parts.append(self._generate_tufte_css())

        parts.append('</head>')
        parts.append('<body>')

        # Main container
        parts.append('    <article>')

        # Quality sidebar if requested
        if self.include_quality_sidebar:
            parts.append(self._generate_quality_sidebar(result))

        # Main content
        parts.append('        <section id="content">')
        parts.append(content)
        parts.append('        </section>')

        parts.append('    </article>')
        parts.append('</body>')
        parts.append('</html>')

        return '\n'.join(parts)

    def _generate_tufte_css(self) -> str:
        """Generate Tufte-inspired CSS."""
        return '''
    <style>
        /* Tufte-inspired CSS */
        body {
            font-family: "ET Book", Palatino, "Palatino Linotype", "Palatino LT STD", "Book Antiqua", Georgia, serif;
            font-size: 15px;
            line-height: 2;
            color: #111;
            max-width: 1400px;
            margin: 0 auto;
            padding: 5% 12%;
            background-color: #fffff8;
        }

        article {
            position: relative;
        }

        h1 {
            font-weight: 400;
            margin-top: 4rem;
            margin-bottom: 1.5rem;
            font-size: 3.2rem;
            line-height: 1;
        }

        h2 {
            font-style: italic;
            font-weight: 400;
            margin-top: 2.1rem;
            margin-bottom: 1.4rem;
            font-size: 2.2rem;
            line-height: 1;
        }

        h3 {
            font-style: italic;
            font-weight: 400;
            font-size: 1.7rem;
            margin-top: 2rem;
            margin-bottom: 1.4rem;
            line-height: 1;
        }

        p {
            margin-top: 1.4rem;
            margin-bottom: 1.4rem;
            padding-right: 0;
            vertical-align: baseline;
        }

        /* Quality sidebar */
        .quality-sidebar {
            float: right;
            clear: right;
            margin-right: -60%;
            width: 50%;
            margin-top: 0;
            margin-bottom: 0;
            font-size: 1.1rem;
            line-height: 1.3;
            vertical-align: baseline;
            position: relative;
            padding: 1rem;
            background: #f5f5f0;
            border-left: 3px solid #ccc;
        }

        .quality-sidebar h4 {
            margin-top: 0;
            font-size: 1.2rem;
        }

        .quality-metric {
            margin: 0.5rem 0;
        }

        .quality-bar {
            height: 4px;
            background: #ddd;
            margin-top: 2px;
        }

        .quality-bar-fill {
            height: 100%;
            background: #2c5282;
        }

        /* Code blocks */
        pre, code {
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.9rem;
            line-height: 1.42;
        }

        pre {
            overflow-x: auto;
            padding: 1rem;
            background: #f5f5f0;
            border-left: 3px solid #2c5282;
        }

        /* Links */
        a {
            color: #2c5282;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* Lists */
        ul, ol {
            padding-left: 1.5rem;
        }

        li {
            margin: 0.5rem 0;
        }

        /* Blockquotes */
        blockquote {
            font-style: italic;
            margin-left: 1.5rem;
            padding-left: 1rem;
            border-left: 3px solid #ccc;
        }

        /* Responsive */
        @media (max-width: 760px) {
            .quality-sidebar {
                float: none;
                margin-right: 0;
                width: 100%;
                margin-bottom: 2rem;
            }

            body {
                padding: 5%;
            }
        }
    </style>
'''

    def _generate_quality_sidebar(self, result: WritingResult) -> str:
        """Generate quality metrics sidebar."""
        parts = []

        parts.append('        <aside class="quality-sidebar">')
        parts.append('            <h4>Quality Metrics</h4>')

        # Overall quality
        parts.append('            <div class="quality-metric">')
        parts.append(f'                <strong>Overall Quality:</strong> {result.quality_score:.0%}')
        parts.append('                <div class="quality-bar">')
        parts.append(f'                    <div class="quality-bar-fill" style="width: {result.quality_score * 100}%"></div>')
        parts.append('                </div>')
        parts.append('            </div>')

        # Confidence
        parts.append('            <div class="quality-metric">')
        parts.append(f'                <strong>Confidence:</strong> {result.confidence:.0%}')
        parts.append('                <div class="quality-bar">')
        parts.append(f'                    <div class="quality-bar-fill" style="width: {result.confidence * 100}%"></div>')
        parts.append('                </div>')
        parts.append('            </div>')

        # Refinement passes
        if result.refinement_passes:
            parts.append('            <div class="quality-metric">')
            parts.append(f'                <strong>Refinement:</strong> {len(result.refinement_passes)} passes')
            trajectory = ' → '.join(f"{p.quality_score:.0%}" for p in result.refinement_passes)
            parts.append(f'                <div style="font-size: 0.9rem; color: #666; margin-top: 0.25rem;">{trajectory}</div>')
            parts.append('            </div>')

        # Mode and style
        parts.append('            <div class="quality-metric" style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #ddd;">')
        parts.append(f'                <div><strong>Mode:</strong> {result.mode.value}</div>')
        parts.append(f'                <div><strong>Style:</strong> {result.style.value}</div>')
        parts.append('            </div>')

        parts.append('        </aside>')

        return '\n'.join(parts)

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert basic Markdown to HTML.

        Simple converter for common Markdown patterns.
        """
        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Code blocks
        html = re.sub(r'```(\w+)?\n(.*?)\n```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)

        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Bold
        html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)

        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)

        # Lists (simple)
        lines = html.split('\n')
        in_list = False
        result_lines = []

        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                result_lines.append(f'<li>{line.strip()[2:]}</li>')
            elif line.strip().startswith(tuple(f'{i}. ' for i in range(10))):
                if not in_list:
                    result_lines.append('<ol>')
                    in_list = True
                result_lines.append(f'<li>{line.strip().split(". ", 1)[1]}</li>')
            else:
                if in_list:
                    # Check if previous list was ul or ol
                    if '<ul>' in result_lines[-10:]:
                        result_lines.append('</ul>')
                    else:
                        result_lines.append('</ol>')
                    in_list = False

                # Paragraphs
                if line.strip() and not line.strip().startswith('<'):
                    result_lines.append(f'<p>{line}</p>')
                else:
                    result_lines.append(line)

        if in_list:
            result_lines.append('</ul>')

        return '\n'.join(result_lines)
