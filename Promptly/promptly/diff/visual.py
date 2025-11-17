#!/usr/bin/env python3
"""
Visual Diff - Terminal colors and HTML diff reports
"""

from typing import Optional, Dict, List
from .engine import DiffResult, DiffChunk, DiffType, DiffStats
from .compare import VersionComparison, BranchComparison
import html


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class TerminalDiff:
    """Render diffs with terminal colors"""

    @staticmethod
    def render(diff_result: DiffResult, show_stats: bool = True) -> str:
        """
        Render diff result with ANSI colors

        Args:
            diff_result: DiffResult to render
            show_stats: Whether to show statistics

        Returns:
            Colored terminal output
        """
        lines = []

        if show_stats:
            lines.append(TerminalDiff._render_stats(diff_result.stats))
            lines.append("")

        # Render chunks
        for chunk in diff_result.chunks:
            if chunk.type == DiffType.EQUAL:
                # Show abbreviated context
                context_lines = chunk.old_text.splitlines()
                if len(context_lines) > 4:
                    lines.append(f"{Colors.DIM}  {context_lines[0]}{Colors.RESET}")
                    lines.append(f"{Colors.DIM}  ...{Colors.RESET}")
                    lines.append(f"{Colors.DIM}  {context_lines[-1]}{Colors.RESET}")
                else:
                    for line in context_lines:
                        lines.append(f"{Colors.DIM}  {line}{Colors.RESET}")

            elif chunk.type == DiffType.DELETE:
                for line in chunk.old_text.splitlines():
                    lines.append(f"{Colors.RED}- {line}{Colors.RESET}")

            elif chunk.type == DiffType.INSERT:
                for line in chunk.new_text.splitlines():
                    lines.append(f"{Colors.GREEN}+ {line}{Colors.RESET}")

            elif chunk.type == DiffType.REPLACE:
                for line in chunk.old_text.splitlines():
                    lines.append(f"{Colors.RED}- {line}{Colors.RESET}")
                for line in chunk.new_text.splitlines():
                    lines.append(f"{Colors.GREEN}+ {line}{Colors.RESET}")

        return '\n'.join(lines)

    @staticmethod
    def render_inline(diff_result: DiffResult) -> str:
        """
        Render diff with inline highlighting (word-level changes highlighted)

        Args:
            diff_result: DiffResult to render

        Returns:
            Colored inline diff
        """
        lines = []

        for chunk in diff_result.chunks:
            if chunk.type == DiffType.EQUAL:
                lines.append(f"{Colors.DIM}{chunk.old_text}{Colors.RESET}")

            elif chunk.type == DiffType.DELETE:
                lines.append(f"{Colors.BG_RED}{Colors.WHITE}{chunk.old_text}{Colors.RESET}")

            elif chunk.type == DiffType.INSERT:
                lines.append(f"{Colors.BG_GREEN}{Colors.WHITE}{chunk.new_text}{Colors.RESET}")

            elif chunk.type == DiffType.REPLACE:
                lines.append(f"{Colors.BG_RED}{Colors.WHITE}{chunk.old_text}{Colors.RESET}")
                lines.append(" -> ")
                lines.append(f"{Colors.BG_GREEN}{Colors.WHITE}{chunk.new_text}{Colors.RESET}")

        return ''.join(lines)

    @staticmethod
    def render_side_by_side(
        old_text: str,
        new_text: str,
        width: int = 80,
        colored: bool = True
    ) -> str:
        """
        Render side-by-side diff with colors

        Args:
            old_text: Original text
            new_text: Modified text
            width: Total width
            colored: Use ANSI colors

        Returns:
            Side-by-side colored diff
        """
        from .engine import DiffEngine

        engine = DiffEngine()
        diff_output = engine.side_by_side(old_text, new_text, width)

        if not colored:
            return diff_output

        # Add colors to side-by-side output
        lines = diff_output.splitlines()
        colored_lines = []

        for line in lines:
            if line.startswith('- '):
                colored_lines.append(f"{Colors.RED}{line}{Colors.RESET}")
            elif line.startswith('+ '):
                colored_lines.append(f"{Colors.GREEN}{line}{Colors.RESET}")
            elif line.startswith('  '):
                colored_lines.append(f"{Colors.DIM}{line}{Colors.RESET}")
            else:
                colored_lines.append(line)

        return '\n'.join(colored_lines)

    @staticmethod
    def _render_stats(stats: DiffStats) -> str:
        """Render statistics with colors"""
        parts = []

        if stats.additions > 0:
            parts.append(f"{Colors.GREEN}+{stats.additions}{Colors.RESET}")

        if stats.deletions > 0:
            parts.append(f"{Colors.RED}-{stats.deletions}{Colors.RESET}")

        if stats.changes > 0:
            parts.append(f"{Colors.YELLOW}~{stats.changes}{Colors.RESET}")

        similarity_color = Colors.GREEN if stats.similarity > 0.8 else Colors.YELLOW if stats.similarity > 0.5 else Colors.RED
        parts.append(f"{similarity_color}{stats.similarity:.1%} similar{Colors.RESET}")

        return f"{Colors.BOLD}Stats:{Colors.RESET} " + ", ".join(parts)

    @staticmethod
    def render_comparison(comparison: VersionComparison) -> str:
        """Render version comparison with colors"""
        lines = []

        # Header
        lines.append(f"{Colors.BOLD}{Colors.CYAN}Prompt Comparison: {comparison.name}{Colors.RESET}")
        lines.append(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        lines.append(f"Versions: v{comparison.old_version} -> v{comparison.new_version}")
        lines.append(f"Commits:  {comparison.old_commit} -> {comparison.new_commit}")

        if comparison.time_delta:
            lines.append(f"Time:     {comparison.time_delta}")

        lines.append("")

        # Stats
        lines.append(TerminalDiff._render_stats(comparison.diff_result.stats))
        lines.append("")

        # Metadata changes
        if comparison.metadata_diff:
            meta_diff = comparison.metadata_diff
            if meta_diff['added']:
                lines.append(f"{Colors.GREEN}Metadata added:{Colors.RESET}")
                for key, value in meta_diff['added'].items():
                    lines.append(f"  + {key}: {value}")

            if meta_diff['removed']:
                lines.append(f"{Colors.RED}Metadata removed:{Colors.RESET}")
                for key, value in meta_diff['removed'].items():
                    lines.append(f"  - {key}: {value}")

            if meta_diff['changed']:
                lines.append(f"{Colors.YELLOW}Metadata changed:{Colors.RESET}")
                for key, change in meta_diff['changed'].items():
                    lines.append(f"  ~ {key}: {change['old']} -> {change['new']}")

            lines.append("")

        # Content diff
        lines.append(f"{Colors.BOLD}Content changes:{Colors.RESET}")
        lines.append(TerminalDiff.render(comparison.diff_result, show_stats=False))

        return '\n'.join(lines)

    @staticmethod
    def render_branch_comparison(comparison: BranchComparison) -> str:
        """Render branch comparison with colors"""
        lines = []

        lines.append(f"{Colors.BOLD}{Colors.CYAN}Branch Comparison{Colors.RESET}")
        lines.append(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        lines.append(f"{comparison.branch_old} <-> {comparison.branch_new}")
        lines.append("")

        # Summary
        lines.append(f"{Colors.BOLD}Summary:{Colors.RESET}")
        lines.append(f"  {Colors.RED}Only in {comparison.branch_old}:{Colors.RESET} {len(comparison.prompts_only_in_old)}")
        lines.append(f"  {Colors.GREEN}Only in {comparison.branch_new}:{Colors.RESET} {len(comparison.prompts_only_in_new)}")
        lines.append(f"  {Colors.YELLOW}Modified:{Colors.RESET} {len(comparison.prompts_modified)}")
        lines.append(f"  {Colors.DIM}Identical:{Colors.RESET} {len(comparison.prompts_identical)}")
        lines.append("")

        # Details
        if comparison.prompts_only_in_old:
            lines.append(f"{Colors.RED}Prompts only in {comparison.branch_old}:{Colors.RESET}")
            for name in comparison.prompts_only_in_old[:10]:
                lines.append(f"  - {name}")
            if len(comparison.prompts_only_in_old) > 10:
                lines.append(f"  {Colors.DIM}... and {len(comparison.prompts_only_in_old) - 10} more{Colors.RESET}")
            lines.append("")

        if comparison.prompts_only_in_new:
            lines.append(f"{Colors.GREEN}Prompts only in {comparison.branch_new}:{Colors.RESET}")
            for name in comparison.prompts_only_in_new[:10]:
                lines.append(f"  + {name}")
            if len(comparison.prompts_only_in_new) > 10:
                lines.append(f"  {Colors.DIM}... and {len(comparison.prompts_only_in_new) - 10} more{Colors.RESET}")
            lines.append("")

        if comparison.prompts_modified:
            lines.append(f"{Colors.YELLOW}Modified prompts:{Colors.RESET}")
            for name in comparison.prompts_modified[:15]:
                if name in comparison.diff_results:
                    stats = comparison.diff_results[name].stats
                    similarity_color = Colors.GREEN if stats.similarity > 0.8 else Colors.YELLOW if stats.similarity > 0.5 else Colors.RED
                    lines.append(
                        f"  ~ {name}: "
                        f"{Colors.GREEN}+{stats.additions}{Colors.RESET} "
                        f"{Colors.RED}-{stats.deletions}{Colors.RESET} "
                        f"{similarity_color}{stats.similarity:.0%}{Colors.RESET}"
                    )
            if len(comparison.prompts_modified) > 15:
                lines.append(f"  {Colors.DIM}... and {len(comparison.prompts_modified) - 15} more{Colors.RESET}")

        return '\n'.join(lines)


class HTMLDiff:
    """Generate HTML diff reports"""

    CSS = """
    <style>
        body {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .diff-container {
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 20px;
            margin: 20px 0;
        }
        .diff-header {
            background: #f0f0f0;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 3px;
        }
        .diff-stats {
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-left: 3px solid #666;
        }
        .diff-line {
            margin: 0;
            padding: 2px 5px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .diff-delete {
            background-color: #ffebee;
            color: #c62828;
        }
        .diff-insert {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .diff-context {
            background-color: #fafafa;
            color: #666;
        }
        .diff-replace-old {
            background-color: #ffcdd2;
            color: #b71c1c;
            text-decoration: line-through;
        }
        .diff-replace-new {
            background-color: #c8e6c9;
            color: #1b5e20;
        }
        .stats-addition {
            color: #2e7d32;
            font-weight: bold;
        }
        .stats-deletion {
            color: #c62828;
            font-weight: bold;
        }
        .stats-change {
            color: #f57c00;
            font-weight: bold;
        }
        h1, h2, h3 {
            color: #333;
        }
        .metadata-diff {
            margin: 15px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 3px;
        }
        .side-by-side {
            display: flex;
            gap: 20px;
        }
        .side-by-side .column {
            flex: 1;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .side-by-side .header {
            background: #e0e0e0;
            padding: 5px 10px;
            font-weight: bold;
        }
    </style>
    """

    @staticmethod
    def render(diff_result: DiffResult, title: str = "Diff Report") -> str:
        """
        Generate HTML diff report

        Args:
            diff_result: DiffResult to render
            title: Report title

        Returns:
            Complete HTML document
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{html.escape(title)}</title>",
            HTMLDiff.CSS,
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
        ]

        # Stats
        html_parts.append("<div class='diff-stats'>")
        html_parts.append(HTMLDiff._render_stats_html(diff_result.stats))
        html_parts.append("</div>")

        # Diff content
        html_parts.append("<div class='diff-container'>")

        for chunk in diff_result.chunks:
            if chunk.type == DiffType.EQUAL:
                # Show limited context
                lines = chunk.old_text.splitlines()
                if len(lines) > 6:
                    html_parts.append(f"<div class='diff-line diff-context'>{html.escape(lines[0])}</div>")
                    html_parts.append(f"<div class='diff-line diff-context'>...</div>")
                    html_parts.append(f"<div class='diff-line diff-context'>{html.escape(lines[-1])}</div>")
                else:
                    for line in lines:
                        html_parts.append(f"<div class='diff-line diff-context'>{html.escape(line)}</div>")

            elif chunk.type == DiffType.DELETE:
                for line in chunk.old_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-delete'>- {html.escape(line)}</div>")

            elif chunk.type == DiffType.INSERT:
                for line in chunk.new_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-insert'>+ {html.escape(line)}</div>")

            elif chunk.type == DiffType.REPLACE:
                for line in chunk.old_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-replace-old'>- {html.escape(line)}</div>")
                for line in chunk.new_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-replace-new'>+ {html.escape(line)}</div>")

        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return '\n'.join(html_parts)

    @staticmethod
    def render_comparison(comparison: VersionComparison) -> str:
        """Generate HTML for version comparison"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Comparison: {html.escape(comparison.name)}</title>",
            HTMLDiff.CSS,
            "</head>",
            "<body>",
            f"<h1>Prompt Comparison: {html.escape(comparison.name)}</h1>",
        ]

        # Header info
        html_parts.append("<div class='diff-header'>")
        html_parts.append(f"<p><strong>Versions:</strong> v{comparison.old_version} → v{comparison.new_version}</p>")
        html_parts.append(f"<p><strong>Commits:</strong> {comparison.old_commit} → {comparison.new_commit}</p>")
        if comparison.time_delta:
            html_parts.append(f"<p><strong>Time delta:</strong> {comparison.time_delta}</p>")
        html_parts.append("</div>")

        # Stats
        html_parts.append("<div class='diff-stats'>")
        html_parts.append("<h3>Statistics</h3>")
        html_parts.append(HTMLDiff._render_stats_html(comparison.diff_result.stats))
        html_parts.append("</div>")

        # Metadata changes
        if comparison.metadata_diff:
            html_parts.append(HTMLDiff._render_metadata_diff_html(comparison.metadata_diff))

        # Content diff
        html_parts.append("<h2>Content Changes</h2>")
        html_parts.append("<div class='diff-container'>")

        for chunk in comparison.diff_result.chunks:
            if chunk.type == DiffType.DELETE:
                for line in chunk.old_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-delete'>- {html.escape(line)}</div>")
            elif chunk.type == DiffType.INSERT:
                for line in chunk.new_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-insert'>+ {html.escape(line)}</div>")
            elif chunk.type == DiffType.REPLACE:
                for line in chunk.old_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-replace-old'>- {html.escape(line)}</div>")
                for line in chunk.new_text.splitlines():
                    html_parts.append(f"<div class='diff-line diff-replace-new'>+ {html.escape(line)}</div>")

        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return '\n'.join(html_parts)

    @staticmethod
    def render_side_by_side(old_text: str, new_text: str, old_label: str = "Old", new_label: str = "New") -> str:
        """Generate side-by-side HTML diff"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Side-by-Side Diff</title>",
            HTMLDiff.CSS,
            "</head>",
            "<body>",
            "<h1>Side-by-Side Comparison</h1>",
            "<div class='side-by-side'>",
        ]

        # Old version
        html_parts.append("<div class='column'>")
        html_parts.append(f"<div class='header'>{html.escape(old_label)}</div>")
        html_parts.append("<div class='diff-container'>")
        for line in old_text.splitlines():
            html_parts.append(f"<div class='diff-line'>{html.escape(line)}</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

        # New version
        html_parts.append("<div class='column'>")
        html_parts.append(f"<div class='header'>{html.escape(new_label)}</div>")
        html_parts.append("<div class='diff-container'>")
        for line in new_text.splitlines():
            html_parts.append(f"<div class='diff-line'>{html.escape(line)}</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return '\n'.join(html_parts)

    @staticmethod
    def _render_stats_html(stats: DiffStats) -> str:
        """Render statistics as HTML"""
        return (
            f"<p>"
            f"<span class='stats-addition'>+{stats.additions}</span> additions, "
            f"<span class='stats-deletion'>-{stats.deletions}</span> deletions, "
            f"<span class='stats-change'>~{stats.changes}</span> changes | "
            f"<strong>{stats.similarity:.1%}</strong> similar"
            f"</p>"
        )

    @staticmethod
    def _render_metadata_diff_html(metadata_diff: Dict) -> str:
        """Render metadata diff as HTML"""
        html_parts = ["<div class='metadata-diff'>", "<h3>Metadata Changes</h3>"]

        if metadata_diff.get('added'):
            html_parts.append("<h4>Added:</h4><ul>")
            for key, value in metadata_diff['added'].items():
                html_parts.append(f"<li class='stats-addition'>{html.escape(key)}: {html.escape(str(value))}</li>")
            html_parts.append("</ul>")

        if metadata_diff.get('removed'):
            html_parts.append("<h4>Removed:</h4><ul>")
            for key, value in metadata_diff['removed'].items():
                html_parts.append(f"<li class='stats-deletion'>{html.escape(key)}: {html.escape(str(value))}</li>")
            html_parts.append("</ul>")

        if metadata_diff.get('changed'):
            html_parts.append("<h4>Changed:</h4><ul>")
            for key, change in metadata_diff['changed'].items():
                html_parts.append(
                    f"<li class='stats-change'>{html.escape(key)}: "
                    f"{html.escape(str(change['old']))} → {html.escape(str(change['new']))}</li>"
                )
            html_parts.append("</ul>")

        html_parts.append("</div>")
        return '\n'.join(html_parts)


def render_terminal(diff_result: DiffResult) -> str:
    """Quick terminal rendering"""
    return TerminalDiff.render(diff_result)


def render_html(diff_result: DiffResult, title: str = "Diff Report") -> str:
    """Quick HTML rendering"""
    return HTMLDiff.render(diff_result, title)
