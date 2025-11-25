"""
BossPig Command-Line Interface

Enhanced CLI with rich output, multiple formats, and filtering.

Created: 2025-11-22
Status: Week 11 Beta Launch - Priority 4
"""

import sys
import json
from pathlib import Path
from typing import Optional, List
import argparse

# Rich terminal formatting (graceful fallback if not installed)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import track
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .detector import BossPigDetector, BossPigFindings, Finding


class BossPigCLI:
    """Enhanced CLI for BossPig with rich formatting and multiple output formats."""

    def __init__(self, use_rich: bool = True):
        """
        Initialize CLI.

        Args:
            use_rich: Whether to use rich formatting (if available)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()

    def print_error(self, message: str):
        """Print error message."""
        if self.use_rich:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"ERROR: {message}", file=sys.stderr)

    def print_success(self, message: str):
        """Print success message."""
        if self.use_rich:
            self.console.print(f"[bold green]Success:[/bold green] {message}")
        else:
            print(f"SUCCESS: {message}")

    def print_info(self, message: str):
        """Print info message."""
        if self.use_rich:
            self.console.print(f"[bold blue]Info:[/bold blue] {message}")
        else:
            print(f"INFO: {message}")

    def format_summary(self, results: BossPigFindings) -> str:
        """
        Format analysis summary with rich formatting.

        Args:
            results: Analysis results

        Returns:
            Formatted summary string
        """
        if not self.use_rich:
            # Fallback to plain text
            return results.summary()

        # Rich formatted output
        metrics = results.quality_metrics
        findings = results.findings

        # Overall score with color based on grade
        grade_colors = {
            "A": "green",
            "B": "green",
            "C": "yellow",
            "D": "red",
            "F": "red"
        }
        color = grade_colors.get(metrics.grade(), "white")

        # Create summary panel
        summary_text = f"""
[bold]Overall Quality Score:[/bold] [{color}]{metrics.grade()}[/{color}] ({metrics.overall_score()}%)

[bold]Quality Breakdown:[/bold]
  Clarity:          {metrics.clarity_score:.1%}
  Specificity:      {metrics.specificity_score:.1%}
  Actionability:    {metrics.actionability_score:.1%}
  Professionalism:  {metrics.professionalism_score:.1%}
  Completeness:     {metrics.completeness_score:.1%}

[bold]Total Findings:[/bold] {len(findings)}
"""

        # Category breakdown
        if findings:
            categories = {}
            for finding in findings:
                cat = finding.category.value
                categories[cat] = categories.get(cat, 0) + 1

            summary_text += "\n[bold]Findings by Category:[/bold]\n"
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                summary_text += f"  {cat}: {count}\n"

        return summary_text

    def display_summary(self, results: BossPigFindings):
        """Display analysis summary."""
        summary = self.format_summary(results)

        if self.use_rich:
            self.console.print(Panel(summary, title="BossPig Analysis Report", border_style="blue"))
        else:
            print("\n" + "=" * 60)
            print("BossPig Analysis Report")
            print("=" * 60)
            print(summary)
            print("=" * 60)

    def display_findings_table(
        self,
        findings: List[Finding],
        severity_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        limit: Optional[int] = None
    ):
        """
        Display findings in a formatted table.

        Args:
            findings: List of findings to display
            severity_filter: Filter by severity (info, warning, error, critical)
            category_filter: Filter by category
            limit: Maximum number of findings to display
        """
        # Apply filters
        filtered = findings

        if severity_filter:
            filtered = [f for f in filtered if f.severity.value == severity_filter.lower()]

        if category_filter:
            filtered = [f for f in filtered if category_filter.lower() in f.category.value.lower()]

        if limit:
            filtered = filtered[:limit]

        if not filtered:
            self.print_info("No findings match the specified filters.")
            return

        if self.use_rich:
            # Rich table
            table = Table(title=f"Findings ({len(filtered)} total)")
            table.add_column("Line", style="cyan", no_wrap=True)
            table.add_column("Severity", style="magenta")
            table.add_column("Category", style="green")
            table.add_column("Issue", style="yellow")
            table.add_column("Suggestion", style="blue")

            # Severity colors
            severity_colors = {
                "info": "blue",
                "warning": "yellow",
                "error": "red",
                "critical": "bold red"
            }

            for finding in filtered:
                severity_color = severity_colors.get(finding.severity.value, "white")
                table.add_row(
                    str(finding.line_number),
                    f"[{severity_color}]{finding.severity.value.upper()}[/{severity_color}]",
                    finding.category.value,
                    finding.message[:50] + "..." if len(finding.message) > 50 else finding.message,
                    finding.suggestion[:50] + "..." if len(finding.suggestion) > 50 else finding.suggestion
                )

            self.console.print(table)
        else:
            # Plain text table
            print("\nFindings:")
            print("-" * 80)
            print(f"{'Line':<6} {'Severity':<10} {'Category':<25} {'Issue'}")
            print("-" * 80)

            for finding in filtered:
                print(f"{finding.line_number:<6} {finding.severity.value.upper():<10} {finding.category.value:<25} {finding.message[:40]}...")

            print("-" * 80)
            print(f"Total: {len(filtered)} findings")

    def export_json(self, results: BossPigFindings, output_path: Path):
        """
        Export results to JSON format.

        Args:
            results: Analysis results
            output_path: Output file path
        """
        data = {
            "quality_metrics": {
                "overall_score": results.quality_metrics.overall_score(),
                "grade": results.quality_metrics.grade(),
                "clarity_score": results.quality_metrics.clarity_score,
                "specificity_score": results.quality_metrics.specificity_score,
                "actionability_score": results.quality_metrics.actionability_score,
                "professionalism_score": results.quality_metrics.professionalism_score,
                "completeness_score": results.quality_metrics.completeness_score
            },
            "findings": [
                {
                    "line_number": f.line_number,
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "text": f.text,
                    "message": f.message,
                    "suggestion": f.suggestion
                }
                for f in results.findings
            ],
            "document_stats": {
                "total_words": results.document_stats.total_words,
                "total_sentences": results.document_stats.total_sentences,
                "total_paragraphs": results.document_stats.total_paragraphs
            }
        }

        output_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self.print_success(f"Results exported to {output_path}")

    def export_html(self, results: BossPigFindings, output_path: Path):
        """
        Export results to HTML format.

        Args:
            results: Analysis results
            output_path: Output file path
        """
        # Generate HTML report
        html = self._generate_html_report(results)
        output_path.write_text(html, encoding='utf-8')
        self.print_success(f"HTML report exported to {output_path}")

    def _generate_html_report(self, results: BossPigFindings) -> str:
        """Generate HTML report (simplified version)."""
        metrics = results.quality_metrics

        # Grade color
        grade_colors = {
            "A": "#28a745",
            "B": "#5cb85c",
            "C": "#ffc107",
            "D": "#fd7e14",
            "F": "#dc3545"
        }
        color = grade_colors.get(metrics.grade(), "#6c757d")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BossPig Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; padding: 20px; background: #f8f9fa; }}
        .score {{ font-size: 48px; font-weight: bold; color: {color}; }}
        .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .metric-name {{ font-weight: bold; color: #6c757d; }}
        .metric-value {{ font-size: 24px; color: #333; }}
        .findings {{ margin-top: 30px; }}
        .finding {{ padding: 10px; margin: 10px 0; border-left: 4px solid #6c757d; background: #f8f9fa; }}
        .finding.critical {{ border-left-color: #dc3545; }}
        .finding.error {{ border-left-color: #fd7e14; }}
        .finding.warning {{ border-left-color: #ffc107; }}
        .finding.info {{ border-left-color: #17a2b8; }}
        .severity {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }}
        .severity.critical {{ background: #dc3545; color: white; }}
        .severity.error {{ background: #fd7e14; color: white; }}
        .severity.warning {{ background: #ffc107; color: black; }}
        .severity.info {{ background: #17a2b8; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BossPig Analysis Report</h1>
        <div class="score">{metrics.grade()}</div>
        <p>Overall Quality: {metrics.overall_score()}%</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-name">Clarity</div>
            <div class="metric-value">{metrics.clarity_score:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-name">Specificity</div>
            <div class="metric-value">{metrics.specificity_score:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-name">Actionability</div>
            <div class="metric-value">{metrics.actionability_score:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-name">Professionalism</div>
            <div class="metric-value">{metrics.professionalism_score:.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-name">Completeness</div>
            <div class="metric-value">{metrics.completeness_score:.1%}</div>
        </div>
    </div>

    <div class="findings">
        <h2>Findings ({len(results.findings)} total)</h2>
"""

        for finding in results.findings:
            html += f"""
        <div class="finding {finding.severity.value}">
            <span class="severity {finding.severity.value}">{finding.severity.value.upper()}</span>
            <strong>Line {finding.line_number}</strong> - {finding.category.value}
            <p><strong>Issue:</strong> {finding.message}</p>
            <p><strong>Suggestion:</strong> {finding.suggestion}</p>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BossPig - Business Writing Quality Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a document
  bosspig analyze document.md

  # Filter by severity
  bosspig analyze document.md --severity error

  # Export to JSON
  bosspig analyze document.md --format json --output report.json

  # Export to HTML
  bosspig analyze document.md --format html --output report.html

  # Limit findings displayed
  bosspig analyze document.md --limit 10
"""
    )

    parser.add_argument(
        "command",
        choices=["analyze", "version"],
        help="Command to run"
    )

    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="File to analyze"
    )

    parser.add_argument(
        "--severity",
        choices=["info", "warning", "error", "critical"],
        help="Filter by severity"
    )

    parser.add_argument(
        "--category",
        help="Filter by category (substring match)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of findings displayed"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (for json/html formats)"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--document-type",
        default="technical_documentation",
        help="Document type (default: technical_documentation)"
    )

    args = parser.parse_args()

    # Handle version command
    if args.command == "version":
        print("BossPig v1.0.0 (Beta)")
        print("Business Writing Quality Analyzer")
        return 0

    # Analyze command requires a file
    if not args.file:
        print("Error: file argument is required for analyze command", file=sys.stderr)
        parser.print_help()
        return 1

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    # Create CLI instance
    cli = BossPigCLI(use_rich=not args.no_color)

    # Create detector
    try:
        detector = BossPigDetector(document_type=args.document_type)
    except Exception as e:
        cli.print_error(f"Failed to initialize detector: {e}")
        return 1

    # Analyze document
    try:
        cli.print_info(f"Analyzing {args.file}...")
        results = detector.analyze(args.file)
    except Exception as e:
        cli.print_error(f"Analysis failed: {e}")
        return 1

    # Display/export results
    if args.format == "text":
        cli.display_summary(results)
        cli.display_findings_table(
            results.findings,
            severity_filter=args.severity,
            category_filter=args.category,
            limit=args.limit
        )

        # Exit code based on findings
        critical_count = sum(1 for f in results.findings if f.severity.value == "critical")
        error_count = sum(1 for f in results.findings if f.severity.value == "error")

        if critical_count > 0:
            return 2  # Critical issues found
        elif error_count > 0:
            return 1  # Errors found
        else:
            return 0  # Success

    elif args.format == "json":
        if not args.output:
            cli.print_error("--output required for JSON format")
            return 1
        cli.export_json(results, args.output)
        return 0

    elif args.format == "html":
        if not args.output:
            cli.print_error("--output required for HTML format")
            return 1
        cli.export_html(results, args.output)
        return 0


if __name__ == "__main__":
    sys.exit(main())
