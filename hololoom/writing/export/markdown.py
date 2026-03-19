"""
Markdown Exporter
=================

Export content to formatted Markdown.
"""

import logging

from ..core.protocol import WritingResult

logger = logging.getLogger(__name__)


class MarkdownExporter:
    """
    Export writing results to Markdown format.

    Features:
    - Standard Markdown formatting
    - Metadata as frontmatter (YAML)
    - Quality metrics as comments
    """

    def __init__(self, include_metadata: bool = True, include_quality: bool = False):
        """
        Initialize Markdown exporter.

        Args:
            include_metadata: Include YAML frontmatter with metadata
            include_quality: Include quality metrics as HTML comments
        """
        self.include_metadata = include_metadata
        self.include_quality = include_quality
        self.logger = logger

    def export(self, result: WritingResult) -> str:
        """
        Export writing result to Markdown.

        Args:
            result: Writing result to export

        Returns:
            Markdown-formatted string
        """
        parts = []

        # Add YAML frontmatter if requested
        if self.include_metadata:
            parts.append(self._generate_frontmatter(result))
            parts.append("")

        # Add quality metrics as HTML comment if requested
        if self.include_quality:
            parts.append(self._generate_quality_comment(result))
            parts.append("")

        # Add main content
        parts.append(result.content)

        return '\n'.join(parts)

    def _generate_frontmatter(self, result: WritingResult) -> str:
        """Generate YAML frontmatter."""
        lines = ["---"]
        lines.append(f"mode: {result.mode.value}")
        lines.append(f"style: {result.style.value}")
        lines.append(f"quality: {result.quality_score:.2f}")
        lines.append(f"confidence: {result.confidence:.2f}")

        if result.refinement_passes:
            lines.append(f"passes: {len(result.refinement_passes)}")

        lines.append("---")

        return '\n'.join(lines)

    def _generate_quality_comment(self, result: WritingResult) -> str:
        """Generate quality metrics as HTML comment."""
        lines = ["<!--"]
        lines.append(f"Quality Score: {result.quality_score:.2f}")
        lines.append(f"Confidence: {result.confidence:.2f}")

        if result.refinement_passes:
            lines.append(f"Refinement Passes: {len(result.refinement_passes)}")
            trajectory = ' → '.join(f"{p.quality_score:.2f}" for p in result.refinement_passes)
            lines.append(f"Quality Trajectory: {trajectory}")

        lines.append("-->")

        return '\n'.join(lines)
