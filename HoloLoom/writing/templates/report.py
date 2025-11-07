"""
Report Template
===============

Templates for structured report generation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import (
    TemplateProtocol,
    TemplateType,
    TemplateContext,
    TemplateResult
)

logger = logging.getLogger(__name__)


class ReportTemplate:
    """
    Report template with customizable structure.

    Supports:
    - Executive, technical, research report types
    - Standard report sections
    - Data-driven insights
    """

    def __init__(
        self,
        report_type: str = 'executive',  # executive, technical, research
        include_toc: bool = False,
        include_executive_summary: bool = True
    ):
        """
        Initialize report template.

        Args:
            report_type: Type of report (executive, technical, research)
            include_toc: Include table of contents
            include_executive_summary: Include executive summary
        """
        self.report_type = report_type
        self.include_toc = include_toc
        self.include_executive_summary = include_executive_summary
        self.logger = logger

    async def generate(self, context: TemplateContext) -> TemplateResult:
        """
        Generate report from template context.

        Args:
            context: Template context with variables and memories

        Returns:
            Template result with report content
        """
        variables = context.variables
        memories = context.memories

        # Extract variables
        title = variables.get('title', 'Report')
        author = variables.get('author', '')
        date = variables.get('date', datetime.now().strftime('%Y-%m-%d'))

        # Build report sections
        sections = {}

        # 1. Title page
        sections['title_page'] = self._generate_title_page(title, author, date)

        # 2. Executive Summary (if enabled)
        if self.include_executive_summary:
            sections['executive_summary'] = self._generate_executive_summary(memories)

        # 3. Table of Contents (if enabled)
        if self.include_toc:
            sections['toc'] = self._generate_toc()

        # 4. Introduction
        sections['introduction'] = self._generate_introduction(title, memories)

        # 5. Main Body (based on report type)
        if self.report_type == 'executive':
            sections['body'] = self._generate_executive_body(memories)
        elif self.report_type == 'technical':
            sections['body'] = self._generate_technical_body(memories)
        else:  # research
            sections['body'] = self._generate_research_body(memories)

        # 6. Conclusions
        sections['conclusions'] = self._generate_conclusions(memories)

        # 7. Recommendations (for executive reports)
        if self.report_type == 'executive':
            sections['recommendations'] = self._generate_recommendations(memories)

        # Assemble report
        report_parts = []

        # Add sections in order
        report_parts.append(sections['title_page'])

        if self.include_executive_summary:
            report_parts.append('\n---\n')
            report_parts.append(sections['executive_summary'])

        if self.include_toc:
            report_parts.append('\n---\n')
            report_parts.append(sections['toc'])

        report_parts.append('\n---\n')
        report_parts.append(sections['introduction'])

        report_parts.append('\n---\n')
        report_parts.append(sections['body'])

        report_parts.append('\n---\n')
        report_parts.append(sections['conclusions'])

        if 'recommendations' in sections:
            report_parts.append('\n---\n')
            report_parts.append(sections['recommendations'])

        content = '\n'.join(report_parts)

        return TemplateResult(
            content=content,
            template_type=TemplateType.REPORT,
            variables_used=list(variables.keys()),
            sections=sections,
            metadata={
                'report_type': self.report_type,
                'page_count': content.count('---') + 1,
                'word_count': len(content.split())
            }
        )

    def _generate_title_page(
        self,
        title: str,
        author: str,
        date: str
    ) -> str:
        """Generate title page."""
        parts = [
            f"# {title}",
            "",
        ]

        if author:
            parts.append(f"**Author:** {author}")

        parts.append(f"**Date:** {date}")
        parts.append(f"**Report Type:** {self.report_type.title()}")

        return '\n'.join(parts)

    def _generate_executive_summary(self, memories: List) -> str:
        """Generate executive summary."""
        parts = ["## Executive Summary", ""]

        # Use top 2 highest-relevance memories
        if memories:
            sorted_memories = sorted(
                memories,
                key=lambda m: m.metadata.get('relevance', 0.5),
                reverse=True
            )

            for memory in sorted_memories[:2]:
                parts.append(memory.text)

        else:
            parts.append("No summary available.")

        return '\n\n'.join(parts)

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        parts = ["## Table of Contents", ""]

        sections = [
            "1. Executive Summary" if self.include_executive_summary else None,
            "2. Introduction",
            "3. Findings" if self.report_type != 'research' else "3. Methodology",
            "4. Analysis",
            "5. Conclusions",
            "6. Recommendations" if self.report_type == 'executive' else None
        ]

        # Filter out None entries
        sections = [s for s in sections if s]

        parts.extend(sections)

        return '\n'.join(parts)

    def _generate_introduction(self, title: str, memories: List) -> str:
        """Generate introduction."""
        parts = ["## 1. Introduction", ""]

        # Context setting
        parts.append(f"This report examines {title.lower()}.")

        # Add scope from memories
        if memories:
            # Use medium-relevance memories for context
            context_memories = [
                m for m in memories
                if 0.6 <= m.metadata.get('relevance', 0.5) <= 0.8
            ]

            if context_memories:
                parts.append("")
                parts.append(context_memories[0].text)

        return '\n\n'.join(parts)

    def _generate_executive_body(self, memories: List) -> str:
        """Generate executive report body."""
        parts = ["## 2. Key Findings", ""]

        # Present findings as numbered list
        for i, memory in enumerate(memories[:5], 1):
            relevance = memory.metadata.get('relevance', 0.5)
            parts.append(f"**Finding {i}:** {memory.text} _(Confidence: {relevance:.0%})_")
            parts.append("")

        return '\n'.join(parts)

    def _generate_technical_body(self, memories: List) -> str:
        """Generate technical report body."""
        parts = ["## 2. Technical Details", ""]

        # Group by topic if available
        topics = {}
        for memory in memories:
            topic = memory.metadata.get('topic', 'General')
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(memory)

        # Present by topic
        for topic, topic_memories in topics.items():
            parts.append(f"### {topic.replace('_', ' ').title()}")
            parts.append("")

            for memory in topic_memories:
                parts.append(memory.text)
                parts.append("")

        return '\n'.join(parts)

    def _generate_research_body(self, memories: List) -> str:
        """Generate research report body."""
        parts = []

        # Methodology section
        parts.append("## 2. Methodology")
        parts.append("")
        parts.append("This research draws from multiple sources to provide comprehensive analysis.")
        parts.append("")

        # Findings section
        parts.append("## 3. Findings")
        parts.append("")

        for memory in memories:
            parts.append(f"- {memory.text}")

        parts.append("")

        # Analysis section
        parts.append("## 4. Analysis")
        parts.append("")

        # Use top memories for analysis
        if memories:
            top_memories = sorted(
                memories,
                key=lambda m: m.metadata.get('relevance', 0.5),
                reverse=True
            )[:3]

            for memory in top_memories:
                parts.append(memory.text)
                parts.append("")

        return '\n'.join(parts)

    def _generate_conclusions(self, memories: List) -> str:
        """Generate conclusions."""
        parts = ["## Conclusions", ""]

        if memories:
            # Use highest-relevance memory for main conclusion
            top_memory = max(
                memories,
                key=lambda m: m.metadata.get('relevance', 0.5)
            )

            parts.append(f"Based on the analysis, {top_memory.text.lower()}")

            # Add supporting conclusion
            if len(memories) > 1:
                parts.append("")
                parts.append("This conclusion is supported by the findings presented in this report.")

        else:
            parts.append("Insufficient data to draw definitive conclusions.")

        return '\n\n'.join(parts)

    def _generate_recommendations(self, memories: List) -> str:
        """Generate recommendations (for executive reports)."""
        parts = ["## Recommendations", ""]

        # Extract actionable items from memories
        recommendations = []

        for memory in memories:
            text = memory.text.lower()
            # Look for action words
            if any(word in text for word in ['should', 'recommend', 'suggest', 'consider']):
                recommendations.append(memory.text)

        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                parts.append(f"{i}. {rec}")
        else:
            # Generic recommendations based on findings
            parts.append("1. Continue monitoring key metrics")
            parts.append("2. Implement findings from this analysis")
            parts.append("3. Review recommendations in next planning cycle")

        return '\n'.join(parts)

    def get_required_variables(self) -> List[str]:
        """Get required template variables."""
        return ['title']

    def get_optional_variables(self) -> List[str]:
        """Get optional template variables."""
        return ['author', 'date']

    @property
    def template_type(self) -> TemplateType:
        """Template type."""
        return TemplateType.REPORT
