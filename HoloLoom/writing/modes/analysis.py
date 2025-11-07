"""
Analysis Mode Writer
====================

Generates analytical reports and comparisons from memory context.

Philosophy: Balanced presentation with evidence and conclusions.
"""

import logging
from typing import List, Dict, Any, Optional, Set

from ..core.protocol import (
    ModeWriterProtocol,
    WritingContext,
    WritingMode,
    StyleGuide,
    MemoryShard
)

logger = logging.getLogger(__name__)


class AnalysisWriter:
    """
    Analysis mode writer - transforms memory into analytical reports.

    Approach:
    1. Executive summary
    2. Key findings with evidence
    3. Comparisons (if applicable)
    4. Conclusions and implications
    5. Data-driven insights
    """

    def __init__(self):
        self.logger = logger

    async def generate(self, context: WritingContext) -> str:
        """
        Generate analytical report from context.

        Args:
            context: Writing context with query and memories

        Returns:
            Analytical report draft
        """
        if not context.memories:
            return self._generate_empty_response(context.query)

        # Detect analysis type
        analysis_type = self._detect_analysis_type(context.query)

        # Extract analytical elements
        analysis = self._analyze_content(context.memories, analysis_type)

        # Generate structured report
        report = self._generate_report_structure(
            query=context.query,
            memories=context.memories,
            analysis=analysis,
            analysis_type=analysis_type,
            style=context.style
        )

        return report

    def _detect_analysis_type(self, query: str) -> str:
        """
        Detect type of analysis from query.

        Args:
            query: User query

        Returns:
            Analysis type: comparison, evaluation, summary, breakdown
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['evaluate', 'assess', 'analyze']):
            return 'evaluation'
        elif any(word in query_lower for word in ['summary', 'overview', 'report']):
            return 'summary'
        elif any(word in query_lower for word in ['breakdown', 'components', 'parts']):
            return 'breakdown'
        else:
            return 'general'

    def _analyze_content(
        self,
        memories: List[MemoryShard],
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Analyze memories for analytical content.

        Args:
            memories: Memory shards
            analysis_type: Type of analysis

        Returns:
            Analysis dict with findings, comparisons, etc.
        """
        findings = []
        entities = set()
        comparisons = []
        data_points = []

        for memory in memories:
            text = memory.text
            metadata = memory.metadata

            # Extract key findings (high relevance)
            if metadata.get('relevance', 0.5) > 0.8:
                findings.append({
                    'text': text,
                    'relevance': metadata.get('relevance', 0.5),
                    'source': metadata.get('source', 'unknown')
                })

            # Extract entities
            if 'entities' in metadata:
                entities.update(metadata['entities'])

            # Detect comparisons
            if any(word in text.lower() for word in ['compared to', 'unlike', 'whereas', 'while']):
                comparisons.append(text)

            # Extract data points (numbers, percentages, etc.)
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
            if numbers:
                data_points.extend(numbers[:3])  # Top 3 per memory

        return {
            'findings': findings[:5],  # Top 5 findings
            'entities': list(entities)[:10],
            'comparisons': comparisons[:3],
            'data_points': data_points[:10],
            'memory_count': len(memories)
        }

    def _generate_report_structure(
        self,
        query: str,
        memories: List[MemoryShard],
        analysis: Dict[str, Any],
        analysis_type: str,
        style: StyleGuide
    ) -> str:
        """
        Generate structured analytical report.

        Args:
            query: User query
            memories: Ordered memories
            analysis: Content analysis
            analysis_type: Type of analysis
            style: Style guide

        Returns:
            Analytical report
        """
        sections = []

        # TITLE
        sections.append(f"# {query}\n")

        # EXECUTIVE SUMMARY
        summary = self._generate_executive_summary(query, memories, analysis)
        sections.append(f"## Executive Summary\n\n{summary}\n")

        # KEY FINDINGS
        if analysis['findings']:
            findings = self._generate_findings(analysis['findings'])
            sections.append(f"## Key Findings\n\n{findings}\n")

        # COMPARISON (if applicable)
        if analysis_type == 'comparison' and analysis['comparisons']:
            comparison = self._generate_comparison(analysis['comparisons'], analysis['entities'])
            sections.append(f"## Comparison\n\n{comparison}\n")

        # DETAILED ANALYSIS
        detailed = self._generate_detailed_analysis(memories, analysis)
        sections.append(f"## Analysis\n\n{detailed}\n")

        # DATA INSIGHTS (if data points available)
        if analysis['data_points']:
            insights = self._generate_data_insights(analysis['data_points'])
            sections.append(f"## Data Insights\n\n{insights}\n")

        # CONCLUSIONS
        conclusions = self._generate_conclusions(query, memories, analysis)
        sections.append(f"## Conclusions\n\n{conclusions}\n")

        return '\n'.join(sections)

    def _generate_executive_summary(
        self,
        query: str,
        memories: List[MemoryShard],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate executive summary."""
        # Use top finding or highest relevance memory
        if analysis['findings']:
            summary = analysis['findings'][0]['text']
        elif memories:
            top_memory = max(memories, key=lambda m: m.metadata.get('relevance', 0.5))
            summary = top_memory.text
        else:
            summary = "Insufficient data for analysis."

        # Add entity context if available
        if analysis['entities']:
            entities_str = ', '.join(analysis['entities'][:3])
            summary += f" This analysis focuses on {entities_str}."

        return summary.strip()

    def _generate_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Generate key findings section."""
        parts = []

        for i, finding in enumerate(findings, 1):
            text = finding['text']
            relevance = finding['relevance']
            source = finding['source']

            # Format as numbered finding
            parts.append(f"{i}. **{text}** (confidence: {relevance:.0%}, source: {source})")

        return '\n\n'.join(parts)

    def _generate_comparison(
        self,
        comparisons: List[str],
        entities: List[str]
    ) -> str:
        """Generate comparison section."""
        if not comparisons:
            return "No direct comparisons available."

        # Create comparison table if we have entities
        if len(entities) >= 2:
            parts = [f"Comparing **{entities[0]}** and **{entities[1]}**:\n"]
        else:
            parts = []

        # Add comparison statements
        for comp in comparisons:
            parts.append(f"- {comp}")

        return '\n'.join(parts)

    def _generate_detailed_analysis(
        self,
        memories: List[MemoryShard],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate detailed analysis section."""
        parts = []

        # Group memories by theme/topic if available
        themed_memories = {}
        for memory in memories:
            topic = memory.metadata.get('topic', 'general')
            if topic not in themed_memories:
                themed_memories[topic] = []
            themed_memories[topic].append(memory)

        # Generate analysis by theme
        for topic, topic_memories in themed_memories.items():
            if len(themed_memories) > 1:  # Only add header if multiple topics
                parts.append(f"### {topic.replace('_', ' ').title()}\n")

            for memory in topic_memories:
                parts.append(memory.text.strip())

        return '\n\n'.join(parts)

    def _generate_data_insights(self, data_points: List[str]) -> str:
        """Generate data insights section."""
        parts = ["Key metrics and data points:\n"]

        for point in data_points[:5]:  # Top 5
            parts.append(f"- {point}")

        if len(data_points) > 5:
            parts.append(f"\n_{len(data_points) - 5} additional data points analyzed._")

        return '\n'.join(parts)

    def _generate_conclusions(
        self,
        query: str,
        memories: List[MemoryShard],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate conclusions section."""
        # Synthesize from top findings
        if analysis['findings']:
            top_finding = analysis['findings'][0]['text']
            conclusion = f"Based on the analysis, {top_finding.lower()}"
        elif memories:
            top_memory = max(memories, key=lambda m: m.metadata.get('relevance', 0.5))
            conclusion = top_memory.text
        else:
            conclusion = "Insufficient data to draw conclusions."

        # Add implications if we have entities
        if analysis['entities']:
            conclusion += f" This has implications for {', '.join(analysis['entities'][:2])}."

        return conclusion.strip()

    def _generate_empty_response(self, query: str) -> str:
        """Generate response when no memories available."""
        return (
            f"# {query}\n\n"
            f"## Executive Summary\n\n"
            f"No data available for analysis. Additional information is required "
            f"to generate meaningful insights.\n\n"
            f"## Recommendations\n\n"
            f"- Gather relevant data sources\n"
            f"- Define analysis criteria\n"
            f"- Identify comparison points\n"
        )

    @property
    def mode(self) -> WritingMode:
        """Writing mode this writer handles."""
        return WritingMode.ANALYSIS

    @property
    def default_style(self) -> StyleGuide:
        """Default style guide for analysis mode."""
        return StyleGuide.ACADEMIC
