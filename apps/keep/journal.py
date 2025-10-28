"""
Narrative journal integration for Keep.

Provides rich narrative tracking and synthesis for beekeeping activities,
enabling temporal storytelling and pattern recognition across journal entries.

Inspired by food_e journal patterns.
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from apps.keep.models import Hive, Colony, Inspection
from apps.keep.apiary import Apiary


class EntryType(str, Enum):
    """Types of journal entries."""
    OBSERVATION = "observation"
    DECISION = "decision"
    REFLECTION = "reflection"
    MILESTONE = "milestone"
    CONCERN = "concern"
    CELEBRATION = "celebration"


class Sentiment(str, Enum):
    """Sentiment of journal entries."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    WORRIED = "worried"


@dataclass
class JournalEntry:
    """
    A narrative journal entry about beekeeping activities.

    Attributes:
        entry_id: Unique identifier
        timestamp: When entry was created
        entry_type: Type of entry
        content: Natural language content
        sentiment: Emotional tone
        hive_ids: Related hives
        tags: Categorical tags
        metadata: Additional structured data
    """
    entry_id: str
    timestamp: datetime
    entry_type: EntryType
    content: str
    sentiment: Sentiment = Sentiment.NEUTRAL
    hive_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"JournalEntry({self.entry_type.value}, {self.timestamp.date()})"


@dataclass
class NarrativeSynthesis:
    """
    Synthesized narrative from multiple journal entries.

    Attributes:
        timeframe: Time range covered
        summary: Narrative summary
        key_themes: Identified themes
        sentiment_arc: How sentiment changed over time
        highlights: Notable moments
        concerns: Identified concerns
    """
    timeframe: tuple[datetime, datetime]
    summary: str
    key_themes: List[str]
    sentiment_arc: List[tuple[datetime, Sentiment]]
    highlights: List[str]
    concerns: List[str]


class BeekeepingJournal:
    """
    Narrative journal for beekeeping with temporal synthesis.

    Provides rich storytelling capabilities for tracking the apiary journey,
    enabling reflection, pattern recognition, and knowledge accumulation.
    """

    def __init__(self, apiary: Apiary):
        """
        Initialize journal.

        Args:
            apiary: Associated apiary
        """
        self.apiary = apiary
        self.entries: List[JournalEntry] = []

    def record(
        self,
        content: str,
        entry_type: EntryType = EntryType.OBSERVATION,
        sentiment: Sentiment = Sentiment.NEUTRAL,
        hive_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> JournalEntry:
        """
        Record a journal entry.

        Args:
            content: Natural language entry
            entry_type: Type of entry
            sentiment: Emotional tone
            hive_ids: Related hives
            tags: Categorical tags
            **metadata: Additional data

        Returns:
            Created journal entry

        Example:
            journal.record(
                "First honey harvest from Hive 001! 45 lbs of beautiful amber honey. "
                "The bees have been thriving since the spring buildup.",
                entry_type=EntryType.CELEBRATION,
                sentiment=Sentiment.POSITIVE,
                hive_ids=[hive1.hive_id],
                tags=["harvest", "milestone"],
                quantity_lbs=45.0
            )
        """
        from uuid import uuid4

        entry = JournalEntry(
            entry_id=str(uuid4()),
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content,
            sentiment=sentiment,
            hive_ids=hive_ids or [],
            tags=tags or [],
            metadata=metadata,
        )

        self.entries.append(entry)
        return entry

    def observe(self, content: str, **kwargs) -> JournalEntry:
        """Record an observation."""
        return self.record(content, entry_type=EntryType.OBSERVATION, **kwargs)

    def decide(self, content: str, **kwargs) -> JournalEntry:
        """Record a decision."""
        return self.record(content, entry_type=EntryType.DECISION, **kwargs)

    def reflect(self, content: str, **kwargs) -> JournalEntry:
        """Record a reflection."""
        return self.record(content, entry_type=EntryType.REFLECTION, **kwargs)

    def celebrate(self, content: str, **kwargs) -> JournalEntry:
        """Record a celebration or milestone."""
        kwargs["sentiment"] = Sentiment.POSITIVE
        return self.record(content, entry_type=EntryType.CELEBRATION, **kwargs)

    def concern(self, content: str, **kwargs) -> JournalEntry:
        """Record a concern."""
        kwargs["sentiment"] = kwargs.get("sentiment", Sentiment.CONCERNED)
        return self.record(content, entry_type=EntryType.CONCERN, **kwargs)

    def get_entries(
        self,
        hive_id: Optional[str] = None,
        entry_type: Optional[EntryType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
    ) -> List[JournalEntry]:
        """
        Query journal entries with filters.

        Args:
            hive_id: Filter by hive
            entry_type: Filter by entry type
            since: Start date
            until: End date
            tags: Filter by tags

        Returns:
            Filtered list of entries
        """
        filtered = self.entries

        if hive_id:
            filtered = [e for e in filtered if hive_id in e.hive_ids]

        if entry_type:
            filtered = [e for e in filtered if e.entry_type == entry_type]

        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        if until:
            filtered = [e for e in filtered if e.timestamp <= until]

        if tags:
            filtered = [
                e for e in filtered
                if any(tag in e.tags for tag in tags)
            ]

        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)

    def synthesize_narrative(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        hive_id: Optional[str] = None,
    ) -> NarrativeSynthesis:
        """
        Synthesize narrative from journal entries.

        Creates a coherent story from individual entries, identifying
        themes, sentiment arcs, and key moments.

        Args:
            since: Start of timeframe
            until: End of timeframe
            hive_id: Focus on specific hive

        Returns:
            Narrative synthesis
        """
        # Default to last 30 days
        if not since:
            since = datetime.now() - timedelta(days=30)
        if not until:
            until = datetime.now()

        entries = self.get_entries(since=since, until=until, hive_id=hive_id)

        if not entries:
            return NarrativeSynthesis(
                timeframe=(since, until),
                summary="No journal entries found for this period.",
                key_themes=[],
                sentiment_arc=[],
                highlights=[],
                concerns=[],
            )

        # Extract themes from tags
        tag_counts = defaultdict(int)
        for entry in entries:
            for tag in entry.tags:
                tag_counts[tag] += 1

        key_themes = [
            tag for tag, count in
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Build sentiment arc
        sentiment_arc = [(e.timestamp, e.sentiment) for e in entries]

        # Extract highlights (celebrations and milestones)
        highlights = [
            e.content for e in entries
            if e.entry_type in [EntryType.CELEBRATION, EntryType.MILESTONE]
        ]

        # Extract concerns
        concerns = [
            e.content for e in entries
            if e.entry_type == EntryType.CONCERN or e.sentiment == Sentiment.WORRIED
        ]

        # Generate summary
        summary = self._generate_summary(entries, since, until, hive_id)

        return NarrativeSynthesis(
            timeframe=(since, until),
            summary=summary,
            key_themes=key_themes,
            sentiment_arc=sentiment_arc,
            highlights=highlights,
            concerns=concerns,
        )

    def _generate_summary(
        self,
        entries: List[JournalEntry],
        since: datetime,
        until: datetime,
        hive_id: Optional[str]
    ) -> str:
        """Generate narrative summary from entries."""
        if not entries:
            return "No activity recorded."

        # Count by type
        type_counts = defaultdict(int)
        for entry in entries:
            type_counts[entry.entry_type] += 1

        # Build summary
        hive_context = ""
        if hive_id:
            hive = self.apiary.hives.get(hive_id)
            hive_context = f" for {hive.name}" if hive else ""

        period = (until - since).days
        parts = [
            f"Over the past {period} days{hive_context}, ",
            f"{len(entries)} journal entries were recorded. "
        ]

        # Describe activity
        if type_counts[EntryType.OBSERVATION] > 0:
            parts.append(f"{type_counts[EntryType.OBSERVATION]} observations were made. ")

        if type_counts[EntryType.DECISION] > 0:
            parts.append(f"{type_counts[EntryType.DECISION]} decisions were documented. ")

        # Sentiment summary
        sentiments = [e.sentiment for e in entries]
        positive_count = sentiments.count(Sentiment.POSITIVE)
        concerned_count = sentiments.count(Sentiment.CONCERNED) + sentiments.count(Sentiment.WORRIED)

        if positive_count > concerned_count:
            parts.append("Overall sentiment was positive. ")
        elif concerned_count > 0:
            parts.append(f"{concerned_count} concerns were noted. ")

        # Highlights
        if type_counts[EntryType.CELEBRATION] > 0:
            parts.append(f"{type_counts[EntryType.CELEBRATION]} milestones were celebrated. ")

        return "".join(parts)

    def get_timeline(
        self,
        days: int = 30,
        hive_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of events.

        Args:
            days: Number of days to include
            hive_id: Filter by hive

        Returns:
            Timeline with entries and inspections merged
        """
        since = datetime.now() - timedelta(days=days)

        # Get journal entries
        entries = self.get_entries(since=since, hive_id=hive_id)

        # Get inspections
        inspections = [
            i for i in self.apiary.inspections
            if i.timestamp >= since
        ]
        if hive_id:
            inspections = [i for i in inspections if i.hive_id == hive_id]

        # Merge into timeline
        timeline = []

        for entry in entries:
            timeline.append({
                "timestamp": entry.timestamp,
                "type": "journal",
                "entry_type": entry.entry_type.value,
                "content": entry.content,
                "sentiment": entry.sentiment.value,
            })

        for inspection in inspections:
            hive = self.apiary.hives.get(inspection.hive_id)
            timeline.append({
                "timestamp": inspection.timestamp,
                "type": "inspection",
                "hive_name": hive.name if hive else "Unknown",
                "inspection_type": inspection.inspection_type.value,
                "summary": f"Inspected {hive.name if hive else 'hive'} - {inspection.inspection_type.value}",
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"], reverse=True)

        return timeline

    async def extract_insights(self) -> Dict[str, Any]:
        """
        Extract insights from journal entries using pattern recognition.

        Returns:
            Dict with identified patterns and insights
        """
        entries = self.entries

        if not entries:
            return {"insights": [], "patterns": [], "recommendations": []}

        insights = []
        patterns = []
        recommendations = []

        # Pattern: Repeated concerns
        concern_entries = [e for e in entries if e.entry_type == EntryType.CONCERN]
        if len(concern_entries) >= 3:
            recent_concerns = concern_entries[-3:]
            common_tags = set.intersection(*[set(e.tags) for e in recent_concerns])
            if common_tags:
                patterns.append(f"Recurring concern about: {', '.join(common_tags)}")
                recommendations.append("Consider developing systematic approach to recurring concerns")

        # Pattern: Sentiment trends
        recent = entries[-10:] if len(entries) >= 10 else entries
        positive_ratio = len([e for e in recent if e.sentiment == Sentiment.POSITIVE]) / len(recent)

        if positive_ratio > 0.7:
            insights.append("Recent trend is very positive - apiary is thriving")
        elif positive_ratio < 0.3:
            insights.append("Recent sentiment is concerning - may need intervention")

        # Pattern: Celebration frequency
        celebrations = [e for e in entries if e.entry_type == EntryType.CELEBRATION]
        if celebrations:
            avg_days_between = (entries[-1].timestamp - entries[0].timestamp).days / max(len(celebrations), 1)
            insights.append(f"Milestones celebrated approximately every {avg_days_between:.0f} days")

        return {
            "insights": insights,
            "patterns": patterns,
            "recommendations": recommendations,
        }

    def __repr__(self) -> str:
        return f"BeekeepingJournal(entries={len(self.entries)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_journal(apiary: Apiary) -> BeekeepingJournal:
    """Create a journal for an apiary."""
    return BeekeepingJournal(apiary)


def from_inspection_to_entry(
    inspection: Inspection,
    journal: BeekeepingJournal
) -> JournalEntry:
    """
    Convert an inspection to a journal entry.

    Args:
        inspection: Inspection to convert
        journal: Journal to add entry to

    Returns:
        Created journal entry
    """
    hive = journal.apiary.hives.get(inspection.hive_id)
    hive_name = hive.name if hive else "Unknown hive"

    # Determine sentiment from findings
    findings = inspection.findings
    has_concerns = (
        findings.get("mites_observed") or
        findings.get("beetles_observed") or
        findings.get("disease_signs") or
        not findings.get("queen_seen")
    )

    sentiment = Sentiment.CONCERNED if has_concerns else Sentiment.NEUTRAL

    # Build narrative content
    content_parts = [f"Inspected {hive_name}."]

    if findings.get("queen_seen"):
        content_parts.append("Queen was observed.")
    if findings.get("eggs_seen"):
        content_parts.append("Fresh eggs present.")

    if findings.get("mites_observed"):
        content_parts.append("CONCERN: Varroa mites detected.")
    if findings.get("beetles_observed"):
        content_parts.append("CONCERN: Small hive beetles seen.")

    if inspection.actions_taken:
        content_parts.append(f"Actions: {', '.join(inspection.actions_taken)}")

    content = " ".join(content_parts)

    # Determine tags
    tags = [inspection.inspection_type.value]
    if has_concerns:
        tags.append("concern")
    if findings.get("swarm_cells", 0) > 0:
        tags.append("swarm_prep")

    return journal.record(
        content=content,
        entry_type=EntryType.OBSERVATION,
        sentiment=sentiment,
        hive_ids=[inspection.hive_id],
        tags=tags,
        inspection_id=inspection.inspection_id,
    )
