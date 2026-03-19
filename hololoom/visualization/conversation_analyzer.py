"""
Conversation Analyzer — Topic/Entity Extraction + Trajectory Detection
========================================================================
Extracts topics and entities from conversation turns using pure
regex/heuristic methods (no LLM calls, no external dependencies).

Designed to run in the background jenny pass (<100ms per turn).
Upgradeable to LLM-based extraction later without changing the interface.

Author: HoloLoom Team
Date: 2026-03-11
"""

import re
from datetime import datetime, timedelta

from .conversation_graph import ConversationGraph, Trajectory

# Session boundary: gap > 15 minutes = new conversation
SESSION_GAP = timedelta(minutes=15)

# Common stop words to filter out of topic extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "not", "no", "nor", "but", "or", "and", "if", "then", "else",
    "so", "too", "very", "just", "also", "about", "more", "some", "any",
    "each", "every", "all", "both", "few", "many", "much", "most",
    "other", "another", "such", "only", "own", "same",
    "than", "into", "with", "from", "for", "on", "at", "to", "in", "of",
    "by", "up", "out", "off", "over", "under", "again", "once",
    "here", "there",
    "don", "didn", "doesn", "isn", "aren", "wasn", "weren", "won",
    "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn",
    "let", "lets", "let's", "get", "got", "getting", "make", "making",
    "know", "think", "want", "like", "look", "use", "using", "used",
    "say", "said", "tell", "told", "ask", "asked",
    "yes", "ok", "okay", "sure", "right", "yeah", "yep", "nope",
    "hi", "hello", "hey", "thanks", "thank", "please",
    "really", "actually", "basically", "probably", "maybe",
    "something", "anything", "everything", "nothing", "thing", "things",
    "way", "time", "well", "still", "even", "back", "now",
})

# Comparison signal phrases
_COMPARISON_PATTERNS = [
    re.compile(r"\b(\w[\w\s]{1,30}?)\s+(?:vs\.?|versus)\s+(\w[\w\s]{1,30})\b", re.I),
    re.compile(r"\bcompare\s+(\w[\w\s]{1,30}?)\s+(?:and|with|to)\s+(\w[\w\s]{1,30})\b", re.I),
    re.compile(r"\bdifference\s+between\s+(\w[\w\s]{1,30}?)\s+and\s+(\w[\w\s]{1,30})\b", re.I),
    re.compile(r"\b(\w[\w\s]{1,30}?)\s+or\s+(\w[\w\s]{1,30})\s*\?\s*$", re.I | re.M),
]

# Action/decision signal phrases (checked via `phrase in text`)
_DECISION_SIGNALS = frozenset({
    "should", "choose", "pick", "decide", "select", "recommend",
    "better", "best", "prefer", "go with", "opt for", "which one",
})

# Technical compound terms to preserve
_COMPOUND_TERMS = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # PascalCase
    r"|[a-z]+_[a-z_]+"               # snake_case
    r"|[a-z]+-[a-z-]+"               # kebab-case
    r"|[A-Z]{2,}"                     # ACRONYMS
    r")\b"
)

# Quoted terms
_QUOTED = re.compile(r'["\u201c]([^"\u201d]{2,40})["\u201d]')

# Multi-word noun phrase (adjective + noun patterns)
_NOUN_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # "Knowledge Graph", "Matrix Channel"
)


class ConversationAnalyzer:
    """
    Stateless analyzer — extract topics/entities from text,
    detect conversation trajectory from graph shape.
    """

    def extract_topics(self, text: str) -> list[str]:
        """
        Extract topic-like terms from message text.

        Uses word frequency + compound term detection + noun phrases.
        Returns deduplicated list, most salient first.
        """
        topics: dict[str, int] = {}

        # 1. Multi-word noun phrases ("Knowledge Graph", "Docker Compose")
        for match in _NOUN_PHRASE.finditer(text):
            phrase = match.group(1).strip()
            if len(phrase) > 2:
                topics[phrase] = topics.get(phrase, 0) + 3

        # 2. Compound terms (PascalCase, snake_case, ACRONYMS)
        for match in _COMPOUND_TERMS.finditer(text):
            term = match.group(0)
            if len(term) > 1:
                topics[term] = topics.get(term, 0) + 2

        # 3. Quoted terms
        for match in _QUOTED.finditer(text):
            term = match.group(1).strip()
            if len(term) > 1:
                topics[term] = topics.get(term, 0) + 2

        # 4. Significant single words (not stop words, 4+ chars)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        for word in words:
            if word not in _STOP_WORDS and word not in topics:
                topics[word] = topics.get(word, 0) + 1

        # Sort by salience, return top terms
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_topics[:10]]

    def extract_entities(self, text: str) -> list[str]:
        """
        Extract entity-like terms: @mentions, URLs, identifiers.
        """
        entities: list[str] = []

        # @mentions
        for match in re.finditer(r"@(\w+)", text):
            entities.append(f"@{match.group(1)}")

        # URLs (extract domain)
        for match in re.finditer(r"https?://([^\s/]+)", text):
            entities.append(match.group(1))

        # File paths
        for match in re.finditer(r"[\w/\\]+\.(?:py|ts|js|tsx|rs|go|java|md)\b", text):
            entities.append(match.group(0))

        return entities[:8]

    def detect_trajectory(self, graph: ConversationGraph) -> Trajectory:
        """
        Detect the conversation's current trajectory from graph shape.
        """
        if not graph.nodes:
            return Trajectory.EXPLORING

        weights = sorted(
            [n.weight for n in graph.nodes.values()],
            reverse=True,
        )

        top = weights[0] if weights else 0
        second = weights[1] if len(weights) > 1 else 0

        # Check for comparison edges
        has_comparison = any(e.relation == "compared_with" for e in graph.edges)

        # Deep dive: one topic dominates (>2x the second)
        if top > 3 and (second == 0 or top > second * 2):
            return Trajectory.DEEP_DIVE

        # Comparing: two topics near-equal and dominant
        if has_comparison or (second > 0 and top <= second * 1.5 and top >= 3):
            return Trajectory.COMPARING

        # Exploring: many topics, no clear dominance
        if len(weights) >= 3 and top <= 3:
            return Trajectory.EXPLORING

        # Wrapping up: most topics haven't been mentioned recently
        if graph.turn_count > 5:
            recent_threshold = graph.turn_count - 2
            active_nodes = sum(
                1 for n in graph.nodes.values()
                if n.last_seen_turn >= recent_threshold
            )
            if active_nodes <= 1 and len(graph.nodes) > 3:
                return Trajectory.WRAPPING_UP

        return Trajectory.EXPLORING

    def detect_session_boundary(
        self,
        last_time: datetime | None,
        current_time: datetime,
    ) -> bool:
        """True if enough time has passed to consider this a new session."""
        if last_time is None:
            return False
        return (current_time - last_time) > SESSION_GAP

    def detect_comparisons(
        self,
        text: str,
        graph: ConversationGraph,
    ) -> tuple[str, str] | None:
        """
        Detect explicit comparison between two topics.

        Returns (topic_a, topic_b) if found, None otherwise.
        """
        for pattern in _COMPARISON_PATTERNS:
            match = pattern.search(text)
            if match:
                a = match.group(1).strip()
                b = match.group(2).strip()
                if len(a) > 1 and len(b) > 1:
                    return (a, b)

        return None

    def has_decision_language(self, text: str) -> bool:
        """True if the text contains decision/selection language."""
        lower = text.lower()
        return any(signal in lower for signal in _DECISION_SIGNALS)
