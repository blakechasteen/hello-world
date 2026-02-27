#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proactive Agent for ChatOps
============================
Monitors conversations and provides proactive suggestions, insights, and automation.

Features:
- Pattern detection (decisions, action items, questions)
- Automatic suggestions based on conversation flow
- Meeting notes extraction
- Task tracking
- Follow-up reminders

Architecture:
    Conversation Monitor
        ↓
    Pattern Detection
        ├→ Decision Detection
        ├→ Action Item Extraction
        ├→ Question Identification
        └→ Topic Tracking
        ↓
    Proactive Suggestions
"""

import logging
import re
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Data Structures
# ============================================================================

@dataclass
class Decision:
    """A decision made in conversation."""
    text: str
    decided_by: str
    timestamp: datetime
    conversation_id: str
    message_id: str
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionItem:
    """An action item extracted from conversation."""
    text: str
    assigned_to: Optional[str] = None
    mentioned_by: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    conversation_id: str = ""
    message_id: str = ""
    due_date: Optional[datetime] = None
    status: str = "open"  # open, in_progress, completed
    priority: str = "medium"  # low, medium, high
    tags: List[str] = field(default_factory=list)


@dataclass
class Question:
    """An unanswered question from conversation."""
    text: str
    asked_by: str
    timestamp: datetime
    conversation_id: str
    message_id: str
    answered: bool = False
    answer_message_id: Optional[str] = None


@dataclass
class MeetingNotes:
    """Structured meeting notes."""
    conversation_id: str
    title: str
    start_time: datetime
    end_time: datetime
    participants: List[str]
    topics: List[str]
    decisions: List[Decision]
    action_items: List[ActionItem]
    questions: List[Question]
    summary: str
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Pattern Detectors
# ============================================================================

class DecisionDetector:
    """Detects decision-making patterns in conversation."""

    DECISION_PATTERNS = [
        r"let['\s]+s\s+(use|go\s+with|do|implement|choose)",
        r"we['\s]+(decided|agreed|will|should)\s+",
        r"(decision|agreed):\s*(.+)",
        r"final\s+decision",
        r"we['\s]+re\s+going\s+with",
    ]

    def detect(self, messages: List[Dict]) -> List[Decision]:
        """
        Detect decisions in message list.

        Args:
            messages: List of message dicts

        Returns:
            List of Decision objects
        """
        decisions = []

        for msg in messages:
            text = msg.get("text", "").lower()

            for pattern in self.DECISION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    decision = Decision(
                        text=msg.get("text", ""),
                        decided_by=msg.get("sender", ""),
                        timestamp=msg.get("timestamp", datetime.now()),
                        conversation_id=msg.get("conversation_id", ""),
                        message_id=msg.get("message_id", "")
                    )
                    decisions.append(decision)
                    break

        return decisions


class ActionItemDetector:
    """Detects action items and tasks."""

    ACTION_PATTERNS = [
        r"(todo|to\s+do|task):\s*(.+)",
        r"(need|needs)\s+to\s+(.+)",
        r"(should|must|will)\s+(.+)",
        r"@(\w+)[,\s]+(please|can\s+you)\s+(.+)",
        r"action\s+item:\s*(.+)",
    ]

    ASSIGNMENT_PATTERNS = [
        r"@(\w+)[,\s]+(.+)",
        r"(\w+)\s+will\s+(.+)",
        r"assign(ed)?\s+to\s+(\w+)",
    ]

    def detect(self, messages: List[Dict]) -> List[ActionItem]:
        """Detect action items in messages."""
        items = []

        for msg in messages:
            text = msg.get("text", "")

            # Check action patterns
            for pattern in self.ACTION_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract assigned user if mentioned
                    assigned_to = None
                    for assign_pattern in self.ASSIGNMENT_PATTERNS:
                        assign_match = re.search(assign_pattern, text, re.IGNORECASE)
                        if assign_match:
                            assigned_to = assign_match.group(1)
                            break

                    item = ActionItem(
                        text=text,
                        assigned_to=assigned_to,
                        mentioned_by=msg.get("sender", ""),
                        timestamp=msg.get("timestamp", datetime.now()),
                        conversation_id=msg.get("conversation_id", ""),
                        message_id=msg.get("message_id", "")
                    )
                    items.append(item)
                    break

        return items


class QuestionDetector:
    """Detects questions in conversation."""

    def detect(self, messages: List[Dict]) -> List[Question]:
        """Detect questions in messages."""
        questions = []

        for msg in messages:
            text = msg.get("text", "")

            # Simple question detection
            if "?" in text:
                question = Question(
                    text=text,
                    asked_by=msg.get("sender", ""),
                    timestamp=msg.get("timestamp", datetime.now()),
                    conversation_id=msg.get("conversation_id", ""),
                    message_id=msg.get("message_id", "")
                )
                questions.append(question)

        return questions


# ============================================================================
# Proactive Agent
# ============================================================================

class ProactiveAgent:
    """
    Monitors conversations and provides proactive insights.

    Features:
    - Detects decisions, action items, questions
    - Generates automatic suggestions
    - Extracts meeting notes
    - Tracks conversation topics
    - Provides periodic summaries

    Usage:
        agent = ProactiveAgent()

        # Process new messages
        insights = agent.process_messages(messages)

        # Get suggestions
        suggestions = agent.get_suggestions(conversation_id)

        # Generate meeting notes
        notes = agent.generate_meeting_notes(conversation_id)
    """

    def __init__(
        self,
        suggestion_threshold: int = 5,  # Messages before suggesting summary
        question_timeout_hours: int = 24  # Hours before flagging unanswered questions
    ):
        """
        Initialize proactive agent.

        Args:
            suggestion_threshold: Messages before proactive suggestions
            question_timeout_hours: Hours before flagging unanswered questions
        """
        self.suggestion_threshold = suggestion_threshold
        self.question_timeout_hours = question_timeout_hours

        # Pattern detectors
        self.decision_detector = DecisionDetector()
        self.action_detector = ActionItemDetector()
        self.question_detector = QuestionDetector()

        # Tracking
        self.decisions: Dict[str, List[Decision]] = defaultdict(list)
        self.action_items: Dict[str, List[ActionItem]] = defaultdict(list)
        self.questions: Dict[str, List[Question]] = defaultdict(list)
        self.message_counts: Dict[str, int] = defaultdict(int)

        logger.info("ProactiveAgent initialized")

    def process_messages(
        self,
        messages: List[Dict],
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Process messages and extract insights.

        Args:
            messages: List of message dicts
            conversation_id: Conversation ID

        Returns:
            Insights dict with detected patterns
        """
        insights = {
            "decisions": [],
            "action_items": [],
            "questions": [],
            "suggestions": []
        }

        # Detect patterns
        decisions = self.decision_detector.detect(messages)
        action_items = self.action_detector.detect(messages)
        questions = self.question_detector.detect(messages)

        # Store
        self.decisions[conversation_id].extend(decisions)
        self.action_items[conversation_id].extend(action_items)
        self.questions[conversation_id].extend(questions)
        self.message_counts[conversation_id] += len(messages)

        # Add to insights
        insights["decisions"] = decisions
        insights["action_items"] = action_items
        insights["questions"] = questions

        # Generate suggestions
        suggestions = self._generate_suggestions(conversation_id)
        insights["suggestions"] = suggestions

        return insights

    def _generate_suggestions(self, conversation_id: str) -> List[str]:
        """Generate proactive suggestions based on conversation state."""
        suggestions = []

        # Check message count for summary suggestion
        if self.message_counts[conversation_id] % self.suggestion_threshold == 0:
            suggestions.append(
                f"💡 You've had {self.message_counts[conversation_id]} messages. "
                "Would you like a summary? Try `!summarize`"
            )

        # Check for unanswered questions
        unanswered = [
            q for q in self.questions[conversation_id]
            if not q.answered and
            (datetime.now() - q.timestamp).total_seconds() > 3600  # 1 hour
        ]
        if len(unanswered) >= 3:
            suggestions.append(
                f"❓ There are {len(unanswered)} unanswered questions. "
                "Consider addressing them or using `!summarize questions`"
            )

        # Check for action items without assignment
        unassigned = [
            item for item in self.action_items[conversation_id]
            if not item.assigned_to and item.status == "open"
        ]
        if len(unassigned) >= 3:
            suggestions.append(
                f"📋 {len(unassigned)} action items need assignment. "
                "Try `!action items` to review"
            )

        # Check for multiple decisions - suggest documentation
        recent_decisions = [
            d for d in self.decisions[conversation_id]
            if (datetime.now() - d.timestamp).total_seconds() < 3600
        ]
        if len(recent_decisions) >= 2:
            suggestions.append(
                "✍️ Several decisions made recently. "
                "Consider documenting with `!remember <decision>`"
            )

        return suggestions

    def generate_meeting_notes(
        self,
        conversation_id: str,
        messages: List[Dict],
        title: Optional[str] = None
    ) -> MeetingNotes:
        """
        Generate structured meeting notes from conversation.

        Args:
            conversation_id: Conversation ID
            messages: List of messages
            title: Optional meeting title

        Returns:
            MeetingNotes object
        """
        if not messages:
            raise ValueError("No messages provided")

        # Extract participants
        participants = list(set(msg.get("sender", "") for msg in messages))

        # Extract topics (simplified - would use topic modeling)
        topics = self._extract_topics(messages)

        # Get decisions, actions, questions
        decisions = self.decision_detector.detect(messages)
        action_items = self.action_detector.detect(messages)
        questions = self.question_detector.detect(messages)

        # Generate summary
        summary = self._generate_summary(messages, decisions, action_items)

        # Create meeting notes
        notes = MeetingNotes(
            conversation_id=conversation_id,
            title=title or f"Meeting - {datetime.now().strftime('%Y-%m-%d')}",
            start_time=messages[0].get("timestamp", datetime.now()),
            end_time=messages[-1].get("timestamp", datetime.now()),
            participants=participants,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            questions=questions,
            summary=summary,
            raw_messages=messages
        )

        return notes

    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """Extract discussion topics (simplified)."""
        # In production, would use topic modeling or LLM
        # For now, extract common keywords
        text = " ".join(msg.get("text", "") for msg in messages).lower()

        keywords = [
            "architecture", "implementation", "design", "testing",
            "deployment", "performance", "security", "api",
            "database", "integration", "chatops", "matrix", "hololoom"
        ]

        topics = [kw for kw in keywords if kw in text]
        return topics[:5]  # Top 5

    def _generate_summary(
        self,
        messages: List[Dict],
        decisions: List[Decision],
        action_items: List[ActionItem]
    ) -> str:
        """Generate meeting summary."""
        summary_parts = []

        # Overview
        summary_parts.append(
            f"Meeting with {len(set(m.get('sender') for m in messages))} participants, "
            f"{len(messages)} messages exchanged."
        )

        # Decisions
        if decisions:
            summary_parts.append(
                f"\n**Decisions Made ({len(decisions)}):**"
            )
            for d in decisions[:3]:  # Top 3
                summary_parts.append(f"• {d.text[:100]}")

        # Action Items
        if action_items:
            summary_parts.append(
                f"\n**Action Items ({len(action_items)}):**"
            )
            for item in action_items[:3]:
                assignee = f"[@{item.assigned_to}]" if item.assigned_to else "[Unassigned]"
                summary_parts.append(f"• {assignee} {item.text[:100]}")

        return "\n".join(summary_parts)

    def format_meeting_notes(self, notes: MeetingNotes) -> str:
        """Format meeting notes as markdown."""
        lines = [
            f"# {notes.title}",
            "",
            f"**Date:** {notes.start_time.strftime('%Y-%m-%d')}",
            f"**Duration:** {(notes.end_time - notes.start_time).total_seconds() / 60:.0f} minutes",
            f"**Participants:** {', '.join(notes.participants)}",
            "",
            "## Topics Discussed",
            ""
        ]

        for topic in notes.topics:
            lines.append(f"• {topic}")

        lines.extend(["", "## Decisions", ""])
        for decision in notes.decisions:
            lines.append(f"• {decision.text}")

        lines.extend(["", "## Action Items", ""])
        for item in notes.action_items:
            assignee = f"[@{item.assigned_to}]" if item.assigned_to else "[ ]"
            lines.append(f"• {assignee} {item.text}")

        if notes.questions:
            lines.extend(["", "## Questions", ""])
            for q in notes.questions:
                status = "✓" if q.answered else "○"
                lines.append(f"• {status} {q.text}")

        lines.extend(["", "## Summary", "", notes.summary])

        return "\n".join(lines)

    def get_statistics(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation."""
        return {
            "total_messages": self.message_counts[conversation_id],
            "decisions": len(self.decisions[conversation_id]),
            "action_items": len(self.action_items[conversation_id]),
            "questions": len(self.questions[conversation_id]),
            "open_action_items": len([
                item for item in self.action_items[conversation_id]
                if item.status == "open"
            ]),
            "unanswered_questions": len([
                q for q in self.questions[conversation_id]
                if not q.answered
            ])
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Proactive Agent Demo")
    print("="*80)
    print()

    # Create agent
    agent = ProactiveAgent()

    # Simulate meeting conversation
    messages = [
        {
            "text": "Let's discuss the chatops integration architecture",
            "sender": "@alice:matrix.org",
            "timestamp": datetime.now(),
            "conversation_id": "meeting_1",
            "message_id": "msg_1"
        },
        {
            "text": "We should use Matrix for messaging. What do you think?",
            "sender": "@bob:matrix.org",
            "timestamp": datetime.now(),
            "conversation_id": "meeting_1",
            "message_id": "msg_2"
        },
        {
            "text": "Let's go with Matrix.org. Decision made.",
            "sender": "@alice:matrix.org",
            "timestamp": datetime.now(),
            "conversation_id": "meeting_1",
            "message_id": "msg_3"
        },
        {
            "text": "@charlie, can you implement the bot client?",
            "sender": "@alice:matrix.org",
            "timestamp": datetime.now(),
            "conversation_id": "meeting_1",
            "message_id": "msg_4"
        },
        {
            "text": "TODO: Integrate with HoloLoom orchestrator",
            "sender": "@bob:matrix.org",
            "timestamp": datetime.now(),
            "conversation_id": "meeting_1",
            "message_id": "msg_5"
        }
    ]

    # Process messages
    print("Processing conversation...\n")
    insights = agent.process_messages(messages, "meeting_1")

    print(f"Detected {len(insights['decisions'])} decisions:")
    for d in insights['decisions']:
        print(f"  • {d.text}")
    print()

    print(f"Detected {len(insights['action_items'])} action items:")
    for item in insights['action_items']:
        assignee = f"@{item.assigned_to}" if item.assigned_to else "[Unassigned]"
        print(f"  • {assignee}: {item.text}")
    print()

    print(f"Detected {len(insights['questions'])} questions:")
    for q in insights['questions']:
        print(f"  • {q.text}")
    print()

    # Generate meeting notes
    print("Generating meeting notes...\n")
    notes = agent.generate_meeting_notes("meeting_1", messages, "ChatOps Architecture Discussion")

    formatted = agent.format_meeting_notes(notes)
    print(formatted)
    print()

    # Statistics
    stats = agent.get_statistics("meeting_1")
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Demo complete!")
