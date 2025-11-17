# -*- coding: utf-8 -*-
"""
Query Engine
============

Natural language query interface for HoloLoom knowledge graphs.

Supports queries like:
- "What should I work on?"
- "What's due this week?"
- "When did I finish X?"
- "Show my notes about Y"
- "What am I working on?"
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from HoloLoom.memory.graph import KG
    from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor
except ImportError:
    KG = None
    OrgLiveMonitor = None


# ============================================================================
# Query Types
# ============================================================================

class QueryType(Enum):
    """Types of queries we can handle."""
    CURRENT_TASKS = "current_tasks"           # What am I working on?
    NEXT_TASK = "next_task"                   # What should I work on next?
    DEADLINES = "deadlines"                   # What's due soon?
    SEARCH = "search"                         # Find notes about X
    TEMPORAL = "temporal"                     # When did X happen?
    STATUS = "status"                         # What's the status of X?
    COMPLETED = "completed"                   # What did I finish?
    BLOCKED = "blocked"                       # What's blocked?
    PRIORITY = "priority"                     # What's high priority?
    STATS = "stats"                           # Show me statistics
    UNKNOWN = "unknown"                       # Couldn't classify


@dataclass
class QueryIntent:
    """
    Classified query intent.
    """
    query_type: QueryType
    entities: List[str] = field(default_factory=list)      # Extracted entities
    timeframe: Optional[str] = None                        # "today", "this week", etc.
    filters: Dict[str, Any] = field(default_factory=dict)  # Additional filters
    confidence: float = 1.0                                # Classification confidence
    original_query: str = ""                               # Original text


# ============================================================================
# Query Classifier
# ============================================================================

class QueryClassifier:
    """
    Classifies natural language queries into intents.

    Uses regex patterns for now (could upgrade to ML later).
    """

    # Pattern matching rules
    PATTERNS = {
        QueryType.CURRENT_TASKS: [
            r"what.*(?:am i|am i currently).*work(?:ing)? on",
            r"current.*task",
            r"what.*(?:doing|in progress)",
        ],
        QueryType.NEXT_TASK: [
            r"what.*should.*work on",
            r"what.*next",
            r"what.*do.*next",
            r"suggest.*task",
            r"recommend.*task",
        ],
        QueryType.DEADLINES: [
            r"what.*due",
            r"deadline",
            r"when.*due",
            r"what.*coming up",
        ],
        QueryType.SEARCH: [
            r"(?:show|find|search).*notes? about",
            r"(?:show|find|search).*related to",
            r"notes? (?:on|about)",
        ],
        QueryType.TEMPORAL: [
            r"when (?:did|was).*(?:finish|complete|done)",
            r"when (?:did|was).*(?:start|begin)",
            r"how long.*take",
        ],
        QueryType.STATUS: [
            r"(?:status|progress) of",
            r"what.*status",
            r"how.*going",
        ],
        QueryType.COMPLETED: [
            r"what.*(?:finish|complete|done)",
            r"(?:show|list).*completed",
            r"what.*accomplished",
        ],
        QueryType.BLOCKED: [
            r"what.*blocked",
            r"what.*stuck",
            r"what.*waiting",
        ],
        QueryType.PRIORITY: [
            r"what.*(?:high )?priority",
            r"what.*urgent",
            r"what.*important",
        ],
        QueryType.STATS: [
            r"(?:show|give).*stat(?:istic)?s",
            r"how many",
            r"analytics",
            r"summary",
        ],
    }

    # Timeframe patterns
    TIMEFRAMES = {
        'today': [r'\btoday\b', r'\bthis day\b'],
        'this week': [r'\bthis week\b', r'\bweek\b'],
        'this month': [r'\bthis month\b', r'\bmonth\b'],
        'tomorrow': [r'\btomorrow\b'],
        'next week': [r'\bnext week\b'],
    }

    def classify(self, query: str) -> QueryIntent:
        """
        Classify query into intent.

        Args:
            query: Natural language query string

        Returns:
            QueryIntent with classification
        """
        query_lower = query.lower()

        # Try to match patterns
        query_type = QueryType.UNKNOWN
        confidence = 0.0

        for qtype, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    query_type = qtype
                    confidence = 0.9  # High confidence for pattern match
                    break
            if query_type != QueryType.UNKNOWN:
                break

        # Extract timeframe
        timeframe = None
        for tf_name, tf_patterns in self.TIMEFRAMES.items():
            for pattern in tf_patterns:
                if re.search(pattern, query_lower):
                    timeframe = tf_name
                    break
            if timeframe:
                break

        # Extract entities (simple: capitalized words and quoted strings)
        entities = []

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Find capitalized words (potential entities)
        # Skip common words
        skip_words = {'what', 'when', 'where', 'who', 'how', 'should', 'could', 'would', 'i'}
        words = query.split()
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in skip_words:
                entities.append(cleaned)

        return QueryIntent(
            query_type=query_type,
            entities=entities,
            timeframe=timeframe,
            confidence=confidence,
            original_query=query
        )


# ============================================================================
# Query Handlers
# ============================================================================

class QueryHandler:
    """
    Base class for query handlers.
    """

    def __init__(self, kg: 'KG', monitor: Optional['OrgLiveMonitor'] = None):
        self.kg = kg
        self.monitor = monitor

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Handle query intent.

        Returns:
            Dict with 'results' and metadata
        """
        raise NotImplementedError


class CurrentTasksHandler(QueryHandler):
    """Handle queries about current tasks."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        # Find all nodes with IN-PROGRESS state
        # This requires scanning the graph for nodes with TODO state metadata

        current_tasks = []

        # Search graph for tasks
        for node in self.kg.G.nodes():
            # Get edges related to TODO states
            for _, neighbor, data in self.kg.G.edges(node, data=True):
                if data.get('type', '').startswith('CHANGE_TODO_STATE'):
                    # Check if latest state is IN-PROGRESS
                    if data.get('new_value') == 'IN-PROGRESS':
                        current_tasks.append({
                            'id': node,
                            'title': node,  # Would need to get actual title from metadata
                            'state': 'IN-PROGRESS'
                        })
                        break

        return {
            'results': current_tasks,
            'count': len(current_tasks),
            'query_type': 'current_tasks'
        }


class DeadlinesHandler(QueryHandler):
    """Handle queries about deadlines."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        # Find all DEADLINE edges
        deadline_tasks = []

        for src, dst, data in self.kg.G.edges(data=True):
            if data.get('type') == 'DEADLINE':
                # dst is time node like "time::2025-11-20"
                deadline_str = dst.replace('time::', '')

                try:
                    # Parse deadline
                    deadline = datetime.fromisoformat(deadline_str.split()[0])

                    # Filter by timeframe if specified
                    if intent.timeframe:
                        if not self._in_timeframe(deadline, intent.timeframe):
                            continue

                    deadline_tasks.append({
                        'id': src,
                        'title': src,
                        'deadline': deadline.isoformat(),
                        'days_until': (deadline - datetime.now()).days
                    })
                except Exception:
                    # Skip invalid dates
                    continue

        # Sort by deadline
        deadline_tasks.sort(key=lambda x: x['deadline'])

        return {
            'results': deadline_tasks,
            'count': len(deadline_tasks),
            'query_type': 'deadlines'
        }

    def _in_timeframe(self, deadline: datetime, timeframe: str) -> bool:
        """Check if deadline is in specified timeframe."""
        now = datetime.now()

        if timeframe == 'today':
            return deadline.date() == now.date()
        elif timeframe == 'tomorrow':
            return deadline.date() == (now + timedelta(days=1)).date()
        elif timeframe == 'this week':
            # This week = next 7 days
            return deadline <= now + timedelta(days=7)
        elif timeframe == 'this month':
            return deadline.month == now.month and deadline.year == now.year
        elif timeframe == 'next week':
            next_week_start = now + timedelta(days=7)
            next_week_end = now + timedelta(days=14)
            return next_week_start <= deadline <= next_week_end

        return True


class TemporalHandler(QueryHandler):
    """Handle temporal queries (when did X happen?)."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        if not self.monitor:
            return {
                'results': [],
                'error': 'Temporal queries require OrgLiveMonitor',
                'query_type': 'temporal'
            }

        # Search change history for entities mentioned in query
        relevant_changes = []

        for entity in intent.entities:
            # Find changes related to this entity
            changes = self.monitor.get_change_history(heading_id=entity)

            # Also search by partial match
            if not changes:
                all_changes = self.monitor.get_change_history()
                changes = [
                    c for c in all_changes
                    if entity.lower() in c.heading_id.lower()
                    or (c.metadata.get('title', '').lower().find(entity.lower()) >= 0)
                ]

            relevant_changes.extend(changes)

        # Format results
        results = []
        for change in relevant_changes:
            results.append({
                'heading_id': change.heading_id,
                'timestamp': change.timestamp.isoformat(),
                'change_type': change.change_type,
                'old_value': change.old_value,
                'new_value': change.new_value,
                'title': change.metadata.get('title', change.heading_id)
            })

        return {
            'results': results,
            'count': len(results),
            'query_type': 'temporal'
        }


class StatsHandler(QueryHandler):
    """Handle statistics queries."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        stats = self.kg.stats()

        # Add temporal stats if monitor available
        if self.monitor:
            change_count = len(self.monitor.change_history)
            todo_transitions = self.monitor.get_todo_transitions()

            stats['total_changes'] = change_count
            stats['tasks_tracked'] = len(todo_transitions)

        return {
            'results': stats,
            'query_type': 'stats'
        }


class SearchHandler(QueryHandler):
    """Handle semantic search queries."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        # Simple keyword search for now
        # Could upgrade to semantic embeddings later

        search_terms = intent.entities
        if not search_terms:
            # Extract from query
            search_terms = intent.original_query.lower().replace('search', '').replace('find', '').replace('notes', '').replace('about', '').strip().split()

        matching_nodes = []

        for node in self.kg.G.nodes():
            node_lower = node.lower()

            # Check if any search term matches
            for term in search_terms:
                if term.lower() in node_lower:
                    matching_nodes.append({
                        'id': node,
                        'title': node,
                        'match_score': 1.0
                    })
                    break

        return {
            'results': matching_nodes,
            'count': len(matching_nodes),
            'query_type': 'search'
        }


class BlockedHandler(QueryHandler):
    """Handle queries about blocked tasks."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        # Find tasks with BLOCKED state or BLOCKED_BY edges
        blocked_tasks = []

        for node in self.kg.G.nodes():
            # Check for BLOCKED_BY edges
            blocked_by = self.kg.get_related_by_type(node, 'BLOCKED_BY', 'out')

            if blocked_by:
                blocked_tasks.append({
                    'id': node,
                    'title': node,
                    'blocked_by': blocked_by,
                    'blocker_count': len(blocked_by)
                })

            # Also check for WAITING state in change history
            if self.monitor:
                changes = self.monitor.get_change_history(heading_id=node)
                for change in changes:
                    if (change.change_type == 'todo_state_change' and
                        change.new_value in ['WAITING', 'BLOCKED']):
                        if node not in [t['id'] for t in blocked_tasks]:
                            blocked_tasks.append({
                                'id': node,
                                'title': node,
                                'state': change.new_value,
                                'since': change.timestamp.isoformat()
                            })

        return {
            'results': blocked_tasks,
            'count': len(blocked_tasks),
            'query_type': 'blocked'
        }


class PriorityHandler(QueryHandler):
    """Handle queries about high-priority tasks."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        # Find high priority tasks
        # Priority can come from:
        # 1. [#A] priority markers
        # 2. :priority: tag
        # 3. Deadline proximity
        # 4. Explicit HIGH_PRIORITY edges

        priority_tasks = []

        for node in self.kg.G.nodes():
            priority_score = 0.0
            reasons = []

            # Check for priority edges
            if self.kg.G.has_edge(node, 'HIGH_PRIORITY'):
                priority_score += 10.0
                reasons.append('marked_high_priority')

            # Check for urgent deadlines
            for src, dst, data in self.kg.G.edges(node, data=True):
                if data.get('type') == 'DEADLINE':
                    deadline_str = dst.replace('time::', '')
                    try:
                        deadline = datetime.fromisoformat(deadline_str.split()[0])
                        days_until = (deadline - datetime.now()).days

                        if days_until <= 1:
                            priority_score += 8.0
                            reasons.append('deadline_today_or_tomorrow')
                        elif days_until <= 3:
                            priority_score += 5.0
                            reasons.append('deadline_this_week')
                    except:
                        pass

            # Check for priority tag in node name
            if 'priority' in node.lower():
                priority_score += 3.0
                reasons.append('priority_tag')

            if priority_score > 0:
                priority_tasks.append({
                    'id': node,
                    'title': node,
                    'priority_score': priority_score,
                    'reasons': reasons
                })

        # Sort by priority score
        priority_tasks.sort(key=lambda x: -x['priority_score'])

        return {
            'results': priority_tasks,
            'count': len(priority_tasks),
            'query_type': 'priority'
        }


class CompletedHandler(QueryHandler):
    """Handle queries about completed tasks."""

    async def handle(self, intent: QueryIntent) -> Dict[str, Any]:
        if not self.monitor:
            return {
                'results': [],
                'error': 'Completed task queries require OrgLiveMonitor',
                'query_type': 'completed'
            }

        # Find tasks that changed to DONE state
        completed_tasks = []

        transitions = self.monitor.get_todo_transitions()

        for heading_id, trans_list in transitions.items():
            # Find transitions to DONE
            for timestamp, old_state, new_state in trans_list:
                if new_state == 'DONE':
                    # Check timeframe if specified
                    if intent.timeframe:
                        if not self._in_timeframe(timestamp, intent.timeframe):
                            continue

                    completed_tasks.append({
                        'id': heading_id,
                        'title': heading_id,
                        'completed_at': timestamp.isoformat(),
                        'duration': self._calculate_duration(heading_id, trans_list, timestamp)
                    })

        # Sort by completion time (most recent first)
        completed_tasks.sort(key=lambda x: x['completed_at'], reverse=True)

        return {
            'results': completed_tasks,
            'count': len(completed_tasks),
            'query_type': 'completed'
        }

    def _in_timeframe(self, timestamp: datetime, timeframe: str) -> bool:
        """Check if timestamp is in specified timeframe."""
        now = datetime.now()

        if timeframe == 'today':
            return timestamp.date() == now.date()
        elif timeframe == 'this week':
            # Last 7 days
            return timestamp >= now - timedelta(days=7)
        elif timeframe == 'this month':
            return timestamp.month == now.month and timestamp.year == now.year

        return True

    def _calculate_duration(self, heading_id: str, transitions: List[tuple], completion_time: datetime) -> Optional[float]:
        """Calculate how long task took from start to completion."""
        # Find when task was started (TODO -> IN-PROGRESS)
        start_time = None
        for timestamp, old_state, new_state in transitions:
            if new_state == 'IN-PROGRESS':
                start_time = timestamp
                break

        if start_time:
            duration = (completion_time - start_time).total_seconds() / 3600  # hours
            return round(duration, 1)

        return None


# ============================================================================
# Query Engine
# ============================================================================

class QueryEngine:
    """
    Main query engine that orchestrates classification and execution.
    """

    def __init__(self, kg: 'KG', monitor: Optional['OrgLiveMonitor'] = None):
        self.kg = kg
        self.monitor = monitor
        self.classifier = QueryClassifier()

        # Register handlers
        self.handlers = {
            QueryType.CURRENT_TASKS: CurrentTasksHandler(kg, monitor),
            QueryType.DEADLINES: DeadlinesHandler(kg, monitor),
            QueryType.TEMPORAL: TemporalHandler(kg, monitor),
            QueryType.STATS: StatsHandler(kg, monitor),
            QueryType.SEARCH: SearchHandler(kg, monitor),
            QueryType.BLOCKED: BlockedHandler(kg, monitor),
            QueryType.PRIORITY: PriorityHandler(kg, monitor),
            QueryType.COMPLETED: CompletedHandler(kg, monitor),
            # More handlers can be added here
        }

    async def query(self, query_str: str) -> Dict[str, Any]:
        """
        Execute natural language query.

        Args:
            query_str: Natural language query

        Returns:
            Dict with results and metadata
        """
        # Classify query
        intent = self.classifier.classify(query_str)

        # Get appropriate handler
        handler = self.handlers.get(intent.query_type)

        if not handler:
            return {
                'results': [],
                'error': f'No handler for query type: {intent.query_type}',
                'intent': intent,
                'query': query_str
            }

        # Execute handler
        result = await handler.handle(intent)

        # Add metadata
        result['intent'] = {
            'type': intent.query_type.value,
            'confidence': intent.confidence,
            'entities': intent.entities,
            'timeframe': intent.timeframe
        }
        result['query'] = query_str

        return result


# ============================================================================
# Convenience Function
# ============================================================================

# Global engine instance (lazy loaded)
_engine: Optional[QueryEngine] = None

def initialize_engine(kg: 'KG', monitor: Optional['OrgLiveMonitor'] = None):
    """Initialize global query engine."""
    global _engine
    _engine = QueryEngine(kg, monitor)


async def query(query_str: str, kg: Optional['KG'] = None, monitor: Optional['OrgLiveMonitor'] = None) -> Dict[str, Any]:
    """
    Convenience function for queries.

    Usage:
        result = await query("What should I work on?")

        # Or with explicit KG
        result = await query("What's due?", kg=my_kg)
    """
    global _engine

    # Use provided KG or require initialization
    if kg:
        engine = QueryEngine(kg, monitor)
        return await engine.query(query_str)

    if _engine is None:
        raise RuntimeError(
            "Query engine not initialized. Call initialize_engine(kg) first "
            "or pass kg parameter to query()"
        )

    return await _engine.query(query_str)
