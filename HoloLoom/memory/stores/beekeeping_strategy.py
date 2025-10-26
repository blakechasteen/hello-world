"""
Beekeeping-Specific Retrieval Strategy
=======================================
Custom memory retrieval optimized for beekeeping workflows.

Scoring factors:
1. **Seasonal relevance** - Winter prep in fall > spring splits in winter
2. **Hive priority** - Strong hives get higher weight than weak
3. **Urgency** - Critical tasks > routine observations
4. **Recency with decay** - Recent but seasonally-adjusted
5. **Genetic lineage** - Memories about related hives cluster

This strategy "knows" beekeeping context!
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import math


class Season(Enum):
    """Beekeeping seasons."""
    SPRING = "spring"  # March-May: Buildup, splits
    SUMMER = "summer"  # June-Aug: Honey flow, swarming
    FALL = "fall"      # Sept-Nov: Harvest, winter prep
    WINTER = "winter"  # Dec-Feb: Cluster, survival


class HivePriority(Enum):
    """Hive priority levels."""
    CRITICAL = 1.0   # Weak hives needing attention
    HIGH = 0.8       # Strong producers, breeding stock
    MEDIUM = 0.5     # Average hives
    LOW = 0.3        # Observer hives, experiments


class TaskUrgency(Enum):
    """Task urgency levels."""
    EMERGENCY = 1.0      # Hive failure, disease outbreak
    CRITICAL = 0.8       # Winter prep, treatment windows
    HIGH = 0.6           # Regular inspections, feeding
    ROUTINE = 0.3        # Observations, notes


def get_current_season() -> Season:
    """Determine current beekeeping season."""
    month = datetime.now().month

    if 3 <= month <= 5:
        return Season.SPRING
    elif 6 <= month <= 8:
        return Season.SUMMER
    elif 9 <= month <= 11:
        return Season.FALL
    else:
        return Season.WINTER


def calculate_seasonal_relevance(memory_season: Season, current_season: Season) -> float:
    """
    Calculate seasonal relevance score.

    Examples:
    - Winter prep notes in fall: 1.0 (highly relevant)
    - Winter prep notes in spring: 0.3 (less relevant)
    - Split notes in spring: 1.0 (perfect timing)
    """
    # Seasonal transition matrix
    transitions = {
        Season.SPRING: {
            Season.SPRING: 1.0,
            Season.SUMMER: 0.6,  # Planning for summer
            Season.FALL: 0.3,    # Less relevant
            Season.WINTER: 0.4   # Spring buildup from winter
        },
        Season.SUMMER: {
            Season.SPRING: 0.5,
            Season.SUMMER: 1.0,
            Season.FALL: 0.7,    # Planning for fall
            Season.WINTER: 0.2
        },
        Season.FALL: {
            Season.SPRING: 0.3,
            Season.SUMMER: 0.4,
            Season.FALL: 1.0,
            Season.WINTER: 0.9   # Preparing for winter
        },
        Season.WINTER: {
            Season.SPRING: 0.8,  # Planning for spring
            Season.SUMMER: 0.2,
            Season.FALL: 0.4,
            Season.WINTER: 1.0
        }
    }

    return transitions[current_season][memory_season]


def extract_season_from_context(context: Dict[str, Any]) -> Season:
    """Extract season from memory context."""
    # Try explicit season field
    if 'season' in context:
        try:
            return Season(context['season'])
        except ValueError:
            pass

    # Infer from timestamp
    if 'timestamp' in context:
        timestamp = context['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        month = timestamp.month
        if 3 <= month <= 5:
            return Season.SPRING
        elif 6 <= month <= 8:
            return Season.SUMMER
        elif 9 <= month <= 11:
            return Season.FALL
        else:
            return Season.WINTER

    # Default to current season
    return get_current_season()


def extract_hive_priority(context: Dict[str, Any]) -> float:
    """
    Extract hive priority from context.

    Factors:
    - Population strength
    - Genetics (Dennis, Jodi = high priority)
    - Status (weakest = critical, strongest = high)
    """
    priority = HivePriority.MEDIUM.value

    # Check population strength
    if 'population_strength' in context:
        strength = context['population_strength']
        if strength in ['very_weak', 'weak']:
            priority = max(priority, HivePriority.CRITICAL.value)
        elif strength in ['very_strong', 'strong']:
            priority = max(priority, HivePriority.HIGH.value)

    # Check colony status
    if 'colony_status' in context:
        status = context['colony_status']
        if status == 'weakest':
            priority = max(priority, HivePriority.CRITICAL.value)
        elif status == 'strongest':
            priority = max(priority, HivePriority.HIGH.value)

    # Check genetics
    if 'genetics' in context:
        genetics = context['genetics']
        if 'dennis' in genetics.lower() or 'jodi' in genetics.lower():
            priority = max(priority, HivePriority.HIGH.value)

    return priority


def extract_task_urgency(context: Dict[str, Any], text: str) -> float:
    """
    Extract task urgency from context and text.

    Keywords:
    - EMERGENCY: "failing", "dead", "disease", "robbing"
    - CRITICAL: "winter prep", "treatment", "combining"
    - HIGH: "inspection", "feeding", "check"
    - ROUTINE: "observation", "note"
    """
    urgency = TaskUrgency.ROUTINE.value

    text_lower = text.lower()

    # Emergency keywords
    emergency_keywords = ['failing', 'dead', 'disease', 'robbing', 'collapse']
    if any(kw in text_lower for kw in emergency_keywords):
        urgency = TaskUrgency.EMERGENCY.value

    # Critical keywords
    critical_keywords = ['winter prep', 'treatment', 'combining', 'split', 'critical']
    if any(kw in text_lower for kw in critical_keywords):
        urgency = max(urgency, TaskUrgency.CRITICAL.value)

    # High priority keywords
    high_keywords = ['inspection', 'feeding', 'check', 'monitor']
    if any(kw in text_lower for kw in high_keywords):
        urgency = max(urgency, TaskUrgency.HIGH.value)

    # Check context priority
    if 'priority' in context:
        priority = context['priority']
        if priority == 'critical':
            urgency = max(urgency, TaskUrgency.CRITICAL.value)
        elif priority == 'high':
            urgency = max(urgency, TaskUrgency.HIGH.value)

    return urgency


def calculate_recency_score(timestamp: datetime, max_age_days: int = 365) -> float:
    """
    Calculate recency score with exponential decay.

    Score = exp(-age_days / decay_constant)

    Recent memories get higher scores, but decay gracefully.
    """
    age = (datetime.now() - timestamp).days
    decay_constant = max_age_days / 3  # 33% decay point

    return math.exp(-age / decay_constant)


def calculate_beekeeping_score(
    memory_text: str,
    memory_context: Dict[str, Any],
    memory_timestamp: datetime,
    query_text: str,
    query_context: Dict[str, Any]
) -> float:
    """
    Calculate beekeeping-specific relevance score.

    Score = weighted average of:
    - Seasonal relevance (30%)
    - Hive priority (25%)
    - Task urgency (25%)
    - Recency (10%)
    - Semantic match (10%)

    Returns:
        score: Float [0, 1]
    """
    # 1. Seasonal relevance
    memory_season = extract_season_from_context(memory_context)
    current_season = get_current_season()
    seasonal_score = calculate_seasonal_relevance(memory_season, current_season)

    # 2. Hive priority
    hive_priority_score = extract_hive_priority(memory_context)

    # 3. Task urgency
    urgency_score = extract_task_urgency(memory_context, memory_text)

    # 4. Recency
    recency_score = calculate_recency_score(memory_timestamp)

    # 5. Semantic match (simple keyword overlap)
    query_words = set(query_text.lower().split())
    memory_words = set(memory_text.lower().split())
    common_words = query_words & memory_words
    semantic_score = len(common_words) / max(len(query_words), 1)

    # Weighted combination
    score = (
        0.30 * seasonal_score +
        0.25 * hive_priority_score +
        0.25 * urgency_score +
        0.10 * recency_score +
        0.10 * semantic_score
    )

    return min(score, 1.0)  # Clamp to [0, 1]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=== Beekeeping Retrieval Strategy Demo ===\n")

    # Simulate memories
    memories = [
        {
            "text": "Hive Jodi very strong - 15 frames, prepare for spring split",
            "context": {
                "hive": "hive-jodi-primary-001",
                "genetics": "jodi-line",
                "population_strength": "very_strong",
                "season": "fall"
            },
            "timestamp": datetime(2024, 10, 12)
        },
        {
            "text": "Hive 5 critically weak - combine immediately before winter",
            "context": {
                "hive": "hive-small-001",
                "population_strength": "very_weak",
                "priority": "critical",
                "colony_status": "weakest"
            },
            "timestamp": datetime(2024, 10, 12)
        },
        {
            "text": "Routine observation - bees foraging on goldenrod",
            "context": {
                "season": "fall"
            },
            "timestamp": datetime(2024, 10, 1)
        },
    ]

    # Query in fall
    query = "What needs attention before winter?"
    query_context = {}

    print(f"Query: '{query}'")
    print(f"Current season: {get_current_season().value}\n")

    scores = []
    for mem in memories:
        score = calculate_beekeeping_score(
            mem["text"],
            mem["context"],
            mem["timestamp"],
            query,
            query_context
        )
        scores.append((score, mem))

    # Sort by score
    scores.sort(reverse=True, key=lambda x: x[0])

    print("Ranked results:")
    for score, mem in scores:
        print(f"\n[{score:.3f}] {mem['text']}")
        print(f"        Season: {extract_season_from_context(mem['context']).value}")
        print(f"        Priority: {extract_hive_priority(mem['context']):.2f}")
        print(f"        Urgency: {extract_task_urgency(mem['context'], mem['text']):.2f}")

    print("\n✓ Demo complete!")