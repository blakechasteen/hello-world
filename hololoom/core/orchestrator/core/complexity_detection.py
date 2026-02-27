#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complexity Detection and Provenance
====================================

Query complexity assessment and provenance trace creation.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 469-573 (~112 lines)

This module handles:
- Hybrid word count + intent detection for complexity assessment
- Progressive complexity levels (LITE/FAST/FULL/RESEARCH)
- Provenance trace creation with operation tracking

Author: Claude Code (Elegance Pass Refactoring - Phase 3)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.weaving_orchestrator import WeavingOrchestrator

from hololoom.core.protocols.types import Query
from hololoom.core.protocols import ComplexityLevel, ProvenanceTrace
from hololoom.core.loom.command import PatternCard


logger = logging.getLogger(__name__)


def assess_complexity_level(
    orchestrator: 'WeavingOrchestrator',
    query: Query,
    trace: Optional[ProvenanceTrace] = None
) -> ComplexityLevel:
    """
    Assess query complexity using hybrid word count + intent detection.

    **Progressive Complexity (3-5-7-9):**
    - LITE (3): Greetings, simple commands (1-3 words, no questions)
    - FAST (5): Questions, knowledge queries (4-20 words OR question indicators)
    - FULL (7): Detailed analysis (21-50 words, complex questions)
    - RESEARCH (9): Deep research (analysis verbs, research keywords, 50+ words)

    **Intent Detection:**
    - Greetings: "hi", "hello", "thanks" → LITE
    - Questions: "what", "how", "why" → FAST (minimum)
    - Knowledge: "explain", "describe" → FAST (minimum)
    - Analysis: "analyze", "compare" → RESEARCH
    - Research: "comprehensive", "detailed" → RESEARCH

    Args:
        orchestrator: The WeavingOrchestrator instance
        query: User query
        trace: Optional provenance trace to record decision

    Returns:
        ComplexityLevel (LITE/FAST/FULL/RESEARCH)

    Example:
        >>> level = assess_complexity_level(orchestrator, Query(text="hi"))
        >>> assert level == ComplexityLevel.LITE

        >>> level = assess_complexity_level(orchestrator, Query(text="What is Thompson Sampling?"))
        >>> assert level == ComplexityLevel.FAST
    """
    if not orchestrator.enable_complexity_auto_detect:
        # Map pattern card to complexity level
        if orchestrator.default_pattern == PatternCard.BARE:
            return ComplexityLevel.LITE
        elif orchestrator.default_pattern == PatternCard.FAST:
            return ComplexityLevel.FAST
        else:  # FUSED
            return ComplexityLevel.FULL

    text = query.text.lower()
    words = text.split()
    word_count = len(words)

    # Extract intent patterns
    thresholds = orchestrator._complexity_thresholds

    # Check for specific intent patterns
    is_greeting = any(pattern in text for pattern in thresholds['greeting_patterns'])
    is_simple_command = any(cmd in words for cmd in thresholds['simple_commands'])
    has_question_word = any(q in words for q in thresholds['question_words'])
    has_question_mark = '?' in text
    has_knowledge_verb = any(verb in words for verb in thresholds['knowledge_verbs'])
    has_analysis_verb = any(verb in words for verb in thresholds['analysis_verbs'])
    has_research_keyword = any(keyword in text for keyword in thresholds['research_keywords'])

    # Determine complexity with sophisticated intent detection
    # Priority: Research > Full > Fast > Lite

    # RESEARCH: Analysis verbs, research keywords, or very long queries
    if has_analysis_verb or has_research_keyword or word_count > thresholds['full_max_words']:
        level = ComplexityLevel.RESEARCH
        reason = f"analysis_verb={has_analysis_verb}, research_keyword={has_research_keyword}, long_query={word_count > thresholds['full_max_words']}"

    # FULL: Detailed questions (21-50 words) or complex knowledge requests
    elif word_count >= thresholds['fast_max_words']:  # Changed > to >= (includes 20-word boundary)
        level = ComplexityLevel.FULL
        reason = f"word_count={word_count} >= {thresholds['fast_max_words']}"

    # FAST: Questions, knowledge verbs, or 4-20 words
    elif has_question_word or has_question_mark or has_knowledge_verb or word_count > thresholds['lite_max_words']:
        # Exception: Pure greetings stay LITE even if >3 words (but not if they have question words)
        if is_greeting and word_count <= 5 and not (has_question_word or has_question_mark):
            level = ComplexityLevel.LITE
            reason = f"greeting_pattern (word_count={word_count})"
        else:
            level = ComplexityLevel.FAST
            reason = f"question={has_question_word or has_question_mark}, knowledge_verb={has_knowledge_verb}, word_count={word_count}"

    # LITE: Simple greetings, short commands (1-3 words, no questions)
    else:
        level = ComplexityLevel.LITE
        reason = f"simple_query (word_count={word_count})"

    if trace:
        trace.add_shuttle_event(
            "complexity_assessment",
            f"Assessed complexity: {level.name}",
            {
                'word_count': word_count,
                'is_greeting': is_greeting,
                'has_question': has_question_word or has_question_mark,
                'has_knowledge_verb': has_knowledge_verb,
                'has_analysis_verb': has_analysis_verb,
                'has_research_keyword': has_research_keyword,
                'complexity_level': level.name,
                'reason': reason
            }
        )

    return level


def create_provenance_trace(
    orchestrator: 'WeavingOrchestrator',
    query: Query,
    complexity: ComplexityLevel
) -> ProvenanceTrace:
    """
    Create a new provenance trace for this operation.

    Args:
        orchestrator: The WeavingOrchestrator instance
        query: User query
        complexity: Assessed complexity level

    Returns:
        ProvenanceTrace with operation ID and start time

    Example:
        >>> trace = create_provenance_trace(orchestrator, query, ComplexityLevel.FAST)
        >>> assert trace.operation_id.startswith("weave_")
        >>> assert trace.complexity_level == ComplexityLevel.FAST
    """
    operation_id = f"weave_{int(time.time() * 1000)}"
    trace = ProvenanceTrace(
        operation_id=operation_id,
        complexity_level=complexity,
        start_time=time.perf_counter()
    )
    trace.add_shuttle_event("weave_start", f"Beginning weave for query: {query.text[:50]}...")
    return trace
