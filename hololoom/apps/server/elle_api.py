#!/usr/bin/env python3
from __future__ import annotations
"""
Elle API
========

Context-aware guidance endpoints for EdWIN persona.

Features:
- Scene analysis (understand code context)
- Intent detection (what user wants to do)
- Action suggestions (next steps)
- Context-aware guidance (AR-style help)

Created: 2025-01-20
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/elle", tags=["elle"])
logger = logging.getLogger(__name__)

# Import Elle components
try:
    from hololoom.elle.domain.action import Action, ActionType
    from hololoom.elle.domain.scene import Intent, Scene
    from hololoom.elle.engine import ElleEngine
    ELLE_AVAILABLE = True
    logger.info("✓ Elle components loaded successfully")
except ImportError as e:
    ELLE_AVAILABLE = False
    logger.warning(f"⚠ Elle components not available: {e}")
    ElleEngine = None

# Global Elle engine instance
elle_engine: Any | None = None


def get_elle_engine():
    """Get or create Elle engine instance."""
    global elle_engine
    if elle_engine is None and ELLE_AVAILABLE:
        elle_engine = ElleEngine()
    return elle_engine


# ============== REQUEST/RESPONSE MODELS ==============

class SceneRequest(BaseModel):
    code: str
    language: str
    file_name: str | None = None
    cursor_position: dict[str, int] | None = None  # {line, column}
    selection: dict[str, Any] | None = None  # {start, end}
    context: dict[str, Any] | None = None


class SceneAnalysis(BaseModel):
    complexity: str  # "trivial", "moderate", "complex"
    focus_areas: list[str]
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    suggestions: list[str]


class GuideRequest(BaseModel):
    scene: SceneRequest
    intent: str  # "seeking_guidance", "debugging", "learning", "refactoring"
    user_query: str | None = None


class GuideResponse(BaseModel):
    guidance: str
    suggested_actions: list[dict[str, Any]]
    focus_highlights: list[dict[str, Any]]  # Areas to highlight in editor
    related_concepts: list[str]
    confidence: float


class SuggestRequest(BaseModel):
    code: str
    language: str
    action_history: list[str] | None = None  # Recent actions taken


class SuggestResponse(BaseModel):
    suggestions: list[dict[str, Any]]
    reasoning: str
    priority: str  # "high", "medium", "low"


# ============== 1. SCENE ANALYSIS ==============

@router.post("/scene", response_model=SceneAnalysis)
async def analyze_scene(request: SceneRequest):
    """
    Analyze code context (Elle's scene understanding).

    Returns:
        - Complexity assessment
        - Focus areas (where to look)
        - Entities (functions, classes, variables)
        - Relationships between entities
        - Initial suggestions
    """

    if not ELLE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Elle not available. Check installation."
        )

    try:
        # Simple complexity heuristic (TODO: use Elle's actual analysis)
        lines = request.code.split('\n')
        line_count = len(lines)

        if line_count < 20:
            complexity = "trivial"
        elif line_count < 100:
            complexity = "moderate"
        else:
            complexity = "complex"

        # Extract entities (basic regex, TODO: use AST)
        entities = []
        relationships = []

        import re

        if request.language == "python":
            # Find functions
            func_pattern = r'def\s+(\w+)'
            for match in re.finditer(func_pattern, request.code):
                entities.append({
                    'name': match.group(1),
                    'type': 'function',
                    'line': request.code[:match.start()].count('\n') + 1
                })

            # Find classes
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, request.code):
                entities.append({
                    'name': match.group(1),
                    'type': 'class',
                    'line': request.code[:match.start()].count('\n') + 1
                })

        # Focus areas (where cursor is, or first entity)
        focus_areas = []
        if request.cursor_position:
            focus_areas.append(f"Line {request.cursor_position['line']}")
        elif entities:
            focus_areas.append(f"{entities[0]['type']}: {entities[0]['name']}")

        # Suggestions based on complexity
        suggestions = []
        if complexity == "complex":
            suggestions.append("Consider breaking this into smaller functions")
            suggestions.append("Add docstrings to improve clarity")
        elif complexity == "moderate":
            suggestions.append("Review for potential edge cases")
        else:
            suggestions.append("Code looks clean!")

        return SceneAnalysis(
            complexity=complexity,
            focus_areas=focus_areas,
            entities=entities,
            relationships=relationships,
            suggestions=suggestions
        )

    except Exception as e:
        logger.error(f"Scene analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============== 2. CONTEXT-AWARE GUIDANCE ==============

@router.post("/guide", response_model=GuideResponse)
async def get_guidance(request: GuideRequest):
    """
    Get context-aware guidance based on scene and intent.

    This is the main EdWIN endpoint - provides AR-style guidance.
    """

    if not ELLE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Elle not available. Check installation."
        )

    try:
        # Analyze scene first
        scene_analysis = await analyze_scene(request.scene)

        # Generate guidance based on intent
        intent_map = {
            "seeking_guidance": "Let me help you understand this code...",
            "debugging": "Let's debug this step by step...",
            "learning": "Here's what this code does...",
            "refactoring": "Here are some refactoring suggestions..."
        }

        guidance_prefix = intent_map.get(request.intent, "Let me analyze this...")

        # Build guidance text
        guidance_parts = [guidance_prefix]

        if scene_analysis.entities:
            entity_summary = f"I found {len(scene_analysis.entities)} entities: "
            entity_summary += ", ".join([f"{e['type']} '{e['name']}'" for e in scene_analysis.entities[:3]])
            guidance_parts.append(entity_summary)

        if scene_analysis.complexity == "complex":
            guidance_parts.append("This is fairly complex code. Would you like me to break it down?")

        guidance = " ".join(guidance_parts)

        # Suggested actions
        suggested_actions = []

        if request.intent == "refactoring":
            suggested_actions.append({
                'action': 'extract_function',
                'description': 'Extract repeated code into a function',
                'confidence': 0.85
            })
            suggested_actions.append({
                'action': 'add_docstrings',
                'description': 'Add documentation to functions',
                'confidence': 0.90
            })
        elif request.intent == "debugging":
            suggested_actions.append({
                'action': 'add_logging',
                'description': 'Add debug logging statements',
                'confidence': 0.80
            })
            suggested_actions.append({
                'action': 'check_edge_cases',
                'description': 'Verify edge case handling',
                'confidence': 0.75
            })
        else:
            suggested_actions.append({
                'action': 'explain_code',
                'description': 'Explain what this code does',
                'confidence': 0.95
            })

        # Focus highlights (areas to highlight in editor)
        focus_highlights = []
        for entity in scene_analysis.entities[:3]:
            focus_highlights.append({
                'line': entity['line'],
                'type': entity['type'],
                'description': f"{entity['type'].capitalize()}: {entity['name']}"
            })

        # Related concepts
        related_concepts = []
        if request.scene.language == "python":
            related_concepts = ["PEP 8", "Type hints", "Docstrings", "Unit testing"]
        elif request.scene.language in ["typescript", "javascript"]:
            related_concepts = ["ESLint", "TypeScript types", "JSDoc", "Jest testing"]

        return GuideResponse(
            guidance=guidance,
            suggested_actions=suggested_actions,
            focus_highlights=focus_highlights,
            related_concepts=related_concepts,
            confidence=0.85
        )

    except Exception as e:
        logger.error(f"Guidance generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Guidance failed: {str(e)}")


# ============== 3. ACTION SUGGESTIONS ==============

@router.post("/suggest", response_model=SuggestResponse)
async def suggest_next_actions(request: SuggestRequest):
    """
    Suggest next actions based on code context.

    Uses action history to avoid repetition.
    """

    if not ELLE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Elle not available. Check installation."
        )

    try:
        suggestions = []

        # Analyze code quality
        lines = request.code.split('\n')
        has_docstrings = '"""' in request.code or "'''" in request.code
        has_tests = 'test_' in request.code or 'def test' in request.code
        has_type_hints = '->' in request.code or ': int' in request.code or ': str' in request.code

        # Suggest based on what's missing
        if not has_docstrings and len(lines) > 10:
            suggestions.append({
                'action': 'add_docstrings',
                'description': 'Add docstrings to improve documentation',
                'priority': 'high',
                'estimated_time': '5 minutes'
            })

        if not has_type_hints and request.language == "python":
            suggestions.append({
                'action': 'add_type_hints',
                'description': 'Add type hints for better type safety',
                'priority': 'medium',
                'estimated_time': '10 minutes'
            })

        if not has_tests:
            suggestions.append({
                'action': 'write_tests',
                'description': 'Write unit tests for this code',
                'priority': 'high',
                'estimated_time': '15 minutes'
            })

        # Filter out recently done actions
        if request.action_history:
            suggestions = [
                s for s in suggestions
                if s['action'] not in request.action_history
            ]

        # Determine priority
        if any(s['priority'] == 'high' for s in suggestions):
            priority = "high"
        elif suggestions:
            priority = "medium"
        else:
            priority = "low"

        reasoning = f"Found {len(suggestions)} opportunities for improvement. "
        if not suggestions:
            reasoning += "Code looks good!"
        else:
            reasoning += f"Focus on {suggestions[0]['action']} first."

        return SuggestResponse(
            suggestions=suggestions,
            reasoning=reasoning,
            priority=priority
        )

    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")
