"""
EdWIN - Educational Weaving Intelligence Network

AI-powered adaptive tutoring system built on HoloLoom infrastructure.

**Version**: 1.0.0
**Date**: November 15, 2025

Core Components:
- EdWINKnowledgeGraph: Curriculum as knowledge graph
- EdWINTutor: RAG-powered tutoring engine
- EdWINStudentModel: Student progress tracking
- AdaptiveDifficultyEngine: Thompson Sampling for optimal challenge
- EdWINSafetyLayer: K-12 content filtering and compliance

Example:
    >>> from edwin import EdWIN
    >>>
    >>> edwin = EdWIN(
    ...     student_id="student_001",
    ...     student_name="Jane Doe",
    ...     grade=8
    ... )
    >>>
    >>> # Ask a question
    >>> result = await edwin.teach("What is a linear equation?")
    >>> print(result)
"""

from .curriculum_graph import EdWINKnowledgeGraph
from .tutoring_engine import EdWINTutor, TutoringResponse
from .student_model import EdWINStudentModel
from .adaptive_difficulty import AdaptiveDifficultyEngine
from .safety_layer import EdWINSafetyLayer
from .core import EdWIN

__version__ = "1.0.0"
__author__ = "EdWIN Development Team"

__all__ = [
    "EdWIN",
    "EdWINKnowledgeGraph",
    "EdWINTutor",
    "EdWINStudentModel",
    "AdaptiveDifficultyEngine",
    "EdWINSafetyLayer",
    "TutoringResponse",
]
