"""
Elle Core - Farm & Kitchen Cooperative Intelligence System

Comprehensive operational intelligence for Coz using HoloLoom/MirrorCore.

Key components:
- Voice-editable SOPs
- Real-time time/profit tracking
- Decision support engine
- Knowledge management (HoloLoom RAG)
- Predictive analytics

Created: 2025-11-15
Author: Blake Chasteen
Version: 0.1.0-alpha
"""

from elle.sop_schema import SOP, SOPStep, Ingredient, StepType, UnitType
from elle.tracker import TaskTracker, TaskResult, TaskStatus
from elle.voice_interface import VoiceSOPEditor
from elle.mirrorcore import DecisionEngine, ElleKnowledge, Recommendation
from elle.budget import BudgetBuilder, Budget, BudgetLine, BudgetCategory, BudgetPeriod

__version__ = "0.1.0-alpha"
__all__ = [
    "SOP",
    "SOPStep",
    "Ingredient",
    "StepType",
    "UnitType",
    "TaskTracker",
    "TaskResult",
    "TaskStatus",
    "VoiceSOPEditor",
    "DecisionEngine",
    "ElleKnowledge",
    "Recommendation",
    "BudgetBuilder",
    "Budget",
    "BudgetLine",
    "BudgetCategory",
    "BudgetPeriod",
]
