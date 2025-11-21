"""
Permaculture Design Toolset for HoloLoom

This module provides plant database management and guild recommendation
capabilities for permaculture design.

**Created: 2025-11-21**
"""

from HoloLoom.permaculture.plant_database import (
    Plant,
    GuildRole,
    CompanionRelationship,
    Guild,
    PlantDatabase,
    Layer,
    Function,
    RelationshipType,
)

from HoloLoom.permaculture.guild_engine import GuildRecommendationEngine

__all__ = [
    "Plant",
    "GuildRole",
    "CompanionRelationship",
    "Guild",
    "PlantDatabase",
    "Layer",
    "Function",
    "RelationshipType",
    "GuildRecommendationEngine",
]
