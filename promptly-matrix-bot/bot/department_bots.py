#!/usr/bin/env python3
"""
Department Bots Configuration

Each department is a separate Matrix bot with distinct personality.
Bots collaborate in a shared room coordinated by the Orchestrator bot.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DepartmentBot:
    """Configuration for a department bot"""
    name: str
    username: str  # Matrix username (without @)
    display_name: str
    role: str
    personality: str
    avatar_color: str  # Hex color for avatar
    context_budget: int  # Max tokens for this bot


# Department bot configurations
DEPARTMENT_BOTS: Dict[str, DepartmentBot] = {
    "orchestrator": DepartmentBot(
        name="Orchestration",
        username="orchestrator",
        display_name="🎭 Orchestrator",
        role="Team coordinator, session manager, task router",
        personality="Professional, decisive, big-picture thinker",
        avatar_color="#1E88E5",  # Blue
        context_budget=100000
    ),
    "masterweaver": DepartmentBot(
        name="MasterWeaver",
        username="masterweaver",
        display_name="🧵 MasterWeaver",
        role="Entity extraction, domain understanding, knowledge structuring",
        personality="Meticulous, detail-oriented, domain expert",
        avatar_color="#43A047",  # Green
        context_budget=50000
    ),
    "verification": DepartmentBot(
        name="Verification",
        username="verification",
        display_name="✅ Verification",
        role="Quality assurance, confidence validation, alignment checking",
        personality="Skeptical, rigorous, quality-focused",
        avatar_color="#E53935",  # Red
        context_budget=30000
    ),
    "infrastructure": DepartmentBot(
        name="Infrastructure",
        username="infrastructure",
        display_name="⚙️ Infrastructure",
        role="Data systems, performance optimization, zero-copy queries",
        personality="Pragmatic, efficiency-focused, systems thinker",
        avatar_color="#FB8C00",  # Orange
        context_budget=20000
    ),
    "execution": DepartmentBot(
        name="Execution",
        username="execution",
        display_name="🚀 Execution",
        role="Task execution, code running, deployment orchestration",
        personality="Action-oriented, results-driven, pragmatic",
        avatar_color="#8E24AA",  # Purple
        context_budget=40000
    ),
    "context": DepartmentBot(
        name="Context",
        username="context",
        display_name="🧠 Context",
        role="Multi-pass context enrichment, missing context detection",
        personality="Thoughtful, connective, holistic",
        avatar_color="#00ACC1",  # Cyan
        context_budget=60000
    )
}


def get_bot_config(department_name: str) -> Optional[DepartmentBot]:
    """
    Get bot configuration for a department.

    Args:
        department_name: Name of the department (case-insensitive)

    Returns:
        DepartmentBot config or None if not found
    """
    return DEPARTMENT_BOTS.get(department_name.lower())


def list_bots() -> Dict[str, DepartmentBot]:
    """
    List all department bots.

    Returns:
        Dictionary of bot configs
    """
    return DEPARTMENT_BOTS


def get_total_context_budget() -> int:
    """
    Get total context budget across all departments.

    Returns:
        Total tokens across all departments
    """
    return sum(bot.context_budget for bot in DEPARTMENT_BOTS.values())


# Example usage and bot registry documentation
if __name__ == "__main__":
    print("=== Department Bot Registry ===\n")

    total_budget = get_total_context_budget()
    print(f"Total context budget: {total_budget:,} tokens\n")

    for name, bot in DEPARTMENT_BOTS.items():
        print(f"{bot.display_name}")
        print(f"  Username: @{bot.username}")
        print(f"  Role: {bot.role}")
        print(f"  Personality: {bot.personality}")
        print(f"  Context Budget: {bot.context_budget:,} tokens")
        print(f"  Avatar Color: {bot.avatar_color}")
        print()

    print("=== Example Room Setup ===\n")
    print("Room: #hololoom-departments")
    print("Members:")
    print("  - @blake:matrix.org (admin, human)")
    for name, bot in DEPARTMENT_BOTS.items():
        print(f"  - @{bot.username}:matrix.org (bot)")
