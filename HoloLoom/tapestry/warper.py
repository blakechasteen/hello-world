"""
Warper - Tapestry Initializer

Sets up warp threads before weaving begins.

The Warper:
1. Decomposes a goal into threads (tasks)
2. Creates the initial tapestry
3. Makes initial git commit
4. Saves tapestry to backend

Combines existing patterns:
- planning_department.py (goal decomposition)
- verification/chain.py (setup validation)

Created: December 2025
"""

import logging
from typing import Optional, List, Dict, Any

from HoloLoom.tapestry.protocol import (
    Tapestry,
    TapestryBackend,
    TapestryError
)
from HoloLoom.tapestry.git import GitIntegration

logger = logging.getLogger(__name__)


class Warper:
    """
    Sets up warp threads before weaving begins.

    The Warper is responsible for:
    1. Decomposing a goal into discrete threads
    2. Creating the initial tapestry structure
    3. Recording initial git state
    4. Persisting tapestry to backend

    Uses HoloLoom's planning capabilities when available,
    falls back to simple decomposition otherwise.
    """

    def __init__(
        self,
        backend: TapestryBackend,
        git: Optional[GitIntegration] = None,
        use_llm_planning: bool = True
    ):
        """
        Initialize Warper.

        Args:
            backend: TapestryBackend for persistence
            git: GitIntegration for version control (optional)
            use_llm_planning: Whether to use LLM for goal decomposition
        """
        self.backend = backend
        self.git = git or GitIntegration()
        self.use_llm_planning = use_llm_planning
        self._planning_available = None

    async def _check_planning_available(self) -> bool:
        """Check if planning department is available."""
        if self._planning_available is not None:
            return self._planning_available

        try:
            from HoloLoom.apps.departments.planning_department import PlanningDepartment
            self._planning_available = True
        except ImportError:
            self._planning_available = False
            logger.debug("PlanningDepartment not available, using basic decomposition")

        return self._planning_available

    async def setup(
        self,
        goal: str,
        threads: Optional[List[str]] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tapestry:
        """
        Create tapestry with threads for goal.

        Process:
        1. Decompose goal → threads (if not provided)
        2. Create tapestry
        3. Initial commit (if git enabled)
        4. Save to backend

        Args:
            goal: Overall goal for this session
            threads: Optional explicit thread descriptions
            dependencies: Optional thread dependency map
            metadata: Optional additional metadata

        Returns:
            Created Tapestry
        """
        logger.info(f"Setting up tapestry for goal: {goal[:50]}...")

        # Step 1: Get thread descriptions
        if threads:
            thread_descriptions = threads
        else:
            thread_descriptions = await self._decompose_goal(goal)

        if not thread_descriptions:
            raise TapestryError("Could not decompose goal into threads")

        # Step 2: Create tapestry
        tapestry = Tapestry.create(
            goal=goal,
            thread_descriptions=thread_descriptions,
            dependencies=dependencies
        )

        # Add metadata
        if metadata:
            tapestry.metadata.update(metadata)

        # Step 3: Initial commit
        try:
            initial_commit = await self.git.get_current_commit()
            if initial_commit:
                tapestry.initial_commit = initial_commit
                tapestry.current_commit = initial_commit
                logger.debug(f"Initial commit: {initial_commit}")
        except Exception as e:
            logger.debug(f"Could not get initial commit: {e}")

        # Step 4: Save to backend
        await self.backend.save(tapestry)
        logger.info(
            f"Created tapestry '{tapestry.loom_id}' with "
            f"{len(tapestry.threads)} threads"
        )

        return tapestry

    async def _decompose_goal(self, goal: str) -> List[str]:
        """
        Decompose a goal into discrete threads.

        Uses PlanningDepartment if available, otherwise
        falls back to simple heuristic decomposition.

        Args:
            goal: Goal to decompose

        Returns:
            List of thread descriptions
        """
        # Try LLM-based planning
        if self.use_llm_planning and await self._check_planning_available():
            try:
                return await self._llm_decompose(goal)
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}, using fallback")

        # Fallback: basic heuristic decomposition
        return self._basic_decompose(goal)

    async def _llm_decompose(self, goal: str) -> List[str]:
        """Use PlanningDepartment for goal decomposition."""
        from HoloLoom.apps.departments.planning_department import PlanningDepartment

        planning = PlanningDepartment()
        result = await planning.request({
            "action": "plan",
            "goal": goal,
            "max_steps": 10
        })

        steps = result.get("steps", [])
        if steps:
            logger.debug(f"LLM decomposition: {len(steps)} steps")
            return steps

        # Fallback if planning returns empty
        return self._basic_decompose(goal)

    def _basic_decompose(self, goal: str) -> List[str]:
        """
        Basic heuristic decomposition.

        Splits goal into logical steps based on common patterns.
        """
        goal_lower = goal.lower()

        # Check for numbered list in goal
        if any(f"{i}." in goal or f"{i})" in goal for i in range(1, 10)):
            # Extract numbered items
            import re
            items = re.split(r'\d+[.)]\s*', goal)
            items = [item.strip() for item in items if item.strip()]
            if items:
                return items

        # Check for comma-separated list
        if "," in goal and len(goal.split(",")) <= 5:
            items = [item.strip() for item in goal.split(",") if item.strip()]
            if len(items) >= 2:
                return items

        # Check for common action words
        action_words = ["implement", "create", "add", "fix", "update", "remove", "test"]

        # Try to identify sub-tasks from goal structure
        if " and " in goal_lower:
            parts = goal.split(" and ")
            if 2 <= len(parts) <= 4:
                return [p.strip() for p in parts]

        # Single-task fallback with common decomposition
        threads = []

        # Understand/research phase
        if any(word in goal_lower for word in ["implement", "create", "build", "develop"]):
            threads.append(f"Research and understand requirements for: {goal}")

        # Main implementation
        threads.append(goal)

        # Testing phase
        if any(word in goal_lower for word in ["implement", "create", "build", "add", "fix"]):
            threads.append(f"Test and verify: {goal}")

        # Documentation phase (optional)
        if "document" in goal_lower or len(goal) > 50:
            threads.append(f"Document: {goal}")

        return threads if threads else [goal]

    async def resume(self) -> Optional[Tapestry]:
        """
        Resume existing tapestry from backend.

        Returns:
            Existing Tapestry or None if none exists
        """
        if not await self.backend.exists():
            return None

        tapestry = await self.backend.load()
        if tapestry:
            logger.info(
                f"Resumed tapestry '{tapestry.loom_id}': "
                f"{tapestry.get_status_summary()}"
            )
        return tapestry

    async def clear(self) -> None:
        """Delete existing tapestry."""
        await self.backend.delete()
        logger.info("Cleared existing tapestry")

    def describe(self) -> str:
        """Get description of warper configuration."""
        return (
            f"Warper(llm_planning={self.use_llm_planning}, "
            f"planning_available={self._planning_available})"
        )


async def quick_setup(
    goal: str,
    threads: Optional[List[str]] = None,
    path: str = ".hololoom/tapestry.json"
) -> Tapestry:
    """
    Quick setup helper.

    Creates a tapestry with minimal configuration.

    Args:
        goal: Overall goal
        threads: Optional explicit threads
        path: Path to tapestry file

    Returns:
        Created Tapestry
    """
    from HoloLoom.tapestry.backends.json_backend import JsonTapestryBackend

    backend = JsonTapestryBackend(path)
    warper = Warper(backend)
    return await warper.setup(goal, threads)
