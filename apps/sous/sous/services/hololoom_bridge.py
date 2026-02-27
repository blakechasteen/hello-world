#!/usr/bin/env python3
"""
HoloLoom Bridge for SOUS

Integrates SOUS with HoloLoom's:
- Unified memory system for cooking knowledge
- RAG system for recipe suggestions
- Agentic reasoning for meal planning
- Voice assistant for hands-free cooking
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add HoloLoom to path
hololoom_path = Path(__file__).parent.parent.parent.parent / "hololoom"
if hololoom_path.exists():
    sys.path.insert(0, str(hololoom_path.parent))

try:
    from hololoom import hololoom
    from hololoom.rag import SimpleRAG
    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
    from hololoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("⚠️  HoloLoom not available - running in basic mode")


class SousHoloLoomBridge:
    """
    Bridge between SOUS kitchen management and HoloLoom AI system.

    Provides:
    - Memory storage for cooking experiences (multi-user aware)
    - AI-powered recipe suggestions
    - Intelligent meal planning
    - Voice-controlled cooking assistance

    Multi-User Features:
    - User-specific memory tagging
    - Household-level memory sharing
    - Personalized recommendations based on user history
    """

    def __init__(
        self,
        household_id: Optional[str] = None,
        enable_rag: bool = True,
        enable_agentic: bool = True
    ):
        """
        Initialize HoloLoom integration.

        Args:
            household_id: Optional household ID for multi-user support
            enable_rag: Enable RAG system
            enable_agentic: Enable agentic reasoning
        """
        self.available = HOLOLOOM_AVAILABLE
        self.enable_rag = enable_rag and HOLOLOOM_AVAILABLE
        self.enable_agentic = enable_agentic and HOLOLOOM_AVAILABLE

        self.household_id = household_id

        self.loom = None
        self.rag = None
        self.agent = None

        if not self.available:
            return

        # Initialize components
        self.config = Config.fast()  # Use FAST mode for good balance

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.available:
            return self

        # Initialize HoloLoom memory system
        self.loom = HoloLoom(config=self.config)
        await self.loom.__aenter__()

        # Initialize RAG if enabled
        if self.enable_rag:
            self.rag = SimpleRAG(config=self.config)
            await self.rag.__aenter__()

        # Initialize agentic orchestrator if enabled
        if self.enable_agentic:
            from hololoom.memory.backend_factory import create_memory_backend
            memory = await create_memory_backend(self.config)
            shards = []  # Will be populated as we learn

            self.agent = await create_agentic_orchestrator(
                self.config,
                shards,
                memory=memory
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.loom:
            await self.loom.__aexit__(exc_type, exc_val, exc_tb)
        if self.rag:
            await self.rag.__aexit__(exc_type, exc_val, exc_tb)
        if self.agent:
            await self.agent.__aexit__(exc_type, exc_val, exc_tb)

    # ========================================================================
    # Memory Integration - Store cooking experiences and knowledge
    # ========================================================================

    async def remember_recipe(
        self,
        recipe: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a recipe in HoloLoom's memory system.

        Args:
            recipe: Recipe data dictionary
            user_id: Optional user ID who added/cooked the recipe

        Returns:
            Memory metadata including ID and embedding
        """
        if not self.loom:
            return {"status": "skipped", "reason": "HoloLoom not available"}

        # Create rich memory from recipe with user context
        memory_text = self._recipe_to_memory_text(recipe, user_id)

        # Store in HoloLoom
        memory = await self.loom.experience(memory_text)

        return {
            "status": "success",
            "memory_id": memory.id,
            "text": memory_text[:100] + "..." if len(memory_text) > 100 else memory_text,
            "user_id": user_id,
            "household_id": self.household_id
        }

    async def remember_cooking_session(
        self,
        recipes_made: List[str],
        success: bool,
        notes: str = "",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a cooking session experience.

        Args:
            recipes_made: List of recipe names
            success: Whether cooking went well
            notes: Additional notes
            user_id: Optional user ID who completed the session

        Returns:
            Memory metadata
        """
        if not self.loom:
            return {"status": "skipped"}

        outcome = "successfully" if success else "with some challenges"
        user_context = f"[User: {user_id}] " if user_id else ""
        household_context = f"[Household: {self.household_id}] " if self.household_id else ""

        memory_text = (
            f"{household_context}{user_context}"
            f"Cooking session: Made {', '.join(recipes_made)} {outcome}. {notes}"
        )

        memory = await self.loom.experience(memory_text)

        # Add reflection if there were challenges
        if not success and notes:
            await self.loom.reflect(
                [memory],
                feedback={"helpful": False, "notes": notes}
            )

        return {
            "status": "success",
            "memory_id": memory.id,
            "learned": not success,  # We learn more from failures
            "user_id": user_id,
            "household_id": self.household_id
        }

    async def remember_ingredient_substitution(
        self,
        original: str,
        substitute: str,
        recipe: str,
        worked_well: bool,
        notes: str = "",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store ingredient substitution experience.

        Args:
            original: Original ingredient
            substitute: Substitute used
            recipe: Recipe name
            worked_well: Whether substitution worked
            notes: Additional notes
            user_id: Optional user ID who tried the substitution

        Returns:
            Memory metadata
        """
        if not self.loom:
            return {"status": "skipped"}

        outcome = "worked well" if worked_well else "didn't work well"
        user_context = f"[User: {user_id}] " if user_id else ""

        memory_text = (
            f"{user_context}Substitution: Used {substitute} instead of {original} "
            f"in {recipe}. It {outcome}. {notes}"
        )

        memory = await self.loom.experience(memory_text)

        return {
            "status": "success",
            "memory_id": memory.id,
            "substitution": f"{original} → {substitute}",
            "success": worked_well,
            "user_id": user_id,
            "household_id": self.household_id
        }

    # ========================================================================
    # RAG Integration - AI-powered recipe suggestions
    # ========================================================================

    async def suggest_recipes_for_ingredients(
        self,
        ingredients: List[str],
        dietary_restrictions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Suggest recipes based on available ingredients.

        Args:
            ingredients: List of ingredients you have
            dietary_restrictions: Optional dietary restrictions

        Returns:
            Recipe suggestions with confidence scores
        """
        if not self.rag:
            return {"status": "skipped", "suggestions": []}

        # Build query
        query = f"What recipes can I make with: {', '.join(ingredients)}?"
        if dietary_restrictions:
            query += f" (Dietary restrictions: {', '.join(dietary_restrictions)})"

        # Query RAG system
        result = await self.rag.query(query, mode="research")

        return {
            "status": "success",
            "query": query,
            "response": result.response,
            "confidence": result.confidence,
            "sources": result.sources
        }

    async def find_recipe_by_description(
        self,
        description: str
    ) -> Dict[str, Any]:
        """
        Find recipes matching a description.

        Args:
            description: What kind of recipe you're looking for

        Returns:
            Matching recipes with relevance scores
        """
        if not self.rag:
            return {"status": "skipped"}

        result = await self.rag.query(
            f"Find recipes for: {description}",
            mode="research"
        )

        return {
            "status": "success",
            "description": description,
            "response": result.response,
            "confidence": result.confidence,
            "reasoning_mode": result.reasoning_mode
        }

    async def suggest_substitutions(
        self,
        missing_ingredient: str,
        recipe_context: str = ""
    ) -> Dict[str, Any]:
        """
        Suggest ingredient substitutions.

        Args:
            missing_ingredient: Ingredient you don't have
            recipe_context: Recipe it's for (optional)

        Returns:
            Substitution suggestions with explanations
        """
        if not self.rag:
            return {"status": "skipped"}

        query = f"What can I substitute for {missing_ingredient}?"
        if recipe_context:
            query += f" (in {recipe_context})"

        result = await self.rag.query(query, mode="verify")

        return {
            "status": "success",
            "missing": missing_ingredient,
            "response": result.response,
            "confidence": result.confidence,
            "verified": result.verification.verified if hasattr(result, 'verification') else None
        }

    # ========================================================================
    # Agentic Reasoning - Intelligent meal planning
    # ========================================================================

    async def plan_meals_for_week(
        self,
        dietary_preferences: Optional[List[str]] = None,
        budget: Optional[float] = None,
        time_constraints: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent meal plan for the week.

        Args:
            dietary_preferences: Optional dietary preferences
            budget: Optional budget constraint
            time_constraints: Optional time limits per day (in minutes)

        Returns:
            Comprehensive meal plan with reasoning
        """
        if not self.agent:
            return {"status": "skipped", "plan": None}

        # Build query
        query = "Plan a week of meals"
        if dietary_preferences:
            query += f" with these preferences: {', '.join(dietary_preferences)}"
        if budget:
            query += f" within ${budget:.2f} budget"
        if time_constraints:
            avg_time = sum(time_constraints.values()) / len(time_constraints)
            query += f" with average {avg_time:.0f} minutes cooking time per day"

        # Use research mode for comprehensive planning
        result = await self.agent.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=5
        )

        return {
            "status": "success",
            "query": query,
            "plan": result.response,
            "confidence": result.confidence,
            "reasoning_steps": len(result.steps_taken),
            "considerations": self._extract_considerations(result)
        }

    async def optimize_shopping_list(
        self,
        shopping_list: List[str],
        pantry_items: List[str],
        local_stores: List[str]
    ) -> Dict[str, Any]:
        """
        Optimize shopping list using AI reasoning.

        Args:
            shopping_list: Items to buy
            pantry_items: What you already have
            local_stores: Available stores

        Returns:
            Optimized shopping strategy
        """
        if not self.agent:
            return {"status": "skipped"}

        query = (
            f"Optimize this shopping list: {', '.join(shopping_list[:10])}{'...' if len(shopping_list) > 10 else ''}. "
            f"I have: {', '.join(pantry_items[:5])}{'...' if len(pantry_items) > 5 else ''}. "
            f"Stores: {', '.join(local_stores)}"
        )

        result = await self.agent.reason(
            query,
            mode=ReasoningMode.PLAN_EXECUTE,
            max_steps=3
        )

        return {
            "status": "success",
            "original_count": len(shopping_list),
            "strategy": result.response,
            "confidence": result.confidence
        }

    async def solve_cooking_problem(
        self,
        problem: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a cooking problem using agentic reasoning.

        Args:
            problem: The problem to solve
            context: Optional context

        Returns:
            Solution with reasoning steps
        """
        if not self.agent:
            return {"status": "skipped"}

        query = problem
        if context:
            query += f" (Context: {context})"

        result = await self.agent.reason(
            query,
            mode=ReasoningMode.RESEARCH,
            max_steps=4
        )

        return {
            "status": "success",
            "problem": problem,
            "solution": result.response,
            "confidence": result.confidence,
            "steps": [step.get('type') for step in result.steps_taken]
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _recipe_to_memory_text(
        self,
        recipe: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Convert recipe to rich memory text with user context"""
        parts = []

        # Add user context if provided
        if user_id:
            parts.append(f"[User: {user_id}]")

        # Add household context if available
        if self.household_id:
            parts.append(f"[Household: {self.household_id}]")

        parts.append(f"Recipe: {recipe.get('name', 'Unknown')}")

        if 'description' in recipe:
            parts.append(recipe['description'])

        if 'prep_time' in recipe:
            parts.append(f"Prep time: {recipe['prep_time']} minutes")

        if 'ingredients' in recipe:
            parts.append(f"Ingredients: {', '.join(recipe['ingredients'][:5])}")

        if 'tags' in recipe:
            parts.append(f"Tags: {', '.join(recipe['tags'])}")

        return ". ".join(parts)

    def _extract_considerations(self, result: Any) -> List[str]:
        """Extract key considerations from reasoning result"""
        considerations = []

        if hasattr(result, 'steps_taken'):
            for step in result.steps_taken:
                if 'consideration' in step:
                    considerations.append(step['consideration'])

        return considerations[:5]  # Top 5

    # ========================================================================
    # Status and Metrics
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current status of HoloLoom integration"""
        return {
            "available": self.available,
            "memory_active": self.loom is not None,
            "rag_active": self.rag is not None,
            "agentic_active": self.agent is not None,
            "timestamp": datetime.now().isoformat()
        }

    async def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory system metrics"""
        if not self.loom:
            return {"status": "unavailable"}

        metrics = self.loom.get_metrics()

        return {
            "status": "active",
            "activation": metrics.get('activation', {}),
            "coherence": metrics.get('coherence', {}),
            "temporal": metrics.get('temporal', {})
        }


# Convenience functions for quick integration
async def remember_cooking_experience(
    description: str,
    success: bool = True
) -> Dict[str, Any]:
    """Quick function to remember a cooking experience"""
    async with SousHoloLoomBridge() as bridge:
        if not bridge.loom:
            return {"status": "unavailable"}

        memory = await bridge.loom.experience(description)
        return {
            "status": "success",
            "memory_id": memory.id
        }


async def ask_cooking_question(question: str) -> Dict[str, Any]:
    """Quick function to ask a cooking question"""
    async with SousHoloLoomBridge(enable_rag=True) as bridge:
        if not bridge.rag:
            return {"status": "unavailable", "answer": "HoloLoom not available"}

        result = await bridge.rag.query(question)
        return {
            "status": "success",
            "question": question,
            "answer": result.response,
            "confidence": result.confidence
        }


async def plan_meal_intelligently(
    requirements: str,
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Quick function to plan a meal"""
    async with SousHoloLoomBridge(enable_agentic=True) as bridge:
        if not bridge.agent:
            return {"status": "unavailable"}

        query = requirements
        if constraints:
            query += f" with constraints: {constraints}"

        result = await bridge.agent.reason(
            query,
            mode=ReasoningMode.PLAN_EXECUTE,
            max_steps=4
        )

        return {
            "status": "success",
            "requirements": requirements,
            "plan": result.response,
            "confidence": result.confidence
        }
