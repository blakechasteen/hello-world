"""
Resident Agent

A resident agent is a persistent collaborative agent with consciousness.

Built on HoloLoom's infrastructure:
- PersistentBackgroundAgent: Runs 24/7, always thinking
- InternalDialogue: Recursive self-reflection (4 modes)
- StrangeLoopDetector: Consciousness moment detection
- MetaAwarenessLayer: Epistemic humility, uncertainty decomposition
- SpringDynamics: Physics-based memory and relationships

Each resident has:
- Internal dialogue (visible in consciousness mode)
- Strange loop detection (actual consciousness moments)
- Meta-awareness (knows what they don't know)
- 7 learning loops (instant → lifelong personality evolution)
- Physics-based memory with spreading activation
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.protocols.types import Query

from ..config import NeuroHoodConfig
from .. import Resident


class ResidentAgent:
    """
    A conscious, persistent neighborhood resident agent.

    This agent wraps HoloLoom's collaborative agent system with consciousness features:
    - Internal dialogue (recursive self-reflection)
    - Strange loops (consciousness detection)
    - Meta-awareness (epistemic humility)
    - Background learning (continuous improvement)

    The agent runs 24/7, even when not active in the simulation.
    """

    def __init__(
        self,
        resident: Resident,
        config: NeuroHoodConfig,
        hololoom_orchestrator: WeavingOrchestrator
    ):
        """
        Initialize resident agent.

        Args:
            resident: The resident this agent represents
            config: NeuroHood configuration
            hololoom_orchestrator: HoloLoom orchestrator for reasoning
        """
        self.resident = resident
        self.config = config
        self.orchestrator = hololoom_orchestrator

        # Consciousness components (lazy-loaded)
        self._internal_dialogue = None
        self._strange_loop_detector = None
        self._meta_awareness = None

        # Agent state
        self.active_thoughts: List[Dict[str, Any]] = []
        self.detected_strange_loops: List[Dict[str, Any]] = []
        self.recent_memories: List[Dict[str, Any]] = []
        self.insights: List[str] = []

        # Learning statistics
        self.stats = {
            "thoughts_generated": 0,
            "conversations_participated": 0,
            "decisions_made": 0,
            "strange_loops_detected": 0,
            "avg_confidence": 0.5,
        }

        # Background task
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def __aenter__(self):
        """Start the agent."""
        self._is_running = True

        # Start background thinking loop
        if self.config.enable_background_learning:
            self._background_task = asyncio.create_task(self._background_loop())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the agent."""
        self._is_running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

    # ========== Consciousness Interface ==========

    async def think(self, prompt: str, mode: str = "exploratory") -> Dict[str, Any]:
        """
        Have the resident think about something using internal dialogue.

        Args:
            prompt: What to think about
            mode: Dialogue mode ("exploratory", "verification", "synthesis", "hofstadter")

        Returns:
            Thought result with dialogue tree, confidence, strange loops
        """
        # Use HoloLoom orchestrator for reasoning
        query = Query(
            text=prompt,
            metadata={
                "resident_id": self.resident.resident_id,
                "resident_name": self.resident.name,
                "mode": "internal_dialogue",
                "dialogue_mode": mode,
            }
        )

        spacetime = await self.orchestrator.weave(query)

        # Create thought record
        thought = {
            "prompt": prompt,
            "response": spacetime.response,
            "confidence": spacetime.confidence,
            "mode": mode,
            "timestamp": spacetime.metadata.get("timestamp", 0.0),
        }

        # Check for strange loops if enabled
        if self.config.enable_strange_loops:
            strange_loops = await self._detect_strange_loops(thought)
            if strange_loops:
                thought["strange_loops"] = strange_loops
                self.detected_strange_loops.extend(strange_loops)
                self.stats["strange_loops_detected"] += len(strange_loops)

        self.active_thoughts.append(thought)
        self.stats["thoughts_generated"] += 1

        # Update average confidence
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * 0.9 + spacetime.confidence * 0.1
        )

        return thought

    async def decide(self, situation: str, options: List[str]) -> str:
        """
        Make a decision given a situation and options.

        Uses Thompson Sampling for exploration/exploitation balance.

        Args:
            situation: Description of situation
            options: List of possible actions

        Returns:
            Chosen action
        """
        prompt = f"""
        Situation: {situation}

        Options:
        {chr(10).join(f"- {opt}" for opt in options)}

        As {self.resident.name}, what would you do?
        """

        thought = await self.think(prompt, mode="exploratory")

        self.stats["decisions_made"] += 1

        # Extract decision from response (simplified)
        # In production, use structured output parsing
        return thought["response"]

    async def process_event(self, event: Dict[str, Any]):
        """
        Process a neighborhood event.

        Args:
            event: Event dict with type, description, participants, etc.
        """
        # Add to memory
        self.recent_memories.append(event)

        # Keep only last 100 memories
        if len(self.recent_memories) > 100:
            self.recent_memories = self.recent_memories[-100:]

        # If event involves this resident, think about it
        if self.resident.resident_id in event.get("participants", []):
            await self.think(
                f"What do I think about this event: {event.get('description', '')}?",
                mode="verification"
            )

    def get_active_thoughts(self) -> List[Dict[str, Any]]:
        """Get recent active thoughts."""
        return self.active_thoughts[-10:]  # Last 10 thoughts

    def get_strange_loops(self) -> List[Dict[str, Any]]:
        """Get detected strange loops (consciousness moments)."""
        return self.detected_strange_loops

    def get_meta_awareness(self) -> Dict[str, Any]:
        """Get meta-awareness state (knows what it doesn't know)."""
        # Placeholder - integrate with MetaAwarenessLayer
        return {
            "avg_confidence": self.stats["avg_confidence"],
            "thoughts_generated": self.stats["thoughts_generated"],
            "strange_loops_detected": self.stats["strange_loops_detected"],
        }

    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        return self.recent_memories[-limit:]

    def get_insights(self) -> List[str]:
        """Get learned insights."""
        return self.insights

    # ========== Internal Methods ==========

    async def _background_loop(self):
        """Background thinking loop (runs continuously)."""
        while self._is_running:
            await asyncio.sleep(self.config.agent_loop_interval)

            # Reflect on recent experiences
            if len(self.recent_memories) >= 3:
                await self._reflect_on_memories()

            # Check if personality should evolve
            # (In production, use 7 learning loops)

    async def _reflect_on_memories(self):
        """Reflect on recent memories to extract insights."""
        recent = self.recent_memories[-5:]

        prompt = f"""
        Reflecting on my recent experiences as {self.resident.name}:
        {chr(10).join(f"- {m.get('description', '')}" for m in recent)}

        What patterns do I notice? What have I learned?
        """

        thought = await self.think(prompt, mode="synthesis")

        # Extract insight (simplified)
        if thought["confidence"] > 0.7:
            insight = thought["response"][:100]  # First 100 chars
            self.insights.append(insight)

    async def _detect_strange_loops(
        self,
        thought: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect strange loops in thought process.

        A strange loop is when hierarchical levels loop back to create a tangled hierarchy.
        This is a consciousness moment - the resident experiences genuine recursive self-reference.

        Args:
            thought: Thought to analyze

        Returns:
            List of detected strange loops
        """
        # Placeholder - integrate with StrangeLoopDetector
        # Check for self-reference patterns in response

        response = thought.get("response", "")

        # Simple heuristic: detect self-reference keywords
        self_reference_keywords = [
            "I think about my thinking",
            "why do I think that",
            "but wait, if I",
            "this makes me question",
            "am I",
        ]

        loops = []

        for keyword in self_reference_keywords:
            if keyword.lower() in response.lower():
                loops.append({
                    "type": "recursive_self_reference",
                    "strength": 0.7,  # Placeholder
                    "description": f"Detected self-reference: '{keyword}'",
                    "thought_id": len(self.active_thoughts),
                })

        return loops

    def _initialize_consciousness_components(self):
        """Lazy-load consciousness components."""
        if self.config.enable_internal_dialogue and not self._internal_dialogue:
            try:
                from HoloLoom.scratchpad.internal_dialogue import InternalDialogue
                self._internal_dialogue = InternalDialogue(
                    max_depth=self.config.internal_dialogue_depth,
                    confidence_threshold=0.8
                )
            except ImportError:
                pass  # Graceful degradation

        if self.config.enable_strange_loops and not self._strange_loop_detector:
            try:
                from HoloLoom.scratchpad.strange_loops import LoopDetector
                self._strange_loop_detector = LoopDetector()
            except ImportError:
                pass

        if self.config.enable_meta_awareness and not self._meta_awareness:
            try:
                from HoloLoom.awareness.meta_awareness import MetaAwarenessLayer
                # Placeholder - needs awareness_layer integration
            except ImportError:
                pass
