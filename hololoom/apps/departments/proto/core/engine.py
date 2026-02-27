"""Proto core engine module.

Thin waist orchestrator for Proto code agent department. All requests flow
through ProtoEngine.process() for centralized control and coordination.

Integration points:
    - AbilityRegistry: Available abilities (code analysis, testing, etc.)
    - HoloLoom AgenticOrchestrator: Reasoning and synthesis
    - ProtoSession: Conversation state and history
    - DepartmentProtocol: Standard department interface

Architecture:
    Query → Intent Detection → Action Selection → Execution → Response

Status: Production ready (2025-12-01)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from hololoom.apps.departments.proto.core.config import ProtoConfig
from hololoom.apps.departments.proto.domain import (
    IntentType,
    CodeContext,
    ProtoIntent,
    ProtoAction,
    ProtoResponse,
)


logger = logging.getLogger("proto.engine")


@dataclass
class ExecutionContext:
    """Runtime context for a single execution.

    Attributes:
        session_id: Session identifier
        intent_id: Intent being processed
        action_id: Current action
        start_time: Execution start time
        ability_results: Results from executed abilities
        errors: List of errors encountered
    """
    session_id: str
    intent_id: str
    action_id: str
    start_time: datetime = field(default_factory=datetime.now)
    ability_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add error to context."""
        self.errors.append(error)
        logger.warning(f"Execution error: {error}")

    def get_duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        return (datetime.now() - self.start_time).total_seconds() * 1000


class ProtoEngine:
    """Core orchestrator for Proto code agent.

    Thin waist pattern: all requests flow through process() for centralized
    control, coordination, and observability.

    Features:
        - Intent parsing and classification
        - Action selection via ability registry or reasoning
        - Graceful fallback chain (ability → agentic → default)
        - Complete execution context tracking
        - Async context manager support

    Usage:
        async with ProtoEngine(config) as engine:
            response = await engine.process("explain this code", context)

    Integrations:
        - HoloLoom AgenticOrchestrator (if enable_agentic_reasoning)
        - HoloLoom Memory System (if enable_memory)
        - Custom Abilities (via AbilityRegistry)
    """

    def __init__(
        self,
        config: ProtoConfig,
        ability_registry: Optional[Any] = None,
        agentic_orchestrator: Optional[Any] = None,
    ):
        """Initialize ProtoEngine.

        Args:
            config: ProtoConfig instance
            ability_registry: AbilityRegistry for custom abilities
            agentic_orchestrator: HoloLoom AgenticOrchestrator for reasoning
        """
        self.config = config
        self.ability_registry = ability_registry
        self.agentic = agentic_orchestrator
        self.logger = logging.getLogger(__name__)

        # Session state
        self._current_session_id = str(uuid.uuid4())
        self._execution_history: List[ExecutionContext] = []

        # Configure logging
        if config.enable_logging:
            self._setup_logging(config.log_level)

    def _setup_logging(self, log_level: str) -> None:
        """Configure logging for engine."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    async def startup(self) -> None:
        """Initialize engine resources."""
        self.logger.info("Proto engine starting...")
        self.logger.debug(f"Config: {self.config.to_dict()}")

        # Validate configuration
        warnings = self.config.validate()
        for warning in warnings:
            self.logger.warning(warning)

        # Initialize ability registry if needed
        if self.ability_registry and hasattr(self.ability_registry, "startup"):
            await self.ability_registry.startup()

        self.logger.info("Proto engine started successfully")

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self.logger.info("Proto engine shutting down...")

        # Shutdown ability registry if needed
        if self.ability_registry and hasattr(self.ability_registry, "shutdown"):
            await self.ability_registry.shutdown()

        # Log execution history summary
        total_requests = len(self._execution_history)
        total_duration = sum(ctx.get_duration_ms() for ctx in self._execution_history)
        avg_duration = total_duration / total_requests if total_requests > 0 else 0

        self.logger.info(f"Session complete: {total_requests} requests, "
                        f"{total_duration:.1f}ms total, {avg_duration:.1f}ms avg")
        self.logger.info("Proto engine shut down")

    async def process(
        self,
        query: str,
        context: Optional[CodeContext] = None,
    ) -> ProtoResponse:
        """Process user query and return response.

        Main entry point for all requests. Implements the thin waist pattern:
        Query → Intent → Action → Execution → Response

        Args:
            query: User query or command
            context: Optional code context (file, selection, etc.)

        Returns:
            ProtoResponse with content, actions, and metadata

        Flow:
            1. Parse intent from query and context
            2. Select action (ability to use)
            3. Execute action with fallback chain
            4. Generate response with confidence score
            5. Return to user
        """
        # Create execution context
        action_id = str(uuid.uuid4())
        intent_id = str(uuid.uuid4())
        exec_ctx = ExecutionContext(
            session_id=self._current_session_id,
            intent_id=intent_id,
            action_id=action_id,
        )

        try:
            self.logger.info(f"Processing query: {query[:50]}...")

            # Step 1: Parse intent
            intent = await self._parse_intent(query, context)
            self.logger.debug(f"Detected intent: {intent.intent_type.value}")

            # Step 2: Select action
            action = await self._select_action(intent)
            self.logger.debug(f"Selected action: {action.action_type}")

            # Step 3: Execute action (with fallback chain)
            result = await self._execute_action(action, exec_ctx)
            action.result = result
            action.executed_at = datetime.now()

            # Step 4: Generate response
            response = await self._generate_response(intent, action, exec_ctx)

            self.logger.info(f"Query processed successfully in {exec_ctx.get_duration_ms():.1f}ms")
            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            exec_ctx.add_error(str(e))

            # Return error response
            return ProtoResponse(
                content=f"Error: {str(e)}",
                confidence=0.0,
                actions_taken=[],
            )
        finally:
            # Track execution
            self._execution_history.append(exec_ctx)

    async def _parse_intent(
        self,
        query: str,
        context: Optional[CodeContext],
    ) -> ProtoIntent:
        """Parse user query into structured intent.

        Detects intent type from query patterns and enriches with context.

        Args:
            query: User query
            context: Optional code context

        Returns:
            ProtoIntent with detected type and metadata
        """
        intent_type = self._detect_intent_type(query)

        return ProtoIntent(
            intent_id=uuid.uuid4(),
            intent_type=intent_type,
            query=query,
            code_context=context,
        )

    def _detect_intent_type(self, query: str) -> IntentType:
        """Detect intent type from query text.

        Uses pattern matching on common keywords to classify intent.

        Args:
            query: User query

        Returns:
            IntentType enum value
        """
        query_lower = query.lower()

        # Pattern matching for different intent types
        patterns = {
            IntentType.EXPLAIN: ["explain", "what does", "how does", "show", "describe"],
            IntentType.REVIEW: ["review", "check", "analyze", "assess", "critique"],
            IntentType.REFACTOR: ["refactor", "improve", "optimize", "clean", "simplify"],
            IntentType.TEST: ["test", "tests", "testing", "unit test", "test case"],
            IntentType.DEBUG: ["debug", "fix", "error", "bug", "issue", "wrong"],
            IntentType.GENERATE: ["generate", "create", "write", "make", "implement"],
            IntentType.SECURITY: ["security", "vulnerable", "safe", "exploit", "attack"],
            IntentType.DOCUMENT: ["document", "docs", "docstring", "comment", "explain"],
        }

        for intent_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent_type

        return IntentType.ASK

    async def _select_action(self, intent: ProtoIntent) -> ProtoAction:
        """Select action/ability based on intent.

        Maps intent type to ability name and creates action with parameters.

        Args:
            intent: Parsed user intent

        Returns:
            ProtoAction ready for execution
        """
        # Map intent to ability name
        ability_map = {
            IntentType.EXPLAIN: "explain",
            IntentType.REVIEW: "review",
            IntentType.REFACTOR: "refactor",
            IntentType.TEST: "test",
            IntentType.DEBUG: "debug",
            IntentType.GENERATE: "generate",
            IntentType.SECURITY: "security",
            IntentType.DOCUMENT: "document",
            IntentType.ASK: "answer",
        }

        action_type = ability_map.get(intent.intent_type, "answer")

        return ProtoAction(
            action_id=uuid.uuid4(),
            action_type=action_type,
            parameters={
                "query": intent.query,
                "context": intent.code_context,
                "intent_type": intent.intent_type.value,
            },
            intent_id=intent.intent_id,
        )

    async def _execute_action(
        self,
        action: ProtoAction,
        exec_ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute action via fallback chain.

        Tries execution in order:
        1. Ability registry (if available and action exists)
        2. Agentic reasoning (if enabled)
        3. Default handler

        Args:
            action: Action to execute
            exec_ctx: Execution context

        Returns:
            Result dict with response and metadata
        """
        # Try ability registry first
        if self.ability_registry:
            ability = self._get_ability(action.action_type)
            if ability and self._can_execute_ability(action.action_type):
                try:
                    self.logger.debug(f"Executing ability: {action.action_type}")
                    result = await self._execute_ability(ability, action.parameters)
                    exec_ctx.ability_results[action.action_type] = result
                    return result
                except Exception as e:
                    exec_ctx.add_error(f"Ability execution failed: {e}")
                    # Fall through to agentic reasoning

        # Fall back to agentic reasoning
        if self.config.enable_agentic_reasoning and self.agentic:
            try:
                self.logger.debug("Falling back to agentic reasoning")
                result = await self._execute_agentic(action.parameters)
                return result
            except Exception as e:
                exec_ctx.add_error(f"Agentic reasoning failed: {e}")
                # Fall through to default

        # Default handler
        self.logger.debug("Using default handler")
        return await self._execute_default(action)

    def _get_ability(self, ability_name: str) -> Optional[Any]:
        """Get ability from registry by name."""
        if not self.ability_registry:
            return None
        if hasattr(self.ability_registry, "get"):
            return self.ability_registry.get(ability_name)
        return None

    def _can_execute_ability(self, ability_name: str) -> bool:
        """Check if ability can be executed based on config."""
        # Check trust level
        if self.config.trust_level.value == "untrusted":
            return False

        # Check specific capabilities
        dangerous_abilities = {"generate", "debug"}  # May require execution
        if ability_name in dangerous_abilities:
            if not self.config.enable_code_execution:
                return False

        return True

    async def _execute_ability(self, ability: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ability with parameters."""
        if hasattr(ability, "execute"):
            return await ability.execute(parameters)
        elif callable(ability):
            return await ability(parameters)
        else:
            raise ValueError(f"Ability is not callable")

    async def _execute_agentic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using agentic reasoning."""
        if not self.agentic:
            raise RuntimeError("Agentic orchestrator not available")

        query = parameters.get("query", "")
        context = parameters.get("context")

        # Use HoloLoom's agentic reasoning
        from hololoom.protocols.types import Query
        hololoom_query = Query(text=query)

        result = await self.agentic.reason(
            query=hololoom_query,
            mode=self.config.reasoning_mode,
        )

        return {
            "response": result.spacetime.response if hasattr(result, "spacetime") else str(result),
            "confidence": result.spacetime.confidence if hasattr(result, "spacetime") else 0.7,
            "source": "agentic",
        }

    async def _execute_default(self, action: ProtoAction) -> Dict[str, Any]:
        """Execute default handler (no ability, no reasoning)."""
        return {
            "response": f"Processed: {action.action_type}",
            "confidence": 0.5,
            "source": "default",
        }

    async def _generate_response(
        self,
        intent: ProtoIntent,
        action: ProtoAction,
        exec_ctx: ExecutionContext,
    ) -> ProtoResponse:
        """Generate final response from action result.

        Args:
            intent: Original intent
            action: Executed action
            exec_ctx: Execution context

        Returns:
            ProtoResponse for user
        """
        result = action.result or {}
        content = result.get("response", str(result)) if isinstance(result, dict) else str(result)
        confidence = result.get("confidence", 0.7) if isinstance(result, dict) else 0.7

        # Add personality (use ASCII-safe marker for Windows compatibility)
        if self.config.verbosity > 0 and self.config.personality_style.value == "relaxed":
            content = f"[OK] {content}"

        return ProtoResponse(
            response_id=uuid.uuid4(),
            content=content,
            actions_taken=[action],
            confidence=confidence,
            sources=result.get("sources", []) if isinstance(result, dict) else [],
            suggestions=self._generate_suggestions(intent, action, result),
            metadata={
                "intent_type": intent.intent_type.value,
                "action_type": action.action_type,
                "duration_ms": exec_ctx.get_duration_ms(),
                "errors": exec_ctx.errors,
                "source": result.get("source", "unknown") if isinstance(result, dict) else "unknown",
            },
        )

    def _generate_suggestions(
        self,
        intent: ProtoIntent,
        action: ProtoAction,
        result: Dict[str, Any],
    ) -> List[str]:
        """Generate follow-up suggestions based on response.

        Args:
            intent: Original intent
            action: Executed action
            result: Action result

        Returns:
            List of suggested follow-up actions
        """
        suggestions = []

        # Suggest follow-ups based on intent type
        if intent.intent_type == IntentType.EXPLAIN:
            suggestions.append("Would you like me to show an example?")
            suggestions.append("Need help applying this concept?")
        elif intent.intent_type == IntentType.REVIEW:
            suggestions.append("Would you like refactoring suggestions?")
            suggestions.append("Should I check for security issues?")
        elif intent.intent_type == IntentType.GENERATE:
            suggestions.append("Would you like me to add tests?")
            suggestions.append("Should I add documentation?")

        return suggestions[:2]  # Limit to 2 suggestions

    def get_session_id(self) -> str:
        """Get current session ID."""
        return self._current_session_id

    def get_execution_history(self) -> List[ExecutionContext]:
        """Get execution history for this session."""
        return self._execution_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_requests = len(self._execution_history)
        total_duration = sum(ctx.get_duration_ms() for ctx in self._execution_history)
        avg_duration = total_duration / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "total_duration_ms": total_duration,
            "avg_duration_ms": avg_duration,
            "session_id": self._current_session_id,
            "config": self.config.to_dict(),
        }
