"""The thin waist: ElleEngine orchestrates the decision loop."""

from typing import Protocol, Optional, TYPE_CHECKING
from datetime import datetime
import logging

from ..domain import ElleRequest, ElleAction, Followup

if TYPE_CHECKING:
    from ..memory import MemorySnapshot


class EllePolicy(Protocol):
    """Interface for Elle's decision-making core."""
    
    def decide(self, request: ElleRequest, memory_snapshot: 'MemorySnapshot') -> ElleAction:
        """Make a decision given request and memory."""
        ...


class MemoryStore(Protocol):
    """Interface for memory persistence."""
    
    def load_memory(self, user_id: str) -> 'MemorySnapshot':
        """Load read-only memory snapshot."""
        ...
    
    def record_interaction(self, request: ElleRequest, action: ElleAction) -> None:
        """Record decision for learning."""
        ...


class ToolRegistry(Protocol):
    """Interface for dispatching follow-up actions."""
    
    def dispatch(self, followup: Followup) -> None:
        """Send follow-up action to appropriate service."""
        ...


class ElleEngine:
    """
    The orchestrator. Thin waist of the system.
    
    Responsibilities:
    1. Accept normalized ElleRequest
    2. Load memory context
    3. Call policy to decide
    4. Dispatch follow-ups
    5. Record outcome
    6. Return ElleAction
    
    Pure-ish function with injected dependencies = testable.
    """
    
    def __init__(
        self,
        policy: EllePolicy,
        memory: MemoryStore,
        tools: ToolRegistry,
        logger: Optional[logging.Logger] = None,
    ):
        self.policy = policy
        self.memory = memory
        self.tools = tools
        self.logger = logger or logging.getLogger(__name__)
    
    def handle(self, request: ElleRequest) -> ElleAction:
        """
        Main decision loop.
        
        This is the golden path: request → decision → action.
        """
        
        self.logger.info(
            f"Handling request {request.request_id} "
            f"from {request.source_adapter} "
            f"for user {request.user.user_id}"
        )
        
        try:
            # 1. Load memory context
            memory_snapshot = self.memory.load_memory(request.user.user_id)
            self.logger.debug(f"Loaded memory for user {request.user.user_id}")
            
            # 2. Make decision
            action = self.policy.decide(request, memory_snapshot)
            self.logger.info(
                f"Policy decided: mode={action.mode.value}, "
                f"has_task={action.has_task}, "
                f"has_visuals={action.has_visuals}"
            )
            
            # 3. Dispatch follow-ups
            for followup in action.followups:
                self.logger.debug(
                    f"Dispatching followup: {followup.service}.{followup.action}"
                )
                self.tools.dispatch(followup)
            
            # 4. Record outcome (for learning)
            self.memory.record_interaction(request, action)
            self.logger.debug("Recorded interaction to memory")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error handling request: {e}", exc_info=True)
            
            # Return safe fallback action
            from ..domain import ElleMode
            return ElleAction(
                mode=ElleMode.SILENT,
                utterance=None,
                reasoning=f"Error: {str(e)}"
            )
    
    def health_check(self) -> dict:
        """Basic health check for monitoring."""
        return {
            "engine": "ok",
            "timestamp": datetime.now().isoformat(),
        }
