"""Elle's decision-making policy."""

from __future__ import annotations
import json
from typing import Optional, List, TYPE_CHECKING
import logging

from .prompt import PromptBuilder
from .llm_client import LLMClient
from ..domain import ElleRequest, ElleAction, ElleMode

if TYPE_CHECKING:
    from ..memory import MemorySnapshot


class PolicyError(Exception):
    """Error in policy execution."""
    pass


class EllePolicy:
    """
    The core decision-maker.
    
    Takes request + memory → returns action.
    
    This is where the magic happens:
    1. Build prompt from context
    2. Call LLM
    3. Parse JSON response
    4. Return ElleAction
    """
    
    def __init__(
        self,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        logger: Optional[logging.Logger] = None,
    ):
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
    
    def decide(
        self,
        request: ElleRequest,
        memory_snapshot: 'MemorySnapshot',
    ) -> ElleAction:
        """
        Make a decision.
        
        Pure function: same inputs → same outputs (modulo LLM variance).
        
        Args:
            request: Current scene + intent + user
            memory_snapshot: History and patterns
        
        Returns:
            What Elle should do/say
        """
        
        self.logger.debug(
            f"Making decision for scene={request.scene.location}, "
            f"intent={request.intent.mode.value}"
        )
        
        try:
            # 1. Build prompt
            prompt = self.prompt_builder.build(
                request=request,
                memory_snapshot=memory_snapshot,
                symbol_names=self._select_symbols(request)
            )
            
            self.logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # 2. Call LLM
            response_text = self.llm_client.complete(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )
            
            self.logger.debug(f"LLM response length: {len(response_text)} chars")
            
            # 3. Parse JSON
            action = self._parse_response(response_text)
            
            self.logger.info(
                f"Decision: mode={action.mode.value}, "
                f"utterance={'yes' if action.utterance else 'no'}, "
                f"task={'yes' if action.has_task else 'no'}"
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Policy error: {e}", exc_info=True)
            
            # Fallback: silent mode
            return ElleAction(
                mode=ElleMode.SILENT,
                reasoning=f"Policy error: {str(e)}"
            )
    
    def _select_symbols(self, request: ElleRequest) -> List[str]:
        """
        Choose which symbols to include in prompt.
        
        Based on scene tags, user state, etc.
        Later: this can get smarter with ML/rules.
        """
        
        symbols = []
        
        # If scene is cluttered, include Chimborazo (context matters)
        if request.scene.has_tag("cluttered"):
            symbols.append("chimborazo")
        
        # If user is exploring, include Plato's Cave
        if request.intent.is_exploring:
            symbols.append("platos_cave")
        
        # If there's an active project, include Penelope's Hands (undoing/patience)
        if request.user.current_projects:
            symbols.append("penelope_hands")
        
        return symbols
    
    def _parse_response(self, response_text: str) -> ElleAction:
        """
        Parse LLM response into ElleAction.
        
        Expects JSON matching the format in base_prompt.txt.
        """
        
        try:
            # Try to extract JSON from response
            # (LLMs sometimes wrap it in markdown or explanation)
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            
            # Parse into ElleAction
            return self._json_to_action(data)
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse response: {e}")
            self.logger.debug(f"Raw response: {response_text}")
            raise PolicyError(f"Invalid response format: {e}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially wrapped response."""
        
        # Try to find JSON block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        
        # Try to find raw JSON
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        
        # No JSON found
        raise PolicyError("No JSON found in response")
    
    def _json_to_action(self, data: dict) -> ElleAction:
        """Convert parsed JSON dict to ElleAction."""
        
        from ..domain import (
            Visual, VisualStyle, Priority, Followup
        )
        
        # Parse mode (required)
        mode = ElleMode(data["mode"])
        
        # Parse visuals (optional)
        visuals = None
        if "visuals" in data and data["visuals"]:
            v = data["visuals"]
            visuals = Visual(
                style=VisualStyle(v["style"]),
                target_object_ids=v.get("target_object_ids", []),
                color=v.get("color"),
                animation=v.get("animation"),
                position=v.get("position"),
            )
        
        # Parse priority (optional)
        priority = None
        if "task_priority" in data and data["task_priority"]:
            priority = Priority(data["task_priority"])
        
        # Parse followups (optional)
        followups = []
        if "followups" in data:
            followups = [
                Followup(
                    service=f["service"],
                    action=f["action"],
                    params=f.get("params", {})
                )
                for f in data["followups"]
            ]
        
        return ElleAction(
            mode=mode,
            utterance=data.get("utterance"),
            visuals=visuals,
            suggested_task_id=data.get("suggested_task_id"),
            suggested_task_summary=data.get("suggested_task_summary"),
            task_priority=priority,
            task_estimate_minutes=data.get("task_estimate_minutes"),
            followups=followups,
            reasoning=data.get("reasoning"),
        )
