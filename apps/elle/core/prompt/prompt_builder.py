"""Prompt building: context + symbols → full LLM prompt."""

from typing import Optional, List, TYPE_CHECKING, Dict
from pathlib import Path
import logging

from ...domain import ElleRequest

if TYPE_CHECKING:
    from ...memory import MemorySnapshot

# Import HoloLoom prompt refinement (graceful degradation if not available)
try:
    from hololoom.prompting.metaprompt import create_metaprompt_auto
    from hololoom.config import Config
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False


class PromptBuilder:
    """
    Builds the full prompt for Elle's LLM.
    
    Combines:
    - Base prompt (myth + rules)
    - Current scene context
    - User state
    - Memory snapshot
    - Optional symbol snippets
    """
    
    def __init__(
        self,
        base_prompt_path: Optional[Path] = None,
        enable_refinement: bool = False,
        refinement_provider: str = "anthropic",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize with path to base prompt.

        Args:
            base_prompt_path: Path to base_prompt.txt (defaults to same directory)
            enable_refinement: Enable HoloLoom prompt refinement (+30% quality)
            refinement_provider: LLM provider for refinement ("anthropic", "google", "openai")
            logger: Optional logger instance
        """
        if base_prompt_path is None:
            base_prompt_path = Path(__file__).parent / "base_prompt.txt"

        self.base_prompt_path = base_prompt_path
        self.enable_refinement = enable_refinement and REFINEMENT_AVAILABLE
        self.refinement_provider = refinement_provider
        self.logger = logger or logging.getLogger(__name__)

        # Caches
        self._base_prompt_cache: Optional[str] = None
        self._refined_prompt_cache: Dict[str, str] = {}  # key → refined prompt

        # Warn if refinement requested but unavailable
        if enable_refinement and not REFINEMENT_AVAILABLE:
            self.logger.warning(
                "Prompt refinement requested but HoloLoom.prompting.metaprompt "
                "not available. Falling back to standard prompts."
            )
    
    @property
    def base_prompt(self) -> str:
        """Load base prompt (cached)."""
        if self._base_prompt_cache is None:
            self._base_prompt_cache = self.base_prompt_path.read_text()
        return self._base_prompt_cache
    
    def build(
        self,
        request: ElleRequest,
        memory_snapshot: 'MemorySnapshot',
        symbol_names: Optional[List[str]] = None,
        refine_this_prompt: Optional[bool] = None,
    ) -> str:
        """
        Build complete prompt for LLM.

        Args:
            request: The current request with scene + intent + user
            memory_snapshot: Recent history and patterns
            symbol_names: Optional list of symbols to include (e.g., ["chimborazo"])
            refine_this_prompt: Override instance-level refinement setting for this call

        Returns:
            Complete prompt string ready for LLM
        """

        # Build standard prompt
        parts = [
            self.base_prompt,
            "",
            "---",
            "",
            "## Current Context",
            "",
            self._format_scene(request.scene),
            "",
            self._format_intent(request.intent),
            "",
            self._format_user(request.user),
            "",
            self._format_memory(memory_snapshot),
        ]

        # Add conversation history if available in request metadata
        conversation_history = request.metadata.get('conversation_history') if hasattr(request, 'metadata') else None
        if conversation_history and conversation_history != "(No conversation history yet)":
            parts.extend([
                "",
                "---",
                "",
                "## Recent Conversation",
                "",
                conversation_history,
                "",
            ])

        # Add symbols if requested
        if symbol_names:
            parts.extend([
                "",
                "---",
                "",
                "## Relevant Symbols",
                "",
            ])
            for name in symbol_names:
                symbol_text = self._load_symbol(name)
                if symbol_text:
                    parts.append(symbol_text)
                    parts.append("")

        # Add RAG context if available in request metadata
        rag_context = request.metadata.get('rag_context') if hasattr(request, 'metadata') else None
        if rag_context:
            parts.extend([
                "",
                "---",
                "",
                "## Relevant Knowledge",
                "",
                rag_context,
                "",
            ])

        parts.extend([
            "",
            "---",
            "",
            "## Your Response",
            "",
            "Based on the above, return your decision as JSON following the format specified in the base prompt.",
        ])

        standard_prompt = "\n".join(parts)

        # Determine if refinement should be applied
        should_refine = refine_this_prompt if refine_this_prompt is not None else self.enable_refinement

        if not should_refine:
            return standard_prompt

        # Refine prompt using HoloLoom metaprompt system
        return self._refine_prompt(standard_prompt)
    
    def _format_scene(self, scene) -> str:
        """Format scene snapshot for prompt."""
        lines = [
            "### Scene",
            f"- Location: {scene.location}",
            f"- Time: {scene.time_of_day or 'unknown'}",
            f"- Weather: {scene.weather or 'unknown'}",
            f"- Objects: {scene.object_count}",
        ]
        
        if scene.tags:
            lines.append(f"- Tags: {', '.join(scene.tags)}")
        
        if scene.summary:
            lines.append(f"- Summary: {scene.summary}")
        
        # List objects
        if scene.objects:
            lines.append("")
            lines.append("Objects in scene:")
            for obj in scene.objects[:10]:  # Limit to 10 for prompt size
                condition = f" ({obj.condition})" if obj.condition else ""
                lines.append(f"  - {obj.name}{condition} @ {obj.location}")
        
        return "\n".join(lines)
    
    def _format_intent(self, intent) -> str:
        """Format user intent for prompt."""
        lines = [
            "### User Intent",
            f"- Mode: {intent.mode.value}",
            f"- Scan type: {intent.scan_type.value}",
            f"- Tired: {intent.is_tired}",
            f"- Rushed: {intent.is_rushed}",
            f"- Exploring: {intent.is_exploring}",
        ]
        
        if intent.explicit_request:
            lines.append(f"- Request: \"{intent.explicit_request}\"")
        
        return "\n".join(lines)
    
    def _format_user(self, user) -> str:
        """Format user state for prompt."""
        lines = [
            "### User State",
            f"- Name: {user.name}",
            f"- Energy: {user.current_energy_level}",
            f"- Preferred pace: {user.preferred_pace}",
        ]
        
        if user.time_available:
            lines.append(f"- Time available: {user.time_available}")
        
        if user.current_projects:
            lines.append(f"- Active projects: {', '.join(user.current_projects)}")
        
        return "\n".join(lines)
    
    def _format_memory(self, memory_snapshot) -> str:
        """Format memory snapshot for prompt."""
        # TODO: Implement when MemorySnapshot is defined
        return "### Memory\n(not yet implemented)"
    
    def _load_symbol(self, name: str) -> Optional[str]:
        """Load a symbol text by name."""
        symbols_dir = Path(__file__).parent.parent.parent / "symbols"
        symbol_path = symbols_dir / f"{name}.txt"

        if symbol_path.exists():
            return symbol_path.read_text()

        return None

    def _refine_prompt(self, standard_prompt: str) -> str:
        """
        Refine prompt using HoloLoom's 7-component metaprompt framework.

        Applies:
        - ROLE: Expert AR guide perspective
        - OBJECTIVE: Clear decision goals
        - PROCESS: Step-by-step reasoning
        - FORMAT: Structured output (preserves JSON)
        - CONSTRAINTS: Anti-patterns to avoid
        - UNCERTAINTY: Fallback when info incomplete
        - VALIDATION: Success criteria

        Args:
            standard_prompt: The standard Elle prompt to refine

        Returns:
            Refined prompt with +20-30% quality improvement
        """

        # Generate cache key from prompt content
        import hashlib
        cache_key = hashlib.md5(standard_prompt.encode()).hexdigest()

        # Return cached refinement if available
        if cache_key in self._refined_prompt_cache:
            self.logger.debug("Using cached refined prompt")
            return self._refined_prompt_cache[cache_key]

        # Refine using HoloLoom metaprompt system
        try:
            self.logger.info(
                f"Refining prompt using HoloLoom (provider: {self.refinement_provider})"
            )

            # Create temporary config
            config = Config.fast()
            config.llm_provider = self.refinement_provider

            # Apply 7-component refinement with explicit JSON preservation
            refinement_instructions = (
                f"{standard_prompt}\n\n"
                "IMPORTANT: Preserve the JSON response format exactly. "
                "The refined prompt MUST still instruct the LLM to return valid JSON "
                "matching the schema in the base prompt."
            )

            refined = create_metaprompt_auto(
                request=refinement_instructions,
                config=config,
                confidence_threshold=0.7
            )

            # Cache the refined prompt
            self._refined_prompt_cache[cache_key] = refined

            self.logger.info(
                f"Prompt refined: {len(standard_prompt)} → {len(refined)} chars "
                f"({round(len(refined) / len(standard_prompt), 1)}x expansion)"
            )

            return refined

        except Exception as e:
            self.logger.error(
                f"Prompt refinement failed: {e}. Falling back to standard prompt.",
                exc_info=True
            )
            # Graceful fallback
            return standard_prompt
