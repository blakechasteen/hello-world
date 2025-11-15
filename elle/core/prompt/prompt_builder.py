"""Prompt building: context + symbols → full LLM prompt."""

from typing import Optional, List, TYPE_CHECKING
from pathlib import Path

from ...domain import ElleRequest

if TYPE_CHECKING:
    from ...memory import MemorySnapshot


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
    
    def __init__(self, base_prompt_path: Optional[Path] = None):
        """
        Initialize with path to base prompt.
        
        If not provided, looks for base_prompt.txt in same directory.
        """
        if base_prompt_path is None:
            base_prompt_path = Path(__file__).parent / "base_prompt.txt"
        
        self.base_prompt_path = base_prompt_path
        self._base_prompt_cache: Optional[str] = None
    
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
    ) -> str:
        """
        Build complete prompt for LLM.
        
        Args:
            request: The current request with scene + intent + user
            memory_snapshot: Recent history and patterns
            symbol_names: Optional list of symbols to include (e.g., ["chimborazo"])
        
        Returns:
            Complete prompt string ready for LLM
        """
        
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
        
        parts.extend([
            "",
            "---",
            "",
            "## Your Response",
            "",
            "Based on the above, return your decision as JSON following the format specified in the base prompt.",
        ])
        
        return "\n".join(parts)
    
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
