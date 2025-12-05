"""
Jenny UI Protocol Definitions
==============================
Protocol interfaces for the Jenny Generative UI Runtime.

Philosophy:
> "Disposable pixels, durable decisions."
>
> Every UI panel is temporary and should dissolve when no longer needed.
> Every decision that generated it is permanent and fully replayable.

These protocols define WHAT the Jenny system does, not HOW.
Implementations are swappable via dependency injection.

Three Core Protocols:
1. JennyCompilerProtocol - Compiles Spacetime → JennySpec[]
2. JennyRendererProtocol - Renders JennySpec → Output (HTML/React/AR)
3. JennyLifecycleProtocol - Manages NASCENT → STABLE → DISSOLVING → ARCHIVED

References:
- jenny_spec.py (JennySpec dataclass)
- fabric/spacetime.py (Spacetime input)
- alignment/audit_trail.py (ProvenanceTracer pattern)

Author: HoloLoom Team
Date: 2025-12-01 (Jenny MVP Week 1)
"""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, AsyncIterator
from enum import Enum


# ============================================================================
# Forward Type References (avoid circular imports)
# ============================================================================

# These will be properly typed when implementations import from jenny_spec.py
# Using string annotations for forward references
JennySpecType = Any  # from visualization.jenny_spec import JennySpec
SpacetimeType = Any  # from fabric.spacetime import Spacetime
LifecycleStageType = Any  # from visualization.jenny_spec import LifecycleStage
DissolutionTriggerType = Any  # from visualization.jenny_spec import DissolutionTrigger


# ============================================================================
# Compilation Strategy Enum
# ============================================================================

class CompilationStrategy(str, Enum):
    """
    Strategy for compiling Spacetime to UI panels.

    Determines how aggressively to generate panels.
    """
    MINIMAL = "minimal"      # Only essential panels (1-2 panels)
    BALANCED = "balanced"    # Standard panels based on content (2-4 panels)
    COMPREHENSIVE = "comprehensive"  # Full panel suite (4-6 panels)
    AUTO = "auto"           # Auto-detect based on query complexity


class RenderTarget(str, Enum):
    """
    Target format for rendering JennySpec.
    """
    HTML = "html"           # Static HTML (for dashboards)
    REACT = "react"         # React component props (for web apps)
    AR = "ar"               # AR overlay format (for Elle/spatial)
    JSON = "json"           # Raw JSON (for API responses)
    TERMINAL = "terminal"   # Terminal/CLI output (for debugging)


# ============================================================================
# JennyCompilerProtocol - Spacetime → JennySpec[]
# ============================================================================

@runtime_checkable
class JennyCompilerProtocol(Protocol):
    """
    Protocol for compiling Spacetime output into UI specifications.

    The compiler analyzes Spacetime (weaving output + trace) and generates
    appropriate JennySpec panels. This is the "brain" that decides:
    - Which panels to show
    - What size/type each panel should be
    - What actions to attach
    - How to layout the panels

    Philosophy:
    - Query intent determines panel types
    - Response content determines panel data
    - Trace metadata informs debugging panels
    - User context affects layout hints

    Implementations:
    - LLMJennyCompiler: Uses LLM to determine panel strategy
    - RuleBasedCompiler: Uses heuristics and rules
    - HybridCompiler: Combines LLM + rules

    Usage:
        compiler = LLMJennyCompiler(llm_client)
        specs = await compiler.compile(spacetime, strategy=CompilationStrategy.BALANCED)
        # specs is List[JennySpec] ready for rendering
    """

    async def compile(
        self,
        spacetime: SpacetimeType,
        strategy: CompilationStrategy = CompilationStrategy.AUTO,
        context: Optional[Dict[str, Any]] = None
    ) -> List[JennySpecType]:
        """
        Compile Spacetime into UI specifications.

        Args:
            spacetime: Woven output from HoloLoom weaving cycle
            strategy: How aggressively to generate panels
            context: Optional user/session context for personalization

        Returns:
            List of JennySpec panels to render

        Raises:
            CompilationError: If compilation fails
        """
        ...

    async def compile_stream(
        self,
        spacetime: SpacetimeType,
        strategy: CompilationStrategy = CompilationStrategy.AUTO,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[JennySpecType]:
        """
        Stream-compile Spacetime for progressive rendering.

        Yields panels as they are determined, enabling:
        - Progressive UI rendering
        - Early display of high-priority panels
        - Reduced time-to-first-panel

        Args:
            spacetime: Woven output from HoloLoom weaving cycle
            strategy: How aggressively to generate panels
            context: Optional user/session context

        Yields:
            JennySpec panels in priority order
        """
        ...

    def get_panel_strategy(
        self,
        spacetime: SpacetimeType
    ) -> Dict[str, Any]:
        """
        Analyze Spacetime and recommend panel strategy (dry run).

        Does not generate specs, just returns strategy recommendation.
        Useful for UI that wants to show "Jenny will generate X panels".

        Args:
            spacetime: Woven output to analyze

        Returns:
            Strategy recommendation:
            {
                "recommended_strategy": "balanced",
                "estimated_panels": 3,
                "panel_types": ["text", "graph", "sources"],
                "layout_hint": "grid",
                "reasoning": "Query is factual with graph context"
            }
        """
        ...

    @property
    def compiler_version(self) -> str:
        """Return compiler version for provenance tracking."""
        ...


# ============================================================================
# JennyRendererProtocol - JennySpec → Output Format
# ============================================================================

@runtime_checkable
class JennyRendererProtocol(Protocol):
    """
    Protocol for rendering JennySpec into output format.

    The renderer converts abstract JennySpec into concrete output:
    - HTML for static dashboards
    - React props for web applications
    - AR format for spatial computing (Elle)
    - JSON for API responses

    Philosophy:
    - Renderers are stateless (spec → output)
    - Target format determines renderer implementation
    - Animations handled by renderer (spawn, dissolve)
    - Accessibility built into all renderers

    Implementations:
    - HTMLRenderer: Generates semantic HTML + CSS
    - ReactRenderer: Generates React component props
    - ARRenderer: Generates AR overlay specifications
    - TerminalRenderer: ASCII art for debugging

    Usage:
        renderer = HTMLRenderer()
        html = await renderer.render(specs)
        # html is ready to display
    """

    async def render(
        self,
        specs: List[JennySpecType],
        target: RenderTarget = RenderTarget.HTML,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render JennySpecs to output format.

        Args:
            specs: List of JennySpec panels to render
            target: Output format (HTML, React, AR, etc.)
            options: Renderer-specific options

        Returns:
            Rendered output as string

        Raises:
            RenderError: If rendering fails
        """
        ...

    async def render_single(
        self,
        spec: JennySpecType,
        target: RenderTarget = RenderTarget.HTML,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render a single JennySpec panel.

        Convenience method for rendering one panel.

        Args:
            spec: Single JennySpec to render
            target: Output format
            options: Renderer-specific options

        Returns:
            Rendered panel as string
        """
        ...

    async def render_update(
        self,
        old_spec: JennySpecType,
        new_spec: JennySpecType,
        target: RenderTarget = RenderTarget.HTML
    ) -> str:
        """
        Render differential update between specs.

        For reactive updates, render only the diff:
        - Changed content fields
        - Lifecycle transitions (with animations)
        - Action state changes

        Args:
            old_spec: Previous panel state
            new_spec: New panel state
            target: Output format

        Returns:
            Differential update for efficient patching
        """
        ...

    def supports_target(self, target: RenderTarget) -> bool:
        """Check if renderer supports a target format."""
        ...

    @property
    def supported_targets(self) -> List[RenderTarget]:
        """Return list of supported render targets."""
        ...


# ============================================================================
# JennyLifecycleProtocol - Panel Lifecycle Management
# ============================================================================

@runtime_checkable
class JennyLifecycleProtocol(Protocol):
    """
    Protocol for managing panel lifecycle transitions.

    Handles the state machine:
        NASCENT → STABLE → DISSOLVING → ARCHIVED
              ↘           ↙
           (timeout/superseded)

    The lifecycle manager:
    - Tracks active panels across sessions
    - Handles dissolution triggers
    - Archives panels to SpecLedger
    - Manages pinned panel persistence

    Philosophy:
    - Panels are disposable by default
    - User can pin panels to make them stable
    - All transitions logged for provenance
    - Memory pressure can force dissolution

    Implementations:
    - InMemoryLifecycleManager: For single sessions
    - PersistentLifecycleManager: With database backing
    - DistributedLifecycleManager: For multi-node deployments

    Usage:
        lifecycle = PersistentLifecycleManager(spec_ledger)
        lifecycle.spawn(spec)  # NASCENT
        lifecycle.pin(spec.spec_id)  # → STABLE
        lifecycle.dissolve(spec.spec_id, trigger=DissolutionTrigger.MANUAL)  # → DISSOLVING
        # After animation: → ARCHIVED (automatic)
    """

    async def spawn(
        self,
        spec: JennySpecType
    ) -> JennySpecType:
        """
        Spawn a new panel (transition to NASCENT).

        Records panel creation, starts any timers.

        Args:
            spec: Panel specification to spawn

        Returns:
            Updated spec with spawn metadata
        """
        ...

    async def pin(
        self,
        spec_id: str
    ) -> JennySpecType:
        """
        Pin a panel (transition NASCENT → STABLE).

        User has explicitly pinned this panel.
        Panel will not auto-dissolve until manually dismissed.

        Args:
            spec_id: ID of panel to pin

        Returns:
            Updated spec with STABLE lifecycle

        Raises:
            PanelNotFoundError: If panel doesn't exist
            InvalidTransitionError: If panel not in NASCENT state
        """
        ...

    async def dissolve(
        self,
        spec_id: str,
        trigger: DissolutionTriggerType
    ) -> JennySpecType:
        """
        Dissolve a panel (transition to DISSOLVING).

        Begins dissolution animation, schedules archival.

        Args:
            spec_id: ID of panel to dissolve
            trigger: What caused dissolution

        Returns:
            Updated spec with DISSOLVING lifecycle

        Raises:
            PanelNotFoundError: If panel doesn't exist
        """
        ...

    async def archive(
        self,
        spec_id: str
    ) -> JennySpecType:
        """
        Archive a panel (transition DISSOLVING → ARCHIVED).

        Called after dissolution animation completes.
        Panel is removed from active set, stored in SpecLedger.

        Args:
            spec_id: ID of panel to archive

        Returns:
            Archived spec (for provenance)
        """
        ...

    async def get_active_panels(
        self,
        session_id: Optional[str] = None
    ) -> List[JennySpecType]:
        """
        Get all active panels (NASCENT or STABLE).

        Args:
            session_id: Optional session filter

        Returns:
            List of active panels
        """
        ...

    async def get_panel(
        self,
        spec_id: str
    ) -> Optional[JennySpecType]:
        """
        Get a panel by ID (any lifecycle stage).

        Args:
            spec_id: Panel ID

        Returns:
            Panel if found, None otherwise
        """
        ...

    async def cleanup_stale(
        self,
        max_age_seconds: int = 300
    ) -> List[str]:
        """
        Clean up stale NASCENT panels.

        Dissolves unpinned panels that have been NASCENT
        longer than max_age_seconds.

        Args:
            max_age_seconds: Maximum age before auto-dissolution

        Returns:
            List of dissolved panel IDs
        """
        ...

    async def handle_memory_pressure(
        self,
        target_panel_count: int
    ) -> List[str]:
        """
        Handle memory pressure by dissolving low-priority panels.

        Prioritization for dissolution:
        1. NASCENT panels (never pinned)
        2. Oldest STABLE panels
        3. Never dissolve SYSTEM panels

        Args:
            target_panel_count: Target number of panels to keep

        Returns:
            List of dissolved panel IDs
        """
        ...


# ============================================================================
# SpecLedgerProtocol - Audit Trail for UI Specs
# ============================================================================

@runtime_checkable
class SpecLedgerProtocol(Protocol):
    """
    Protocol for audit trail of all generated UI specifications.

    Extends HoloLoom's AuditTrail pattern to UI specs:
    - Complete provenance (which Spacetime generated which specs)
    - Full replay capability (recreate any past UI)
    - Query by time, spacetime_id, panel_type
    - JSONL persistence for efficiency

    Philosophy:
    - "Decisions are durable" - every generated spec is logged
    - Replay enables debugging ("why did it show that?")
    - Provenance connects UI to reasoning
    - Searchable for pattern analysis

    Usage:
        ledger = SpecLedger(persist_path="./spec_ledger")
        ledger.log_spec(spec, spacetime_id="st_123")

        # Later replay
        specs = ledger.query_by_spacetime("st_123")
        specs = ledger.query_by_time_range(start, end)
    """

    async def log_spec(
        self,
        spec: JennySpecType,
        spacetime_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a spec to the ledger.

        Args:
            spec: JennySpec to log
            spacetime_id: ID of Spacetime that generated this spec
            metadata: Optional additional metadata

        Returns:
            Ledger entry ID
        """
        ...

    async def get_spec(
        self,
        spec_id: str
    ) -> Optional[JennySpecType]:
        """
        Retrieve a spec by ID.

        Args:
            spec_id: Spec ID to retrieve

        Returns:
            Spec if found, None otherwise
        """
        ...

    async def query_by_spacetime(
        self,
        spacetime_id: str
    ) -> List[JennySpecType]:
        """
        Query all specs generated from a Spacetime.

        Args:
            spacetime_id: Spacetime ID to query

        Returns:
            List of specs generated from that Spacetime
        """
        ...

    async def query_by_time_range(
        self,
        start_time: str,
        end_time: str,
        limit: int = 100
    ) -> List[JennySpecType]:
        """
        Query specs within a time range.

        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            limit: Maximum results

        Returns:
            List of specs within range
        """
        ...

    async def query_by_panel_type(
        self,
        panel_type: str,
        limit: int = 100
    ) -> List[JennySpecType]:
        """
        Query specs by panel type.

        Args:
            panel_type: Panel type to query (e.g., "graph", "text")
            limit: Maximum results

        Returns:
            List of specs of that type
        """
        ...

    async def get_statistics(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ledger statistics.

        Args:
            start_time: Optional start filter
            end_time: Optional end filter

        Returns:
            Statistics dict:
            {
                "total_specs": 1234,
                "panel_type_counts": {"text": 500, "graph": 400, ...},
                "avg_panels_per_spacetime": 2.3,
                "dissolution_trigger_counts": {...}
            }
        """
        ...

    async def replay_session(
        self,
        session_id: str
    ) -> List[JennySpecType]:
        """
        Replay all specs from a session in order.

        Enables full UI reconstruction for debugging.

        Args:
            session_id: Session ID to replay

        Returns:
            Ordered list of specs from session
        """
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'CompilationStrategy',
    'RenderTarget',
    # Protocols
    'JennyCompilerProtocol',
    'JennyRendererProtocol',
    'JennyLifecycleProtocol',
    'SpecLedgerProtocol',
]
