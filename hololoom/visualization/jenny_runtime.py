"""
Jenny Runtime - Unified Orchestrator
=====================================
The complete Jenny Generative UI Runtime in a single, elegant interface.

Week 4 MVP - December 2025

Philosophy:
> "Disposable pixels, durable decisions."

This module is the crown jewel - it brings together:
- JennyCompiler: Query → JennySpec
- LifecycleManager: Panel state machine
- ActionHandler: User interactions
- StreamingManager: Live data bindings
- SpecLedger: Complete provenance
- Renderers: Multi-target output

Architecture:
                    ┌──────────────────────┐
                    │    JennyRuntime      │
                    │   (Orchestrator)     │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────┐          ┌───────────┐          ┌──────────┐
   │Compiler │          │ Lifecycle │          │ Renderer │
   │         │◄────────►│  Manager  │◄────────►│          │
   └─────────┘          └───────────┘          └──────────┘
        │                      │                      │
        │                      ▼                      │
        │               ┌───────────┐                 │
        │               │  Action   │                 │
        └──────────────►│  Handler  │◄────────────────┘
                        └───────────┘
                               │
                               ▼
                        ┌───────────┐
                        │  Spec     │
                        │  Ledger   │ ← Complete Provenance
                        └───────────┘

Usage:
    from hololoom.visualization import JennyRuntime

    # One-liner magic
    async with JennyRuntime() as jenny:
        # Generate panel from query
        panel = await jenny.ask("What is Thompson Sampling?")
        print(panel.html)  # Rendered HTML

        # User pins the panel
        result = await jenny.act(panel, "pin")

        # View complete history
        history = jenny.history()

References:
- jenny_spec.py (JennySpec data model)
- jenny_compiler.py (Query → JennySpec)
- jenny_lifecycle.py (State machine)
- jenny_actions.py (Action handlers)
- jenny_streaming.py (Live data)
- jenny_renderer.py (Multi-target rendering)
- spec_ledger.py (Provenance)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Set, Type

from .jenny_config_base import BaseConfigWithDefaults
from datetime import datetime
from uuid import uuid4
import asyncio
from contextlib import asynccontextmanager

# Core components
from .jenny_spec import (
    JennySpec,
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    PanelTypeJenny,
    PanelSizeJenny,
    LayoutHint,
    create_action,
    get_default_actions,
    ACTION_PIN,
    ACTION_DISMISS,
    ACTION_WHY,
)
from .jenny_compiler import JennyCompiler, QueryAnalysis
from .jenny_lifecycle import (
    JennyLifecycleManagerBase,
    InMemoryLifecycleManager,
    PanelState,
    PanelNotFoundError,
    InvalidTransitionError,
    create_lifecycle_manager,
)
from .jenny_actions import (
    ActionStatus,
    ActionResult,
    JennyActionHandler,
    create_action_handler,
)

# MRF integration for learning callback (Phase C)
try:
    from .jenny_mrf import JennyMRFCompiler, create_learning_callback
    MRF_AVAILABLE = True
except ImportError:
    MRF_AVAILABLE = False
    JennyMRFCompiler = None  # type: ignore
from .jenny_streaming import (
    StreamStatus,
    UpdateType,
    StreamUpdate,
    Subscription,
    StreamingManager,
    create_streaming_manager,
)
from .jenny_renderer import (
    JennyRendererBase,
    HTMLRenderer,
    TerminalRenderer,
    JSONRenderer,
    RenderTarget,
    create_renderer,
)
from .spec_ledger import SpecLedger, SpecLedgerEntry


# ============================================================================
# Runtime Configuration
# ============================================================================

@dataclass
class JennyConfig(BaseConfigWithDefaults):
    """
    Configuration for Jenny Runtime.

    Sensible defaults for immediate productivity.
    Override for production customization.
    """
    # Rendering
    default_renderer: RenderTarget = RenderTarget.HTML
    theme: str = "default"

    # Lifecycle
    auto_cleanup: bool = True
    cleanup_interval_seconds: float = 60.0
    nascent_ttl_seconds: float = 300.0  # 5 minutes

    # Streaming
    enable_streaming: bool = True
    max_subscriptions: int = 100
    default_refresh_interval_ms: int = 5000

    # Ledger
    enable_ledger: bool = True
    ledger_persist_path: Optional[str] = None

    # Compiler
    compiler_version: str = "jenny-v1.0.0"
    default_spacetime_id: str = "default"


# ============================================================================
# Panel Result (User-Facing)
# ============================================================================

@dataclass
class JennyPanel:
    """
    User-facing panel result.

    Wraps JennySpec with rendered output and convenience methods.
    This is what users interact with - clean and simple.
    """
    spec: JennySpec
    html: str
    terminal: str
    json_data: Dict[str, Any]

    # Convenience accessors
    @property
    def id(self) -> str:
        return self.spec.spec_id

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def lifecycle(self) -> LifecycleStage:
        return self.spec.lifecycle

    @property
    def panel_type(self) -> PanelTypeJenny:
        return self.spec.panel_type

    @property
    def content(self) -> Dict[str, Any]:
        return self.spec.content

    @property
    def actions(self) -> List[Dict[str, Any]]:
        return self.spec.actions

    def __repr__(self) -> str:
        return f"JennyPanel(id={self.id[:8]}..., title='{self.title}', lifecycle={self.lifecycle.value})"


# ============================================================================
# Jenny Runtime - The Main Event
# ============================================================================

class JennyRuntime:
    """
    The complete Jenny Generative UI Runtime.

    Unified orchestrator that brings all Jenny components together
    into a single, elegant interface.

    Philosophy:
    - One class to rule them all
    - Sensible defaults, infinite customization
    - Every operation logged with provenance
    - Async-first, but sync-friendly

    Usage:
        # Quick start (async context manager)
        async with JennyRuntime() as jenny:
            panel = await jenny.ask("What is Thompson Sampling?")
            await jenny.act(panel, "pin")
            print(jenny.history())

        # Manual lifecycle
        jenny = JennyRuntime()
        await jenny.start()
        panel = await jenny.ask("Explain recursion")
        await jenny.stop()

        # Sync-friendly (for demos)
        jenny = JennyRuntime()
        panel = jenny.ask_sync("What is Python?")
    """

    def __init__(
        self,
        config: Optional[JennyConfig] = None,
        compiler: Optional["JennyMRFCompiler"] = None,
    ):
        """
        Initialize Jenny Runtime.

        Args:
            config: Optional configuration (sensible defaults used otherwise)
            compiler: Optional JennyMRFCompiler for MRF-enhanced panel generation.
                     If provided with enable_learning=True, user actions (PIN, DISMISS, etc.)
                     will automatically update Thompson Sampling priors for panel type selection.
        """
        self._config = config or JennyConfig()
        self._compiler = compiler  # MRF compiler for learning integration (Phase C)
        self._started = False

        # Core components (created on start)
        self._compiler_version: str = self._config.compiler_version
        self._lifecycle: Optional[JennyLifecycleManagerBase] = None
        self._actions: Optional[JennyActionHandler] = None
        self._streaming: Optional[StreamingManager] = None
        self._ledger: Optional[SpecLedger] = None

        # Renderers (lazy-loaded)
        self._renderers: Dict[RenderTarget, JennyRendererBase] = {}

        # Statistics
        self._stats = {
            "panels_created": 0,
            "actions_executed": 0,
            "queries_processed": 0,
            "start_time": None,
        }

        # Track all created panel IDs (needed for list_panels with DISSOLVING/ARCHIVED)
        self._panel_ids: Set[str] = set()

        # Cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self) -> "JennyRuntime":
        """
        Start the runtime (initialize all components).

        Returns:
            Self for chaining
        """
        if self._started:
            return self

        # Initialize lifecycle manager
        self._lifecycle = create_lifecycle_manager()

        # Initialize ledger
        if self._config.enable_ledger:
            self._ledger = SpecLedger(persist_path=self._config.ledger_persist_path)

        # Create learning callback if MRF compiler has learning enabled (Phase C)
        learning_callback = None
        if (
            MRF_AVAILABLE
            and self._compiler is not None
            and hasattr(self._compiler, 'enable_learning')
            and self._compiler.enable_learning
        ):
            learning_callback = await create_learning_callback(self._compiler)

        # Initialize action handler with optional learning callback
        self._actions = create_action_handler(
            self._lifecycle,
            self._ledger,
            learning_callback=learning_callback,
        )

        # Initialize streaming manager
        if self._config.enable_streaming:
            self._streaming = create_streaming_manager(
                max_subscriptions=self._config.max_subscriptions
            )

        # Start cleanup loop
        if self._config.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._stats["start_time"] = datetime.now()
        self._started = True

        return self

    async def stop(self) -> None:
        """Stop the runtime (cleanup all components)."""
        if not self._started:
            return

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close streaming manager
        if self._streaming:
            await self._streaming.close()

        # Flush ledger
        if self._ledger:
            self._ledger.flush_all()

        self._started = False

    async def __aenter__(self) -> "JennyRuntime":
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    # ========================================================================
    # Core API - The Magic Happens Here
    # ========================================================================

    async def ask(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        panel_type: Optional[PanelTypeJenny] = None,
        binding_mode: BindingMode = BindingMode.STATIC,
        data_source: Optional[str] = None,
    ) -> JennyPanel:
        """
        Ask Jenny a question - generate a panel.

        This is the main entry point. Give Jenny a query,
        get back a beautiful, actionable panel.

        Args:
            query: Natural language query
            context: Optional context for compilation
            panel_type: Force specific panel type (auto-detected otherwise)
            binding_mode: STATIC, REACTIVE, or STREAMING
            data_source: Required for REACTIVE/STREAMING modes

        Returns:
            JennyPanel with rendered output and actions

        Example:
            panel = await jenny.ask("What is Thompson Sampling?")
            print(panel.html)
        """
        if not self._started:
            await self.start()

        # Create spec directly from query (standalone mode)
        spec = self._create_spec_from_query(
            query=query,
            context=context or {},
            panel_type=panel_type,
            binding_mode=binding_mode,
            data_source=data_source,
        )

        # Register with lifecycle manager
        await self._lifecycle.spawn(spec)

        # Log to ledger
        if self._ledger:
            await self._ledger.log_spec(spec, spacetime_id=spec.spacetime_id)

        # Render to all targets
        panel = await self._render_panel(spec)

        # Update stats and track panel ID
        self._stats["panels_created"] += 1
        self._stats["queries_processed"] += 1
        self._panel_ids.add(spec.spec_id)

        return panel

    def _create_spec_from_query(
        self,
        query: str,
        context: Dict[str, Any],
        panel_type: Optional[PanelTypeJenny] = None,
        binding_mode: BindingMode = BindingMode.STATIC,
        data_source: Optional[str] = None,
    ) -> JennySpec:
        """
        Create JennySpec directly from a query string.

        This enables standalone runtime operation without requiring
        the full HoloLoom weaving cycle (Spacetime).

        Query analysis determines:
        - Panel type (if not specified)
        - Content structure
        - Confidence level
        - Actions available
        """
        query_lower = query.lower()

        # Detect query type for auto panel selection
        detected_type = PanelTypeJenny.TEXT  # default

        if any(word in query_lower for word in ["how to", "steps", "guide", "tutorial"]):
            detected_type = PanelTypeJenny.TEXT  # Use TEXT with ordered list content
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference", "tradeoff"]):
            detected_type = PanelTypeJenny.TABLE
        elif any(word in query_lower for word in ["show me", "visualize", "graph", "chart", "plot"]):
            detected_type = PanelTypeJenny.GRAPH  # CHART doesn't exist, use GRAPH
        elif any(word in query_lower for word in ["code", "function", "write code"]):
            detected_type = PanelTypeJenny.CODE
        elif any(word in query_lower for word in ["what is", "define", "explain", "describe", "implement"]):
            detected_type = PanelTypeJenny.TEXT

        # Use specified type or detected type
        final_type = panel_type or detected_type

        # Detect complexity for confidence estimate
        word_count = len(query.split())
        complexity = "simple" if word_count <= 8 else ("moderate" if word_count <= 15 else "complex")
        confidence = 0.9 if complexity == "simple" else (0.75 if complexity == "moderate" else 0.6)

        # Generate title from query
        title = query[:50] + "..." if len(query) > 50 else query
        title = title.replace("\n", " ").strip()

        # Create content based on panel type
        content = self._generate_content_for_type(final_type, query, context)

        # Create spec
        spec = JennySpec(
            spacetime_id=self._config.default_spacetime_id,
            panel_type=final_type,
            title=title,
            content=content,
            actions=tuple(get_default_actions(final_type)),  # Must be tuple
            lifecycle=LifecycleStage.NASCENT,
            binding_mode=binding_mode,
            data_source=data_source,
            refresh_interval_ms=self._config.default_refresh_interval_ms if binding_mode != BindingMode.STATIC else None,
            compiler_version=self._compiler_version,
            metadata={"confidence": confidence, "complexity": complexity},  # Confidence in metadata
        )

        return spec

    def _generate_content_for_type(
        self,
        panel_type: PanelTypeJenny,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate appropriate content structure for panel type."""
        if panel_type == PanelTypeJenny.TEXT:
            return {
                "text": f"Response to: {query}",
                "format": "markdown",
                "query": query,
            }
        elif panel_type == PanelTypeJenny.TABLE:
            return {
                "headers": ["Aspect", "Option A", "Option B"],
                "rows": [
                    ["Feature 1", "Value A1", "Value B1"],
                    ["Feature 2", "Value A2", "Value B2"],
                ],
                "query": query,
            }
        elif panel_type == PanelTypeJenny.GRAPH:
            return {
                "nodes": [
                    {"id": "a", "label": "A"},
                    {"id": "b", "label": "B"},
                    {"id": "c", "label": "C"},
                ],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "c"},
                ],
                "query": query,
            }
        elif panel_type == PanelTypeJenny.CODE:
            return {
                "code": "# Code example\ndef example():\n    pass",
                "language": context.get("language", "python"),
                "query": query,
            }
        elif panel_type == PanelTypeJenny.CONFIDENCE:
            return {
                "value": 0.85,
                "threshold_low": 0.6,
                "threshold_high": 0.8,
                "query": query,
            }
        elif panel_type == PanelTypeJenny.METRIC:
            return {
                "value": 42,
                "unit": "ms",
                "format": "number",
                "query": query,
            }
        else:
            # Default: text-like content
            return {
                "text": f"Panel content for: {query}",
                "query": query,
            }

    async def compile_spacetime(
        self,
        spacetime: Any,
        features: Optional[Dict[str, Any]] = None,
        panel_count: int = 4,
    ) -> List[JennyPanel]:
        """
        Compile a Spacetime object into Jenny panels.

        Bridge between the weaving pipeline and Jenny's panel system.
        Extracts query, response, confidence, and metadata from Spacetime
        and generates appropriate panels.

        Args:
            spacetime: Spacetime object from the weaving pipeline
            features: Optional extracted features from the pipeline
            panel_count: Maximum number of panels to generate

        Returns:
            List of JennyPanel objects
        """
        if not self._started:
            await self.start()

        # Extract data from Spacetime
        query = getattr(spacetime, 'query_text', '') or ''
        response = getattr(spacetime, 'response', '') or ''
        confidence = getattr(spacetime, 'confidence', 0.8)
        tool_used = getattr(spacetime, 'tool_used', '')
        sources = getattr(spacetime, 'sources_used', [])
        metadata = getattr(spacetime, 'metadata', {})
        trace = getattr(spacetime, 'trace', None)

        context = {
            'response': response,
            'confidence': confidence,
            'tool_used': tool_used,
            'sources': sources,
            'features': features or {},
            **metadata,
        }

        # Build panels based on what the Spacetime contains
        panels: List[JennyPanel] = []

        # 1. Always: primary response panel
        primary = await self.ask(query, context={
            **context,
            'text': response,
        })
        panels.append(primary)

        if len(panels) >= panel_count:
            return panels

        # 2. Confidence panel if below threshold
        if confidence < 0.7:
            conf_panel = await self.ask(
                query,
                context={**context, 'value': confidence},
                panel_type=PanelTypeJenny.CONFIDENCE,
            )
            panels.append(conf_panel)
            if len(panels) >= panel_count:
                return panels

        # 3. Timeline panel if trace has multiple stages
        threads_activated = metadata.get('threads_activated', 0)
        if trace and threads_activated > 2:
            graph_panel = await self.ask(
                query,
                context=context,
                panel_type=PanelTypeJenny.GRAPH,
            )
            panels.append(graph_panel)
            if len(panels) >= panel_count:
                return panels

        # 4. Sources panel if multiple sources used
        if len(sources) > 1:
            sources_panel = await self.ask(
                query,
                context={**context, 'sources': sources},
                panel_type=PanelTypeJenny.SOURCES,
            )
            panels.append(sources_panel)
            if len(panels) >= panel_count:
                return panels

        # 5. Metric panel if duration is notable
        duration_ms = metadata.get('duration_ms', 0)
        if duration_ms > 100:
            metric_panel = await self.ask(
                query,
                context={
                    **context,
                    'value': duration_ms,
                    'unit': 'ms',
                    'format': 'number',
                },
                panel_type=PanelTypeJenny.METRIC,
            )
            panels.append(metric_panel)

        return panels[:panel_count]

    async def act(
        self,
        panel: Union[JennyPanel, str],
        action: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """
        Execute an action on a panel.

        Args:
            panel: JennyPanel or panel ID
            action: Action name ("pin", "dismiss", "why") or action dict
            context: Optional additional context

        Returns:
            ActionResult with outcome details

        Example:
            result = await jenny.act(panel, "pin")
            if result.status == ActionStatus.SUCCESS:
                print("Panel pinned!")
        """
        if not self._started:
            await self.start()

        # Get spec
        panel_id = panel.id if isinstance(panel, JennyPanel) else panel
        spec = await self._lifecycle.get_panel(panel_id)

        if not spec:
            return ActionResult(
                action_id=str(uuid4()),
                handler=action if isinstance(action, str) else action.get("handler", "unknown"),
                status=ActionStatus.FAILED,
                spec_id=panel_id,
                error="Panel not found",
            )

        # Resolve action
        if isinstance(action, str):
            action_dict = self._resolve_action(action)
        else:
            action_dict = action

        # Execute action
        result = await self._actions.execute(
            action=action_dict,
            spec=spec,
            context=context,
        )

        # Update stats
        self._stats["actions_executed"] += 1

        return result

    def _resolve_action(self, action_name: str) -> Dict[str, Any]:
        """Resolve action name to action dict."""
        actions = {
            "pin": ACTION_PIN,
            "dismiss": ACTION_DISMISS,
            "why": ACTION_WHY,
            "expand": create_action("expand", "expand_panel"),
            "collapse": create_action("collapse", "collapse_panel"),
            "copy": create_action("copy", "copy_content"),
            "export": create_action("export", "export_panel", requires_confirmation=True),
        }
        return actions.get(action_name, create_action(action_name, action_name))

    async def get_panel(self, panel_id: str) -> Optional[JennyPanel]:
        """
        Get a panel by ID.

        Args:
            panel_id: Panel spec ID

        Returns:
            JennyPanel if found, None otherwise
        """
        if not self._started:
            return None

        spec = await self._lifecycle.get_panel(panel_id)
        if not spec:
            return None

        return await self._render_panel(spec)

    async def list_panels(
        self,
        lifecycle: Optional[LifecycleStage] = None,
    ) -> List[JennyPanel]:
        """
        List panels, optionally filtered by lifecycle stage.

        Args:
            lifecycle: Filter by lifecycle stage (None for all)

        Returns:
            List of JennyPanels
        """
        if not self._started:
            return []

        # For DISSOLVING or ARCHIVED, we need to look up all tracked panel IDs
        # because get_active_panels() only returns NASCENT and STABLE
        if lifecycle in (LifecycleStage.DISSOLVING, LifecycleStage.ARCHIVED):
            specs = []
            for panel_id in self._panel_ids:
                spec = await self._lifecycle.get_panel(panel_id)
                if spec and spec.lifecycle == lifecycle:
                    specs.append(spec)
        elif lifecycle is None:
            # Get all panels by looking up each tracked ID
            specs = []
            for panel_id in self._panel_ids:
                spec = await self._lifecycle.get_panel(panel_id)
                if spec:
                    specs.append(spec)
        else:
            # For NASCENT/STABLE, use the optimized active panels method
            specs = await self._lifecycle.get_active_panels()
            specs = [s for s in specs if s.lifecycle == lifecycle]

        panels = []
        for spec in specs:
            panel = await self._render_panel(spec)
            panels.append(panel)

        return panels

    # ========================================================================
    # Streaming API
    # ========================================================================

    async def subscribe(
        self,
        panel: Union[JennyPanel, str],
        callback: Callable[[StreamUpdate], Awaitable[None]],
    ) -> str:
        """
        Subscribe to panel data updates.

        Only works for REACTIVE or STREAMING binding modes.

        Args:
            panel: JennyPanel or panel ID
            callback: Async function called on each update

        Returns:
            Subscription ID
        """
        if not self._streaming:
            raise RuntimeError("Streaming not enabled")

        panel_id = panel.id if isinstance(panel, JennyPanel) else panel
        spec = await self._lifecycle.get_panel(panel_id)

        if not spec:
            raise ValueError(f"Panel not found: {panel_id}")

        return await self._streaming.subscribe(spec, callback)

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from panel updates."""
        if not self._streaming:
            return False
        return await self._streaming.unsubscribe(subscription_id)

    # ========================================================================
    # Provenance & History
    # ========================================================================

    def history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get complete interaction history.

        Every panel creation, action, and state change is logged.
        This is Jenny's killer feature: complete provenance.

        Args:
            limit: Maximum entries to return

        Returns:
            List of ledger entries (most recent first)
        """
        history_items = []

        # Add panel creation and lifecycle events from ledger
        if self._ledger:
            entries = list(self._ledger.entries.values())
            entries.sort(key=lambda e: e.logged_at, reverse=True)

            for entry in entries[:limit]:
                entry_dict = entry.to_dict()

                # Add creation event
                history_items.append({
                    **entry_dict,
                    "event_type": "panel_creation",
                })

                # Add lifecycle transition events from history
                for transition in entry.lifecycle_history[1:]:  # Skip initial state
                    history_items.append({
                        "event_type": "lifecycle_transition",
                        "spec_id": entry.spec.spec_id,
                        "stage": transition.get("stage"),
                        "trigger": transition.get("trigger"),
                        "timestamp": transition.get("timestamp"),
                    })

        # Add action events from action history
        if self._actions and hasattr(self._actions, '_history'):
            for action_result in self._actions._history:
                history_items.append({
                    "event_type": "action",
                    "action_id": action_result.action_id,
                    "handler": action_result.handler,
                    "status": action_result.status.value if hasattr(action_result.status, 'value') else str(action_result.status),
                    "spec_id": action_result.spec_id,
                    "timestamp": datetime.now().isoformat(),  # Action results don't have timestamps
                })

        # Sort all events by timestamp (newest first)
        history_items.sort(
            key=lambda h: h.get("timestamp") or h.get("logged_at", ""),
            reverse=True
        )

        return history_items[:limit]

    def action_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get action execution history."""
        if not self._actions:
            return []
        # ActionHandler has _history attribute (List[ActionResult])
        history = self._actions._history if hasattr(self._actions, '_history') else []
        # Convert ActionResult to dict
        result = []
        for action_result in history[-limit:]:
            result.append({
                "action_id": action_result.action_id,
                "handler": action_result.handler,
                "status": action_result.status.value if hasattr(action_result.status, 'value') else str(action_result.status),
                "spec_id": action_result.spec_id,
                "error": action_result.error,
                "data": action_result.data,
            })
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        stats = dict(self._stats)

        # Add lifecycle stats - note: can't await in sync method
        # so we track active_panels via our own counter
        if self._lifecycle:
            stats["lifecycle"] = {
                "active_panels": self._stats.get("panels_created", 0),
            }

        # Add streaming stats
        if self._streaming:
            stats["streaming"] = self._streaming.get_statistics()

        # Uptime
        if stats["start_time"]:
            stats["uptime_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()

        return stats

    # ========================================================================
    # Rendering
    # ========================================================================

    async def _render_panel(self, spec: JennySpec) -> JennyPanel:
        """Render spec to all targets."""
        # Get or create renderers
        html_renderer = self._get_renderer(RenderTarget.HTML)
        terminal_renderer = self._get_renderer(RenderTarget.TERMINAL)
        json_renderer = self._get_renderer(RenderTarget.JSON)

        # Render single spec - each renderer needs its own target
        html = await html_renderer.render_single(spec, target=RenderTarget.HTML)
        terminal = await terminal_renderer.render_single(spec, target=RenderTarget.TERMINAL)
        json_str = await json_renderer.render_single(spec, target=RenderTarget.JSON)

        import json
        json_data = json.loads(json_str)

        return JennyPanel(
            spec=spec,
            html=html,
            terminal=terminal,
            json_data=json_data,
        )

    def _get_renderer(self, target: RenderTarget) -> JennyRendererBase:
        """Get or create renderer for target."""
        if target not in self._renderers:
            self._renderers[target] = create_renderer(target)
        return self._renderers[target]

    # ========================================================================
    # Background Tasks
    # ========================================================================

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired panels."""
        while True:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)

                # Get expired NASCENT panels
                specs = await self._lifecycle.get_active_panels()
                now = datetime.now()

                for spec in specs:
                    if spec.lifecycle == LifecycleStage.NASCENT:
                        age = (now - spec.timestamp).total_seconds()
                        if age > self._config.nascent_ttl_seconds:
                            # Auto-dissolve expired NASCENT
                            await self._lifecycle.dissolve(
                                spec.spec_id,
                                DissolutionTrigger.TTL_EXPIRED,
                            )
                            if self._ledger:
                                await self._ledger.log_lifecycle_transition(
                                    spec_id=spec.spec_id,
                                    new_stage=LifecycleStage.DISSOLVING,
                                    trigger=DissolutionTrigger.TTL_EXPIRED,
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash
                print(f"Jenny cleanup error: {e}")

    # ========================================================================
    # Sync Helpers (for demos and simple scripts)
    # ========================================================================

    def ask_sync(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> JennyPanel:
        """
        Synchronous version of ask() for simple scripts.

        Creates event loop if needed. Use async version for production.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.ask(query, context))

    def act_sync(
        self,
        panel: Union[JennyPanel, str],
        action: str,
    ) -> ActionResult:
        """Synchronous version of act()."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.act(panel, action))


# ============================================================================
# Factory Functions
# ============================================================================

def create_runtime(
    config: Optional[JennyConfig] = None,
    compiler: Optional["JennyMRFCompiler"] = None,
) -> JennyRuntime:
    """
    Create a Jenny Runtime instance.

    Args:
        config: Optional configuration
        compiler: Optional JennyMRFCompiler for learning-enabled panel generation.
                 When provided with enable_learning=True, user actions automatically
                 update Thompson Sampling priors for improved panel type selection.

    Returns:
        JennyRuntime (not started - use async with or call start())
    """
    return JennyRuntime(config, compiler=compiler)


@asynccontextmanager
async def jenny_session(
    config: Optional[JennyConfig] = None,
    compiler: Optional["JennyMRFCompiler"] = None,
):
    """
    Convenience async context manager for Jenny sessions.

    Args:
        config: Optional configuration
        compiler: Optional JennyMRFCompiler for learning-enabled panel generation

    Usage:
        async with jenny_session() as jenny:
            panel = await jenny.ask("What is Python?")

        # With MRF learning:
        from hololoom.visualization.jenny_mrf import JennyMRFCompiler
        compiler = JennyMRFCompiler(enable_learning=True)
        async with jenny_session(compiler=compiler) as jenny:
            panel = await jenny.ask("What is Thompson Sampling?")
            await jenny.act(panel, "pin")  # Updates learning priors
    """
    runtime = JennyRuntime(config, compiler=compiler)
    try:
        await runtime.start()
        yield runtime
    finally:
        await runtime.stop()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Configuration
    "JennyConfig",

    # User-facing result
    "JennyPanel",

    # Main class
    "JennyRuntime",

    # Factories
    "create_runtime",
    "jenny_session",
]
