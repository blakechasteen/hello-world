#!/usr/bin/env python3
"""
WeavingOrchestrator Integration - Self-Constructing Dashboards
================================================================
Extends WeavingOrchestrator with automatic dashboard generation.

This module adds the "Wolfram Alpha" capability to HoloLoom:
  Query -> Weave -> Auto-Dashboard -> HTML/Browser

Usage:
    from HoloLoom.visualization import DashboardOrchestrator

    orchestrator = DashboardOrchestrator(cfg, shards)
    result = await orchestrator.weave_with_dashboard(query, save_path="dashboard.html")
    # result.spacetime = Spacetime artifact
    # result.dashboard = Dashboard object
    # result.html = HTML string
    # result.file_path = Saved file path (if save_path provided)

Author: Claude Code
Date: October 29, 2025
"""

import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import webbrowser

from ..weaving_orchestrator import WeavingOrchestrator
from ..documentation.types import Query
from ..fabric.spacetime import Spacetime
from .constructor import DashboardConstructor
from .html_renderer import HTMLRenderer
from .dashboard import Dashboard


@dataclass
class DashboardResult:
    """Result from weaving with dashboard generation."""
    spacetime: Spacetime
    dashboard: Dashboard
    html: str
    file_path: Optional[Path] = None


class DashboardOrchestrator(WeavingOrchestrator):
    """
    Extended WeavingOrchestrator with automatic dashboard generation.

    This class inherits from WeavingOrchestrator and adds dashboard
    generation capabilities. All original weaving functionality is preserved.

    New capabilities:
    - weave_with_dashboard(): Weave + auto-generate dashboard
    - generate_dashboard(): Create dashboard from existing Spacetime
    - serve_dashboard(): Generate and open in browser
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize DashboardOrchestrator.

        Accepts all WeavingOrchestrator arguments plus:
            enable_dashboard_generation: Enable auto-dashboard (default True)
            dashboard_theme: 'light' or 'dark' (default 'light')
        """
        # Extract dashboard-specific kwargs
        self.enable_dashboard_generation = kwargs.pop('enable_dashboard_generation', True)
        self.dashboard_theme = kwargs.pop('dashboard_theme', 'light')

        # Initialize parent WeavingOrchestrator
        super().__init__(*args, **kwargs)

        # Initialize dashboard components
        if self.enable_dashboard_generation:
            self.dashboard_constructor = DashboardConstructor()
            self.dashboard_renderer = HTMLRenderer(theme=self.dashboard_theme)
            self.logger.info("[DASHBOARD] Dashboard generation enabled")

    async def weave_with_dashboard(
        self,
        query: Query,
        save_path: Optional[str] = None,
        open_browser: bool = False,
        **weave_kwargs
    ) -> DashboardResult:
        """
        Execute weaving cycle and auto-generate dashboard.

        This is the main "Wolfram Alpha" API - one call gives you:
        1. Complete Spacetime artifact with full trace
        2. Auto-generated dashboard optimized for the query
        3. Rendered HTML ready to display
        4. Optionally saved to file and opened in browser

        Args:
            query: User query
            save_path: Optional path to save HTML file
            open_browser: If True, open dashboard in browser
            **weave_kwargs: Additional arguments for weave() method

        Returns:
            DashboardResult with spacetime, dashboard, html, and path

        Example:
            result = await orch.weave_with_dashboard(
                Query(text="How does Thompson Sampling work?"),
                save_path="output.html",
                open_browser=True
            )
        """
        if not self.enable_dashboard_generation:
            raise RuntimeError("Dashboard generation is disabled")

        self.logger.info(f"[DASHBOARD] Weaving with auto-dashboard for: '{query.text}'")

        # Step 1: Execute weaving cycle
        spacetime = await self.weave(query, **weave_kwargs)

        # Step 2: Generate dashboard
        dashboard = self.dashboard_constructor.construct(spacetime)
        self.logger.info(
            f"[DASHBOARD] Generated dashboard: {len(dashboard.panels)} panels, "
            f"layout={dashboard.layout.value}"
        )

        # Step 3: Render to HTML
        html = self.dashboard_renderer.render(dashboard)
        self.logger.info(f"[DASHBOARD] Rendered HTML ({len(html)} chars)")

        # Step 4: Optionally save to file
        file_path = None
        if save_path:
            file_path = Path(save_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(html, encoding='utf-8')
            self.logger.info(f"[DASHBOARD] Saved to: {file_path}")

        # Step 5: Optionally open in browser
        if open_browser and file_path:
            webbrowser.open(f'file://{file_path.absolute()}')
            self.logger.info(f"[DASHBOARD] Opened in browser")

        return DashboardResult(
            spacetime=spacetime,
            dashboard=dashboard,
            html=html,
            file_path=file_path
        )

    def generate_dashboard(self, spacetime: Spacetime) -> Dashboard:
        """
        Generate dashboard from existing Spacetime artifact.

        Use this when you already have a Spacetime (e.g., from cache)
        and just want to generate a fresh dashboard view.

        Args:
            spacetime: Spacetime artifact

        Returns:
            Dashboard object

        Example:
            spacetime = await orchestrator.weave(query)
            dashboard = orchestrator.generate_dashboard(spacetime)
        """
        if not self.enable_dashboard_generation:
            raise RuntimeError("Dashboard generation is disabled")

        return self.dashboard_constructor.construct(spacetime)

    def render_dashboard(self, dashboard: Dashboard) -> str:
        """
        Render dashboard to HTML string.

        Args:
            dashboard: Dashboard object

        Returns:
            HTML string
        """
        if not self.enable_dashboard_generation:
            raise RuntimeError("Dashboard generation is disabled")

        return self.dashboard_renderer.render(dashboard)

    async def serve_dashboard(
        self,
        query: Query,
        output_dir: Optional[Path] = None,
        **weave_kwargs
    ) -> DashboardResult:
        """
        Weave query and immediately open dashboard in browser.

        Convenience method for interactive exploration. Automatically:
        1. Executes weaving
        2. Generates dashboard
        3. Saves to temp/output directory
        4. Opens in default browser

        Args:
            query: User query
            output_dir: Directory to save dashboards (default: ./dashboards/)
            **weave_kwargs: Additional arguments for weave()

        Returns:
            DashboardResult

        Example:
            # Interactive exploration
            await orch.serve_dashboard(Query(text="What is Thompson Sampling?"))
        """
        if output_dir is None:
            output_dir = Path("./dashboards")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from query
        safe_query = "".join(c if c.isalnum() else "_" for c in query.text[:50])
        filename = f"dashboard_{safe_query}_{int(asyncio.get_event_loop().time())}.html"
        save_path = output_dir / filename

        return await self.weave_with_dashboard(
            query,
            save_path=str(save_path),
            open_browser=True,
            **weave_kwargs
        )


# ============================================================================
# Convenience Functions
# ============================================================================

async def weave_and_visualize(
    query: str,
    cfg,
    shards=None,
    memory=None,
    save_path: Optional[str] = None,
    open_browser: bool = False
) -> DashboardResult:
    """
    One-shot function: weave query and generate dashboard.

    This is the simplest API - pass a query string and get back everything.

    Args:
        query: Query string
        cfg: Config object
        shards: Optional memory shards
        memory: Optional unified memory backend
        save_path: Optional path to save HTML
        open_browser: If True, open in browser

    Returns:
        DashboardResult

    Example:
        from HoloLoom.config import Config
        from HoloLoom.visualization import weave_and_visualize

        result = await weave_and_visualize(
            "How does the weaving orchestrator work?",
            cfg=Config.fast(),
            shards=test_shards,
            save_path="output.html",
            open_browser=True
        )
    """
    from ..documentation.types import Query as QueryType

    orchestrator = DashboardOrchestrator(
        cfg=cfg,
        shards=shards,
        memory=memory,
        enable_dashboard_generation=True
    )

    query_obj = QueryType(text=query)

    return await orchestrator.weave_with_dashboard(
        query_obj,
        save_path=save_path,
        open_browser=open_browser
    )
