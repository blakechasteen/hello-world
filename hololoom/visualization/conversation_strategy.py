"""
Conversation Visualization Strategy
======================================
Maps conversation trajectory → Jenny panel type selection.

Decides WHAT to show and WHEN. Returns None when visualization would
be noise (too early, wrapping up, no significant change).

Author: HoloLoom Team
Date: 2026-03-11
"""


from .conversation_graph import ConversationGraph, Trajectory
from .jenny_spec import PanelTypeJenny

# Minimum turns before any conversation visualization
MIN_TURNS_TO_VISUALIZE = 3

# Trajectory → panel type (None = skip visualization)
_TRAJECTORY_PANEL: dict[Trajectory, PanelTypeJenny | None] = {
    Trajectory.EXPLORING: PanelTypeJenny.TABLE,        # Fan-out: show topics touched
    Trajectory.COMPARING: PanelTypeJenny.COMPARISON,   # Two topics being weighed
    Trajectory.DEEP_DIVE: PanelTypeJenny.TABLE,        # One dominant topic — detail
    Trajectory.DECIDING: PanelTypeJenny.COMPARISON,    # Decision matrix
    Trajectory.WRAPPING_UP: None,                      # Winding down — stop
}


class ConversationVisualizationStrategy:
    """
    Stateless strategy: given a trajectory and graph shape,
    pick the right panel type (or None to skip).
    """

    def select(
        self,
        trajectory: Trajectory,
        graph: ConversationGraph,
        turn_count: int,
    ) -> PanelTypeJenny | None:
        """
        Select panel type based on conversation trajectory.

        Returns None to skip visualization (too early, winding down, etc.)
        """
        if turn_count < MIN_TURNS_TO_VISUALIZE:
            return None

        panel = _TRAJECTORY_PANEL.get(trajectory)
        if panel is not None:
            return panel

        # Explicit None in the map (wrapping_up) vs. unknown trajectory
        if trajectory in _TRAJECTORY_PANEL:
            return None

        # Unknown trajectory — default to table if enough content
        if len(graph.nodes) >= 3:
            return PanelTypeJenny.TABLE

        return None

    def should_render(
        self,
        graph: ConversationGraph,
        trajectory: Trajectory,
    ) -> bool:
        """
        Combined check: significant change AND useful panel type.
        """
        if not graph.has_significant_change():
            return False

        panel_type = self.select(trajectory, graph, graph.turn_count)
        return panel_type is not None
