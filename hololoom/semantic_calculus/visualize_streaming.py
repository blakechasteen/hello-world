#!/usr/bin/env python3
"""
📊🌊 REAL-TIME MULTI-SCALE SEMANTIC VISUALIZATION
=================================================
Live visualization of word-by-word semantic calculus across nested scales.

Visualization Panels:
1. Position Trajectories - Where each scale is in 244D space
2. Velocity Heatmap - Which dimensions are actively changing
3. Acceleration Spikes - Narrative force moments
4. Phase Portrait - Attractors and cycles in semantic space
5. Scale Resonance Matrix - Coupling between scales
6. Narrative Momentum Curve - Overall flow quality

The visualization updates in real-time as text streams in.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
from collections import deque

from .streaming_multi_scale import (
    MultiScaleSnapshot,
    TemporalScale,
    ScaleResonance
)


class RealtimeSemanticVisualizer:
    """
    Real-time multi-panel visualization of streaming semantic calculus.

    Updates live as snapshots arrive from StreamingSemanticCalculus.
    """

    def __init__(
        self,
        window_size: int = 50,
        update_interval_ms: int = 100,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Initialize visualizer.

        Args:
            window_size: How many snapshots to show in time series
            update_interval_ms: Milliseconds between visual updates
            figsize: Figure size in inches
        """
        self.window_size = window_size
        self.update_interval_ms = update_interval_ms

        # Data buffers
        self.snapshot_buffer: deque = deque(maxlen=window_size)
        self.momentum_history: deque = deque(maxlen=window_size)
        self.complexity_history: deque = deque(maxlen=window_size)
        self.word_count_history: deque = deque(maxlen=window_size)

        # Create figure with subplots
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.gs = GridSpec(3, 3, figure=self.fig)

        # Initialize subplots
        self._init_subplots()

        # Animation control
        self.animation = None
        self.is_running = False

    def _init_subplots(self):
        """Initialize all subplot axes."""
        # Row 1: Position trajectories (large), Velocity heatmap
        self.ax_position = self.fig.add_subplot(self.gs[0, :2])
        self.ax_velocity = self.fig.add_subplot(self.gs[0, 2])

        # Row 2: Acceleration spikes, Phase portrait, Resonance matrix
        self.ax_acceleration = self.fig.add_subplot(self.gs[1, 0])
        self.ax_phase = self.fig.add_subplot(self.gs[1, 1])
        self.ax_resonance = self.fig.add_subplot(self.gs[1, 2])

        # Row 3: Momentum curve (spans all columns)
        self.ax_momentum = self.fig.add_subplot(self.gs[2, :])

        # Style all axes
        for ax in [self.ax_position, self.ax_velocity, self.ax_acceleration,
                   self.ax_phase, self.ax_resonance, self.ax_momentum]:
            ax.set_facecolor('#f8f8f8')
            ax.grid(True, alpha=0.3, linestyle='--')

        # Set titles
        self.ax_position.set_title('Position: Top Dimensions Over Time', fontweight='bold', fontsize=11)
        self.ax_velocity.set_title('Velocity Heatmap', fontweight='bold', fontsize=11)
        self.ax_acceleration.set_title('Acceleration Spikes', fontweight='bold', fontsize=11)
        self.ax_phase.set_title('Phase Portrait (2D)', fontweight='bold', fontsize=11)
        self.ax_resonance.set_title('Scale Resonance', fontweight='bold', fontsize=11)
        self.ax_momentum.set_title('Narrative Momentum & Complexity', fontweight='bold', fontsize=11)

        # Labels
        self.ax_position.set_xlabel('Word Count')
        self.ax_position.set_ylabel('Semantic Position')
        self.ax_velocity.set_xlabel('Time →')
        self.ax_velocity.set_ylabel('Dimensions')
        self.ax_acceleration.set_xlabel('Word Count')
        self.ax_acceleration.set_ylabel('Force Magnitude')
        self.ax_phase.set_xlabel('Position')
        self.ax_phase.set_ylabel('Momentum')
        self.ax_momentum.set_xlabel('Word Count')
        self.ax_momentum.set_ylabel('Score')

        # Set main title
        self.fig.suptitle('🌊📐 Multi-Scale Streaming Semantic Calculus',
                         fontsize=16, fontweight='bold')

    def add_snapshot(self, snapshot: MultiScaleSnapshot):
        """Add new snapshot to visualization buffers."""
        self.snapshot_buffer.append(snapshot)
        self.momentum_history.append(snapshot.narrative_momentum)
        self.complexity_history.append(snapshot.complexity_index)
        self.word_count_history.append(snapshot.word_count)

    def update_all(self, frame: int = 0):
        """Update all subplots with latest data."""
        if len(self.snapshot_buffer) == 0:
            return

        latest_snapshot = self.snapshot_buffer[-1]

        # Update position trajectories
        self._update_position_plot(latest_snapshot)

        # Update velocity heatmap
        self._update_velocity_heatmap(latest_snapshot)

        # Update acceleration spikes
        self._update_acceleration_plot(latest_snapshot)

        # Update phase portrait
        self._update_phase_portrait(latest_snapshot)

        # Update resonance matrix
        self._update_resonance_matrix(latest_snapshot)

        # Update momentum curve
        self._update_momentum_curve()

        # Refresh canvas
        self.fig.canvas.draw_idle()

    def _update_position_plot(self, snapshot: MultiScaleSnapshot):
        """Update position trajectories for top dimensions."""
        self.ax_position.clear()

        if not snapshot.dominant_dimensions:
            return

        # Plot top 3 dimensions across word scale
        word_states = snapshot.states_by_scale.get(TemporalScale.WORD)
        if not word_states or word_states['position'] is None:
            return

        # Get position history from snapshot buffer
        word_counts = list(self.word_count_history)
        if len(word_counts) < 2:
            return

        # Extract positions for top dimensions over time
        from .dimensions import EXTENDED_244_DIMENSIONS
        dim_names = [d.name for d in EXTENDED_244_DIMENSIONS]

        for dim_name in snapshot.dominant_dimensions[:5]:
            if dim_name in dim_names:
                dim_idx = dim_names.index(dim_name)
                positions = []

                for snap in self.snapshot_buffer:
                    ws = snap.states_by_scale.get(TemporalScale.WORD)
                    if ws and ws['position'] is not None:
                        positions.append(ws['position'][dim_idx])
                    else:
                        positions.append(np.nan)

                if positions:
                    self.ax_position.plot(word_counts[-len(positions):],
                                         positions, label=dim_name,
                                         linewidth=2, alpha=0.7, marker='o', markersize=3)

        self.ax_position.set_xlabel('Word Count')
        self.ax_position.set_ylabel('Semantic Position')
        self.ax_position.set_title('Position: Top Dimensions Over Time', fontweight='bold')
        self.ax_position.legend(loc='upper left', fontsize=8)
        self.ax_position.grid(True, alpha=0.3)

    def _update_velocity_heatmap(self, snapshot: MultiScaleSnapshot):
        """Update velocity heatmap for recent history."""
        self.ax_velocity.clear()

        if not snapshot.dominant_dimensions:
            return

        # Collect velocity data for top dimensions
        from .dimensions import EXTENDED_244_DIMENSIONS
        dim_names = [d.name for d in EXTENDED_244_DIMENSIONS]

        velocity_matrix = []
        for snap in list(self.snapshot_buffer)[-20:]:  # Last 20 snapshots
            word_states = snap.states_by_scale.get(TemporalScale.WORD)
            if word_states and word_states['velocity'] is not None:
                row = []
                for dim_name in snapshot.dominant_dimensions[:8]:
                    if dim_name in dim_names:
                        dim_idx = dim_names.index(dim_name)
                        row.append(word_states['velocity'][dim_idx])
                    else:
                        row.append(0.0)
                velocity_matrix.append(row)

        if velocity_matrix:
            velocity_array = np.array(velocity_matrix).T
            im = self.ax_velocity.imshow(velocity_array, aspect='auto',
                                         cmap='RdBu_r', interpolation='nearest',
                                         vmin=-0.5, vmax=0.5)
            self.ax_velocity.set_yticks(range(len(snapshot.dominant_dimensions[:8])))
            self.ax_velocity.set_yticklabels(snapshot.dominant_dimensions[:8], fontsize=8)
            self.ax_velocity.set_xlabel('Time →')
            self.ax_velocity.set_title('Velocity Heatmap', fontweight='bold')

    def _update_acceleration_plot(self, snapshot: MultiScaleSnapshot):
        """Update acceleration spikes."""
        self.ax_acceleration.clear()

        word_counts = list(self.word_count_history)
        accelerations = []

        for snap in self.snapshot_buffer:
            word_states = snap.states_by_scale.get(TemporalScale.WORD)
            if word_states and word_states['acceleration'] is not None:
                # Total acceleration magnitude
                mag = np.linalg.norm(word_states['acceleration'])
                accelerations.append(mag)
            else:
                accelerations.append(0.0)

        if accelerations:
            self.ax_acceleration.bar(word_counts[-len(accelerations):],
                                    accelerations, color='red', alpha=0.6)
            self.ax_acceleration.set_xlabel('Word Count')
            self.ax_acceleration.set_ylabel('Force Magnitude')
            self.ax_acceleration.set_title('Acceleration Spikes (Narrative Forces)', fontweight='bold')
            self.ax_acceleration.grid(True, alpha=0.3)

    def _update_phase_portrait(self, snapshot: MultiScaleSnapshot):
        """Update 2D phase portrait."""
        self.ax_phase.clear()

        # Use dominant dimension for phase space
        if not snapshot.dominant_dimensions:
            return

        from .dimensions import EXTENDED_244_DIMENSIONS
        dim_names = [d.name for d in EXTENDED_244_DIMENSIONS]

        if snapshot.dominant_dimensions[0] not in dim_names:
            return

        dim_idx = dim_names.index(snapshot.dominant_dimensions[0])

        positions = []
        momenta = []

        for snap in self.snapshot_buffer:
            word_states = snap.states_by_scale.get(TemporalScale.WORD)
            if word_states and word_states['position'] is not None:
                positions.append(word_states['position'][dim_idx])
                if word_states['velocity'] is not None:
                    momenta.append(word_states['velocity'][dim_idx])
                else:
                    momenta.append(0.0)

        if positions and momenta:
            self.ax_phase.plot(positions, momenta, 'b-', alpha=0.5, linewidth=2)
            self.ax_phase.scatter(positions[0], momenta[0],
                                 c='green', s=100, marker='o', label='Start', zorder=5)
            self.ax_phase.scatter(positions[-1], momenta[-1],
                                 c='red', s=100, marker='X', label='Current', zorder=5)
            self.ax_phase.set_xlabel(f'{snapshot.dominant_dimensions[0]} Position')
            self.ax_phase.set_ylabel(f'{snapshot.dominant_dimensions[0]} Momentum')
            self.ax_phase.set_title('Phase Portrait (2D)', fontweight='bold')
            self.ax_phase.legend(fontsize=8)
            self.ax_phase.grid(True, alpha=0.3)

    def _update_resonance_matrix(self, snapshot: MultiScaleSnapshot):
        """Update scale resonance matrix."""
        self.ax_resonance.clear()

        if not snapshot.resonances:
            return

        # Build resonance matrix
        scales = list(TemporalScale)
        n_scales = len(scales)
        matrix = np.zeros((n_scales, n_scales))

        for resonance in snapshot.resonances:
            s1, s2 = resonance.scale_pair
            i1 = scales.index(s1)
            i2 = scales.index(s2)
            score = resonance.resonance_score
            matrix[i1, i2] = score
            matrix[i2, i1] = score  # Symmetric

        # Plot heatmap
        im = self.ax_resonance.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1,
                                      interpolation='nearest')
        self.ax_resonance.set_xticks(range(n_scales))
        self.ax_resonance.set_yticks(range(n_scales))
        self.ax_resonance.set_xticklabels([s.value[:4] for s in scales], fontsize=8)
        self.ax_resonance.set_yticklabels([s.value[:4] for s in scales], fontsize=8)
        self.ax_resonance.set_title('Scale Resonance', fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=self.ax_resonance, fraction=0.046, pad=0.04)

    def _update_momentum_curve(self):
        """Update momentum and complexity curves."""
        self.ax_momentum.clear()

        word_counts = list(self.word_count_history)
        momentum = list(self.momentum_history)
        complexity = list(self.complexity_history)

        if word_counts and momentum:
            self.ax_momentum.plot(word_counts, momentum,
                                 label='Momentum (alignment)', color='blue',
                                 linewidth=2, marker='o', markersize=4)
            self.ax_momentum.plot(word_counts, complexity,
                                 label='Complexity (divergence)', color='orange',
                                 linewidth=2, marker='s', markersize=4)

            self.ax_momentum.set_xlabel('Word Count')
            self.ax_momentum.set_ylabel('Score (0-1)')
            self.ax_momentum.set_ylim(-0.1, 1.1)
            self.ax_momentum.set_title('Narrative Momentum & Complexity', fontweight='bold')
            self.ax_momentum.legend(loc='upper left')
            self.ax_momentum.grid(True, alpha=0.3)

            # Add interpretation zones
            self.ax_momentum.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Strong')
            self.ax_momentum.axhspan(0.4, 0.7, alpha=0.1, color='yellow')
            self.ax_momentum.axhspan(0.0, 0.4, alpha=0.1, color='red')

    def start_animation(self):
        """Start real-time animation."""
        if not self.is_running:
            self.animation = FuncAnimation(
                self.fig,
                self.update_all,
                interval=self.update_interval_ms,
                blit=False
            )
            self.is_running = True
            plt.show(block=False)

    def stop_animation(self):
        """Stop animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False

    def save_snapshot(self, filepath: str):
        """Save current visualization to file."""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📸 Saved visualization: {filepath}")


async def demonstrate_live_visualization():
    """Demonstrate live visualization with streaming text."""
    import asyncio
    from .streaming_multi_scale import StreamingSemanticCalculus

    print("📊🌊 LIVE MULTI-SCALE VISUALIZATION DEMO")
    print("=" * 80)
    print()

    # Simple embedding for demo
    def simple_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text.lower()) % (2**32))
        return np.random.randn(384)

    # Sample narrative
    text = """
    The ordinary world was comfortable but empty. Then came the call, unexpected and
    disruptive. She refused at first, afraid of the unknown. But the mentor appeared,
    offering wisdom and courage. With a deep breath, she crossed the threshold into
    a new world. Tests came immediately - some she passed, others she failed. Allies
    emerged, but so did enemies. The deeper she went, the harder it became. Then came
    the supreme ordeal, the moment of truth. Everything hung in the balance. In the
    darkness, she found her power. The reward was hard-won and precious. Now begins
    the journey home, forever changed. She returns with the elixir to transform others.
    """

    # Create analyzer
    analyzer = StreamingSemanticCalculus(
        embed_fn=simple_embed,
        snapshot_interval=0.5
    )

    # Create visualizer
    visualizer = RealtimeSemanticVisualizer(
        window_size=30,
        update_interval_ms=500
    )

    # Connect analyzer to visualizer
    def on_snapshot(snapshot):
        visualizer.add_snapshot(snapshot)
        visualizer.update_all()

    analyzer.on_snapshot(on_snapshot)

    # Start visualization
    visualizer.start_animation()

    # Stream text
    async def word_stream():
        for word in text.split():
            yield word
            await asyncio.sleep(0.2)

    print("🚀 Starting live visualization...")
    print("   (Close plot window to end)")
    print()

    async for snapshot in analyzer.stream_analyze(word_stream()):
        await asyncio.sleep(0.01)  # Let matplotlib update

    print("✅ Visualization complete!")
    visualizer.save_snapshot('semantic_calculus_visualization.png')


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_live_visualization())
