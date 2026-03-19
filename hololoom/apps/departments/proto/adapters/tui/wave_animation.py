"""Proto TUI Wave Animation System.

Semantic wave animations that communicate system state:
- Top waves = INCOMING (waiting for response from LLM)
- Bottom waves = OUTGOING (generating/sending request)
- Waves intermix and grow deeper as processes increase

Status: Phase 3 TUI (2025-12-05)
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum


class WaveState(Enum):
    """Wave animation states with semantic meaning."""
    # Top waves (incoming)
    IDLE = "idle"
    WAITING = "waiting"
    RECEIVING = "receiving"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

    # Bottom waves (outgoing)
    TYPING = "typing"
    PREPARING = "preparing"
    SENDING = "sending"
    SENT = "sent"


class WaveTheme(Enum):
    """Visual wave themes."""
    OCEAN = "ocean"           # Unicode ocean (default)
    KANAGAWA = "kanagawa"     # Japanese style
    CYBERPUNK = "cyberpunk"   # Block characters
    MINIMAL = "minimal"       # Simple sine


# Wave character sets
WAVE_CHARS = {
    # Water/wave symbols
    'tilde': '～',
    'wave': '∿',
    'approx': '≈',
    'triple': '≋',

    # Particles/foam
    'sparkle': '✧✦',
    'dot': '·°',
    'star': '゚',

    # Blocks (for gradient)
    'blocks': '░▒▓█',

    # Lines
    'heavy': '━',
    'double': '═',

    # Collision markers
    'up': '⬆',
    'down': '⬇',

    # Effects
    'lightning': '⚡',
    'wave_emoji': '🌊',
}


# State-based colors (Textual/Rich compatible)
WAVE_STATE_COLORS = {
    # Top waves (incoming)
    WaveState.IDLE: 'dim',
    WaveState.WAITING: 'cyan',
    WaveState.RECEIVING: 'bright_cyan',
    WaveState.PROCESSING: 'magenta',
    WaveState.COMPLETE: 'green',
    WaveState.ERROR: 'red',

    # Bottom waves (outgoing)
    WaveState.TYPING: 'yellow',
    WaveState.PREPARING: 'blue',
    WaveState.SENDING: 'bright_blue',
    WaveState.SENT: 'dim',
}


@dataclass
class WaveProcess:
    """Represents an active process affecting wave display."""
    name: str
    intensity: float = 1.0  # 0.0-1.0
    direction: str = "incoming"  # "incoming" or "outgoing"
    started_at: float = field(default_factory=time.time)


@dataclass
class WaveConfig:
    """Wave animation configuration."""
    theme: WaveTheme = WaveTheme.OCEAN
    base_frequency: float = 0.15  # Wave frequency
    base_amplitude: float = 1.0   # Wave amplitude multiplier
    animation_speed: float = 0.1  # Scroll speed per frame
    foam_probability: float = 0.3 # Chance of foam particles
    min_rows: int = 2            # Minimum wave rows
    max_rows: int = 8            # Maximum wave rows
    fps: int = 12                # Animation frames per second


class WaveGenerator:
    """Generates animated wave patterns with semantic state communication."""

    def __init__(self, config: WaveConfig | None = None):
        self.config = config or WaveConfig()
        self.frame = 0
        self.processes: list[WaveProcess] = []

        # Top and bottom state
        self.top_state = WaveState.IDLE
        self.bottom_state = WaveState.IDLE

        # Animation offset (for scrolling)
        self.offset = 0.0

    def set_top_state(self, state: WaveState) -> None:
        """Set the top wave state (incoming/response)."""
        self.top_state = state

    def set_bottom_state(self, state: WaveState) -> None:
        """Set the bottom wave state (outgoing/request)."""
        self.bottom_state = state

    def add_process(self, name: str, intensity: float = 1.0,
                    direction: str = "incoming") -> None:
        """Add an active process that affects wave depth."""
        self.processes.append(WaveProcess(
            name=name,
            intensity=intensity,
            direction=direction
        ))

    def remove_process(self, name: str) -> None:
        """Remove a process by name."""
        self.processes = [p for p in self.processes if p.name != name]

    def clear_processes(self) -> None:
        """Clear all active processes."""
        self.processes.clear()

    def calculate_wave_depth(self) -> int:
        """Calculate wave row depth based on active processes."""
        if not self.processes:
            return self.config.min_rows

        base_depth = self.config.min_rows
        depth_per_process = 1.5

        total_depth = base_depth + sum(
            p.intensity * depth_per_process
            for p in self.processes
        )

        return min(int(total_depth), self.config.max_rows)

    def _get_ocean_chars(self, row: int, depth: int) -> str:
        """Get wave characters for ocean theme based on row depth."""
        if row == 0:
            # Top foam layer
            return '゚·✧ ゚·✦ '
        elif row == 1:
            # Surface
            return '≋≋∿∿≈≈～～'
        elif row == 2:
            # Mid water
            return '≈≈≋≋∿∿～～'
        elif row == 3:
            # Deeper
            return '～～≈≈≋≋∿∿'
        else:
            # Deep water / base
            return '▓▓▓▓▓▓▓▓'

    def _get_cyberpunk_chars(self, row: int, depth: int) -> str:
        """Get wave characters for cyberpunk theme."""
        if row == 0:
            return '░▒▓▓▒░  '
        elif row == 1:
            return '▓██▓▓██▓'
        else:
            return '████████'

    def _get_kanagawa_chars(self, row: int, depth: int) -> str:
        """Get wave characters for Kanagawa (Japanese) theme."""
        if row == 0:
            return '  ╭───╮  '
        elif row == 1:
            return '╭╯ ● ╰──╮'
        else:
            return '─╯       ╰'

    def _get_minimal_chars(self, row: int, depth: int) -> str:
        """Get wave characters for minimal theme."""
        if row == 0:
            return 'ᨒ  ᨓ  '
        else:
            return '∿∿∿∿∿∿'

    def _get_wave_chars(self, row: int, depth: int) -> str:
        """Get wave characters based on theme and row."""
        if self.config.theme == WaveTheme.OCEAN:
            return self._get_ocean_chars(row, depth)
        elif self.config.theme == WaveTheme.CYBERPUNK:
            return self._get_cyberpunk_chars(row, depth)
        elif self.config.theme == WaveTheme.KANAGAWA:
            return self._get_kanagawa_chars(row, depth)
        else:  # MINIMAL
            return self._get_minimal_chars(row, depth)

    def _add_collision_markers(self, line: str,
                                incoming_active: bool,
                                outgoing_active: bool) -> str:
        """Add collision markers when both flows active."""
        if not (incoming_active and outgoing_active):
            return line

        # Add collision markers at random positions
        result = list(line)
        marker_probability = 0.15

        for i in range(len(result)):
            if random.random() < marker_probability:
                if result[i] not in ' \t\n':
                    # Alternate between up and down markers
                    result[i] = '⬆' if random.random() < 0.5 else '⬇'

        return ''.join(result)

    def generate_wave_row(self, width: int, row: int, depth: int,
                          is_top: bool,
                          incoming_active: bool,
                          outgoing_active: bool) -> str:
        """Generate a single wave row with animation offset."""
        chars = self._get_wave_chars(row, depth)
        pattern_len = len(chars)

        # Apply animation offset for scrolling effect
        offset = int(self.offset) % pattern_len

        # Build the row by repeating and offsetting the pattern
        result = []
        for i in range(width):
            char_idx = (i + offset) % pattern_len
            result.append(chars[char_idx])

        line = ''.join(result)

        # Add collision markers if both processes active
        if incoming_active and outgoing_active and depth > 3:
            line = self._add_collision_markers(line, incoming_active, outgoing_active)

        return line

    def generate_top_waves(self, width: int, title: str = "P R O T O") -> list[str]:
        """Generate top wave rows (incoming/response)."""
        depth = self.calculate_wave_depth()
        incoming_active = self.top_state not in [WaveState.IDLE, WaveState.COMPLETE]
        outgoing_active = self.bottom_state not in [WaveState.IDLE, WaveState.SENT]

        rows = []

        for row_idx in range(depth):
            line = self.generate_wave_row(
                width, row_idx, depth,
                is_top=True,
                incoming_active=incoming_active,
                outgoing_active=outgoing_active
            )

            # Center title in first row
            if row_idx == 0 and title:
                title_with_sparkles = f'゚・✧ {title} ゚・✧'
                padding = (width - len(title_with_sparkles)) // 2
                if padding > 0:
                    line = line[:padding] + title_with_sparkles + line[padding + len(title_with_sparkles):]

            rows.append(line)

        return rows

    def generate_bottom_waves(self, width: int) -> list[str]:
        """Generate bottom wave rows (outgoing/request)."""
        depth = self.calculate_wave_depth()
        incoming_active = self.top_state not in [WaveState.IDLE, WaveState.COMPLETE]
        outgoing_active = self.bottom_state not in [WaveState.IDLE, WaveState.SENT]

        rows = []

        # Bottom waves are inverted (deep at top, foam at bottom)
        for row_idx in range(depth - 1, -1, -1):
            line = self.generate_wave_row(
                width, row_idx, depth,
                is_top=False,
                incoming_active=incoming_active,
                outgoing_active=outgoing_active
            )
            rows.append(line)

        # Add bubbles at the very bottom
        if depth >= 2:
            bubbles = self._generate_bubbles(width)
            rows.append(bubbles)

        return rows

    def _generate_bubbles(self, width: int) -> str:
        """Generate random bubble/sparkle row."""
        result = []
        for _ in range(width):
            if random.random() < self.config.foam_probability:
                result.append(random.choice('·✧✦°'))
            else:
                result.append(' ')
        return ''.join(result)

    def advance_frame(self) -> None:
        """Advance animation by one frame."""
        self.frame += 1
        self.offset += self.config.animation_speed

        # Adjust speed based on activity level
        activity = len(self.processes)
        if activity > 2:
            self.offset += self.config.animation_speed * 0.5  # Faster when busy

    def get_top_color(self) -> str:
        """Get Rich/Textual color for top waves."""
        return WAVE_STATE_COLORS.get(self.top_state, 'cyan')

    def get_bottom_color(self) -> str:
        """Get Rich/Textual color for bottom waves."""
        return WAVE_STATE_COLORS.get(self.bottom_state, 'blue')


class SemanticWaveController:
    """High-level controller for semantic wave communication.

    Provides a simple interface for setting wave states based on
    Proto's processing stages.
    """

    def __init__(self, generator: WaveGenerator | None = None):
        self.generator = generator or WaveGenerator()

    def on_idle(self) -> None:
        """System is idle, ready for input."""
        self.generator.set_top_state(WaveState.IDLE)
        self.generator.set_bottom_state(WaveState.IDLE)
        self.generator.clear_processes()

    def on_user_typing(self) -> None:
        """User is typing a query."""
        self.generator.set_bottom_state(WaveState.TYPING)
        self.generator.add_process("typing", intensity=0.3, direction="outgoing")

    def on_preparing_request(self) -> None:
        """Building context and preparing request."""
        self.generator.set_bottom_state(WaveState.PREPARING)
        self.generator.add_process("preparing", intensity=0.6, direction="outgoing")

    def on_sending_request(self) -> None:
        """Request is being sent."""
        self.generator.set_bottom_state(WaveState.SENDING)
        self.generator.add_process("sending", intensity=0.8, direction="outgoing")

    def on_waiting_response(self) -> None:
        """Waiting for LLM response."""
        self.generator.set_top_state(WaveState.WAITING)
        self.generator.set_bottom_state(WaveState.SENT)
        self.generator.remove_process("sending")
        self.generator.add_process("waiting", intensity=0.5, direction="incoming")

    def on_receiving_response(self) -> None:
        """Response is streaming in."""
        self.generator.set_top_state(WaveState.RECEIVING)
        self.generator.add_process("receiving", intensity=0.7, direction="incoming")

    def on_processing(self) -> None:
        """LLM is processing (thinking)."""
        self.generator.set_top_state(WaveState.PROCESSING)
        self.generator.add_process("processing", intensity=1.0, direction="incoming")

    def on_complete(self) -> None:
        """Response complete."""
        self.generator.set_top_state(WaveState.COMPLETE)
        self.generator.clear_processes()

    def on_error(self) -> None:
        """Error occurred."""
        self.generator.set_top_state(WaveState.ERROR)
        self.generator.set_bottom_state(WaveState.ERROR)
        self.generator.add_process("error", intensity=1.0, direction="incoming")
        self.generator.add_process("error_out", intensity=1.0, direction="outgoing")


# Convenience function for simple usage
def create_wave_system(theme: WaveTheme = WaveTheme.OCEAN) -> SemanticWaveController:
    """Create a semantic wave controller with the specified theme."""
    config = WaveConfig(theme=theme)
    generator = WaveGenerator(config)
    return SemanticWaveController(generator)
