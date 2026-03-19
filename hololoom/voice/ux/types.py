"""
Voice UX Type Definitions
Milestone 1: Conversational Mode + Basic Threads (6 weeks)
Date: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ============================================================================
# Voice Modes
# ============================================================================

class VoiceMode(Enum):
    """Voice interaction modes"""
    CONVERSATIONAL = "conversational"  # Natural back-and-forth
    COMMAND = "command"                # Structured commands (Milestone 2)
    STREAMING = "streaming"            # Continuous cognition (Milestone 3)
    DISABLED = "disabled"              # Voice disabled


class IntentType(Enum):
    """User intent classification"""
    # Thread management
    CREATE_THREAD = "create_thread"
    SWITCH_THREAD = "switch_thread"
    CLOSE_THREAD = "close_thread"
    SUMMARIZE_THREADS = "summarize_threads"

    # Conversational
    QUESTION = "question"
    STATEMENT = "statement"
    ACKNOWLEDGMENT = "acknowledgment"

    # System control
    PAUSE = "pause"
    STOP = "stop"
    HELP = "help"

    # Unknown
    UNKNOWN = "unknown"


class ThreadState(Enum):
    """Thread lifecycle states"""
    INACTIVE = "inactive"      # Exists but not active
    ACTIVE = "active"          # Current focus
    BACKGROUND = "background"  # Paused, context maintained
    ARCHIVED = "archived"      # Closed and stored


class STTEngine(Enum):
    """Speech-to-text engine options"""
    WEB_SPEECH_API = "web_speech_api"  # Browser Web Speech API (Milestone 1)
    WHISPER_LOCAL = "whisper_local"    # OpenAI Whisper (Milestone 2)
    GOOGLE_CLOUD = "google_cloud"      # Google Cloud Speech-to-Text (Milestone 2)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VoiceConfig:
    """Voice system configuration"""
    # Mode
    default_mode: VoiceMode = VoiceMode.CONVERSATIONAL

    # STT
    stt_engine: STTEngine = STTEngine.WEB_SPEECH_API  # Default to browser API
    stt_language: str = "en-US"
    stt_continuous: bool = True

    # NLU
    intent_confidence_threshold: float = 0.7

    # Performance
    target_latency_ms: int = 750
    enable_performance_monitoring: bool = True

    # Battery/Network integration (Phase 3.11)
    disable_streaming_on_low_battery: bool = True
    low_battery_threshold: int = 30
    use_local_stt_on_slow_network: bool = True
    slow_network_threshold: float = 1.0

    # Memory integration (Phase 3.11)
    limit_thread_history_on_high_memory: bool = True
    memory_threshold: int = 75


@dataclass
class VoiceInput:
    """Raw voice input from STT"""
    transcript: str
    confidence: float
    language: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    engine: STTEngine = STTEngine.WEB_SPEECH_API
    is_final: bool = True


@dataclass
class Intent:
    """Classified user intent from NLU"""
    type: IntentType
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    raw_transcript: str = ""
    requires_confirmation: bool = False
    can_execute_immediately: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Thread:
    """Conversation thread"""
    id: str
    name: str
    state: ThreadState
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    activation_count: int = 0
    context_snapshot: dict[str, Any] | None = None


@dataclass
class VoiceResponse:
    """System response to user input"""
    text: str
    should_speak: bool = True
    visual_feedback: dict[str, Any] | None = None
    thread_id: str | None = None
    new_thread_state: ThreadState | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: int = 0


@dataclass
class VoiceMetrics:
    """Performance metrics for voice system"""
    total_latency_ms: int = 0
    stt_latency_ms: int = 0
    nlu_latency_ms: int = 0
    execution_latency_ms: int = 0
    intent_confidence: float = 0.0
    stt_confidence: float = 0.0
    command_success: bool = False
    error_message: str | None = None
    voice_mode: VoiceMode = VoiceMode.CONVERSATIONAL
    battery_level: int | None = None
    network_quality: str | None = None
    memory_usage_percent: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
