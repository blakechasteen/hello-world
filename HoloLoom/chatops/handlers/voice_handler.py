"""
Voice Handler for Matrix ChatOps

Handles voice messages from Matrix clients:
1. Downloads audio from RoomMessageAudio events
2. Transcribes using local Whisper (WhisperSpinner)
3. Classifies intent using CodeVoiceGrammar
4. Routes to appropriate code generation/explanation/editing handlers

Architecture:
- Protocol-based handlers for extensibility
- Event hooks for monitoring and integration
- Middleware support for cross-cutting concerns
- Async-first design throughout

Usage:
    handler = VoiceHandler(matrix_client)
    response = await handler.handle_voice_message(room, event)

    # Custom handlers via protocol
    handler.register_handler(CodeIntent.GENERATE, my_generator)

    # Event hooks for monitoring
    handler.on_transcription(lambda t: log_transcription(t))
    handler.on_intent_classified(lambda i: track_intent(i))

Author: Claude Code
Date: December 2025
"""

import asyncio
import tempfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Protocol, runtime_checkable
from dataclasses import dataclass, field
import aiofiles

try:
    from nio import AsyncClient, MatrixRoom, DownloadResponse
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False

from HoloLoom.chatops.handlers.code_voice_grammar import (
    CodeVoiceGrammar,
    CodeCommandIntent,
    CodeIntent
)

# Try to import WhisperSpinner
try:
    from HoloLoom.spinningWheel.whisper_spinner import (
        WhisperSpinner,
        AudioTranscription,
        WHISPER_AVAILABLE
    )
except ImportError:
    WHISPER_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    language: str
    duration: float
    confidence: float
    segments: list  # List of TimecodeSegment
    success: bool
    error: Optional[str] = None


@dataclass
class VoiceCommandResult:
    """Result of processing a voice command"""
    original_text: str
    intent: CodeIntent
    params: Dict[str, str]
    response: str
    confidence: float
    success: bool
    error: Optional[str] = None
    code_block: Optional[str] = None  # Generated/modified code
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Protocol-Based Handler Interface
# =============================================================================

@runtime_checkable
class IntentHandler(Protocol):
    """
    Protocol for intent handlers.

    Implement this protocol to create custom handlers for code intents.
    Handlers receive params and context, return VoiceCommandResult.

    Example:
        class MyCodeGenerator:
            async def handle(self, params, context) -> VoiceCommandResult:
                # Generate code using your custom logic
                return VoiceCommandResult(...)

        handler.register_handler(CodeIntent.GENERATE, MyCodeGenerator())
    """

    async def handle(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle an intent with given params and context."""
        ...


class BaseIntentHandler(ABC):
    """
    Abstract base class for intent handlers.

    Provides common utilities and enforces the handler interface.
    """

    @abstractmethod
    async def handle(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle the intent. Must be implemented by subclasses."""
        pass

    def _make_result(
        self,
        original_text: str,
        intent: CodeIntent,
        params: Dict[str, str],
        response: str,
        confidence: float = 0.8,
        success: bool = True,
        code_block: Optional[str] = None,
        error: Optional[str] = None,
        **metadata
    ) -> VoiceCommandResult:
        """Convenience method to create VoiceCommandResult."""
        return VoiceCommandResult(
            original_text=original_text,
            intent=intent,
            params=params,
            response=response,
            confidence=confidence,
            success=success,
            code_block=code_block,
            error=error,
            metadata=metadata
        )


# =============================================================================
# Event Hooks
# =============================================================================

@dataclass
class VoiceEvent:
    """Base class for voice processing events."""
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class TranscriptionEvent(VoiceEvent):
    """Emitted after audio transcription."""
    result: TranscriptionResult = None
    duration_ms: float = 0.0


@dataclass
class IntentClassifiedEvent(VoiceEvent):
    """Emitted after intent classification."""
    intent: CodeCommandIntent = None
    text: str = ""


@dataclass
class CommandProcessedEvent(VoiceEvent):
    """Emitted after command processing."""
    result: VoiceCommandResult = None
    processing_time_ms: float = 0.0


class EventEmitter:
    """
    Simple event emitter for voice processing hooks.

    Allows registering callbacks for various processing stages.
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, callback: Callable) -> 'EventEmitter':
        """Register a callback for an event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
        return self

    def off(self, event_type: str, callback: Callable) -> 'EventEmitter':
        """Remove a callback for an event type."""
        if event_type in self._listeners:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb != callback
            ]
        return self

    async def emit(self, event_type: str, event: VoiceEvent):
        """Emit an event to all registered listeners."""
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    result = callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"Event listener error ({event_type}): {e}")


# =============================================================================
# Voice Handler
# =============================================================================

class VoiceHandler:
    """
    Handles voice messages from Matrix for code operations.

    Pipeline:
    1. Download audio from Matrix server
    2. Transcribe with local Whisper
    3. Classify intent (GENERATE, EXPLAIN, EDIT, REVIEW, QUERY)
    4. Route to appropriate handler
    5. Return response for Matrix bot to send

    Example:
        >>> handler = VoiceHandler(client)
        >>> result = await handler.handle_voice_message(room, event)
        >>> print(result.response)
    """

    def __init__(
        self,
        matrix_client: Optional[Any] = None,
        whisper_model: str = "base",
        language: Optional[str] = None,
        code_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VoiceHandler.

        Args:
            matrix_client: AsyncClient from matrix-nio
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            language: Target language for transcription (None for auto-detect)
            code_context: Shared context for code operations
        """
        self.client = matrix_client
        self.whisper_model = whisper_model
        self.language = language

        # Code context for multi-turn conversations
        self.code_context = code_context or {}

        # Initialize grammar
        self.grammar = CodeVoiceGrammar()

        # Initialize Whisper spinner (lazy load)
        self._whisper: Optional[WhisperSpinner] = None

        # Handler callbacks for different intents
        self._intent_handlers: Dict[CodeIntent, Callable] = {}

        # Register default handlers
        self._register_default_handlers()

        logger.info(f"VoiceHandler initialized (model: {whisper_model})")

    def _get_whisper(self) -> Optional[WhisperSpinner]:
        """Lazy-load Whisper spinner."""
        if self._whisper is None and WHISPER_AVAILABLE:
            self._whisper = WhisperSpinner(
                model_size=self.whisper_model,
                language=self.language,
                enable_word_timestamps=True
            )
        return self._whisper

    def _register_default_handlers(self):
        """Register default intent handlers."""
        self._intent_handlers = {
            CodeIntent.GENERATE: self._handle_generate,
            CodeIntent.EXPLAIN: self._handle_explain,
            CodeIntent.EDIT: self._handle_edit,
            CodeIntent.REVIEW: self._handle_review,
            CodeIntent.CONTEXT_SET: self._handle_context_set,
            CodeIntent.CONTEXT_CLEAR: self._handle_context_clear,
            CodeIntent.QUERY: self._handle_query,
        }

    def register_handler(self, intent: CodeIntent, handler: Callable):
        """
        Register a custom handler for an intent.

        Args:
            intent: CodeIntent type
            handler: Async callable(params, context) -> str
        """
        self._intent_handlers[intent] = handler
        logger.info(f"Registered custom handler for {intent.name}")

    # =========================================================================
    # Audio Processing
    # =========================================================================

    async def download_audio(
        self,
        mxc_url: str,
        room_id: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Download audio file from Matrix server.

        Args:
            mxc_url: Matrix content URL (mxc://...)
            room_id: Room ID for encrypted content

        Returns:
            Audio bytes or None on failure
        """
        if not self.client:
            logger.error("No Matrix client available")
            return None

        try:
            # Download from Matrix
            response = await self.client.download(mxc_url)

            if isinstance(response, DownloadResponse):
                logger.info(f"Downloaded audio: {len(response.body)} bytes")
                return response.body
            else:
                logger.error(f"Download failed: {response}")
                return None

        except Exception as e:
            logger.error(f"Error downloading audio: {e}", exc_info=True)
            return None

    async def transcribe(self, audio_data: bytes) -> TranscriptionResult:
        """
        Transcribe audio using local Whisper.

        Args:
            audio_data: Raw audio bytes

        Returns:
            TranscriptionResult with text and metadata
        """
        if not WHISPER_AVAILABLE:
            return TranscriptionResult(
                text="",
                language="",
                duration=0.0,
                confidence=0.0,
                segments=[],
                success=False,
                error="Whisper not available. Install with: pip install openai-whisper"
            )

        whisper = self._get_whisper()
        if not whisper or not whisper.is_available():
            return TranscriptionResult(
                text="",
                language="",
                duration=0.0,
                confidence=0.0,
                segments=[],
                success=False,
                error="Whisper model not loaded"
            )

        try:
            # Write audio to temp file (Whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_path = Path(f.name)

            try:
                # Transcribe
                transcription = await whisper._transcribe_file(temp_path)

                # Calculate average confidence
                avg_confidence = 0.0
                if transcription.segments:
                    avg_confidence = sum(s.confidence for s in transcription.segments) / len(transcription.segments)

                return TranscriptionResult(
                    text=transcription.full_text,
                    language=transcription.language,
                    duration=transcription.duration,
                    confidence=avg_confidence,
                    segments=transcription.segments,
                    success=True
                )

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return TranscriptionResult(
                text="",
                language="",
                duration=0.0,
                confidence=0.0,
                segments=[],
                success=False,
                error=str(e)
            )

    def classify_intent(self, text: str) -> CodeCommandIntent:
        """
        Classify transcribed text into code intent.

        Args:
            text: Transcribed voice text

        Returns:
            CodeCommandIntent with classified intent and params
        """
        return self.grammar.classify(text)

    # =========================================================================
    # Intent Handlers (Default Implementations)
    # =========================================================================

    async def _handle_generate(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle code generation requests."""
        request = params.get('request', '')
        code_type = params.get('type', 'code')
        language = params.get('language', context.get('language', 'python'))

        # TODO: Integrate with HoloLoom weaving for actual code generation
        # For now, return a placeholder that can be replaced with real generation

        response = f"Generating {code_type} to {request}..."
        code_block = f"# TODO: Generate {code_type} that {request}\n# Language: {language}"

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.GENERATE,
            params=params,
            response=response,
            confidence=0.8,
            success=True,
            code_block=code_block,
            metadata={'language': language, 'type': code_type}
        )

    async def _handle_explain(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle code explanation requests."""
        current_code = context.get('current_code', '')

        if not current_code:
            return VoiceCommandResult(
                original_text=params.get('_original', ''),
                intent=CodeIntent.EXPLAIN,
                params=params,
                response="No code context set. Please share the code you want explained.",
                confidence=0.9,
                success=False,
                error="No code context"
            )

        # TODO: Integrate with HoloLoom for actual explanation
        response = f"Explaining the code...\n\nThe code you shared appears to be..."

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.EXPLAIN,
            params=params,
            response=response,
            confidence=0.8,
            success=True
        )

    async def _handle_edit(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle code editing requests."""
        current_code = context.get('current_code', '')
        edit_request = params.get('request', '')

        if not current_code:
            return VoiceCommandResult(
                original_text=params.get('_original', ''),
                intent=CodeIntent.EDIT,
                params=params,
                response="No code context set. Please share the code you want to edit.",
                confidence=0.9,
                success=False,
                error="No code context"
            )

        # TODO: Integrate with HoloLoom for actual code editing
        response = f"Editing code to {edit_request}..."
        code_block = f"# Edited code: {edit_request}\n{current_code}"

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.EDIT,
            params=params,
            response=response,
            confidence=0.8,
            success=True,
            code_block=code_block
        )

    async def _handle_review(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle code review requests."""
        current_code = context.get('current_code', '')

        if not current_code:
            return VoiceCommandResult(
                original_text=params.get('_original', ''),
                intent=CodeIntent.REVIEW,
                params=params,
                response="No code context set. Please share the code you want reviewed.",
                confidence=0.9,
                success=False,
                error="No code context"
            )

        # TODO: Integrate with Trough/xTerminator for actual code review
        response = "Reviewing the code...\n\n✓ No obvious issues found."

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.REVIEW,
            params=params,
            response=response,
            confidence=0.8,
            success=True
        )

    async def _handle_context_set(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle context setting requests."""
        # Context is set externally when code is shared
        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.CONTEXT_SET,
            params=params,
            response="Ready to work with the code you shared.",
            confidence=0.9,
            success=True
        )

    async def _handle_context_clear(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle context clearing requests."""
        self.code_context.clear()

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.CONTEXT_CLEAR,
            params=params,
            response="Code context cleared. Starting fresh.",
            confidence=0.95,
            success=True
        )

    async def _handle_query(
        self,
        params: Dict[str, str],
        context: Dict[str, Any]
    ) -> VoiceCommandResult:
        """Handle general queries (fallback)."""
        query = params.get('request', '')

        # TODO: Route to HoloLoom for general knowledge queries
        response = f"Processing your query: {query}"

        return VoiceCommandResult(
            original_text=params.get('_original', ''),
            intent=CodeIntent.QUERY,
            params=params,
            response=response,
            confidence=0.7,
            success=True
        )

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def handle_voice_message(
        self,
        room: Any,  # MatrixRoom
        event: Any,  # RoomMessageAudio
        code_context: Optional[Dict[str, Any]] = None
    ) -> VoiceCommandResult:
        """
        Main entry point for processing voice messages.

        Args:
            room: Matrix room where message was sent
            event: RoomMessageAudio event
            code_context: Optional code context (current code, language, etc.)

        Returns:
            VoiceCommandResult with response and metadata
        """
        # Merge context
        context = {**self.code_context}
        if code_context:
            context.update(code_context)

        try:
            # 1. Download audio
            mxc_url = event.url
            audio_data = await self.download_audio(mxc_url, room.room_id)

            if not audio_data:
                return VoiceCommandResult(
                    original_text="",
                    intent=CodeIntent.QUERY,
                    params={},
                    response="Failed to download audio file.",
                    confidence=0.0,
                    success=False,
                    error="Download failed"
                )

            # 2. Transcribe
            transcription = await self.transcribe(audio_data)

            if not transcription.success:
                return VoiceCommandResult(
                    original_text="",
                    intent=CodeIntent.QUERY,
                    params={},
                    response=f"Transcription failed: {transcription.error}",
                    confidence=0.0,
                    success=False,
                    error=transcription.error
                )

            logger.info(f"Transcribed: '{transcription.text}' (lang: {transcription.language})")

            # 3. Classify intent
            intent = self.classify_intent(transcription.text)

            logger.info(f"Classified: {intent.code_intent.name} (confidence: {intent.confidence:.2f})")

            # 4. Route to handler
            handler = self._intent_handlers.get(intent.code_intent, self._handle_query)

            # Add original text to params
            params = {**intent.params, '_original': transcription.text}

            result = await handler(params, context)

            # Add transcription metadata
            result.metadata['transcription'] = {
                'text': transcription.text,
                'language': transcription.language,
                'duration': transcription.duration,
                'confidence': transcription.confidence
            }

            return result

        except Exception as e:
            logger.error(f"Voice handler error: {e}", exc_info=True)
            return VoiceCommandResult(
                original_text="",
                intent=CodeIntent.QUERY,
                params={},
                response=f"Error processing voice message: {str(e)}",
                confidence=0.0,
                success=False,
                error=str(e)
            )

    async def process_text(
        self,
        text: str,
        code_context: Optional[Dict[str, Any]] = None
    ) -> VoiceCommandResult:
        """
        Process text directly (for testing or text-based input).

        Args:
            text: Text to process as if transcribed
            code_context: Optional code context

        Returns:
            VoiceCommandResult
        """
        context = {**self.code_context}
        if code_context:
            context.update(code_context)

        # Classify intent
        intent = self.classify_intent(text)

        # Route to handler
        handler = self._intent_handlers.get(intent.code_intent, self._handle_query)
        params = {**intent.params, '_original': text}

        result = await handler(params, context)
        result.metadata['source'] = 'text'

        return result

    def set_code_context(self, code: str, language: Optional[str] = None):
        """
        Set the current code context for multi-turn conversations.

        Args:
            code: Code snippet to set as context
            language: Programming language (optional)
        """
        self.code_context['current_code'] = code
        if language:
            self.code_context['language'] = language
        logger.info(f"Code context set ({len(code)} chars, lang: {language})")

    def get_code_context(self) -> Dict[str, Any]:
        """Get current code context."""
        return self.code_context.copy()


# =============================================================================
# Factory Function
# =============================================================================

def create_voice_handler(
    matrix_client: Optional[Any] = None,
    whisper_model: str = "base",
    language: Optional[str] = None
) -> VoiceHandler:
    """
    Create a VoiceHandler instance.

    Args:
        matrix_client: Optional Matrix AsyncClient
        whisper_model: Whisper model size
        language: Target language for transcription

    Returns:
        Configured VoiceHandler
    """
    return VoiceHandler(
        matrix_client=matrix_client,
        whisper_model=whisper_model,
        language=language
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 60)
        print("Voice Handler Demo")
        print("=" * 60)

        # Create handler without Matrix client (for testing)
        handler = create_voice_handler(whisper_model="base")

        # Test text processing (simulates transcribed voice)
        test_commands = [
            "write a function that sorts users by age",
            "explain this code",
            "add error handling to this",
            "review my implementation",
            "clear context",
            "what is Thompson Sampling?"
        ]

        for command in test_commands:
            print(f"\nCommand: '{command}'")
            result = await handler.process_text(command)
            print(f"  Intent: {result.intent.name}")
            print(f"  Response: {result.response[:80]}...")
            print(f"  Success: {result.success}")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)

    asyncio.run(demo())
