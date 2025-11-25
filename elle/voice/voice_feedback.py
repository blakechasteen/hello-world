"""
Voice Feedback Manager - Voice interaction during task execution

Provides:
- Progress throttling (minimum time between announcements)
- Confirmation prompts
- Interrupt handling (stop, pause, cancel)
- Streaming responses (optional)

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 4
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable, Dict
from enum import Enum
from dataclasses import dataclass


# ============================================================================
# Voice Feedback Protocol
# ============================================================================

class FeedbackType(Enum):
    """Type of voice feedback"""
    PROGRESS = "progress"              # Progress update (throttled)
    CONFIRMATION = "confirmation"      # Requires user response
    ERROR = "error"                    # Error message (immediate)
    COMPLETION = "completion"          # Task completion (immediate)
    INFO = "info"                      # General info (immediate)


@dataclass
class VoiceFeedback:
    """Voice feedback message"""
    feedback_type: FeedbackType
    message: str
    task_id: Optional[str] = None
    requires_response: bool = False
    timeout_seconds: Optional[float] = None  # For confirmations


class InterruptType(Enum):
    """Type of user interrupt"""
    STOP = "stop"        # Cancel task
    PAUSE = "pause"      # Pause task
    RESUME = "resume"    # Resume paused task
    SKIP = "skip"        # Skip current step
    NONE = "none"        # No interrupt


# ============================================================================
# Voice Feedback Manager
# ============================================================================

class VoiceFeedbackManager:
    """
    Manages voice feedback during task execution

    Responsibilities:
    - Throttle progress announcements (avoid spam)
    - Handle confirmation prompts
    - Detect interrupts (stop, pause, resume)
    - Coordinate TTS and STT

    Usage:
        manager = VoiceFeedbackManager(
            tts_handler=assistant.speak,
            stt_handler=assistant.stt.listen,
            throttle_seconds=5.0
        )

        # Announce progress (throttled)
        await manager.announce(VoiceFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message="Analyzing data: 50% complete",
            task_id="task_123"
        ))

        # Request confirmation
        response = await manager.announce(VoiceFeedback(
            feedback_type=FeedbackType.CONFIRMATION,
            message="Delete all files?",
            requires_response=True,
            timeout_seconds=10.0
        ))
    """

    def __init__(
        self,
        tts_handler: Callable[[str], Awaitable[bool]],
        stt_handler: Callable[[float], Awaitable[str]],
        throttle_seconds: float = 5.0
    ):
        """
        Initialize voice feedback manager

        Args:
            tts_handler: Async function to speak text (e.g., assistant.speak)
            stt_handler: Async function to listen for speech (e.g., stt.listen)
            throttle_seconds: Minimum seconds between progress announcements
        """
        self.tts = tts_handler
        self.stt = stt_handler
        self.throttle_seconds = throttle_seconds

        # Track last announcement time per task
        self._last_announcement: Dict[str, float] = {}

        # Interrupt detection
        self._listening_for_interrupts = False
        self._interrupt_task: Optional[asyncio.Task] = None

    async def announce(self, feedback: VoiceFeedback) -> Optional[str]:
        """
        Announce feedback to user

        Args:
            feedback: Voice feedback to announce

        Returns:
            User response (if requires_response=True), else None
        """
        # Check throttling for PROGRESS feedback
        if feedback.feedback_type == FeedbackType.PROGRESS:
            if not self._should_announce(feedback.task_id or "default"):
                return None

        # Speak message
        await self.tts(feedback.message)

        # Update last announcement time
        if feedback.task_id:
            self._last_announcement[feedback.task_id] = time.time()

        # Wait for response if needed
        if feedback.requires_response:
            response = await self.stt(feedback.timeout_seconds or 10.0)
            return response

        return None

    async def confirm(self, message: str, timeout: float = 10.0) -> bool:
        """
        Request user confirmation

        Args:
            message: Confirmation message (e.g., "Delete all files?")
            timeout: Timeout in seconds

        Returns:
            True if user confirms (yes/ok/sure), False otherwise
        """
        feedback = VoiceFeedback(
            feedback_type=FeedbackType.CONFIRMATION,
            message=message,
            requires_response=True,
            timeout_seconds=timeout
        )

        response = await self.announce(feedback)

        if not response:
            return False  # Timeout = no

        # Check for affirmative responses
        response_lower = response.lower().strip()
        affirmative = ["yes", "yeah", "yep", "ok", "okay", "sure", "confirm", "do it"]

        return any(word in response_lower for word in affirmative)

    async def announce_progress(
        self,
        message: str,
        task_id: str,
        force: bool = False
    ) -> bool:
        """
        Announce progress update (with throttling)

        Args:
            message: Progress message
            task_id: Task identifier (for throttling)
            force: Force announcement (skip throttling)

        Returns:
            True if announced, False if throttled
        """
        # Check throttling
        if not force and not self._should_announce(task_id):
            return False

        feedback = VoiceFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message=message,
            task_id=task_id
        )

        await self.announce(feedback)
        return True

    async def announce_completion(self, message: str):
        """Announce task completion (immediate, no throttling)"""
        feedback = VoiceFeedback(
            feedback_type=FeedbackType.COMPLETION,
            message=message
        )
        await self.announce(feedback)

    async def announce_error(self, message: str):
        """Announce error (immediate, no throttling)"""
        feedback = VoiceFeedback(
            feedback_type=FeedbackType.ERROR,
            message=message
        )
        await self.announce(feedback)

    def _should_announce(self, task_id: str) -> bool:
        """
        Check if we should announce progress for this task

        Args:
            task_id: Task identifier

        Returns:
            True if enough time has passed since last announcement
        """
        if task_id not in self._last_announcement:
            return True

        elapsed = time.time() - self._last_announcement[task_id]
        return elapsed >= self.throttle_seconds

    async def start_interrupt_detection(
        self,
        interrupt_callback: Callable[[InterruptType], Awaitable[None]]
    ):
        """
        Start listening for interrupts in background

        Args:
            interrupt_callback: Async function to call when interrupt detected
        """
        if self._listening_for_interrupts:
            return  # Already listening

        self._listening_for_interrupts = True
        self._interrupt_task = asyncio.create_task(
            self._interrupt_loop(interrupt_callback)
        )

    async def stop_interrupt_detection(self):
        """Stop listening for interrupts"""
        self._listening_for_interrupts = False

        if self._interrupt_task:
            self._interrupt_task.cancel()
            try:
                await self._interrupt_task
            except asyncio.CancelledError:
                pass
            self._interrupt_task = None

    async def _interrupt_loop(
        self,
        callback: Callable[[InterruptType], Awaitable[None]]
    ):
        """Background loop that listens for interrupts"""
        while self._listening_for_interrupts:
            try:
                # Listen for short utterance
                utterance = await self.stt(timeout=2.0)

                if not utterance:
                    continue  # Timeout, try again

                # Detect interrupt type
                interrupt = self._detect_interrupt(utterance)

                if interrupt != InterruptType.NONE:
                    # Call callback
                    await callback(interrupt)

            except Exception as e:
                # Log error but continue listening
                print(f"[Interrupt detection error] {e}")
                await asyncio.sleep(0.5)

    def _detect_interrupt(self, utterance: str) -> InterruptType:
        """
        Detect interrupt type from utterance

        Args:
            utterance: User speech

        Returns:
            InterruptType detected
        """
        utterance_lower = utterance.lower().strip()

        # Stop/Cancel
        if any(word in utterance_lower for word in ["stop", "cancel", "abort", "quit"]):
            return InterruptType.STOP

        # Pause
        if any(word in utterance_lower for word in ["pause", "wait", "hold"]):
            return InterruptType.PAUSE

        # Resume
        if any(word in utterance_lower for word in ["resume", "continue", "go"]):
            return InterruptType.RESUME

        # Skip
        if any(word in utterance_lower for word in ["skip", "next"]):
            return InterruptType.SKIP

        return InterruptType.NONE

    def reset_throttle(self, task_id: Optional[str] = None):
        """
        Reset throttle timer

        Args:
            task_id: Task to reset (or None for all tasks)
        """
        if task_id is None:
            self._last_announcement.clear()
        elif task_id in self._last_announcement:
            del self._last_announcement[task_id]


# ============================================================================
# Example Usage
# ============================================================================

async def demo_voice_feedback():
    """Demo VoiceFeedbackManager"""
    print("\n" + "="*60)
    print("VOICE FEEDBACK MANAGER DEMO")
    print("="*60)

    # Mock TTS/STT handlers
    async def mock_tts(text: str) -> bool:
        print(f"[TTS] {text}")
        return True

    async def mock_stt(timeout: float) -> str:
        # Simulate user response
        await asyncio.sleep(0.5)
        return "yes"

    # Create feedback manager
    manager = VoiceFeedbackManager(
        tts_handler=mock_tts,
        stt_handler=mock_stt,
        throttle_seconds=2.0
    )

    # Test 1: Progress announcements (throttled)
    print("\n[Test 1: Throttled Progress]")
    task_id = "task_123"

    for i in range(5):
        announced = await manager.announce_progress(
            f"Progress: {(i+1)*20}%",
            task_id
        )
        if announced:
            print(f"  Announced: {(i+1)*20}%")
        else:
            print(f"  Throttled: {(i+1)*20}%")

        await asyncio.sleep(0.5)

    # Test 2: Confirmation prompt
    print("\n[Test 2: Confirmation Prompt]")
    confirmed = await manager.confirm("Delete all files?")
    print(f"  User confirmed: {confirmed}")

    # Test 3: Completion announcement (immediate)
    print("\n[Test 3: Completion Announcement]")
    await manager.announce_completion("Task complete!")

    # Test 4: Error announcement (immediate)
    print("\n[Test 4: Error Announcement]")
    await manager.announce_error("Task failed: Network timeout")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("Voice Feedback Manager")
    print("Manages voice interaction during task execution\n")

    asyncio.run(demo_voice_feedback())

    print("\n[DEMO COMPLETE]")
