"""
Wake Word Detection for Elle Core

Listens for "Hey Elle" or "Elle" to activate listening mode.

Created: 2025-11-15
"""

import asyncio
import threading
from typing import Callable, Optional
import re


class WakeWordDetector:
    """
    Detect wake words to activate Elle

    Supports:
    - "Hey Elle"
    - "Elle"
    - "OK Elle"
    - Fuzzy matching for speech recognition errors
    """

    def __init__(
        self,
        wake_words: Optional[list] = None,
        timeout: float = 30.0,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize wake word detector

        Args:
            wake_words: List of wake words to listen for
            timeout: How long to listen for wake word
            confidence_threshold: Confidence threshold for detection
        """
        self.wake_words = wake_words or [
            "hey elle",
            "ok elle",
            "hey el",
            "okay elle",
            "ella",  # Common speech recognition error
        ]

        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.listening = False
        self.detected = False

    def detect(self, text: str) -> tuple[bool, float]:
        """
        Detect wake word in text

        Args:
            text: Transcribed text to check

        Returns:
            Tuple of (detected: bool, confidence: float 0.0-1.0)
        """
        text_lower = text.lower().strip()

        # Exact matches (high confidence)
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                return True, 1.0
            if text_lower == wake_word:
                return True, 1.0

        # Partial matches with confidence scoring
        best_confidence = 0.0

        for wake_word in self.wake_words:
            confidence = self._calculate_similarity(text_lower, wake_word)
            if confidence > best_confidence:
                best_confidence = confidence

        # Return detection if above threshold
        detected = best_confidence >= self.confidence_threshold
        return detected, best_confidence

    def _calculate_similarity(self, text: str, wake_word: str) -> float:
        """
        Calculate similarity score between text and wake word

        Args:
            text: Input text
            wake_word: Wake word to compare

        Returns:
            Similarity score 0.0-1.0
        """
        text = text.strip()
        wake_word = wake_word.strip()

        # Exact match
        if text == wake_word:
            return 1.0

        # Substring match
        if text.startswith(wake_word):
            # Bonus points for exact start, minus points for extra length
            extra_length = len(text) - len(wake_word)
            similarity = 1.0 - (extra_length / (len(text) + 1))
            return max(0.5, similarity)

        if wake_word in text:
            # Substring match (lower confidence)
            return 0.7

        # Levenshtein distance
        distance = self._levenshtein_distance(text, wake_word)
        max_len = max(len(text), len(wake_word))

        if max_len == 0:
            return 1.0

        # Convert distance to similarity
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of single-character edits)
        """
        if len(s1) < len(s2):
            return WakeWordDetector._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    async def listen_for_wake_word(
        self,
        transcriber_func: Callable,
        on_detected: Optional[Callable] = None
    ) -> bool:
        """
        Listen for wake word using provided transcriber

        Args:
            transcriber_func: Async function that returns transcribed text
            on_detected: Optional callback when wake word detected

        Returns:
            True if wake word detected, False if timeout
        """
        self.listening = True
        self.detected = False

        print(f"🎤 Listening for wake word... (timeout in {self.timeout}s)")

        try:
            # Try transcriber with timeout
            text = await asyncio.wait_for(
                transcriber_func(),
                timeout=self.timeout
            )

            if text:
                detected, confidence = self.detect(text)

                print(f"Heard: '{text}' (confidence: {confidence:.1%})")

                if detected:
                    print("✓ Wake word detected!")
                    self.detected = True

                    if on_detected:
                        await on_detected(text)

                    return True
                else:
                    print("✗ Wake word not detected. Please try again.")
                    return False
            else:
                print("✗ No speech detected. Please try again.")
                return False

        except asyncio.TimeoutError:
            print(f"✗ Timeout waiting for wake word ({self.timeout}s)")
            return False

        finally:
            self.listening = False

    async def continuous_listening(
        self,
        transcriber_func: Callable,
        on_detected: Optional[Callable] = None,
        max_attempts: int = 5
    ):
        """
        Continuously listen for wake word

        Args:
            transcriber_func: Async function that returns transcribed text
            on_detected: Optional callback when wake word detected
            max_attempts: Maximum attempts before giving up
        """
        print(f"🎤 Continuous listening mode (max {max_attempts} attempts)")

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")

            detected = await self.listen_for_wake_word(
                transcriber_func,
                on_detected
            )

            if detected:
                return True

            await asyncio.sleep(1)

        print("✗ Max attempts reached. Exiting listening mode.")
        return False

    def extract_command(self, text: str) -> Optional[str]:
        """
        Extract command text after wake word

        Args:
            text: Full transcribed text including wake word

        Returns:
            Command text without wake word, or None
        """
        text_lower = text.lower()

        # Find which wake word was detected
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                # Remove wake word and strip
                command = text[len(wake_word):].strip()

                # Remove leading punctuation
                command = re.sub(r"^[,:\s]+", "", command)

                return command if command else None

        return None


async def demo_wake_word():
    """Demo: Wake word detection"""
    print("\n" + "="*60)
    print("WAKE WORD DETECTOR DEMO")
    print("="*60)

    detector = WakeWordDetector()

    # Test detection with various inputs
    test_cases = [
        ("Hey Elle, what time is it?", True),
        ("Elle, start baking", True),
        ("OK Elle, show me the bread SOP", True),
        ("Hello there", False),
        ("Hey el", True),  # Fuzzy match
        ("ella", True),    # Fuzzy match
        ("What time is it?", False),
        ("Elle", True),    # Just the wake word
    ]

    print("\nTesting wake word detection:")
    for text, expected in test_cases:
        detected, confidence = detector.detect(text)
        match = "✓" if detected == expected else "✗"
        print(f"{match} '{text}'")
        print(f"   Detected: {detected}, Confidence: {confidence:.1%}")

        # Extract command if detected
        if detected:
            command = detector.extract_command(text)
            if command:
                print(f"   Command: {command}")
        print()


if __name__ == "__main__":
    asyncio.run(demo_wake_word())
