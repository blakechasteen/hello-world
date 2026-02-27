#!/usr/bin/env python3
"""
Voice-Enhanced Live Scratchpad
===============================

Conversational scratchpad with TTS feedback for:
- Human-in-the-loop confirmations
- Clarity requests for ambiguous data
- Voice-guided field completion
- Read-back verification

Integrates:
- Safety Guardrails (HiTL for high-risk actions)
- TTS (text-to-speech feedback)
- Whisper (speech-to-text input)

Author: Claude Code
Date: November 2025
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import re

from hololoom.spinningWheel.live_scratchpad import LiveScratchpad, TemplateField
from hololoom.alignment.safety_guardrails import SafetyGuardrails, RiskLevel, ActionCategory


# ============================================================================
# Voice Interaction Types
# ============================================================================

@dataclass
class VoicePrompt:
    """TTS prompt for user interaction"""
    text: str
    prompt_type: str  # 'confirmation', 'clarification', 'guidance', 'readback'
    field_name: Optional[str] = None
    expected_response: Optional[str] = None  # 'yes/no', 'value', 'choice'
    choices: Optional[List[str]] = None
    timeout_seconds: float = 30.0


@dataclass
class VoiceResponse:
    """User's voice response"""
    transcription: str
    understood: bool
    extracted_value: Any
    confidence: float


# ============================================================================
# Voice-Enhanced Scratchpad
# ============================================================================

class VoiceScratchpad(LiveScratchpad):
    """
    Conversational scratchpad with voice feedback.

    Features:
    - TTS prompts for clarity
    - Voice confirmations for high-risk actions
    - Read-back verification
    - Guided field completion

    Usage:
        scratchpad = VoiceScratchpad(enable_voice=True)
        await scratchpad.start_recording('recipe')

        # Voice interaction happens automatically:
        # TTS: "Did you say 24 servings?"
        # User: "Yes"
        # TTS: "Great! Recipe saved."
    """

    def __init__(
        self,
        enable_voice: bool = True,
        enable_safety_confirmations: bool = True,
        enable_clarity_requests: bool = True,
        enable_readback: bool = True,
        tts_voice: str = "nova"  # OpenAI TTS voice
    ):
        """
        Initialize voice-enhanced scratchpad.

        Args:
            enable_voice: Enable TTS feedback
            enable_safety_confirmations: Ask for confirmation on high-risk actions
            enable_clarity_requests: Ask for clarification on ambiguous data
            enable_readback: Read back final entry for verification
            tts_voice: TTS voice to use (nova, alloy, echo, fable, onyx, shimmer)
        """
        super().__init__()

        self.enable_voice = enable_voice
        self.enable_safety_confirmations = enable_safety_confirmations
        self.enable_clarity_requests = enable_clarity_requests
        self.enable_readback = enable_readback
        self.tts_voice = tts_voice

        # Safety guardrails for HiTL
        self.safety = SafetyGuardrails(testing_mode=False) if enable_safety_confirmations else None

        # Voice interaction callbacks
        self.on_voice_prompt: Optional[Callable[[VoicePrompt], None]] = None
        self.on_voice_response: Optional[Callable[[VoiceResponse], None]] = None

        # Interaction history
        self.voice_history: List[Tuple[VoicePrompt, VoiceResponse]] = []

    def register_voice_callbacks(
        self,
        on_prompt: Callable[[VoicePrompt], None],
        on_response: Callable[[VoiceResponse], None]
    ):
        """Register callbacks for voice interactions"""
        self.on_voice_prompt = on_prompt
        self.on_voice_response = on_response

    async def _speak(self, text: str, prompt_type: str = 'guidance') -> VoicePrompt:
        """
        Speak text via TTS.

        Args:
            text: Text to speak
            prompt_type: Type of prompt

        Returns:
            VoicePrompt object
        """
        prompt = VoicePrompt(
            text=text,
            prompt_type=prompt_type
        )

        if self.enable_voice and self.on_voice_prompt:
            self.on_voice_prompt(prompt)

        return prompt

    async def _ask_confirmation(
        self,
        question: str,
        field_name: Optional[str] = None
    ) -> bool:
        """
        Ask yes/no confirmation via voice.

        TTS: "Did you say 24 servings?"
        User: "Yes" / "No"

        Returns:
            True if confirmed, False otherwise
        """
        prompt = VoicePrompt(
            text=question,
            prompt_type='confirmation',
            field_name=field_name,
            expected_response='yes/no'
        )

        if self.enable_voice and self.on_voice_prompt:
            self.on_voice_prompt(prompt)

        # TODO: Wait for voice response
        # For now, return True (auto-confirm)
        # In production, would wait for user's voice input
        return True

    async def _ask_clarification(
        self,
        field_name: str,
        prompt_text: str,
        choices: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Ask for clarification on ambiguous field.

        TTS: "I heard 'Hive tree'. Did you mean 'Hive 3'?"
        User: "Yes"

        Returns:
            Clarified value or None
        """
        prompt = VoicePrompt(
            text=prompt_text,
            prompt_type='clarification',
            field_name=field_name,
            expected_response='choice' if choices else 'value',
            choices=choices
        )

        if self.enable_voice and self.on_voice_prompt:
            self.on_voice_prompt(prompt)

        # TODO: Wait for voice response
        # For now, return first choice
        return choices[0] if choices else None

    async def _readback_entry(self) -> bool:
        """
        Read back complete entry for verification.

        TTS: "Recipe for chocolate chip cookies. Serves 24. Ingredients: butter, sugar..."
        TTS: "Is this correct?"
        User: "Yes" / "No, edit servings"

        Returns:
            True if verified, False if needs editing
        """
        if not self.enable_readback:
            return True

        # Build readable summary
        summary_parts = []

        if self.active_template:
            summary_parts.append(f"{self.active_template.name.replace('_', ' ').title()} entry.")

            for field in self.active_template.fields:
                value = self.current_data.get(field.name)
                if value:
                    field_label = field.name.replace('_', ' ').title()
                    if isinstance(value, list):
                        summary_parts.append(f"{field_label}: {', '.join(str(v) for v in value)}.")
                    else:
                        summary_parts.append(f"{field_label}: {value}.")

        summary = " ".join(summary_parts)

        # Speak summary
        await self._speak(summary, prompt_type='readback')

        # Ask for confirmation
        confirmed = await self._ask_confirmation("Is this correct?")

        return confirmed

    async def update_from_transcription(
        self,
        new_transcription: str,
        detected_domain: Optional[str] = None
    ):
        """
        Update template from transcription with voice feedback.

        Adds:
        - Confirmation for ambiguous values
        - Clarification requests
        - Field completion guidance
        """
        # Standard update
        await super().update_from_transcription(new_transcription, detected_domain)

        if not self.enable_voice:
            return

        # Voice feedback for populated fields
        if self.active_template:
            # Find newly populated fields
            for field in self.active_template.fields:
                value = self.current_data.get(field.name)

                if value and self.enable_clarity_requests:
                    # Check for ambiguous values
                    if await self._needs_clarification(field, value):
                        clarified = await self._clarify_field(field, value)
                        if clarified:
                            self.current_data[field.name] = clarified

            # Guide user on required missing fields
            missing_required = [
                f for f in self.active_template.fields
                if f.required and not self.current_data.get(f.name)
            ]

            if missing_required:
                field_names = ', '.join(f.name.replace('_', ' ') for f in missing_required[:3])
                await self._speak(
                    f"I still need: {field_names}.",
                    prompt_type='guidance'
                )

    async def _needs_clarification(self, field: TemplateField, value: Any) -> bool:
        """Check if field value needs clarification"""
        # Numeric fields with unusual values
        if field.field_type == 'number' and isinstance(value, (int, float)):
            if value > 1000:  # Unusually large
                return True

        # Currency with typos
        if field.field_type == 'currency':
            if not re.match(r'^\d+\.\d{2}$', str(value)):
                return True

        # List fields with single item (might be parsing error)
        if field.field_type == 'list' and isinstance(value, list):
            if len(value) == 1 and len(value[0]) > 50:  # One very long item
                return True

        return False

    async def _clarify_field(self, field: TemplateField, value: Any) -> Optional[Any]:
        """Request clarification for ambiguous field"""
        field_label = field.name.replace('_', ' ').title()

        if field.field_type == 'number':
            confirmed = await self._ask_confirmation(
                f"Did you say {value} for {field_label}? That seems high.",
                field_name=field.name
            )
            return value if confirmed else None

        elif field.field_type == 'currency':
            # Suggest correction
            try:
                corrected = f"{float(value):.2f}"
                confirmed = await self._ask_confirmation(
                    f"Did you mean ${corrected} for {field_label}?",
                    field_name=field.name
                )
                return corrected if confirmed else None
            except ValueError:
                return None

        elif field.field_type == 'list':
            await self._speak(
                f"I detected one item for {field_label}. Say 'add more' if you have additional items.",
                prompt_type='clarification'
            )
            return value

        return value

    async def finalize(self, skip_readback: bool = False) -> Any:
        """
        Finalize entry with voice confirmation.

        Adds:
        - Read-back verification
        - Safety confirmation for high-risk entries
        - Success feedback

        Args:
            skip_readback: Skip read-back verification

        Returns:
            MemoryShard
        """
        # Safety check for high-risk domains
        if self.enable_safety_confirmations and self.safety:
            risk_assessment = await self._assess_entry_risk()

            if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                # Voice confirmation for high-risk
                await self._speak(
                    f"This is a {risk_assessment.risk_level.value} risk entry. {risk_assessment.reason}",
                    prompt_type='confirmation'
                )

                confirmed = await self._ask_confirmation(
                    "Are you sure you want to save this?"
                )

                if not confirmed:
                    raise ValueError("Entry cancelled by user")

        # Read back entry
        if not skip_readback:
            verified = await self._readback_entry()
            if not verified:
                await self._speak("Let me know what to change.", prompt_type='guidance')
                raise ValueError("Entry not verified by user")

        # Finalize
        shard = await super().finalize()

        # Success feedback
        await self._speak(
            f"{self.active_domain.replace('_', ' ').title()} entry saved successfully!",
            prompt_type='guidance'
        )

        return shard

    async def _assess_entry_risk(self) -> Any:
        """Assess risk level of entry using safety guardrails"""
        # Map domains to risk levels
        domain_risks = {
            'expense': (ActionCategory.FINANCIAL, "Financial transaction"),
            'budget': (ActionCategory.FINANCIAL, "Budget entry"),
            'sop': (ActionCategory.SYSTEM, "Standard operating procedure"),
            'bee_inspection': (ActionCategory.STORAGE, "Bee inspection record"),
            'recipe': (ActionCategory.STORAGE, "Recipe entry"),
            'time_tracking': (ActionCategory.STORAGE, "Time tracking entry"),
        }

        if self.active_domain in domain_risks:
            category, description = domain_risks[self.active_domain]

            # Use safety guardrails
            from hololoom.alignment.safety_guardrails import ActionRequest
            request = ActionRequest(
                action=f"save_{self.active_domain}",
                category=category,
                context={
                    'domain': self.active_domain,
                    'data': self.current_data
                }
            )
            decision = self.safety.check_action(request)

            return decision

        # Default: low risk
        from hololoom.alignment.safety_guardrails import SafetyDecision
        return SafetyDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Standard data entry"
        )


# ============================================================================
# Voice Interaction Helpers
# ============================================================================

class VoiceInteractionManager:
    """
    Manages voice interactions between scratchpad and UI.

    Handles:
    - TTS queue
    - Voice response waiting
    - Conversation flow
    """

    def __init__(self):
        self.pending_prompt: Optional[VoicePrompt] = None
        self.response_queue: asyncio.Queue = asyncio.Queue()

    async def prompt_user(self, prompt: VoicePrompt) -> VoiceResponse:
        """
        Prompt user via TTS and wait for voice response.

        Args:
            prompt: Voice prompt to speak

        Returns:
            User's voice response
        """
        self.pending_prompt = prompt

        # In production, would:
        # 1. Send prompt to TTS API
        # 2. Play audio
        # 3. Wait for user's voice response
        # 4. Transcribe with Whisper
        # 5. Parse response

        # For now, simulate response
        response = await self._simulate_response(prompt)

        self.pending_prompt = None
        return response

    async def _simulate_response(self, prompt: VoicePrompt) -> VoiceResponse:
        """Simulate user voice response (for testing)"""
        if prompt.expected_response == 'yes/no':
            return VoiceResponse(
                transcription="yes",
                understood=True,
                extracted_value=True,
                confidence=0.95
            )
        elif prompt.expected_response == 'choice' and prompt.choices:
            return VoiceResponse(
                transcription=prompt.choices[0],
                understood=True,
                extracted_value=prompt.choices[0],
                confidence=0.90
            )
        else:
            return VoiceResponse(
                transcription="",
                understood=False,
                extracted_value=None,
                confidence=0.0
            )


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_voice_scratchpad(
    enable_all: bool = True,
    tts_voice: str = "nova"
) -> VoiceScratchpad:
    """
    Create voice-enhanced scratchpad with default settings.

    Args:
        enable_all: Enable all voice features
        tts_voice: TTS voice to use

    Returns:
        VoiceScratchpad instance
    """
    return VoiceScratchpad(
        enable_voice=enable_all,
        enable_safety_confirmations=enable_all,
        enable_clarity_requests=enable_all,
        enable_readback=enable_all,
        tts_voice=tts_voice
    )
