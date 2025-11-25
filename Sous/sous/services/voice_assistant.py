#!/usr/bin/env python3
"""
Voice Assistant for SOUS

Hands-free cooking assistant using HoloLoom's voice capabilities.

Supports:
- Voice commands for recipe navigation
- Hands-free timer management
- Ingredient queries while cooking
- Step-by-step guidance
- Emergency substitution suggestions
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Try to import HoloLoom voice capabilities
try:
    from .hololoom_bridge import SousHoloLoomBridge, HOLOLOOM_AVAILABLE
except ImportError:
    HOLOLOOM_AVAILABLE = False


class VoiceCommandType(Enum):
    """Types of voice commands"""
    RECIPE_NAVIGATION = "recipe_navigation"
    TIMER = "timer"
    INGREDIENT_QUERY = "ingredient_query"
    SUBSTITUTION = "substitution"
    MEASUREMENT = "measurement"
    TEMPERATURE = "temperature"
    GENERAL_HELP = "general_help"


@dataclass
class VoiceCommand:
    """Parsed voice command"""
    command_type: VoiceCommandType
    raw_text: str
    parsed_intent: str
    parameters: Dict[str, Any]
    confidence: float
    user_id: Optional[str] = None  # User who issued the command


@dataclass
class Timer:
    """Active cooking timer"""
    name: str
    duration_seconds: int
    started_at: datetime
    notes: str = ""

    def remaining_seconds(self) -> int:
        """Calculate remaining time"""
        elapsed = (datetime.now() - self.started_at).total_seconds()
        remaining = self.duration_seconds - elapsed
        return max(0, int(remaining))

    def is_expired(self) -> bool:
        """Check if timer has expired"""
        return self.remaining_seconds() == 0


class SousVoiceAssistant:
    """
    Voice assistant for hands-free cooking.

    Integrates with HoloLoom's AI system for intelligent responses.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        household_id: Optional[str] = None,
        user_names: Optional[Dict[str, str]] = None,
        voice_enabled: bool = True,
        use_hololoom: bool = True
    ):
        """
        Initialize voice assistant with multi-user support.

        Args:
            user_id: Initial user ID (for single-user or default user)
            household_id: Household ID for HoloLoom integration
            user_names: Mapping of user_id -> display name for personalized responses
            voice_enabled: Enable voice input/output
            use_hololoom: Use HoloLoom AI for intelligence
        """
        self.voice_enabled = voice_enabled
        self.use_hololoom = use_hololoom and HOLOLOOM_AVAILABLE

        # Multi-user state
        self.current_user_id = user_id
        self.household_id = household_id
        self.user_names = user_names or {}

        # State
        self.current_recipe = None
        self.current_step = 0
        self.active_timers: List[Timer] = []
        self.command_history: List[VoiceCommand] = []

        # HoloLoom bridge (initialized in context manager)
        self.bridge = None

    async def __aenter__(self):
        """Async context manager entry"""
        if self.use_hololoom:
            self.bridge = SousHoloLoomBridge(
                household_id=self.household_id,
                enable_rag=True,
                enable_agentic=True
            )
            await self.bridge.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.bridge:
            await self.bridge.__aexit__(exc_type, exc_val, exc_tb)

    # ========================================================================
    # Multi-User Management
    # ========================================================================

    def switch_user(self, user_id: str, user_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Switch the active user.

        Args:
            user_id: New active user ID
            user_name: Optional display name for the user

        Returns:
            Status message
        """
        old_user = self.current_user_id
        self.current_user_id = user_id

        if user_name:
            self.user_names[user_id] = user_name

        return {
            "status": "success",
            "previous_user": old_user,
            "current_user": user_id,
            "display_name": self.user_names.get(user_id, user_id),
            "message": f"Switched to {self.user_names.get(user_id, user_id)}"
        }

    def get_current_user_name(self) -> str:
        """Get display name for current user"""
        if not self.current_user_id:
            return "there"
        return self.user_names.get(self.current_user_id, self.current_user_id)

    # ========================================================================
    # Voice Command Processing
    # ========================================================================

    async def process_voice_command(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a voice command with user identification.

        Args:
            text: Voice command text
            user_id: Optional user ID (overrides current_user_id)

        Returns:
            Response with action taken
        """
        # Determine which user is issuing the command
        active_user_id = user_id if user_id else self.current_user_id

        # Parse command
        command = self._parse_command(text, active_user_id)
        self.command_history.append(command)

        # Route to appropriate handler
        handlers = {
            VoiceCommandType.RECIPE_NAVIGATION: self._handle_recipe_navigation,
            VoiceCommandType.TIMER: self._handle_timer,
            VoiceCommandType.INGREDIENT_QUERY: self._handle_ingredient_query,
            VoiceCommandType.SUBSTITUTION: self._handle_substitution,
            VoiceCommandType.MEASUREMENT: self._handle_measurement,
            VoiceCommandType.TEMPERATURE: self._handle_temperature,
            VoiceCommandType.GENERAL_HELP: self._handle_general_help,
        }

        handler = handlers.get(command.command_type, self._handle_unknown)
        response = await handler(command)

        return {
            "command": text,
            "user_id": active_user_id,
            "user_name": self.user_names.get(active_user_id, active_user_id) if active_user_id else None,
            "understood": command.confidence > 0.5,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

    def _parse_command(self, text: str, user_id: Optional[str] = None) -> VoiceCommand:
        """Parse voice command text into structured command with user context"""
        text_lower = text.lower()

        # Recipe navigation commands
        if any(word in text_lower for word in ["next step", "continue", "what's next"]):
            return VoiceCommand(
                command_type=VoiceCommandType.RECIPE_NAVIGATION,
                raw_text=text,
                parsed_intent="next_step",
                parameters={},
                confidence=0.9,
                user_id=user_id
            )

        if any(word in text_lower for word in ["previous step", "go back", "repeat"]):
            return VoiceCommand(
                command_type=VoiceCommandType.RECIPE_NAVIGATION,
                raw_text=text,
                parsed_intent="previous_step",
                parameters={},
                confidence=0.9,
                user_id=user_id
            )

        if "start recipe" in text_lower or "begin cooking" in text_lower:
            return VoiceCommand(
                command_type=VoiceCommandType.RECIPE_NAVIGATION,
                raw_text=text,
                parsed_intent="start_recipe",
                parameters={},
                confidence=0.9,
                user_id=user_id
            )

        # Timer commands
        if any(word in text_lower for word in ["set timer", "timer for", "start timer"]):
            # Extract duration
            duration_seconds = self._extract_duration(text)
            timer_name = self._extract_timer_name(text)

            return VoiceCommand(
                command_type=VoiceCommandType.TIMER,
                raw_text=text,
                parsed_intent="set_timer",
                parameters={
                    "duration_seconds": duration_seconds,
                    "name": timer_name
                },
                confidence=0.85,
                user_id=user_id
            )

        if any(word in text_lower for word in ["check timer", "time remaining", "how long"]):
            return VoiceCommand(
                command_type=VoiceCommandType.TIMER,
                raw_text=text,
                parsed_intent="check_timer",
                parameters={},
                confidence=0.9,
                user_id=user_id
            )

        if any(word in text_lower for word in ["cancel timer", "stop timer"]):
            return VoiceCommand(
                command_type=VoiceCommandType.TIMER,
                raw_text=text,
                parsed_intent="cancel_timer",
                parameters={},
                confidence=0.9,
                user_id=user_id
            )

        # Ingredient queries
        if any(word in text_lower for word in ["how much", "how many", "what ingredient"]):
            return VoiceCommand(
                command_type=VoiceCommandType.INGREDIENT_QUERY,
                raw_text=text,
                parsed_intent="ingredient_quantity",
                parameters={},
                confidence=0.8,
                user_id=user_id
            )

        # Substitution requests
        if any(word in text_lower for word in ["substitute", "instead of", "replacement"]):
            return VoiceCommand(
                command_type=VoiceCommandType.SUBSTITUTION,
                raw_text=text,
                parsed_intent="find_substitution",
                parameters={"query": text},
                confidence=0.85,
                user_id=user_id
            )

        # Measurement conversions
        if any(word in text_lower for word in ["convert", "cups to", "ounces to"]):
            return VoiceCommand(
                command_type=VoiceCommandType.MEASUREMENT,
                raw_text=text,
                parsed_intent="convert_measurement",
                parameters={"query": text},
                confidence=0.8,
                user_id=user_id
            )

        # Temperature conversions
        if any(word in text_lower for word in ["fahrenheit", "celsius", "degrees"]):
            return VoiceCommand(
                command_type=VoiceCommandType.TEMPERATURE,
                raw_text=text,
                parsed_intent="convert_temperature",
                parameters={"query": text},
                confidence=0.8,
                user_id=user_id
            )

        # General help
        if any(word in text_lower for word in ["help", "what can you do"]):
            return VoiceCommand(
                command_type=VoiceCommandType.GENERAL_HELP,
                raw_text=text,
                parsed_intent="show_help",
                parameters={},
                confidence=0.95,
                user_id=user_id
            )

        # Unknown command
        return VoiceCommand(
            command_type=VoiceCommandType.GENERAL_HELP,
            raw_text=text,
            parsed_intent="unknown",
            parameters={"query": text},
            confidence=0.3,
            user_id=user_id
        )

    # ========================================================================
    # Command Handlers
    # ========================================================================

    async def _handle_recipe_navigation(self, command: VoiceCommand) -> str:
        """Handle recipe navigation commands with personalized responses"""
        user_name = self.user_names.get(command.user_id, "") if command.user_id else ""
        greeting = f"{user_name}, " if user_name else ""

        if not self.current_recipe:
            return f"{greeting}No recipe is currently active. Please start a recipe first."

        intent = command.parsed_intent

        if intent == "start_recipe":
            self.current_step = 0
            return self._get_current_step_text()

        elif intent == "next_step":
            if self.current_step < len(self.current_recipe.get('steps', [])) - 1:
                self.current_step += 1
                return self._get_current_step_text()
            else:
                return f"{greeting}You've completed all steps! Great job cooking!"

        elif intent == "previous_step":
            if self.current_step > 0:
                self.current_step -= 1
                return self._get_current_step_text()
            else:
                return f"{greeting}You're already at the first step."

        return f"{greeting}I didn't understand that navigation command."

    async def _handle_timer(self, command: VoiceCommand) -> str:
        """Handle timer commands"""
        intent = command.parsed_intent

        if intent == "set_timer":
            duration = command.parameters.get("duration_seconds", 0)
            name = command.parameters.get("name", "Timer")

            if duration <= 0:
                return "Sorry, I couldn't understand the timer duration. Please say something like 'set timer for 15 minutes'."

            timer = Timer(
                name=name,
                duration_seconds=duration,
                started_at=datetime.now()
            )
            self.active_timers.append(timer)

            minutes = duration // 60
            seconds = duration % 60
            time_str = f"{minutes} minutes" if seconds == 0 else f"{minutes} minutes and {seconds} seconds"

            return f"Timer '{name}' set for {time_str}. I'll let you know when it's done!"

        elif intent == "check_timer":
            if not self.active_timers:
                return "No active timers."

            responses = []
            for timer in self.active_timers:
                remaining = timer.remaining_seconds()
                if remaining > 0:
                    minutes = remaining // 60
                    seconds = remaining % 60
                    responses.append(f"{timer.name}: {minutes}m {seconds}s remaining")
                else:
                    responses.append(f"{timer.name}: Time's up!")

            return "\n".join(responses)

        elif intent == "cancel_timer":
            if self.active_timers:
                cancelled = self.active_timers.pop()
                return f"Cancelled timer: {cancelled.name}"
            return "No active timers to cancel."

        return "I didn't understand that timer command."

    async def _handle_ingredient_query(self, command: VoiceCommand) -> str:
        """Handle ingredient queries"""
        if not self.current_recipe:
            return "No recipe is currently active."

        # Get current step ingredients
        step = self.current_recipe.get('steps', [])[self.current_step]
        step_text = step.get('description', '')

        # Use HoloLoom RAG if available
        if self.bridge and self.bridge.rag:
            result = await self.bridge.rag.query(
                f"In this cooking step: '{step_text}', answer this question: {command.raw_text}"
            )
            return result.response

        # Fallback: return step text
        return f"Current step: {step_text}"

    async def _handle_substitution(self, command: VoiceCommand) -> str:
        """Handle substitution requests"""
        query = command.parameters.get("query", "")

        # Use HoloLoom RAG for intelligent substitutions
        if self.bridge and self.bridge.rag:
            result = await self.bridge.suggest_substitutions(
                missing_ingredient=query,
                recipe_context=self.current_recipe.get('name', '') if self.current_recipe else ""
            )

            if result['status'] == 'success':
                return result['response']

        # Fallback: basic substitutions
        basic_subs = {
            "butter": "Use oil or margarine (3/4 cup oil for 1 cup butter)",
            "egg": "Use 1/4 cup applesauce or mashed banana per egg",
            "milk": "Use water, plant milk, or cream",
            "sugar": "Use honey (3/4 cup honey for 1 cup sugar)"
        }

        for ingredient, sub in basic_subs.items():
            if ingredient in query.lower():
                return sub

        return "I don't have a substitution suggestion for that. Try searching online or asking a more specific question."

    async def _handle_measurement(self, command: VoiceCommand) -> str:
        """Handle measurement conversions"""
        query = command.parameters.get("query", "").lower()

        # Common conversions
        conversions = {
            "cups to tablespoons": "1 cup = 16 tablespoons",
            "tablespoons to teaspoons": "1 tablespoon = 3 teaspoons",
            "cups to ounces": "1 cup = 8 fluid ounces",
            "ounces to grams": "1 ounce = 28.35 grams",
            "pounds to kilograms": "1 pound = 0.45 kilograms"
        }

        for pattern, result in conversions.items():
            if pattern in query:
                return result

        # Use HoloLoom for complex conversions
        if self.bridge and self.bridge.rag:
            result = await self.bridge.rag.query(
                f"Convert this measurement: {query}"
            )
            return result.response

        return "I couldn't understand that conversion. Try being more specific, like 'convert 2 cups to tablespoons'."

    async def _handle_temperature(self, command: VoiceCommand) -> str:
        """Handle temperature conversions"""
        query = command.parameters.get("query", "")

        # Extract numbers
        import re
        numbers = re.findall(r'\d+', query)

        if numbers:
            temp = int(numbers[0])

            if "fahrenheit" in query.lower() or "f" in query.lower():
                celsius = (temp - 32) * 5/9
                return f"{temp}°F is {celsius:.0f}°C"

            elif "celsius" in query.lower() or "c" in query.lower():
                fahrenheit = (temp * 9/5) + 32
                return f"{temp}°C is {fahrenheit:.0f}°F"

        return "I couldn't understand that temperature. Try saying something like '350 degrees fahrenheit to celsius'."

    async def _handle_general_help(self, command: VoiceCommand) -> str:
        """Handle general help requests"""
        if command.parsed_intent == "unknown" and self.bridge and self.bridge.rag:
            # Use HoloLoom for unknown queries
            result = await self.bridge.rag.query(command.parameters.get("query", command.raw_text))
            return result.response

        return """I can help you with:
- Recipe navigation: "next step", "previous step", "start recipe"
- Timers: "set timer for 10 minutes", "check timer", "cancel timer"
- Ingredients: "how much sugar?", "what's in this step?"
- Substitutions: "what can I use instead of butter?"
- Measurements: "convert 2 cups to tablespoons"
- Temperature: "350 fahrenheit to celsius"

Just talk naturally and I'll do my best to help!"""

    async def _handle_unknown(self, command: VoiceCommand) -> str:
        """Handle unknown commands"""
        if self.bridge and self.bridge.rag:
            result = await self.bridge.rag.query(command.raw_text)
            return result.response

        return "I didn't understand that command. Say 'help' to see what I can do."

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_current_step_text(self) -> str:
        """Get current recipe step text"""
        if not self.current_recipe:
            return "No recipe loaded."

        steps = self.current_recipe.get('steps', [])
        if self.current_step >= len(steps):
            return "Recipe complete!"

        step = steps[self.current_step]
        step_num = self.current_step + 1
        total_steps = len(steps)

        return f"Step {step_num} of {total_steps}: {step.get('description', 'No description')}"

    def _extract_duration(self, text: str) -> int:
        """Extract duration in seconds from text"""
        import re

        # Extract numbers
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return 0

        duration = int(numbers[0])

        # Determine unit
        if any(word in text.lower() for word in ["hour", "hours", "hr"]):
            return duration * 3600
        elif any(word in text.lower() for word in ["minute", "minutes", "min"]):
            return duration * 60
        elif any(word in text.lower() for word in ["second", "seconds", "sec"]):
            return duration
        else:
            # Default to minutes
            return duration * 60

    def _extract_timer_name(self, text: str) -> str:
        """Extract timer name from text"""
        # Look for "for [name]" pattern
        import re
        match = re.search(r'for (\w+)', text.lower())
        if match:
            return match.group(1).capitalize()

        # Default name based on current step
        if self.current_recipe:
            return f"Step {self.current_step + 1}"

        return "Timer"

    # ========================================================================
    # Recipe Loading
    # ========================================================================

    def load_recipe(
        self,
        recipe: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a recipe for voice-guided cooking.

        Args:
            recipe: Recipe dictionary
            user_id: Optional user who is loading the recipe

        Returns:
            Status message
        """
        self.current_recipe = recipe
        self.current_step = 0

        # Get user name for personalized message
        active_user = user_id if user_id else self.current_user_id
        user_name = self.user_names.get(active_user, "") if active_user else ""
        greeting = f"{user_name}, " if user_name else ""

        return {
            "status": "success",
            "recipe_name": recipe.get('name', 'Unknown'),
            "total_steps": len(recipe.get('steps', [])),
            "loaded_by": active_user,
            "message": f"{greeting}Recipe '{recipe.get('name')}' loaded. Say 'start recipe' to begin."
        }

    # ========================================================================
    # Status Methods
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current voice assistant status with user context"""
        return {
            "voice_enabled": self.voice_enabled,
            "hololoom_active": self.bridge is not None,
            "current_user_id": self.current_user_id,
            "current_user_name": self.user_names.get(self.current_user_id) if self.current_user_id else None,
            "household_id": self.household_id,
            "registered_users": len(self.user_names),
            "current_recipe": self.current_recipe.get('name') if self.current_recipe else None,
            "current_step": self.current_step if self.current_recipe else None,
            "active_timers": len(self.active_timers),
            "commands_processed": len(self.command_history),
            "commands_by_user": self._get_command_counts_by_user()
        }

    def _get_command_counts_by_user(self) -> Dict[str, int]:
        """Get command counts grouped by user"""
        counts = {}
        for cmd in self.command_history:
            user_id = cmd.user_id or "unknown"
            counts[user_id] = counts.get(user_id, 0) + 1
        return counts
