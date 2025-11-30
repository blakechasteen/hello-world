"""
Ollama Client - Local LLM integration for NeuroHood

Enables residents to have real AI-powered conversations using local models.

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import aiohttp


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"  # Fast, good for dialogue
    temperature: float = 0.8  # Creative but coherent
    max_tokens: int = 100  # Short responses for natural dialogue
    timeout: float = 30.0


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from Ollama."""
        session = await self._get_session()

        payload = {
            "model": self.config.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }

        try:
            async with session.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error_text}")

                result = await response.json()
                return result.get("response", "").strip()

        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Chat-style generation with message history."""
        session = await self._get_session()

        # Build messages list
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        try:
            async with session.post(
                f"{self.config.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error_text}")

                result = await response.json()
                return result.get("message", {}).get("content", "").strip()

        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as response:
                if response.status == 200:
                    result = await response.json()
                    return [m["name"] for m in result.get("models", [])]
        except Exception:
            pass
        return []


# =============================================================================
# DIALOGUE GENERATOR WITH OLLAMA
# =============================================================================

class OllamaDialogueGenerator:
    """LLM-powered dialogue using Ollama."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self._available: Optional[bool] = None

    async def check_availability(self) -> bool:
        """Check if Ollama is available."""
        if self._available is None:
            self._available = await self.client.is_available()
        return self._available

    def _build_personality_prompt(self, agent: "AgentState") -> str:
        """Build personality prompt from agent state."""
        traits = []

        # Big Five interpretation
        if agent.personality.get("openness", 0.5) > 0.6:
            traits.append("creative, curious, and imaginative")
        elif agent.personality.get("openness", 0.5) < 0.4:
            traits.append("practical and down-to-earth")

        if agent.personality.get("extraversion", 0.5) > 0.6:
            traits.append("outgoing, warm, and talkative")
        elif agent.personality.get("extraversion", 0.5) < 0.4:
            traits.append("quiet, thoughtful, and reserved")

        if agent.personality.get("agreeableness", 0.5) > 0.6:
            traits.append("kind, cooperative, and caring")
        elif agent.personality.get("agreeableness", 0.5) < 0.4:
            traits.append("direct and competitive")

        if agent.personality.get("neuroticism", 0.5) > 0.6:
            traits.append("sensitive and emotionally aware")
        elif agent.personality.get("neuroticism", 0.5) < 0.4:
            traits.append("calm and emotionally stable")

        # Current emotional state
        emotions = []
        if agent.emotions.get("happiness", 0) > 0.3:
            emotions.append("happy")
        elif agent.emotions.get("happiness", 0) < -0.3:
            emotions.append("sad")
        if agent.emotions.get("anxiety", 0) > 0.3:
            emotions.append("a bit anxious")
        if agent.emotions.get("loneliness", 0) > 0.3:
            emotions.append("lonely")
        if agent.emotions.get("energy", 0.5) < 0.3:
            emotions.append("tired")

        prompt = f"""You are {agent.name}, a resident of a small neighborhood.

Personality: You are {', '.join(traits) if traits else 'a balanced individual'}.
Current mood: You're feeling {', '.join(emotions) if emotions else 'neutral'}.

Speaking style:
- Keep responses SHORT (1-2 sentences max)
- Be natural and conversational
- Express your personality through word choice
- React authentically to what the other person says
- You can use *actions* like *smiles* or *looks thoughtful*

DO NOT:
- Be overly formal or robotic
- Give long explanations
- Ask multiple questions at once
- Break character"""

        # Add recent memories for context
        if agent.memories:
            recent = agent.memories[-3:]
            memory_text = "; ".join(m.get("description", "") for m in recent if isinstance(m, dict))
            if memory_text:
                prompt += f"\n\nRecent experiences: {memory_text}"

        return prompt

    async def generate_line(
        self,
        speaker: "AgentState",
        listener: "AgentState",
        context: Dict[str, Any],
        conversation_history: List["DialogueLine"]
    ) -> str:
        """Generate a dialogue line using Ollama."""

        # Build system prompt
        system_prompt = self._build_personality_prompt(speaker)

        # Build conversation context
        location = context.get("location", "the neighborhood")
        time_of_day = context.get("time_of_day", "day")

        # Format history
        history_lines = []
        for line in conversation_history[-6:]:  # Last 6 lines
            history_lines.append(f"{line.speaker_name}: {line.text}")
        history_text = "\n".join(history_lines) if history_lines else "(conversation just started)"

        # Build user prompt
        user_prompt = f"""You're talking with {listener.name} at {location} ({time_of_day}).

Conversation so far:
{history_text}

Now respond as {speaker.name}. Keep it short and natural (1-2 sentences):"""

        try:
            response = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=80
            )

            # Clean up response
            response = response.strip()

            # Remove any "Name:" prefix if the model added it
            if response.startswith(f"{speaker.name}:"):
                response = response[len(speaker.name) + 1:].strip()

            # Remove quotes if wrapped
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            return response if response else "..."

        except Exception as e:
            print(f"[Ollama] Error: {e}")
            # Fallback to simple template
            return self._fallback_response(speaker, listener)

    def _fallback_response(self, speaker: "AgentState", listener: "AgentState") -> str:
        """Fallback if LLM fails."""
        import random
        templates = [
            f"Hey {listener.name}!",
            "How's it going?",
            "*nods thoughtfully*",
            "That's interesting.",
            "I was just thinking about that.",
        ]
        return random.choice(templates)


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Test Ollama integration."""
    print("=" * 60)
    print("OLLAMA CLIENT TEST")
    print("=" * 60)

    client = OllamaClient()

    # Check availability
    print("\nChecking Ollama availability...")
    available = await client.is_available()

    if not available:
        print("Ollama is not running!")
        print("Start it with: ollama serve")
        print("Then pull a model: ollama pull llama3.2:3b")
        return

    print("Ollama is running!")

    # List models
    models = await client.list_models()
    print(f"Available models: {models}")

    # Test generation
    print("\n--- Test Generation ---")
    response = await client.generate(
        system_prompt="You are Maya, a creative artist who speaks thoughtfully.",
        user_prompt="Someone just asked you about your latest painting. Respond briefly.",
        max_tokens=50
    )
    print(f"Maya: {response}")

    # Test dialogue generator
    print("\n--- Test Dialogue Generator ---")

    # Create mock agent states
    from dataclasses import dataclass, field
    from typing import Dict, List, Any

    @dataclass
    class MockAgent:
        agent_id: str
        name: str
        personality: Dict[str, float] = field(default_factory=dict)
        emotions: Dict[str, float] = field(default_factory=dict)
        memories: List[Dict] = field(default_factory=list)

    @dataclass
    class MockLine:
        speaker_name: str
        text: str

    maya = MockAgent(
        agent_id="maya_001",
        name="Maya",
        personality={"openness": 0.9, "extraversion": 0.4, "agreeableness": 0.7},
        emotions={"happiness": 0.3, "energy": 0.6},
        memories=[{"description": "Just finished a new painting"}]
    )

    jorge = MockAgent(
        agent_id="jorge_002",
        name="Jorge",
        personality={"openness": 0.6, "extraversion": 0.7, "agreeableness": 0.8},
        emotions={"happiness": 0.5}
    )

    generator = OllamaDialogueGenerator(client)

    # Simulate conversation
    history = []

    # Jorge starts
    print("\n[At the Corner Cafe]")
    line1 = MockLine("Jorge", "Maya! I haven't seen you in a while. How's the art going?")
    history.append(line1)
    print(f"Jorge: \"{line1.text}\"")

    # Maya responds
    response = await generator.generate_line(
        speaker=maya,
        listener=jorge,
        context={"location": "Corner Cafe", "time_of_day": "afternoon"},
        conversation_history=history
    )
    line2 = MockLine("Maya", response)
    history.append(line2)
    print(f"Maya: \"{response}\"")

    # Jorge responds
    response2 = await generator.generate_line(
        speaker=jorge,
        listener=maya,
        context={"location": "Corner Cafe", "time_of_day": "afternoon"},
        conversation_history=history
    )
    print(f"Jorge: \"{response2}\"")

    await client.close()
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
