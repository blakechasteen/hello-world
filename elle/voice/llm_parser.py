"""
LLM-Powered Natural Language Parser for Voice Commands

Parses natural language variations into structured commands.
Supports semantic understanding for commands like "increase proofing"
vs "make it longer" vs "bump it up".

Created: 2025-11-15
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import re


@dataclass
class ParsedCommand:
    """Structured representation of a parsed command"""
    command_type: str  # "sop_update", "query", "task_start", etc.
    entity: Optional[str] = None  # "bread", "biochar", etc.
    action: Optional[str] = None  # "increase", "change", "update", etc.
    intent: Optional[str] = None  # "increase_duration", "change_temperature", etc.
    parameters: Dict[str, Any] = None
    confidence: float = 1.0
    raw_text: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self):
        return asdict(self)


class LLMParser:
    """
    Parse natural language voice commands using pattern matching and optional LLM

    Supports:
    - Pattern-based parsing (fallback)
    - LLM-enhanced parsing (if available)
    - Semantic similarity for command variations
    - Confidence scoring
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize parser

        Args:
            use_llm: Whether to use LLM for enhanced parsing
        """
        self.use_llm = use_llm
        self.llm_available = self._check_llm()

        # Semantic command aliases
        self.intent_aliases = {
            # Duration changes
            "increase": ["increase", "raise", "extend", "lengthen", "make it longer",
                        "bump up", "add more time", "make longer", "go longer"],
            "decrease": ["decrease", "reduce", "shorten", "make shorter", "cut down",
                        "minus", "remove time", "make less"],
            "set": ["set", "change", "update", "make it", "set to"],

            # Temperature changes
            "increase_temp": ["increase temperature", "heat up", "make it hotter",
                             "raise temperature", "crank it up"],
            "decrease_temp": ["decrease temperature", "cool down", "lower temperature",
                             "make it cooler"],

            # Process changes
            "start": ["start", "begin", "commence", "launch", "run"],
            "stop": ["stop", "halt", "finish", "end", "pause"],
            "pause": ["pause", "suspend", "hold"],
            "resume": ["resume", "continue", "restart", "go again"],

            # Query
            "ask": ["what", "how", "why", "tell me about", "explain"],
        }

        # Measurement patterns
        self.unit_patterns = {
            "minutes": [r"(\d+)\s*(?:minutes?|mins?)", r"(\d+)\'"],
            "hours": [r"(\d+)\s*(?:hours?|hrs?)", r"(\d+)h"],
            "degrees": [r"(\d+)\s*(?:degrees?|deg|°)?(?:f|fahrenheit)?"],
            "cups": [r"(\d+(?:\.\d+)?)\s*cups?"],
            "pounds": [r"(\d+(?:\.\d+)?)\s*(?:pounds?|lbs?)"],
            "percent": [r"(\d+)\s*%"],
        }

    def _check_llm(self) -> bool:
        """Check if LLM is available"""
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse voice command text

        Args:
            text: Voice transcription text

        Returns:
            ParsedCommand with structured information
        """
        text = text.lower().strip()

        # Remove "Elle" prefix
        text = re.sub(r"^(?:hey |ok )?elle[,:]?\s*", "", text)

        # Try LLM parsing first
        if self.use_llm and self.llm_available:
            try:
                return self._parse_with_llm(text)
            except Exception as e:
                print(f"⚠ LLM parsing failed: {e}. Falling back to pattern matching.")

        # Fall back to pattern matching
        return self._parse_with_patterns(text)

    def _parse_with_patterns(self, text: str) -> ParsedCommand:
        """Parse using regex patterns"""

        # SOP Update: "update X sop: Y"
        match = re.match(r"(?:update|change|modify)\s+(\w+)\s+sop[:\s]+(.+)", text)
        if match:
            sop_name = match.group(1)
            update_text = match.group(2)
            return self._parse_sop_update(text, sop_name, update_text)

        # Show SOP: "show X sop"
        match = re.match(r"(?:show|display)\s+(?:me\s+)?(?:the\s+)?(\w+)\s+sop", text)
        if match:
            return ParsedCommand(
                command_type="sop_show",
                entity=match.group(1),
                raw_text=text
            )

        # Query: "what/how/why ..."
        match = re.match(r"(?:what|how|why)(?:'s|is)?\s+(.+?)(?:\?|$)", text)
        if match:
            query_text = match.group(1)
            return ParsedCommand(
                command_type="query",
                action="ask",
                parameters={"query_text": query_text},
                raw_text=text
            )

        # Task Start: "start X"
        match = re.match(r"start\s+(.+?)(?:\s+(?:now|timer))?(?:\?|$)", text)
        if match:
            return ParsedCommand(
                command_type="task_start",
                action="start",
                parameters={"task_name": match.group(1)},
                raw_text=text
            )

        # Task End: "finish/end X"
        match = re.match(r"(?:finish|end|complete)\s+(?:task:?)?\s*(.+)?", text)
        if match:
            return ParsedCommand(
                command_type="task_end",
                action="end",
                parameters={"end_info": match.group(1) or ""},
                raw_text=text
            )

        # Task Pause: "pause"
        if "pause" in text:
            return ParsedCommand(
                command_type="task_pause",
                action="pause",
                raw_text=text
            )

        # Task Resume: "resume/continue"
        if any(word in text for word in ["resume", "continue", "go again", "restart"]):
            return ParsedCommand(
                command_type="task_resume",
                action="resume",
                raw_text=text
            )

        # Unknown command
        return ParsedCommand(
            command_type="unknown",
            confidence=0.0,
            raw_text=text
        )

    def _parse_sop_update(
        self,
        raw_text: str,
        sop_name: str,
        update_text: str
    ) -> ParsedCommand:
        """Parse SOP update command with semantic understanding"""

        intent = self._detect_intent(update_text)
        parameters = self._extract_parameters(update_text)

        return ParsedCommand(
            command_type="sop_update",
            entity=sop_name,
            action=intent,
            intent=intent,
            parameters=parameters,
            confidence=self._calculate_confidence(update_text),
            raw_text=raw_text
        )

    def _detect_intent(self, text: str) -> str:
        """Detect intent from text"""

        # Check intent aliases
        for intent, aliases in self.intent_aliases.items():
            for alias in aliases:
                if alias in text:
                    return intent

        # Default to "update"
        return "update"

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from text"""

        parameters = {}

        # Extract what to update
        if "time" in text or "duration" in text:
            parameters["property"] = "duration"
        elif "temperature" in text or "temp" in text:
            parameters["property"] = "temperature"
        elif "step" in text:
            parameters["property"] = "step"
        else:
            parameters["property"] = "unknown"

        # Extract step name
        step_match = re.search(r"(?:step|for)\s+(\w+)", text)
        if step_match:
            parameters["step_name"] = step_match.group(1)

        # Extract numeric values
        for unit_type, patterns in self.unit_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        value = float(match.group(1))
                        parameters[f"value_{unit_type}"] = value
                        parameters["value"] = value
                        parameters["unit"] = unit_type
                        break
                    except ValueError:
                        continue

        # Extract description
        parameters["description"] = text

        return parameters

    def _calculate_confidence(self, text: str) -> float:
        """Calculate parsing confidence"""

        # High confidence if contains specific keywords
        confidence = 0.5

        if any(word in text for word in
               ["increase", "decrease", "minutes", "temperature", "degrees"]):
            confidence += 0.2

        if re.search(r"\d+", text):  # Contains numbers
            confidence += 0.2

        # Lower confidence if too vague
        if len(text.split()) < 2:
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _parse_with_llm(self, text: str) -> ParsedCommand:
        """Parse using LLM (Anthropic Claude)"""

        try:
            import anthropic

            client = anthropic.Anthropic()

            # Create LLM prompt for parsing
            prompt = f"""Parse this voice command for Elle Core (farm/kitchen AI) into a structured format.

Voice command: "{text}"

Return JSON with:
- command_type: "sop_update", "sop_show", "query", "task_start", "task_end", "task_pause", "task_resume", "add_step", "remove_step", or "unknown"
- entity: What thing (e.g., "bread", "biochar", "temperature")
- action: What to do (e.g., "increase", "decrease", "set", "show", "start")
- intent: Semantic intent (e.g., "increase_duration", "change_temperature")
- parameters: Dict with extracted values (step_name, value, unit, etc.)
- confidence: 0.0-1.0

Only return the JSON, no markdown or explanation."""

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = message.content[0].text.strip()

            # Extract JSON
            import json
            data = json.loads(response_text)

            return ParsedCommand(
                command_type=data.get("command_type", "unknown"),
                entity=data.get("entity"),
                action=data.get("action"),
                intent=data.get("intent"),
                parameters=data.get("parameters", {}),
                confidence=float(data.get("confidence", 0.8)),
                raw_text=text
            )

        except Exception as e:
            raise RuntimeError(f"LLM parsing failed: {e}")

    def parse_batch(self, texts: List[str]) -> List[ParsedCommand]:
        """Parse multiple commands"""
        return [self.parse(text) for text in texts]


class CommandVariationGenerator:
    """Generate variations of commands for testing"""

    @staticmethod
    def get_sop_update_variations(sop_name: str, value: int, unit: str) -> List[str]:
        """Generate variations for SOP update commands"""

        variations = [
            f"update {sop_name} sop: increase proofing time to {value} {unit}",
            f"Elle, increase {sop_name} proofing to {value} {unit}",
            f"change {sop_name} duration to {value} {unit}",
            f"make {sop_name} proofing longer, set to {value} {unit}",
            f"bump {sop_name} proofing up to {value} {unit}",
            f"set {sop_name} proofing duration to {value} {unit}",
            f"update {sop_name}: proofing should be {value} {unit}",
            f"Elle, I want to increase the {sop_name} proofing to {value} {unit}",
        ]

        return variations

    @staticmethod
    def get_query_variations(query: str) -> List[str]:
        """Generate variations for query commands"""

        variations = [
            f"what is {query}?",
            f"how {query}?",
            f"tell me about {query}",
            f"explain {query}",
            f"what's the {query}?",
            f"can you tell me {query}?",
        ]

        return variations


async def demo_parsing():
    """Demo: Parse various voice commands"""
    print("\n" + "="*60)
    print("LLM PARSER DEMO")
    print("="*60)

    parser = LLMParser(use_llm=False)  # Start with pattern matching

    # Test SOP updates
    sop_updates = CommandVariationGenerator.get_sop_update_variations("bread", 50, "minutes")
    print("\nSOP Update Commands:")
    for text in sop_updates:
        cmd = parser.parse(text)
        print(f"  Input:  {text}")
        print(f"  Parsed: {cmd.command_type} - {cmd.entity} - {cmd.intent}")
        print(f"  Confidence: {cmd.confidence:.1%}")
        if cmd.parameters.get("value"):
            print(f"  Value: {cmd.parameters['value']} {cmd.parameters.get('unit')}")
        print()

    # Test queries
    queries = CommandVariationGenerator.get_query_variations("inoculation ratio")
    print("Query Commands:")
    for text in queries:
        cmd = parser.parse(text)
        print(f"  Input:  {text}")
        print(f"  Parsed: {cmd.command_type}")
        print(f"  Query: {cmd.parameters.get('query_text')}")
        print()

    # Test task commands
    task_commands = [
        "start baking bread",
        "finish baking, made 24 loaves for $180",
        "pause the timer",
        "resume",
    ]
    print("Task Commands:")
    for text in task_commands:
        cmd = parser.parse(text)
        print(f"  Input:  {text}")
        print(f"  Parsed: {cmd.command_type} - {cmd.action}")
        print()


if __name__ == "__main__":
    asyncio.run(demo_parsing())
