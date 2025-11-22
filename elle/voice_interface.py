"""
Voice-Editable SOP Interface for Elle Core

Hands-free operation using:
- Voice commands for SOP updates ("Elle, increase proofing time to 45 minutes")
- Voice queries for knowledge retrieval ("Elle, what's the biochar ratio?")
- Voice-controlled task tracking ("Elle, start baking bread")
- Integration with HoloLoom AudioSpinner and RAG

Created: 2025-11-15
Author: Blake Chasteen
"""

import asyncio
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

# Will integrate with HoloLoom
try:
    from HoloLoom.spinningWheel.audio import AudioSpinner
    from HoloLoom.rag import SimpleRAG
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("Warning: HoloLoom not available. Voice features will be limited.")

from elle.sop_schema import SOP, StepType, UnitType
from elle.tracker import TaskTracker


class VoiceCommand:
    """Parsed voice command"""

    def __init__(
        self,
        raw_text: str,
        command_type: str,
        entity: Optional[str] = None,
        action: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        self.raw_text = raw_text
        self.command_type = command_type  # "sop_update", "query", "task", "info"
        self.entity = entity  # "bread", "biochar", etc.
        self.action = action  # "update", "create", "start", "end"
        self.params = params or {}


class VoiceSOPEditor:
    """
    Voice-controlled SOP editor with HoloLoom integration

    Supports natural language commands like:
    - "Elle, update bread SOP: increase proofing time to 45 minutes"
    - "Elle, add step to biochar SOP: check temperature after 30 minutes"
    - "Elle, what's the biochar inoculation ratio?"
    - "Elle, start baking bread"
    """

    def __init__(
        self,
        sop_dir: str = "elle/sops",
        use_hololoom: bool = True
    ):
        self.sop_dir = Path(sop_dir)
        self.sop_dir.mkdir(parents=True, exist_ok=True)

        # Loaded SOPs (in-memory cache)
        self.sops: Dict[str, SOP] = {}

        # Task tracker
        self.tracker = TaskTracker()

        # HoloLoom integration
        self.use_hololoom = use_hololoom and HOLOLOOM_AVAILABLE
        self.rag: Optional[Any] = None  # SimpleRAG instance

        # Command patterns (regex)
        self.patterns = {
            "sop_update": r"update (\w+) sop:?\s*(.+)",
            "sop_create": r"create (?:new )?sop for (\w+)",
            "sop_show": r"show(?: me)?(?: the)? (\w+) sop",
            "query": r"what(?:'s| is) (.+)\?",
            "task_start": r"start (.+)",
            "task_end": r"(?:finish|end|complete)(?: task)?(?:,)?\s*(.+)?",
            "task_pause": r"pause(?: timer)?",
            "task_resume": r"resume(?: timer)?",
            "add_step": r"add step to (\w+) sop:?\s*(.+)",
            "remove_step": r"remove step (\d+) from (\w+) sop",
        }

        # Load existing SOPs
        self._load_sops()

    def _load_sops(self):
        """Load all SOPs from disk"""
        for sop_file in self.sop_dir.glob("*.json"):
            try:
                with open(sop_file) as f:
                    data = json.load(f)
                    sop = SOP.from_dict(data)
                    self.sops[sop.sop_id] = sop
                    print(f"✓ Loaded SOP: {sop.name} ({sop.sop_id})")
            except Exception as e:
                print(f"✗ Failed to load {sop_file}: {e}")

    def _save_sop(self, sop: SOP):
        """Save SOP to disk"""
        sop_file = self.sop_dir / f"{sop.sop_id}.json"
        with open(sop_file, "w") as f:
            json.dump(sop.to_dict(), f, indent=2)
        print(f"✓ Saved SOP: {sop.name} ({sop.sop_id})")

    async def initialize_hololoom(self):
        """Initialize HoloLoom RAG for knowledge queries"""
        if not self.use_hololoom:
            return

        try:
            from HoloLoom.rag import SimpleRAG

            self.rag = SimpleRAG()

            # Ingest all SOPs into HoloLoom memory
            for sop in self.sops.values():
                markdown = sop.to_markdown()
                await self.rag.ingest(markdown)
                print(f"✓ Ingested SOP into HoloLoom: {sop.name}")

            print("✓ HoloLoom RAG initialized")
        except Exception as e:
            print(f"✗ Failed to initialize HoloLoom: {e}")
            self.use_hololoom = False

    async def process_voice_command(self, text: str, thread_context: Optional[List[str]] = None) -> str:
        """
        Process a voice command

        Args:
            text: Transcribed voice input
            thread_context: Optional list of recent conversation messages for context

        Returns:
            Response text (for text-to-speech)
        """
        # Normalize text
        text = text.lower().strip()

        # Remove "elle" prefix if present
        text = re.sub(r"^(?:hey |ok )?elle,?\s*", "", text)

        # Parse command
        command = self._parse_command(text)

        if not command:
            return "I didn't understand that command. Try 'show bread SOP' or 'start baking bread'."

        # Store thread context for handlers to use
        command.params['thread_context'] = thread_context

        # Route to appropriate handler
        if command.command_type == "sop_show":
            return await self._handle_show_sop(command)
        elif command.command_type == "sop_update":
            return await self._handle_update_sop(command)
        elif command.command_type == "sop_create":
            return await self._handle_create_sop(command)
        elif command.command_type == "add_step":
            return await self._handle_add_step(command)
        elif command.command_type == "remove_step":
            return await self._handle_remove_step(command)
        elif command.command_type == "query":
            return await self._handle_query(command)
        elif command.command_type == "task_start":
            return await self._handle_task_start(command)
        elif command.command_type == "task_end":
            return await self._handle_task_end(command)
        elif command.command_type == "task_pause":
            return await self._handle_task_pause(command)
        elif command.command_type == "task_resume":
            return await self._handle_task_resume(command)
        else:
            return f"Command type '{command.command_type}' not yet implemented."

    def _parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse voice command text"""
        for cmd_type, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()

                if cmd_type == "sop_update":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        entity=groups[0],
                        params={"update_text": groups[1]}
                    )
                elif cmd_type in ["sop_show", "sop_create"]:
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        entity=groups[0]
                    )
                elif cmd_type == "query":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        params={"query_text": groups[0]}
                    )
                elif cmd_type == "task_start":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        params={"task_name": groups[0]}
                    )
                elif cmd_type == "task_end":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        params={"end_info": groups[0] if groups[0] else ""}
                    )
                elif cmd_type in ["task_pause", "task_resume"]:
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type
                    )
                elif cmd_type == "add_step":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        entity=groups[0],
                        params={"step_description": groups[1]}
                    )
                elif cmd_type == "remove_step":
                    return VoiceCommand(
                        raw_text=text,
                        command_type=cmd_type,
                        entity=groups[1],
                        params={"step_number": int(groups[0])}
                    )

        return None

    async def _handle_show_sop(self, command: VoiceCommand) -> str:
        """Show an SOP"""
        sop_id = self._find_sop_id(command.entity)

        if not sop_id:
            return f"I couldn't find an SOP for {command.entity}."

        sop = self.sops[sop_id]

        # Generate spoken summary (full markdown would be too verbose)
        summary = f"{sop.name}, version {sop.version}. "
        summary += f"This produces {sop.batch_size} {sop.batch_unit} in {sop.total_time_minutes} minutes. "
        summary += f"There are {len(sop.steps)} steps. "
        summary += f"Profit per batch is ${sop.profit_per_batch:.2f} with a {sop.profit_margin:.0f}% margin. "
        summary += f"Hourly ROI is ${sop.hourly_roi:.2f} per hour. "

        # Also save full markdown to file for reference
        md_file = self.sop_dir / f"{sop.sop_id}.md"
        with open(md_file, "w") as f:
            f.write(sop.to_markdown())

        summary += f"Full details saved to {md_file.name}."

        return summary

    async def _handle_update_sop(self, command: VoiceCommand) -> str:
        """Update an SOP based on natural language"""
        sop_id = self._find_sop_id(command.entity)

        if not sop_id:
            return f"I couldn't find an SOP for {command.entity}."

        sop = self.sops[sop_id]
        update_text = command.params["update_text"]

        # Parse update (simple patterns for now)
        # TODO: Use LLM for more sophisticated parsing

        # Pattern: "increase proofing time to 45 minutes"
        time_match = re.search(r"(?:increase|change|set|update) (\w+) (?:time|duration) to (\d+) minutes", update_text)
        if time_match:
            step_keyword = time_match.group(1)
            new_duration = int(time_match.group(2))

            # Find step containing keyword
            for step in sop.steps:
                if step_keyword.lower() in step.title.lower() or step_keyword.lower() in step.description.lower():
                    sop.update_step(step.step_number, duration_minutes=new_duration)
                    self._save_sop(sop)
                    return f"Updated {step.title} duration to {new_duration} minutes in {sop.name}."

            return f"Couldn't find a step matching '{step_keyword}' in {sop.name}."

        # Pattern: "change temperature to 450 degrees"
        temp_match = re.search(r"(?:change|set|update) temperature to (\d+)", update_text)
        if temp_match:
            new_temp = int(temp_match.group(1))

            # Find step with temperature
            for step in sop.steps:
                if step.temperature:
                    sop.update_step(step.step_number, temperature=new_temp)
                    self._save_sop(sop)
                    return f"Updated {step.title} temperature to {new_temp}°F in {sop.name}."

            return f"Couldn't find a step with temperature in {sop.name}."

        return f"I couldn't parse the update '{update_text}'. Try being more specific."

    async def _handle_create_sop(self, command: VoiceCommand) -> str:
        """Create a new SOP"""
        # TODO: Interactive SOP creation with voice guidance
        return f"Interactive SOP creation for {command.entity} not yet implemented. Use the Python API for now."

    async def _handle_add_step(self, command: VoiceCommand) -> str:
        """Add a step to an SOP"""
        sop_id = self._find_sop_id(command.entity)

        if not sop_id:
            return f"I couldn't find an SOP for {command.entity}."

        sop = self.sops[sop_id]
        description = command.params["step_description"]

        # Simple step creation (defaults to PROCESS type)
        sop.add_step(
            StepType.PROCESS,
            title="Voice-added step",
            description=description
        )

        self._save_sop(sop)

        return f"Added new step to {sop.name}: {description}"

    async def _handle_remove_step(self, command: VoiceCommand) -> str:
        """Remove a step from an SOP"""
        sop_id = self._find_sop_id(command.entity)

        if not sop_id:
            return f"I couldn't find an SOP for {command.entity}."

        sop = self.sops[sop_id]
        step_num = command.params["step_number"]

        try:
            sop.remove_step(step_num)
            self._save_sop(sop)
            return f"Removed step {step_num} from {sop.name}."
        except ValueError as e:
            return str(e)

    async def _handle_query(self, command: VoiceCommand) -> str:
        """Answer a knowledge query using HoloLoom RAG with thread context"""
        query_text = command.params["query_text"]
        thread_context = command.params.get("thread_context", [])

        if self.use_hololoom and self.rag:
            try:
                # Build enriched query with thread context
                if thread_context and len(thread_context) > 0:
                    # Add recent conversation context to help RAG understand query better
                    context_str = "\n".join(thread_context[-5:])  # Last 5 messages
                    enriched_query = f"Conversation context:\n{context_str}\n\nCurrent question: {query_text}"
                else:
                    enriched_query = query_text

                # Use HoloLoom RAG for sophisticated retrieval
                result = await self.rag.query(enriched_query, mode="direct")
                return result.response
            except Exception as e:
                print(f"✗ HoloLoom query failed: {e}")

        # Fallback: Simple keyword search in SOPs
        query_lower = query_text.lower()

        for sop in self.sops.values():
            # Check ingredients
            for ing in sop.ingredients:
                if ing.name.lower() in query_lower:
                    return f"For {ing.name}, we use {ing.amount} {ing.unit} in {sop.name}."

            # Check steps
            for step in sop.steps:
                if query_lower in step.description.lower():
                    return f"In {sop.name}, {step.title}: {step.description}"

        return f"I couldn't find information about '{query_text}' in the SOPs."

    async def _handle_task_start(self, command: VoiceCommand) -> str:
        """Start a task"""
        task_name = command.params["task_name"]

        # Try to infer SOP and category from task name
        sop_id = None
        category = "general"

        for sop in self.sops.values():
            if sop.sop_id.lower() in task_name.lower() or sop.name.lower() in task_name.lower():
                sop_id = sop.sop_id
                category = sop.category
                break

        task_id = await self.tracker.start(
            task_name=task_name,
            sop_id=sop_id,
            category=category
        )

        # Store active task ID for later reference
        self.active_task_id = task_id

        return f"Started task: {task_name}. Timer is running."

    async def _handle_task_end(self, command: VoiceCommand) -> str:
        """End a task"""
        if not hasattr(self, 'active_task_id'):
            return "No active task to end."

        # Parse end info (e.g., "made 24 loaves, sold for $144")
        end_info = command.params.get("end_info", "")

        units = 1
        revenue = 0.0

        # Extract units
        units_match = re.search(r"(?:made|produced) (\d+)", end_info)
        if units_match:
            units = int(units_match.group(1))

        # Extract revenue
        revenue_match = re.search(r"(?:sold for|revenue) \$?(\d+(?:\.\d{2})?)", end_info)
        if revenue_match:
            revenue = float(revenue_match.group(1))

        result = await self.tracker.end(
            task_id=self.active_task_id,
            units=units,
            revenue=revenue
        )

        delattr(self, 'active_task_id')

        return (
            f"Task complete. "
            f"Duration: {result.active_time_hours:.1f} hours. "
            f"Profit: ${result.profit:.2f}. "
            f"Hourly ROI: ${result.hourly_roi:.2f} per hour."
        )

    async def _handle_task_pause(self, command: VoiceCommand) -> str:
        """Pause active task"""
        if not hasattr(self, 'active_task_id'):
            return "No active task to pause."

        await self.tracker.pause(self.active_task_id)
        return "Timer paused. Take your break."

    async def _handle_task_resume(self, command: VoiceCommand) -> str:
        """Resume paused task"""
        if not hasattr(self, 'active_task_id'):
            return "No active task to resume."

        await self.tracker.resume(self.active_task_id)
        return "Timer resumed. Back to work."

    def _find_sop_id(self, entity_name: str) -> Optional[str]:
        """Find SOP ID from entity name (fuzzy match)"""
        entity_lower = entity_name.lower()

        # Direct ID match
        for sop_id in self.sops:
            if entity_lower in sop_id.lower():
                return sop_id

        # Name match
        for sop_id, sop in self.sops.items():
            if entity_lower in sop.name.lower():
                return sop_id

        # Tag match
        for sop_id, sop in self.sops.items():
            if any(entity_lower in tag.lower() for tag in sop.tags):
                return sop_id

        return None


# Text-to-speech helper (optional)
def speak(text: str):
    """Convert text to speech (requires pyttsx3)"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        print(f"[SPEAK]: {text}")


# Example interactive session
async def interactive_session():
    """Run an interactive voice command session"""
    print("\n" + "="*60)
    print("ELLE CORE - Voice Interface")
    print("="*60)
    print("\nInitializing...")

    editor = VoiceSOPEditor()
    await editor.initialize_hololoom()

    print("\n✓ Ready! Try commands like:")
    print("  - 'show bread SOP'")
    print("  - 'what's the biochar inoculation ratio?'")
    print("  - 'start baking bread'")
    print("  - 'update bread SOP: increase proofing time to 50 minutes'")
    print("\nType 'quit' to exit.\n")

    while True:
        try:
            # Get command (in real use, this would be from speech-to-text)
            user_input = input("Elle> ").strip()

            if user_input.lower() in ['quit', 'exit', 'stop']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Process command
            response = await editor.process_voice_command(user_input)

            # Output response (in real use, this would be text-to-speech)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(interactive_session())
