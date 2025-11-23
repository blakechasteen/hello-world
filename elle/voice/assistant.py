"""
Elle Voice Assistant

Integrates STT, NLP, TTS, and wake word detection into a complete voice interface.

Created: 2025-11-15
"""

import asyncio
from typing import Optional, Callable, List
from pathlib import Path

from elle.voice.whisper_stt import WhisperSTT
from elle.voice.llm_parser import LLMParser, ParsedCommand
from elle.voice.command_parser import CommandGrammarParser, StructuredCommand, CommandType
from elle.voice.tts import TextToSpeech, VoiceGender
from elle.voice.wake_word import WakeWordDetector
from elle.voice.threads import ThreadManager
from elle.voice_interface import VoiceSOPEditor


class VoiceAssistant:
    """
    Complete voice assistant for Elle Core

    Combines:
    - Wake word detection ("Hey Elle")
    - Speech-to-text (Whisper)
    - Natural language parsing (LLM + patterns)
    - Command execution (VoiceSOPEditor)
    - Text-to-speech responses
    """

    def __init__(
        self,
        sop_dir: str = "elle/sops",
        whisper_model: str = "tiny",
        tts_rate: int = 150,
        use_llm_parser: bool = True,
        verbose: bool = True
    ):
        """
        Initialize voice assistant

        Args:
            sop_dir: Directory for SOP files
            whisper_model: Whisper model size (tiny, base, small)
            tts_rate: TTS speech rate (words per minute)
            use_llm_parser: Use LLM for enhanced parsing
            verbose: Print debug information
        """
        self.sop_dir = sop_dir
        self.verbose = verbose

        # Initialize components
        self.stt = WhisperSTT(model=whisper_model)
        self.parser = LLMParser(use_llm=use_llm_parser)  # Milestone 1: Conversational
        self.command_parser = CommandGrammarParser()  # Milestone 2: Command Mode
        self.tts = TextToSpeech(rate=tts_rate, voice_gender=VoiceGender.FEMALE)
        self.wake_detector = WakeWordDetector()
        self.editor = VoiceSOPEditor(sop_dir=sop_dir)
        self.thread_manager = ThreadManager()  # Thread management for conversations

        # Session state
        self.is_listening = False
        self.context = {}
        self.command_mode = True  # Milestone 2: Default to Command Mode (concise)

        # Navigation history (Milestone 2)
        self.navigation_history = []  # List of thread IDs
        self.navigation_position = -1  # Current position in history

    async def initialize(self):
        """Initialize assistant (loads SOPs, initializes RAG, etc.)"""
        if self.verbose:
            print("Initializing Elle Voice Assistant...")

        await self.editor.initialize_hololoom()

    async def process_voice_input(self, text: str) -> str:
        """
        Process voice input: parse and execute command

        Milestone 1: Conversational Mode (natural language)
        Milestone 2: Command Mode (structured shortcuts)

        Args:
            text: Transcribed voice text

        Returns:
            Response text to speak
        """
        if not text:
            return "I didn't hear that. Please try again."

        # Store user message in active thread
        self.thread_manager.add_message_to_active("user", text)

        # Try Command Mode first (Milestone 2)
        if self.command_mode:
            commands = self.command_parser.parse(text)

            # Handle command chaining (t3; run analyze)
            if len(commands) > 1:
                responses = []
                for cmd in commands:
                    resp = await self._handle_structured_command(cmd)
                    if resp:
                        responses.append(resp)
                response = ". ".join(responses) if responses else "Done"
            else:
                # Single command
                cmd = commands[0]

                # Check if fallback to conversational mode
                if cmd.command_type == CommandType.CONVERSATIONAL:
                    # Switch to conversational parser
                    response = await self._handle_conversational(text)
                else:
                    # Handle structured command
                    response = await self._handle_structured_command(cmd)
        else:
            # Conversational Mode only (Milestone 1)
            response = await self._handle_conversational(text)

        # Store assistant response in active thread
        self.thread_manager.add_message_to_active("assistant", response)

        return response

    async def _handle_conversational(self, text: str) -> str:
        """Handle conversational mode (Milestone 1)"""
        command = self.parser.parse(text)

        if self.verbose:
            print(f"[Conversational] Parsed: {command.command_type}")
            print(f"  Confidence: {command.confidence:.1%}")

        # Handle thread commands (Milestone 1 syntax)
        if command.command_type == "thread_create":
            return self._handle_thread_create(command)
        elif command.command_type == "thread_switch":
            return self._handle_thread_switch(command)
        elif command.command_type == "thread_list":
            return self._handle_thread_list(command)
        elif command.command_type == "thread_summarize":
            return self._handle_thread_summarize(command)
        # Handle unknown commands
        elif command.command_type == "unknown":
            return "I didn't understand that command. Try 't3' or 'threads'."
        # Process other commands through voice SOP editor
        else:
            try:
                # Get thread context (recent messages) for better query understanding
                thread_context = self._get_thread_context()
                response = await self.editor.process_voice_command(text, thread_context=thread_context)
                return response
            except Exception as e:
                return f"Error processing command: {str(e)}"

    async def _handle_structured_command(self, cmd: StructuredCommand) -> str:
        """
        Handle structured command (Milestone 2)

        Args:
            cmd: StructuredCommand from CommandGrammarParser

        Returns:
            Response text (brief, <500ms)
        """
        if self.verbose:
            print(f"[Command Mode] {cmd.command_type.value} {cmd.parameters}")

        # Navigation commands
        if cmd.command_type == CommandType.BACK:
            return self._handle_nav_back()
        elif cmd.command_type == CommandType.NEXT:
            return self._handle_nav_next()
        elif cmd.command_type == CommandType.HOME:
            return self._handle_nav_home()

        # Thread operations
        elif cmd.command_type == CommandType.THREAD_SWITCH:
            return self._handle_thread_switch_structured(cmd)
        elif cmd.command_type == CommandType.THREAD_LIST:
            return self._handle_thread_list_structured(cmd)
        elif cmd.command_type == CommandType.THREAD_CREATE:
            return self._handle_thread_create_structured(cmd)
        elif cmd.command_type == CommandType.THREAD_DELETE:
            return self._handle_thread_delete(cmd)

        # Task operations
        elif cmd.command_type == CommandType.TASK_RUN:
            return await self._handle_task_run(cmd)
        elif cmd.command_type == CommandType.TASK_STOP:
            return self._handle_task_stop()
        elif cmd.command_type == CommandType.TASK_PAUSE:
            return self._handle_task_pause()
        elif cmd.command_type == CommandType.TASK_RESUME:
            return self._handle_task_resume()
        elif cmd.command_type == CommandType.TASK_STATUS:
            return self._handle_task_status()

        # Query operations
        elif cmd.command_type == CommandType.ENTITY_LOOKUP:
            return await self._handle_entity_lookup(cmd)
        elif cmd.command_type == CommandType.SEARCH:
            return await self._handle_search(cmd)

        else:
            return "Unknown command"

    def _handle_thread_create(self, command: ParsedCommand) -> str:
        """Handle thread creation command"""
        topic = command.parameters.get("thread_topic", "").strip()

        if not topic:
            return "What topic should this thread be about?"

        # Create thread with topic as both name and topic
        thread = self.thread_manager.create_thread(name=topic, topic=topic)

        return f"Created new thread '{topic}'. What would you like to discuss?"

    def _handle_thread_switch(self, command: ParsedCommand) -> str:
        """Handle thread switch command"""
        thread_name = command.parameters.get("thread_name", "").strip()

        if not thread_name:
            return "Which thread do you want to switch to?"

        # Switch to thread
        thread = self.thread_manager.switch_thread(thread_name)

        if thread:
            msg_count = len(thread.messages)
            return f"Switched to thread '{thread.name}'. {msg_count} messages in this thread."
        else:
            return f"Thread '{thread_name}' not found. Say 'list my threads' to see available threads."

    def _handle_thread_list(self, command: ParsedCommand) -> str:
        """Handle thread list command"""
        threads = self.thread_manager.list_threads()

        if not threads:
            return "You have no threads yet. Say 'start a new thread for [topic]' to create one."

        # Build response
        active_id = self.thread_manager.active_thread_id
        thread_list = []

        for i, thread in enumerate(threads, 1):
            active_marker = "★ " if thread.id == active_id else "  "
            msg_count = len(thread.messages)
            thread_list.append(f"{active_marker}{i}. {thread.name} ({msg_count} messages)")

        response = "Your threads:\n" + "\n".join(thread_list)
        return response

    def _handle_thread_summarize(self, command: ParsedCommand) -> str:
        """Handle thread summarize command"""
        thread_name = command.parameters.get("thread_name")

        # Get thread (active if no name specified)
        if thread_name:
            thread = self.thread_manager.get_thread_by_name(thread_name)
            if not thread:
                return f"Thread '{thread_name}' not found."
        else:
            thread = self.thread_manager.get_active_thread()
            if not thread:
                return "No active thread to summarize."

        # Generate summary
        summary = thread.get_summary()
        return summary

    def _get_thread_context(self, n: int = 10) -> List[str]:
        """
        Get recent conversation context from active thread

        Args:
            n: Number of recent messages to include

        Returns:
            List of message strings in format "role: content"
        """
        thread = self.thread_manager.get_active_thread()
        if not thread:
            return []

        # Get last N messages
        recent_messages = thread.get_last_n_messages(n)

        # Format as "role: content" strings
        context = []
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Elle"
            context.append(f"{role_label}: {msg.content}")

        return context

    # ========================================================================
    # Milestone 2: Structured Command Handlers
    # ========================================================================

    # Navigation Handlers
    def _handle_nav_back(self) -> str:
        """Navigate to previous thread in history"""
        if len(self.navigation_history) < 2:
            return "No previous thread"

        # Current thread is at end of history, go to previous
        self.navigation_history.pop()  # Remove current
        prev_thread_id = self.navigation_history[-1]

        # Switch to previous thread
        thread = self.thread_manager.get_thread(prev_thread_id)
        if thread:
            self.thread_manager.active_thread_id = prev_thread_id
            return f"Back to {thread.name}"
        else:
            return "Thread not found"

    def _handle_nav_next(self) -> str:
        """Navigate to next thread in history (forward)"""
        if self.navigation_position >= len(self.navigation_history) - 1:
            return "No next thread"

        self.navigation_position += 1
        next_thread_id = self.navigation_history[self.navigation_position]

        thread = self.thread_manager.get_thread(next_thread_id)
        if thread:
            self.thread_manager.active_thread_id = next_thread_id
            return f"Next: {thread.name}"
        else:
            return "Thread not found"

    def _handle_nav_home(self) -> str:
        """Navigate to default/first thread"""
        threads = self.thread_manager.list_threads()
        if not threads:
            return "No threads available"

        # Switch to first thread
        first_thread = threads[0]
        self.thread_manager.active_thread_id = first_thread.id
        self._add_to_navigation_history(first_thread.id)
        return f"Home: {first_thread.name}"

    # Structured Thread Handlers (Brief Responses)
    def _handle_thread_switch_structured(self, cmd: StructuredCommand) -> str:
        """Switch thread using ID or name - brief response"""
        if "thread_id" in cmd.parameters:
            thread_id = cmd.parameters["thread_id"]
            threads = self.thread_manager.list_threads()

            # Find thread by index (1-based)
            if 1 <= thread_id <= len(threads):
                thread = threads[thread_id - 1]
                self.thread_manager.active_thread_id = thread.id
                self._add_to_navigation_history(thread.id)
                return f"Thread {thread_id}"
            else:
                return f"No thread {thread_id}"

        elif "thread_name" in cmd.parameters:
            thread_name = cmd.parameters["thread_name"]
            thread = self.thread_manager.switch_thread(thread_name)

            if thread:
                self._add_to_navigation_history(thread.id)
                return f"{thread.name}"
            else:
                return f"No thread {thread_name}"

        return "Invalid thread"

    def _handle_thread_list_structured(self, cmd: StructuredCommand) -> str:
        """List threads - brief response"""
        threads = self.thread_manager.list_threads()

        if not threads:
            return "No threads"

        # Brief list: just count and names
        names = [t.name for t in threads[:3]]  # First 3 only
        response = f"{len(threads)} threads"

        if len(threads) <= 3:
            response += f": {', '.join(names)}"

        return response

    def _handle_thread_create_structured(self, cmd: StructuredCommand) -> str:
        """Create thread - brief response"""
        topic = cmd.parameters.get("topic", "").strip()

        if not topic:
            return "Need topic"

        thread = self.thread_manager.create_thread(name=topic, topic=topic)
        self._add_to_navigation_history(thread.id)
        return f"Created {topic}"

    def _handle_thread_delete(self, cmd: StructuredCommand) -> str:
        """Delete thread by ID"""
        thread_id_index = cmd.parameters.get("thread_id")
        threads = self.thread_manager.list_threads()

        if 1 <= thread_id_index <= len(threads):
            thread = threads[thread_id_index - 1]
            thread_name = thread.name

            # Remove from navigation history
            if thread.id in self.navigation_history:
                self.navigation_history.remove(thread.id)

            # Delete thread
            self.thread_manager.threads = [t for t in self.thread_manager.threads if t.id != thread.id]

            # Switch to first thread if deleted active
            if self.thread_manager.active_thread_id == thread.id and threads:
                self.thread_manager.active_thread_id = threads[0].id

            return f"Deleted {thread_name}"
        else:
            return f"No thread {thread_id_index}"

    # Task Handlers
    async def _handle_task_run(self, cmd: StructuredCommand) -> str:
        """Execute task"""
        task_name = cmd.parameters.get("task_name", "")

        if not task_name:
            return "Need task name"

        # Placeholder: Task execution system to be implemented in Phase 3
        # For now, acknowledge the command
        return f"Running {task_name}"

    def _handle_task_stop(self) -> str:
        """Stop current task"""
        # Placeholder: Task control to be implemented in Phase 3
        return "Stopped"

    def _handle_task_pause(self) -> str:
        """Pause current task"""
        # Placeholder: Task control to be implemented in Phase 3
        return "Paused"

    def _handle_task_resume(self) -> str:
        """Resume paused task"""
        # Placeholder: Task control to be implemented in Phase 3
        return "Resumed"

    def _handle_task_status(self) -> str:
        """Show task status"""
        # Placeholder: Task status to be implemented in Phase 3
        return "No tasks running"

    # Query Handlers
    async def _handle_entity_lookup(self, cmd: StructuredCommand) -> str:
        """Quick entity reference lookup"""
        entity = cmd.parameters.get("entity", "")

        if not entity:
            return "Need entity name"

        # Use VoiceSOPEditor for entity lookup
        try:
            query = f"What is {entity}?"
            thread_context = self._get_thread_context()
            response = await self.editor.process_voice_command(query, thread_context=thread_context)
            return response
        except Exception as e:
            return f"Lookup failed: {str(e)}"

    async def _handle_search(self, cmd: StructuredCommand) -> str:
        """Knowledge base search"""
        query = cmd.parameters.get("query", "")

        if not query:
            return "Need search query"

        # Use VoiceSOPEditor for search
        try:
            search_query = f"Find {query}"
            thread_context = self._get_thread_context()
            response = await self.editor.process_voice_command(search_query, thread_context=thread_context)
            return response
        except Exception as e:
            return f"Search failed: {str(e)}"

    # Helper Methods
    def _add_to_navigation_history(self, thread_id: str):
        """Add thread to navigation history"""
        # Avoid duplicates - only add if different from current
        if not self.navigation_history or self.navigation_history[-1] != thread_id:
            self.navigation_history.append(thread_id)

            # Limit history to last 20 threads
            if len(self.navigation_history) > 20:
                self.navigation_history.pop(0)

            # Reset forward navigation position
            self.navigation_position = len(self.navigation_history) - 1

    # ========================================================================
    # End of Milestone 2 Handlers
    # ========================================================================

    async def listen_and_respond(self) -> bool:
        """
        Listen for voice input, process it, and respond

        Returns:
            True if command was processed, False if error
        """
        try:
            # Record audio
            audio_path = await self.stt.record_audio(duration=5.0)
            if not audio_path:
                await self.tts.speak("I couldn't hear you. Please try again.")
                return False

            # Transcribe
            text, metadata = await self.stt.transcribe_with_fallback(audio_path)
            if not text:
                await self.tts.speak("Transcription failed. Please try again.")
                return False

            if self.verbose:
                print(f"Transcribed: {text}")

            # Process command
            response = await self.process_voice_input(text)

            # Speak response
            await self.tts.speak(response)

            return True

        except Exception as e:
            print(f"✗ Error: {e}")
            await self.tts.speak("An error occurred. Please try again.")
            return False

    async def wake_word_loop(self):
        """
        Continuous listening for wake word

        Listens for "Hey Elle", then listens for command
        """
        print("\n" + "="*60)
        print("ELLE VOICE ASSISTANT")
        print("="*60)
        print("\n👂 Listening for wake word...")
        await self.tts.speak("Elle is ready. Say hey Elle to begin.")

        while True:
            try:
                # Listen for wake word
                async def record_and_transcribe():
                    audio_path = await self.stt.record_audio(duration=5.0)
                    if audio_path:
                        text, _ = await self.stt.transcribe_with_fallback(audio_path)
                        return text
                    return None

                # Wait for wake word
                detected = await self.wake_detector.listen_for_wake_word(
                    record_and_transcribe,
                    on_detected=self._on_wake_word
                )

                if detected:
                    # Listen for command
                    await self.tts.speak("I'm listening.")
                    success = await self.listen_and_respond()

                    if success:
                        await asyncio.sleep(1)  # Pause before next wake word
                        print("\n👂 Listening for wake word...")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                await self.tts.speak("Goodbye.")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                await asyncio.sleep(1)

    async def _on_wake_word(self, text: str):
        """Callback when wake word is detected"""
        # Extract command from wake word phrase
        command = self.wake_detector.extract_command(text)
        if command and len(command) > 2:
            # Full command was spoken with wake word
            if self.verbose:
                print(f"Full command heard: {command}")
            response = await self.process_voice_input(command)
            await self.tts.speak(response)

    async def interactive_mode(self):
        """
        Interactive mode: text input instead of voice

        Useful for testing without audio hardware
        """
        print("\n" + "="*60)
        print("ELLE VOICE ASSISTANT - INTERACTIVE MODE")
        print("="*60)
        print("\nType commands like:")
        print("  - 'show bread SOP'")
        print("  - 'update bread SOP: increase proofing to 50 minutes'")
        print("  - 'what's the biochar inoculation ratio?'")
        print("  - 'start baking bread'")
        print("  - 'Type 'quit' to exit'\n")

        await self.tts.speak("Elle is ready in interactive mode.")

        while True:
            try:
                text = input("You> ").strip()

                if not text:
                    continue

                if text.lower() in ["quit", "exit", "bye"]:
                    print("Goodbye!")
                    await self.tts.speak("Goodbye.")
                    break

                response = await self.process_voice_input(text)
                print(f"Elle> {response}")
                await self.tts.speak(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                await self.tts.speak("Goodbye.")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

    async def run(self, mode: str = "voice"):
        """
        Run the assistant

        Args:
            mode: "voice" for wake word listening, "interactive" for text input
        """
        await self.initialize()

        if mode == "voice":
            await self.wake_word_loop()
        elif mode == "interactive":
            await self.interactive_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}")


async def main():
    """Main entry point"""
    import sys

    # Determine mode from command line
    mode = "interactive"  # Default to interactive for easier testing

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode not in ["voice", "interactive"]:
        print(f"Usage: python assistant.py [voice|interactive]")
        print(f"  voice: Listen for 'Hey Elle' wake word (requires microphone)")
        print(f"  interactive: Type commands (for testing)")
        sys.exit(1)

    # Create and run assistant
    assistant = VoiceAssistant(
        whisper_model="tiny",  # Use tiny for speed
        tts_rate=150,
        use_llm_parser=False,  # Pattern matching only for now
        verbose=True
    )

    await assistant.run(mode=mode)


if __name__ == "__main__":
    asyncio.run(main())
