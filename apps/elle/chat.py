"""Interactive chat with Elle.

Usage:
    python -m elle.chat
    python -m elle.chat --provider ollama --model llama3.2:3b  # Local via Ollama
    python -m elle.chat --provider anthropic
    python -m elle.chat --provider openai --model gpt-4
    python -m elle.chat --user "Blake"
    python -m elle.chat --enable-ar  # Enable AR mode transitions
    python -m elle.chat --enable-rag  # Enable RAG knowledge retrieval

Elle is a calm AR companion. In chat mode, she:
- Listens to your thoughts
- Offers minimal, grounded guidance
- Speaks plainly, one thing at a time

With AR enabled, Elle can transition to visual mode:
- Say "show me where to start" to activate AR
- Say "go back to chat" to return to text mode

With RAG enabled, Elle can remember and retrieve knowledge:
- Use /learn <text> to teach Elle something
- Knowledge persists during the session
"""

import os
import sys
import argparse
import logging
import asyncio
import webbrowser
from pathlib import Path
from typing import Optional

# Elle imports
from elle.adapters.chat_adapter import ChatAdapter, create_chat_adapter
from elle.adapters.chat_ar_bridge import ElleMode as BridgeMode
from elle.core.llm_client import (
    LLMProvider,
    AnthropicClient,
    OpenAIClient,
    OllamaClient,
    LLMClient,
)
from elle.core.policy import EllePolicy
from elle.core.prompt import PromptBuilder
from elle.engine.elle_engine import ElleEngine
from elle.memory.in_memory import InMemoryMemoryStore
from elle.tools import create_default_registry

# RAG bridge (optional - graceful degradation)
try:
    from elle.adapters.rag_bridge import ElleRAGBridge, create_rag_bridge
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    ElleRAGBridge = None
    create_rag_bridge = None


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ChatSession:
    """Interactive chat session with Elle, with AR and RAG support."""

    def __init__(
        self,
        engine: ElleEngine,
        chat_adapter: ChatAdapter,
        user_name: str = "User",
        enable_ar: bool = False,
        ar_client_url: Optional[str] = None,
        enable_rag: bool = False,
        rag_bridge: Optional["ElleRAGBridge"] = None,
    ):
        self.engine = engine
        self.chat_adapter = chat_adapter
        self.user_name = user_name
        self.running = False
        self.enable_ar = enable_ar
        self.ar_client_url = ar_client_url or "http://localhost:8000/ar"
        self.ar_mode_active = False
        self.enable_rag = enable_rag
        self.rag_bridge = rag_bridge

    def start(self):
        """Start the chat session (sync wrapper)."""
        asyncio.run(self._async_start())

    async def _async_start(self):
        """Start the chat session (async)."""
        self.running = True
        self.chat_adapter.start_session()

        # Initialize RAG bridge if enabled
        if self.enable_rag and self.rag_bridge:
            try:
                await self.rag_bridge.__aenter__()
                if self.rag_bridge.available:
                    print("  RAG knowledge system ready.")
                else:
                    print("  RAG: HoloLoom not available, operating without knowledge retrieval.")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")
                print(f"  RAG initialization failed: {e}")
                self.enable_rag = False

        self._print_welcome()

        while self.running:
            try:
                # Get user input
                user_input = self._get_input()

                if user_input is None:
                    # EOF (Ctrl+D)
                    self.running = False
                    break

                # Handle special commands
                if self._handle_command(user_input):
                    continue

                # Skip empty input
                if not user_input.strip():
                    continue

                # Check for AR mode trigger (if AR enabled)
                response_prefix = ""
                if self.enable_ar:
                    ar_result = await self.chat_adapter.check_ar_trigger(user_input)
                    if ar_result.get("should_transition"):
                        await self._handle_ar_transition(ar_result)
                    if ar_result.get("response_prefix"):
                        response_prefix = ar_result["response_prefix"]

                # Process through Elle (with RAG context if enabled)
                response = await self._process_message(user_input)

                # Display response with optional prefix
                self._print_response(response_prefix + response)

            except KeyboardInterrupt:
                # Ctrl+C
                print("\n")
                self.running = False
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\n(Something went wrong. Let's continue.)\n")

        self._print_goodbye()
        self.chat_adapter.end_session()

        # Cleanup RAG bridge
        if self.rag_bridge:
            try:
                await self.rag_bridge.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"RAG cleanup error: {e}")

    async def _handle_ar_transition(self, ar_result: dict):
        """Handle transition to/from AR mode."""
        transition = ar_result.get("transition")
        if not transition:
            return

        if transition.to_mode == BridgeMode.AR:
            # Transitioning TO AR mode
            self.ar_mode_active = True
            print("\n" + "-" * 50)
            print("  [AR Mode Activated]")
            print("  Opening visual view...")
            print("-" * 50)

            # Open AR client in browser
            try:
                ar_url = self._get_ar_client_url()
                webbrowser.open(ar_url)
                print(f"  AR client: {ar_url}")
            except Exception as e:
                logger.warning(f"Could not open AR client: {e}")
                print("  (AR client not available - continuing in chat)")
                self.ar_mode_active = False

            print("-" * 50 + "\n")

        elif transition.to_mode == BridgeMode.CHAT:
            # Transitioning back TO chat mode
            self.ar_mode_active = False
            print("\n" + "-" * 50)
            print("  [Chat Mode]")
            print("  Back to text conversation.")
            print("-" * 50 + "\n")

    def _get_ar_client_url(self) -> str:
        """Get URL for AR web client."""
        # Check if demo.html exists locally
        demo_path = Path(__file__).parent / "ar_web_client" / "demo.html"
        if demo_path.exists():
            return f"file://{demo_path.absolute()}"
        # Fall back to server URL
        return self.ar_client_url

    def _get_input(self) -> Optional[str]:
        """Get input from user."""
        try:
            return input(f"\n{self.user_name}: ")
        except EOFError:
            return None

    def _handle_command(self, text: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        text_lower = text.strip().lower()

        # Exit commands
        if text_lower in ("exit", "quit", "bye", "/quit", "/exit"):
            self.running = False
            return True

        # Help command
        if text_lower in ("help", "/help", "?"):
            self._print_help()
            return True

        # Context/history command
        if text_lower in ("/context", "/history"):
            self._print_context()
            return True

        # Clear command
        if text_lower in ("/clear", "/reset"):
            self.chat_adapter.end_session()
            self.chat_adapter.start_session()
            print("\n(Starting fresh.)\n")
            return True

        # AR commands (only if AR enabled)
        if self.enable_ar:
            if text_lower == "/ar":
                self._toggle_ar_mode()
                return True

            if text_lower == "/mode":
                mode = self.chat_adapter.get_ar_mode()
                ar_status = " (AR active)" if self.ar_mode_active else ""
                print(f"\n  Current mode: {mode}{ar_status}\n")
                return True

        # RAG commands (only if RAG enabled)
        if self.enable_rag and self.rag_bridge:
            if text_lower.startswith("/learn "):
                knowledge = text[7:].strip()
                if knowledge:
                    asyncio.create_task(self._learn_knowledge(knowledge))
                else:
                    print("\n  Usage: /learn <something to remember>\n")
                return True

            if text_lower == "/knowledge":
                self._show_rag_status()
                return True

        return False

    async def _learn_knowledge(self, knowledge: str):
        """Teach Elle something via RAG."""
        if not self.rag_bridge:
            print("\n  RAG not available.\n")
            return

        try:
            success = await self.rag_bridge.ingest(knowledge)
            if success:
                print(f"\n  (Elle remembers: {knowledge[:50]}...)\n")
            else:
                print("\n  (Failed to store knowledge)\n")
        except Exception as e:
            logger.warning(f"RAG ingest failed: {e}")
            print("\n  (Something went wrong storing that)\n")

    def _show_rag_status(self):
        """Show RAG system status."""
        if not self.rag_bridge:
            print("\n  RAG: Not initialized\n")
            return

        try:
            metrics = self.rag_bridge.get_metrics()
            if metrics.get("available"):
                print("\n  RAG Status:")
                print(f"    Available: Yes")
                print(f"    Memories: {metrics.get('n_memories', 0)}")
                print(f"    Cache size: {metrics.get('cache_size', 0)}")
                print()
            else:
                print("\n  RAG: Not available\n")
        except Exception:
            print("\n  RAG: Status unknown\n")

    def _toggle_ar_mode(self):
        """Manually toggle AR mode."""
        if self.ar_mode_active:
            # Deactivate AR
            self.ar_mode_active = False
            print("\n  [Chat Mode] AR deactivated.\n")
        else:
            # Activate AR
            self.ar_mode_active = True
            print("\n  [AR Mode] Opening visual view...")
            try:
                ar_url = self._get_ar_client_url()
                webbrowser.open(ar_url)
                print(f"  AR client: {ar_url}\n")
            except Exception as e:
                logger.warning(f"Could not open AR client: {e}")
                print("  (AR client not available)\n")
                self.ar_mode_active = False

    async def _process_message(self, text: str) -> str:
        """Process user message through Elle, with optional RAG context."""
        # Get RAG context if enabled
        rag_context = None
        if self.enable_rag and self.rag_bridge and self.rag_bridge.available:
            try:
                rag_context = await self.rag_bridge.get_context(text, max_sources=3)
                if rag_context.sources:
                    logger.debug(f"RAG found {len(rag_context.sources)} sources (confidence: {rag_context.confidence:.2f})")
            except Exception as e:
                logger.warning(f"RAG context retrieval failed: {e}")

        # Convert text to ElleRequest (with RAG context in metadata)
        request = self.chat_adapter.text_to_request(
            text,
            rag_context=rag_context.to_prompt_section() if rag_context and rag_context.sources else None
        )

        # Get Elle's decision
        action = self.engine.handle(request)

        # Convert action to text
        response = self.chat_adapter.action_to_text(action)

        return response

    def _print_welcome(self):
        """Print welcome message."""
        print()
        print("=" * 50)
        print("         Elle - Chat Mode")
        print("=" * 50)
        print()
        print("  She's here. Listening.")
        print()
        print("  Type your thoughts. She'll respond when it helps.")
        print("  Type 'exit' or Ctrl+C to leave.")
        print("  Type '/help' for commands.")
        if self.enable_ar:
            print()
            print("  AR mode enabled.")
            print("  Say 'show me...' to switch to visual mode.")
        if self.enable_rag:
            print()
            print("  RAG mode enabled.")
            print("  Use '/learn <text>' to teach Elle.")
        print()
        print("-" * 50)

    def _print_response(self, response: str):
        """Print Elle's response."""
        print(f"\nElle: {response}")

    def _print_help(self):
        """Print help message."""
        print()
        print("Commands:")
        print("  exit, quit, bye   - End the session")
        print("  /help, ?          - Show this help")
        print("  /context          - Show conversation context")
        print("  /clear            - Start fresh conversation")
        if self.enable_ar:
            print()
            print("AR Commands:")
            print("  /ar               - Toggle AR mode")
            print("  /mode             - Show current mode")
            print()
            print("AR Trigger Phrases:")
            print("  'show me...'      - Switch to visual mode")
            print("  'look at...'      - Visual guidance")
            print("  'guide me...'     - Step-by-step AR help")
            print("  'go back to chat' - Return to text mode")
        if self.enable_rag:
            print()
            print("RAG Commands:")
            print("  /learn <text>     - Teach Elle something")
            print("  /knowledge        - Show RAG system status")
        print()
        print("Tips:")
        print("  - Elle responds best to concrete situations")
        print("  - She may stay quiet if you're in flow")
        print("  - Short, plain questions work well")
        print()

    def _print_context(self):
        """Print conversation context."""
        context = self.chat_adapter.get_context_summary()
        print()
        print("Recent conversation:")
        print(context if context else "(No conversation yet)")
        print()

    def _print_goodbye(self):
        """Print goodbye message."""
        print()
        print("-" * 50)
        print("  (Elle nods.)")
        print("-" * 50)
        print()


def create_llm_client(
    provider: str,
    model: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client based on provider."""

    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it or use --provider openai"
            )
        return AnthropicClient(
            api_key=api_key,
            model=model or "claude-sonnet-4-5-20250514",
        )

    elif provider_lower == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it or use --provider anthropic"
            )
        return OpenAIClient(
            api_key=api_key,
            model=model or "gpt-4",
        )

    elif provider_lower in ("ollama", "local"):
        return OllamaClient(
            model=model or "qwen3.5:9b",
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'ollama'")


def create_elle_engine(llm_client: LLMClient) -> ElleEngine:
    """Create Elle engine with chat prompt."""

    # Use chat-specific prompt
    chat_prompt_path = Path(__file__).parent / "core" / "prompt" / "chat_prompt.txt"

    # Fallback to base prompt if chat prompt doesn't exist
    if not chat_prompt_path.exists():
        logger.warning("Chat prompt not found, using base prompt")
        chat_prompt_path = Path(__file__).parent / "core" / "prompt" / "base_prompt.txt"

    prompt_builder = PromptBuilder(base_prompt_path=chat_prompt_path)

    # Create policy
    policy = EllePolicy(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    # Create memory store
    memory = InMemoryMemoryStore()

    # Create tool registry
    tools = create_default_registry()

    # Create engine
    engine = ElleEngine(
        policy=policy,
        memory=memory,
        tools=tools,
    )

    return engine


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Chat with Elle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m elle.chat
  python -m elle.chat --provider anthropic
  python -m elle.chat --provider openai --model gpt-4
  python -m elle.chat --user "Blake"

Environment variables:
  ANTHROPIC_API_KEY - Required for Anthropic provider
  OPENAI_API_KEY    - Required for OpenAI provider
"""
    )

    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )

    parser.add_argument(
        "--model",
        help="Model name (default: provider-specific)"
    )

    parser.add_argument(
        "--user",
        default="You",
        help="Your name for the chat (default: You)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--enable-ar",
        action="store_true",
        help="Enable AR mode transitions"
    )

    parser.add_argument(
        "--ar-url",
        default=None,
        help="AR client URL (default: local demo.html)"
    )

    parser.add_argument(
        "--enable-rag",
        action="store_true",
        default=True,  # RAG enabled by default if available
        help="Enable RAG knowledge retrieval (default: enabled if available)"
    )

    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG knowledge retrieval"
    )

    parser.add_argument(
        "--rag-provider",
        choices=["ollama", "anthropic", "openai"],
        default="ollama",
        help="LLM provider for RAG (default: ollama)"
    )

    parser.add_argument(
        "--rag-model",
        default=None,
        help="Model for RAG (default: provider-specific)"
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("elle").setLevel(logging.DEBUG)

    try:
        # Create LLM client
        print(f"Connecting to {args.provider}...")
        llm_client = create_llm_client(args.provider, args.model)

        # Create Elle engine
        engine = create_elle_engine(llm_client)

        # Create chat adapter (with AR bridge if enabled)
        chat_adapter = create_chat_adapter(
            user_name=args.user,
            enable_ar_bridge=args.enable_ar,
        )

        # Create RAG bridge if enabled (default: on, unless --no-rag)
        rag_bridge = None
        enable_rag = args.enable_rag and not args.no_rag

        if enable_rag:
            if not RAG_AVAILABLE:
                print("  RAG: HoloLoom not installed, continuing without knowledge retrieval.")
            else:
                print(f"  Enabling RAG with {args.rag_provider}...")
                rag_bridge = create_rag_bridge(
                    llm_provider=args.rag_provider,
                    llm_model=args.rag_model or (
                        "qwen3.5:9b" if args.rag_provider == "ollama" else None
                    ),
                    enable_caching=True,
                )

        # Create and start session
        session = ChatSession(
            engine=engine,
            chat_adapter=chat_adapter,
            user_name=args.user,
            enable_ar=args.enable_ar,
            ar_client_url=args.ar_url,
            enable_rag=enable_rag and rag_bridge is not None,
            rag_bridge=rag_bridge,
        )

        session.start()

    except ValueError as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
