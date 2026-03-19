"""
HoloLoom Lite - Simple REPL

A minimal read-eval-print loop for HoloLoom Lite.
No dependencies required - uses standard library only.

Usage:
    python -m HoloLoom.lite repl

Commands:
    /help       Show help
    /history    Show conversation history
    /memories   Show stored memories
    /clear      Clear history
    /quit       Exit REPL

Date: December 2025
"""

import asyncio


def start_repl(**kwargs) -> None:
    """Start the simple REPL."""
    asyncio.run(_repl_main(**kwargs))


async def _repl_main(**kwargs) -> None:
    """Main REPL loop."""
    from hololoom.lite import HoloLoomLite

    print()
    print("=" * 60)
    print("  HoloLoom Lite REPL")
    print("=" * 60)
    print()
    print("  Type a query to interact with HoloLoom.")
    print("  Commands: /help, /history, /memories, /clear, /quit")
    print()
    print("  Initializing...")
    print()

    # Track conversation history
    history: list[tuple[str, str]] = []

    async with HoloLoomLite() as loom:
        print("  Ready! Type your first query.\n")

        while True:
            try:
                # Get input
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input[1:].lower().split()[0]

                    if command in ("quit", "exit", "q"):
                        print("\nGoodbye!")
                        break

                    elif command == "help":
                        _print_help()

                    elif command == "history":
                        _print_history(history)

                    elif command == "memories":
                        await _print_memories(loom)

                    elif command == "clear":
                        history.clear()
                        print("History cleared.")

                    elif command == "experience":
                        # Store explicit memory: /experience This is important
                        content = user_input[12:].strip()
                        if content:
                            mem = await loom.experience(content)
                            print(f"Stored memory: {mem.id}")
                        else:
                            print("Usage: /experience <content>")

                    elif command == "reason":
                        # Agentic reasoning: /reason verify What is Thompson Sampling?
                        parts = user_input[8:].strip().split(" ", 1)
                        if len(parts) >= 2:
                            mode, query = parts
                            if mode in ("direct", "verify", "research", "plan_execute"):
                                result = await loom.reason(query, mode=mode)
                                print(f"\n{result.text}\n")
                                print(f"[Confidence: {result.confidence:.2%}, Mode: {mode}]\n")
                            else:
                                print("Mode must be: direct, verify, research, or plan_execute")
                        else:
                            print("Usage: /reason <mode> <query>")

                    else:
                        print(f"Unknown command: /{command}")
                        print("Type /help for available commands.")

                    continue

                # Regular query - recall memories and reason
                print()

                # First, store the query as an experience
                await loom.experience(user_input, context={"type": "query"})

                # Try to reason about it
                result = await loom.reason(user_input, mode="direct")

                # Display result
                print(result.text)
                print()
                if result.confidence > 0:
                    print(f"[Confidence: {result.confidence:.2%}]")
                print()

                # Track history
                history.append((user_input, result.text))

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")

            except EOFError:
                print("\nGoodbye!")
                break

            except Exception as e:
                print(f"\nError: {e}\n")


def _print_help():
    """Print help message."""
    print("""
Commands:
  /help                     Show this help
  /history                  Show conversation history
  /memories                 Show stored memories
  /clear                    Clear history
  /experience <content>     Store explicit memory
  /reason <mode> <query>    Agentic reasoning (modes: direct, verify, research, plan_execute)
  /quit                     Exit REPL

Regular input is treated as a query and will:
  1. Store the query as a memory
  2. Use direct reasoning to find an answer
  3. Display the result with confidence score
""")


def _print_history(history: list[tuple[str, str]]):
    """Print conversation history."""
    if not history:
        print("No history yet.")
        return

    print(f"\nConversation history ({len(history)} turns):\n")
    for i, (query, response) in enumerate(history, 1):
        print(f"  [{i}] You: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"      Bot: {response[:60]}{'...' if len(response) > 60 else ''}")
        print()


async def _print_memories(loom):
    """Print stored memories."""
    metrics = loom.get_metrics()
    print(f"\nMemories: {metrics['n_memories']}")
    print(f"Connections: {metrics['n_connections']}")
    print()

    # Try to get some memories
    if metrics['n_memories'] > 0:
        memories = await loom.recall("", limit=10)
        if memories:
            print("Recent memories:")
            for mem in memories[:10]:
                text = mem.text[:60] + "..." if len(mem.text) > 60 else mem.text
                print(f"  - {text}")
    print()


if __name__ == "__main__":
    start_repl()
