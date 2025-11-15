"""
Elle Core - Interactive Demo

Run an interactive CLI session with Elle.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from elle_core.engine import create_engine
from elle_core.core import create_llm_client
from elle_core.adapters.cli import CLI


async def main():
    """Run interactive CLI."""
    print("\n" + "=" * 70)
    print("  Elle Core - Interactive Demo")
    print("=" * 70)

    # Check for API key
    if os.getenv("ANTHROPIC_API_KEY"):
        provider = "claude"
        print("\n✅ Using Claude (Anthropic)")
    elif os.getenv("OPENAI_API_KEY"):
        provider = "openai"
        print("\n✅ Using OpenAI")
    else:
        provider = "ollama"
        print("\n⚠️  No API key found, using Ollama (local)")
        print("Make sure Ollama is running: ollama serve")

    # Create LLM client
    try:
        llm_client = create_llm_client(provider=provider)
    except Exception as e:
        print(f"\n❌ Failed to create LLM client: {e}")
        print("\nSet API key or start Ollama:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("  # OR")
        print("  ollama serve")
        return

    # Create engine
    engine = create_engine(llm_client=llm_client)

    # Create CLI
    cli = CLI(engine=engine)

    # Run interactive session
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
