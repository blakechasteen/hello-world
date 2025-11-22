"""
Demo: Multi-Provider LLM Integration
======================================

Shows how to use 20+ LLM providers through unified interface.

Author: Claude Code
Date: 2025-11-20
"""

from pathlib import Path
import sys
import os

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Try to import, but don't fail if not available
try:
    from HoloLoom.integrations.langchain import (
        MultiProviderLLM,
        create_llm,
        list_llm_providers,
        create_best_available_llm
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


def demo_basic_usage():
    """Demo basic LLM usage."""
    print("\n" + "="*60)
    print("Demo 1: Basic LLM Usage")
    print("="*60 + "\n")

    print("[OpenAI (GPT-4)]")
    print("""
    llm = MultiProviderLLM(provider='openai', model='gpt-4')
    response = llm('Explain quantum computing')
    print(response)
    """)

    print("\n[Anthropic (Claude)]")
    print("""
    llm = MultiProviderLLM(provider='anthropic', model='claude-3-5-sonnet-20241022')
    response = llm('Write a Python function')
    print(response)
    """)

    print("\n[Local (Ollama)]")
    print("""
    llm = MultiProviderLLM(provider='ollama', model='llama3.2:3b')
    response = llm('Hello, how are you?')
    print(response)
    """)


def demo_chat_mode():
    """Demo chat-style generation."""
    print("\n" + "="*60)
    print("Demo 2: Chat Mode")
    print("="*60 + "\n")

    print("[Multi-Turn Conversation]")
    print("""
    llm = MultiProviderLLM(provider='anthropic')

    response = llm.chat([
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is Python?'},
        {'role': 'assistant', 'content': 'Python is a programming language...'},
        {'role': 'user', 'content': 'Show me an example'}
    ])

    print(response)
    """)


def demo_streaming():
    """Demo streaming generation."""
    print("\n" + "="*60)
    print("Demo 3: Streaming")
    print("="*60 + "\n")

    print("[Token-by-Token Streaming]")
    print("""
    llm = MultiProviderLLM(provider='openai')

    for token in llm.stream('Write a short story'):
        print(token, end='', flush=True)
    """)


def demo_auto_fallback():
    """Demo auto-fallback."""
    print("\n" + "="*60)
    print("Demo 4: Auto-Fallback")
    print("="*60 + "\n")

    print("[Automatic Provider Selection]")
    print("""
    from HoloLoom.integrations.langchain import create_best_available_llm

    llm = create_best_available_llm()
    # Tries in order:
    # 1. Anthropic (if ANTHROPIC_API_KEY set)
    # 2. OpenAI (if OPENAI_API_KEY set)
    # 3. Cohere (if COHERE_API_KEY set)
    # 4. Ollama (local fallback)

    response = llm('Hello!')
    """)


def demo_available_providers():
    """Show available providers."""
    print("\n" + "="*60)
    print("Demo 5: Available Providers")
    print("="*60 + "\n")

    try:
        providers = list_llm_providers()

        for provider, models in providers.items():
            print(f"\n{provider.upper()}:")
            for model in models[:3]:
                print(f"   • {model}")
            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more models")

    except Exception as e:
        print(f"⚠️  {e}\n")
        print("Example providers:")
        print("\nOPENAI:")
        print("   • gpt-4-turbo-preview")
        print("   • gpt-3.5-turbo")
        print("\nANTHROPIC:")
        print("   • claude-3-5-sonnet-20241022")
        print("   • claude-3-opus-20240229")
        print("\nCOHERE:")
        print("   • command-r-plus")
        print("\nOLLAMA:")
        print("   • llama3.2:3b")
        print("   • mistral:7b")


def demo_usage_tracking():
    """Demo usage tracking."""
    print("\n" + "="*60)
    print("Demo 6: Usage Tracking")
    print("="*60 + "\n")

    print("[Track Token Usage]")
    print("""
    llm = MultiProviderLLM(provider='openai')

    # Generate responses
    llm('Question 1')
    llm('Question 2')
    llm('Question 3')

    # Get statistics
    stats = llm.get_usage_stats()
    print(f"Provider: {stats['provider']}")
    print(f"Model: {stats['model']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")
    """)


def demo_integration_with_hololoom():
    """Demo HoloLoom integration."""
    print("\n" + "="*60)
    print("Demo 7: Integration with HoloLoom")
    print("="*60 + "\n")

    print("[HoloLoom Recall + LLM Generation]")
    print("""
    from HoloLoom import HoloLoom
    from HoloLoom.integrations.langchain import MultiProviderLLM

    async def main():
        async with HoloLoom() as loom:
            # Recall memories
            memories = await loom.recall('What is Thompson Sampling?')
            context = '\\n'.join([m.content for m in memories[:3]])

            # Generate answer with LLM
            llm = MultiProviderLLM(provider='anthropic')
            prompt = f'Context:\\n{context}\\n\\nQuestion: What is Thompson Sampling?\\n\\nAnswer:'
            response = llm(prompt)

            print(response)

    import asyncio
    asyncio.run(main())
    """)


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "Multi-Provider LLM Demo")
    print("="*70)

    demo_basic_usage()
    demo_chat_mode()
    demo_streaming()
    demo_auto_fallback()
    demo_available_providers()
    demo_usage_tracking()
    demo_integration_with_hololoom()

    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70 + "\n")

    print("Next Steps:")
    print("1. Set API keys:")
    print("   export OPENAI_API_KEY='sk-...'")
    print("   export ANTHROPIC_API_KEY='sk-ant-...'")
    print("2. Install providers:")
    print("   pip install langchain-openai langchain-anthropic")
    print("3. Try different providers and models\n")


if __name__ == "__main__":
    main()
