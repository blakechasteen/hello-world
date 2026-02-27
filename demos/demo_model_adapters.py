"""
Model Adapter Demonstration

Demonstrates all three model-specific adapters:
- ClaudeAdapter (thinking tags, artifacts, XML constraints)
- GeminiAdapter (multimodal, code execution, grounding)
- GPTAdapter (structured outputs, function calling)

Shows:
1. Creating adapters programmatically
2. Applying enhancements to core prompts
3. Comparing performance characteristics
4. Integration with HoloLoom Config

Run:
    python demos/demo_model_adapters.py
"""

from hololoom.prompting import (
    create_adapter,
    create_metaprompt,
    create_metaprompt_auto,
    enhance_request
)
from hololoom.config import Config


def demo_individual_adapters():
    """Demonstrate each adapter individually"""
    print("=" * 80)
    print("DEMO 1: Individual Model Adapters")
    print("=" * 80)

    request = "Explain Thompson Sampling and show a Python implementation"

    # Claude Adapter
    print("\n--- Claude Adapter ---")
    adapter = create_adapter("anthropic")
    print(f"Adapter name: {adapter.name}")
    print(f"LLM provider: {adapter.llm_provider}")
    print(f"Supported features: {adapter.supported_features}")

    perf = adapter.get_performance_characteristics()
    print(f"Latency overhead: +{perf.latency_overhead_ms}ms")
    print(f"Quality boost: +{perf.quality_boost_percent}%")
    print(f"Thinking tags: {perf.thinking_tags}")
    print(f"Artifacts: {perf.artifacts}")

    # Gemini Adapter
    print("\n--- Gemini Adapter ---")
    adapter = create_adapter("google")
    print(f"Adapter name: {adapter.name}")
    print(f"LLM provider: {adapter.llm_provider}")
    print(f"Supported features: {adapter.supported_features}")

    perf = adapter.get_performance_characteristics()
    print(f"Latency overhead: +{perf.latency_overhead_ms}ms")
    print(f"Quality boost: +{perf.quality_boost_percent}%")
    print(f"Multimodal: {perf.multimodal}")
    print(f"Code execution: {perf.code_execution}")

    # GPT Adapter
    print("\n--- GPT Adapter ---")
    adapter = create_adapter("openai")
    print(f"Adapter name: {adapter.name}")
    print(f"LLM provider: {adapter.llm_provider}")
    print(f"Supported features: {adapter.supported_features}")

    perf = adapter.get_performance_characteristics()
    print(f"Latency overhead: +{perf.latency_overhead_ms}ms")
    print(f"Quality boost: +{perf.quality_boost_percent}%")
    print(f"JSON mode: {perf.json_mode}")
    print(f"Function calling: {perf.function_calling}")


def demo_metaprompt_integration():
    """Demonstrate metaprompt creation with adapters"""
    print("\n" + "=" * 80)
    print("DEMO 2: Metaprompt Integration with Config")
    print("=" * 80)

    request = "Help me prepare for tomorrow's meeting about Q4 planning"

    # Claude (via Config)
    print("\n--- Using Claude via Config ---")
    config = Config.fused()
    config.llm_provider = "anthropic"
    config.llm_model = "claude-3-5-sonnet-20241022"

    enhanced = create_metaprompt(request, config=config)
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Contains thinking tags: {'<thinking>' in enhanced}")
    print(f"Contains artifact tags: {'<antArtifact>' in enhanced}")
    print("\nFirst 500 chars:")
    print(enhanced[:500])

    # Gemini (via Config)
    print("\n--- Using Gemini via Config ---")
    config = Config.fused()
    config.llm_provider = "google"
    config.llm_model = "gemini-pro"

    enhanced = create_metaprompt(request, config=config)
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Mentions multimodal: {'Multimodal' in enhanced}")
    print("\nFirst 500 chars:")
    print(enhanced[:500])

    # GPT (via Config)
    print("\n--- Using GPT via Config ---")
    config = Config.fused()
    config.llm_provider = "openai"
    config.llm_model = "gpt-4"

    enhanced = create_metaprompt(request, config=config)
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Mentions structured outputs: {'Structured Outputs' in enhanced}")
    print("\nFirst 500 chars:")
    print(enhanced[:500])


def demo_one_liner_api():
    """Demonstrate the simplest one-liner API"""
    print("\n" + "=" * 80)
    print("DEMO 3: One-Liner API")
    print("=" * 80)

    request = "Write a function to calculate Fibonacci numbers"

    # Claude (one-liner)
    print("\n--- Claude (one-liner) ---")
    enhanced = enhance_request(request, provider="anthropic")
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Contains <thinking>: {'<thinking>' in enhanced}")

    # Gemini (one-liner)
    print("\n--- Gemini (one-liner) ---")
    enhanced = enhance_request(request, provider="google")
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Mentions code execution: {'code execution' in enhanced.lower()}")

    # GPT (one-liner)
    print("\n--- GPT (one-liner) ---")
    enhanced = enhance_request(request, provider="openai")
    print(f"Enhanced prompt length: {len(enhanced)} chars")
    print(f"Mentions JSON: {'JSON' in enhanced}")


def demo_auto_detection():
    """Demonstrate auto-detection with metaprompting"""
    print("\n" + "=" * 80)
    print("DEMO 4: Auto-Detection + Metaprompting")
    print("=" * 80)

    requests = [
        "Analyze this security vulnerability",
        "Help me write a blog post",
        "Optimize this SQL query"
    ]

    config = Config.fused()
    config.llm_provider = "anthropic"

    for request in requests:
        print(f"\nRequest: \"{request}\"")
        enhanced = create_metaprompt_auto(request, config=config)
        print(f"Enhanced prompt length: {len(enhanced)} chars")
        print(f"Strategy applied: {'---' in enhanced}")  # Separator indicates strategy


def demo_feature_flags():
    """Demonstrate custom feature flags"""
    print("\n" + "=" * 80)
    print("DEMO 5: Custom Feature Flags")
    print("=" * 80)

    request = "Refactor this code for better performance"
    config = Config.fused()
    config.llm_provider = "anthropic"

    # Default features
    print("\n--- Default Features ---")
    enhanced = create_metaprompt(request, config=config)
    print(f"Length: {len(enhanced)} chars")

    # Custom features (disable artifacts)
    print("\n--- Custom Features (no artifacts) ---")
    enhanced = create_metaprompt(
        request,
        config=config,
        features={'thinking_tags': True, 'artifacts': False}
    )
    print(f"Length: {len(enhanced)} chars")
    print(f"Contains artifacts: {'<antArtifact>' in enhanced}")

    # Multi-pass validation enabled
    print("\n--- Multi-Pass Validation Enabled ---")
    enhanced = create_metaprompt(
        request,
        config=config,
        features={'multi_pass_validation': True}
    )
    print(f"Length: {len(enhanced)} chars")
    print(f"Contains validation passes: {'Pass 1:' in enhanced}")


def demo_performance_comparison():
    """Compare performance characteristics across adapters"""
    print("\n" + "=" * 80)
    print("DEMO 6: Performance Comparison")
    print("=" * 80)

    adapters = [
        ("Claude", "anthropic"),
        ("Gemini", "google"),
        ("GPT", "openai")
    ]

    print("\n{:<10} {:<15} {:<12} {:<20}".format(
        "Adapter", "Latency (ms)", "Quality (%)", "Key Features"
    ))
    print("-" * 80)

    for name, provider in adapters:
        adapter = create_adapter(provider)
        perf = adapter.get_performance_characteristics()

        # Get key features
        features = adapter.supported_features[:3]  # First 3
        features_str = ", ".join(features)

        print("{:<10} +{:<14} +{:<11} {}".format(
            name,
            f"{perf.latency_overhead_ms}",
            f"{perf.quality_boost_percent}",
            features_str
        ))


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("MODEL ADAPTER DEMONSTRATION")
    print("=" * 80)
    print("\nShowing integration of model-specific adapters with HoloLoom")
    print("metaprompting system.")

    try:
        demo_individual_adapters()
        demo_metaprompt_integration()
        demo_one_liner_api()
        demo_auto_detection()
        demo_feature_flags()
        demo_performance_comparison()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("\nAll adapters working correctly!")
        print("\nNext steps:")
        print("1. Try with real LLM calls")
        print("2. Measure actual performance improvements")
        print("3. Test with complex prompts")
        print("4. Integrate with prompt chaining (Phase 2)")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
