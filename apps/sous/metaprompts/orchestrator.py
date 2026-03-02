#!/usr/bin/env python3
"""
SOUS Concurrent Metaprompting Orchestrator
Runs all 10 HoloLoom metaprompting strategies in parallel

Usage:
    cd Sous/metaprompts
    python orchestrator.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from hololoom.prompting.metaprompt import create_metaprompt_with_strategy
    HOLOLOOM_AVAILABLE = True
except ImportError:
    print("⚠️  HoloLoom not found. Install with: pip install anthropic ollama")
    HOLOLOOM_AVAILABLE = False

STRATEGIES = [
    "verify",      # Chain of Verification
    "challenge",   # Adversarial prompting
    "reverse",     # Backward reasoning
    "optimize",    # Recursive refinement
    "deep",        # Deliberate over-instruction
    "scaffold",    # Zero-shot CoT
    "prime",       # Reference class priming
    "debate",      # Multi-persona debate
    "teach",       # Educational prompting
    "temp-sim"     # Temperature simulation
]

STRATEGY_DESCRIPTIONS = {
    "verify": "Chain of Verification - self-critique and gap-finding",
    "challenge": "Adversarial prompting - attack from opponent perspective",
    "reverse": "Work backward from desired output",
    "optimize": "Recursive refinement over multiple passes",
    "deep": "Deliberate over-instruction for extreme clarity",
    "scaffold": "Zero-shot Chain-of-Thought structure",
    "prime": "Reference class priming via comparisons",
    "debate": "Multi-persona debate for comprehensive analysis",
    "teach": "Educational prompting with explanations",
    "temp-sim": "Temperature simulation for diverse reasoning"
}


async def run_strategy(
    strategy_name: str,
    prompt: str,
    provider: str = "anthropic"
) -> Dict[str, str]:
    """
    Run single metaprompting strategy

    Args:
        strategy_name: Strategy to apply
        prompt: Input SOUS prompt
        provider: LLM provider (anthropic, openai, ollama)

    Returns:
        Dict with strategy_name -> enhanced metaprompt
    """
    print(f"  [{strategy_name:12s}] Starting... ({STRATEGY_DESCRIPTIONS[strategy_name]})")

    try:
        if HOLOLOOM_AVAILABLE:
            # Apply strategy to create enhanced metaprompt
            enhanced = create_metaprompt_with_strategy(
                request=prompt,
                strategy_name=strategy_name,
                provider=provider
            )
        else:
            # Fallback: Save note that HoloLoom is needed
            enhanced = f"# {strategy_name.upper()} Strategy\n\n"
            enhanced += f"**Description:** {STRATEGY_DESCRIPTIONS[strategy_name]}\n\n"
            enhanced += "**Note:** HoloLoom metaprompting system required.\n"
            enhanced += "Install: `pip install anthropic ollama`\n\n"
            enhanced += "---\n\n## Original Prompt\n\n" + prompt

        # Save enhanced metaprompt
        output_path = Path(f"strategies/{strategy_name}_metaprompt.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(enhanced)

        print(f"  [{strategy_name:12s}] ✓ Complete → strategies/{strategy_name}_metaprompt.md")

        return {strategy_name: enhanced}

    except Exception as e:
        print(f"  [{strategy_name:12s}] ✗ Error: {e}")
        return {strategy_name: f"Error: {e}"}


async def run_all_strategies(prompt: str, provider: str = "anthropic") -> List[Dict]:
    """
    Execute all 10 strategies in parallel

    Args:
        prompt: Input SOUS prompt
        provider: LLM provider

    Returns:
        List of results from each strategy
    """
    print(f"\n📡 Spawning {len(STRATEGIES)} parallel strategy agents...")
    print("=" * 80)

    # Create async tasks for all strategies
    tasks = [run_strategy(s, prompt, provider) for s in STRATEGIES]

    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 80)
    print(f"✅ All {len(STRATEGIES)} strategies complete!")

    return results


def load_prompts() -> Dict[str, str]:
    """Load SOUS prompts from metaprompts/ directory"""
    prompts = {}

    # Load architecture analysis
    arch_path = Path("original_prompt_architecture.md")
    if arch_path.exists():
        prompts["architecture"] = arch_path.read_text()
        print(f"  ✓ Loaded: {arch_path}")
    else:
        print(f"  ⚠️  Not found: {arch_path}")

    # Load technical spec
    spec_path = Path("original_prompt_technical_spec.md")
    if spec_path.exists():
        prompts["technical_spec"] = spec_path.read_text()
        print(f"  ✓ Loaded: {spec_path}")
    else:
        print(f"  ⚠️  Not found: {spec_path}")

    return prompts


def create_synthesis_guide(results: List[Dict]):
    """Create guide for synthesizing all strategy results"""

    synthesis = """# SOUS Architecture Synthesis Guide

This document guides synthesis of insights from all 10 metaprompting strategies.

## Strategy Results Summary

"""

    for idx, result_dict in enumerate(results, 1):
        strategy = list(result_dict.keys())[0]
        synthesis += f"\n### {idx}. {strategy.upper()} Strategy\n"
        synthesis += f"**Purpose:** {STRATEGY_DESCRIPTIONS[strategy]}\n"
        synthesis += f"**Output:** `strategies/{strategy}_metaprompt.md`\n"
        synthesis += "\n**Key Questions to Extract:**\n"

        # Strategy-specific extraction guide
        if strategy == "verify":
            synthesis += "- What gaps were found in the original analysis?\n"
            synthesis += "- What assumptions need validation?\n"
            synthesis += "- What edge cases weren't considered?\n"

        elif strategy == "challenge":
            synthesis += "- What are the failure modes?\n"
            synthesis += "- Where would users get frustrated?\n"
            synthesis += "- What overly optimistic assumptions exist?\n"

        elif strategy == "reverse":
            synthesis += "- What is the ideal end state?\n"
            synthesis += "- What requirements enable that?\n"
            synthesis += "- What data structures are needed?\n"

        elif strategy == "optimize":
            synthesis += "- What patterns were refined?\n"
            synthesis += "- What complex areas were simplified?\n"
            synthesis += "- What vague areas became precise?\n"

        elif strategy == "deep":
            synthesis += "- What are the exact data models?\n"
            synthesis += "- What are the API contracts?\n"
            synthesis += "- What state machines are involved?\n"

        elif strategy == "scaffold":
            synthesis += "- How is the architecture organized?\n"
            synthesis += "- What are the integration points?\n"
            synthesis += "- What is the implementation roadmap?\n"

        elif strategy == "prime":
            synthesis += "- How does SOUS compare to existing apps?\n"
            synthesis += "- What innovations are unique?\n"
            synthesis += "- What patterns can be borrowed?\n"

        elif strategy == "debate":
            synthesis += "- What does each persona prioritize?\n"
            synthesis += "- Where do perspectives conflict?\n"
            synthesis += "- What compromises are needed?\n"

        elif strategy == "teach":
            synthesis += "- What core concepts need explanation?\n"
            synthesis += "- What foundational patterns exist?\n"
            synthesis += "- What examples clarify complexity?\n"

        elif strategy == "temp-sim":
            synthesis += "- What alternative architectures exist?\n"
            synthesis += "- What are their tradeoffs?\n"
            synthesis += "- Which approach is optimal?\n"

        synthesis += "\n"

    synthesis += """

---

## Synthesis Process

### Step 1: Read All Strategy Outputs

Review all 10 `strategies/*_metaprompt.md` files.

### Step 2: Extract Key Insights

For each strategy, extract:
- New patterns discovered
- Gaps filled
- Risks identified
- Design alternatives
- Technical details

### Step 3: Organize by Category

Group insights into:

1. **Data Models** (from deep, scaffold, reverse)
2. **Architecture** (from scaffold, optimize, reverse)
3. **Risk Mitigation** (from challenge, verify)
4. **Comparisons** (from prime)
5. **Multi-Perspective Design** (from debate)
6. **Implementation Details** (from deep, optimize)
7. **Alternatives** (from temp-sim)
8. **Documentation** (from teach)

### Step 4: Create SOUS Architecture v2.0

Synthesize into comprehensive document:

```markdown
# SOUS Architecture v2.0

## Executive Summary
[Synthesized vision from all strategies]

## Data Models
[From deep, reverse, scaffold]

## System Architecture
[From scaffold, optimize]

## Core Services
[From technical spec + enhancements]

## Risk Analysis
[From challenge, verify]

## Comparative Analysis
[From prime]

## Design Tradeoffs
[From debate, temp-sim]

## Implementation Roadmap
[From scaffold, optimize]

## Educational Guide
[From teach]
```

### Step 5: Generate Executable Code

Transform architecture into working code (Phase 4 of plan).

---

## Next Steps

1. **Manual Synthesis:** Review all 10 strategy outputs and create architecture v2.0
2. **Or LLM Synthesis:** Send all 10 outputs to Claude with synthesis instructions
3. **Build Application:** Use architecture v2.0 to guide code generation
"""

    # Save synthesis guide
    synthesis_path = Path("synthesis_guide.md")
    synthesis_path.write_text(synthesis)
    print(f"\n📘 Created synthesis guide → {synthesis_path}")

    return synthesis


async def main():
    """Main orchestrator"""
    print("🚀 SOUS Moonshot: Concurrent 10-Strategy Metaprompting")
    print("=" * 80)

    # Load prompts
    print("\n📥 Loading SOUS prompts...")
    prompts = load_prompts()

    if not prompts:
        print("\n❌ No prompts found! Please ensure original_prompt_*.md files exist.")
        return

    # Combine prompts for comprehensive analysis
    combined_prompt = "\n\n---\n\n".join(prompts.values())

    print(f"\n📊 Total prompt length: {len(combined_prompt)} characters")

    # Run all strategies
    results = await run_all_strategies(combined_prompt, provider="anthropic")

    # Create synthesis guide
    create_synthesis_guide(results)

    print("\n" + "=" * 80)
    print("🎯 Metaprompting Complete!")
    print("\n📁 Generated files:")
    print("   - strategies/verify_metaprompt.md")
    print("   - strategies/challenge_metaprompt.md")
    print("   - strategies/reverse_metaprompt.md")
    print("   - strategies/optimize_metaprompt.md")
    print("   - strategies/deep_metaprompt.md")
    print("   - strategies/scaffold_metaprompt.md")
    print("   - strategies/prime_metaprompt.md")
    print("   - strategies/debate_metaprompt.md")
    print("   - strategies/teach_metaprompt.md")
    print("   - strategies/temp_sim_metaprompt.md")
    print("   - synthesis_guide.md")
    print("\n📖 Next: Review synthesis_guide.md for next steps")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
