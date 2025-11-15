#!/usr/bin/env python3
"""
HoloLoom-Promptly Integration Demo
===================================
Demonstrates bidirectional integration between HoloLoom and Promptly.

This demo shows:
1. Loading Promptly prompts into HoloLoom memory
2. Using HoloLoom's neural policy to evaluate Promptly prompts
3. Converting HoloLoom memories to Promptly prompts
4. End-to-end workflow examples

Requirements:
- Both HoloLoom and Promptly installed
- Initialized Promptly repository with sample prompts
- integration_config.yaml in repository root
"""

import sys
import os
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

print("=" * 80)
print("HoloLoom-Promptly Integration Demo")
print("=" * 80)

# Check if both systems are available
try:
    from Promptly.promptly.promptly import Promptly
    print("[OK] Promptly loaded successfully")
    PROMPTLY_OK = True
except ImportError as e:
    print(f"[ERROR] Promptly not available: {e}")
    PROMPTLY_OK = False

try:
    from HoloLoom.Documentation.types import MemoryShard, Query, Features
    from HoloLoom.config import Config
    print("[OK] HoloLoom loaded successfully")
    HOLOLOOM_OK = True
except ImportError as e:
    print(f"[ERROR] HoloLoom not available: {e}")
    HOLOLOOM_OK = False

if not (PROMPTLY_OK and HOLOLOOM_OK):
    print("\n[FATAL] Both systems must be available for integration demo")
    sys.exit(1)

print()


# ==============================================================================
# Demo 1: Load Promptly prompts into HoloLoom
# ==============================================================================

def demo_1_promptly_to_hololoom():
    """Demo: Load Promptly prompts as HoloLoom memory shards."""
    print("\n" + "=" * 80)
    print("Demo 1: Loading Promptly Prompts into HoloLoom")
    print("=" * 80)

    try:
        from HoloLoom.integrations.promptly import PromptlyLoader

        # Initialize Promptly
        promptly = Promptly(root_dir=str(repo_root))

        # Create loader
        loader = PromptlyLoader(promptly_instance=promptly)

        # Load prompts as shards
        print("\nLoading prompts from Promptly repository...")
        shards = loader.prompts_to_shards()

        print(f"\nLoaded {len(shards)} prompts as HoloLoom MemoryShards:")
        for i, shard in enumerate(shards[:5], 1):  # Show first 5
            print(f"\n  Shard {i}:")
            print(f"    ID: {shard.id}")
            print(f"    Text: {shard.text[:80]}...")
            print(f"    Episode: {shard.episode}")
            print(f"    Entities: {shard.entities}")
            print(f"    Motifs: {shard.motifs[:3]}")
            print(f"    Metadata: {list(shard.metadata.keys())}")

        if len(shards) > 5:
            print(f"\n  ... and {len(shards) - 5} more shards")

        print("\n[SUCCESS] Promptly prompts loaded into HoloLoom format")
        return shards

    except Exception as e:
        print(f"\n[ERROR] Demo 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==============================================================================
# Demo 2: Use HoloLoom to evaluate Promptly prompts
# ==============================================================================

def demo_2_hololoom_evaluator():
    """Demo: Use HoloLoom neural policy to evaluate prompts."""
    print("\n" + "=" * 80)
    print("Demo 2: Evaluating Prompts with HoloLoom Neural Policy")
    print("=" * 80)

    try:
        from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

        # Initialize evaluator
        print("\nInitializing HoloLoom evaluator (fast mode)...")
        evaluator = HoloLoomEvaluator(config_mode='fast')

        # Create test prompts
        test_prompts = [
            {
                'name': 'summarizer',
                'content': 'Provide a concise summary of the following text: {text}',
                'metadata': {'tags': ['summarization', 'text-processing']}
            },
            {
                'name': 'classifier',
                'content': 'Classify the sentiment of this text as positive, negative, or neutral: {text}',
                'metadata': {'tags': ['classification', 'sentiment']}
            },
            {
                'name': 'extractor',
                'content': 'Extract all entities (people, places, organizations) from: {text}',
                'metadata': {'tags': ['extraction', 'ner']}
            }
        ]

        # Evaluate each prompt
        print("\nEvaluating prompts...")
        results = []

        for prompt in test_prompts:
            test_input = {'text': 'HoloLoom is a neural decision-making system.'}
            result = evaluator.evaluate_prompt(
                prompt['name'],
                prompt['content'],
                test_input
            )
            results.append(result)

            print(f"\n  Prompt: {result.prompt_name}")
            print(f"    Score: {result.score:.3f}")
            print(f"    Tool Selection: {result.tool_selection}")
            print(f"    Metadata: {result.metadata}")

        print("\n[SUCCESS] HoloLoom evaluation completed")
        return results

    except Exception as e:
        print(f"\n[ERROR] Demo 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==============================================================================
# Demo 3: Batch evaluation with HoloLoom
# ==============================================================================

def demo_3_batch_evaluation():
    """Demo: Batch evaluate multiple prompts."""
    print("\n" + "=" * 80)
    print("Demo 3: Batch Evaluation with HoloLoom")
    print("=" * 80)

    try:
        from Promptly.promptly.integrations.hololoom import HoloLoomEvaluator

        evaluator = HoloLoomEvaluator(config_mode='fast')

        # Batch of prompts
        prompts = [
            {
                'name': f'test_prompt_{i}',
                'content': f'Process this data with method {i}: {{data}}',
                'metadata': {'tags': [f'method-{i}']}
            }
            for i in range(5)
        ]

        print(f"\nBatch evaluating {len(prompts)} prompts...")

        test_inputs = {'data': 'sample input data'}
        results = evaluator.batch_evaluate(prompts, test_inputs)

        print("\nBatch evaluation results:")
        scores = []
        for result in results:
            print(f"  {result.prompt_name}: {result.score:.3f}")
            scores.append(result.score)

        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nAverage score: {avg_score:.3f}")

        print("\n[SUCCESS] Batch evaluation completed")
        return results

    except Exception as e:
        print(f"\n[ERROR] Demo 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ==============================================================================
# Demo 4: Load integration config
# ==============================================================================

def demo_4_integration_config():
    """Demo: Load and display integration configuration."""
    print("\n" + "=" * 80)
    print("Demo 4: Integration Configuration")
    print("=" * 80)

    try:
        from HoloLoom.integrations.promptly import load_integration_config

        config_path = repo_root / 'integration_config.yaml'
        print(f"\nLoading integration config from: {config_path}")

        config = load_integration_config(str(config_path))

        if config:
            print("\nIntegration configuration:")
            print(f"  Version: {config.get('version')}")
            print(f"  Integration enabled: {config.get('integration', {}).get('enabled')}")
            print(f"  HoloLoom mode: {config.get('hololoom', {}).get('mode')}")
            print(f"  Promptly default branch: {config.get('promptly', {}).get('default_branch')}")
            print(f"  Default evaluator: {config.get('evaluators', {}).get('default')}")

            # Display feature flags
            features = config.get('features', {})
            print("\n  Feature flags:")
            for key, value in features.items():
                print(f"    {key}: {value}")

            print("\n[SUCCESS] Configuration loaded")
        else:
            print("\n[WARNING] No configuration found, using defaults")

        return config

    except Exception as e:
        print(f"\n[ERROR] Demo 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ==============================================================================
# Demo 5: Full workflow - Promptly -> HoloLoom -> Evaluation
# ==============================================================================

def demo_5_full_workflow():
    """Demo: Complete workflow combining all features."""
    print("\n" + "=" * 80)
    print("Demo 5: Full Integration Workflow")
    print("=" * 80)

    try:
        from HoloLoom.integrations.promptly import PromptlyLoader
        from Promptly.promptly.integrations.hololoom import (
            HoloLoomEvaluator,
            create_evaluator_function
        )

        # Step 1: Initialize Promptly and add sample prompts
        print("\nStep 1: Setting up Promptly repository...")
        promptly = Promptly(root_dir=str(repo_root))

        # Try to add a sample prompt (may fail if already exists)
        try:
            promptly.add(
                'workflow_test',
                'Analyze this text for key insights: {text}',
                metadata={'tags': ['analysis', 'demo']}
            )
            print("  Added sample prompt to Promptly")
        except Exception as e:
            print(f"  Using existing prompts (add failed: {e})")

        # Step 2: Load prompts into HoloLoom format
        print("\nStep 2: Loading prompts into HoloLoom...")
        loader = PromptlyLoader(promptly_instance=promptly)
        shards = loader.prompts_to_shards()
        print(f"  Loaded {len(shards)} prompts as MemoryShards")

        # Step 3: Evaluate prompts with HoloLoom
        print("\nStep 3: Evaluating prompts with HoloLoom neural policy...")
        evaluator = HoloLoomEvaluator(config_mode='fast')

        # Get latest prompt
        prompts = promptly.list_prompts()
        if prompts:
            latest = prompts[0]
            prompt_data = promptly.get(latest['name'])

            if prompt_data:
                result = evaluator.evaluate_prompt(
                    prompt_data['name'],
                    prompt_data['content'],
                    {'text': 'Integration test data'}
                )

                print(f"  Evaluated: {result.prompt_name}")
                print(f"  Score: {result.score:.3f}")

        # Step 4: Create evaluator function for Promptly
        print("\nStep 4: Creating Promptly-compatible evaluator...")
        eval_fn = create_evaluator_function(config_mode='fast')
        print("  Evaluator function created")

        # Step 5: Show integration stats
        print("\nStep 5: Integration statistics:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Total shards: {len(shards)}")
        print("  Integration: Active")

        print("\n[SUCCESS] Full workflow completed")

    except Exception as e:
        print(f"\n[ERROR] Demo 5 failed: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# Main demo runner
# ==============================================================================

def main():
    """Run all demos."""
    print("\nStarting integration demos...\n")

    # Check if Promptly is initialized
    promptly_dir = repo_root / '.promptly'
    if not promptly_dir.exists():
        print("[WARNING] Promptly repository not initialized")
        print("Run 'cd Promptly/promptly && python promptly.py init' first")
        print("\nWould you like to continue with available demos? (y/n): ", end="")
        response = input().lower()
        if response != 'y':
            return

    # Run demos
    demos = [
        ("Load Promptly -> HoloLoom", demo_1_promptly_to_hololoom),
        ("HoloLoom Evaluator", demo_2_hololoom_evaluator),
        ("Batch Evaluation", demo_3_batch_evaluation),
        ("Integration Config", demo_4_integration_config),
        ("Full Workflow", demo_5_full_workflow),
    ]

    results = {}
    for name, demo_fn in demos:
        try:
            print(f"\n{'=' * 80}")
            print(f"Running: {name}")
            result = demo_fn()
            results[name] = result
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Demo cancelled by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Demo '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("Demo Summary")
    print("=" * 80)
    for name, result in results.items():
        status = "[OK]" if result or result == {} else "[FAILED]"
        print(f"{status} {name}")

    print("\n" + "=" * 80)
    print("Integration demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
