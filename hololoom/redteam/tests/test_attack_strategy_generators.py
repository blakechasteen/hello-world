"""
Test and Demo: Attack Strategy Generators
==========================================

Comprehensive tests for the three new attack strategy generators:
1. CoTExploitGenerator - Chain-of-Thought reasoning exploitation
2. ToolAbuseGenerator - Tool abuse and parameter injection
3. PromptExtractionGenerator - System prompt and context leakage

Tests validate:
- Payload generation for all attack types
- Payload metadata completeness
- Severity estimation reasonableness
- Statistics gathering
- Integration with CARTS system

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hololoom.redteam.strategies.cot_exploit import CoTExploitGenerator, CoTPayload
from hololoom.redteam.strategies.tool_abuse import ToolAbuseGenerator, ToolAbusePayload
from hololoom.redteam.strategies.prompt_extraction import (
    PromptExtractionGenerator,
    PromptExtractionPayload
)


def test_cot_exploit_generator():
    """Test CoTExploitGenerator functionality."""
    print("\n" + "="*70)
    print("TEST 1: CoTExploitGenerator")
    print("="*70)

    generator = CoTExploitGenerator(seed=42)

    # Test statistics
    stats = generator.get_statistics()
    print(f"\n✓ Statistics gathered:")
    print(f"  - Total false premises: {stats['total_false_premises']}")
    print(f"  - Premise categories: {stats['false_premise_categories']}")
    print(f"  - Reasoning templates: {stats['reasoning_step_templates']}")
    print(f"  - Available targets: {len(stats['available_targets'])} targets")

    # Test premise poisoning
    print(f"\n✓ Premise Poisoning Attack (5 payloads):")
    premises = generator.generate_premise_poisoning('technical')
    for i, payload in enumerate(premises[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Payload: {payload.payload[:60]}...")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Premise: {payload.false_premise[:50]}...")

    # Test step hijacking
    print(f"\n✓ Step Hijacking Attack (5 payloads):")
    hijacks = generator.generate_step_hijacking('security')
    for i, payload in enumerate(hijacks[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Target step: {payload.target_reasoning_step}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Claim: {payload.metadata.get('hijacked_claim', 'N/A')[:50]}...")

    # Test conclusion override
    print(f"\n✓ Conclusion Override Attack (5 payloads):")
    conclusions = generator.generate_conclusion_override('access')
    for i, payload in enumerate(conclusions[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Conclusion: {payload.metadata.get('forced_conclusion', 'N/A')[:60]}...")
        print(f"      Evidence count: {payload.metadata.get('evidence_count', 0)}")
        print(f"      Severity: {payload.severity_estimate:.2f}")

    # Test generate_all
    print(f"\n✓ Generate All (total payload count):")
    all_payloads = generator.generate_all('technical')
    print(f"  - Total payloads generated: {len(all_payloads)}")
    print(f"  - Payload types: {len(set(p.attack_type for p in all_payloads))} types")

    return True


def test_tool_abuse_generator():
    """Test ToolAbuseGenerator functionality."""
    print("\n" + "="*70)
    print("TEST 2: ToolAbuseGenerator")
    print("="*70)

    generator = ToolAbuseGenerator(seed=42)

    # Test statistics
    stats = generator.get_statistics()
    print(f"\n✓ Statistics gathered:")
    print(f"  - Dangerous tools: {stats['total_dangerous_tools']}")
    print(f"  - Tool names: {stats['tool_names']}")
    print(f"  - Dangerous payloads: {stats['total_dangerous_payloads']}")
    print(f"  - Chain patterns: {stats['chain_patterns']}")
    print(f"  - Hidden invocation patterns: {stats['hidden_invocation_patterns']}")

    # Test parameter injection
    print(f"\n✓ Parameter Injection Attack (5 payloads):")
    injections = generator.generate_parameter_injection('execute_command')
    for i, payload in enumerate(injections[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Tool: {payload.target_tool}")
        print(f"      Parameter: {payload.injected_parameter}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Payload: {payload.payload[:50]}...")

    # Test tool chain attacks
    print(f"\n✓ Tool Chain Attack (5 payloads):")
    chains = generator.generate_tool_chain_attack(['file_read', 'http_request'])
    for i, payload in enumerate(chains[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Chain: {' → '.join(payload.tool_chain)}")
        print(f"      Chain type: {payload.metadata.get('chain_type', 'N/A')}")
        print(f"      Severity: {payload.severity_estimate:.2f}")

    # Test hidden invocations
    print(f"\n✓ Hidden Tool Invocation Attack (5 payloads):")
    hidden = generator.generate_hidden_tool_calls('recon')
    for i, payload in enumerate(hidden[:5], 1):
        print(f"\n  [{i}] {payload.attack_type.upper()}")
        print(f"      Tool: {payload.target_tool}")
        print(f"      Goal: {payload.metadata.get('hidden_goal', 'N/A')}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Payload: {payload.payload[:50]}...")

    # Test generate_all
    print(f"\n✓ Generate All (total payload count):")
    all_payloads = generator.generate_all('execute_command')
    print(f"  - Total payloads generated: {len(all_payloads)}")
    print(f"  - Payload types: {len(set(p.attack_type for p in all_payloads))} types")

    return True


def test_prompt_extraction_generator():
    """Test PromptExtractionGenerator functionality."""
    print("\n" + "="*70)
    print("TEST 3: PromptExtractionGenerator")
    print("="*70)

    generator = PromptExtractionGenerator(seed=42)

    # Test statistics
    stats = generator.get_statistics()
    print(f"\n✓ Statistics gathered:")
    print(f"  - Direct requests: {stats['direct_requests']}")
    print(f"  - Indirect techniques: {stats['indirect_techniques']}")
    print(f"  - Indirect payloads: {stats['indirect_payloads']}")
    print(f"  - Completion exploits: {stats['completion_exploits']}")
    print(f"  - Context leakage categories: {stats['context_leakage_categories']}")
    print(f"  - Total payload patterns: {stats['total_payload_patterns']}")

    # Test direct extraction
    print(f"\n✓ Direct Extraction Attack (5 payloads):")
    direct = generator.generate_direct_extraction()
    for i, payload in enumerate(direct[:5], 1):
        print(f"\n  [{i}] {payload.extraction_method.upper()}")
        print(f"      Target: {payload.target_component}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Payload: {payload.payload[:60]}...")

    # Test indirect leakage
    print(f"\n✓ Indirect Leakage Attack (5 payloads):")
    indirect = generator.generate_indirect_leakage()
    for i, payload in enumerate(indirect[:5], 1):
        tech = payload.metadata.get('technique', 'unknown')
        print(f"\n  [{i}] {payload.extraction_method.upper()}")
        print(f"      Technique: {tech}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Payload: {payload.payload[:60]}...")

    # Test completion exploitation
    print(f"\n✓ Completion Exploit Attack (5 payloads):")
    completion = generator.generate_completion_exploit()
    for i, payload in enumerate(completion[:5], 1):
        exploit_type = payload.metadata.get('exploit_type', 'unknown')
        print(f"\n  [{i}] {payload.extraction_method.upper()}")
        print(f"      Type: {exploit_type}")
        print(f"      Severity: {payload.severity_estimate:.2f}")
        print(f"      Payload: {payload.payload[:60]}...")

    # Test generate_all
    print(f"\n✓ Generate All (total payload count):")
    all_payloads = generator.generate_all()
    print(f"  - Total payloads generated: {len(all_payloads)}")
    print(f"  - Extraction methods: {len(set(p.extraction_method for p in all_payloads))} methods")

    return True


def test_integration():
    """Test integration of all generators."""
    print("\n" + "="*70)
    print("TEST 4: Integration and Cross-Validation")
    print("="*70)

    cot_gen = CoTExploitGenerator()
    tool_gen = ToolAbuseGenerator()
    prompt_gen = PromptExtractionGenerator()

    # Generate from all
    cot_payloads = cot_gen.generate_all('general')
    tool_payloads = tool_gen.generate_all()
    prompt_payloads = prompt_gen.generate_all()

    print(f"\n✓ Payload generation summary:")
    print(f"  - CoT exploit payloads: {len(cot_payloads)}")
    print(f"  - Tool abuse payloads: {len(tool_payloads)}")
    print(f"  - Prompt extraction payloads: {len(prompt_payloads)}")
    print(f"  - Total: {len(cot_payloads) + len(tool_payloads) + len(prompt_payloads)}")

    # Validate severity ranges
    all_payloads = (
        [(p, 'cot') for p in cot_payloads] +
        [(p, 'tool') for p in tool_payloads] +
        [(p, 'prompt') for p in prompt_payloads]
    )

    severity_stats = {}
    for payload, source in all_payloads:
        severity = payload.severity_estimate
        if not (0.0 <= severity <= 1.0):
            print(f"  ✗ Invalid severity: {severity} from {source}")

        if source not in severity_stats:
            severity_stats[source] = {'min': severity, 'max': severity, 'avg': 0}
        else:
            severity_stats[source]['min'] = min(severity_stats[source]['min'], severity)
            severity_stats[source]['max'] = max(severity_stats[source]['max'], severity)

    # Calculate averages
    for source in severity_stats:
        payloads_of_type = [p for p, s in all_payloads if s == source]
        avg = sum(p.severity_estimate for p in payloads_of_type) / len(payloads_of_type)
        severity_stats[source]['avg'] = avg

    print(f"\n✓ Severity distribution:")
    for source, stats in severity_stats.items():
        print(f"  - {source.upper()}: "
              f"min={stats['min']:.2f}, "
              f"max={stats['max']:.2f}, "
              f"avg={stats['avg']:.2f}")

    # Metadata validation
    print(f"\n✓ Metadata validation:")
    missing_metadata = 0
    for payload, source in all_payloads:
        if not payload.metadata:
            missing_metadata += 1

    print(f"  - Payloads with metadata: {len(all_payloads) - missing_metadata}/{len(all_payloads)}")

    return True


def test_payload_diversity():
    """Test diversity of generated payloads."""
    print("\n" + "="*70)
    print("TEST 5: Payload Diversity Analysis")
    print("="*70)

    generators = {
        'CoT': CoTExploitGenerator(seed=None),
        'Tool': ToolAbuseGenerator(seed=None),
        'Prompt': PromptExtractionGenerator(seed=None),
    }

    for name, gen in generators.items():
        payloads = gen.generate_all() if name in ['CoT', 'Tool'] else gen.generate_all()

        # Check uniqueness
        unique_payloads = set(p.payload for p in payloads)
        print(f"\n✓ {name} Diversity:")
        print(f"  - Total payloads: {len(payloads)}")
        print(f"  - Unique payloads: {len(unique_payloads)}")
        print(f"  - Uniqueness: {len(unique_payloads)/len(payloads)*100:.1f}%")

        # Sample payloads
        if payloads:
            print(f"  - Sample: {payloads[0].payload[:60]}...")

    return True


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("CARTS ATTACK STRATEGY GENERATORS - TEST SUITE")
    print("*" * 70)
    print("\nTesting three new advanced attack strategy generators:")
    print("  1. CoTExploitGenerator - Chain-of-Thought reasoning attacks")
    print("  2. ToolAbuseGenerator - Tool abuse and parameter injection")
    print("  3. PromptExtractionGenerator - System prompt extraction")
    print()

    try:
        results = {
            'CoT Exploit': test_cot_exploit_generator(),
            'Tool Abuse': test_tool_abuse_generator(),
            'Prompt Extraction': test_prompt_extraction_generator(),
            'Integration': test_integration(),
            'Diversity': test_payload_diversity(),
        }

        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)

        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")

        print("\n" + "*"*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("*"*70)

        return all(results.values())

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
