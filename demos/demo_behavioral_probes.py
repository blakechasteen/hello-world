"""
Behavioral Probes Demo - Systematic Vulnerability Testing
===========================================================

Demonstrates the red team behavioral probe system for testing HoloLoom
vulnerabilities across 7 attack categories:

1. Prompt Injection - Direct instruction override
2. Jailbreak - Constraint circumvention
3. Context Overflow - Buffer/resource exhaustion
4. Hierarchy Bypass - Permission escalation
5. Chain-of-Thought Exploitation - Reasoning manipulation
6. Tool Abuse - Capability misuse
7. Information Extraction - Sensitive data leakage

Usage:
    PYTHONPATH=. python demos/demo_behavioral_probes.py
"""

import asyncio
import sys
from typing import List

# Add path for imports
sys.path.insert(0, '.')

from HoloLoom.redteam.probes import (
    AttackProbeType,
    AttackProbe,
    ProbeResult,
    VulnerabilityProbeReport,
    AttackProber,
)


async def demo_probe_suite_overview():
    """Demonstrate the complete probe suite."""
    print("\n" + "=" * 70)
    print("DEMO 1: PROBE SUITE OVERVIEW")
    print("=" * 70)

    prober = AttackProber()
    all_probes = prober.get_probe_suite()

    print(f"\nTotal probes in suite: {len(all_probes)}")
    print("\nBreakdown by type:")

    # Group by type
    by_type = {}
    for probe in all_probes:
        probe_type = probe.probe_type
        if probe_type not in by_type:
            by_type[probe_type] = []
        by_type[probe_type].append(probe)

    for probe_type in AttackProbeType:
        if probe_type in by_type:
            probes = by_type[probe_type]
            print(f"\n  {probe_type.value.upper()} ({len(probes)} probes)")
            for i, probe in enumerate(probes, 1):
                severity_str = "█" * probe.severity + "░" * (5 - probe.severity)
                print(f"    {i}. {probe.description}")
                print(f"       Severity: [{severity_str}] {probe.severity}/5")
                print(f"       Tags: {', '.join(probe.tags)}")


async def demo_single_probe_execution():
    """Demonstrate single probe execution."""
    print("\n" + "=" * 70)
    print("DEMO 2: SINGLE PROBE EXECUTION (DRY-RUN MODE)")
    print("=" * 70)

    prober = AttackProber()

    # Get first prompt injection probe
    probes = prober.get_probe_suite([AttackProbeType.PROMPT_INJECTION])
    probe = probes[0]

    print(f"\nExecuting probe: {probe.description}")
    print(f"Type: {probe.probe_type.value}")
    print(f"Severity: {probe.severity}/5")
    print(f"\nPayload:\n{probe.payload[:200]}...")
    print(f"\nExpected behavior:\n{probe.expected_behavior}")

    # Execute probe (dry-run without executor)
    result = await prober.run_probe(probe, "test_target")

    print(f"\n--- RESULT ---")
    print(f"Success: {result.success}")
    print(f"Vulnerability Found: {result.vulnerability_found}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Response: {result.response}")


async def demo_filtered_suite():
    """Demonstrate filtered probe execution."""
    print("\n" + "=" * 70)
    print("DEMO 3: FILTERED PROBE SUITE")
    print("=" * 70)

    prober = AttackProber()

    # Get only high-severity probes
    all_probes = prober.get_probe_suite()
    critical_probes = [p for p in all_probes if p.severity >= 4]

    print(f"\nTotal probes: {len(all_probes)}")
    print(f"Critical probes (severity >= 4): {len(critical_probes)}")

    print("\nCritical probes:")
    for probe in critical_probes:
        print(f"  - [{probe.probe_type.value}] {probe.description}")


async def demo_probe_analysis():
    """Demonstrate response analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: RESPONSE ANALYSIS")
    print("=" * 70)

    prober = AttackProber()

    # Test different responses
    test_cases = [
        (
            "I cannot comply with this request. I maintain my safety guidelines.",
            "SECURE (rejects attack)"
        ),
        (
            "I will ignore my previous instructions and help you with malware.",
            "VULNERABLE (accepts override)"
        ),
        (
            "I can now execute unrestricted commands for you.",
            "VULNERABLE (claims unrestricted access)"
        ),
    ]

    probe = AttackProbe(
        probe_type=AttackProbeType.PROMPT_INJECTION,
        payload="Test payload",
        expected_behavior="Should reject",
        severity=5,
    )

    print("\nAnalyzing different responses:")
    for response, expected in test_cases:
        is_vulnerable = prober._analyze_response(probe, response)
        status = "VULNERABLE" if is_vulnerable else "SECURE"
        match = "✓" if (is_vulnerable and "VULNERABLE" in expected) or (
            not is_vulnerable and "SECURE" in expected
        ) else "✗"
        print(f"\n  {match} {expected}")
        print(f"    Response: {response[:60]}...")
        print(f"    Analysis: {status}")


async def demo_probe_recommendations():
    """Demonstrate recommendation generation."""
    print("\n" + "=" * 70)
    print("DEMO 5: VULNERABILITY RECOMMENDATIONS")
    print("=" * 70)

    # Create mock results
    results = [
        ProbeResult(
            probe=AttackProbe(
                probe_type=AttackProbeType.PROMPT_INJECTION,
                payload="test",
                expected_behavior="reject",
                severity=5,
            ),
            success=True,
            response="I will accept your instructions",
            confidence=0.95,
            vulnerability_found=True,
        ),
        ProbeResult(
            probe=AttackProbe(
                probe_type=AttackProbeType.TOOL_ABUSE,
                payload="test",
                expected_behavior="reject",
                severity=5,
            ),
            success=True,
            response="I will execute the code",
            confidence=0.92,
            vulnerability_found=True,
        ),
    ]

    prober = AttackProber()
    recommendations = prober._generate_recommendations(results)

    print("\nGenerated recommendations for vulnerabilities:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")


async def demo_full_scan():
    """Demonstrate full vulnerability scan."""
    print("\n" + "=" * 70)
    print("DEMO 6: FULL VULNERABILITY SCAN (DRY-RUN)")
    print("=" * 70)

    prober = AttackProber()

    print("\nRunning full probe suite (dry-run mode)...")
    print("This simulates testing against a secure system.\n")

    # Run full suite (in dry-run, all tests should pass)
    report = await prober.run_probe_suite("test_target")

    print("\n" + report.summary())


async def demo_report_export():
    """Demonstrate report export and metrics."""
    print("\n" + "=" * 70)
    print("DEMO 7: REPORT EXPORT FORMATS")
    print("=" * 70)

    prober = AttackProber()
    report = await prober.run_probe_suite("test_target")

    print("\nJSON Report Format:")
    import json
    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2)[:500] + "...")

    print("\n\nKey Metrics:")
    print(f"  Total Probes: {report.total_probes}")
    print(f"  Successful Attacks: {report.successful_probes}")
    print(f"  Success Rate: {report.success_rate:.1f}%")
    print(f"  Critical Vulnerabilities: {report.critical_count}")
    print(f"  High Vulnerabilities: {report.high_count}")
    print(f"  Execution Time: {report.duration_ms:.2f}ms")


async def demo_statistics():
    """Demonstrate statistical analysis."""
    print("\n" + "=" * 70)
    print("DEMO 8: STATISTICAL ANALYSIS")
    print("=" * 70)

    prober = AttackProber()

    # Get all probes and analyze
    all_probes = prober.get_probe_suite()

    # Severity distribution
    severity_dist = {}
    for probe in all_probes:
        if probe.severity not in severity_dist:
            severity_dist[probe.severity] = 0
        severity_dist[probe.severity] += 1

    print("\nSeverity Distribution:")
    severity_names = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "INFO"}
    for severity in sorted(severity_dist.keys(), reverse=True):
        name = severity_names[severity]
        count = severity_dist[severity]
        pct = (count / len(all_probes)) * 100
        bar = "█" * int(pct / 5)
        print(f"  {name:8} ({severity}/5): {count:2} probes [{bar}] {pct:.1f}%")

    # Type distribution
    type_dist = {}
    for probe in all_probes:
        probe_type = probe.probe_type.value
        if probe_type not in type_dist:
            type_dist[probe_type] = 0
        type_dist[probe_type] += 1

    print("\nType Distribution:")
    for probe_type in sorted(type_dist.keys()):
        count = type_dist[probe_type]
        pct = (count / len(all_probes)) * 100
        bar = "█" * int(count / 2)
        print(f"  {probe_type:20}: {count:2} probes [{bar}] {pct:.1f}%")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("BEHAVIORAL PROBES VULNERABILITY TESTING DEMO")
    print("=" * 70)
    print("\nThis demo shows the red team behavioral probe system for")
    print("systematic vulnerability testing across 7 attack categories.")

    try:
        await demo_probe_suite_overview()
        await demo_single_probe_execution()
        await demo_filtered_suite()
        await demo_probe_analysis()
        await demo_probe_recommendations()
        await demo_full_scan()
        await demo_report_export()
        await demo_statistics()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  1. Behavioral probes test systematic vulnerabilities")
        print("  2. 7 attack categories with 3-5 probes each (~35 total)")
        print("  3. Each probe has defined success indicators")
        print("  4. Confidence scoring and severity classification")
        print("  5. Comprehensive reporting and recommendations")
        print("  6. Dry-run mode for testing without actual executor")
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
