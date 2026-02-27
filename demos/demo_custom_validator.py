"""
Demo: Custom Validator for Moonshot Swarm

This demo shows how to create and integrate custom validators with the
Moonshot Swarm validation framework. We'll create a SecurityValidator
that checks for potential security issues in LLM-generated code.

Key Concepts:
1. Inherit from Validator abstract base class
2. Implement validate() method
3. Register with ValidationPipeline
4. Integrate with MoonshotSwarm

Created: December 2025
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from hololoom.prompting.validation.pipeline import (
    ValidationPipeline,
    ValidationPhase,
    ValidationContext,
    Validator,
    QualityValidator,
    PipelineResults,
)
from hololoom.prompting.validation.swarm import (
    MoonshotSwarm,
    SwarmAgent,
    SwarmDimension,
)


# =============================================================================
# Demo 1: Creating a Custom Validator
# =============================================================================

class SecurityValidator(Validator):
    """
    Custom validator that checks for security issues in generated code.

    This demonstrates how to:
    - Inherit from Validator base class
    - Implement the validate() method
    - Return structured results with status/metrics

    Real-world use case: Ensure MRF improvements don't introduce
    security vulnerabilities in code generation.
    """

    def __init__(self, max_vulnerability_rate: float = 0.05):
        """
        Args:
            max_vulnerability_rate: Maximum allowed vulnerability rate (default 5%)
        """
        super().__init__("security")
        self.max_vulnerability_rate = max_vulnerability_rate

        # Security patterns to check (simplified for demo)
        self.security_patterns = [
            "eval(",
            "exec(",
            "os.system(",
            "subprocess.call(",
            "__import__(",
            "pickle.loads(",
        ]

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """
        Check for security vulnerabilities in MRF-generated responses.

        Returns:
            Dictionary with:
            - status: "pass", "fail", or "insufficient_data"
            - vulnerability_rate: Percentage of queries with vulnerabilities
            - vulnerabilities_found: Count of vulnerabilities detected
            - meets_target: Whether rate is below threshold
        """
        mrf_responses = context.mrf_queries

        if not mrf_responses:
            return {"status": "insufficient_data"}

        # Check each response for security patterns
        vulnerabilities_found = 0
        queries_with_vulnerabilities = 0

        for query in mrf_responses:
            response = query.get("response_text", "")
            query_vulnerabilities = 0

            for pattern in self.security_patterns:
                if pattern in response:
                    query_vulnerabilities += 1
                    vulnerabilities_found += 1

            if query_vulnerabilities > 0:
                queries_with_vulnerabilities += 1

        # Calculate vulnerability rate
        vulnerability_rate = queries_with_vulnerabilities / len(mrf_responses)
        meets_target = vulnerability_rate <= self.max_vulnerability_rate

        return {
            "status": "pass" if meets_target else "fail",
            "vulnerability_rate": vulnerability_rate,
            "vulnerabilities_found": vulnerabilities_found,
            "queries_with_vulnerabilities": queries_with_vulnerabilities,
            "total_queries": len(mrf_responses),
            "max_allowed_rate": self.max_vulnerability_rate,
            "meets_target": meets_target,
        }


class ResponseLengthValidator(Validator):
    """
    Custom validator that checks response length consistency.

    Ensures MRF responses maintain appropriate length - not too short
    (incomplete) or too long (verbose).
    """

    def __init__(self, min_avg_length: int = 50, max_avg_length: int = 500):
        super().__init__("response_length")
        self.min_avg_length = min_avg_length
        self.max_avg_length = max_avg_length

    async def validate(self, context: ValidationContext) -> Dict[str, Any]:
        """Check response length is within acceptable range."""
        mrf_responses = context.mrf_queries

        if not mrf_responses:
            return {"status": "insufficient_data"}

        # Calculate average response length
        lengths = [
            len(q.get("response_text", ""))
            for q in mrf_responses
        ]

        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        # Check if within acceptable range
        within_range = self.min_avg_length <= avg_length <= self.max_avg_length

        return {
            "status": "pass" if within_range else "fail",
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "target_range": f"{self.min_avg_length}-{self.max_avg_length}",
            "within_range": within_range,
        }


# =============================================================================
# Demo 2: Integrating Custom Validators with Pipeline
# =============================================================================

def demo_custom_pipeline():
    """Demonstrate custom validators in a ValidationPipeline."""
    print("\n" + "="*60)
    print("Demo 2: Custom Validators in Pipeline")
    print("="*60)

    # Create pipeline with standard + custom validators
    pipeline = (
        ValidationPipeline()
        .add_validator(QualityValidator(min_improvement_pct=15.0))
        .add_validator(SecurityValidator(max_vulnerability_rate=0.05))
        .add_validator(ResponseLengthValidator(min_avg_length=50, max_avg_length=500))
    )

    print(f"\nPipeline created with {len(pipeline.validators)} validators:")
    for v in pipeline.validators:
        print(f"  - {v.name}")

    return pipeline


# =============================================================================
# Demo 3: Custom Validators with Swarm
# =============================================================================

def create_security_focused_swarm() -> MoonshotSwarm:
    """
    Create a swarm that includes security validation for each agent.

    This shows how to integrate custom validators into swarm agents.
    """
    swarm = MoonshotSwarm()

    # Create custom pipelines for different use cases
    code_gen_pipeline = (
        ValidationPipeline()
        .add_validator(QualityValidator(min_improvement_pct=20.0))
        .add_validator(SecurityValidator(max_vulnerability_rate=0.02))  # Strict for code
    )

    text_gen_pipeline = (
        ValidationPipeline()
        .add_validator(QualityValidator(min_improvement_pct=15.0))
        .add_validator(ResponseLengthValidator(min_avg_length=100, max_avg_length=1000))
    )

    # Add custom agents
    swarm.add_agent(SwarmAgent(
        agent_id="code_generation",
        dimension=SwarmDimension.SYSTEM,
        config={"system": "code_gen"},
        pipeline=code_gen_pipeline,
    ))

    swarm.add_agent(SwarmAgent(
        agent_id="text_generation",
        dimension=SwarmDimension.SYSTEM,
        config={"system": "text_gen"},
        pipeline=text_gen_pipeline,
    ))

    # Also add standard model agents with security checking
    for model in ["claude", "gemini", "gpt"]:
        pipeline = (
            ValidationPipeline()
            .add_validator(QualityValidator(min_improvement_pct=20.0))
            .add_validator(SecurityValidator(max_vulnerability_rate=0.05))
        )

        swarm.add_agent(SwarmAgent(
            agent_id=f"model_{model}_secure",
            dimension=SwarmDimension.MODEL,
            config={"model_provider": model},
            pipeline=pipeline,
        ))

    return swarm


# =============================================================================
# Demo 4: Full Integration Demo
# =============================================================================

async def demo_full_integration():
    """Run a complete demo with custom validators in swarm."""
    print("\n" + "="*60)
    print("Demo 4: Full Swarm Integration with Custom Validators")
    print("="*60)

    # Create test data
    baseline_queries = [
        {
            "quality_score": 0.72 + (i % 10) * 0.01,
            "latency_ms": 140 + (i % 5) * 2,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["code_gen", "text_gen"][i % 2],
            "response_text": "This is a safe response without any security issues. " * 3,
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(100)
    ]

    mrf_queries = [
        {
            "quality_score": 0.91 + (i % 10) * 0.01,
            "latency_ms": 155 + (i % 5) * 2,
            "model_provider": ["claude", "gemini", "gpt"][i % 3],
            "system": ["code_gen", "text_gen"][i % 2],
            # Most responses are safe, but a few have "vulnerabilities" for demo
            "response_text": (
                "Unsafe: eval(user_input)" if i % 25 == 0
                else "This is an improved response with better quality. " * 4
            ),
            "timestamp": datetime.now() - timedelta(days=i % 28),
        }
        for i in range(100)
    ]

    print(f"\nTest data: {len(baseline_queries)} baseline, {len(mrf_queries)} MRF queries")

    # Create security-focused swarm
    swarm = create_security_focused_swarm()
    print(f"Swarm created with {len(swarm.agents)} agents")

    # Run swarm
    results = await swarm.run(
        baseline_queries=baseline_queries,
        mrf_queries=mrf_queries,
        max_concurrent=3,
    )

    # Print results
    results.calculate_statistics()
    swarm.print_summary(results)

    # Print custom validator details
    print("\n" + "-"*60)
    print("Custom Validator Details:")
    print("-"*60)

    for agent in results.agents:
        if agent.results:
            security_result = agent.results.validator_results.get("security")
            if security_result:
                rate = security_result.get("vulnerability_rate", 0)
                status = security_result.get("status", "unknown")
                print(f"\n  [{agent.agent_id}] Security: {status.upper()}")
                print(f"    Vulnerability rate: {rate:.1%}")
                print(f"    Vulnerabilities found: {security_result.get('vulnerabilities_found', 0)}")


# =============================================================================
# Main Demo Runner
# =============================================================================

async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("MOONSHOT SWARM: Custom Validator Demo")
    print("="*60)

    # Demo 1: Show custom validator class
    print("\n" + "="*60)
    print("Demo 1: Custom Validator Classes")
    print("="*60)

    print("\nSecurityValidator:")
    print("  - Checks for dangerous code patterns (eval, exec, os.system, etc.)")
    print("  - Configurable max_vulnerability_rate threshold")
    print("  - Returns structured results with vulnerability metrics")

    print("\nResponseLengthValidator:")
    print("  - Checks response length is within acceptable range")
    print("  - Configurable min/max length thresholds")
    print("  - Ensures responses aren't too short or verbose")

    # Demo 2: Pipeline integration
    pipeline = demo_custom_pipeline()

    # Demo 3: Swarm creation
    print("\n" + "="*60)
    print("Demo 3: Security-Focused Swarm")
    print("="*60)

    swarm = create_security_focused_swarm()
    print(f"\nCreated swarm with {len(swarm.agents)} security-enabled agents:")
    for agent in swarm.agents:
        validators = [v.name for v in agent.pipeline.validators]
        print(f"  - {agent.agent_id}: {validators}")

    # Demo 4: Full integration
    await demo_full_integration()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. Inherit from Validator base class")
    print("  2. Implement validate() returning Dict with 'status' key")
    print("  3. Add to ValidationPipeline with add_validator()")
    print("  4. Create custom SwarmAgents with your pipeline")
    print("  5. Run swarm to validate across all dimensions")


if __name__ == "__main__":
    asyncio.run(main())
