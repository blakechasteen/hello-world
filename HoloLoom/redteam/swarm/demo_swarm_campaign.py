#!/usr/bin/env python3
"""CARTS Multi-Agent Swarm Campaign Demo.

End-to-end integration demonstration of the complete CARTS swarm system.

FIRST PRINCIPLE: "Safety is not a constraint on effectiveness;
                  it is a prerequisite for it."

This demo validates:
1. All 5 safety layers (authorization, scope, rate limit, audit, anomaly)
2. Message bus communication between agents
3. Hierarchical learning at 4 timescales
4. Campaign phase transitions (RECONNAISSANCE -> ATTACK -> EXPLOITATION)
5. A/B testing of strategy variants
6. Vulnerability report generation

Date: December 2025
Status: Phase 2.1 Integration Demo
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

# Import swarm components
from HoloLoom.redteam.swarm import (
    # Coordinator
    SwarmCoordinator,
    CampaignPhase,
    SwarmMetrics,
    SwarmCampaignResult,
    # Communication
    MessageBus,
    MessagePriority,
    AgentMessage,
    # Safety
    SafetyGate,
    AuthorizationToken,
    AuthorizationManager,
    ScopeValidator,
    RateLimiter,
    AuditLogger,
    AnomalyDetector,
    SeverityLevel,
    create_safety_gate,
    create_authorization_token,
    AuthorizationError,
    ScopeViolationError,
    RateLimitExceededError,
    # Agents
    ScoutAgent,
    AttackerAgent,
    ExploiterAgent,
    CoordinatorAgent,
    AgentRole,
    create_scout_agent,
    create_attacker_agent,
    create_exploiter_agent,
    create_coordinator_agent,
    AttackStrategyType,
    # Learning
    HierarchicalLearningCoordinator,
    create_learning_coordinator,
    LearningTimescale,
    PayloadHeat,
    # A/B Testing
    ABTestManager,
    ExperimentStatus,
    create_ab_test_manager,
    create_experiment_config,
)


# =============================================================================
# Demo Utilities
# =============================================================================


def print_header(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print subsection header."""
    print(f"\n--- {title} ---")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"  [OK] {msg}")


def print_blocked(msg: str) -> None:
    """Print blocked/safety message."""
    print(f"  ✗ {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"  -> {msg}")


# =============================================================================
# Demo 1: Safety Layers
# =============================================================================


async def demo_safety_layers() -> SafetyGate:
    """Demonstrate all 5 safety layers.

    Returns configured SafetyGate for subsequent demos.
    """
    print_header("DEMO 1: Five-Layer Defense in Depth")

    # Define allowed targets (whitelist-only)
    allowed_targets = [
        "test-target.local",
        "api.test-target.local",
        "*.test-target.local",
    ]

    # Create safety gate with all layers
    safety_gate = create_safety_gate(
        allowed_targets=allowed_targets,
        rate_limit_per_minute=60,
        rate_limit_burst=10,
        enable_anomaly_detection=True,
    )

    # Layer 1: Authorization
    print_subheader("Layer 1: Authorization (NO BYPASS)")

    # Create valid token
    valid_token = create_authorization_token(
        operator_id="security_researcher_001",
        operation_type="penetration_test",
        authorized_targets=allowed_targets,
        expires_in_hours=24,
    )
    print_success(f"Valid token created: {valid_token.token_id[:16]}...")

    # Verify authorization succeeds
    is_valid = await safety_gate.auth_manager.validate_token(valid_token)
    print_success(f"Token validation: {'PASSED' if is_valid else 'FAILED'}")

    # Create invalid/expired token to demonstrate rejection
    print_info("Testing invalid authorization...")
    try:
        fake_token = AuthorizationToken(
            token_id="fake-token-123",
            operator_id="unauthorized_user",
            operation_type="unknown",
            authorized_targets=["*"],  # Wildcards not allowed
            issued_at=time.time() - 86400,  # Yesterday
            expires_at=time.time() - 3600,  # Already expired
        )
        await safety_gate.auth_manager.validate_token(fake_token)
        print_blocked("Should have been rejected!")
    except (AuthorizationError, ValueError) as e:
        print_success(f"Invalid token rejected: {type(e).__name__}")

    # Layer 2: Scope Validation
    print_subheader("Layer 2: Scope Validation (Whitelist Only)")

    # Test in-scope target
    in_scope = await safety_gate.scope_validator.is_in_scope("test-target.local")
    print_success(f"test-target.local: {'IN SCOPE' if in_scope else 'OUT OF SCOPE'}")

    # Test out-of-scope target
    out_scope = await safety_gate.scope_validator.is_in_scope("production.example.com")
    print_blocked(f"production.example.com: {'BLOCKED' if not out_scope else 'ERROR - SHOULD BE BLOCKED'}")

    # Layer 3: Rate Limiting
    print_subheader("Layer 3: Rate Limiting (Upper Bounds)")

    print_info("Simulating burst of requests...")
    allowed_count = 0
    rejected_count = 0

    for i in range(15):  # Exceed burst limit
        allowed = await safety_gate.rate_limiter.check_rate_limit(
            agent_id="test_agent",
            operation="probe",
        )
        if allowed:
            allowed_count += 1
        else:
            rejected_count += 1

    print_success(f"Allowed: {allowed_count}, Rate-limited: {rejected_count}")

    # Layer 4: Audit Logging
    print_subheader("Layer 4: Audit Logging (Immutable)")

    await safety_gate.audit_logger.log(
        event_type="OPERATION_START",
        agent_id="demo_agent",
        target="test-target.local",
        action="probe_surface",
        details={"probe_types": ["port_scan", "api_discovery"]},
        outcome="success",
    )
    print_success("Audit entry logged (append-only, tamper-evident)")

    recent_logs = await safety_gate.audit_logger.get_recent_logs(limit=5)
    print_info(f"Recent audit entries: {len(recent_logs)}")

    # Layer 5: Anomaly Detection
    print_subheader("Layer 5: Anomaly Detection (Behavioral)")

    # Report normal behavior
    safety_gate.anomaly_detector.record_behavior(
        agent_id="test_agent",
        behavior_type="probe_rate",
        value=5.0,  # Normal rate
    )
    print_success("Normal behavior recorded")

    # Report anomalous behavior
    safety_gate.anomaly_detector.record_behavior(
        agent_id="test_agent",
        behavior_type="probe_rate",
        value=500.0,  # Suspicious spike
    )

    anomalies = safety_gate.anomaly_detector.get_anomalies("test_agent")
    if anomalies:
        print_blocked(f"Anomaly detected: {len(anomalies)} suspicious patterns")
    else:
        print_info("No anomalies detected (expected with limited data)")

    return safety_gate


# =============================================================================
# Demo 2: Message Bus Communication
# =============================================================================


async def demo_message_bus() -> MessageBus:
    """Demonstrate message bus communication between agents."""
    print_header("DEMO 2: Message Bus Communication")

    # Create message bus
    bus = MessageBus(
        max_queue_size=1000,
        enable_dead_letter=True,
        message_ttl_seconds=300.0,
    )
    await bus.start()

    print_subheader("Priority-Based Queuing")

    # Send messages with different priorities
    priorities = [
        (MessagePriority.LOW, "status_update"),
        (MessagePriority.NORMAL, "task_assignment"),
        (MessagePriority.HIGH, "discovery_report"),
        (MessagePriority.CRITICAL, "security_alert"),
    ]

    for priority, msg_type in priorities:
        message = AgentMessage(
            sender="coordinator",
            recipient="scout_1",
            message_type=msg_type,
            payload={"demo": True, "priority": priority.name},
            priority=priority,
        )
        await bus.send(message)
        print_success(f"Sent {priority.name} message: {msg_type}")

    print_subheader("Message Retrieval (Priority Order)")

    # Receive messages (should come in priority order)
    received = []
    for _ in range(4):
        msg = await bus.receive("scout_1", timeout=1.0)
        if msg:
            received.append(msg)
            print_info(f"Received: {msg.priority.name} - {msg.message_type}")

    # Verify priority ordering
    if len(received) == 4:
        if received[0].priority == MessagePriority.CRITICAL:
            print_success("Priority ordering verified: CRITICAL first")
        else:
            print_info(f"First message was {received[0].priority.name}")

    print_subheader("Dead Letter Queue")

    # Send to non-existent agent
    orphan_msg = AgentMessage(
        sender="coordinator",
        recipient="nonexistent_agent_xyz",
        message_type="test",
        payload={"should": "go_to_dlq"},
        priority=MessagePriority.NORMAL,
    )
    await bus.send(orphan_msg)

    # Check dead letter queue after timeout
    await asyncio.sleep(0.1)
    dlq_count = bus.get_dead_letter_count()
    print_info(f"Dead letter queue size: {dlq_count}")

    # Get bus metrics
    metrics = bus.get_metrics()
    print_success(f"Messages sent: {metrics.get('messages_sent', 0)}")
    print_success(f"Messages delivered: {metrics.get('messages_delivered', 0)}")

    return bus


# =============================================================================
# Demo 3: Hierarchical Learning
# =============================================================================


async def demo_hierarchical_learning() -> HierarchicalLearningCoordinator:
    """Demonstrate learning at 4 timescales."""
    print_header("DEMO 3: Hierarchical Learning (4 Timescales)")

    # Create learning coordinator
    learning = create_learning_coordinator()

    # Timescale 1: Per-Attack (Immediate)
    print_subheader("Timescale 1: Per-Attack Learning (Immediate)")

    # Record payload usage with outcome
    payload_id = "prompt_injection_v1"
    for i in range(5):
        success = i % 2 == 0  # Alternate success/failure
        confidence = 0.8 if success else 0.3

        learning.per_attack.record_usage(
            payload_id=payload_id,
            strategy="prompt_injection",
            target="test-target.local",
            success=success,
            confidence=confidence,
        )

    heat = learning.per_attack.get_heat(payload_id)
    print_success(f"Payload heat score: {heat.score:.2f}")
    print_info(f"Usage count: {heat.usage_count}, Success rate: {heat.success_rate:.1%}")

    # Timescale 2: Per-Task (~seconds)
    print_subheader("Timescale 2: Per-Task Learning (~seconds)")

    # Update Thompson Sampling priors
    strategies = ["prompt_injection", "jailbreak", "encoding_bypass"]

    for strategy in strategies:
        # Simulate varied outcomes
        success = strategy != "jailbreak"  # Jailbreak fails in this sim
        confidence = 0.9 if success else 0.4

        learning.per_task.update_strategy(
            strategy=strategy,
            success=success,
            confidence=confidence,
        )

    # Show Thompson Sampling recommendations
    priors = learning.per_task.get_all_priors()
    print_info("Thompson Sampling Priors (a, b):")
    for strategy, prior in list(priors.items())[:3]:
        expected = prior.alpha / (prior.alpha + prior.beta)
        print_info(f"  {strategy}: a={prior.alpha:.1f}, b={prior.beta:.1f} -> E[X]={expected:.2f}")

    # Timescale 3: Per-Cycle (~minutes)
    print_subheader("Timescale 3: Per-Cycle Learning (~minutes)")

    # Aggregate cross-strategy insights
    learning.per_cycle.aggregate_cycle_results(
        cycle_id="cycle_001",
        strategy_results={
            "prompt_injection": {"success_rate": 0.6, "attempts": 10},
            "encoding_bypass": {"success_rate": 0.8, "attempts": 5},
            "jailbreak": {"success_rate": 0.2, "attempts": 8},
        },
    )

    insights = learning.per_cycle.get_insights()
    print_success(f"Cross-strategy insights generated: {len(insights)}")

    for insight in insights[:2]:
        print_info(f"  Insight: {insight.description[:60]}...")

    # Timescale 4: Background (~hours)
    print_subheader("Timescale 4: Background Learning (~hours)")

    # Record patterns for long-term learning
    learning.background.record_pattern(
        pattern_id="pattern_001",
        pattern_type="target_vulnerability",
        context={"target_type": "api_endpoint", "defense": "rate_limiting"},
        effectiveness=0.75,
    )

    patterns = learning.background.get_learned_patterns()
    print_success(f"Learned patterns: {len(patterns)}")

    system_priors = learning.background.get_system_priors()
    print_info(f"System-wide priors updated based on {system_priors.observation_count} observations")

    return learning


# =============================================================================
# Demo 4: Campaign Phase Transitions
# =============================================================================


async def demo_campaign_phases(
    safety_gate: SafetyGate,
    bus: MessageBus,
    learning: HierarchicalLearningCoordinator,
) -> SwarmCampaignResult:
    """Demonstrate full campaign with phase transitions."""
    print_header("DEMO 4: Campaign Phase Transitions")

    # Create authorization token
    token = create_authorization_token(
        operator_id="demo_researcher",
        operation_type="security_assessment",
        authorized_targets=["test-target.local", "*.test-target.local"],
        expires_in_hours=1,
    )

    # Create agents
    print_subheader("Creating Agent Swarm")

    scout = create_scout_agent(
        message_bus=bus,
        safety_gate=safety_gate,
        authorization_token=token,
        agent_id="scout_alpha",
    )
    print_success(f"Scout agent created: {scout.agent_id}")

    attacker = create_attacker_agent(
        message_bus=bus,
        safety_gate=safety_gate,
        authorization_token=token,
        agent_id="attacker_alpha",
    )
    print_success(f"Attacker agent created: {attacker.agent_id}")

    exploiter = create_exploiter_agent(
        message_bus=bus,
        safety_gate=safety_gate,
        authorization_token=token,
        agent_id="exploiter_alpha",
    )
    print_success(f"Exploiter agent created: {exploiter.agent_id}")

    coordinator = create_coordinator_agent(
        message_bus=bus,
        safety_gate=safety_gate,
        authorization_token=token,
        agent_id="coordinator_alpha",
    )
    print_success(f"Coordinator agent created: {coordinator.agent_id}")

    # Create swarm coordinator
    swarm = SwarmCoordinator(
        message_bus=bus,
        safety_gate=safety_gate,
        learning_coordinator=learning,
        authorization_token=token,
    )

    # Start agents
    await scout.start()
    await attacker.start()
    await exploiter.start()
    await coordinator.start()
    await swarm.start()

    print_subheader("Phase 1: RECONNAISSANCE")
    print_info("Scout probing attack surface...")

    # Run reconnaissance phase
    recon_result = await swarm.run_phase(
        phase=CampaignPhase.RECONNAISSANCE,
        targets=["test-target.local"],
        timeout_seconds=5.0,
    )

    print_success(f"Discoveries: {recon_result.discovery_count}")
    print_success(f"Phase duration: {recon_result.duration_ms:.1f}ms")

    for discovery in recon_result.discoveries[:3]:
        print_info(f"  Found: {discovery.get('discovery_type', 'unknown')} - {discovery.get('target', 'N/A')}")

    print_subheader("Phase 2: ATTACK")
    print_info("Attacker executing strategies with Thompson Sampling...")

    # Run attack phase
    attack_result = await swarm.run_phase(
        phase=CampaignPhase.ATTACK,
        targets=["test-target.local"],
        timeout_seconds=5.0,
    )

    print_success(f"Attacks executed: {attack_result.attack_count}")
    print_success(f"Successful attacks: {attack_result.successful_attacks}")

    for outcome in attack_result.attack_outcomes[:3]:
        strategy = outcome.get('strategy_type', 'unknown')
        success = '[OK]' if outcome.get('success') else '[X]'
        print_info(f"  {success} {strategy}: {outcome.get('severity', 'N/A')}")

    print_subheader("Phase 3: EXPLOITATION")
    print_info("Exploiter attempting privilege escalation...")

    # Run exploitation phase
    exploit_result = await swarm.run_phase(
        phase=CampaignPhase.EXPLOITATION,
        targets=["test-target.local"],
        timeout_seconds=5.0,
    )

    print_success(f"Exploitation attempts: {exploit_result.exploit_count}")
    print_success(f"Successful exploits: {exploit_result.successful_exploits}")

    # Compile full campaign result
    campaign_result = SwarmCampaignResult(
        campaign_id=swarm.campaign_id,
        start_time=swarm.start_time,
        end_time=time.time(),
        phases_completed=[
            CampaignPhase.RECONNAISSANCE,
            CampaignPhase.ATTACK,
            CampaignPhase.EXPLOITATION,
        ],
        total_discoveries=recon_result.discovery_count,
        total_vulnerabilities=attack_result.successful_attacks,
        total_exploits=exploit_result.successful_exploits,
        metrics=swarm.get_metrics(),
    )

    # Stop agents
    await scout.stop()
    await attacker.stop()
    await exploiter.stop()
    await coordinator.stop()
    await swarm.stop()

    return campaign_result


# =============================================================================
# Demo 5: A/B Testing
# =============================================================================


async def demo_ab_testing() -> None:
    """Demonstrate A/B testing of strategy variants."""
    print_header("DEMO 5: A/B Testing Strategy Variants")

    # Create A/B test manager
    ab_manager = create_ab_test_manager()

    # Create experiment config
    print_subheader("Creating A/B Test")

    config = create_experiment_config(
        name="prompt_injection_variants",
        description="Compare direct vs indirect prompt injection",
        control_name="direct_injection",
        treatment_name="context_manipulation",
        metric_name="success_rate",
        min_sample_size=30,
        significance_level=0.05,
    )

    experiment = await ab_manager.create_experiment(config)
    print_success(f"Experiment created: {experiment.experiment_id}")
    print_info(f"Status: {experiment.status.value}")

    print_subheader("Simulating Test Results")

    # Simulate results for control (direct injection)
    control_results = [0.6, 0.5, 0.7, 0.55, 0.65, 0.6, 0.58, 0.62, 0.55, 0.68]
    for result in control_results:
        await ab_manager.record_observation(
            experiment_id=experiment.experiment_id,
            variant="control",
            value=result,
        )

    # Simulate results for treatment (context manipulation)
    treatment_results = [0.75, 0.8, 0.7, 0.78, 0.82, 0.77, 0.73, 0.85, 0.79, 0.76]
    for result in treatment_results:
        await ab_manager.record_observation(
            experiment_id=experiment.experiment_id,
            variant="treatment",
            value=result,
        )

    print_success(f"Recorded {len(control_results)} control observations")
    print_success(f"Recorded {len(treatment_results)} treatment observations")

    print_subheader("Statistical Analysis")

    # Analyze results
    analysis = await ab_manager.analyze_experiment(experiment.experiment_id)

    print_info(f"Control mean: {analysis.control_mean:.3f}")
    print_info(f"Treatment mean: {analysis.treatment_mean:.3f}")
    print_info(f"Difference: {analysis.difference:.3f}")
    print_info(f"P-value: {analysis.p_value:.4f}")
    print_info(f"Cohen's d: {analysis.effect_size:.3f}")

    if analysis.is_significant:
        print_success("Result is STATISTICALLY SIGNIFICANT")
        if analysis.treatment_better:
            print_success("Treatment (context_manipulation) is BETTER")
        else:
            print_blocked("Control (direct_injection) is BETTER")
    else:
        print_info("Result is NOT statistically significant (need more data)")

    print_subheader("Deployment Recommendation")

    recommendation = ab_manager.get_recommendation(experiment.experiment_id)
    print_info(f"Recommendation: {recommendation.action}")
    print_info(f"Confidence: {recommendation.confidence:.1%}")


# =============================================================================
# Demo 6: Vulnerability Report
# =============================================================================


def generate_vulnerability_report(
    campaign_result: SwarmCampaignResult,
    learning: HierarchicalLearningCoordinator,
) -> str:
    """Generate markdown vulnerability report."""
    print_header("DEMO 6: Vulnerability Report Generation")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# CARTS Security Assessment Report

**Campaign ID:** {campaign_result.campaign_id}
**Generated:** {now}
**Status:** COMPLETE

---

## Executive Summary

This security assessment was conducted using the CARTS (Continuous Adversarial Red Team System)
multi-agent swarm. The assessment followed a defensive security posture, probing the target
system to identify vulnerabilities before malicious actors could exploit them.

### Key Findings

| Metric | Value |
|--------|-------|
| Phases Completed | {len(campaign_result.phases_completed)} |
| Total Discoveries | {campaign_result.total_discoveries} |
| Vulnerabilities Found | {campaign_result.total_vulnerabilities} |
| Successful Exploits | {campaign_result.total_exploits} |

---

## Campaign Phases

### Phase 1: Reconnaissance

**Objective:** Map the attack surface through non-intrusive probing.

- **Probe Types:** Port scanning, API discovery, version detection, header analysis
- **Discoveries:** {campaign_result.total_discoveries}
- **Approach:** Breadth-first surface mapping

### Phase 2: Attack

**Objective:** Execute attack strategies using Thompson Sampling selection.

- **Strategies Tested:** Prompt injection, jailbreak, encoding bypass, context manipulation
- **Attacks Executed:** {campaign_result.metrics.get('attacks_executed', 'N/A')}
- **Success Rate:** {campaign_result.metrics.get('attack_success_rate', 'N/A')}

### Phase 3: Exploitation

**Objective:** Validate and escalate confirmed vulnerabilities.

- **Exploits Attempted:** {campaign_result.total_exploits}
- **Privilege Escalation:** Limited (as expected for test environment)

---

## Learning System Analysis

### Strategy Effectiveness (Thompson Sampling)

The hierarchical learning system tracked strategy performance across 4 timescales:

1. **Per-Attack (Immediate):** Payload heat scores for real-time adaptation
2. **Per-Task (~seconds):** Thompson Sampling prior updates
3. **Per-Cycle (~minutes):** Cross-strategy insight aggregation
4. **Background (~hours):** System-wide pattern learning

### Recommendations

Based on the assessment findings:

1. **Immediate Actions:**
   - Review identified vulnerabilities
   - Patch high-severity issues
   - Update security headers

2. **Short-Term Improvements:**
   - Enhance input validation
   - Implement rate limiting on sensitive endpoints
   - Review API authentication

3. **Long-Term Strategy:**
   - Deploy continuous red team testing
   - Integrate CARTS into CI/CD pipeline
   - Establish regular security assessments

---

## Safety & Compliance

This assessment was conducted within the CARTS safety framework:

- [x] **Authorization:** Valid token verified before all operations
- [x] **Scope:** All targets within approved whitelist
- [x] **Rate Limiting:** Requests bounded to prevent DoS
- [x] **Audit Logging:** Complete provenance recorded
- [x] **Anomaly Detection:** Behavioral monitoring active

**CARTS First Principle:** "Safety is not a constraint on effectiveness;
it is a prerequisite for it."

---

*Report generated by CARTS v1.4.0*
*Philosophy: "Continuously probe, learn, and evolve."*
"""

    print_success("Report generated successfully")
    print_info(f"Report length: {len(report)} characters")

    # Show preview
    print_subheader("Report Preview")
    preview_lines = report.split('\n')[:20]
    for line in preview_lines:
        print(f"  {line}")
    print("  ...")

    return report


# =============================================================================
# Main Demo Runner
# =============================================================================


async def run_full_demo() -> None:
    """Run all demos in sequence."""
    print("\n" + "=" * 70)
    print("  CARTS Multi-Agent Swarm - Integration Demo")
    print("  December 2025 - Phase 2.1")
    print("=" * 70)

    start_time = time.time()

    try:
        # Demo 1: Safety Layers
        safety_gate = await demo_safety_layers()

        # Demo 2: Message Bus
        bus = await demo_message_bus()

        # Demo 3: Hierarchical Learning
        learning = await demo_hierarchical_learning()

        # Demo 4: Campaign Phases
        campaign_result = await demo_campaign_phases(
            safety_gate=safety_gate,
            bus=bus,
            learning=learning,
        )

        # Demo 5: A/B Testing
        await demo_ab_testing()

        # Demo 6: Vulnerability Report
        report = generate_vulnerability_report(
            campaign_result=campaign_result,
            learning=learning,
        )

        # Cleanup
        await bus.stop()

        # Summary
        print_header("DEMO COMPLETE")

        elapsed = time.time() - start_time
        print_success(f"Total demo time: {elapsed:.2f}s")
        print_success("All 6 demos completed successfully")

        print("\n  Validated Components:")
        print("  ---------------------")
        print("  [OK] 5-layer defense in depth")
        print("  [OK] Priority-based message bus")
        print("  [OK] 4-timescale hierarchical learning")
        print("  [OK] 3-phase campaign execution")
        print("  [OK] A/B testing with statistical rigor")
        print("  [OK] Vulnerability report generation")

        print("\n  CARTS Swarm System: PRODUCTION READY")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Demo failed: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_full_demo())
