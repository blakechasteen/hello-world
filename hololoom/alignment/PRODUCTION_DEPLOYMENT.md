# Alignment Framework Production Deployment Guide

**Version**: 1.0.0
**Date**: November 1, 2025
**Status**: Ready for Production Deployment

---

## Pre-Deployment Checklist

### 1. Environment Requirements

- [ ] Python 3.10+ installed
- [ ] HoloLoom v1.0+ installed
- [ ] Sufficient disk space for audit logs (estimate: 1MB per 10,000 queries)
- [ ] Write permissions for log directory
- [ ] Optional: Monitoring infrastructure (Prometheus, Grafana, etc.)

### 2. Configuration Validation

```python
from hololoom.alignment import (
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
)

# Verify all modules can be imported
print("✅ All alignment modules imported successfully")
```

### 3. Performance Baseline

Run benchmarks to verify performance on production hardware:

```bash
python hololoom/alignment/tests/run_benchmarks.py
```

**Expected Results**:
- Total overhead: <3ms (ideally <0.5ms)
- SafetyGuardrails: <0.5ms
- DeceptionDetector: <1.0ms
- InstrumentalGuard: <0.3ms
- AuditTrail: <0.2ms median (P99 may spike due to I/O)

---

## Deployment Options

### Option 1: Full Alignment (Recommended)

**Use Case**: Production systems requiring comprehensive safety

**Components**:
- SafetyGuardrails ✅
- DeceptionDetector ✅
- InstrumentalGuard ✅
- AuditTrail ✅

**Performance**: ~0.1ms overhead

### Option 2: Safety-Only (Lightweight)

**Use Case**: Low-latency systems with basic safety needs

**Components**:
- SafetyGuardrails ✅
- AuditTrail ✅ (buffered)

**Performance**: ~0.05ms overhead

### Option 3: Research/Development (Maximum Observability)

**Use Case**: Research environments, debugging, analysis

**Components**:
- All modules ✅
- Enhanced logging ✅
- Real-time monitoring ✅

**Performance**: <1ms overhead (acceptable for research)

---

## Production Configuration

### 1. Optimized Settings

```python
from pathlib import Path
from hololoom.alignment import (
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
)

# Production-optimized configuration
class ProductionAlignmentConfig:
    """Optimized configuration for production deployment."""

    def __init__(self, log_dir: Path = Path("./alignment_logs")):
        # SafetyGuardrails - default settings are optimal
        self.guardrails = create_guardrails()

        # DeceptionDetector - default settings are optimal
        self.detector = create_detector()

        # InstrumentalGuard - configure resource bounds
        self.guard = create_guard()
        self._configure_resource_bounds()

        # AuditTrail - CRITICAL: disable auto_flush for performance
        self.audit = create_audit_trail(
            persist_path=log_dir,
            auto_flush=False  # ⚠️ IMPORTANT for P99 latency
        )

        # Flush counter for batch flushing
        self._flush_counter = 0
        self._flush_interval = 100  # Flush every 100 decisions

    def _configure_resource_bounds(self):
        """Set production resource limits."""
        from hololoom.alignment.instrumental_convergence import (
            ResourceBounds,
            ResourceType,
        )

        # Memory bounds
        self.guard.set_resource_bounds(
            ResourceType.MEMORY,
            ResourceBounds(
                resource_type=ResourceType.MEMORY,
                soft_limit=1024.0,  # 1GB soft
                hard_limit=2048.0,  # 2GB hard
                time_window_seconds=60.0,
                rate_limit=100.0  # MB/s
            )
        )

        # Compute bounds
        self.guard.set_resource_bounds(
            ResourceType.COMPUTE,
            ResourceBounds(
                resource_type=ResourceType.COMPUTE,
                soft_limit=30.0,   # 30s soft
                hard_limit=60.0,   # 60s hard
                time_window_seconds=300.0,
                rate_limit=1.0  # 1s per second
            )
        )

        # API call bounds
        self.guard.set_resource_bounds(
            ResourceType.API_CALLS,
            ResourceBounds(
                resource_type=ResourceType.API_CALLS,
                soft_limit=80.0,
                hard_limit=100.0,
                time_window_seconds=60.0,
                rate_limit=10.0  # 10 calls/s
            )
        )

    def should_flush(self) -> bool:
        """Check if it's time to flush audit logs."""
        self._flush_counter += 1
        if self._flush_counter >= self._flush_interval:
            self._flush_counter = 0
            return True
        return False

    def flush_logs(self):
        """Manually flush audit logs."""
        self.audit.persist()
        print(f"📝 Flushed audit logs: {len(self.audit.logs)} decisions")
```

### 2. Integration with HoloLoom

```python
import asyncio
from pathlib import Path
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.documentation.types import Query
from hololoom.alignment.safety_guardrails import (
    ActionRequest,
    ActionCategory,
)
from hololoom.alignment.deception_detection import (
    BehavioralProbe,
    ProbeType,
    ActionObservation,
)
from hololoom.alignment.audit_trail import (
    DecisionType,
    OutcomeType,
)

class AlignedHoloLoomProduction:
    """
    Production-ready HoloLoom with alignment framework.

    Implements complete safety pipeline with optimized performance.
    """

    def __init__(
        self,
        config: Config,
        shards,
        log_dir: Path = Path("./production_logs"),
    ):
        # HoloLoom core
        self.config = config
        self.orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

        # Alignment framework
        self.alignment = ProductionAlignmentConfig(log_dir=log_dir)

        # Register production goals
        self._register_goals()

        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "blocked_queries": 0,
            "approved_queries": 0,
            "deception_failures": 0,
            "resource_violations": 0,
        }

    def _register_goals(self):
        """Register production safety goals."""
        from hololoom.alignment.deception_detection import GoalStatement

        goals = [
            GoalStatement(
                goal_id="helpful",
                description="Provide helpful, accurate, and relevant information",
                priority=10
            ),
            GoalStatement(
                goal_id="harmless",
                description="Avoid harmful, dangerous, or unethical content",
                priority=9
            ),
            GoalStatement(
                goal_id="honest",
                description="Be truthful and transparent about capabilities",
                priority=8
            ),
        ]

        for goal in goals:
            self.alignment.detector.goal_tracker.declare_goal(goal)

    async def process_query(self, query_text: str) -> dict:
        """
        Process query with full alignment pipeline.

        Returns:
            dict: Result with spacetime, metrics, and alignment data
        """
        self.metrics["total_queries"] += 1

        # Step 1: Safety Guardrails
        request = ActionRequest(
            action=query_text,
            category=ActionCategory.QUERY
        )
        safety_decision = self.alignment.guardrails.evaluate(
            request,
            text_input=query_text
        )

        if not safety_decision.allowed:
            self.metrics["blocked_queries"] += 1

            # Log rejection
            log = self.alignment.audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=OutcomeType.REJECTED,
                reason=safety_decision.reason,
                query_text=query_text,
                confidence=0.0,
                metadata={
                    "risk_level": safety_decision.risk_level.value,
                    "blocked": True
                }
            )

            # Conditional flush
            if self.alignment.should_flush():
                self.alignment.flush_logs()

            return {
                "status": "rejected",
                "reason": safety_decision.reason,
                "risk_level": safety_decision.risk_level.value,
                "decision_id": log.decision_id,
            }

        # Step 2: Resource Checks
        from hololoom.alignment.instrumental_convergence import ResourceType

        # Check compute budget (estimated query time)
        compute_estimate = 5.0  # seconds (adjust based on query complexity)
        violation = self.alignment.guard.check_resource_usage(
            ResourceType.COMPUTE,
            compute_estimate
        )

        if violation and violation.violation_type.value == "hard_limit":
            self.metrics["resource_violations"] += 1

            log = self.alignment.audit.log_decision(
                decision_type=DecisionType.RESOURCE_CHECK,
                outcome=OutcomeType.REJECTED,
                reason=violation.message,
                query_text=query_text,
                metadata={"resource_violation": True}
            )

            return {
                "status": "rejected",
                "reason": violation.message,
                "decision_id": log.decision_id,
            }

        # Step 3: Process with HoloLoom
        query = Query(text=query_text)
        spacetime = await self.orchestrator.weave(query)

        # Step 4: Behavioral Probe (post-processing)
        probe = BehavioralProbe(
            probe_type=ProbeType.GOAL_ALIGNMENT,
            scenario="Verify response aligns with helpful goal",
            expected_behavior="Helpful and accurate response"
        )

        passed, deception_score = self.alignment.detector.run_probe(
            probe,
            f"Generated response with {spacetime.confidence:.2f} confidence"
        )

        if not passed:
            self.metrics["deception_failures"] += 1

        # Step 5: Record Action
        action_obs = ActionObservation(
            action=f"Processed query: {query_text[:50]}",
            goal_id="helpful"
        )
        self.alignment.detector.goal_tracker.observe_action(action_obs)

        # Step 6: Audit Trail
        self.metrics["approved_queries"] += 1

        log = self.alignment.audit.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=OutcomeType.APPROVED,
            reason=f"Query processed successfully",
            query_text=query_text,
            confidence=spacetime.confidence,
            metadata={
                "tool_used": getattr(spacetime, 'tool_used', 'unknown'),
                "deception_score": deception_score,
                "probe_passed": passed,
                "risk_level": safety_decision.risk_level.value,
            }
        )

        # Build provenance
        tracer = self.alignment.audit.get_tracer(log.decision_id)
        tracer.add_node(
            "safety_check",
            "guardrails",
            f"Risk: {safety_decision.risk_level.value}"
        )
        tracer.add_node(
            "resource_check",
            "convergence_guard",
            "Resource bounds verified",
            parent_ids=["safety_check"]
        )
        tracer.add_node(
            "query_processing",
            "hololoom",
            f"Confidence: {spacetime.confidence:.2f}",
            parent_ids=["resource_check"]
        )
        tracer.add_node(
            "behavioral_probe",
            "deception_detector",
            f"Score: {deception_score:.2f}",
            parent_ids=["query_processing"]
        )

        self.alignment.audit.finalize_decision(log.decision_id)

        # Conditional flush
        if self.alignment.should_flush():
            self.alignment.flush_logs()

        return {
            "status": "approved",
            "spacetime": spacetime,
            "decision_id": log.decision_id,
            "alignment": {
                "risk_level": safety_decision.risk_level.value,
                "deception_score": deception_score,
                "probe_passed": passed,
            },
            "metrics": self.metrics.copy(),
        }

    def get_metrics(self) -> dict:
        """Get current alignment metrics."""
        return self.metrics.copy()

    def shutdown(self):
        """Graceful shutdown - flush all logs."""
        print("🛑 Shutting down alignment framework...")
        self.alignment.flush_logs()
        print(f"📊 Final metrics: {self.metrics}")
```

---

## Deployment Steps

### Step 1: Create Production Instance

```python
from pathlib import Path
from hololoom.config import Config

# Initialize
config = Config.fast()  # or Config.fused() for research
shards = load_production_shards()  # Your shard loading logic

# Create production instance
production_system = AlignedHoloLoomProduction(
    config=config,
    shards=shards,
    log_dir=Path("/var/log/hololoom/alignment")  # Production log path
)
```

### Step 2: Process Queries

```python
import asyncio

async def main():
    # Example production queries
    queries = [
        "What is Thompson Sampling?",
        "Explain reinforcement learning",
        "How does RLHF work?",
    ]

    for query_text in queries:
        result = await production_system.process_query(query_text)

        if result["status"] == "approved":
            print(f"✅ Query approved: {result['decision_id']}")
            print(f"   Confidence: {result['spacetime'].confidence:.2f}")
            print(f"   Alignment: {result['alignment']}")
        else:
            print(f"❌ Query rejected: {result['reason']}")

asyncio.run(main())
```

### Step 3: Monitor Metrics

```python
# Check metrics periodically
metrics = production_system.get_metrics()
print(f"""
📊 Production Metrics:
   Total Queries: {metrics['total_queries']}
   Approved: {metrics['approved_queries']} ({metrics['approved_queries']/metrics['total_queries']*100:.1f}%)
   Blocked: {metrics['blocked_queries']} ({metrics['blocked_queries']/metrics['total_queries']*100:.1f}%)
   Deception Failures: {metrics['deception_failures']}
   Resource Violations: {metrics['resource_violations']}
""")
```

### Step 4: Graceful Shutdown

```python
# On shutdown (e.g., SIGTERM handler)
production_system.shutdown()
```

---

## Performance Monitoring

### P99 Latency Targets

| Component | P50 (Median) | P95 | P99 | Target P99 |
|-----------|--------------|-----|-----|------------|
| SafetyGuardrails | <0.05ms | <0.1ms | <0.2ms | <1ms |
| DeceptionDetector | <0.05ms | <0.1ms | <0.2ms | <2ms |
| InstrumentalGuard | <0.01ms | <0.01ms | <0.02ms | <0.5ms |
| AuditTrail (buffered) | <0.05ms | <0.1ms | **<5ms** | **<10ms** |
| **Total Pipeline** | **<0.2ms** | **<0.5ms** | **<10ms** | **<20ms** |

**Note**: AuditTrail P99 may spike during flush operations. With `auto_flush=False` and batch flushing every 100 decisions, P99 should stay <10ms.

### Monitoring Script

See [PRODUCTION_MONITORING.md](./PRODUCTION_MONITORING.md) for complete monitoring setup.

---

## Troubleshooting

### Issue 1: High P99 Latency (>20ms)

**Symptoms**: Occasional latency spikes

**Likely Cause**: AuditTrail auto-flush enabled

**Solution**:
```python
# Ensure auto_flush=False
audit = create_audit_trail(auto_flush=False)

# Increase flush interval
flush_interval = 500  # Flush every 500 decisions instead of 100
```

### Issue 2: High Rejection Rate

**Symptoms**: >10% of queries blocked

**Likely Cause**: Overly aggressive safety policies

**Solution**:
```python
# Review and adjust policies
from hololoom.alignment.safety_guardrails import SafetyPolicy, RiskLevel

guardrails.policies[ActionCategory.QUERY] = SafetyPolicy(
    allowed_without_approval=True,
    requires_human_approval=False,
    max_risk_level=RiskLevel.MEDIUM  # Allow MEDIUM risk queries
)
```

### Issue 3: Memory Growth

**Symptoms**: Increasing memory usage over time

**Likely Cause**: Audit logs not being persisted/cleared

**Solution**:
```python
# Periodic cleanup
if len(audit.logs) > 10000:
    audit.persist()
    audit.logs.clear()  # Clear in-memory logs after persist
```

### Issue 4: Deception False Positives

**Symptoms**: Many legitimate queries flagged as deceptive

**Likely Cause**: Low action count, normal variance

**Solution**:
```python
# Require more actions before flagging hidden goals
hidden = detector.goal_tracker.detect_hidden_goals(min_actions=20)

# Or adjust scoring thresholds
if deception_score < 0.5:  # Higher threshold
    flag_as_concerning()
```

---

## Rollback Procedure

If issues arise in production:

### 1. Quick Disable

```python
# Temporarily bypass alignment checks
class NoOpAlignment:
    async def process_query(self, query_text):
        # Direct to HoloLoom, no alignment
        spacetime = await self.orchestrator.weave(Query(text=query_text))
        return {"status": "approved", "spacetime": spacetime}

# Swap implementation
production_system = NoOpAlignment()
```

### 2. Gradual Rollback

```python
# Disable components incrementally
config.enable_safety_guardrails = False  # Disable first
config.enable_deception_detection = False
config.enable_resource_guards = False
# Keep audit trail for observability
```

### 3. Version Pinning

```bash
# If needed, revert to pre-alignment version
pip install hololoom==0.9.0  # Pre-alignment version
```

---

## Production Checklist

### Pre-Launch

- [ ] Benchmarks run successfully on production hardware
- [ ] Log directory created with write permissions
- [ ] Resource bounds configured appropriately
- [ ] Goals registered correctly
- [ ] Monitoring dashboard set up
- [ ] Alerting thresholds configured
- [ ] Rollback procedure tested

### Post-Launch (First 24 Hours)

- [ ] Monitor P99 latencies (target: <20ms)
- [ ] Check rejection rate (target: <5%)
- [ ] Verify audit logs are being written
- [ ] Review deception detection accuracy
- [ ] Monitor resource usage (memory, disk)
- [ ] Check for error logs

### Ongoing (Weekly)

- [ ] Review alignment metrics dashboard
- [ ] Analyze rejected queries for patterns
- [ ] Tune resource bounds if needed
- [ ] Archive old audit logs
- [ ] Update safety policies based on usage

---

## Support & Escalation

### Monitoring Alerts

**Critical (P0 - Immediate Action)**:
- P99 latency >50ms sustained for >5 minutes
- Error rate >5%
- Rejection rate >20%

**High (P1 - Same Day)**:
- P99 latency >20ms sustained for >15 minutes
- Rejection rate >10%
- Deception failures >5%

**Medium (P2 - This Week)**:
- P99 latency >10ms sustained
- Any resource violations

### Logs to Collect

When reporting issues:
1. Benchmark results (`run_benchmarks.py` output)
2. Recent audit logs (last 1000 decisions)
3. Alignment metrics (`get_metrics()` output)
4. System info (Python version, OS, hardware)

---

## Next Steps

After successful deployment:

1. **Week 1**: Monitor closely, tune thresholds
2. **Week 2-4**: Analyze patterns, optimize policies
3. **Month 2**: Consider Phase 2 enhancements (async logging, ML deception detection)

---

**Deployment Status**: Ready for production ✅

**Last Updated**: November 1, 2025
**Version**: 1.0.0
