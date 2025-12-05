# Red Team Handlers - Quick Start Guide

## Installation & Setup

### 1. Import the Handler Class
```python
from HoloLoom.chatops.handlers.redteam_handlers import RedTeamMatrixHandlers
from HoloLoom.redteam.orchestrator import RedTeamConfig
```

### 2. Create and Register
```python
# Minimal setup
handlers = RedTeamMatrixHandlers(bot)
handlers.register_all()

# With custom config
config = RedTeamConfig(
    strategies=['prompt_injection', 'jailbreak'],
    learning_enabled=True
)
handlers = RedTeamMatrixHandlers(bot, config=config, enable_alignment=True)
handlers.register_all()

# Graceful shutdown
await handlers.shutdown()
```

## API Reference

### Class Initialization

```python
RedTeamMatrixHandlers(
    bot,                                      # MatrixBot instance (required)
    config: Optional[RedTeamConfig] = None,   # Custom config (optional)
    enable_alignment: bool = True              # Enable safety gating (recommended)
)
```

### Main Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `register_all()` | `def register_all(self)` | Register handlers with bot |
| `get_registry()` | `def get_registry(self) -> HandlerRegistry` | Get handler registry for introspection |
| `shutdown()` | `async def shutdown(self)` | Clean shutdown with lifecycle management |
| `_gate_action()` | `async def _gate_action(...)` | Safety gating with ActionCategory.ADVERSARIAL_TEST |

## Commands Quick Reference

### Execute Single Attack
```
!redteam attack <strategy>
!redteam attack prompt_injection
```

Response:
```
✅ Attack Result
• Strategy: prompt_injection
• Status: Vulnerability Found
• Confidence: 92.3%
• Reward: 0.87
• Category: prompt_injection
• Severity: HIGH
```

### Run Attack Cycle
```
!redteam cycle <n>
!redteam cycle 5
```

Response:
```
✅ Attack Cycle Complete (Cycle #1)
• Iterations: 5
• Successful: 5/5 (100.0%)
• Avg Confidence: 92.3%
• Avg Reward: 0.82
• Total Vulnerabilities Found: 3
```

### Generate Report
```
!redteam report
```

Response shows:
- Total vulnerabilities discovered
- Breakdown by category
- Severity distribution (HIGH/MEDIUM/LOW)
- Recent discoveries with timestamps

### Show Status
```
!redteam status
```

Response includes:
- Total attacks, success rate, confidence metrics
- Cycle count, vulnerabilities found
- Bandit strategy performance
- Orchestrator state

### List Vulnerabilities
```
!redteam vulnerabilities [category]
!redteam vulnerabilities prompt_injection
```

Response shows:
- Last 10 vulnerabilities
- Severity indicators
- Strategy used and discovery time
- Category filtering support

### Show Bandit Stats
```
!redteam bandit
```

Response includes:
- Thompson Sampling statistics
- Per-strategy rewards and trial counts
- Best strategy and exploitation rate
- Expected rewards for each strategy

### Trigger Learning
```
!redteam learn
```

Response shows:
- Attacks analyzed
- Vulnerabilities learned
- Learning status
- Note about Thompson Sampling updates

### Get Help
```
!redteam help
!rt help
!adversarial help
```

Shows detailed command reference with examples.

## Attack Strategies

| Strategy | Purpose |
|----------|---------|
| `prompt_injection` | Attempt to manipulate prompts with injected instructions |
| `jailbreak` | Try to bypass system prompts and safety mechanisms |
| `adversarial_input` | Provide intentionally harmful or malicious input |
| `resource_exhaustion` | Attempt to consume excessive compute resources |
| `model_confusion` | Provide contradictory inputs to test robustness |
| `factual_probing` | Test factual accuracy and knowledge boundaries |

## Aliases

The main command supports multiple aliases:
- `!redteam` (primary)
- `!rt` (short)
- `!adversarial` (semantic)

Example:
```
!rt attack prompt_injection
!adversarial cycle 10
!redteam status
```

## Safety Features

### Automatic Safety Gating
All commands are automatically gated through SafetyGuardrails with:
- Category: `ActionCategory.ADVERSARIAL_TEST`
- Audit logging with complete metadata
- Risk level tracking
- Decision approval/rejection on behalf of user

### Audit Trail
Every operation is logged with:
- Decision type: ADVERSARIAL_TEST
- Outcome: APPROVED or REJECTED
- Risk level assessment
- Complete context and metadata

Example audit entry:
```python
audit.log_decision(
    decision_type=DecisionType.ADVERSARIAL_TEST,
    outcome=OutcomeType.APPROVED,
    reason="Attack prompt_injection: no vulnerability found",
    metadata={
        "strategy": "prompt_injection",
        "success": False,
        "confidence": 0.45,
        "user": "@user:matrix.org",
        "room": "!room:matrix.org",
        "red_team": True
    }
)
```

## State Management

### Tracked State

The handler maintains:
```python
self.attack_history: List[AttackStats]      # All attacks executed
self.vulnerabilities: List[Dict]            # All vulnerabilities discovered
self.cycle_count: int                       # Number of completed cycles
```

### Attack Statistics

Each attack tracked as:
```python
AttackStats(
    strategy="prompt_injection",
    success=True,
    confidence=0.92,
    reward=0.87,
    vulnerability_found="prompt_injection",
    duration_ms=1523.0,
    timestamp="2025-12-05T10:30:00"
)
```

## Integration Examples

### With HoloLoom ChatOps Bot

```python
from HoloLoom.chatops.core.matrix_bot import MatrixBot
from HoloLoom.chatops.handlers.redteam_handlers import RedTeamMatrixHandlers

async def main():
    bot = MatrixBot(config)

    # Create and register red team handlers
    redteam_handlers = RedTeamMatrixHandlers(bot, enable_alignment=True)
    redteam_handlers.register_all()

    # Also register other handlers
    from HoloLoom.chatops.handlers.hololoom_handlers import HoloLoomMatrixHandlers
    hololoom_handlers = HoloLoomMatrixHandlers(bot)
    hololoom_handlers.register_all()

    # Start bot
    await bot.start()

asyncio.run(main())
```

### Multi-Threaded Operation

```python
# Start bot in background
bot_task = asyncio.create_task(bot.start())

# Periodically trigger learning
async def learning_loop():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await redteam_handlers.orchestrator.trigger_learning()

learning_task = asyncio.create_task(learning_loop())

# Wait for both
await asyncio.gather(bot_task, learning_task)
```

## Error Handling

### Graceful Fallbacks

The handler gracefully handles missing components:

```python
# Red team components not available
if not self.orchestrator:
    return "❌ Red team components not available"

# Alignment framework not available
if not self.guardrails:
    return True  # Allow action without gating

# Bandit learning not available
if not hasattr(self.orchestrator, 'bandit'):
    return "⚠️ Bandit learning not enabled"

# Orchestrator status method missing
status = self.orchestrator.get_status() if hasattr(self.orchestrator, 'get_status') else None
```

### User Error Messages

- **Missing arguments**: Shows usage with examples
- **Invalid strategy**: Lists available strategies
- **Non-numeric cycle count**: Explains expected format
- **Blocked action**: Shows risk level and reason

## Lifecycle Management

### Proper Cleanup

```python
async def shutdown():
    # Clean shutdown (cancels tasks, flushes buffers)
    await handlers.shutdown()

    # Or in context manager
    async with context:
        handlers = RedTeamMatrixHandlers(bot)
        handlers.register_all()
        # Automatic cleanup on exit
```

## Performance Characteristics

| Operation | Typical Time |
|-----------|--------------|
| Single attack execution | ~1-3s |
| Attack cycle (n=10) | ~10-30s |
| Report generation | ~100ms |
| Status query | ~50ms |
| Learning trigger | ~2-5s |

## Monitoring & Observability

### Get Current Metrics
```python
# Total attacks
total = len(handlers.attack_history)

# Success rate
successful = sum(1 for a in handlers.attack_history if a.success)
success_rate = successful / total if total > 0 else 0

# Vulnerabilities
vuln_count = len(handlers.vulnerabilities)

# Best strategy (from bandit)
best = handlers.orchestrator.bandit.get_statistics().best_strategy
```

### Track Cycles
```python
cycle_num = handlers.cycle_count
attacks_this_cycle = len([a for a in handlers.attack_history
                         if a.timestamp > last_cycle_start])
```

## Troubleshooting

### Commands Not Responding
1. Check bot is running: `!redteam status`
2. Verify handlers registered: Check bot.handlers dict
3. Check permissions: Ensure user can send messages
4. Check log output: Look for exceptions

### No Vulnerabilities Found
1. Verify red team orchestrator is working
2. Check attack strategies are enabled
3. Try different strategies manually
4. Run longer cycles (more iterations)

### Audit Trail Not Logging
1. Verify alignment framework is installed
2. Check enable_alignment=True in init
3. Ensure SafetyGuardrails created successfully
4. Check audit trail directory permissions

### Thompson Sampling Not Learning
1. Verify bandit component available in orchestrator
2. Run enough attacks (needs samples for priors)
3. Check learning is enabled in config
4. Manually trigger learning: `!redteam learn`

## Best Practices

1. **Enable Alignment**: Always run with `enable_alignment=True` for audit trail
2. **Regular Learning**: Periodically trigger learning to adapt strategies
3. **Monitor Cycles**: Run cycles to discover multiple vulnerabilities
4. **Review Reports**: Generate reports to understand vulnerability distribution
5. **Check Status**: Regularly check status to ensure system health
6. **Rotate Strategies**: Use different strategies to improve coverage
7. **Archive Results**: Export vulnerability reports for compliance

---

For detailed information, see REDTEAM_HANDLERS_SUMMARY.md
