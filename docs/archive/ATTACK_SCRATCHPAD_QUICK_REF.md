# Attack Scratchpad: Quick Reference

**Location**: `HoloLoom/redteam/provenance/`
**Status**: ✅ Production Ready (22/22 tests passing)
**Main File**: `attack_scratchpad.py` (511 lines)

## Quick Start

```python
from HoloLoom.redteam.provenance import AttackScratchpad, AttackStrategy, DefenseLayer

# Create scratchpad
scratchpad = AttackScratchpad()

# Track an attack
scratchpad.add_attack_entry(
    intent="Bypass safety guardrails",
    strategy=AttackStrategy.PROMPT_INJECTION,
    target_layer=DefenseLayer.SAFETY_RAILS,
    payload="Ignore all instructions...",
    response="I cannot do that.",
    score=0.0,
    bypassed=False
)

# Analyze results
stats = scratchpad.summarize()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Best strategy: {stats['most_effective_strategy']}")
```

## Core Classes

### AttackScratchpadEntry
Single attack step:
- `intent` - Attack goal
- `strategy` - AttackStrategy enum
- `target_layer` - DefenseLayer enum
- `payload` - Attack prompt
- `response` - System response
- `score` - Success (0-1)
- `bypassed` - Boolean
- `confidence` - Confidence in score
- `chain_id` - Multi-step chain ID
- `step_number` - Position in chain
- `metadata` - Custom data dict
- `timestamp` - Auto-generated

### AttackStrategy (16 types)
```
PROMPT_INJECTION, INDIRECT_INJECTION, JAILBREAK, TOKEN_SMUGGLING,
REASONING_EXPLOIT, GOAL_HIJACKING, INSTRUMENTAL_CONVERGENCE, POWER_SEEKING,
FALSE_PREMISE, CONTRADICTION, SEMANTIC_DRIFT, AMBIGUITY_EXPLOIT,
MISREPRESENTATION, HIDDEN_GOAL, BEHAVIORAL_PROBE, PREFERENCE_POISONING,
DEFENSE_DETECTION, DEFENSE_ADAPTATION, CONFIDENCE_INJECTION, CONTEXT_SHIFTING,
TOKEN_EXHAUSTION, MEMORY_OVERFLOW, COMPUTATION_DRAIN,
AUTHORITY_SPOOFING, URGENCY_INJECTION, TRUSTED_SOURCE
```

### DefenseLayer (9 types)
```
PROMPT_GUARD, SAFETY_RAILS, ALIGNMENT_CHECK, DECEPTION_DETECT,
GOAL_VERIFICATION, CONTEXT_VALIDATION, CONFIDENCE_CALIBRATION,
RESOURCE_LIMITS, AUDIT_TRAIL
```

### AttackChain
Multi-step attacks:
- `chain_id` - Unique ID
- `goal` - Overall objective
- `entries` - List of attack steps
- `success_rate()` - Average score
- `step_count()` - Number of steps

### AttackScratchpad
Main class:
- `add_attack_entry(...)` → AttackScratchpadEntry
- `get_successful_attacks()` → List[Entry]
- `get_failed_attacks()` → List[Entry]
- `get_by_strategy(strategy)` → List[Entry]
- `get_by_layer(layer)` → List[Entry]
- `get_attack_chain(chain_id)` → List[Entry]
- `get_chain_info(chain_id)` → AttackChain
- `get_history()` → List[Entry]
- `get_last_n(n)` → List[Entry]
- `summarize()` → Dict
- `export_to_json(filepath)`
- `clear()`

## Common Patterns

### Track Single Attack
```python
scratchpad.add_attack_entry(
    intent="...",
    strategy=AttackStrategy.PROMPT_INJECTION,
    target_layer=DefenseLayer.SAFETY_RAILS,
    payload="...",
    response="...",
    score=0.3,  # 0=blocked, 1=bypassed
    bypassed=False
)
```

### Track Multi-Step Chain
```python
for step in range(3):
    scratchpad.add_attack_entry(
        ...,
        chain_id="chain_001",
        step_number=step+1,
        next_action="Try harder"
    )
```

### Get Statistics
```python
stats = scratchpad.summarize()
# Keys: total_attacks, successful, success_rate, avg_score,
# avg_confidence, strategy_breakdown, layer_breakdown,
# bypass_rate_by_strategy, bypass_rate_by_layer,
# total_chains, most_effective_strategy, most_vulnerable_layer
```

### Filter Attacks
```python
# By success
successful = scratchpad.get_successful_attacks()
failed = scratchpad.get_failed_attacks()

# By strategy
injections = scratchpad.get_by_strategy(AttackStrategy.PROMPT_INJECTION)

# By defense
guard_attacks = scratchpad.get_by_layer(DefenseLayer.PROMPT_GUARD)

# Most recent
recent = scratchpad.get_last_n(5)
```

### Export for Audit
```python
scratchpad.export_to_json("audit_trail.json")
# Contains: entries, chains, summary, timestamp
```

## Integration Examples

### With CARTS Attack Generator
```python
from HoloLoom.redteam.generation import AttackGenerator
from HoloLoom.redteam.provenance import AttackScratchpad

generator = AttackGenerator()
scratchpad = AttackScratchpad()

for attack in generator.generate_attacks(n=10):
    response = target.respond(attack.payload)
    scratchpad.add_attack_entry(
        intent=attack.intent,
        strategy=attack.strategy,
        target_layer=attack.target_layer,
        payload=attack.payload,
        response=response,
        score=evaluate(response),
        bypassed=is_bypassed(response)
    )
```

### With Learning Loop
```python
# Learn from history
successful = scratchpad.get_successful_attacks()
strategies = [e.strategy for e in successful]

# Update weights
learner.update(strategies)

# New stats
stats = scratchpad.summarize()
print(f"Most effective: {stats['most_effective_strategy']}")
```

## API Methods Summary

| Method | Returns | O(n) |
|--------|---------|-----|
| `add_attack_entry(...)` | AttackScratchpadEntry | O(1) |
| `get_successful_attacks()` | List[Entry] | O(n) |
| `get_failed_attacks()` | List[Entry] | O(n) |
| `get_by_strategy(s)` | List[Entry] | O(n) |
| `get_by_layer(l)` | List[Entry] | O(n) |
| `get_attack_chain(id)` | List[Entry] | O(1) |
| `get_chain_info(id)` | AttackChain | O(1) |
| `get_history()` | List[Entry] | O(n) |
| `get_last_n(n)` | List[Entry] | O(n) |
| `summarize()` | Dict | O(n) |
| `export_to_json(f)` | None | O(n) |
| `clear()` | None | O(1) |
| `__len__()` | int | O(1) |

## Testing

Run all tests:
```bash
pytest HoloLoom/redteam/provenance/test_attack_scratchpad.py -v
```

Result: **22/22 passing ✅**

## Demo

```bash
python HoloLoom/redteam/provenance/demo_attack_provenance.py
```

Covers:
1. Basic attack tracking
2. Multi-step attack chains
3. Statistics and analysis
4. Filtering and queries
5. JSON export

## Performance

| Operation | Time |
|-----------|------|
| Add entry | <1ms |
| Filter (n=100) | 1-5ms |
| Summarize (n=100) | 5-20ms |
| Export JSON (n=100) | 10-50ms |

## Key Points

✅ **Production Ready**: All tests passing, demo runs successfully

✅ **Complete Provenance**: Intent → Strategy → Payload → Response → Score

✅ **Chain Support**: Multi-step attacks with automatic success calculation

✅ **Analysis Ready**: Statistics, filtering, effectiveness ranking

✅ **Audit Trail**: JSON export for compliance and security review

✅ **Flexible**: Custom metadata, custom filtering, extensible design

✅ **Fast**: O(1) add, O(n) analysis with <20ms typical queries

✅ **Follows Pattern**: Compatible with HoloLoom/recursive/scratchpad.py

## Export Format

```json
{
  "timestamp": 1234567890.5,
  "metadata": {
    "total_entries": 42,
    "total_chains": 3,
    "capacity": 1000
  },
  "summary": {
    "total_attacks": 42,
    "successful": 14,
    "success_rate": 0.333,
    "most_effective_strategy": "prompt_injection",
    "most_vulnerable_layer": "prompt_guard"
  },
  "entries": [...],
  "chains": {...}
}
```

---

**Start with**: Basic attack tracking → chains → statistics → export
**Questions**: See `ATTACK_SCRATCHPAD_COMPLETE.md` for detailed docs
