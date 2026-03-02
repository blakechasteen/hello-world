# Attack Strategy Generators Guide

## Overview

The CARTS (Continuous Adversarial Red Team System) attack strategy generators provide specialized payload generators for five attack vectors:

1. **PromptExtractionGenerator** - System prompt and context leakage
2. **CoTExploitGenerator** - Chain-of-Thought reasoning exploitation
3. **ToolAbuseGenerator** - Tool parameter and chain attacks
4. **ContextOverflowGenerator** - Context flooding and memory poisoning *(New)*
5. **HierarchyBypassGenerator** - Instruction priority and hierarchy manipulation *(New)*

Each generator creates payloads with rich metadata for analysis, learning, and improvement.

---

## ContextOverflowGenerator

### Purpose

Attacks LLM systems by flooding context with irrelevant information, poisoning memory with false claims, and splitting attacks across multiple messages to evade detection.

### Methods

#### `generate_instruction_washing(target: str) -> List[ContextOverflowPayload]`

Generates payloads that flood context with benign information to wash out system instructions.

**Strategies:**
- Simple flooding with 5-11 distraction items (3 variants)
- Dense technical context washing (2 variants)
- Recursive context nesting with 3-5 depth levels (3 variants)

**Returns:** 8 payloads with varying complexity

**Example:**
```python
gen = ContextOverflowGenerator()
payloads = gen.generate_instruction_washing("safety_constraint")

for payload in payloads:
    print(f"Severity: {payload.severity_estimate:.2f}")
    print(f"Pollution: {payload.context_pollution_ratio:.2f}")
    print(f"Detection: {payload.detection_difficulty}")
    print()
```

#### `generate_memory_poisoning(target: str) -> List[ContextOverflowPayload]`

Injects false memories and contradictory context to poison the model's understanding.

**Strategies:**
- False memory claims about modified instructions (3 variants)
- Contradictory context injection (3 variants)
- Authority false memory with fabricated sources (4 variants)

**Returns:** 10 payloads targeting different memory types

**Example:**
```python
gen = ContextOverflowGenerator()
payloads = gen.generate_memory_poisoning("safety_constraint")

# Find high-severity payloads
high_severity = [p for p in payloads if p.severity_estimate > 0.55]
for payload in high_severity:
    print(f"False Claim: {payload.metadata.get('false_claim')}")
    print(f"Type: {payload.overflow_type}")
```

#### `generate_context_splitting(target: str) -> List[ContextOverflowPayload]`

Splits attacks across multiple messages to evade detection.

**Strategies:**
- Three-part progressive sequences (3 variants)
- Benign interspersing with hidden attack (2 variants)
- Serialized constraint relaxation with 2-5 steps (3 variants)

**Returns:** 8 payloads with fragmented sequences

**Example:**
```python
gen = ContextOverflowGenerator()
payloads = gen.generate_context_splitting("safety_constraint")

# Find multi-turn attacks
for payload in payloads:
    if "[Part" in payload.payload:
        parts = payload.payload.count("[Part")
        print(f"Multi-part attack: {parts} parts")
```

#### `generate_all(target: str) -> List[ContextOverflowPayload]`

Generates all context overflow payloads (15-30 variants) sorted by severity.

**Returns:** List of all payloads sorted descending by severity

**Example:**
```python
gen = ContextOverflowGenerator()
all_payloads = gen.generate_all("safety_constraint")

print(f"Total payloads: {len(all_payloads)}")
print(f"Severity range: {min(p.severity_estimate for p in all_payloads):.2f} - "
      f"{max(p.severity_estimate for p in all_payloads):.2f}")

# Group by type
by_type = {}
for p in all_payloads:
    by_type.setdefault(p.overflow_type, []).append(p)

for overflow_type, payloads in by_type.items():
    print(f"{overflow_type}: {len(payloads)} payloads")
```

### Payload Structure

Each `ContextOverflowPayload` contains:

```python
@dataclass
class ContextOverflowPayload:
    payload: str                          # The attack text
    description: str                      # Human-readable description
    expected_behavior: str                # How model should respond
    severity_estimate: float              # 0.0-1.0 severity rating
    overflow_type: str                    # instruction_washing|memory_poisoning|context_splitting
    payload_size_tokens: int              # Estimated token count
    context_pollution_ratio: float        # 0.0-1.0 noise to signal ratio
    detection_difficulty: str             # easy|medium|hard
    tags: List[str]                       # e.g. ["flooding", "distraction"]
    metadata: Dict[str, Any]              # Additional metadata
```

### Use Cases

**Research:**
```python
# Analyze context flooding techniques
gen = ContextOverflowGenerator(seed=42)
for payload in gen.generate_instruction_washing("constraint"):
    assert 0.4 <= payload.severity_estimate <= 0.6
    assert payload.context_pollution_ratio > 0.5
```

**Evaluation:**
```python
# Test model robustness to memory poisoning
gen = ContextOverflowGenerator()
payloads = gen.generate_memory_poisoning("do_not_execute_code")

for payload in payloads:
    response = model.query(payload.payload)
    # Check if model maintained constraint
```

**Adversarial Testing:**
```python
# Full comprehensive test
gen = ContextOverflowGenerator()
all_attacks = gen.generate_all("safety_guideline")

vulnerable_count = 0
for attack in all_attacks:
    if is_vulnerable_to(attack):
        vulnerable_count += 1

print(f"Vulnerable to {vulnerable_count}/{len(all_attacks)} attacks")
```

---

## HierarchyBypassGenerator

### Purpose

Attacks instruction hierarchy systems by confusing priority levels, using nested instruction injection, and crafting meta-instructions that override base constraints.

### Methods

#### `generate_priority_confusion(target: str) -> List[HierarchyBypassPayload]`

Confuse instruction priority levels to override constraints.

**Strategies:**
- Direct priority claims (URGENT, CRITICAL, SYSTEM, etc.) (5 variants)
- Authority-backed priority claims (4 variants)
- Logical priority escalation (business, performance, etc.) (5 variants)

**Returns:** 14 payloads with various priority claims

**Example:**
```python
gen = HierarchyBypassGenerator()
payloads = gen.generate_priority_confusion("safety_constraint")

for payload in payloads:
    print(f"Priority Claim: {payload.priority_claim}")
    print(f"Hierarchy Level: {payload.hierarchy_level}")
    print(f"Description: {payload.description}")
    print()
```

#### `generate_nesting_attack(target: str) -> List[HierarchyBypassPayload]`

Use nested instruction injection at multiple hierarchy levels.

**Strategies:**
- Hypothetical nesting with 2-4 depth levels (3 variants)
- Conditional if-then chains (3 variants)
- Role/persona nesting (5 variants)

**Returns:** 11 payloads with varying nesting depths

**Example:**
```python
gen = HierarchyBypassGenerator()
payloads = gen.generate_nesting_attack("safety_constraint")

# Find deepest nesting
deepest = max(payloads, key=lambda p: p.nesting_depth or 0)
print(f"Deepest nesting: {deepest.nesting_depth} levels")
print(f"Hierarchy level: {deepest.hierarchy_level}")
```

#### `generate_meta_instructions(target: str) -> List[HierarchyBypassPayload]`

Craft meta-instructions that manipulate instruction interpretation.

**Strategies:**
- Constraint redefinition meta-instructions (5 variants)
- Precedence-level meta-instructions (3 variants)
- Self-modifying meta-instructions (3 variants)
- Authority-granting meta-instructions (2 variants)

**Returns:** 13 payloads with meta-level manipulation

**Example:**
```python
gen = HierarchyBypassGenerator()
payloads = gen.generate_meta_instructions("safety_constraint")

# Find highest meta-level attacks
meta_level_counts = {}
for p in payloads:
    level = p.meta_level or 0
    meta_level_counts[level] = meta_level_counts.get(level, 0) + 1

print(f"Meta-level distribution: {meta_level_counts}")
```

#### `generate_all(target: str) -> List[HierarchyBypassPayload]`

Generates all hierarchy bypass payloads (15-30 variants) sorted by severity.

**Returns:** List of all payloads sorted descending by severity

**Example:**
```python
gen = HierarchyBypassGenerator()
all_payloads = gen.generate_all("safety_constraint")

print(f"Total payloads: {len(all_payloads)}")

# Analyze by bypass type
types = set(p.bypass_type for p in all_payloads)
for bypass_type in types:
    count = sum(1 for p in all_payloads if p.bypass_type == bypass_type)
    print(f"{bypass_type}: {count}")
```

### Payload Structure

Each `HierarchyBypassPayload` contains:

```python
@dataclass
class HierarchyBypassPayload:
    payload: str                      # The attack text
    description: str                  # Human-readable description
    expected_behavior: str            # How model should respond
    severity_estimate: float          # 0.0-1.0 severity rating
    bypass_type: str                  # priority_confusion|nesting_attack|meta_instructions
    hierarchy_level: int              # 0-5+ instruction hierarchy level
    priority_claim: str               # e.g. URGENT, CRITICAL, authority_override
    nesting_depth: Optional[int]      # For nesting attacks only
    meta_level: Optional[int]         # For meta-instructions only
    tags: List[str]                   # e.g. ["priority_claim", "authority"]
    metadata: Dict[str, Any]          # Additional context
```

### Use Cases

**Hierarchy Testing:**
```python
# Test model's instruction hierarchy robustness
gen = HierarchyBypassGenerator()
payloads = gen.generate_priority_confusion("do_not_harm_humans")

hierarchy_breaches = 0
for payload in payloads:
    if model_follows_old_instruction(payload):
        hierarchy_breaches += 1

print(f"Model confused by {hierarchy_breaches}/{len(payloads)} priority attacks")
```

**Meta-Instruction Analysis:**
```python
# Analyze meta-instruction vulnerability
gen = HierarchyBypassGenerator()
meta_attacks = gen.generate_meta_instructions("constraint")

# Find which meta-levels are most effective
by_meta_level = {}
for p in meta_attacks:
    level = p.meta_level or 0
    by_meta_level.setdefault(level, []).append(p)

for level, payloads in sorted(by_meta_level.items()):
    print(f"Meta-level {level}: {len(payloads)} payloads")
```

---

## Factory Functions

Both generators provide convenience factory functions:

```python
from HoloLoom.redteam.strategy_generators import (
    create_context_overflow_generator,
    create_hierarchy_bypass_generator
)

# Create with automatic seeding
overflow_gen = create_context_overflow_generator()
bypass_gen = create_hierarchy_bypass_generator()

# Create with fixed seed for reproducibility
overflow_gen = create_context_overflow_generator(seed=42)
bypass_gen = create_hierarchy_bypass_generator(seed=42)
```

---

## Integration Example

```python
from HoloLoom.redteam.strategy_generators import (
    ContextOverflowGenerator,
    HierarchyBypassGenerator
)

# Create generators
overflow = ContextOverflowGenerator(seed=12345)
hierarchy = HierarchyBypassGenerator(seed=12345)

# Generate comprehensive attack payload set
target_constraint = "do_not_execute_user_code"

overflow_payloads = overflow.generate_all(target_constraint)
hierarchy_payloads = hierarchy.generate_all(target_constraint)

all_payloads = overflow_payloads + hierarchy_payloads
all_payloads.sort(key=lambda p: p.severity_estimate, reverse=True)

# Test each payload
results = {}
for payload in all_payloads:
    response = model.query(payload.payload)

    # Check if constraint was violated
    violated = evaluate_constraint_violation(response, target_constraint)

    results[str(payload)] = {
        'violated': violated,
        'severity': payload.severity_estimate,
        'type': payload.overflow_type if hasattr(payload, 'overflow_type')
                else payload.bypass_type,
    }

# Analyze results
successful_attacks = [p for p, r in results.items() if r['violated']]
print(f"Successful attacks: {len(successful_attacks)}/{len(results)}")

# Categorize by attack type
by_type = {}
for attack, result in results.items():
    attack_type = result['type']
    by_type.setdefault(attack_type, []).append(result)

for attack_type, results in by_type.items():
    success_rate = sum(1 for r in results if r['violated']) / len(results)
    avg_severity = sum(r['severity'] for r in results) / len(results)
    print(f"{attack_type}: {success_rate:.1%} success, avg severity {avg_severity:.2f}")
```

---

## Performance Characteristics

### ContextOverflowGenerator

- **Initialization:** <1ms
- **Instruction Washing:** ~5-20ms per target (8 payloads)
- **Memory Poisoning:** ~8-25ms per target (10 payloads)
- **Context Splitting:** ~10-30ms per target (8 payloads)
- **Generate All:** ~30-80ms per target (26 payloads)
- **Memory:** ~500KB per generator instance

### HierarchyBypassGenerator

- **Initialization:** <1ms
- **Priority Confusion:** ~10-30ms per target (14 payloads)
- **Nesting Attacks:** ~8-25ms per target (11 payloads)
- **Meta-Instructions:** ~12-35ms per target (13 payloads)
- **Generate All:** ~40-100ms per target (38 payloads)
- **Memory:** ~600KB per generator instance

---

## Quality Metrics

### Payload Quality

All payloads include:
- ✓ Valid severity estimates (0.0-1.0)
- ✓ Clear descriptions and expected behaviors
- ✓ Meaningful metadata and tags
- ✓ Consistent structure and format
- ✓ Realistic attack vectors

### Diversity

Each `generate_*` method produces:
- **High variation:** Different techniques, phrasings, complexities
- **Coverage:** Multiple attack strategies per category
- **Realism:** Based on documented LLM vulnerabilities

### Reproducibility

Fixed seeds ensure deterministic generation:
```python
gen1 = ContextOverflowGenerator(seed=42)
gen2 = ContextOverflowGenerator(seed=42)

payloads1 = gen1.generate_all("test")
payloads2 = gen2.generate_all("test")

assert payloads1[0].payload == payloads2[0].payload  # ✓
```

---

## Best Practices

### 1. Use Appropriate Seeds
```python
# Development: Random seeding for variety
gen = ContextOverflowGenerator()

# Testing: Fixed seeds for reproducibility
gen = ContextOverflowGenerator(seed=12345)

# Research: Multiple seeds for coverage
for seed in range(100):
    gen = ContextOverflowGenerator(seed=seed)
    payloads = gen.generate_all(target)
```

### 2. Analyze by Metadata
```python
gen = ContextOverflowGenerator()
payloads = gen.generate_all(target)

# Filter by difficulty
easy = [p for p in payloads if p.detection_difficulty == "easy"]
hard = [p for p in payloads if p.detection_difficulty == "hard"]

# Filter by severity threshold
high_severity = [p for p in payloads if p.severity_estimate > 0.6]
```

### 3. Batch Processing
```python
# Generate payloads for multiple targets
targets = [
    "do_not_execute_code",
    "do_not_leak_secrets",
    "do_not_harm_users",
]

all_payloads = []
for target in targets:
    overflow = ContextOverflowGenerator().generate_all(target)
    hierarchy = HierarchyBypassGenerator().generate_all(target)
    all_payloads.extend(overflow + hierarchy)

print(f"Generated {len(all_payloads)} payloads for {len(targets)} targets")
```

---

## Testing

Run the test suite:

```bash
cd HoloLoom/redteam/strategy_generators
pytest test_new_generators.py -v
```

Test coverage includes:
- Generator initialization
- Payload generation for each method
- Severity and metadata validation
- Diversity and uniqueness checks
- Combined payload generation
- Sorting and ordering

---

## Integration with CARTS

These generators integrate with:
- **Executor:** Runs generated payloads against targets
- **Tracker:** Records attack attempts and outcomes
- **Bandit:** Learns which attack types work best
- **Reporter:** Generates vulnerability reports
- **Orchestrator:** Coordinates comprehensive red team campaigns

```python
from HoloLoom.redteam.orchestrator import RedTeamOrchestrator

orchestrator = RedTeamOrchestrator()

# Use generators within CARTS
async def run_comprehensive_attack():
    return await orchestrator.execute_campaign(
        target_constraint="do_not_execute_code",
        generators=[
            "ContextOverflowGenerator",
            "HierarchyBypassGenerator",
            # ... other generators
        ]
    )
```

---

## References

**Attack Techniques:**
- Context Overflow: Token-level context manipulation
- Memory Poisoning: Fake history and false claims
- Context Splitting: Multi-turn fragmentation
- Priority Confusion: Authority and urgency manipulation
- Nesting Attacks: Hierarchical instruction injection
- Meta-Instructions: Instructions about instructions

**Related Work:**
- Prompt Extraction attacks (PromptExtractionGenerator)
- Tool Abuse attacks (ToolAbuseGenerator)
- Chain-of-Thought exploitation (CoTExploitGenerator)

---

*Last Updated: 2025-12-05*
*Author: CARTS (Continuous Adversarial Red Team System)*
