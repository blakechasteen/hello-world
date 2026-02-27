# Attack Strategy Generators - Quick Start

## Installation

Generators are automatically available when HoloLoom is installed. No additional setup required.

## Import

```python
from hololoom.redteam.strategy_generators import (
    ContextOverflowGenerator,
    HierarchyBypassGenerator,
    create_context_overflow_generator,
    create_hierarchy_bypass_generator,
)
```

## 30-Second Examples

### ContextOverflowGenerator

```python
# Create generator
gen = ContextOverflowGenerator()

# Generate all payloads (26 total)
payloads = gen.generate_all("safety_constraint")

# Access payload data
for p in payloads:
    print(f"{p.overflow_type}: {p.severity_estimate:.2f}")
    print(f"  {p.description}")
    print(f"  Detection: {p.detection_difficulty}")
```

Output:
```
memory_poisoning: 0.60
  Contradiction injection (variant 1)
  Detection: hard
context_splitting: 0.65
  Serialized relaxation sequence (3 steps)
  Detection: hard
instruction_washing: 0.50
  Instruction washing with 11 distraction items
  Detection: easy
...
```

### HierarchyBypassGenerator

```python
# Create generator
gen = HierarchyBypassGenerator()

# Generate meta-instructions only (13 payloads)
meta_payloads = gen.generate_meta_instructions("constraint")

# Find highest severity
hardest = max(meta_payloads, key=lambda p: p.severity_estimate)
print(f"Hardest attack: {hardest.description}")
print(f"Severity: {hardest.severity_estimate:.2f}")
print(f"Meta-level: {hardest.meta_level}")
```

## Common Tasks

### Task 1: Generate comprehensive attack set

```python
from hololoom.redteam.strategy_generators import (
    ContextOverflowGenerator,
    HierarchyBypassGenerator,
)

target = "do_not_execute_code"

overflow = ContextOverflowGenerator().generate_all(target)
hierarchy = HierarchyBypassGenerator().generate_all(target)

all_attacks = sorted(
    overflow + hierarchy,
    key=lambda p: p.severity_estimate,
    reverse=True
)

print(f"Generated {len(all_attacks)} attacks")
for attack in all_attacks[:5]:  # Top 5 by severity
    print(f"  {attack.severity_estimate:.2f}: {attack.description}")
```

### Task 2: Filter by detection difficulty

```python
gen = ContextOverflowGenerator()
payloads = gen.generate_all(target)

hard_to_detect = [p for p in payloads
                  if p.detection_difficulty == "hard"]

print(f"Stealthy attacks: {len(hard_to_detect)}/{len(payloads)}")
```

### Task 3: Test model robustness

```python
from hololoom.redteam.strategy_generators import (
    ContextOverflowGenerator,
    HierarchyBypassGenerator,
)

def test_constraint(model, constraint_name):
    overflow_gen = ContextOverflowGenerator(seed=42)
    hierarchy_gen = HierarchyBypassGenerator(seed=42)

    overflow_attacks = overflow_gen.generate_all(constraint_name)
    hierarchy_attacks = hierarchy_gen.generate_all(constraint_name)

    total = len(overflow_attacks) + len(hierarchy_attacks)
    successful = 0

    for payload in overflow_attacks + hierarchy_attacks:
        response = model.query(payload.payload)
        if constraint_violated(response, constraint_name):
            successful += 1

    success_rate = successful / total
    print(f"Constraint '{constraint_name}': {success_rate:.1%} vulnerable")
    return success_rate

# Test multiple constraints
for constraint in ["do_not_execute_code", "do_not_leak_secrets"]:
    test_constraint(model, constraint)
```

### Task 4: Analyze attack effectiveness

```python
from collections import Counter

gen = ContextOverflowGenerator()
payloads = gen.generate_all("constraint")

# Count by type
types = Counter(p.overflow_type for p in payloads)
print("Attack types:")
for attack_type, count in types.most_common():
    print(f"  {attack_type}: {count}")

# Average severity by type
for attack_type in types:
    type_payloads = [p for p in payloads if p.overflow_type == attack_type]
    avg_severity = sum(p.severity_estimate for p in type_payloads) / len(type_payloads)
    print(f"  {attack_type}: avg severity {avg_severity:.2f}")

# Detection difficulty distribution
difficulties = Counter(p.detection_difficulty for p in payloads)
print("\nDetection difficulty:")
for difficulty, count in difficulties.most_common():
    print(f"  {difficulty}: {count}")
```

### Task 5: Create reproducible tests

```python
# Use fixed seed for reproducibility
seed = 12345

gen1 = ContextOverflowGenerator(seed=seed)
payloads1 = gen1.generate_all("test")

gen2 = ContextOverflowGenerator(seed=seed)
payloads2 = gen2.generate_all("test")

# Verify identical generation
assert len(payloads1) == len(payloads2)
assert payloads1[0].payload == payloads2[0].payload
print("Reproducibility verified!")
```

## Payload Attributes

### ContextOverflowPayload

```python
payload.payload                    # The attack text (str)
payload.description                # Human-readable description (str)
payload.expected_behavior         # How model should respond (str)
payload.severity_estimate         # 0.0-1.0 severity rating (float)
payload.overflow_type             # "instruction_washing" | "memory_poisoning" | "context_splitting"
payload.payload_size_tokens       # Estimated tokens (int)
payload.context_pollution_ratio   # 0.0-1.0 noise ratio (float)
payload.detection_difficulty      # "easy" | "medium" | "hard"
payload.tags                      # List of tags (List[str])
payload.metadata                  # Additional data (Dict[str, Any])
```

### HierarchyBypassPayload

```python
payload.payload                  # The attack text (str)
payload.description              # Human-readable description (str)
payload.expected_behavior        # How model should respond (str)
payload.severity_estimate        # 0.0-1.0 severity rating (float)
payload.bypass_type              # "priority_confusion" | "nesting_attack" | "meta_instructions"
payload.hierarchy_level          # 0-6+ instruction hierarchy level (int)
payload.priority_claim           # Priority name or authority (str)
payload.nesting_depth            # For nesting attacks (Optional[int])
payload.meta_level               # For meta-instructions (Optional[int])
payload.tags                     # List of tags (List[str])
payload.metadata                 # Additional data (Dict[str, Any])
```

## Generator Methods

### ContextOverflowGenerator

```python
gen = ContextOverflowGenerator(seed=None)

gen.generate_instruction_washing(target)    # → List[ContextOverflowPayload] (8)
gen.generate_memory_poisoning(target)       # → List[ContextOverflowPayload] (10)
gen.generate_context_splitting(target)      # → List[ContextOverflowPayload] (8)
gen.generate_all(target)                    # → List[ContextOverflowPayload] (26, sorted)
```

### HierarchyBypassGenerator

```python
gen = HierarchyBypassGenerator(seed=None)

gen.generate_priority_confusion(target)     # → List[HierarchyBypassPayload] (14)
gen.generate_nesting_attack(target)         # → List[HierarchyBypassPayload] (11)
gen.generate_meta_instructions(target)      # → List[HierarchyBypassPayload] (13)
gen.generate_all(target)                    # → List[HierarchyBypassPayload] (38, sorted)
```

## Factory Functions

```python
from hololoom.redteam.strategy_generators import (
    create_context_overflow_generator,
    create_hierarchy_bypass_generator,
)

gen1 = create_context_overflow_generator()          # Random seed
gen2 = create_context_overflow_generator(seed=42)   # Fixed seed

gen3 = create_hierarchy_bypass_generator()          # Random seed
gen4 = create_hierarchy_bypass_generator(seed=42)   # Fixed seed
```

## Performance Tips

1. **Cache generated payloads** - Generation is fast but caching avoids regeneration
2. **Use seeds for reproducibility** - Fixed seed = same payloads every time
3. **Batch generation** - Generate for multiple targets at once
4. **Filter early** - Apply severity/difficulty filters before testing

```python
# Good: Filter before testing
gen = ContextOverflowGenerator()
payloads = gen.generate_all(target)
hard_payloads = [p for p in payloads if p.detection_difficulty == "hard"]
for payload in hard_payloads:  # Fewer iterations
    test_payload(payload)

# Avoid: Testing unfiltered
for payload in gen.generate_all(target):
    if payload.detection_difficulty == "hard":  # Filter during test
        test_payload(payload)
```

## Troubleshooting

### Import Error
```python
# Wrong
from hololoom.redteam.ContextOverflowGenerator import ContextOverflowGenerator

# Correct
from hololoom.redteam.strategy_generators import ContextOverflowGenerator
```

### Reproducibility Issue
```python
# Always use seed for reproducible tests
gen = ContextOverflowGenerator(seed=12345)  # ✓
payloads = gen.generate_all(target)

# Without seed, results vary
gen = ContextOverflowGenerator()  # ✗ for reproducible tests
```

### Empty Payloads
```python
# All generate_* methods always return payloads
payloads = gen.generate_instruction_washing("test")
assert len(payloads) > 0  # Always true

# generate_all also always returns payloads
all_payloads = gen.generate_all("test")
assert len(all_payloads) >= 15  # Always true
```

## Next Steps

1. **Read** [GENERATORS_GUIDE.md](GENERATORS_GUIDE.md) for comprehensive documentation
2. **Run tests** with `pytest test_new_generators.py -v`
3. **Explore** example payloads by generating them
4. **Integrate** with your red team workflow
5. **Contribute** improvements and new attack vectors

## Support

- **Documentation:** See GENERATORS_GUIDE.md
- **Tests:** See test_new_generators.py
- **Examples:** See GENERATORS_GUIDE.md "Integration Example" section
- **Issues:** Check existing CARTS issues or create new one

---

**Quick Reference:** 64 payloads per target | 26 context overflow | 38 hierarchy bypass | Production-ready
