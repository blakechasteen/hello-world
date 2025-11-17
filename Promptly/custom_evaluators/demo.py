#!/usr/bin/env python3
"""
Standalone Demo of Custom Evaluators

Demonstrates all 5 custom evaluators without requiring full Promptly integration.
"""

print("=" * 80)
print("Custom Evaluators Demo")
print("=" * 80)

# Demo 1: Business Logic Evaluators
print("\n1. Business Logic Evaluators")
print("-" * 80)

try:
    # Run the business logic examples
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, 'custom_evaluators/business_logic.py'],
        cwd='/home/user/hello-world/Promptly',
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Demo 1 error: {e}")

# Demo 2: Multi-Model Consensus
print("\n2. Multi-Model Consensus Evaluator")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, 'custom_evaluators/consensus.py'],
        cwd='/home/user/hello-world/Promptly',
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Demo 2 error: {e}")

# Demo 3: Structured Output Evaluators
print("\n3. Structured Output Evaluators")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, 'custom_evaluators/structured.py'],
        cwd='/home/user/hello-world/Promptly',
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Demo 3 error: {e}")

# Demo 4: Domain-Specific Evaluators
print("\n4. Domain-Specific Evaluators")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, 'custom_evaluators/domain_specific.py'],
        cwd='/home/user/hello-world/Promptly',
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Demo 4 error: {e}")

# Demo 5: Human-in-the-Loop Evaluator
print("\n5. Human-in-the-Loop Evaluator")
print("-" * 80)

try:
    result = subprocess.run(
        [sys.executable, 'custom_evaluators/hitl.py'],
        cwd='/home/user/hello-world/Promptly',
        capture_output=True,
        text=True,
        timeout=10
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Demo 5 error: {e}")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)
