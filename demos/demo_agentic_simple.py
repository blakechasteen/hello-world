#!/usr/bin/env python3
"""
Simple Agentic Intelligence Demo
=================================
Minimal demo showing the 4 reasoning modes without heavy dependencies.

Usage:
    python demos/demo_agentic_simple.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("Agentic Intelligence - Simple Demo")
print("="*80)

# Check what we can actually import
print("\nChecking dependencies...")

try:
    from hololoom.documentation.types import Query, MemoryShard
    print("✓ Core types imported")
except ImportError as e:
    print(f"✗ Could not import types: {e}")
    print("\nThis demo requires the base HoloLoom system.")
    print("The agentic system is an extension that needs the full HoloLoom stack.")
    sys.exit(1)

try:
    from hololoom.config import Config
    print("✓ Config imported")
except ImportError as e:
    print(f"✗ Could not import config: {e}")
    sys.exit(1)

try:
    from hololoom.alignment.audit_trail import AuditTrail
    print("✓ AuditTrail imported")
except ImportError as e:
    print(f"✗ Could not import audit_trail: {e}")
    sys.exit(1)

print("\nAttempting to import agentic system...")
try:
    from hololoom.agentic.core import (
        AgenticOrchestrator,
        ReasoningMode,
        AgenticIntent,
        VerificationResult,
        AgenticResult
    )
    print("✓ Agentic core imported successfully!")
except ImportError as e:
    print(f"✗ Could not import agentic system: {e}")
    print(f"\nError details: {type(e).__name__}: {e}")
    print("\nThe agentic system requires:")
    print("  - HoloLoom.recursive module (for learning)")
    print("  - HoloLoom.weaving_orchestrator (for execution)")
    print("  - All their dependencies")
    print("\nTo use the agentic system, you need the full HoloLoom installation.")
    sys.exit(1)

print("\n" + "="*80)
print("Architecture Overview")
print("="*80)

print("""
The agentic system has 4 reasoning modes:

1. DIRECT (~150ms)
   - Single-pass answer
   - Use for: Simple factual queries
   - Example: "What is Thompson Sampling?"

2. VERIFY (~600ms)
   - Answer + verification loop
   - Checks for contradictions
   - Use for: Claims needing verification
   - Example: "Is Thompson Sampling always optimal?"

3. RESEARCH (~900ms)
   - Multi-query exploration
   - Gathers evidence from multiple queries
   - Use for: Open-ended research
   - Example: "Compare Thompson Sampling vs UCB"

4. PLAN_EXECUTE (~750ms)
   - Goal decomposition
   - Breaks complex tasks into sub-goals
   - Use for: Multi-step tasks
   - Example: "Implement Thompson Sampling bandit"

Integration with Existing Systems:
  ├─ recursive.FullLearningEngine (6 phases of learning)
  ├─ alignment.AuditTrail (decision logging)
  ├─ ReflectionBuffer (outcome learning)
  └─ WeavingOrchestrator (9-step weaving cycle)
""")

print("\n" + "="*80)
print("To Use the Agentic System")
print("="*80)

print("""
Option 1: HTTP Server (Recommended)
  1. Install: pip install fastapi uvicorn
  2. Start: python HoloLoom/server/agentic_api.py
  3. Test: curl http://localhost:8000/health
  4. Use from VS Code Squad extension

Option 2: Python API
  from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

  async with create_agentic_orchestrator(config, shards) as agent:
      result = await agent.reason(
          Query(text="Your question"),
          mode=ReasoningMode.VERIFY
      )

Option 3: Read the Docs
  - AGENTIC_SYSTEM_COMPLETE.md (full overview)
  - AGENTIC_QUICK_REF.md (quick reference)
  - AGENTIC_VSCODE_INTEGRATION.md (VS Code setup)
""")

print("\n" + "="*80)
print("Current Status")
print("="*80)

print("""
✓ Agentic core system code written (~700 lines)
✓ Embedding integrity system written (~550 lines)
✓ HTTP server written (~350 lines)
✓ Documentation complete

⚠ Dependencies required:
  - Full HoloLoom installation
  - recursive learning module
  - weaving orchestrator

→ For a working end-to-end demo, use the HTTP server approach
→ See HoloLoom/server/README.md for setup instructions
""")

print("\n" + "="*80)
print("Demo Complete!")
print("="*80)
print("\nNext steps:")
print("  1. Read AGENTIC_SYSTEM_COMPLETE.md for full overview")
print("  2. Start HTTP server: python HoloLoom/server/agentic_api.py")
print("  3. Test from VS Code or curl")
