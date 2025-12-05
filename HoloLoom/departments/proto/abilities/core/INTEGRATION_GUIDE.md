# Code Execution Ability - Integration Guide

**Status**: Production ready (2025-12-03)
**Audience**: Proto developers, ability registry maintainers

This guide shows how to integrate the Code Execution ability into Proto's ability system.

## Quick Integration

### 1. Register with AbilityRegistry

```python
from HoloLoom.departments.proto.abilities.registry import AbilityRegistry
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility

# Create registry (typically done once at startup)
registry = AbilityRegistry(max_trust_level=AbilityTrustLevel.VERIFIED)

# Create and register code execution ability
ability = CodeExecutionAbility()
success = registry.register(ability)

if success:
    print("Code execution ability registered")
else:
    print("Failed to register code execution ability")
```

### 2. Retrieve and Use

```python
# Get ability from registry
ability = registry.get("code_execution")

if ability is None:
    print("Code execution ability not available")
    return

# Use ability
context = AbilityContext(session_id="user_123", user_confirmed=True)
result = await ability.execute({"code": "print('hello')"}, context)
```

### 3. Full Workflow

```python
from HoloLoom.departments.proto.abilities.registry import AbilityRegistry
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility
from HoloLoom.departments.proto.abilities.protocol import AbilityContext, AbilityTrustLevel

# Setup
registry = AbilityRegistry()
registry.register(CodeExecutionAbility())

# Retrieve
ability = registry.get("code_execution")

# Create context
context = AbilityContext(
    session_id="demo_session",
    user_confirmed=True,
    trust_level=AbilityTrustLevel.VERIFIED
)

# Check preflight
preflight = await ability.preflight(context)
if not preflight.can_execute:
    print(f"Cannot execute: {preflight.reason}")
    return

# Execute
result = await ability.execute(
    {"code": "print('Hello from registry!')"},
    context
)

if result.success:
    print(f"Success: {result.output}")
else:
    print(f"Error: {result.error}")

# Verify
verification = await ability.verify(result)
print(f"Verified: {verification.verified}")
```

## Module Integration

### Add to `__init__.py`

To make the ability discoverable, add to your package's `__init__.py`:

```python
# HoloLoom/departments/proto/abilities/core/__init__.py

from .code_execution import (
    CodeExecutionAbility,
    CodeExecutionConfig,
    create_code_execution_ability,
)

__all__ = [
    # ... existing exports ...
    "CodeExecutionAbility",
    "CodeExecutionConfig",
    "create_code_execution_ability",
]
```

### Auto-Discovery

For automatic registration of all core abilities:

```python
# HoloLoom/departments/proto/abilities/loader.py

from typing import List
from .protocol import Ability
from .core import CodeExecutionAbility

def load_core_abilities() -> List[Ability]:
    """Load all core (Tier 1 + 2) abilities."""
    return [
        # ... Tier 1 skill wrappers ...
        # Tier 2 plugins
        CodeExecutionAbility(),
    ]

def create_default_registry(max_trust_level=AbilityTrustLevel.VERIFIED):
    """Create registry with all core abilities pre-registered."""
    registry = AbilityRegistry(max_trust_level=max_trust_level)

    for ability in load_core_abilities():
        registry.register(ability)

    return registry
```

## Advanced Integration

### Custom Configuration

```python
from HoloLoom.departments.proto.abilities.core.code_execution import (
    CodeExecutionAbility,
    CodeExecutionConfig
)

# Create with custom configuration
config = CodeExecutionConfig(
    max_timeout=60.0,           # 1 minute max
    max_output_length=50_000,   # 50KB output limit
    enable_sandbox=True,
    working_directory="/tmp/work"
)

ability = CodeExecutionAbility(config)
registry.register(ability)
```

### Conditional Registration

```python
import os
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility

registry = AbilityRegistry()

# Only register in trusted environments
if os.getenv("ENVIRONMENT") == "production":
    # More restrictive config
    config = CodeExecutionConfig(max_timeout=10.0)
    ability = CodeExecutionAbility(config)
elif os.getenv("ENVIRONMENT") == "development":
    # More permissive config
    config = CodeExecutionConfig(max_timeout=60.0)
    ability = CodeExecutionAbility(config)
else:
    return  # Don't register at all

registry.register(ability)
```

### Integration with Safety Guardrails

```python
from HoloLoom.alignment import SafetyGuardrails
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility
from HoloLoom.departments.proto.abilities.protocol import AbilityContext

guardrails = SafetyGuardrails()
ability = CodeExecutionAbility()

async def safe_execute(code: str, user_id: str):
    """Execute code with safety guardrails."""

    # Gate through guardrails
    context = AbilityContext(
        session_id=user_id,
        user_confirmed=True,  # Assume user confirmed
        trust_level=AbilityTrustLevel.VERIFIED
    )

    action = "execute_python_code"
    gate_result = await guardrails.gate_action(
        action,
        {"code": code, "user_id": user_id}
    )

    if not gate_result.allowed:
        return {
            "success": False,
            "error": f"Action blocked: {gate_result.reason}",
            "risk_score": gate_result.risk_score
        }

    # Execute
    result = await ability.execute(
        {"code": code},
        context
    )

    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.duration_ms
    }
```

### Integration with Audit Trail

```python
from HoloLoom.alignment import AuditTrail
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility

audit_trail = AuditTrail()
ability = CodeExecutionAbility()

async def logged_execute(code: str, user_id: str):
    """Execute code with audit logging."""

    context = AbilityContext(
        session_id=user_id,
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    # Execute
    result = await ability.execute(
        {"code": code},
        context
    )

    # Log to audit trail
    await audit_trail.log_decision(
        query=f"Execute Python code ({len(code)} bytes)",
        action="code_execution",
        outcome="success" if result.success else "failure",
        safety_score=result.confidence,
        metadata={
            "user_id": user_id,
            "duration_ms": result.duration_ms,
            "return_code": result.metadata.get("return_code"),
            "execution_id": result.metadata.get("execution_id")
        }
    )

    return result
```

## Testing Integration

### Test Registry Integration

```python
# tests/test_code_execution_registry.py

import pytest
from HoloLoom.departments.proto.abilities.registry import AbilityRegistry
from HoloLoom.departments.proto.abilities.core.code_execution import CodeExecutionAbility

@pytest.fixture
def registry():
    """Create registry with code execution ability."""
    reg = AbilityRegistry()
    ability = CodeExecutionAbility()
    reg.register(ability)
    return reg

@pytest.mark.asyncio
async def test_registry_get(registry):
    """Test retrieving ability from registry."""
    ability = registry.get("code_execution")
    assert ability is not None
    assert ability.manifest.name == "code_execution"

@pytest.mark.asyncio
async def test_registry_execute(registry):
    """Test executing through registry."""
    ability = registry.get("code_execution")
    context = AbilityContext(session_id="test", user_confirmed=True)

    result = await ability.execute(
        {"code": "print('test')"},
        context
    )

    assert result.success
    assert "test" in result.output
```

### Mock for Testing

```python
# tests/mocks/mock_code_execution.py

from unittest.mock import AsyncMock
from HoloLoom.departments.proto.abilities.protocol import AbilityResult

def create_mock_code_execution():
    """Create mock code execution ability."""
    mock = AsyncMock()

    # Mock preflight
    mock.preflight.return_value = AsyncMock(can_execute=True)

    # Mock execute
    mock.execute.return_value = AbilityResult(
        success=True,
        output="mock output",
        confidence=0.9,
        metadata={"return_code": 0}
    )

    # Mock verify
    mock.verify.return_value = AsyncMock(verified=True)

    return mock
```

## Deployment Considerations

### Production Checklist

- [ ] Code execution ability registered with appropriate trust level
- [ ] User confirmation requirement enforced
- [ ] Timeout limits configured for your infrastructure
- [ ] Output limits set to prevent memory issues
- [ ] Audit logging enabled for compliance
- [ ] Safety guardrails integrated
- [ ] Resource limits (CPU, memory) set at OS level (optional)
- [ ] Docker sandboxing enabled (Phase 2, optional)
- [ ] Monitoring in place for execution statistics
- [ ] Error handling for failed executions

### Resource Planning

**Per Execution**:
- Memory: ~5-10MB (temporary files + subprocess overhead)
- CPU: Varies (depends on code)
- Disk: Temporary Python files (~1-10KB)

**Limits**:
- Max concurrent executions: OS process limit
- Max output per execution: 100KB (configurable)
- Max execution time: 30-300 seconds (configurable)
- Max code size: 1MB (hardcoded)

### Monitoring

```python
# Example monitoring integration

async def monitor_code_execution(ability: CodeExecutionAbility):
    """Monitor code execution ability performance."""

    print(f"Executions: {ability.execution_count}")
    print(f"Total duration: {ability.total_duration_ms:.1f}ms")
    print(f"Avg duration: {ability.total_duration_ms / ability.execution_count:.1f}ms")

    if ability.last_execution:
        print(f"Last execution: {ability.last_execution}")
```

## Troubleshooting Integration

### Ability Not Found

```python
ability = registry.get("code_execution")
if ability is None:
    print("Code execution ability not registered")
    print("Available abilities:", registry.list_abilities())
```

### Preflight Check Fails

```python
preflight = await ability.preflight(context)
if not preflight.can_execute:
    print(f"Reason: {preflight.reason}")
    for warning in preflight.warnings:
        print(f"Warning: {warning}")
```

### Execution Fails

```python
result = await ability.execute(params, context)
if not result.success:
    print(f"Error: {result.error}")
    print(f"Confidence: {result.confidence}")
    print(f"Metadata: {result.metadata}")
```

## Best Practices

1. **Always check preflight before executing**
   ```python
   preflight = await ability.preflight(context)
   if not preflight.can_execute:
       return handle_error(preflight.reason)
   ```

2. **Use proper context with user confirmation**
   ```python
   context = AbilityContext(
       user_confirmed=True,  # Essential!
       trust_level=AbilityTrustLevel.VERIFIED
   )
   ```

3. **Log execution for audit trail**
   ```python
   result = await ability.execute(params, context)
   await audit_trail.log_decision(
       action="code_execution",
       outcome="success" if result.success else "failure"
   )
   ```

4. **Handle errors gracefully**
   ```python
   try:
       result = await ability.execute(params, context)
   except Exception as e:
       logger.error(f"Code execution failed: {e}")
       return {"success": False, "error": str(e)}
   ```

5. **Verify output before using**
   ```python
   verification = await ability.verify(result)
   if not verification.verified:
       logger.warning(f"Output verification failed: {verification.issues}")
   ```

## Related Documentation

- [Code Execution Ability README](./CODE_EXECUTION_README.md)
- [Ability Protocol](../protocol.py)
- [Ability Registry](../registry.py)
- [Proto Abilities System](../../README.md)

## Support

For issues or questions:
1. Check [CODE_EXECUTION_README.md](./CODE_EXECUTION_README.md) troubleshooting section
2. Review test cases in [test_code_execution.py](./test_code_execution.py)
3. Check [Integration Guide](#integration-guide) above
4. File an issue with reproduction steps

---

**Last Updated**: 2025-12-03
**Maintainer**: HoloLoom Team
