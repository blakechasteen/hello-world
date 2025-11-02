# Exception Handling Improvement Guide

**Date**: November 2, 2025
**Status**: Guidance for future improvement
**Priority**: MEDIUM (code works, but can be improved)

## Overview

The codebase currently has **161 files** with broad `except Exception` catches. While this provides safety (systems don't crash), it can mask specific errors and make debugging harder.

## Current State

**Files with broad catches**: 161
**Critical files identified**: 11 (weaving_orchestrator.py, memory/cache.py, embedding/spectral.py, etc.)

## Recommended Approach

### Phase 1: Critical Core Files (Priority: HIGH)
Focus on files in the hot path:

1. **weaving_orchestrator.py** (6 instances)
2. **memory/cache.py** (2 instances)
3. **embedding/spectral.py** (2 instances)
4. **policy/unified.py** (1 instance)

### Phase 2: Infrastructure Files (Priority: MEDIUM)
5. memory/backend_factory.py
6. memory/neo4j_graph.py
7. memory/hyperspace_backend.py

### Phase 3: Supporting Files (Priority: LOW)
8. All other files (150+ files)

## Exception Hierarchy Guide

### Common Exceptions to Use

#### I/O Operations
```python
# INSTEAD OF:
try:
    with open(file_path) as f:
        data = f.read()
except Exception as e:
    logger.error(f"Error: {e}")

# USE:
try:
    with open(file_path) as f:
        data = f.read()
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
except PermissionError:
    logger.error(f"Permission denied: {file_path}")
except IOError as e:
    logger.error(f"I/O error reading {file_path}: {e}")
```

#### Network Operations
```python
# INSTEAD OF:
try:
    response = requests.get(url)
except Exception as e:
    logger.error(f"Error: {e}")

# USE:
try:
    response = requests.get(url)
except requests.ConnectionError:
    logger.error(f"Connection failed to {url}")
except requests.Timeout:
    logger.error(f"Request timed out to {url}")
except requests.RequestException as e:
    logger.error(f"Request error: {e}")
```

#### Type Operations
```python
# INSTEAD OF:
try:
    value = int(user_input)
except Exception as e:
    logger.error(f"Error: {e}")

# USE:
try:
    value = int(user_input)
except ValueError as e:
    logger.error(f"Invalid integer: {user_input}")
except TypeError as e:
    logger.error(f"Type error: {e}")
```

#### Async Operations
```python
# INSTEAD OF:
try:
    result = await some_async_function()
except Exception as e:
    logger.error(f"Error: {e}")

# USE:
try:
    result = await some_async_function()
except asyncio.TimeoutError:
    logger.error("Operation timed out")
except asyncio.CancelledError:
    logger.error("Operation cancelled")
except RuntimeError as e:
    logger.error(f"Runtime error: {e}")
```

#### Import Operations
```python
# INSTEAD OF:
try:
    import optional_dependency
except Exception:
    optional_dependency = None

# USE:
try:
    import optional_dependency
except ImportError:
    optional_dependency = None
    logger.warning("Optional dependency not available, using fallback")
except ModuleNotFoundError:
    optional_dependency = None
    logger.warning("Module not found, using fallback")
```

## Pattern: Specific Then General

Always catch specific exceptions first, then fall back to general:

```python
try:
    result = risky_operation()
except SpecificError as e:
    # Handle specific case
    logger.error(f"Specific error: {e}")
    result = fallback_value
except AnotherSpecificError as e:
    # Handle another specific case
    logger.error(f"Another error: {e}")
    result = different_fallback
except Exception as e:
    # Only catch general exception as last resort
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise  # Re-raise if truly unexpected
```

## Example Improvements

### Before: weaving_orchestrator.py (lines 1352-1366)
```python
try:
    action_plan = await asyncio.wait_for(
        policy.decide(features=features, context=context),
        timeout=2.0
    )
except Exception as e:
    # Too broad!
    self.logger.error(f"Error: {e}")
    action_plan = fallback_plan
```

### After: Specific Exception Handling
```python
try:
    action_plan = await asyncio.wait_for(
        policy.decide(features=features, context=context),
        timeout=2.0
    )
except asyncio.TimeoutError:
    self.logger.error("Policy decision timed out after 2.0s, using safe default")
    action_plan = ActionPlan(
        tool="answer",
        confidence=0.5,
        tool_probs={"answer": 1.0},
        metadata={"timeout": True, "fallback": True}
    )
except RuntimeError as e:
    self.logger.error(f"Runtime error in policy: {e}")
    action_plan = fallback_plan
except Exception as e:
    self.logger.error(f"Unexpected error in policy: {e}", exc_info=True)
    raise  # Re-raise unexpected errors
```

## When to Use Broad Catches

Broad `except Exception` is acceptable in these scenarios:

1. **Top-level error handlers** (main loops, API endpoints)
2. **Cleanup/finally blocks** (ensure cleanup always runs)
3. **Logging wrappers** (log all errors without crashing)
4. **Plugin systems** (don't trust third-party code)

Example of acceptable broad catch:
```python
# Top-level API endpoint
@app.route("/query")
async def handle_query():
    try:
        result = await process_query(request.json)
        return result
    except Exception as e:
        # Catch everything at API boundary
        logger.error(f"API error: {e}", exc_info=True)
        return {"error": "Internal server error"}, 500
```

## Testing Exception Handling

Add tests for exception paths:

```python
def test_handles_file_not_found():
    """Should handle missing file gracefully."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.json")

def test_handles_timeout():
    """Should handle timeout gracefully."""
    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        result = process_query(query)
        assert result.metadata['timeout'] is True
```

## Migration Strategy

### Step 1: Audit (DONE)
- ✅ Identified 161 files with broad catches
- ✅ Prioritized 11 critical files

### Step 2: Core Files (RECOMMENDED)
- ⬜ Fix weaving_orchestrator.py (6 instances)
- ⬜ Fix memory/cache.py (2 instances)
- ⬜ Fix embedding/spectral.py (2 instances)
- ⬜ Fix policy/unified.py (1 instance)

### Step 3: Infrastructure (OPTIONAL)
- ⬜ Fix memory backend files (3-5 instances each)

### Step 4: Bulk Improvement (FUTURE)
- ⬜ Create linting rule to flag new broad catches
- ⬜ Gradual improvement of remaining 150 files

## Tools and Automation

### Find broad catches:
```bash
grep -r "except Exception" HoloLoom/ | wc -l
```

### Find specific files:
```bash
grep -l "except Exception" HoloLoom/*.py
```

### Suggested pylint rule:
```python
# .pylintrc
[MESSAGES CONTROL]
enable=broad-except
```

## Conclusion

While broad exception catches are currently widespread (161 files), the system works reliably due to:
1. ✅ Logging at all exception points
2. ✅ Graceful fallbacks in critical paths
3. ✅ Comprehensive E2E testing (139 tests)

**Recommendation**: Improve core files (11 files) first, then gradually improve others over time. Current broad catches are **safe but not optimal**.

**Priority**: MEDIUM (system works, but can be improved for better debugging)

---

**Status**: Guidance document complete
**Immediate Action Required**: NONE (system stable)
**Future Improvement**: Yes (when time permits)
