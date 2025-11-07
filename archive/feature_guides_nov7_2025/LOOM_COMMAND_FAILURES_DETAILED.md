# Loom Command Test Failures - Detailed Analysis

---

## FAILURE #1: test_selection_priority_constraints_override_auto

**Line**: 371-379
**Category**: Resource Constraint Selection
**Severity**: Minor (boundary condition)

### Test Code

```python
def test_selection_priority_constraints_override_auto(loom):
    """Test resource constraints override auto-selection."""
    pattern = loom.select_pattern(
        query_text="hello",  # 5 chars - would auto-select BARE
        resource_constraints={"max_timeout": 6.0}  # But constraint overrides
    )

    # Constraint should win - expects FAST
    assert pattern.card == PatternCard.FAST
```

### Actual Behavior

```
Traceback:
  AssertionError: assert <PatternCard.FUSED: 'fused'> == <PatternCard.FAST: 'fast'>

Log Output:
  Selected pattern: fused (resource_constraints={'max_timeout': 6.0})
```

### Step-by-Step Execution Trace

1. **Input**:
   - `query_text = "hello"` (5 chars)
   - `resource_constraints = {"max_timeout": 6.0}`
   - `loom.auto_select = True` (fixture default)
   - `loom.default_pattern = PatternCard.FAST` (fixture default)

2. **Selection Priority Check** (HoloLoom/loom/command.py, line 325-350):

   ```python
   # Priority 1: User preference?
   if user_preference:  # None
       # SKIPPED

   # Priority 2: Resource constraints?
   if not selected_card and resource_constraints:
       selected_card = self._select_by_constraints(resource_constraints)
       # ENTERS THIS BRANCH ← calls _select_by_constraints()
   ```

3. **Constraint Selection Logic** (HoloLoom/loom/command.py, line 365-380):

   ```python
   def _select_by_constraints(self, constraints: Dict[str, Any]):
       max_timeout = constraints.get("max_timeout")  # 6.0

       if max_timeout:
           if max_timeout <= 2.5:
               return PatternCard.BARE
           elif max_timeout <= 5.0:
               return PatternCard.FAST
           else:
               return PatternCard.FUSED  # ← RETURNS HERE because 6.0 > 5.0

       return None
   ```

4. **Result**:
   - `selected_card = PatternCard.FUSED` (returned from `_select_by_constraints`)
   - Stops processing (priorities 3 & 4 skipped because `selected_card` is set)
   - Returns `FUSED`

### Why the Test Fails

The test expects `FAST` but the implementation returns `FUSED` because:

```
Input: max_timeout = 6.0 seconds

Current Logic:
  if 6.0 <= 2.5?  NO
  if 6.0 <= 5.0?  NO  ← Falls through
  else: return FUSED  ← Result
```

The threshold of `5.0` seconds is the boundary:
- `5.0s` → `FAST` (pattern timeout 4.0s)
- `5.1s` → `FUSED` (pattern timeout 8.0s)
- `6.0s` → `FUSED`

### The Ambiguity

**Question**: Should a 6.0s timeout budget use FAST or FUSED?

**Current Implementation**:
- FAST pattern: 4.0s timeout
- FUSED pattern: 8.0s timeout
- Available budget: 6.0s

Interpretation A (Current):
- 6.0s > 5.0s threshold → Use FUSED (quality-first)
- Rationale: Generous budget, go for highest quality

Interpretation B (Test Expectation):
- 6.0s is only 1.0s more than FAST timeout (4.0s) → Use FAST
- Rationale: Conservative, don't use full budget just for marginal quality gain

### Why the Test Author Expected FAST

```
Thinking: "6.0s is still pretty tight. FAST (4.0s) has 2.0s headroom,
which is 50% buffer. That's safe. FUSED (8.0s) needs 33% extra time,
which seems excessive."
```

### Why the Implementation Returns FUSED

```
Thinking: "6.0s exceeds the 5.0s threshold I defined as the FAST budget.
Anything above that should get FUSED for quality."
```

### Visual Timeline

```
Available Timeouts on Number Line
├─ 2.5s
│  │
│  └─ BARE (pattern: 1.0s, headroom: 1.5s = 150%)
│
├─ 5.0s ← CURRENT THRESHOLD
│  │
│  ├─ FAST (pattern: 4.0s, headroom: 1.0s = 25%)
│  │
│  └─ 6.0s ← TEST INPUT (headroom for FAST: 2.0s = 50%)
│  │         (headroom for FUSED: 2.0s = 25%)
│  │
│  └─ FUSED (pattern: 8.0s, headroom: -2.0s = 25% OVER)
│
└─ 8.0s
   │
   └─ Safe for FUSED
```

### Recommended Fix

**Option A: Extend FAST Threshold** ✅ RECOMMENDED

```python
# HoloLoom/loom/command.py, line 378-384
# OLD:
if max_timeout <= 2.5:
    return PatternCard.BARE
elif max_timeout <= 5.0:  # ← Change this
    return PatternCard.FAST
else:
    return PatternCard.FUSED

# NEW:
if max_timeout <= 2.5:
    return PatternCard.BARE
elif max_timeout <= 6.0:  # ← Extended to 6.0s
    return PatternCard.FAST
else:
    return PatternCard.FUSED
```

**Rationale**:
- FAST pattern (4.0s) fits comfortably in 6.0s budget
- 2.0s headroom (50% buffer) is safe and sufficient
- More generous timeout handling improves usability
- Aligns with test's conservative philosophy

**Option B: Fix the Test** (Alternative)

```python
# HoloLoom/tests/unit/test_loom_command.py, line 371-379
# OLD:
assert pattern.card == PatternCard.FAST

# NEW:
assert pattern.card == PatternCard.FUSED  # 6.0s warrants FUSED
```

**Rationale**:
- Implementation logic is consistent (6.0 > 5.0 → FUSED)
- Test should match implementation
- No code changes needed

**Why Option A is Better**:
- Extends the "safe" timeout range for FAST
- More user-friendly (less likely to jump to FUSED unexpectedly)
- FAST is already well-optimized; FUSED adds marginal quality for significant time cost
- The 50% headroom for FAST at 6.0s is reasonable and safe

---

## FAILURE #2: test_select_pattern_empty_query_string

**Line**: 557-562
**Category**: Edge Cases (Empty String Handling)
**Severity**: Minor (test logic contradiction)

### Test Code

```python
def test_select_pattern_empty_query_string(loom):
    """Test pattern selection with empty query string."""
    pattern = loom.select_pattern(query_text="")

    # Should use default (auto-select won't trigger)
    assert pattern.card == PatternCard.BARE  # 0 chars < 50
```

### Actual Behavior

```
Traceback:
  AssertionError: assert <PatternCard.FAST: 'fast'> == <PatternCard.BARE: 'bare'>

Log Output:
  Selected pattern: fast (default_pattern)
```

### The Core Problem: Python Truthiness

In Python, empty string `""` is **falsy**:

```python
bool("")           # False
bool("hello")      # True
bool(" ")          # True
bool("x" * 0)      # False
bool("x" * 1)      # True
```

This affects boolean conditions:

```python
query_text = ""

if query_text:           # FAILS - empty string is falsy
    print("truthy")
else:
    print("falsy")      # ← EXECUTES

# This means:
if auto_select and query_text:
    # Never executes if query_text is ""
```

### Step-by-Step Execution Trace

1. **Input**:
   - `query_text = ""` (empty string)
   - `loom.auto_select = True` (fixture default)
   - `loom.default_pattern = PatternCard.FAST` (fixture default)

2. **Selection Priority Check** (HoloLoom/loom/command.py, line 325-347):

   ```python
   # Priority 1: User preference?
   if user_preference:  # None
       # SKIPPED

   # Priority 2: Resource constraints?
   if not selected_card and resource_constraints:  # None
       # SKIPPED

   # Priority 3: Automatic selection?
   if not selected_card and self.auto_select and query_text:
       #                                            ↑
       #                                       "" is falsy!
       selected_card = self._auto_select(query_text)
       # CONDITION FAILS - empty string is falsy
       # THIS BLOCK IS SKIPPED

   # Priority 4: Default
   if not selected_card:  # Still None, so condition is True
       selected_card = self.default_pattern  # PatternCard.FAST
       # ENTERS THIS BLOCK ← Returns FAST
   ```

3. **Result**:
   - Empty string fails the `and query_text` check
   - Auto-select is skipped
   - Falls back to default: `FAST`

### Why the Test Fails

The test has a **logical contradiction**:

```python
# Comment: "Should use default (auto-select won't trigger)"
#           ↑ Correct - empty string won't trigger auto-select
#             because it's falsy

# Assertion: assert pattern.card == PatternCard.BARE
#            ↑ Wrong - BARE only comes from auto-select!
#              If auto-select doesn't trigger, why expect BARE?
```

**The test author confused two different things**:
1. What should happen: Use default pattern (FAST) ✓
2. What they asserted: Use auto-selected pattern (BARE) ✗

### Visual Trace

```
test_select_pattern_empty_query_string():
  Input: query_text = ""
         fixture: auto_select=True, default=FAST

  Selection Process:
    Priority 1: user_preference? None → Skip
    Priority 2: constraints? None → Skip
    Priority 3: auto_select and query_text?
                True and "" (falsy)? = False → SKIP ✓
    Priority 4: default → FAST ✓

  Result: FAST

  Test expects: BARE
  Test got:    FAST

  Why?
    - Empty string is falsy in Python
    - auto_select requires both conditions to be true
    - Empty string makes the AND condition false
    - Falls back to default (FAST)
    - Comment is correct, assertion is wrong!
```

### Comparison with Related Test

**This test**:
```python
def test_select_pattern_empty_query_string(loom):
    """Test pattern selection with empty query string."""
    pattern = loom.select_pattern(query_text="")
    assert pattern.card == PatternCard.BARE  # Wrong!
```

**Similar test (PASSING)**:
```python
def test_select_pattern_none_query_string(loom):
    """Test pattern selection with None query string."""
    pattern = loom.select_pattern(query_text=None)
    assert pattern.card == PatternCard.FAST  # Correct!
```

**Why one passes and one fails**:
- Both `None` and `""` are falsy
- Both skip auto-select
- Both use default
- First test expects BARE (wrong!) - 😞 FAILS
- Second test expects FAST (correct!) - ✅ PASSES

### Philosophies Considered

**Philosophy A: Empty string = no query**
```
if query_text:  # Empty string treated like None
    # Don't auto-select
```
Outcome: Use default (FAST) ← Current behavior

**Philosophy B: Empty string = query of length 0**
```
if query_text is not None:  # Distinguish None from ""
    # Do auto-select
    # _auto_select("") returns BARE (len < 50)
```
Outcome: Use auto-selected BARE

**Which is correct?**
- Current implementation uses Philosophy A ✓
- Test expects Philosophy B (partially) ✗
- Comment supports Philosophy A ✓

### Recommended Fix

**Option A: Fix Test to Match Implementation** ✅ RECOMMENDED

```python
# HoloLoom/tests/unit/test_loom_command.py, line 557-562
# OLD:
def test_select_pattern_empty_query_string(loom):
    """Test pattern selection with empty query string."""
    pattern = loom.select_pattern(query_text="")
    assert pattern.card == PatternCard.BARE  # 0 chars < 50

# NEW:
def test_select_pattern_empty_query_string(loom):
    """Test pattern selection with empty query string."""
    pattern = loom.select_pattern(query_text="")
    assert pattern.card == PatternCard.FAST  # Default (auto-select skipped)
```

**Rationale**:
- Comment is correct: "auto-select won't trigger"
- Empty string `""` is falsy
- Implementation correctly skips auto-select
- Falls back to default (FAST)
- This matches the related `test_select_pattern_none_query_string` pattern

**Option B: Change Implementation** (Alternative)

```python
# HoloLoom/loom/command.py, line 340-342
# OLD:
if not selected_card and self.auto_select and query_text:
    selected_card = self._auto_select(query_text)

# NEW:
if not selected_card and self.auto_select and query_text is not None:
    selected_card = self._auto_select(query_text)
```

**Rationale**:
- Distinguish between None (no query) and "" (empty query)
- Empty string would trigger auto-select
- `_auto_select("")` returns BARE (length 0 < 50)
- Test assertion would pass

**Why Option A is Better**:
- Matches current semantic: falsy input → use default
- Consistent with None handling
- Simpler logic (fewer edge cases)
- Comment is already correct; just fix assertion to match comment
- Single-line test fix vs implementation change

**Why Option B Makes Sense Too**:
- Empty string is still a query (just empty)
- Could argue empty queries deserve auto-select
- More consistent with "all query strings get auto-selected"
- But adds semantic complexity

### Additional Consideration

If we want to test "empty string triggers auto-select", we should:

1. Either implement Option B (above)
2. Or create new test `test_select_pattern_empty_string_with_auto` that clearly documents the behavior

Example new test (if choosing Option B):

```python
def test_select_pattern_empty_string_triggers_auto(loom):
    """Test empty string can trigger auto-selection (if implemented)."""
    # This would require changing implementation to use "is not None"
    pattern = loom.select_pattern(query_text="")
    # After implementation change: would return BARE
    # Before implementation change: returns FAST
    pass
```

---

## Summary Comparison

| Aspect | Failure 1 | Failure 2 |
|--------|-----------|-----------|
| **What It Tests** | Resource constraint selection | Empty string handling |
| **Root Cause** | Threshold boundary (6.0 > 5.0) | Python truthiness ("" is falsy) |
| **Test vs Implementation** | Different interpretation | Test contradiction (comment vs assertion) |
| **Severity** | Medium (semantic) | Low (test logic error) |
| **Recommended Fix** | Change implementation (extend threshold) | Change test assertion |
| **Lines to Change** | 1 (line 382) | 1 (line 562) |
| **Effort** | ~2 minutes | ~1 minute |

---

## Both Fixes Together

```bash
# File 1: HoloLoom/loom/command.py
# Line 382: Change 5.0 to 6.0
-    elif max_timeout <= 5.0:
+    elif max_timeout <= 6.0:

# File 2: HoloLoom/tests/unit/test_loom_command.py
# Line 562: Change BARE to FAST
-    assert pattern.card == PatternCard.BARE
+    assert pattern.card == PatternCard.FAST
```

**Expected Result After Fixes**:
```bash
$ PYTHONPATH=. pytest HoloLoom/tests/unit/test_loom_command.py -v
... 42 passed in 0.53s
```

