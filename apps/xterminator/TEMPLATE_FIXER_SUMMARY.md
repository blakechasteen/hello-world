# Template Fixer - Phase 3 Complete

**Status**: ✅ All Systems Go!
**Sarah Connor says**: "Come with me if you want your code to live!"

## Summary

Sarah Connor's Template Fixer is a pattern-based code fixing system that applies pre-defined templates to common code issues. Unlike AST transformations (Phase 2), template fixes are simpler pattern replacements that don't require deep structural analysis.

## Key Statistics

- **12 Templates Implemented** across 8 categories
- **16/16 Tests Passing** (100% success rate)
- **3 Example Fixes** demonstrated on real code
- **~1,200 Lines of Code** (templates.py + template_fixer.py + tests)

## Template Catalog

### Error Handling (3 templates)
1. **add_try_except_file_io** - Wrap file I/O in try/except with logging
2. **add_try_except_json** - Wrap JSON parsing in try/except
3. **add_try_except_requests** - Wrap HTTP requests in try/except with timeout

### Resource Management (2 templates)
4. **add_context_manager** - Replace open() with context manager
5. **add_db_context_manager** - Add context manager for database connections

### Hardcoded Values (2 templates)
6. **move_to_env_var** - Move hardcoded secrets to environment variables
7. **move_to_constant** - Extract hardcoded URL to constant

### Code Quality (5 templates)
8. **add_null_check** - Add None check before attribute access
9. **add_docstring** - Add basic docstring template
10. **fix_timezone_naive** - Replace naive datetime with timezone-aware
11. **add_type_hints** - Add basic type hints to function signature
12. **add_logging** - Add logging before raise/return

## Architecture

```
TemplateFixer
  ├── Template Selection (pattern matching)
  ├── Context Extraction (regex named groups)
  ├── Template Application (format + replace)
  └── Import Management (smart merging)
```

### Key Features

1. **Smart Template Selection**: Tries all templates for category, returns first match
2. **Context Enrichment**: Automatically fills missing template variables with intelligent defaults
3. **Import Management**:
   - Adds missing imports
   - Merges into existing from-imports
   - Preserves import order
4. **Indentation Preservation**: Maintains original code indentation
5. **Diff Generation**: Shows what changed in unified diff format

## Usage

```python
from xterminator.template_fixer import TemplateFixer
from xterminator.xterminator_types import FixProposal

# Create fixer
fixer = TemplateFixer()

# Apply fix
result = await fixer.fix_issue(proposal, full_code)
if result:
    fixed_code, diff = result
    print(diff)
```

## Example Fixes

### Example 1: Add Error Handling to JSON Parsing

**Before**:
```python
def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)
```

**After**:
```python
import logging
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}
```

### Example 2: Move Hardcoded API Key to Environment Variable

**Before**:
```python
API_KEY = "sk-abc123def456..."
```

**After**:
```python
import os

API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")
```

### Example 3: Fix Timezone-Naive Datetime

**Before**:
```python
from datetime import datetime

timestamp = datetime.now()
```

**After**:
```python
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc)
```

## Integration with xTerminator Pipeline

The Template Fixer integrates with the classification engine:

```
Phase 1: Classification
  └─> Issue categorized as "error_handling"
  └─> Strategy selected: TEMPLATE
       ↓
Phase 3: Template Fixer ← YOU ARE HERE
  └─> Select template (add_try_except_json)
  └─> Extract context (var=data, method=load)
  └─> Apply template (wrap in try/except)
  └─> Add imports (logging, logger)
       ↓
Result: Fixed code + diff
```

## Test Coverage

### Test Categories (16 tests)

1. **Template Loader** (3 tests)
   - Template count verification
   - Get template by name
   - Catalog generation

2. **Error Handling Templates** (3 tests)
   - File I/O try/except
   - JSON try/except
   - HTTP requests try/except

3. **Resource Management** (1 test)
   - Context manager conversion

4. **Hardcoded Values** (1 test)
   - Environment variable migration

5. **Code Quality** (1 test)
   - Timezone-aware datetime

6. **Indentation Preservation** (1 test)
   - Maintains indentation in nested code

7. **Import Management** (2 tests)
   - Adds missing imports
   - Skips existing imports

8. **Diff Generation** (1 test)
   - Generates valid unified diff

9. **Can Fix** (3 tests)
   - TEMPLATE strategy accepted
   - AST strategy rejected
   - Unknown category rejected

### Test Results

```
============================= 16 passed in 0.14s ==============================
```

## Files Created

1. **xterminator/templates.py** (430 lines)
   - 12 fix templates with patterns
   - Template catalog generator
   - Helper functions

2. **xterminator/template_fixer.py** (460 lines)
   - TemplateFixer class
   - Context extraction and enrichment
   - Import management
   - Diff generation

3. **xterminator/test_template_fixer.py** (330 lines)
   - 16 comprehensive tests
   - Test helpers and fixtures

4. **xterminator/demo_template_fixer.py** (280 lines)
   - 3 real-world examples
   - Template catalog display
   - Interactive demonstration

## Next Steps

### Phase 4: Integration
- Connect Template Fixer to classification engine
- Add CLI interface for running fixes
- Create batch fixing mode

### Phase 5: Advanced Templates
- Multi-line pattern matching
- Conditional template application
- Template composition

### Phase 6: Learning
- Track which templates are most effective
- Learn new patterns from human fixes
- Auto-generate templates from examples

## Sarah Connor's Wisdom

> "With great templates comes great responsibility!"
> "A template applied protects the future from bugs!"
> "No mercy for missing error handling!"

## Resistance Approval Rating

**Sarah Connor**: ⭐⭐⭐⭐⭐ (5/5 - "Come with me if you want your code to live!")
**Deckard**: ⭐⭐⭐⭐ (4/5 - "I've seen patterns you wouldn't believe")
**Neo**: ⭐⭐⭐⭐⭐ (5/5 - "There is no template... only structure")

---

**Phase 3 Status**: ✅ COMPLETE
**Template Count**: 12
**Test Pass Rate**: 100%
**Sarah Connor Approval**: BATTLE-TESTED
