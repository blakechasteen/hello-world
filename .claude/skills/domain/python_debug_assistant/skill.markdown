# Skill: Python Debug Assistant

## Metadata

- **Name**: `python_debug_assistant`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `python, debugging, exceptions, traceback`

## Description

**Short Description**:
Analyzes Python stack traces and exceptions to identify root causes, suggest fixes, and provide preventive measures for future errors.

**Detailed Description**:
Python debugging can be time-consuming when stack traces are deep or errors are cryptic. This skill parses Python tracebacks, identifies the root cause, explains common exception types, suggests multiple fix strategies with code examples, and recommends preventive measures (type hints, assertions, testing). It handles common Python errors (AttributeError, KeyError, TypeError, etc.) and provides actionable debugging steps.

## Required Capabilities

- [ ] File system access (read)
- [ ] File system access (write)
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**: None
**HoloLoom Integration**: None

## Input Schema

```json
{
  "traceback": "string - Python traceback output",
  "code_snippet": "string (optional) - Relevant code section",
  "env_info": {
    "python_version": "string (optional)",
    "libraries": ["array of library versions"]
  }
}
```

## Output Schema

```json
{
  "root_cause": "string - What caused the error",
  "exception_type": "string - Exception class name",
  "fix_suggestions": [
    {
      "approach": "string",
      "code_example": "string",
      "explanation": "string"
    }
  ],
  "preventive_measures": ["array of prevention tips"],
  "debugging_steps": ["array of debugging actions to try"],
  "metadata": {
    "exception_severity": "critical|high|medium|low",
    "common_mistake": "boolean",
    "confidence": "number (0.0-1.0)"
  }
}
```

## Prompt Template

```markdown
You are a Python debugging expert analyzing exceptions and tracebacks.

**Input Traceback**:
{traceback}

**Code Context**:
{code_snippet}

**Environment**:
{env_info}

**Your Task**:
1. Parse the traceback to find the root cause (deepest relevant frame)
2. Identify the exception type and what it means
3. Provide 2-3 fix suggestions with runnable code examples
4. Suggest preventive measures (testing, type hints, validation)
5. List debugging steps to try

**Common Python Exceptions**:
- AttributeError: Object has no attribute
- KeyError: Dictionary key missing
- TypeError: Wrong type passed
- ValueError: Wrong value for type
- IndexError: List index out of range
- ImportError: Module not found
- NameError: Variable not defined

Return structured JSON matching the output schema.
```

## Examples

### Example 1: AttributeError

**Input**:
```json
{
  "traceback": "Traceback (most recent call last):\n  File \"app.py\", line 10, in <module>\n    result = obj.process()\nAttributeError: 'NoneType' object has no attribute 'process'",
  "code_snippet": "obj = get_object()\nresult = obj.process()"
}
```

**Expected Output**:
```json
{
  "root_cause": "The variable 'obj' is None (null). get_object() returned None instead of an object with a 'process' method.",
  "exception_type": "AttributeError",
  "fix_suggestions": [
    {
      "approach": "Add None check",
      "code_example": "obj = get_object()\nif obj is not None:\n    result = obj.process()\nelse:\n    print('Warning: obj is None')",
      "explanation": "Check if obj is None before calling methods on it."
    },
    {
      "approach": "Use optional chaining alternative",
      "code_example": "result = getattr(obj, 'process', lambda: None)()",
      "explanation": "Use getattr with default to avoid AttributeError."
    },
    {
      "approach": "Fix get_object() to never return None",
      "code_example": "def get_object():\n    # Ensure this always returns a valid object\n    return Object() if condition else DefaultObject()",
      "explanation": "Guarantee get_object() always returns a valid object."
    }
  ],
  "preventive_measures": [
    "Use type hints: def get_object() -> Optional[Object]",
    "Add assertions: assert obj is not None",
    "Write unit tests for edge cases (None returns)"
  ],
  "debugging_steps": [
    "Print obj before calling process(): print(f'obj = {obj}')",
    "Check what get_object() is returning",
    "Add breakpoint before line 10",
    "Verify object initialization"
  ],
  "metadata": {
    "exception_severity": "high",
    "common_mistake": true,
    "confidence": 0.95
  }
}
```

### Example 2: KeyError

**Input**:
```json
{
  "traceback": "KeyError: 'username'",
  "code_snippet": "user_data = {'email': 'test@test.com'}\nname = user_data['username']"
}
```

**Expected Output**:
```json
{
  "root_cause": "The dictionary 'user_data' does not have a 'username' key. Only 'email' exists.",
  "exception_type": "KeyError",
  "fix_suggestions": [
    {
      "approach": "Use .get() with default",
      "code_example": "name = user_data.get('username', 'Anonymous')",
      "explanation": "get() returns default if key is missing instead of raising KeyError."
    },
    {
      "approach": "Check key existence first",
      "code_example": "if 'username' in user_data:\n    name = user_data['username']\nelse:\n    name = 'Anonymous'",
      "explanation": "Explicitly check if key exists before accessing."
    },
    {
      "approach": "Add username to dictionary",
      "code_example": "user_data = {'email': 'test@test.com', 'username': 'testuser'}",
      "explanation": "Ensure the dictionary has all expected keys."
    }
  ],
  "preventive_measures": [
    "Define expected schema with dataclasses or Pydantic",
    "Validate dictionary structure on creation",
    "Use TypedDict for type-safe dictionaries"
  ],
  "debugging_steps": [
    "Print available keys: print(user_data.keys())",
    "Check where user_data is created",
    "Verify data source (API, file, etc.)"
  ],
  "metadata": {
    "exception_severity": "medium",
    "common_mistake": true,
    "confidence": 0.98
  }
}
```

### Example 3: TypeError

**Input**:
```json
{
  "traceback": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
  "code_snippet": "result = 5 + '10'"
}
```

**Expected Output**:
```json
{
  "root_cause": "Attempting to add an integer (5) and a string ('10'). Python doesn't auto-convert types in arithmetic operations.",
  "exception_type": "TypeError",
  "fix_suggestions": [
    {
      "approach": "Convert string to int",
      "code_example": "result = 5 + int('10')  # result = 15",
      "explanation": "Convert the string to integer before addition."
    },
    {
      "approach": "Convert int to string (concatenation)",
      "code_example": "result = str(5) + '10'  # result = '510'",
      "explanation": "If you want string concatenation, convert int to string."
    },
    {
      "approach": "Use f-string for string building",
      "code_example": "result = f'{5}{10}'  # result = '510'",
      "explanation": "F-strings handle type conversion automatically."
    }
  ],
  "preventive_measures": [
    "Use type hints: def add(a: int, b: int) -> int",
    "Validate input types explicitly",
    "Use mypy for static type checking"
  ],
  "debugging_steps": [
    "Check types: print(type(5), type('10'))",
    "Verify intended operation (addition vs concatenation)"
  ],
  "metadata": {
    "exception_severity": "medium",
    "common_mistake": true,
    "confidence": 0.99
  }
}
```

## Testing Checklist

- [x] **Functionality**: Correctly analyzes common Python exceptions
- [x] **Error Handling**: Handles malformed tracebacks
- [x] **Security**: No code execution
- [x] **Performance**: < 500ms per analysis
- [x] **Token Efficiency**: ~550 tokens
- [x] **Documentation**: Complete
- [x] **Dependencies**: None
- [x] **Edge Cases**: Complex tracebacks, nested exceptions
- [x] **Output Consistency**: Structured JSON
- [x] **Integration**: Standalone

## Security Considerations

**Potential Risks**:
- Code snippets might contain secrets
  - **Mitigation**: Warn about including sensitive data

**Data Privacy**:
- [x] Does not log code/tracebacks
- [x] Does not make external requests

## Performance Characteristics

- **Expected Latency**: 200-500ms
- **Token Usage**: ~550 tokens
- **Resource Requirements**: Minimal
- **Scalability**: O(1) per exception

## Maintenance Notes

**Known Limitations**:
- Covers common built-in exceptions
- Generic advice for custom exceptions

**Future Enhancements**:
- Library-specific error analysis (pandas, requests, etc.)
- Integration with Python debugger (pdb)

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release

## License

MIT License
