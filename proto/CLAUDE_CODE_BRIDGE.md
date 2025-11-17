# Claude Code Bridge - Complete Documentation

**Status**: ✅ Complete (November 2025)
**Location**: `proto/bot/claude_code_bridge.py`
**Integration**: `proto/bot/claude_code_commands.py`
**Tests**: `proto/test_claude_code_bridge.py`

---

## Overview

The **Claude Code Bridge** integrates Claude Code CLI (claude.ai/code) with Proto, enabling AI-powered code analysis, refactoring, and generation directly from Matrix chat.

### Core Philosophy

> "AI code assistance through conversation - review, refactor, explain, and implement via chat."

### Key Features

- ✅ **Code Review** - Security, performance, and quality analysis
- ✅ **Refactoring** - Extract methods, simplify logic, optimize performance
- ✅ **Code Explanation** - Natural language explanations with Q&A
- ✅ **Implementation** - Generate code from descriptions
- ✅ **Test Generation** - Automatic unit test creation
- ✅ **Async Execution** - Non-blocking subprocess integration
- ✅ **Error Handling** - Comprehensive timeout and error management
- ✅ **Matrix Formatting** - Beautiful responses for chat display

---

## Architecture

```
Matrix Chat (@proto code-review src/auth.py)
    ↓
ClaudeCodeCommands (parse command)
    ↓
ClaudeCodeBridge (execute via subprocess)
    ↓
Claude Code CLI (local or cloud)
    ↓
ClaudeResponse (structured result)
    ↓
Matrix Chat (formatted response)
```

### Components

**1. ClaudeCodeBridge** (`claude_code_bridge.py`)
- Core subprocess integration
- Command execution with timeouts
- Response parsing and structuring
- Health checking and availability

**2. ClaudeCodeCommands** (`claude_code_commands.py`)
- Matrix command handlers
- Syntax parsing and validation
- Context management
- Help system

**3. Test Suite** (`test_claude_code_bridge.py`)
- Health check verification
- Command testing
- Integration validation
- Full workflow simulation

---

## Usage

### Basic Commands

#### Code Review

```
@proto code-review src/auth.py
@proto code-review src/auth.py security
@proto code-review src/auth.py performance
```

**Focus Options**:
- `security` - SQL injection, XSS, hardcoded secrets
- `performance` - N+1 queries, inefficient loops
- `style` - Naming, formatting, conventions
- `readability` - Complexity, documentation

**Example Response**:
```
🔍 Claude Code Review

Security Issues (2):
1. SQL Injection Risk (HIGH) - Line 42
   f"SELECT * FROM users WHERE id={user_id}"
   → Use parameterized queries

2. Hardcoded Secret (CRITICAL) - Line 67
   API_KEY = "sk-abc123..."
   → Move to environment variables

Code Quality:
• Complexity: 6.2 (target: <5)
• Test coverage: 67% (target: 80%)
• Type hints: 85% complete

Suggestions:
1. Split auth_handler() into smaller functions
2. Add input validation for email format
3. Implement rate limiting

Completed in 2.3s
```

#### Refactoring

```
@proto refactor extract_method src/utils.py:42-67
@proto refactor simplify src/complex_logic.py
@proto refactor optimize src/slow_function.py
```

**Refactoring Patterns**:
- `extract_method` - Extract code block into separate method
- `simplify` - Reduce complexity, remove duplication
- `rename` - Suggest better variable/function names
- `optimize` - Performance improvements

**Example**:
```
User: @proto refactor extract_method src/utils.py:42-67

Proto: 🔧 Claude Code Refactor

Suggested Refactoring:

# Before (lines 42-67):
def process_data(data):
    # 25 lines of complex logic
    ...

# After:
def process_data(data):
    validated = _validate_input(data)
    transformed = _transform_data(validated)
    return _format_output(transformed)

def _validate_input(data):
    # Extracted validation logic (8 lines)
    ...

def _transform_data(data):
    # Extracted transformation (10 lines)
    ...

def _format_output(data):
    # Extracted formatting (7 lines)
    ...

Benefits:
• Reduced complexity: 12.5 → 4.2
• Improved testability (3 unit-testable functions)
• Better readability and maintainability

Completed in 1.8s
```

#### Code Explanation

```
@proto explain src/auth.py
@proto explain src/auth.py How does JWT validation work?
@proto explain src/algorithm.py What's the time complexity?
```

**Example**:
```
User: @proto explain src/auth.py How does JWT validation work?

Proto: 📚 Claude Code Explanation

JWT Validation Process (lines 45-89):

1. Token Extraction (line 45-52):
   - Extracts JWT from Authorization header
   - Removes "Bearer " prefix
   - Handles missing/malformed headers

2. Signature Verification (line 54-67):
   - Decodes JWT using secret key
   - Verifies signature with HS256 algorithm
   - Checks token hasn't been tampered with

3. Expiration Check (line 69-75):
   - Extracts 'exp' claim from payload
   - Compares with current Unix timestamp
   - Rejects if expired

4. User Lookup (line 77-89):
   - Extracts 'user_id' from JWT payload
   - Queries database for user record
   - Attaches user object to request context

Security Considerations:
• Secret key stored in environment (✓)
• Constant-time comparison prevents timing attacks (✓)
• Refresh token rotation every 7 days (✓)

Time Complexity: O(1) for verification + O(1) DB lookup
Space Complexity: O(1)

Completed in 1.5s
```

#### Implementation

```
@proto implement Add JWT authentication middleware
@proto implement Create user registration endpoint
@proto implement Add rate limiting to API endpoints
```

**Example**:
```
User: @proto implement Add JWT authentication middleware

Proto: ⚙️ Claude Code Implementation

Suggested Implementation:

```python
# middleware/auth.py
import jwt
from functools import wraps
from flask import request, jsonify
import os

def require_auth(f):
    """JWT authentication middleware"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Extract token
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        if not token:
            return jsonify({'error': 'Missing token'}), 401

        try:
            # Verify token
            payload = jwt.decode(
                token,
                os.getenv('JWT_SECRET'),
                algorithms=['HS256']
            )

            # Attach user to request
            request.user_id = payload['user_id']

        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)

    return decorated


# Usage example:
@app.route('/protected')
@require_auth
def protected_endpoint():
    return jsonify({'user_id': request.user_id})
```

Integration Steps:
1. Add to requirements.txt: PyJWT==2.8.0
2. Set JWT_SECRET environment variable
3. Apply @require_auth decorator to protected routes
4. Add token refresh endpoint

Testing:
```bash
# Valid request
curl -H "Authorization: Bearer <token>" /protected

# Missing token (401)
curl /protected

# Expired token (401)
curl -H "Authorization: Bearer <expired_token>" /protected
```

Completed in 2.7s
```

#### Test Generation

```
@proto test-gen src/auth.py
@proto test-gen src/auth.py pytest
@proto test-gen src/api.py unittest
```

**Example**:
```
User: @proto test-gen src/auth.py pytest

Proto: 🧪 Claude Code Test Generation

Generated Tests (pytest):

```python
# tests/test_auth.py
import pytest
from src.auth import validate_jwt, AuthenticationError

def test_validate_jwt_success():
    """Test JWT validation with valid token"""
    token = generate_test_token(user_id=123, exp=now() + 3600)
    result = validate_jwt(token)

    assert result['user_id'] == 123
    assert result['valid'] is True


def test_validate_jwt_expired():
    """Test JWT validation with expired token"""
    token = generate_test_token(user_id=123, exp=now() - 3600)

    with pytest.raises(AuthenticationError, match='Token expired'):
        validate_jwt(token)


def test_validate_jwt_invalid_signature():
    """Test JWT validation with tampered token"""
    token = generate_test_token(user_id=123)
    tampered = token[:-10] + 'xxxxxxxxxx'

    with pytest.raises(AuthenticationError, match='Invalid signature'):
        validate_jwt(tampered)


def test_validate_jwt_missing_claims():
    """Test JWT validation with missing required claims"""
    token = generate_test_token(user_id=None)  # Missing user_id

    with pytest.raises(AuthenticationError, match='Missing required claim'):
        validate_jwt(token)


@pytest.fixture
def generate_test_token():
    """Fixture to generate test JWT tokens"""
    def _generate(user_id=123, exp=None):
        import jwt
        payload = {'user_id': user_id, 'exp': exp or (now() + 3600)}
        return jwt.encode(payload, 'test-secret', algorithm='HS256')
    return _generate
```

Coverage: 95% (19/20 lines)
Test cases: 4 core + 1 fixture

Run with:
```bash
pytest tests/test_auth.py -v
```

Completed in 2.1s
```

---

## Programmatic Usage

### Python API

```python
from proto.bot.claude_code_bridge import ClaudeCodeBridge
import asyncio

async def main():
    # Initialize bridge
    bridge = ClaudeCodeBridge(
        repo_path='/path/to/repo',
        default_timeout=300
    )

    # Code review
    result = await bridge.code_review('src/auth.py', focus='security')
    print(result.format_for_matrix())

    # Refactoring
    result = await bridge.refactor('extract_method', 'src/utils.py:42-67')
    print(result.output)

    # Explanation
    result = await bridge.explain('src/auth.py', 'How does JWT work?')
    print(result.output)

    # Implementation
    result = await bridge.implement('Add rate limiting middleware')
    if result.success:
        print(result.output)

    # Test generation
    result = await bridge.generate_tests('src/auth.py', test_framework='pytest')
    print(result.output)

asyncio.run(main())
```

### Matrix Integration

```python
from proto.bot.claude_code_commands import ClaudeCodeCommands, CommandContext

# In your Matrix bot:
commands = ClaudeCodeCommands(repo_path='/path/to/repo')

async def on_message(room, event):
    message = event.body

    if '@proto code-review' in message:
        context = CommandContext(
            user_id=event.sender,
            room_id=room.room_id,
            message=message
        )
        response = await commands.handle_code_review(context)
        await bot.send_message(room, response)

    elif '@proto refactor' in message:
        context = CommandContext(
            user_id=event.sender,
            room_id=room.room_id,
            message=message
        )
        response = await commands.handle_refactor(context)
        await bot.send_message(room, response)

    # ... other commands
```

---

## Testing

### Run Test Suite

```bash
# Full test suite
python proto/test_claude_code_bridge.py

# Health check only
python proto/test_claude_code_bridge.py --health-only

# Test specific command
python proto/test_claude_code_bridge.py --command code-review --file src/auth.py
```

### Test Output

```
🧪 Claude Code Bridge Test Suite

Testing Proto's Claude Code integration
======================================================================

======================================================================
TEST 1: Claude Code Health Check
======================================================================

Claude Code Available: True
Claude Path: claude
Repo Path: /home/user/hello-world

Version: Claude Code v1.2.3

✅ Claude Code is ready!

======================================================================
TEST 2: Code Review
======================================================================

Created test file: test_sample.py
Contents:
def authenticate_user(username, password):
    # SQL injection vulnerable!
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)

Requesting code review...
(This may take 1-3 minutes)

🔍 Claude Code Review

Security Issues (2):
1. SQL Injection (CRITICAL) - Line 3
2. Plaintext Password Storage (HIGH) - Line 3

Suggestions:
1. Use parameterized queries
2. Hash passwords with bcrypt
3. Add input validation

Completed in 2.3s

======================================================================
TEST SUMMARY
======================================================================

✅ PASS - Health Check
✅ PASS - Code Review
✅ PASS - Explain
✅ PASS - Command Integration
✅ PASS - Full Workflow

5/5 tests passed (100%)

🎉 All tests passed! Claude Code bridge is ready.
```

---

## Performance

| Operation | Typical Latency | Timeout |
|-----------|-----------------|---------|
| **Code Review** | 1-3 minutes | 5 minutes |
| **Refactor** | 30-90 seconds | 5 minutes |
| **Explain** | 20-60 seconds | 5 minutes |
| **Implement** | 1-2 minutes | 5 minutes |
| **Test Gen** | 1-2 minutes | 5 minutes |
| **Health Check** | <1 second | 5 seconds |

**Note**: Latency varies based on:
- File size
- Complexity of code
- Network conditions (if using cloud)
- Claude Code CLI version

---

## Error Handling

### Timeout Management

```python
# Custom timeout for long operations
result = await bridge.code_review(
    'large_file.py',
    timeout=600  # 10 minutes
)
```

### Error Recovery

```python
result = await bridge.code_review('src/auth.py')

if not result.success:
    # Handle error
    print(f"Error: {result.error}")

    # Possible errors:
    # - "Command timed out after 300s"
    # - "Claude Code CLI not found"
    # - "File not found: src/auth.py"
    # - "Exit code: 1"
else:
    # Success
    print(result.output)
```

### Graceful Degradation

```python
# Check availability before use
health = await bridge.health_check()

if not health['available']:
    # Fallback to alternative or inform user
    await bot.send_message(
        room,
        "⚠️ Claude Code not available. Install from https://claude.ai/code"
    )
else:
    # Proceed with command
    result = await bridge.code_review(file_path)
```

---

## Configuration

### Environment Variables

```bash
# Optional: Custom Claude Code path
export CLAUDE_PATH=/path/to/claude

# Optional: Default repository path
export PROTO_REPO_PATH=/path/to/repo

# Optional: Default timeout (seconds)
export CLAUDE_TIMEOUT=300
```

### Python Configuration

```python
bridge = ClaudeCodeBridge(
    claude_path='claude',              # Claude executable path
    repo_path='/path/to/repo',         # Repository root
    default_timeout=300,               # 5 minutes default
    max_output_lines=1000              # Truncate long output
)
```

---

## Security Considerations

### File Access

The bridge includes security checks:

```python
# Security: Check file is within repo
try:
    full_path.resolve().relative_to(self.repo_path.resolve())
except ValueError:
    raise ValueError(f"File outside repository: {file_path}")
```

**Protections**:
- ✅ Files must be within repository root
- ✅ No directory traversal (../ blocked)
- ✅ Symbolic link validation
- ✅ Read-only operations (no file modification)

### Command Injection

**Safe subprocess execution**:
```python
# BAD: Shell injection vulnerable
subprocess.run(f"claude review {user_input}", shell=True)

# GOOD: List-based arguments (used by bridge)
subprocess.run(['claude', 'review', user_input], shell=False)
```

### Rate Limiting

Consider adding rate limits in production:

```python
# Example: Rate limit per user
from collections import defaultdict
from time import time

class RateLimitedBridge:
    def __init__(self):
        self.bridge = ClaudeCodeBridge()
        self.last_request = defaultdict(lambda: 0)
        self.min_interval = 30  # 30 seconds between requests

    async def code_review(self, file_path, user_id):
        now = time()
        last = self.last_request[user_id]

        if now - last < self.min_interval:
            raise Exception(f"Rate limit: Wait {self.min_interval - (now - last):.0f}s")

        self.last_request[user_id] = now
        return await self.bridge.code_review(file_path)
```

---

## Troubleshooting

### Claude Code Not Found

**Error**: `Claude Code CLI not found`

**Solutions**:
1. Install Claude Code: https://claude.ai/code
2. Verify installation: `claude --version`
3. Check PATH: `which claude`
4. Set custom path: `CLAUDE_PATH=/path/to/claude`

### Timeout Errors

**Error**: `Command timed out after 300s`

**Solutions**:
1. Increase timeout: `timeout=600`
2. Review smaller files
3. Use focus parameter to narrow scope
4. Check network connectivity (if using cloud)

### Permission Denied

**Error**: `Permission denied: src/secret.py`

**Solutions**:
1. Check file permissions: `ls -la src/secret.py`
2. Verify repo_path is correct
3. Ensure file is within repository root

---

## Future Enhancements

### Planned Features

- **Streaming Responses** - Show progress as Claude Code works
- **Batch Operations** - Review multiple files at once
- **Context Caching** - Speed up repeated queries
- **Custom Rules** - Team-specific linting rules
- **Integration Tests** - Generate integration tests, not just unit
- **Auto-Fix** - Automatically apply suggested fixes
- **Diff Preview** - Show before/after for refactorings

### API Improvements

```python
# Future: Streaming responses
async for chunk in bridge.code_review_stream('large_file.py'):
    await bot.send_typing(room)  # Show typing indicator
    print(chunk)

# Future: Batch operations
results = await bridge.code_review_batch([
    'src/auth.py',
    'src/api.py',
    'src/utils.py'
])

# Future: Auto-fix
result = await bridge.code_review('src/auth.py', auto_fix=True)
if result.fixes_applied:
    await git.commit("fix: Apply Claude Code suggestions")
```

---

## Integration with Proto Ecosystem

### HoloLoom Memory

Store code review results in knowledge graph:

```python
from HoloLoom import HoloLoom

async def review_and_store(file_path):
    # Review code
    result = await bridge.code_review(file_path)

    # Store in HoloLoom
    async with HoloLoom() as loom:
        await loom.experience({
            'type': 'code_review',
            'file': file_path,
            'issues': result.output,
            'timestamp': result.timestamp,
            'confidence': 1.0 if result.success else 0.0
        })

    return result
```

### Trough & xTerminator Integration

Combine with Trough's detection:

```python
from HoloLoom.departments import get_department

# Trough detects issues
qa_dept = get_department("quality_assurance")
trough_issues = await qa_dept.process({'file': 'src/auth.py'})

# Claude Code provides detailed analysis
claude_result = await bridge.code_review('src/auth.py', focus='security')

# Combined results
all_issues = trough_issues + parse_claude_issues(claude_result)
```

### Git Integration

Auto-commit fixes:

```python
# Review → Fix → Commit workflow
result = await bridge.code_review('src/auth.py')

if result.success and has_auto_fixes(result):
    # Apply fixes (future feature)
    await apply_fixes(result)

    # Commit with Claude's suggestions
    await git_handler.commit(f"fix: {summarize_fixes(result)}")
```

---

## Links

- **Proto Vision**: [proto/PROTO_VISION.md](PROTO_VISION.md)
- **Main README**: [proto/README.md](README.md)
- **Bridge Code**: [proto/bot/claude_code_bridge.py](bot/claude_code_bridge.py)
- **Commands**: [proto/bot/claude_code_commands.py](bot/claude_code_commands.py)
- **Tests**: [proto/test_claude_code_bridge.py](test_claude_code_bridge.py)
- **Claude Code**: https://claude.ai/code

---

**Built for Proto - Conversational Intelligence Hub**
**Last Updated**: November 17, 2025
