# Claude Code Bridge - Testing Guide

**Last Updated**: November 17, 2025
**Test Status**: ✅ ALL PASSING (15/15)
**Maintainer**: Proto Development Team

---

## Quick Start

### Run All Tests
```bash
cd /home/user/hello-world/proto
python test_claude_code_bridge.py
```

### Check if Claude Code is Available
```bash
python test_claude_code_bridge.py --health-only
```

### Expected Output (Claude Code Available)
```
✅ Claude Code is ready!
Version: claude version X.Y.Z

✅ PASS - Health Check
✅ PASS - Code Review
✅ PASS - Explain
✅ PASS - Command Integration
✅ PASS - Full Workflow

5/5 tests passed (100%)
🎉 All tests passed! Claude Code bridge is ready.
```

### Expected Output (Claude Code Not Available)
```
❌ Claude Code not available

To install Claude Code:
1. Visit: https://claude.ai/code
2. Follow installation instructions
3. Verify with: claude --version

⚠️ Claude Code not available - skipping integration tests
```

---

## Test Structure

### Test File Location
```
/home/user/hello-world/proto/test_claude_code_bridge.py
```

### Code Under Test
```
/home/user/hello-world/proto/bot/claude_code_bridge.py (472 lines)
/home/user/hello-world/proto/bot/claude_code_commands.py (406 lines)
```

---

## Test Categories

### Category 1: Unit Tests (7/7 Passing)
- ClaudeResponse dataclass
- ClaudeCommandType enum
- ClaudeCodeBridge initialization
- ClaudeCodeCommands initialization
- Command parsing (code-review, refactor, explain)

### Category 2: Async Tests (4/4 Passing)
- Help handler
- Invalid syntax detection
- Health check
- Valid command syntax

### Category 3: Error Handling (4/4 Passing)
- Timeout handling
- Missing file handling
- Error response formatting
- Output truncation

---

## Running Specific Tests

```bash
# Health check only
python test_claude_code_bridge.py --health-only

# Test specific command
python test_claude_code_bridge.py --command code-review --file src/auth.py
python test_claude_code_bridge.py --command explain --file src/auth.py
python test_claude_code_bridge.py --command health

# Run all tests (requires Claude Code)
python test_claude_code_bridge.py
```

---

## Test Results Summary

**Total**: 15 tests
**Passed**: 15 (100%)
**Failed**: 0
**Skipped**: 0

### Breakdown by Category
- Unit Tests: 7/7 ✅
- Async Tests: 4/4 ✅
- Error Handling: 4/4 ✅

### Breakdown by Component
- ClaudeResponse: ✅
- ClaudeCommandType: ✅
- ClaudeCodeBridge: ✅
- ClaudeCodeCommands: ✅
- Regex Parsing: ✅
- Error Handling: ✅

---

## Detailed Test List

### Unit Tests
1. ✅ ClaudeResponse dataclass - Response creation and formatting
2. ✅ ClaudeCommandType enum - All 5 command types defined
3. ✅ ClaudeCodeBridge init - Default and custom parameters
4. ✅ ClaudeCodeCommands init - Bridge integration
5. ✅ Code review parsing - File and focus parameters
6. ✅ Refactor parsing - Pattern and target parameters
7. ✅ Explain parsing - File and optional question

### Async Tests
1. ✅ Help handler - Generates complete help message
2. ✅ Invalid syntax - Catches incomplete commands
3. ✅ Health check - Reports availability status
4. ✅ Valid syntax - All 5 commands parse correctly

### Error Handling
1. ✅ Timeout handling - Gracefully handles timeouts
2. ✅ Missing command - Handles missing files
3. ✅ Error formatting - Matrix markdown formatting
4. ✅ Output truncation - Prevents message flooding

---

## Integration Requirements

Full integration tests require:
- [ ] Claude Code CLI installed (`claude --version` works)
- [ ] Network access available
- [ ] 1-3 minutes per command

Unit and error handling tests work without Claude Code.

---

## Troubleshooting

### "Claude Code not available"
- Install from https://claude.ai/code
- Verify: `claude --version`
- Restart terminal

### "Command timed out"
- Check network connectivity
- Verify Claude Code is running
- Check system resources

### "File not found"
- Use relative paths from repo root
- Check write permissions
- Verify file exists

---

## Documentation Files

- **CLAUDE_CODE_BRIDGE.md** - Main documentation (26 KB)
- **TEST_RESULTS.md** - Detailed test report (12 KB)
- **TESTING_GUIDE.md** - This file (testing instructions)

---

See CLAUDE_CODE_BRIDGE.md for complete documentation and troubleshooting.
