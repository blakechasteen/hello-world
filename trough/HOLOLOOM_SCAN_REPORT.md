# Trough Scan Report - HoloLoom Codebase

**Date**: 2025-11-12 23:16
**Files Scanned**: 50
**Total Issues**: 1364
**Files with Issues**: 50

## Issues by Severity

- **CRITICAL**: 11
- **HIGH**: 11
- **MEDIUM**: 913
- **LOW**: 284

## Issues by Category

- **copy_paste**: 560
- **error_handling**: 293
- **hardcoded_values**: 226
- **timezone**: 52
- **dead_code**: 38
- **documentation**: 22
- **incomplete**: 16
- **security**: 9
- **performance**: 3

## Top 10 Files with Most Issues

1. **weaving_orchestrator.py** - 137 issues
   - Line 344: [critical] security
   - Line 345: [critical] security
   - Line 346: [critical] security

2. **alignment/mcp_department_registry.py** - 121 issues
   - Line 115: [medium] copy_paste
   - Line 116: [medium] copy_paste
   - Line 122: [medium] copy_paste

3. **agentic/code_verification.py** - 92 issues
   - Line 27: [medium] copy_paste
   - Line 28: [medium] copy_paste
   - Line 29: [medium] copy_paste

4. **agents/profiles.py** - 68 issues
   - Line 41: [medium] copy_paste
   - Line 42: [medium] copy_paste
   - Line 90: [medium] copy_paste

5. **agents/adversarial_agents.py** - 56 issues
   - Line 163: [medium] copy_paste
   - Line 164: [medium] copy_paste
   - Line 165: [medium] copy_paste

6. **agentic/memory_tools.py** - 50 issues
   - Line 20: [critical] security
   - Line 159: [medium] timezone
   - Line 249: [medium] timezone

7. **config.py** - 43 issues
   - Line 422: [medium] copy_paste
   - Line 423: [medium] copy_paste
   - Line 424: [medium] copy_paste

8. **agentic/web_researcher.py** - 39 issues
   - Line 203: [medium] timezone
   - Line 266: [medium] error_handling
   - Line 267: [medium] error_handling

9. **alignment/human_in_loop.py** - 39 issues
   - Line 330: [high] error_handling
   - Line 457: [high] error_handling
   - Line 84: [medium] copy_paste

10. **alignment/monitoring.py** - 39 issues
   - Line 71: [medium] copy_paste
   - Line 72: [medium] copy_paste
   - Line 161: [medium] timezone

## Example Issues Found

### 1. weaving_orchestrator.py:344
**Severity**: CRITICAL
**Category**: security
**Issue**: Potential SQL injection vulnerability (use parameterized queries)
**Fix**: Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))

### 2. weaving_orchestrator.py:345
**Severity**: CRITICAL
**Category**: security
**Issue**: Potential SQL injection vulnerability (use parameterized queries)
**Fix**: Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))

### 3. weaving_orchestrator.py:346
**Severity**: CRITICAL
**Category**: security
**Issue**: Potential SQL injection vulnerability (use parameterized queries)
**Fix**: Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))

### 4. weaving_orchestrator.py:1246
**Severity**: CRITICAL
**Category**: security
**Issue**: Potential SQL injection vulnerability (use parameterized queries)
**Fix**: Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))

### 5. weaving_orchestrator.py:1249
**Severity**: CRITICAL
**Category**: security
**Issue**: Potential SQL injection vulnerability (use parameterized queries)
**Fix**: Use cursor.execute('SELECT * FROM table WHERE id = ?', (user_id,))
