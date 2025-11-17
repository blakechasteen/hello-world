# File Watcher Integration - Quick Start Guide

**Implementation**: Phase 5 Wave 1 - File Watcher Integration
**Date**: 2025-11-17
**Status**: ✅ Ready to Test

## What's New

File changes in your VS Code workspace now **automatically update HoloLoom's knowledge graph**. No manual indexing needed!

### Example Workflow

```
1. You save main.py
   ↓
2. VS Code detects change
   ↓
3. File added to batch (max 2 second wait)
   ↓
4. HoloLoom indexes the file
   ↓
5. Knowledge graph updated with new entities
   ↓
6. Next query finds updated information ✨
```

## Prerequisites

### 1. Start HoloLoom Server

```bash
cd HoloLoom/server
python -m uvicorn agentic_api:app --reload --port 8000
```

**Output should show**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 2. Open VS Code Extension

```bash
cd promptly-vscode
npm install          # If not already done
code .               # Open VS Code
```

### 3. Extension Settings

Configure in VS Code settings (`.vscode/settings.json` or UI):

```json
{
    "promptly.hololoomUrl": "http://localhost:8000",
    "promptly.enableFileWatcher": true
}
```

## Testing the Integration

### Test 1: Single File Change

**Steps**:
1. Open a Python file in VS Code
2. Add a new function:
   ```python
   def new_test_function():
       """This is a test."""
       pass
   ```
3. Save the file (Ctrl+S)
4. Check console: `Cmd+Shift+J` or `View → Debug Console`

**Expected Output**:
```
WorkspaceWatcher: File changed: /path/to/file.py
WorkspaceWatcher: Indexed 1 file(s), created X entities, Y TODOs found
```

### Test 2: Multiple Files in Batch

**Steps**:
1. Modify 3 different Python files
2. Save them within 2 seconds:
   - Save file1.py (at 0.5s)
   - Save file2.py (at 1.0s)
   - Save file3.py (at 1.5s)
3. Check console around 3.5s

**Expected Output**:
```
WorkspaceWatcher: Indexed 3 file(s), created X entities, Y TODOs found
```

**Verify Batching**: Only 1 API call, not 3. Check network tab.

### Test 3: New File Creation

**Steps**:
1. Create a new Python file: `new_module.py`
2. Add content:
   ```python
   class MyNewClass:
       def method(self):
           pass
   ```
3. Save immediately

**Expected Output** (in ~100ms, not 2 seconds):
```
WorkspaceWatcher: File created: /path/to/new_module.py
WorkspaceWatcher: Indexed 1 file(s), created X entities
```

### Test 4: Server Down Recovery

**Steps**:
1. Stop HoloLoom server (Ctrl+C in terminal)
2. Modify a Python file
3. Save it
4. Check VS Code notification

**Expected**:
```
Warning: "HoloLoom: Server not running. Knowledge graph updates paused."
```

5. Start server again (same command as before)
6. Modify file
7. Check notification

**Expected**:
```
"HoloLoom: Updated X entities from 1 file(s)"
```

### Test 5: TODO/FIXME Tracking

**Steps**:
1. Create a file with comments:
   ```python
   def complex_function():
       # TODO: Add error handling
       # TODO: Add logging
       # FIXME: Remove debug code
       pass
   ```
2. Save the file

**Expected Output**:
```
WorkspaceWatcher: Indexed 1 file(s), created X entities, found 3 TODOs
```

**Verify**: Check HoloLoom response for `todos: 3`

## Monitoring

### Console Output

Enable debug logging in VS Code:

```javascript
// In Debug Console
console.log('Setting: promptly.debug = true');
```

This shows all file watcher events:
```
WorkspaceWatcher: File changed: ...
WorkspaceWatcher: File created: ...
WorkspaceWatcher: Indexed X file(s), created Y entities
```

### Network Activity

**Chrome DevTools / Network Tab**:
```
POST http://localhost:8000/ingest/workspace/incremental
Headers: Content-Type: application/json
Body: {
  "files": ["...file1.py", "...file2.ts"],
  "workspace_path": "/workspace"
}
Response: {
  "success": true,
  "files_indexed": 2,
  "entities_created": 15
}
```

### Server Logs

HoloLoom server shows:
```
INFO: POST /ingest/workspace/incremental
INFO: Incremental workspace ingestion: 2 files
INFO: Indexed: src/main.py
INFO: Indexed: src/utils.ts
INFO: Incremental ingestion complete: 2 files, 15 elements, 3 TODOs
```

## Troubleshooting

### Issue: "Server not running" Warning

**Symptom**: Always see "Server not running" even after starting server

**Solution**:
1. Check server actually started (look for "Uvicorn running on..." message)
2. Check URL in settings matches: `http://localhost:8000`
3. Check firewall isn't blocking port 8000
4. Try accessing http://localhost:8000/health in browser

### Issue: No Notifications Showing

**Symptom**: Files save but no UI feedback

**Causes**:
- Updates too small (< 10 entities, < 2 TODOs)
  - This is expected - notifications only show for significant updates
  - Check console instead
- Server returning error silently
  - Check console for error messages
  - Check server logs for exceptions

**Debug**:
```javascript
// In VS Code Debug Console
vscode.window.showInformationMessage("Test notification");
// Should see popup
```

### Issue: Files Not Being Indexed

**Symptoms**:
- File changed but no indexing happens
- Console shows no output

**Causes**:
1. File not in watch patterns
   - Only watches: `*.{py,ts,tsx,js,jsx,md}`
   - Other files are ignored

2. File in exclude pattern
   - Excludes: `node_modules`, `.git`, `__pycache__`, `dist`, `build`
   - Add files to subdirectories of `src` to index

3. Extension not started
   - Check if WorkspaceWatcher.start() was called
   - Should be called on extension activation

**Solution**:
```typescript
// Add more file patterns in workspaceWatcher.ts
private static readonly WATCH_PATTERNS = [
    '**/*.{ts,tsx,js,jsx,py,md,go,rs}'  // Add more extensions
];
```

### Issue: Slow Indexing / Timeout

**Symptom**: Large batch of files times out

**Causes**:
- Very large files (>1MB)
- Very large batch (>100 files)
- Slow AST parsing
- Server resources exhausted

**Solution**:
```typescript
// Increase timeout in HoloLoomCommands
timeout: 60000  // Increase from 30s to 60s

// Or reduce batch size
// Implement chunking for > 50 files
```

### Issue: Memory Usage Growing

**Symptom**: Extension uses more memory over time

**Causes**:
- Pending files set not cleared
- Timers not cleaned up
- Memory leak in batch processing

**Solution**:
1. Restart VS Code
2. Check WorkspaceWatcher.stop() is called on deactivate
3. Verify no reference cycles

## Running Tests

### TypeScript Tests

```bash
cd promptly-vscode
npm test

# Or run specific test:
npm test -- --grep "batch"
```

**Expected Output**:
```
  WorkspaceWatcher - HoloLoom Integration Tests
    ✓ Debouncing batches multiple file changes (150ms)
    ✓ Debounce timer resets on new changes (200ms)
    ✓ Duplicate files are deduplicated (100ms)
    ✓ New files use shorter debounce (50ms)
    ...

  25 tests passing
```

### Python Tests

```bash
cd HoloLoom/server
pytest test_incremental_ingestion.py -v

# Or specific test class:
pytest test_incremental_ingestion.py::TestIncrementalIngestionEndpoint -v

# Or specific test:
pytest test_incremental_ingestion.py::TestIncrementalIngestionEndpoint::test_empty_file_list -v
```

**Expected Output**:
```
test_incremental_ingestion.py::TestIncrementalIngestionEndpoint::test_incremental_ingestion_with_valid_files PASSED
test_incremental_ingestion.py::TestIncrementalIngestionEndpoint::test_empty_file_list PASSED
...

25 passed in 5.23s
```

## Performance Baseline

For reference, typical performance:

| Operation | Time |
|-----------|------|
| File save detection | <10ms |
| 2-second debounce wait | 2000ms |
| Single file indexing | 150-300ms |
| 3-file batch | 400-600ms |
| Total end-to-end | 2.2-2.6s |

If much slower, check:
1. Server performance (disk I/O, CPU)
2. Network latency (is server on same machine?)
3. File sizes (very large files parse slowly)

## What Gets Indexed

### Python Files
- Functions (name, parameters, docstring)
- Classes (name, methods)
- Imports
- Comments (TODO, FIXME, NOTE)

**Example**:
```python
def my_function(param1, param2):
    """This is indexed."""
    pass

class MyClass:
    def method(self):
        # TODO: Implement this
        pass
```

Gets indexed as:
- Entities: `my_function`, `MyClass`, `method`
- Motifs: `function`, `class`
- TODOs: 1

### TypeScript/JavaScript Files
- Functions
- Classes
- Interfaces
- Exports
- Comments

### Markdown Files
- Headings
- Code blocks
- Link text

## Next Steps

1. **Test the Integration**
   - Run through all 5 tests above
   - Verify console output
   - Check network requests

2. **Run Test Suite**
   - TypeScript: `npm test`
   - Python: `pytest test_incremental_ingestion.py`
   - All tests should pass

3. **Integration with Queries**
   - Open HoloLoom Query view
   - Index a new function
   - Query for it → should find it!

4. **Performance Testing**
   - Create large batch (10+ files)
   - Measure end-to-end time
   - Compare to manual indexing

5. **Provide Feedback**
   - Issues: Open GitHub issue
   - Improvements: Create feature request
   - Performance: Share metrics/profiles

## Common Commands

```bash
# Start HoloLoom server
cd HoloLoom/server
python -m uvicorn agentic_api:app --reload --port 8000

# Run TypeScript tests
cd promptly-vscode
npm test

# Run Python tests
cd HoloLoom/server
pytest test_incremental_ingestion.py -v

# Check server health
curl http://localhost:8000/health

# View recent logs
tail -f HoloLoom/server/agentic_api.log

# Reset workspace indexing (clear KG)
# Stop server, delete in-memory shards, restart
```

## Tips & Tricks

### Faster Testing

**Use small test files**:
```python
# Good - indexes instantly
def func():
    pass

# Bad - slower, lots to parse
def func_with_many_parameters(
    param1: str,
    param2: int,
    param3: float,
    param4: bool,
    param5: List[str],
    param6: Optional[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Very long complex function."""
    pass
```

### Batch Verification

**Check if single API call was made**:
1. Open DevTools (F12)
2. Network tab
3. Look for single `POST /ingest/workspace/incremental`
4. Should have multiple files in body

### TODO Tracking

**Quick TODO count**:
1. Modify file with TODOs
2. Save
3. Check console output for `found X TODOs`
4. Verify with API response

## Success Criteria

✅ You'll know it's working when:

1. **Files index automatically** - No manual "Index Workspace" button needed
2. **No crashes** - Even if server down, extension stays stable
3. **Batching works** - 3 file saves = 1 API call
4. **Notifications appear** - For significant updates (>10 entities)
5. **Queries find new content** - After saving, queries find recent code
6. **Tests pass** - Both TypeScript and Python test suites pass

## Getting Help

- **Extension issues**: Check VS Code Debug Console (Ctrl+Shift+J)
- **Server issues**: Check HoloLoom server terminal output
- **Network issues**: Use browser DevTools Network tab
- **Test failures**: Run with `-vv` flag for verbose output

---

**Happy indexing!** 🚀

For detailed architecture and design decisions, see `FILE_WATCHER_INTEGRATION_NOTES.md`
