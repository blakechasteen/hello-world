# Phase 3.5 Testing Guide

## Quick Start Testing

### Step 1: Start the HoloLoom Server

Open a terminal and run:

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
PYTHONPATH=. uvicorn HoloLoom.server.unified_server:app --reload --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 2: Run Automated Tests

In a **second terminal**, run:

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
python HoloLoom/web_dashboard/test_phase3_4.py
```

**Expected Output**:
```
[TEST 1] Server Health Check
✓ Server online (uptime: X.Xs)

[TEST 2] Making 20 Diverse Queries (for analytics data)
  Query 1/20: What is Thompson Sampling?... (XXms, conf: 0.XX)
  Query 2/20: Explain Bayesian inference... (XXms, conf: 0.XX)
  ...
✓ Completed 20 queries successfully

[TEST 3] Query Classification Verification
Query type distribution:
  factual     :  X (XX.X%)
  procedural  :  X (XX.X%)
  analytical  :  X (XX.X%)
  creative    :  X (XX.X%)
  debugging   :  X (XX.X%)
✓ Good diversity: 5/5 query types represented

[TEST 4] Performance Metrics Verification
Latency stats:
  Average: XX.Xms
  Min: XX.Xms
  Max: XX.Xms
...

All tests passed!
```

### Step 3: Manual Testing in Browser

1. **Open Dashboard**:
   - Navigate to `HoloLoom/web_dashboard/control_panel.html` in your browser
   - Or go to `http://localhost:8000/static/control_panel.html` if served

2. **Navigate to Analytics Tab**:
   - Click "Analytics" tab in navigation
   - You should see 4 cards:
     - Query Comparison Table
     - Historical Confidence Tracking
     - Tool Effectiveness Matrix
     - System Health Dashboard
     - **Data Management (NEW in Phase 3.5)**

3. **Verify Data Persistence** (Phase 3.5):

   **Test A: Auto-Persistence**
   - ✓ Verify all 20 queries from test script are visible
   - ✓ Refresh the browser page (F5)
   - ✓ Navigate back to Analytics tab
   - ✓ **All data should still be there** (queries, charts, metrics)

   **Test B: Storage Usage**
   - ✓ Check "Storage Usage" section shows metrics:
     - Used space: ~X KB
     - Usage percentage: ~X%
     - Query count: 20
     - Confidence score count: 20
     - Tool count: X
   - ✓ Progress bar should be green (low usage)
   - ✓ Click "Refresh Usage" - metrics update immediately

   **Test C: Export Data**
   - ✓ Click "📥 Export Data (JSON)" button
   - ✓ File downloads: `hololoom-analytics-<timestamp>.json`
   - ✓ Open file in text editor
   - ✓ Verify JSON structure:
     ```json
     {
       "version": "3.5.0",
       "exportDate": "2025-11-13T...",
       "queryHistory": [...20 queries...],
       "confidenceHistory": [...],
       "toolStats": {...},
       "systemHealth": {...}
     }
     ```

   **Test D: Clear Data**
   - ✓ Click "🗑️ Clear All Data" button
   - ✓ Confirmation dialog appears
   - ✓ Click "OK"
   - ✓ All analytics reset to empty state
   - ✓ Storage usage shows 0 KB

   **Test E: Import Data**
   - ✓ Click "📤 Import Data" button
   - ✓ File picker opens
   - ✓ Select previously exported JSON file
   - ✓ Confirmation dialog: "Import 20 queries? This will replace current data."
   - ✓ Click "OK"
   - ✓ All 20 queries restored
   - ✓ Charts and metrics repopulated
   - ✓ Storage usage back to ~X KB

   **Test F: Cross-Session Persistence**
   - ✓ With data loaded, close browser completely
   - ✓ Reopen browser
   - ✓ Navigate to control_panel.html
   - ✓ Go to Analytics tab
   - ✓ **Data should automatically load from previous session**

### Step 4: Advanced Testing

**Test G: Quota Management** (Optional)

This test verifies automatic cleanup when storage is full.

1. Open browser console (F12 → Console tab)
2. Manually fill LocalStorage:
   ```javascript
   // Generate large data to fill storage
   let largeData = [];
   for (let i = 0; i < 10000; i++) {
       largeData.push({
           query: `Test query ${i}`.repeat(100),
           confidence: Math.random(),
           latency_ms: Math.random() * 100
       });
   }

   // Try to store (should trigger quota exceeded)
   try {
       localStorage.setItem('test_large', JSON.stringify(largeData));
   } catch (e) {
       console.log('Storage full - expected!');
   }
   ```

3. Run a new query through the dashboard
4. Check console for: `[AnalyticsMonitor] Cleared oldest 25% to free space`
5. Verify newest data is preserved

## Expected Test Results

### Automated Tests
- ✓ 6/6 tests passing
- ✓ 20 diverse queries executed
- ✓ Query types: factual, procedural, analytical, creative, debugging
- ✓ Performance metrics calculated
- ✓ Tool distribution analyzed
- ✓ Cache effectiveness tracked

### Manual Tests (Phase 3.5)
- ✓ Data persists across page refreshes
- ✓ Data persists across browser restarts
- ✓ Storage usage displayed correctly
- ✓ Export downloads JSON file
- ✓ Import restores from JSON file
- ✓ Clear removes all data
- ✓ Auto-save after each query (debounced)
- ✓ Automatic quota management

## Troubleshooting

### Server Won't Start

**Error**: `ModuleNotFoundError: No module named 'HoloLoom'`

**Fix**: Ensure PYTHONPATH is set:
```bash
PYTHONPATH=. uvicorn HoloLoom.server.unified_server:app --port 8000
```

### Tests Fail to Connect

**Error**: `Cannot connect to host localhost:8000`

**Fix**: Make sure server is running in a separate terminal first.

### Unicode Error in Tests

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2717'`

**Fix**: Set console encoding (Windows):
```bash
chcp 65001
python HoloLoom/web_dashboard/test_phase3_4.py
```

### Data Not Persisting

**Issue**: Data disappears after refresh

**Possible Causes**:
1. **Private/Incognito Mode**: LocalStorage is cleared on exit
   - **Fix**: Use normal browser mode
2. **LocalStorage Disabled**: Browser settings block storage
   - **Fix**: Enable LocalStorage in browser settings
3. **Console Errors**: Check browser console for errors
   - **Fix**: Address errors shown in console

### Export Button Doesn't Work

**Issue**: Clicking Export does nothing

**Possible Causes**:
1. **No Data**: Can't export empty dataset
   - **Fix**: Run some queries first
2. **Pop-up Blocked**: Browser blocked download
   - **Fix**: Allow downloads from localhost
3. **Analytics Monitor Not Initialized**:
   - **Fix**: Check console for errors, refresh page

### Import Fails

**Issue**: Import shows "Invalid data format"

**Possible Causes**:
1. **Wrong File Format**: Not a valid HoloLoom export
   - **Fix**: Use files exported from Phase 3.5
2. **Corrupted JSON**: File is malformed
   - **Fix**: Validate JSON at jsonlint.com
3. **Old Format**: Pre-3.5 export (no version field)
   - **Fix**: Re-export from Phase 3.5

## Performance Benchmarks

Expected performance on typical hardware:

| Operation | Expected Latency | Frequency |
|-----------|------------------|-----------|
| Auto-save (debounced) | <10ms | Max 1/second |
| Load on startup | <20ms | Once per session |
| Export to JSON | <50ms | On-demand |
| Import from JSON | <100ms | On-demand |
| Storage usage calc | <5ms | Every 10 seconds |
| Query addition | <2ms | Per query |

**Storage Capacity**:
- 100 queries ≈ 15-25 KB
- 1000 queries ≈ 150-250 KB
- Typical quota: 5-10 MB
- Estimated capacity: 20,000+ queries

## Console Logs to Expect

### On Dashboard Load
```
[AnalyticsMonitor] Initializing...
[AnalyticsMonitor] Loaded 20 queries from 11/13/2025, 3:45:00 PM
[AnalyticsMonitor] Initialized with 20 persisted queries
```

### On Auto-Save
```
[AnalyticsMonitor] Saved 21 queries (18.5 KB)
```

### On Export
```
[AnalyticsMonitor] Data exported successfully
```

### On Import
```
[AnalyticsMonitor] Data imported successfully
```

### On Clear
```
[AnalyticsMonitor] All data cleared
```

### On Quota Exceeded
```
[AnalyticsMonitor] Storage quota exceeded. Clearing old data...
[AnalyticsMonitor] Cleared 5 old queries to free up space
[AnalyticsMonitor] Saved 15 queries (12.3 KB)
```

## Success Criteria

Phase 3.5 is working correctly if:

- [x] All automated tests pass (6/6)
- [x] Data persists across page refreshes
- [x] Data persists across browser restarts
- [x] Storage usage displays correctly
- [x] Export downloads valid JSON
- [x] Import restores data successfully
- [x] Clear removes all data
- [x] Auto-save triggers after queries
- [x] No console errors
- [x] Performance <10ms overhead

## Next Steps After Testing

Once Phase 3.5 is verified:

**Option 1**: Document any issues found during testing

**Option 2**: Proceed to Phase 3.6 (Advanced Filtering)
- Filter queries by date range
- Filter by confidence threshold
- Filter by tool used
- Filter by query type

**Option 3**: Proceed to Phase 3.7 (Custom Dashboards)
- Drag-and-drop card arrangement
- Hide/show specific metrics
- Custom color themes

**Option 4**: Production deployment
- Set up server on production hardware
- Configure HTTPS
- Enable authentication
- Set up monitoring/alerts

---

**Phase 3.5 Testing Status**: Ready for Testing ✅

Last Updated: November 13, 2025
