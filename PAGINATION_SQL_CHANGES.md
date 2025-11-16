# HoloLoom Pagination - SQL Changes Reference

## Summary
All pagination changes use standard SQL `LIMIT` and `OFFSET` clauses. No database schema modifications required.

## SQL Queries Modified

### 1. Get Total Executions Count
**File**: `HoloLoom/analytics/recursive_analytics.py`
**Method**: `get_total_executions_count()`
**Lines**: 405-427

#### Query Pattern
```sql
-- Without strategy filter
SELECT COUNT(*) FROM executions

-- With strategy filter
SELECT COUNT(*) FROM executions WHERE strategy = ?
```

**Parameters**:
- `strategy` (optional): Filter by strategy value

**Python Code**:
```python
def get_total_executions_count(self, strategy: Optional[ReasoningStrategy] = None) -> int:
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    if strategy:
        cursor.execute("""
            SELECT COUNT(*) FROM executions WHERE strategy = ?
        """, (strategy.value,))
    else:
        cursor.execute("SELECT COUNT(*) FROM executions")

    count = cursor.fetchone()[0]
    conn.close()
    return count
```

### 2. Get Recent Executions with Pagination
**File**: `HoloLoom/analytics/recursive_analytics.py`
**Method**: `get_recent_executions()`
**Lines**: 429-482

#### Previous Query (No Pagination)
```sql
SELECT * FROM executions
ORDER BY timestamp DESC
LIMIT ?
```

#### New Query (With Pagination)
```sql
-- Without strategy filter
SELECT * FROM executions
ORDER BY timestamp DESC
LIMIT ? OFFSET ?

-- With strategy filter
SELECT * FROM executions
WHERE strategy = ?
ORDER BY timestamp DESC
LIMIT ? OFFSET ?
```

**Parameters**:
- `limit`: Maximum number of rows to return
- `skip` (OFFSET): Number of rows to skip before returning results
- `strategy` (optional): Filter by strategy value

**Python Code**:
```python
def get_recent_executions(
    self,
    limit: int = 10,
    skip: int = 0,
    strategy: Optional[ReasoningStrategy] = None
) -> List[ExecutionRecord]:
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if strategy:
        cursor.execute("""
            SELECT * FROM executions
            WHERE strategy = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (strategy.value, limit, skip))
    else:
        cursor.execute("""
            SELECT * FROM executions
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, skip))

    records = []
    for row in cursor.fetchall():
        records.append(ExecutionRecord(...))

    conn.close()
    return records
```

## SQL Clauses Explanation

### LIMIT Clause
```sql
LIMIT n
```
- Returns at most `n` rows
- Used to control page size
- Example: `LIMIT 20` returns up to 20 rows

### OFFSET Clause
```sql
LIMIT n OFFSET m
```
- Skips the first `m` rows, then returns up to `n` rows
- `OFFSET` is SQL standard equivalent of MySQL's `LIMIT m, n`
- Used for pagination offset
- Example: `LIMIT 20 OFFSET 40` skips 40 rows, then returns 20 rows

### Combined Pattern
```sql
SELECT * FROM table
ORDER BY column DESC
LIMIT page_size OFFSET (page_number - 1) * page_size
```

## Performance Implications

### Query Efficiency

| Query Type | Complexity | Notes |
|-----------|-----------|-------|
| `COUNT(*)` | O(1) - O(n) | SQLite scans entire table if no index |
| `SELECT ... LIMIT` | O(1) | Very fast, returns first N rows |
| `SELECT ... LIMIT/OFFSET` | O(skip + limit) | Must scan skip + limit rows |
| Indexed ORDER BY | O(log n) | With timestamp index |

### Index Usage
Existing indexes support pagination queries:
```sql
CREATE INDEX idx_timestamp ON executions(timestamp)
CREATE INDEX idx_strategy ON executions(strategy)
```

These indexes accelerate:
- `ORDER BY timestamp DESC` - Uses idx_timestamp
- `WHERE strategy = ?` - Uses idx_strategy
- Combined queries use both indexes via query optimizer

### OFFSET Performance Notes
```sql
-- Fast (small offset)
SELECT * FROM executions ORDER BY timestamp DESC LIMIT 20 OFFSET 0
SELECT * FROM executions ORDER BY timestamp DESC LIMIT 20 OFFSET 100

-- Slower (large offset)
SELECT * FROM executions ORDER BY timestamp DESC LIMIT 20 OFFSET 100000

-- Very slow (huge offset)
SELECT * FROM executions ORDER BY timestamp DESC LIMIT 20 OFFSET 1000000
```

SQLite must scan through OFFSET rows before returning results.
**Recommendation**: For pagination of very large datasets (>1M rows), consider implementing cursor-based pagination instead.

## Database Schema (No Changes)

### Current Schema (Unchanged)
```sql
CREATE TABLE executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    query_text TEXT,
    iterations INTEGER,
    initial_quality REAL,
    final_quality REAL,
    quality_gain REAL,
    duration_ms REAL,
    tokens_used INTEGER,
    cost REAL,
    converged BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
```

No schema modifications needed.

### Indexes (No Changes)
```sql
CREATE INDEX idx_strategy ON executions(strategy)
CREATE INDEX idx_timestamp ON executions(timestamp)
```

Existing indexes support pagination.

## SQLite-Specific Notes

### LIMIT/OFFSET vs Other Databases
| Database | Syntax | Notes |
|----------|--------|-------|
| SQLite | `LIMIT 10 OFFSET 20` | Standard SQL |
| MySQL | `LIMIT 20, 10` | Non-standard, older syntax |
| PostgreSQL | `LIMIT 10 OFFSET 20` | Standard SQL |
| SQL Server | `OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY` | MSSQL syntax |

HoloLoom uses SQLite's standard `LIMIT/OFFSET` syntax.

### SQLite Optimization
SQLite query optimizer automatically:
- Uses available indexes
- Reduces scanned rows when possible
- Caches query plans (prepared statements)

The code uses parameterized queries (`?`) which enables SQLite's query caching:
```python
cursor.execute("SELECT * FROM executions LIMIT ? OFFSET ?", (limit, skip))
```

## Testing SQL Queries

### Direct SQLite Testing
```bash
sqlite3 .hololoom/recursive_analytics.db

# Test COUNT
sqlite> SELECT COUNT(*) FROM executions;
42

# Test pagination
sqlite> SELECT * FROM executions ORDER BY timestamp DESC LIMIT 10 OFFSET 0;
# Returns first 10 records

sqlite> SELECT * FROM executions ORDER BY timestamp DESC LIMIT 10 OFFSET 10;
# Returns next 10 records

# Check indexes
sqlite> .indices executions
idx_strategy
idx_timestamp

# Analyze query performance
sqlite> EXPLAIN QUERY PLAN
   SELECT * FROM executions
   ORDER BY timestamp DESC
   LIMIT 10 OFFSET 20;
```

### Python Testing
```python
from HoloLoom.analytics.recursive_analytics import RecursiveAnalytics

analytics = RecursiveAnalytics()

# Test count
total = analytics.get_total_executions_count()
print(f"Total records: {total}")

# Test pagination
page1 = analytics.get_recent_executions(limit=10, skip=0)
page2 = analytics.get_recent_executions(limit=10, skip=10)

print(f"Page 1: {len(page1)} items")
print(f"Page 2: {len(page2)} items")

# Test with strategy filter
refine_total = analytics.get_total_executions_count(strategy="refine")
print(f"REFINE executions: {refine_total}")
```

## Migration and Rollback

### Forward Migration (Current)
- No schema changes required
- Updated SQL queries are backward compatible
- Simply deploy new code

### Rollback (If Needed)
- Old code without pagination parameters still works (defaults to skip=0)
- Revert code changes without database modifications

## Future Optimization

### Recommended Enhancement: Cursor-Based Pagination
For very large datasets, implement keyset pagination instead of OFFSET:

```sql
-- Current approach (OFFSET-based)
SELECT * FROM executions
ORDER BY timestamp DESC
LIMIT 20 OFFSET 1000
-- This requires scanning 1020 rows

-- Future approach (cursor-based)
SELECT * FROM executions
WHERE timestamp < ? AND id < ?
ORDER BY timestamp DESC, id DESC
LIMIT 20
-- This uses index to skip directly to cursor position
```

**Benefits**:
- O(log n) complexity instead of O(skip + limit)
- Constant performance regardless of page number
- Better for infinite scroll UI patterns

## Summary

**Changes**:
- Added `LIMIT ? OFFSET ?` to `get_recent_executions()`
- Added `COUNT(*)` query in `get_total_executions_count()`

**Schema Impact**: None
**Performance Impact**: Negligible for typical page sizes (10-100 items)
**Backward Compatibility**: Full (default skip=0 matches old behavior)
**Testing Status**: Syntax verified, ready for integration testing

---

**Last Updated**: 2025-11-16
