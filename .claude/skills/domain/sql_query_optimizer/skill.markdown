# Skill: SQL Query Optimizer

## Metadata

- **Name**: `sql_query_optimizer`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `sql, database, performance, optimization`

## Description

**Short Description**:
Analyzes SQL queries for performance issues and provides optimized alternatives with index recommendations and execution plan insights.

**Detailed Description**:
Database performance is critical for application scalability. This skill analyzes SQL queries (SELECT, JOIN, subqueries), identifies performance bottlenecks (missing indexes, N+1 queries, inefficient joins, subquery hell), suggests optimized query rewrites, recommends indexes, and explains execution plans. Supports PostgreSQL, MySQL, SQLite syntax with database-specific optimizations.

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
  "query": "string - SQL query to optimize",
  "schema": {
    "tables": [
      {
        "name": "string",
        "columns": ["array of column names"],
        "indexes": ["array of indexed columns"]
      }
    ]
  },
  "execution_plan": "string (optional) - EXPLAIN output",
  "database_type": "string (optional) - postgresql|mysql|sqlite (default: postgresql)"
}
```

## Output Schema

```json
{
  "issues": [
    {
      "type": "string - Issue category",
      "severity": "critical|high|medium|low",
      "description": "string",
      "query_section": "string - Problematic SQL clause"
    }
  ],
  "optimized_query": "string - Rewritten query",
  "index_recommendations": [
    {
      "table": "string",
      "columns": ["array"],
      "index_type": "btree|hash|gin|gist",
      "sql": "string - CREATE INDEX statement",
      "estimated_improvement": "string"
    }
  ],
  "execution_plan_analysis": "string (optional) - Explain plan insights",
  "metadata": {
    "estimated_speedup": "string - e.g., '5-10x faster'",
    "confidence": "number (0.0-1.0)"
  }
}
```

## Prompt Template

```markdown
You are a database performance expert analyzing SQL queries for optimization.

**Query**:
{query}

**Schema**:
{schema}

**Execution Plan** (if provided):
{execution_plan}

**Database**: {database_type}

**Your Task**:
1. Identify performance issues (missing indexes, inefficient joins, etc.)
2. Rewrite query for better performance
3. Recommend indexes with CREATE INDEX statements
4. Analyze execution plan (if provided)
5. Estimate performance improvement

**Common SQL Performance Issues**:
- Missing indexes on WHERE/JOIN columns
- SELECT * instead of specific columns
- N+1 query patterns
- Subqueries that should be JOINs
- Inefficient JOIN order
- Missing LIMIT on large result sets
- Full table scans
- Inefficient OR conditions (use UNION instead)
- NOT IN with NULLs (use NOT EXISTS)

**Optimization Strategies**:
- Add indexes on frequently filtered/joined columns
- Use EXISTS instead of IN for large datasets
- Replace subqueries with JOINs when possible
- Use CTEs (WITH) for readability and optimization
- Avoid SELECT * (specify columns)
- Use covering indexes
- Partition large tables

Return structured JSON matching the output schema.
```

## Examples

### Example 1: Missing Index

**Input**:
```json
{
  "query": "SELECT * FROM users WHERE email = 'test@test.com';",
  "schema": {
    "tables": [
      {
        "name": "users",
        "columns": ["id", "email", "name", "created_at"],
        "indexes": ["id"]
      }
    ]
  }
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "missing_index",
      "severity": "high",
      "description": "Full table scan on 'email' column without index",
      "query_section": "WHERE email = 'test@test.com'"
    },
    {
      "type": "select_star",
      "severity": "medium",
      "description": "SELECT * retrieves unnecessary columns",
      "query_section": "SELECT *"
    }
  ],
  "optimized_query": "SELECT id, email, name FROM users WHERE email = 'test@test.com';",
  "index_recommendations": [
    {
      "table": "users",
      "columns": ["email"],
      "index_type": "btree",
      "sql": "CREATE INDEX idx_users_email ON users(email);",
      "estimated_improvement": "100-1000x faster for email lookups"
    }
  ],
  "execution_plan_analysis": null,
  "metadata": {
    "estimated_speedup": "100-1000x with index",
    "confidence": 0.95
  }
}
```

### Example 2: Inefficient Subquery

**Input**:
```json
{
  "query": "SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE active = true);",
  "schema": {
    "tables": [
      {
        "name": "orders",
        "columns": ["id", "user_id", "total", "created_at"],
        "indexes": ["id"]
      },
      {
        "name": "users",
        "columns": ["id", "email", "active"],
        "indexes": ["id"]
      }
    ]
  }
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "inefficient_subquery",
      "severity": "high",
      "description": "IN subquery can be replaced with JOIN for better performance",
      "query_section": "WHERE user_id IN (SELECT id FROM users WHERE active = true)"
    },
    {
      "type": "missing_index",
      "severity": "high",
      "description": "Missing index on orders.user_id for JOIN optimization",
      "query_section": "WHERE user_id"
    }
  ],
  "optimized_query": "SELECT o.* FROM orders o INNER JOIN users u ON o.user_id = u.id WHERE u.active = true;",
  "index_recommendations": [
    {
      "table": "orders",
      "columns": ["user_id"],
      "index_type": "btree",
      "sql": "CREATE INDEX idx_orders_user_id ON orders(user_id);",
      "estimated_improvement": "10-100x faster for joins"
    },
    {
      "table": "users",
      "columns": ["active"],
      "index_type": "btree",
      "sql": "CREATE INDEX idx_users_active ON users(active);",
      "estimated_improvement": "5-10x faster for active user filtering"
    }
  ],
  "execution_plan_analysis": null,
  "metadata": {
    "estimated_speedup": "5-10x with JOIN + indexes",
    "confidence": 0.92
  }
}
```

### Example 3: N+1 Query Pattern

**Input**:
```json
{
  "query": "-- First query\nSELECT * FROM posts;\n\n-- Then for each post:\nSELECT * FROM comments WHERE post_id = ?;",
  "schema": {
    "tables": [
      {
        "name": "posts",
        "columns": ["id", "title", "content"],
        "indexes": ["id"]
      },
      {
        "name": "comments",
        "columns": ["id", "post_id", "text"],
        "indexes": ["id"]
      }
    ]
  }
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "n_plus_one",
      "severity": "critical",
      "description": "N+1 query pattern: fetching comments in a loop instead of one query",
      "query_section": "Multiple queries in application loop"
    }
  ],
  "optimized_query": "SELECT p.*, c.id AS comment_id, c.text AS comment_text FROM posts p LEFT JOIN comments c ON p.id = c.post_id;",
  "index_recommendations": [
    {
      "table": "comments",
      "columns": ["post_id"],
      "index_type": "btree",
      "sql": "CREATE INDEX idx_comments_post_id ON comments(post_id);",
      "estimated_improvement": "Eliminates N queries (1 query instead of N+1)"
    }
  ],
  "execution_plan_analysis": "Single query with JOIN eliminates application-level loops and reduces database round-trips from N+1 to 1.",
  "metadata": {
    "estimated_speedup": "10-100x reduction in queries (from N+1 to 1)",
    "confidence": 0.90
  }
}
```

## Testing Checklist

- [x] **Functionality**: Identifies common SQL anti-patterns
- [x] **Error Handling**: Handles malformed SQL gracefully
- [x] **Security**: No SQL execution (analysis only)
- [x] **Performance**: < 1s per query analysis
- [x] **Token Efficiency**: ~700 tokens
- [x] **Documentation**: Complete
- [x] **Dependencies**: None
- [x] **Edge Cases**: Complex queries, CTEs, window functions
- [x] **Output Consistency**: Structured JSON
- [x] **Integration**: Standalone

## Security Considerations

**Potential Risks**:
- SQL injection analysis (identifies vulnerable patterns)
**Data Privacy**:
- [x] Does not execute queries
- [x] Does not connect to databases
**Sandboxing**:
- [x] Static analysis only

## Performance Characteristics

- **Expected Latency**: 500ms - 1s
- **Token Usage**: ~700 tokens
- **Resource Requirements**: Minimal
- **Scalability**: O(n) with query complexity

## Maintenance Notes

**Known Limitations**:
- Static analysis (no actual execution plan)
- Generic recommendations (not workload-specific)

**Future Enhancements**:
- Integration with EXPLAIN ANALYZE
- Workload-specific optimizations
- Query plan visualization

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release

## License

MIT License
