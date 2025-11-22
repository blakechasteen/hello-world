# Skill: React Performance Optimizer

## Metadata

- **Name**: `react_performance_optimizer`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `react, performance, optimization, frontend`

## Description

**Short Description**:
Identifies React anti-patterns and performance bottlenecks, providing optimized code examples and measurable improvement strategies.

**Detailed Description**:
React performance issues often stem from unnecessary re-renders, improper memoization, large component trees, and inefficient state management. This skill analyzes React components, detects performance anti-patterns (missing useMemo/useCallback, inline functions in JSX, large lists without virtualization), suggests optimizations with code examples, and estimates performance improvements. Covers React 18+ features (concurrent rendering, automatic batching, transitions).

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
  "component_code": "string - React component source code",
  "performance_issue": "string (optional) - Specific issue (slow renders, memory leak, etc.)",
  "react_version": "string (optional) - React version (default: 18.x)"
}
```

## Output Schema

```json
{
  "issues": [
    {
      "type": "string - Issue category",
      "severity": "critical|high|medium|low",
      "description": "string - What's wrong",
      "line_numbers": ["array of affected lines"]
    }
  ],
  "optimizations": [
    {
      "issue_type": "string",
      "optimized_code": "string - Fixed code",
      "explanation": "string - Why this is better",
      "estimated_improvement": "string - e.g., '2x fewer renders'"
    }
  ],
  "code_diff": "string - Unified diff showing changes",
  "metadata": {
    "total_issues": "number",
    "estimated_overall_improvement": "string",
    "confidence": "number (0.0-1.0)"
  }
}
```

## Prompt Template

```markdown
You are a React performance expert analyzing components for optimization opportunities.

**Component Code**:
{component_code}

**Reported Issue**:
{performance_issue}

**React Version**:
{react_version}

**Your Task**:
1. Identify performance anti-patterns
2. Categorize by severity (critical/high/medium/low)
3. Provide optimized code for each issue
4. Estimate performance improvements
5. Generate code diff

**Common React Performance Issues**:
- Unnecessary re-renders (missing React.memo, useMemo, useCallback)
- Inline functions/objects in JSX props
- Large lists without virtualization (react-window, react-virtual)
- Expensive computations not memoized
- Context value changes causing cascading re-renders
- useEffect dependency arrays causing loops
- Large component trees without code splitting
- Images without lazy loading

**Optimization Strategies**:
- Use React.memo for expensive pure components
- Wrap callbacks in useCallback
- Wrap expensive computations in useMemo
- Virtualize long lists (>100 items)
- Split code with React.lazy + Suspense
- Use React 18 concurrent features (useTransition, useDeferredValue)
- Optimize context usage (split contexts)

Return structured JSON matching the output schema.
```

## Examples

### Example 1: Unnecessary Re-renders

**Input**:
```json
{
  "component_code": "function UserList({ users }) {\n  return users.map(user => (\n    <UserCard key={user.id} user={user} onDelete={() => deleteUser(user.id)} />\n  ));\n}",
  "performance_issue": "Component re-renders too often"
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "inline_function_in_jsx",
      "severity": "high",
      "description": "Inline arrow function in onDelete prop causes UserCard to re-render every time",
      "line_numbers": [3]
    }
  ],
  "optimizations": [
    {
      "issue_type": "inline_function_in_jsx",
      "optimized_code": "const UserList = React.memo(({ users }) => {\n  const handleDelete = useCallback((id) => deleteUser(id), []);\n  return users.map(user => (\n    <UserCard key={user.id} user={user} onDelete={() => handleDelete(user.id)} />\n  ));\n});",
      "explanation": "useCallback memoizes the delete handler. React.memo prevents re-renders when props don't change.",
      "estimated_improvement": "2-3x fewer re-renders for UserCard components"
    }
  ],
  "code_diff": "- function UserList({ users }) {\n+ const UserList = React.memo(({ users }) => {\n+   const handleDelete = useCallback((id) => deleteUser(id), []);\n    return users.map(user => (\n-     <UserCard key={user.id} user={user} onDelete={() => deleteUser(user.id)} />\n+     <UserCard key={user.id} user={user} onDelete={() => handleDelete(user.id)} />\n    ));\n- }",
  "metadata": {
    "total_issues": 1,
    "estimated_overall_improvement": "2-3x render performance improvement",
    "confidence": 0.92
  }
}
```

### Example 2: Large List Without Virtualization

**Input**:
```json
{
  "component_code": "function ProductList({ products }) {\n  return (\n    <div>\n      {products.map(product => <ProductCard key={product.id} product={product} />)}\n    </div>\n  );\n}",
  "performance_issue": "Slow scrolling with 1000+ products"
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "no_virtualization",
      "severity": "critical",
      "description": "Rendering 1000+ items without virtualization causes performance degradation",
      "line_numbers": [3]
    }
  ],
  "optimizations": [
    {
      "issue_type": "no_virtualization",
      "optimized_code": "import { FixedSizeList } from 'react-window';\n\nfunction ProductList({ products }) {\n  const Row = ({ index, style }) => (\n    <div style={style}>\n      <ProductCard product={products[index]} />\n    </div>\n  );\n\n  return (\n    <FixedSizeList\n      height={600}\n      itemCount={products.length}\n      itemSize={120}\n      width='100%'\n    >\n      {Row}\n    </FixedSizeList>\n  );\n}",
      "explanation": "react-window only renders visible items, drastically reducing DOM nodes and render time.",
      "estimated_improvement": "10-50x performance improvement for large lists"
    }
  ],
  "code_diff": "+ import { FixedSizeList } from 'react-window';\n\n  function ProductList({ products }) {\n+   const Row = ({ index, style }) => (\n+     <div style={style}>\n+       <ProductCard product={products[index]} />\n+     </div>\n+   );\n+\n    return (\n-     <div>\n-       {products.map(product => <ProductCard key={product.id} product={product} />)}\n-     </div>\n+     <FixedSizeList\n+       height={600}\n+       itemCount={products.length}\n+       itemSize={120}\n+       width='100%'\n+     >\n+       {Row}\n+     </FixedSizeList>\n    );\n  }",
  "metadata": {
    "total_issues": 1,
    "estimated_overall_improvement": "10-50x for 1000+ items",
    "confidence": 0.95
  }
}
```

### Example 3: Expensive Computation Not Memoized

**Input**:
```json
{
  "component_code": "function Dashboard({ data }) {\n  const stats = calculateComplexStats(data);\n  return <StatsDisplay stats={stats} />;\n}",
  "performance_issue": "Component lags on every render"
}
```

**Expected Output**:
```json
{
  "issues": [
    {
      "type": "expensive_computation_not_memoized",
      "severity": "high",
      "description": "calculateComplexStats runs on every render, even when data hasn't changed",
      "line_numbers": [2]
    }
  ],
  "optimizations": [
    {
      "issue_type": "expensive_computation_not_memoized",
      "optimized_code": "function Dashboard({ data }) {\n  const stats = useMemo(() => calculateComplexStats(data), [data]);\n  return <StatsDisplay stats={stats} />;\n}",
      "explanation": "useMemo caches the computation result and only recalculates when 'data' changes.",
      "estimated_improvement": "Eliminates unnecessary recalculations (potentially 10-100x faster)"
    }
  ],
  "code_diff": "  function Dashboard({ data }) {\n-   const stats = calculateComplexStats(data);\n+   const stats = useMemo(() => calculateComplexStats(data), [data]);\n    return <StatsDisplay stats={stats} />;\n  }",
  "metadata": {
    "total_issues": 1,
    "estimated_overall_improvement": "10-100x depending on computation complexity",
    "confidence": 0.90
  }
}
```

## Testing Checklist

- [x] **Functionality**: Detects common React anti-patterns
- [x] **Error Handling**: Handles malformed JSX
- [x] **Security**: No code execution
- [x] **Performance**: < 1s per analysis
- [x] **Token Efficiency**: ~650 tokens
- [x] **Documentation**: Complete
- [x] **Dependencies**: None
- [x] **Edge Cases**: Class components, hooks edge cases
- [x] **Output Consistency**: Structured JSON
- [x] **Integration**: Standalone

## Security Considerations

**Potential Risks**:
- Code analysis only (no execution)
**Data Privacy**:
- [x] Does not log component code
**Sandboxing**:
- [x] No external dependencies

## Performance Characteristics

- **Expected Latency**: 500ms - 1s
- **Token Usage**: ~650 tokens
- **Resource Requirements**: Minimal
- **Scalability**: O(n) with component size

## Maintenance Notes

**Known Limitations**:
- Static analysis only (no runtime profiling)
- Covers React 16.8+ (hooks)

**Future Enhancements**:
- Integration with React DevTools Profiler
- Server Component optimization (React 18+)

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release

## License

MIT License
