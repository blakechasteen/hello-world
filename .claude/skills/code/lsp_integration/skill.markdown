# Skill: LSP Integration

## Metadata

- **Name**: `lsp_integration`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `code`
- **Tags**: `lsp, code, intelligence, analysis, navigation, refactoring`

## Description

**Short Description**:
Language Server Protocol integration for deep code intelligence and navigation.

**Detailed Description**:
The LSP Integration skill provides comprehensive code understanding capabilities through the Language Server Protocol. Supports goto definition, find references, hover documentation, symbol search, code completion, diagnostics, and refactoring. Works with any LSP-compatible language server (Python: pyright/pylsp, TypeScript: tsserver, Rust: rust-analyzer, Go: gopls, etc.). Enables IDE-quality code intelligence in HoloLoom workflows, including real-time error detection, smart navigation, and workspace-wide refactoring.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [ ] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- Language servers for target languages:
  - Python: `pyright` or `pylsp` (Python Language Server)
  - TypeScript/JavaScript: `typescript-language-server`
  - Rust: `rust-analyzer`
  - Go: `gopls`
  - Java: `jdtls`
  - C/C++: `clangd`
- LSP client library (e.g., `pygls`, `python-lsp-jsonrpc`)
- Target language runtime/compiler for accurate analysis

**HoloLoom Integration**: Integrates with code analysis pipelines, test runners, documentation generation, and refactoring workflows.

## Input Schema

```json
{
  "operation": "string - goto_definition|find_references|hover|symbols|completion|diagnostics|rename|format",
  "parameters": {
    "file": "string (required for most ops) - File path",
    "position": {
      "line": "number (0-indexed) - Line number",
      "character": "number (0-indexed) - Character offset"
    },
    "workspace": "boolean (optional for symbols) - Search entire workspace",
    "new_name": "string (required for rename) - New symbol name",
    "formatter": "string (optional for format) - Formatter to use (black, autopep8, prettier, etc.)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "file": "string - Target file",
    "position": "object - Position in file (if applicable)",
    "symbol": "string - Symbol name (if applicable)",
    "type": "string - Symbol type signature (for hover)",
    "documentation": "string - Symbol documentation (for hover)",
    "references": "array - List of reference locations (for find_references)",
    "completions": "array - Code completion suggestions (for completion)",
    "diagnostics": "array - Errors and warnings (for diagnostics)",
    "changes": "array - Proposed changes (for rename/format)"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Goto Definition

**Input**:
```json
{
  "operation": "goto_definition",
  "parameters": {
    "file": "src/main.py",
    "position": {
      "line": 42,
      "character": 15
    }
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "file": "src/utils.py",
    "position": {"line": 10, "character": 0},
    "symbol": "process_data",
    "kind": "function"
  },
  "message": "Found definition of 'process_data'",
  "execution_time_ms": 85
}
```

**Explanation**: Navigates from a function call at line 42 to its definition in utils.py line 10. Essential for code exploration and understanding call hierarchies.

### Example 2: Find All References

**Input**:
```json
{
  "operation": "find_references",
  "parameters": {
    "file": "src/api.py",
    "position": {
      "line": 25,
      "character": 8
    }
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "symbol": "UserService",
    "references": [
      {"file": "src/api.py", "line": 25, "character": 8},
      {"file": "src/handlers.py", "line": 12, "character": 20},
      {"file": "src/controllers/auth.py", "line": 45, "character": 15},
      {"file": "tests/test_api.py", "line": 88, "character": 10}
    ],
    "count": 4
  },
  "message": "Found 4 references to 'UserService'",
  "execution_time_ms": 120
}
```

**Explanation**: Finds all usages of UserService class across the codebase. Critical for impact analysis before refactoring or understanding dependencies.

### Example 3: Hover Documentation

**Input**:
```json
{
  "operation": "hover",
  "parameters": {
    "file": "src/database.py",
    "position": {
      "line": 55,
      "character": 12
    }
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "symbol": "execute_query",
    "type": "def execute_query(sql: str, params: Dict[str, Any]) -> List[Dict]",
    "documentation": "Execute a parameterized SQL query safely.\n\nArgs:\n    sql: SQL query with placeholders\n    params: Query parameters\n\nReturns:\n    List of result rows as dictionaries\n\nRaises:\n    DatabaseError: If query execution fails"
  },
  "message": "Retrieved documentation for 'execute_query'",
  "execution_time_ms": 60
}
```

**Explanation**: Provides inline documentation with type signatures. Eliminates need to jump to source for function signatures and docstrings.

### Example 4: Code Completion

**Input**:
```json
{
  "operation": "completion",
  "parameters": {
    "file": "src/app.py",
    "position": {
      "line": 18,
      "character": 7
    }
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "completions": [
      {
        "label": "request",
        "kind": "variable",
        "detail": "request: Request",
        "documentation": "Current HTTP request object"
      },
      {
        "label": "response",
        "kind": "variable",
        "detail": "response: Response",
        "documentation": "HTTP response builder"
      },
      {
        "label": "render_template",
        "kind": "function",
        "detail": "render_template(template: str, **context) -> str",
        "documentation": "Render Jinja2 template with context"
      }
    ],
    "count": 3
  },
  "message": "Retrieved 3 completion suggestions",
  "execution_time_ms": 45
}
```

**Explanation**: Smart code completion based on context. Shows available variables, functions, and methods with types and documentation.

### Example 5: Workspace-Wide Rename

**Input**:
```json
{
  "operation": "rename",
  "parameters": {
    "file": "src/models.py",
    "position": {
      "line": 10,
      "character": 6
    },
    "new_name": "Customer"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "old_name": "User",
    "new_name": "Customer",
    "changes": [
      {"file": "src/models.py", "line": 10, "old": "class User:", "new": "class Customer:"},
      {"file": "src/api.py", "line": 25, "old": "user = User()", "new": "user = Customer()"},
      {"file": "src/handlers.py", "line": 42, "old": "def get_user", "new": "def get_customer"},
      {"file": "tests/test_models.py", "line": 15, "old": "User(", "new": "Customer("}
    ],
    "change_count": 15
  },
  "message": "Renamed 'User' to 'Customer' (15 changes across 8 files)",
  "execution_time_ms": 250
}
```

**Explanation**: Safe workspace-wide symbol renaming. Updates all references, imports, and string literals while preserving code semantics.

## Testing Checklist

- [x] **Functionality**: All 8 operations execute correctly
- [x] **Error Handling**: Graceful handling of missing files, invalid positions
- [x] **Security**: No command injection, safe file path handling
- [x] **Performance**: Operations complete within expected time (<500ms)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Language servers documented
- [x] **Edge Cases**: Handles missing symbols, empty files, syntax errors
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom code analysis pipeline

## Security Considerations

**Potential Risks**:
- **Path Traversal**: File paths could escape workspace -> Validate and sanitize paths
- **Code Execution**: LSP servers execute arbitrary code -> Run in sandboxed environment
- **Resource Exhaustion**: Large workspace analysis -> Implement timeouts and limits

**Data Privacy**:
- [x] Does not send code to external servers (local LSP only)
- [x] Does not log sensitive code content
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Does not attempt privilege escalation
- [x] Does not modify files outside workspace scope (for read operations)

## Performance Characteristics

- **Expected Latency**: 50-500ms (depending on operation and workspace size)
- **Token Usage**: 100-1000 tokens per execution
- **Resource Requirements**: Language server process, sufficient memory for AST parsing
- **Scalability**: Limited by workspace size and language server performance

**Operation-Specific Latencies**:
- `goto_definition`: 50-100ms (simple lookup)
- `find_references`: 100-300ms (workspace scan)
- `hover`: 50-150ms (documentation retrieval)
- `symbols`: 100-400ms (depends on file/workspace size)
- `completion`: 50-200ms (depends on context depth)
- `diagnostics`: 100-500ms (full file analysis)
- `rename`: 200-500ms (workspace-wide changes)
- `format`: 100-300ms (depends on file size)

## License

MIT License

## Related Documentation

- **Language Server Protocol Specification**: [microsoft.github.io/language-server-protocol](https://microsoft.github.io/language-server-protocol)
- **Pyright LSP**: [github.com/microsoft/pyright](https://github.com/microsoft/pyright)
- **LSP Clients**: [langserver.org](https://langserver.org)
- **HoloLoom Code Skills**: [../README.md](../README.md)
