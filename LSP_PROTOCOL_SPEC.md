# HoloLoom Language Server Protocol (LSP) Specification

**Status**: Design Document (Ready for Implementation)
**Date**: 2025-11-16
**Version**: 1.0
**Target Editors**: VS Code, Neovim, Emacs, Vim, Sublime Text (any LSP client)

## Executive Summary

This document specifies HoloLoom's Language Server Protocol (LSP) implementation, enabling code intelligence features across any text editor that supports LSP. Unlike traditional language servers focused on syntax and types, HoloLoom LSP leverages:

- **Semantic search** (knowledge graph + embeddings)
- **Neural memory system** (recall relevant code patterns)
- **Agentic reasoning** (multi-step analysis)
- **Code understanding** (entity extraction, relationships)

**Key Innovation**: HoloLoom LSP bridges the gap between semantic code intelligence and traditional LSP, enabling queries like:
- "Show me similar code patterns" (semantic completion)
- "What does this function relate to?" (knowledge graph navigation)
- "Find all usages of this pattern across the codebase" (semantic search)
- "Explain this code in context of the codebase" (agentic reasoning)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Endpoints (Priority Order)](#core-endpoints-priority-order)
3. [Request/Response Specifications](#requestresponse-specifications)
4. [HoloLoom Integration Mapping](#hololoom-integration-mapping)
5. [Configuration & Capabilities](#configuration--capabilities)
6. [Example Workflows](#example-workflows)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Best Practices](#best-practices)

---

## Architecture Overview

### System Diagram

```
Text Editor (VS Code, Neovim, etc.)
    ↓
    ↑← LSP Protocol (JSON-RPC over stdio/TCP)
    ↓
┌─────────────────────────────────────────────┐
│  HoloLoom Language Server (Python)          │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │ LSP Handler Layer (pygls)           │  │
│  │ - Message routing                   │  │
│  │ - Capability negotiation            │  │
│  │ - Request/response formatting       │  │
│  └─────────────────────────────────────┘  │
│               ↓                             │
│  ┌─────────────────────────────────────┐  │
│  │ HoloLoom Core Services              │  │
│  │ - Orchestrator (weaving)            │  │
│  │ - Knowledge Graph (KG)              │  │
│  │ - Memory System (recall/remember)   │  │
│  │ - Agentic Reasoning                 │  │
│  │ - Code Indexing (WorkspaceSpinner)  │  │
│  └─────────────────────────────────────┘  │
│               ↓                             │
│  ┌─────────────────────────────────────┐  │
│  │ Backend Services                    │  │
│  │ - Neo4j (knowledge graph storage)   │  │
│  │ - Qdrant (semantic embeddings)      │  │
│  │ - In-memory fallback                │  │
│  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Protocol Flow

1. **Initialization**: Client requests server capabilities → Server responds with supported features
2. **File Opening**: Client notifies server of file open (`textDocument/didOpen`) → Server indexes file
3. **Live Editing**: Client sends changes (`textDocument/didChange`) → Server updates indexes incrementally
4. **Intelligence Queries**: Client requests completions/hover/definitions → Server queries HoloLoom
5. **Shutdown**: Client requests shutdown → Server cleans up resources

### Key Principles

1. **Semantic-First**: Prioritize semantic understanding over syntactic analysis
2. **Knowledge Graph Centric**: All intelligence flows through the knowledge graph
3. **Graceful Degradation**: Work with or without code analysis plugins
4. **Non-Blocking**: All operations must be async and non-blocking
5. **Incremental Updates**: Support incremental file synchronization
6. **Memory Efficient**: Cache results for repeated queries

---

## Core Endpoints (Priority Order)

### Phase 1: Foundation (MVP - Week 1-2)

#### 1. **initialize** (Server Capability Negotiation)

**Method**: `initialize`
**Direction**: Client → Server
**Purpose**: Negotiate capabilities and configure server

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "processId": 1234,
    "rootUri": "file:///home/user/project",
    "capabilities": {
      "textDocument": {
        "synchronization": {
          "didSave": true,
          "willSave": true
        },
        "completion": {
          "completionItem": {
            "snippetSupport": true
          }
        }
      }
    },
    "initializationOptions": {
      "max_memory_shards": 1000,
      "semantic_search_enabled": true,
      "knowledge_graph_mode": "hybrid",
      "background_indexing": true
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "capabilities": {
      "textDocumentSync": {
        "openClose": true,
        "change": {
          "syncKind": 2
        },
        "save": {
          "includeText": true
        }
      },
      "completionProvider": {
        "resolveProvider": true,
        "triggerCharacters": [".", " ", "$", "@"],
        "commitCharacters": [";", ",", "."],
        "allCommitCharacters": [";", ",", ".", "(", ")"]
      },
      "hoverProvider": true,
      "definitionProvider": true,
      "referencesProvider": true,
      "documentSymbolProvider": true,
      "workspaceSymbolProvider": true,
      "codeActionProvider": true,
      "diagnosticsProvider": {
        "interFileDependencies": true,
        "workspaceDiagnostics": true
      }
    },
    "serverInfo": {
      "name": "HoloLoom",
      "version": "1.0.0"
    }
  }
}
```

**Integration**: No HoloLoom backend call - purely capability declaration

---

#### 2. **textDocument/didOpen** (File Opening)

**Method**: `textDocument/didOpen`
**Direction**: Client → Server (Notification)
**Purpose**: Notify server of newly opened file, begin indexing

**Notification**:
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didOpen",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts",
      "languageId": "typescript",
      "version": 1,
      "text": "export async function authenticate(username: string, password: string) {\n  const user = await db.users.findOne({username});\n  if (!user) throw new Error('User not found');\n  const valid = await compare(password, user.passwordHash);\n  if (!valid) throw new Error('Invalid password');\n  return user;\n}"
    }
  }
}
```

**HoloLoom Integration**:
1. Call `/ingest/file` with file content, language, path
2. Parse code entities (functions, classes, variables)
3. Extract entities and relationships
4. Add to knowledge graph
5. Store in memory shards for retrieval

**Pseudocode**:
```python
@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def on_file_open(ls, params):
    uri = params.text_document.uri
    file_path = uri_to_path(uri)
    language_id = params.text_document.language_id
    content = params.text_document.text

    # Map language ID to code language enum
    language = map_language_id(language_id)

    # Ingest file to knowledge graph
    result = await POST("/ingest/file", {
        "file_path": file_path,
        "language": language,
        "content": content
    })

    # Store in document cache for incremental updates
    cache[uri] = {
        "content": content,
        "version": params.text_document.version,
        "entities": result.entities,
        "parsed": result
    }
```

---

#### 3. **textDocument/didChange** (File Modification)

**Method**: `textDocument/didChange`
**Direction**: Client → Server (Notification)
**Purpose**: Track file changes for incremental indexing

**Notification** (full document sync):
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/didChange",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts",
      "version": 2
    },
    "contentChanges": [
      {
        "range": {
          "start": {"line": 4, "character": 28},
          "end": {"line": 4, "character": 28}
        },
        "rangeLength": 0,
        "text": "Hash"
      }
    ]
  }
}
```

**HoloLoom Integration**:
1. Update internal document cache
2. Parse changed lines for entities
3. Update knowledge graph with new/changed entities
4. Mark stale memory for refresh

**Implementation Note**:
- Full sync (syncKind: 2) - replace entire document
- Incremental (syncKind: 1) - apply range-based changes
- HoloLoom implementation uses full sync (simpler, acceptable for LSP)

---

#### 4. **textDocument/completion** (Code Completion)

**Method**: `textDocument/completion`
**Direction**: Client → Server (Request)
**Purpose**: Provide intelligent code completions (semantic + memory-based)

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "textDocument/completion",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/main.ts"
    },
    "position": {
      "line": 35,
      "character": 21
    },
    "context": {
      "triggerKind": 1,
      "triggerCharacter": "."
    }
  }
}
```

**Response** (Completion Items):
```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": [
    {
      "label": "authenticate",
      "kind": 6,
      "detail": "function authenticate(username: string, password: string)",
      "documentation": {
        "kind": "markdown",
        "value": "Authenticate user with credentials\n\nRelated: db.users.findOne, compare, PasswordHash"
      },
      "sortText": "0_authenticate",
      "filterText": "authenticate",
      "insertText": "authenticate(${1:username}, ${2:password})",
      "insertTextFormat": 2,
      "range": {
        "start": {"line": 35, "character": 18},
        "end": {"line": 35, "character": 21}
      },
      "data": {
        "source": "memory_semantic",
        "confidence": 0.92,
        "source_file": "src/auth.ts:2",
        "context": "function"
      }
    },
    {
      "label": "authenticateWithMFA",
      "kind": 6,
      "detail": "function authenticateWithMFA(username: string, password: string, mfaToken: string)",
      "documentation": {
        "kind": "markdown",
        "value": "Two-factor authentication flow\n\nRelated: authenticate, MFAProvider, verifyToken"
      },
      "sortText": "1_authenticateWithMFA",
      "filterText": "authenticateWithMFA",
      "insertText": "authenticateWithMFA(${1:username}, ${2:password}, ${3:mfaToken})",
      "insertTextFormat": 2,
      "range": {
        "start": {"line": 35, "character": 18},
        "end": {"line": 35, "character": 21}
      },
      "data": {
        "source": "memory_semantic",
        "confidence": 0.85,
        "source_file": "src/auth.ts:25",
        "context": "function"
      }
    },
    {
      "label": "auth",
      "kind": 9,
      "detail": "module auth",
      "documentation": {
        "kind": "markdown",
        "value": "Authentication module\n\nExports: authenticate, authenticateWithMFA, logout"
      },
      "sortText": "2_auth",
      "filterText": "auth",
      "insertText": "auth",
      "range": {
        "start": {"line": 35, "character": 18},
        "end": {"line": 35, "character": 21}
      }
    }
  ]
}
```

**HoloLoom Integration**:

Completion = semantic search + memory recall:

1. **Extract context** from editor:
   - Current line + surrounding 10 lines
   - Current file path (for path-based filtering)
   - Language type
   - Cursor position

2. **Build query** from partial token:
   - Token: "auth"
   - Context: "function call in main.ts"

3. **Call HoloLoom recall** (semantic search):
   ```python
   # Recall similar functions/modules
   memories = await loom.recall(
       query="authenticate functions, auth modules",
       k=20,  # Get top 20 candidates
       filter={"language": "typescript"}
   )
   ```

4. **Score and rank**:
   - Semantic similarity (memory confidence)
   - Popularity (usage count in codebase)
   - Recency (last modification)
   - Contextual relevance (same module, same type)

5. **Format as LSP CompletionItem**:
   - label: Entity name
   - kind: LSP completion item kind (6=function, 9=module, etc.)
   - detail: Signature
   - documentation: Multi-line markdown with relationships
   - data: Metadata for resolveCompletion

**Trigger Characters**:
- `.` : member access (method/property completion)
- ` ` : space (after keyword)
- `$` : template strings
- `@` : decorator access
- `(` : function argument completion (phase 2)

---

#### 5. **textDocument/hover** (Hover Information)

**Method**: `textDocument/hover`
**Direction**: Client → Server (Request)
**Purpose**: Show symbol definition, related knowledge when hovering

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 43,
  "method": "textDocument/hover",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/main.ts"
    },
    "position": {
      "line": 12,
      "character": 24
    }
  }
}
```

**Response** (Hover Information):
```json
{
  "jsonrpc": "2.0",
  "id": 43,
  "result": {
    "contents": {
      "kind": "markdown",
      "value": "```typescript\nfunction authenticate(username: string, password: string): Promise<User>\n```\n\n---\n\n**Authenticates a user with credentials**\n\nVerifies username and password against database, comparing hashed password. Throws on invalid credentials.\n\n**Location**: `src/auth.ts:2`\n\n**Related Code**:\n- Uses: `db.users.findOne()` (line 3)\n- Uses: `compare()` (password hashing, line 5)\n- Called by: `handleLogin()` (src/handlers/login.ts:15)\n- Called by: `apiLogin()` (src/api.ts:45)\n\n**Knowledge Graph Context**:\n- Part of: Authentication subsystem\n- Related patterns: Password validation, User lookup, Async error handling\n- Related functions: `logout()`, `refreshToken()`, `authenticateWithMFA()`\n\n**Metadata**:\n- Created: 2025-11-10\n- Last modified: 2025-11-15 14:32:18\n- Type: Function\n- Language: TypeScript\n- Confidence: 0.92\n"
    },
    "range": {
      "start": {"line": 12, "character": 14},
      "end": {"line": 12, "character": 26}
    }
  }
}
```

**HoloLoom Integration**:

Hover combines symbol info + knowledge graph context:

1. **Identify symbol** at cursor position:
   - Extract word/identifier from code
   - Determine type (function, class, variable, etc.)

2. **Query knowledge graph**:
   - Find node matching symbol name
   - Get entity metadata (location, type, signature)
   - Get relationships (callers, callees, related)

3. **Get semantic context**:
   - Call `/api/recall` with symbol name
   - Get description, patterns, examples
   - Find similar functions

4. **Synthesize markdown**:
   - Code block (signature)
   - Description (from KG or memory)
   - Location and metadata
   - Relationships (IS_A, USES, CALLED_BY, etc.)

---

#### 6. **textDocument/definition** (Go to Definition)

**Method**: `textDocument/definition`
**Direction**: Client → Server (Request)
**Purpose**: Navigate to symbol definition (jump to code location)

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 44,
  "method": "textDocument/definition",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/main.ts"
    },
    "position": {
      "line": 12,
      "character": 24
    }
  }
}
```

**Response** (Location):
```json
{
  "jsonrpc": "2.0",
  "id": 44,
  "result": [
    {
      "uri": "file:///home/user/project/src/auth.ts",
      "range": {
        "start": {"line": 1, "character": 0},
        "end": {"line": 1, "character": 26}
      }
    }
  ]
}
```

**HoloLoom Integration**:

1. **Find symbol** in knowledge graph
2. **Locate definition**:
   - Query KG for entity with exact name
   - Get file path and line number from metadata
3. **Return Location**:
   - URI: file:// URI of definition file
   - Range: Start/end of symbol definition

**Note**: Unlike traditional LSP, HoloLoom can find semantic definitions across the codebase (e.g., interface implementations).

---

### Phase 2: Advanced Search (Week 3-4)

#### 7. **textDocument/references** (Find References)

**Method**: `textDocument/references`
**Direction**: Client → Server (Request)
**Purpose**: Find all usages of a symbol (with semantic understanding)

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 45,
  "method": "textDocument/references",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts"
    },
    "position": {
      "line": 2,
      "character": 18
    },
    "context": {
      "includeDeclaration": true
    }
  }
}
```

**Response** (Locations):
```json
{
  "jsonrpc": "2.0",
  "id": 45,
  "result": [
    {
      "uri": "file:///home/user/project/src/auth.ts",
      "range": {
        "start": {"line": 2, "character": 18},
        "end": {"line": 2, "character": 28}
      }
    },
    {
      "uri": "file:///home/user/project/src/handlers/login.ts",
      "range": {
        "start": {"line": 15, "character": 10},
        "end": {"line": 15, "character": 20}
      }
    },
    {
      "uri": "file:///home/user/project/src/api.ts",
      "range": {
        "start": {"line": 45, "character": 5},
        "end": {"line": 45, "character": 15}
      }
    }
  ]
}
```

**HoloLoom Integration**:

1. Query knowledge graph: Find all edges with CALLED_BY relationship
2. Return list of file locations where symbol is used
3. Can be semantic (find similar patterns) or exact (only same symbol)

---

#### 8. **workspace/symbol** (Workspace Symbol Search)

**Method**: `workspace/symbol`
**Direction**: Client → Server (Request)
**Purpose**: Semantic search across entire codebase

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 46,
  "method": "workspace/symbol",
  "params": {
    "query": "user authentication"
  }
}
```

**Response** (Symbol Information):
```json
{
  "jsonrpc": "2.0",
  "id": 46,
  "result": [
    {
      "name": "authenticate",
      "kind": 6,
      "location": {
        "uri": "file:///home/user/project/src/auth.ts",
        "range": {
          "start": {"line": 2, "character": 0},
          "end": {"line": 10, "character": 1}
        }
      },
      "containerName": "auth",
      "data": {
        "confidence": 0.96,
        "type": "function"
      }
    },
    {
      "name": "User",
      "kind": 5,
      "location": {
        "uri": "file:///home/user/project/src/types.ts",
        "range": {
          "start": {"line": 15, "character": 0},
          "end": {"line": 25, "character": 1}
        }
      },
      "containerName": "types",
      "data": {
        "confidence": 0.92,
        "type": "interface"
      }
    },
    {
      "name": "verifyToken",
      "kind": 6,
      "location": {
        "uri": "file:///home/user/project/src/auth.ts",
        "range": {
          "start": {"line": 40, "character": 0},
          "end": {"line": 50, "character": 1}
        }
      },
      "containerName": "auth",
      "data": {
        "confidence": 0.85,
        "type": "function"
      }
    }
  ]
}
```

**HoloLoom Integration**:

1. Parse query: "user authentication"
2. Call `/api/recall` with semantic search:
   ```python
   memories = await loom.recall("user authentication", k=50)
   ```
3. Extract entities from returned memories
4. Format as SymbolInformation array
5. Rank by confidence and relevance

---

#### 9. **documentSymbol** (Document Symbols)

**Method**: `textDocument/documentSymbol`
**Direction**: Client → Server (Request)
**Purpose**: Get all symbols (functions, classes, etc.) in current document

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 47,
  "method": "textDocument/documentSymbol",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts"
    }
  }
}
```

**Response** (Document Symbols):
```json
{
  "jsonrpc": "2.0",
  "id": 47,
  "result": [
    {
      "name": "authenticate",
      "kind": 6,
      "range": {
        "start": {"line": 2, "character": 0},
        "end": {"line": 10, "character": 1}
      },
      "selectionRange": {
        "start": {"line": 2, "character": 18},
        "end": {"line": 2, "character": 30}
      },
      "children": []
    },
    {
      "name": "verifyToken",
      "kind": 6,
      "range": {
        "start": {"line": 40, "character": 0},
        "end": {"line": 50, "character": 1}
      },
      "selectionRange": {
        "start": {"line": 40, "character": 18},
        "end": {"line": 40, "character": 29}
      },
      "children": []
    }
  ]
}
```

**HoloLoom Integration**:

1. Query document cache (from didOpen/didChange)
2. Extract entities from parsing
3. Format as DocumentSymbol array
4. Include relationship information in extended fields

---

### Phase 3: Semantic Understanding (Week 5-6)

#### 10. **codeAction** (Quick Fixes & Refactoring)

**Method**: `textDocument/codeAction`
**Direction**: Client → Server (Request)
**Purpose**: Provide code improvements, fixes, refactoring suggestions

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 48,
  "method": "textDocument/codeAction",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/main.ts"
    },
    "range": {
      "start": {"line": 15, "character": 0},
      "end": {"line": 25, "character": 0}
    },
    "context": {
      "diagnostics": [],
      "only": ["quickfix", "refactor"]
    }
  }
}
```

**Response** (Code Actions):
```json
{
  "jsonrpc": "2.0",
  "id": 48,
  "result": [
    {
      "title": "Extract to function",
      "kind": "refactor.extract",
      "edit": {
        "changes": {
          "file:///home/user/project/src/main.ts": [
            {
              "range": {
                "start": {"line": 15, "character": 0},
                "end": {"line": 25, "character": 0}
              },
              "newText": "const extracted = extractFunction();\n"
            }
          ],
          "file:///home/user/project/src/utils.ts": [
            {
              "range": {
                "start": {"line": 100, "character": 0},
                "end": {"line": 100, "character": 0}
              },
              "newText": "\nexport function extractFunction() {\n  // Extracted code here\n}\n"
            }
          ]
        }
      },
      "data": {
        "action_type": "extract_function",
        "confidence": 0.85
      }
    },
    {
      "title": "Replace with pattern: Error handling",
      "kind": "quickfix",
      "edit": {
        "changes": {
          "file:///home/user/project/src/main.ts": [
            {
              "range": {
                "start": {"line": 17, "character": 2},
                "end": {"line": 19, "character": 0}
              },
              "newText": "try {\n    // existing code\n  } catch (error) {\n    logger.error('Error occurred', error);\n    throw error;\n  }\n"
            }
          ]
        }
      },
      "data": {
        "action_type": "apply_pattern",
        "pattern_id": "error_handling_try_catch",
        "confidence": 0.90
      }
    }
  ]
}
```

**HoloLoom Integration**:

1. Query code patterns from knowledge graph
2. Detect code smell/improvements in selection
3. Suggest refactoring based on:
   - Common patterns in codebase
   - Best practices from memory system
   - Semantic similarity (what do similar functions do?)
4. Provide workspace edits to apply changes

---

#### 11. **textDocument/diagnostic** (Linting & Error Detection)

**Method**: `textDocument/diagnostic` (Publishing Diagnostics)
**Direction**: Server → Client (Notification)
**Purpose**: Publish code issues, warnings, hints

**Notification** (Server sends periodically):
```json
{
  "jsonrpc": "2.0",
  "method": "textDocument/publishDiagnostics",
  "params": {
    "uri": "file:///home/user/project/src/main.ts",
    "diagnostics": [
      {
        "range": {
          "start": {"line": 15, "character": 10},
          "end": {"line": 15, "character": 20}
        },
        "severity": 1,
        "code": "missing_error_handling",
        "source": "hololoom",
        "message": "Missing error handling in async operation",
        "relatedInformation": [
          {
            "location": {
              "uri": "file:///home/user/project/src/handlers.ts",
              "range": {
                "start": {"line": 42, "character": 5},
                "end": {"line": 42, "character": 15}
              }
            },
            "message": "Similar pattern handled here with try/catch"
          }
        ]
      },
      {
        "range": {
          "start": {"line": 20, "character": 0},
          "end": {"line": 20, "character": 10}
        },
        "severity": 2,
        "code": "unused_variable",
        "source": "hololoom",
        "message": "Variable declared but never used",
        "relatedInformation": []
      }
    ]
  }
}
```

**HoloLoom Integration**:

1. When file changes, run analysis:
   - ML logic detector: Detect logic errors
   - Pattern matcher: Compare against codebase patterns
   - Best practices: Check code quality

2. Return diagnostics with:
   - Location (range)
   - Severity (error, warning, info, hint)
   - Message
   - Related information (similar patterns in codebase)

3. Publish via `textDocument/publishDiagnostics` notification

---

#### 12. **textDocument/semanticTokens** (Semantic Highlighting)

**Method**: `textDocument/semanticTokens/full`
**Direction**: Client → Server (Request)
**Purpose**: Provide semantic-aware syntax highlighting (entity types, confidence levels)

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 49,
  "method": "textDocument/semanticTokens/full",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts"
    }
  }
}
```

**Response** (Semantic Tokens):
```json
{
  "jsonrpc": "2.0",
  "id": 49,
  "result": {
    "resultId": "auth.ts:1",
    "data": [
      3, 0, 11, 6, 0,
      0, 12, 12, 11, 0,
      1, 18, 30, 6, 0,
      0, 32, 8, 4, 0,
      0, 10, 6, 3, 0,
      0, 8, 8, 3, 0
    ]
  }
}
```

**Legend**:
```
Token types: ["keyword", "variable", "type", "function", "class", "parameter", "comment", "string"]
Token modifiers: ["declaration", "definition", "semantic", "readonly", "high_confidence", "low_confidence"]
```

**HoloLoom Integration**:

1. Extract token positions and types from cached parse
2. Enhance with semantic information:
   - Confidence levels (high/low based on KG)
   - Entity relationships
   - Semantic types (function, class, module, etc.)
3. Encode as LSP semantic tokens (delta encoding)

---

### Phase 4: Agentic Features (Week 7-8)

#### 13. **hololoom/explain** (Agentic Code Explanation)

**Method**: `hololoom/explain` (Custom LSP Extension)
**Direction**: Client → Server (Request)
**Purpose**: Multi-step explanation of code using agentic reasoning

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 50,
  "method": "hololoom/explain",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/auth.ts"
    },
    "range": {
      "start": {"line": 2, "character": 0},
      "end": {"line": 10, "character": 1}
    },
    "reasoning_mode": "research",
    "max_steps": 3
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 50,
  "result": {
    "explanation": "This function authenticates a user by:\n1. Looking up the user in the database by username\n2. Comparing the provided password with the stored hash\n3. Returning the user object on success or throwing an error\n\nThe function follows security best practices by using hashed password comparison instead of plain-text comparison.",
    "confidence": 0.94,
    "steps": [
      {
        "step": 1,
        "query": "What is the purpose of this authenticate function?",
        "finding": "User authentication handler that validates credentials"
      },
      {
        "step": 2,
        "query": "What is db.users.findOne and how is it used?",
        "finding": "Database lookup - retrieves user by username from users table"
      },
      {
        "step": 3,
        "query": "What password security pattern is used here?",
        "finding": "bcrypt compare() function - securely validates hashed passwords"
      }
    ],
    "related_patterns": [
      "logout function (src/auth.ts:40)",
      "User type definition (src/types.ts:15)",
      "Password hashing on registration (src/handlers/register.ts:20)"
    ]
  }
}
```

**HoloLoom Integration**:

Uses agentic orchestrator in RESEARCH mode:

```python
@server.feature("hololoom/explain")
async def on_explain(ls, params):
    file_path = uri_to_path(params.text_document.uri)
    start_line = params.range.start.line
    end_line = params.range.end.line

    # Extract code selection
    code_selection = cached_content[params.text_document.uri][start_line:end_line]

    # Call agentic orchestrator
    from HoloLoom.agentic import ReasoningMode
    result = await orchestrator.reason(
        Query(text=f"Explain this code:\n{code_selection}"),
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )

    return format_explanation(result)
```

---

#### 14. **hololoom/suggest_pattern** (Pattern Suggestion)

**Method**: `hololoom/suggest_pattern` (Custom LSP Extension)
**Direction**: Client → Server (Request)
**Purpose**: Suggest common patterns from codebase for current task

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 51,
  "method": "hololoom/suggest_pattern",
  "params": {
    "textDocument": {
      "uri": "file:///home/user/project/src/handlers.ts"
    },
    "position": {
      "line": 20,
      "character": 0
    },
    "context": "function"
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 51,
  "result": {
    "patterns": [
      {
        "name": "Error Handling with Try/Catch",
        "confidence": 0.95,
        "frequency": 42,
        "example": "try {\n  const result = await someAsync();\n  return result;\n} catch (error) {\n  logger.error('Operation failed', error);\n  throw new CustomError('User-friendly message');\n}",
        "location": "src/handlers.ts:15",
        "description": "Most common pattern in codebase for async error handling"
      },
      {
        "name": "Request Validation",
        "confidence": 0.88,
        "frequency": 38,
        "example": "if (!request.body.username || !request.body.password) {\n  throw new ValidationError('Missing required fields');\n}",
        "location": "src/handlers/login.ts:5",
        "description": "Input validation pattern used in request handlers"
      },
      {
        "name": "Logging Pattern",
        "confidence": 0.92,
        "frequency": 51,
        "example": "logger.info(`Processing ${action} for user ${userId}`);\n// ... operation ...\nlogger.info(`Completed ${action} successfully`);",
        "location": "src/utils/logger.ts:10",
        "description": "Structured logging pattern with context"
      }
    ]
  }
}
```

**HoloLoom Integration**:

1. Query knowledge graph for patterns
2. Find most common patterns in codebase
3. Rank by frequency and confidence
4. Return with code examples

---

---

## Request/Response Specifications

### JSON-RPC Format

All LSP messages follow JSON-RPC 2.0 specification:

```json
{
  "jsonrpc": "2.0",
  "id": <integer or string>,
  "method": "<method name>",
  "params": <object or array>,
  "result": <success response>,
  "error": {
    "code": <integer>,
    "message": "<error message>",
    "data": <optional error data>
  }
}
```

### Error Codes

| Code | Meaning | Usage |
|------|---------|-------|
| -32700 | Parse Error | JSON parsing failed |
| -32600 | Invalid Request | Malformed JSON-RPC |
| -32601 | Method Not Found | Unknown LSP method |
| -32602 | Invalid Params | Method parameters invalid |
| -32603 | Internal Error | Server error during processing |
| -32000 to -32099 | Server Error | HoloLoom-specific errors |

**HoloLoom-Specific Error Codes**:
- `-32001`: Knowledge graph unavailable
- `-32002`: Memory backend offline
- `-32003`: Orchestrator not ready
- `-32004`: File not indexed
- `-32005`: Invalid query

### Position Format

```json
{
  "line": 0,
  "character": 0
}
```

- **line**: Zero-based line number
- **character**: Zero-based character offset (UTF-16)

### Range Format

```json
{
  "start": {"line": 0, "character": 0},
  "end": {"line": 10, "character": 50}
}
```

### URI Format

```
file:///absolute/path/to/file.ext
```

- **Scheme**: `file://`
- **Path**: Absolute path (Unix-style forward slashes)
- **Encoding**: RFC 8089 compliant

---

## HoloLoom Integration Mapping

### API Endpoint Mapping

| LSP Feature | HoloLoom Endpoint | Purpose |
|-------------|-------------------|---------|
| textDocument/completion | `/api/recall` | Semantic search for completion candidates |
| textDocument/hover | `/api/graph/data` + `/api/recall` | Symbol metadata + relationships |
| textDocument/definition | `/api/graph/data` | Find symbol location in KG |
| workspace/symbol | `/api/recall` | Semantic search across codebase |
| textDocument/references | `/api/graph/data` | Query CALLED_BY edges in KG |
| textDocument/diagnostic | `/detect/logic` + `/detect/slop` | Code analysis |
| codeAction | `/api/recall` (pattern search) | Suggest patterns + refactoring |
| hololoom/explain | `/query` (agentic reasoning) | Multi-step code explanation |
| hololoom/suggest_pattern | `/api/graph/data` (pattern query) | Pattern frequency analysis |

### Data Flow Diagram

```
LSP Request (from editor)
    ↓
[Message Routing Layer (pygls)]
    ↓
[Extract: file, position, context, query]
    ↓
┌──────────────────────────────────────┐
│ HoloLoom Query Dispatcher            │
│                                      │
│ ┌──────────────────────────────────┐ │
│ │ Document Cache                   │ │
│ │ (didOpen/didChange events)       │ │
│ └──────────────────────────────────┘ │
│             ↓                         │
│ ┌──────────────────────────────────┐ │
│ │ Determine Query Type:            │ │
│ │ - Completion: semantic search    │ │
│ │ - Definition: KG lookup          │ │
│ │ - References: KG traversal       │ │
│ │ - Diagnostic: Code analysis      │ │
│ └──────────────────────────────────┘ │
│             ↓                         │
│ ┌──────────────────────────────────┐ │
│ │ Call HoloLoom Backend             │ │
│ │ - /api/recall (semantic)          │ │
│ │ - /api/graph/data (structural)    │ │
│ │ - /detect/* (analysis)            │ │
│ │ - /query (agentic)                │ │
│ └──────────────────────────────────┘ │
│             ↓                         │
│ ┌──────────────────────────────────┐ │
│ │ Format Response:                 │ │
│ │ - CompletionItem[]               │ │
│ │ - Hover                          │ │
│ │ - Location[]                     │ │
│ │ - Diagnostic[]                   │ │
│ └──────────────────────────────────┘ │
└──────────────────────────────────────┘
    ↓
[LSP Response Formatting]
    ↓
JSON-RPC Reply (back to editor)
```

### Caching Strategy

**In-Memory Caches** (pygls server):

```python
document_cache = {
    "file:///path/to/file.ts": {
        "content": "...",
        "version": 5,
        "parsed": ParseResult(...),
        "entities": [Entity(...)],
        "last_update": timestamp
    }
}

completion_cache = {
    ("file:///path", "authenticate"): [
        CompletionItem(...),
        CompletionItem(...),
    ],
    "expires": timestamp  # 5 minute TTL
}

hover_cache = {
    ("file:///path", (35, 21)): HoverInfo(...),
    "expires": timestamp  # 10 minute TTL
}
```

**Cache Invalidation**:

- `textDocument/didChange` → Invalidate document cache
- `textDocument/didChange` → Invalidate completion cache for affected lines
- New file opened → Refresh file's symbol cache
- Workspace ingestion → Full cache clear

---

## Configuration & Capabilities

### Server Capabilities

The server declares support for features during initialization:

```python
server.capabilities = ServerCapabilities(
    text_document_sync=TextDocumentSyncOptions(
        open_close=True,
        change=TextDocumentSyncKind.FULL,
        save=SaveOptions(includeText=True)
    ),
    completion_provider=CompletionOptions(
        trigger_characters=[".", " ", "$", "@"],
        all_commit_characters=[";", ",", "."],
        resolve_provider=True
    ),
    hover_provider=True,
    definition_provider=True,
    references_provider=True,
    workspace_symbol_provider=True,
    document_symbol_provider=True,
    code_action_provider=True,
    semantic_tokens_provider=SemanticTokensOptions(
        legend=SemanticTokensLegend(
            token_types=["keyword", "variable", "type", "function", "class"],
            token_modifiers=["declaration", "definition", "semantic"]
        ),
        full=True,
        range=False
    ),
    diagnostic_provider=DiagnosticOptions(
        inter_file_dependencies=True,
        workspace_diagnostics=True
    )
)
```

### Client Capabilities

Clients (editors) declare their features in `initialize` request:

```json
{
  "capabilities": {
    "textDocument": {
      "synchronization": {
        "didSave": true,
        "willSave": true
      },
      "completion": {
        "completionItem": {
          "snippetSupport": true,
          "labelDetailsSupport": true,
          "insertReplaceSupport": true
        }
      },
      "hover": {
        "contentFormat": ["markdown", "plaintext"]
      }
    },
    "workspace": {
      "symbol": {
        "symbolKind": {
          "valueSet": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }
      }
    }
  }
}
```

### Initialization Options

Clients can pass custom options during `initialize`:

```json
{
  "initializationOptions": {
    "max_memory_shards": 1000,
    "semantic_search_enabled": true,
    "knowledge_graph_mode": "hybrid",
    "background_indexing": true,
    "analysis_engines": {
      "ml_logic": true,
      "pattern_suggestion": true,
      "code_quality": true
    },
    "performance": {
      "max_completion_results": 20,
      "completion_timeout_ms": 500,
      "hover_timeout_ms": 1000,
      "diagnostic_interval_ms": 5000
    },
    "workspace": {
      "indexing_languages": ["typescript", "python", "javascript"],
      "exclude_patterns": ["**/node_modules/**", "**/.venv/**", "**/.git/**"]
    }
  }
}
```

---

## Example Workflows

### Workflow 1: Code Completion

**Scenario**: User types `thom` in editor, trigger character is next character

```
User types in editor:
  const handler = thom

VS Code sends textDocument/completion request:
  {
    "position": {"line": 5, "character": 28},
    "context": {"triggerKind": 1}  // Invoked (Ctrl+Space)
  }

HoloLoom Server:
  1. Extract partial token: "thom"
  2. Get document context (surrounding 10 lines)
  3. Build query: "thom" + context
  4. Call /api/recall("thompson sampling, thom*", k=20)
  5. Score results by:
     - Edit distance to "thom"
     - Semantic similarity
     - Frequency in codebase
     - Recency
  6. Return top 5 CompletionItems

VS Code displays:
  [✓] Thompson Sampling (0.95 confidence)
  [ ] throttle (0.82 confidence)
  [ ] throw (built-in keyword)

User selects "Thompson Sampling"
  → Inserts function call
  → Editor triggers didChange notification
  → Server updates knowledge graph
```

---

### Workflow 2: Hover + Go-to-Definition

**Scenario**: User hovers over function name, clicks to go to definition

```
User hovers over "authenticate" at line 12, char 24:

VS Code sends textDocument/hover:
  {
    "position": {"line": 12, "character": 24}
  }

HoloLoom Server:
  1. Extract token: "authenticate"
  2. Query KG: Find node "authenticate"
  3. Get metadata: type=function, location=auth.ts:2
  4. Get relationships: CALLED_BY=[handler.ts:15, api.ts:45]
  5. Get similar functions: [authenticateWithMFA, login, verifyToken]
  6. Build markdown documentation
  7. Return Hover with markdown content

VS Code displays:
  ┌─────────────────────────────────────┐
  │ function authenticate(...)          │
  │ Authenticates a user...             │
  │ Related: db.users.findOne...        │
  └─────────────────────────────────────┘

User clicks "Go to Definition" (F12):

VS Code sends textDocument/definition:
  {
    "position": {"line": 12, "character": 24}
  }

HoloLoom Server:
  1. Extract token: "authenticate"
  2. Query KG: Find "authenticate" node
  3. Get location metadata: uri=auth.ts, line=2
  4. Return Location object
  5. VS Code jumps to auth.ts:2
```

---

### Workflow 3: Semantic Search Across Workspace

**Scenario**: User opens command palette and searches for functionality

```
User opens Command Palette (Ctrl+Shift+P):
  > HoloLoom: Find Code Pattern
  > user authentication

VS Code sends workspace/symbol:
  {
    "query": "user authentication"
  }

HoloLoom Server:
  1. Parse query: "user authentication"
  2. Build embedding of query
  3. Call /api/recall("user authentication", k=50)
  4. Filter results by type (functions, classes, etc.)
  5. Score by:
     - Semantic similarity to query
     - Frequency/popularity
     - Recency
  6. Return SymbolInformation[]

VS Code displays hierarchical results:
  🔍 user authentication
  ├─ authenticate (src/auth.ts:2) [0.96]
  ├─ User (src/types.ts:15) [0.92]
  ├─ verifyToken (src/auth.ts:40) [0.85]
  ├─ handleLogin (src/handlers.ts:15) [0.82]
  └─ authenticateWithMFA (src/auth.ts:60) [0.78]

User clicks on "authenticate":
  → VS Code jumps to src/auth.ts:2
```

---

### Workflow 4: Code Analysis & Diagnostics

**Scenario**: User saves file, server analyzes for issues

```
User saves file (Ctrl+S):

VS Code sends textDocument/didChange with final content

HoloLoom Server (asynchronously):
  1. Parse file content
  2. Run analysis engines:
     - ML Logic Detector: Check for logic errors
     - Pattern Analyzer: Compare against codebase patterns
     - Best Practices: Check code quality
  3. Build list of Diagnostics:
     - Missing error handling (line 15)
     - Unused variable (line 20)
     - Pattern mismatch (line 25)
  4. For each diagnostic, find related patterns:
     - Query KG for similar patterns
     - Get examples from codebase
  5. Publish diagnostics

VS Code displays diagnostics inline:
  Line 15: ⚠️ Missing error handling in async operation
    Hint: Similar pattern at handlers.ts:42

  Line 20: ℹ️ Variable declared but never used

  Line 25: 💡 Pattern inconsistency
    Suggestion: Use try/catch pattern (see error_handling.ts:5)

User hovers over warning:
  → Shows code fix suggestion
  → User applies quick fix
  → Editor triggers didChange
  → Server updates analysis
```

---

### Workflow 5: Agentic Code Explanation

**Scenario**: User requests detailed explanation of function

```
User selects function code in editor:
  export async function authenticate(username, password) {
    const user = await db.users.findOne({username});
    if (!user) throw new Error('User not found');
    const valid = await compare(password, user.passwordHash);
    if (!valid) throw new Error('Invalid password');
    return user;
  }

User opens Command Palette:
  > HoloLoom: Explain Selection

VS Code sends hololoom/explain:
  {
    "range": {
      "start": {"line": 2, "character": 0},
      "end": {"line": 10, "character": 1}
    },
    "reasoning_mode": "research"
  }

HoloLoom Server (agentic reasoning):
  Step 1: Query - "What is the purpose of this authenticate function?"
    Finding: User authentication handler

  Step 2: Query - "What is db.users.findOne?"
    Finding: Database user lookup by username

  Step 3: Query - "What is the compare() function?"
    Finding: bcrypt password comparison for security

  Synthesized Explanation:
    "This function authenticates a user by looking up their
     username in the database and comparing the provided password
     with the stored hash. It follows security best practices by
     using bcrypt instead of plain-text comparison."

VS Code displays explanation in hover/sidebar:
  📖 Code Explanation
  ─────────────────

  This function authenticates a user by:
  1. Looking up the user by username
  2. Throwing error if not found
  3. Comparing passwords using bcrypt
  4. Returning user or throwing error

  Security: Uses hashed password comparison
  Related: logout(), User type, bcrypt lib
```

---

## Implementation Roadmap

### Architecture Prerequisites

Before implementing LSP endpoints, establish:

1. **Document Cache System**
   ```python
   class DocumentCache:
       def __init__(self):
           self.documents = {}  # uri -> {content, version, parsed}

       def open(self, uri, content, language_id):
           # Parse code, extract entities
           # Store in cache
           pass

       def change(self, uri, changes):
           # Apply incremental/full changes
           # Reparse affected sections
           pass

       def close(self, uri):
           # Clean up document
           pass
   ```

2. **HoloLoom Query Dispatcher**
   ```python
   class HoloLoomDispatcher:
       async def query_completion(self, file_path, position, context):
           # Convert LSP request to HoloLoom API call
           # /api/recall with semantic search

       async def query_definition(self, file_path, symbol_name):
           # /api/graph/data to find symbol location

       async def query_hover(self, file_path, symbol_name):
           # /api/graph/data + /api/recall for metadata

       async def analyze_file(self, file_path, content):
           # /detect/logic + /detect/slop for diagnostics
   ```

3. **Result Formatting**
   ```python
   class ResultFormatter:
       def to_completion_item(self, memory, score):
           # Convert memory to CompletionItem

       def to_location(self, entity_location):
           # Convert entity location to LSP Location

       def to_diagnostic(self, issue):
           # Convert code analysis issue to Diagnostic
   ```

### Phase 1 Implementation (MVP)

**Duration**: 2 weeks
**Endpoints**: 6 core endpoints
**Target**: Basic code intelligence

**Files to Create**:
```
HoloLoom/server/lsp/
├── __init__.py
├── server.py              # Main LSP server (pygls setup)
├── handlers.py            # LSP request handlers
├── dispatcher.py          # HoloLoom query dispatcher
├── cache.py               # Document cache
├── formatter.py           # Result formatting
└── types.py               # LSP type extensions
```

**Implementation Order**:
1. Setup pygls server with initialize/shutdown
2. Implement didOpen/didChange/didClose (file tracking)
3. Implement completion (semantic search)
4. Implement hover (KG metadata)
5. Implement definition (KG lookup)
6. Implement workspace/symbol (semantic search)

**Testing**:
- Unit tests for formatter
- Integration tests with mock HoloLoom APIs
- Manual testing with VS Code LSP client

---

### Phase 2 Implementation (Advanced)

**Duration**: 2 weeks
**Endpoints**: 4 advanced endpoints
**Target**: Code analysis and refactoring

**New Endpoints**:
1. textDocument/references (KG traversal)
2. textDocument/documentSymbol (entity extraction)
3. codeAction (pattern suggestions)
4. textDocument/diagnostic (code analysis)

**Features**:
- Incremental diagnostics publishing
- Code action quick fixes
- Pattern matching and suggestions

---

### Phase 3 Implementation (Agentic)

**Duration**: 2 weeks
**Endpoints**: 2-3 custom endpoints
**Target**: Intelligent reasoning

**New Endpoints**:
1. hololoom/explain (agentic explanation)
2. hololoom/suggest_pattern (pattern recommendation)
3. hololoom/semantic_refactor (refactoring with reasoning)

**Features**:
- Multi-step agentic reasoning
- Pattern analysis and suggestions
- Semantic-aware refactoring

---

### Development Checklist

- [ ] Phase 1
  - [ ] pygls server setup
  - [ ] initialize/shutdown handlers
  - [ ] didOpen/didChange/didClose handlers
  - [ ] Document cache implementation
  - [ ] HoloLoom dispatcher (semantic queries)
  - [ ] Completion endpoint (basic)
  - [ ] Hover endpoint (basic)
  - [ ] Definition endpoint
  - [ ] workspace/symbol endpoint
  - [ ] Unit tests (80% coverage)
  - [ ] Integration tests with mock APIs
  - [ ] Documentation and examples

- [ ] Phase 2
  - [ ] References endpoint
  - [ ] Document symbol endpoint
  - [ ] Code action endpoint
  - [ ] Diagnostic publishing
  - [ ] Enhanced completion (snippets)
  - [ ] Semantic tokens (highlighting)
  - [ ] Integration tests (100% coverage)

- [ ] Phase 3
  - [ ] Custom hololoom/explain endpoint
  - [ ] Custom hololoom/suggest_pattern endpoint
  - [ ] Agentic integration
  - [ ] Performance optimization
  - [ ] Production deployment guide

---

## Best Practices

### Performance

1. **Async/Await**: All I/O must be async
   ```python
   @server.feature(TEXT_DOCUMENT_COMPLETION)
   async def on_completion(params):
       # Must be async
       results = await hololoom_dispatcher.query_completion(...)
       return results
   ```

2. **Timeouts**: Set reasonable timeouts for all queries
   ```python
   try:
       result = await asyncio.wait_for(
           query_task,
           timeout=1.0  # 1 second max
       )
   except asyncio.TimeoutError:
       return []  # Return empty results on timeout
   ```

3. **Caching**: Cache results aggressively
   ```python
   @lru_cache(maxsize=1000)
   async def get_symbol_info(symbol_name):
       # Cache symbol lookups
       return await kg.find_symbol(symbol_name)
   ```

4. **Incremental Processing**: Process only changed portions
   ```python
   def on_change(uri, changes):
       # Only reparse changed lines
       for change in changes:
           affected_lines = change.range.start.line...end.line
           reparse_lines(affected_lines)
   ```

### Error Handling

1. **Graceful Degradation**: Continue working with partial data
   ```python
   try:
       results = await hololoom_api.query(...)
   except HoloLoomError:
       # Fallback to basic completion
       return get_basic_completions()
   ```

2. **User-Friendly Errors**: Explain what went wrong
   ```python
   if not results:
       return {
           "message": "No suggestions available",
           "reason": "Knowledge graph temporarily unavailable"
       }
   ```

3. **Logging**: Log all errors for debugging
   ```python
   logger.error(f"Query failed: {query_text}", exc_info=True)
   ```

### Code Quality

1. **Type Hints**: Use LSP types from `lsprotocol`
   ```python
   from lsprotocol.types import (
       TextDocumentPositionParams,
       CompletionList,
       CompletionItem
   )

   async def complete(
       params: TextDocumentPositionParams
   ) -> CompletionList:
       ...
   ```

2. **Documentation**: Document every endpoint
   ```python
   @server.feature(TEXT_DOCUMENT_COMPLETION)
   async def on_completion(params: CompletionParams) -> CompletionList:
       """
       Provide intelligent code completions using semantic search.

       Queries HoloLoom knowledge graph for similar entities and returns
       ranked completion items with confidence scores.
       """
   ```

3. **Testing**: Write tests for all endpoints
   ```python
   @pytest.mark.asyncio
   async def test_completion_returns_items():
       lsp = HoloLoomLanguageServer()
       params = CompletionParams(...)
       result = await lsp.complete(params)
       assert len(result.items) > 0
       assert result.items[0].label == "expected_label"
   ```

---

## Appendix: Setting Up Development Environment

### Installation

```bash
# Clone repository
git clone https://github.com/user/hololoom.git
cd hololoom

# Install dependencies
pip install -r requirements.txt
pip install pygls lsprotocol

# For development
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

### Starting LSP Server

```python
# server.py
from pygls.server import LanguageServer
from pygls.lsp.types import *

server = LanguageServer("hololoom", "1.0.0")

@server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls, params):
    # Handle file open
    pass

if __name__ == "__main__":
    server.start_tcp("localhost", 8080)
```

### VS Code Client Setup

```json
// .vscode/settings.json
{
  "languageServerExample.trace.server": "verbose",
  "[python]": {
    "defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
  }
}
```

### Testing with `lsp-devtools`

```bash
# Install LSP devtools
pip install lsp-devtools

# Start server with logging
lsp-devtools agent --log-file=lsp.log python server.py

# View trace
lsp-devtools recorder --port=9001
```

---

## References

- [Language Server Protocol Specification 3.17](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
- [pygls Documentation](https://pygls.readthedocs.io/)
- [VS Code LSP Extension Guide](https://code.microsoft.com/api/language-extensions/language-server-extension-guide)
- [HoloLoom Architecture Documentation](ARCHITECTURE_VISUAL_MAP.md)
- [HoloLoom Agentic Reasoning](agentic/README.md)

---

**Document Status**: Ready for Implementation Review
**Last Updated**: 2025-11-16
**Next Steps**: Code review, team discussion, Phase 1 implementation kickoff
