# Squad - HoloLoom Code Intelligence for VS Code

AI-powered code intelligence extension that brings HoloLoom's advanced reasoning capabilities to VS Code.

## 🚀 Features

### Code Understanding
- **Explain Code**: Get detailed explanations of selected code with concepts, patterns, and improvements
- **Find Similar Code**: Discover similar patterns and algorithms across your workspace
- **Code Context Analysis**: AST-based parsing extracts function signatures, classes, and dependencies

### Code Generation
- **Generate Unit Tests**: Automatically create comprehensive test suites (pytest, unittest, jest, mocha)
- **Add Documentation**: Generate docstrings, JSDoc, or Markdown documentation for your code
- **Smart Refactoring**: Get AI-powered suggestions for code smells, performance, and best practices

### Code Review
- **Review Changes**: Analyze git diffs with AI-powered insights
- **Detect Logic Errors**: ML-based detection of subtle bugs (infinite loops, null dereferencing, etc.)
- **Security Analysis**: Identify potential security issues and vulnerabilities

### Workspace Intelligence
- **Workspace Indexing**: Build a knowledge graph of your entire codebase
- **Semantic Search**: Find code by meaning, not just keywords
- **Incremental Caching**: Fast responses with SQLite-backed embedding cache

## 📦 Installation

### Prerequisites

1. **HoloLoom Backend** (required):
   ```bash
   cd HoloLoom/server
   python agentic_api.py
   # Server runs on http://localhost:8000
   ```

2. **VS Code**: Version 1.80.0 or higher

### Install Extension

#### From VSIX (Recommended)
```bash
# Build extension
cd squad
npm install
npm run compile
vsce package  # Creates squad-1.0.0.vsix

# Install in VS Code
code --install-extension squad-1.0.0.vsix
```

#### From Source
1. Clone repository
2. Open `squad/` in VS Code
3. Press F5 to launch extension in development mode

## 🎯 Usage

### Quick Start

1. **Start HoloLoom backend**:
   ```bash
   cd HoloLoom/server
   python agentic_api.py
   ```

2. **Select code** in your editor

3. **Right-click** → Choose Squad command:
   - Explain This Code (`Ctrl+Alt+E`)
   - Find Similar Code (`Ctrl+Alt+F`)
   - Generate Unit Tests (`Ctrl+Alt+T`)
   - Add Documentation
   - Suggest Refactorings
   - Review Changes

4. **View results** in side panel or new tab

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `Squad: Explain This Code` | `Ctrl+Alt+E` | Detailed code explanation |
| `Squad: Find Similar Code` | `Ctrl+Alt+F` | Search for similar patterns |
| `Squad: Generate Unit Tests` | `Ctrl+Alt+T` | Create test suite |
| `Squad: Add Documentation` | - | Generate docstrings |
| `Squad: Suggest Refactorings` | - | Code improvement suggestions |
| `Squad: Review Changes` | - | Analyze git diff |
| `Squad: Index Workspace` | - | Build codebase knowledge graph |
| `Squad: Clear Cache` | - | Clear embedding cache |
| `Squad: Show Statistics` | - | View usage stats |

### Configuration

Open VS Code Settings (`Ctrl+,`) and search for "Squad":

```json
{
  "squad.hololoomUrl": "http://localhost:8000",
  "squad.enableCache": true,
  "squad.cacheMaxSize": 10000,
  "squad.autoIndexWorkspace": false,
  "squad.maxContextLines": 100,
  "squad.reasoningMode": "verify",
  "squad.showInlineHints": true
}
```

#### Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `hololoomUrl` | `http://localhost:8000` | HoloLoom backend URL |
| `enableCache` | `true` | Enable embedding cache |
| `cacheMaxSize` | `10000` | Max cached embeddings |
| `autoIndexWorkspace` | `false` | Auto-index on startup |
| `maxContextLines` | `100` | Max context lines (0 = unlimited) |
| `reasoningMode` | `verify` | Default mode: direct/verify/research/plan_execute |
| `showInlineHints` | `true` | Show inline suggestions |

## 🏗️ Architecture

### Components

1. **CodeContextExtractor** (`src/lib/CodeContextExtractor.ts`)
   - Tree-sitter AST parsing for Python, TypeScript, JavaScript
   - Extracts functions, classes, imports, dependencies
   - Minimal context extraction (sends only relevant code)

2. **CacheManager** (`src/lib/CacheManager.ts`)
   - SQLite persistent cache
   - LRU eviction policy
   - File watcher for auto-invalidation
   - Incremental updates (only re-embed changed code)

3. **HoloLoomBridge** (`src/lib/HoloLoomBridge.ts`)
   - Type-safe API client
   - Connection status monitoring
   - Error handling with retries
   - Request/response serialization

4. **SquadCommands** (`src/commands/index.ts`)
   - Command implementations
   - Progress reporting
   - Result formatting

### Data Flow

```
User Selection
    ↓
CodeContextExtractor (AST parsing)
    ↓
CacheManager (check cache)
    ↓ (cache miss)
HoloLoomBridge (API request)
    ↓
HoloLoom Backend (agentic reasoning)
    ↓
Result Display (Markdown panel)
```

### Backend Integration

Squad integrates with HoloLoom's agentic API:

- **Query Endpoint**: `/query` - Main agentic reasoning
- **Memory Endpoint**: `/memories/add` - Store code knowledge
- **Workspace Endpoint**: `/ingest/workspace` - Index codebase
- **Detection Endpoint**: `/detect/logic` - Logic error detection

## 📊 Performance

- **Cache Hit**: <1ms response time
- **Cache Miss**: ~150-600ms (depends on reasoning mode)
- **Workspace Indexing**: ~100-500 files/second
- **Memory Usage**: ~50MB base + cache (configurable)

### Cache Statistics

View cache performance with `Squad: Show Statistics`:

```
Cache:
- Entries: 1,247 / 10,000
- Hit Rate: 87.3%
- Size: 23.4 MB

Server:
- Queries: 532
- Success Rate: 98.1%
- Avg Latency: 287ms
```

## 🛠️ Development

### Build from Source

```bash
cd squad
npm install
npm run compile
npm run lint
npm test
```

### Debug Extension

1. Open `squad/` in VS Code
2. Press F5 to launch Extension Development Host
3. Set breakpoints in TypeScript files
4. Test commands in development instance

### Run Tests

```bash
npm test
```

### Package Extension

```bash
npm install -g vsce
vsce package
# Creates squad-1.0.0.vsix
```

## 🐛 Troubleshooting

### "Squad: Server Offline"

**Solution**: Start HoloLoom backend:
```bash
cd HoloLoom/server
python agentic_api.py
```

### "Rate limit exceeded"

**Solution**: Wait a moment and try again. Default limit: 60 requests/minute.

### Cache Issues

**Solution**: Clear cache:
```
Ctrl+Shift+P → "Squad: Clear Cache"
```

### Parser Not Found

**Solution**: Rebuild extension:
```bash
cd squad
npm install
npm run compile
```

## 📚 Examples

### Example 1: Explain Complex Algorithm

**Input** (select code):
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

**Output**:
```markdown
# Code Explanation

This implements the **QuickSort** algorithm using a divide-and-conquer approach.

**How it works:**
1. Base case: Arrays with ≤1 element are already sorted
2. Choose pivot (middle element)
3. Partition into three lists: less than, equal to, greater than pivot
4. Recursively sort left and right partitions
5. Concatenate results

**Key Concepts:**
- Divide and conquer
- Recursion
- List comprehensions (Pythonic)

**Performance:**
- Average: O(n log n)
- Worst case: O(n²) if pivot is always min/max
- Space: O(n) due to list copies

**Improvements:**
- Use in-place partitioning for O(log n) space
- Choose random pivot to avoid worst case
- Switch to insertion sort for small arrays

Confidence: 95%
```

### Example 2: Generate Tests

**Input** (select function):
```typescript
function validateEmail(email: string): boolean {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}
```

**Output** (new file):
```typescript
import { validateEmail } from './email';

describe('validateEmail', () => {
    test('should accept valid email addresses', () => {
        expect(validateEmail('user@example.com')).toBe(true);
        expect(validateEmail('john.doe@company.co.uk')).toBe(true);
        expect(validateEmail('test+tag@domain.io')).toBe(true);
    });

    test('should reject invalid email addresses', () => {
        expect(validateEmail('invalid')).toBe(false);
        expect(validateEmail('@example.com')).toBe(false);
        expect(validateEmail('user@')).toBe(false);
        expect(validateEmail('user @example.com')).toBe(false);
    });

    test('should handle edge cases', () => {
        expect(validateEmail('')).toBe(false);
        expect(validateEmail('a@b.c')).toBe(true);
    });
});
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **HoloLoom**: Advanced agentic reasoning backend
- **Tree-sitter**: Multi-language AST parsing
- **VS Code**: Extensibility platform

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/squad/issues)
- **Docs**: [HoloLoom Documentation](../CLAUDE.md)
- **Discord**: [Community Server](#)

---

**Built with ❤️ using HoloLoom's agentic intelligence**
