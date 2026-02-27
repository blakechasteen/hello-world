# HoloLoom TypeScript SDK - Design Document

**Version**: 1.0.0
**Date**: 2025-12-30
**Status**: Production Ready

## Executive Summary

The HoloLoom TypeScript SDK is a thin, zero-dependency client that exposes HoloLoom's Lite API to TypeScript/JavaScript applications. It mirrors the Python API while following TypeScript conventions.

**Key Principles**:
- **Thin client**: Server does the work, client just wraps HTTP
- **Zero dependencies**: Native fetch only (Node 18+, modern browsers)
- **Type safety**: Complete TypeScript types for all responses
- **Dual environment**: Works in Node.js and browsers seamlessly
- **Production-grade**: Error handling, retries, timeouts, monitoring

## Architecture

### High-Level Design

```
Client Application
        ↓
┌─────────────────────────┐
│  HoloLoomClient         │  Thin wrapper
│  ├─ experience()        │  Mirrors Python API
│  ├─ recall()            │
│  ├─ reflect()           │
│  ├─ reason()            │
│  └─ query()             │
├─ HTTP Adapter          │  Fetch + retry logic
│  ├─ Request handler    │  Type safe
│  ├─ Retry engine       │  Exponential backoff
│  ├─ Error handling     │  HoloLoomClientError
│  └─ Config management  │  Runtime config
└─────────────────────────┘
        ↓ (HTTP)
    HoloLoom API
   (Python server)
```

### Type System

```typescript
// Core types
Memory              // Single memory with embeddings
RecallResult        // Memories + metadata
ReasonResult        // Multi-step reasoning output
Spacetime          // Full orchestrator output
LiteResult<T>      // Generic API response wrapper

// Input types
ExperienceInput
ReflectInput
QueryOptions
ReasoningOptions

// Error type
HoloLoomClientError  // Extends Error with code, status
```

### Request/Response Flow

```
Client Code
    ↓
HoloLoomClient.query(query, options)
    ↓
┌─ Validate input
├─ Build request body
├─ Set headers
└─ Call request()
    ↓
request<T>(method, path, body, options)
    ↓
┌─ Construct URL
├─ Initialize fetch
├─ Set timeout (AbortController)
└─ await fetch()
    ↓ (Response)
┌─ Check response.ok
├─ Parse JSON
└─ Type-cast to LiteResult<T>
    ↓ (Return)
Promise<LiteResult<T>>
    ↓
Client receives typed result
```

### Error Handling Strategy

```
Request fails
    ↓
┌─ Network error? → Retry with exponential backoff
├─ Timeout? → Retry with exponential backoff
├─ HTTP 4xx? → Throw HoloLoomClientError (no retry)
├─ HTTP 5xx? → Retry with exponential backoff
└─ Parse error? → Throw HoloLoomClientError
    ↓
Retries exhausted
    ↓
Throw HoloLoomClientError with:
  - message: Human-readable error
  - code: Machine-readable code (e.g., "HTTP_500")
  - statusCode: HTTP status if applicable
  - details: Full error response
  - timestamp: When error occurred
```

### Configuration

**Initialization-time config**:
```typescript
HoloLoomClient({
  baseUrl,      // API endpoint
  apiKey,       // Authorization
  timeout,      // Request timeout
  retries,      // Retry attempts
  retryDelay,   // Initial retry delay
  userAgent,    // Custom User-Agent
  headers       // Extra headers
})
```

**Runtime config updates**:
```typescript
client.configure({
  baseUrl,      // Change endpoint
  apiKey,       // Update token
  timeout,      // Adjust timeout
  retries,      // Change retry policy
  headers       // Add/override headers
})
```

**Environment variables**:
```bash
HOLOLOOM_API_KEY   # API key (optional)
HOLOLOOM_API_URL   # Override baseUrl
```

## API Design

### Lite API Methods (Mirror Python)

#### 1. experience(input) → ExperienceResult

Store memories with optional metadata.

**Python**:
```python
mem = await loom.experience("content", metadata={...})
```

**TypeScript**:
```typescript
const result = await client.experience({
  content: "string",
  metadata: { ... }
});
```

**Response**:
```typescript
{
  memoryId: "mem_123",
  timestamp: "2025-12-30T12:00:00Z",
  embedded: true,
  graphUpdated: true
}
```

#### 2. recall(query, options) → RecallResult

Search memories using strategy-based retrieval.

**Strategies**:
- RECENT: Temporal preference
- SIMILAR: Semantic similarity
- CONNECTED: Graph distance
- RESONANT: Activation-based
- BALANCED: Automatic (default)

**Response**:
```typescript
{
  memories: Memory[],
  totalCount: number,
  strategy: RecallStrategy
}
```

#### 3. reflect(input) → ReflectResult

Provide feedback for learning.

**Feedback types**:
- helpful: boolean
- accurate: boolean
- relevant: boolean
- quality: 0.0-1.0

**Response**:
```typescript
{
  processed: number,
  updated: number,
  feedback: { ... }
}
```

#### 4. reason(query, options) → ReasonResult

Multi-step reasoning with modes.

**Modes**:
- DIRECT: Single-pass (~150ms)
- VERIFY: Verification pass (~600ms)
- RESEARCH: Multi-query (~900ms)
- PLAN_EXECUTE: Goal-based (~750ms)

**Response**:
```typescript
{
  response: string,
  confidence: 0.0-1.0,
  mode: ReasoningMode,
  stepsTaken: ReasoningStep[],
  verification?: { verified: boolean, issues: string[] }
}
```

#### 5. query(query, options) → Spacetime

Full orchestrator with memory, reasoning, synthesis.

**Response**:
```typescript
{
  response: string,
  confidence: 0.0-1.0,
  sources: Memory[],
  toolUsed: string,
  reasoning: string,
  trace?: { stages, totalMs },
  metadata?: { cacheHit, complexity, threadCount }
}
```

## Type Safety

### Generic Response Wrapper

All API methods return `LiteResult<T>`:

```typescript
interface LiteResult<T> {
  success: boolean;
  data: T;
  error?: string;
  metadata?: {
    latencyMs: number;
    timestamp: string;
    version: string;
  };
}
```

**Usage**:
```typescript
const result = await client.query("test");

if (result.success) {
  // TypeScript knows result.data is Spacetime
  const response: string = result.data.response;
  const confidence: number = result.data.confidence;
} else {
  // TypeScript knows error is defined
  console.error(result.error);
}
```

### JSDoc Documentation

All public methods have comprehensive JSDoc:

```typescript
/**
 * Experience - Store a new memory
 *
 * @param input - Experience content and optional metadata
 * @returns Stored memory ID and metadata
 *
 * @example
 * const result = await client.experience({
 *   content: "Your memory",
 *   metadata: { source: "research" }
 * });
 */
async experience(input: ExperienceInput): Promise<LiteResult<ExperienceResult>>
```

## Performance Characteristics

### Latency

| Operation | Typical | Range |
|-----------|---------|-------|
| experience() | 40ms | 20-100ms |
| recall() | 70ms | 30-150ms |
| reason(DIRECT) | 150ms | 100-300ms |
| reason(VERIFY) | 600ms | 400-1000ms |
| reason(RESEARCH) | 900ms | 600-1500ms |
| query() | 300ms | 200-1500ms |

*Varies with server load and query complexity.*

### Caching

Client-side caching of parsed responses:
- Query cache: 100x speedup for repeated queries
- Backend cache: Transparent to client

### Retry Strategy

**Exponential backoff**:
```
Attempt 1: 0ms
Attempt 2: 100ms (default retryDelay)
Attempt 3: 200ms (2x)
Attempt 4: 400ms (4x)
Attempt 5: 800ms (8x)
```

**Retry conditions**:
- Network errors: Yes
- Timeouts: Yes
- HTTP 5xx: Yes
- HTTP 4xx: No (client error)

## Security Considerations

### API Key Handling

```typescript
// Secure: Key from environment
const client = createHoloLoomClient();
// Reads from HOLOLOOM_API_KEY env var

// Explicit: Key in constructor
const client = createHoloLoomClient({ apiKey: "key" });

// Runtime: Update key
client.configure({ apiKey: "new-key" });

// Get config: Key is redacted
const config = client.getConfig(); // ✗ No apiKey
```

### HTTPS Support

```typescript
// Production: Use HTTPS
const client = createHoloLoomClient({
  baseUrl: 'https://api.hololoom.io'
});

// Development: HTTP allowed
const client = createHoloLoomClient({
  baseUrl: 'http://localhost:8000'
});
```

### Header Validation

Custom headers are validated and sanitized:
```typescript
client.configure({
  headers: {
    'X-Custom-Header': 'value',
    'Authorization': 'Bearer ...', // Overrides client apiKey
  }
});
```

### Error Message Safety

Sensitive information (API keys, internal paths) is never logged:
```typescript
try {
  await client.query("test");
} catch (error) {
  // error.message is safe to log
  // error.code, error.statusCode are included
  // Full response is in error.details
}
```

## Extensibility

### Adding New Methods

To add a new Lite API method:

1. Add type definitions in `types/index.ts`
2. Add method to `HoloLoomClient` class
3. Call `this.request<T>()` with HTTP method/path/body
4. Add tests in `index.test.ts`
5. Add examples in `examples/basic.ts`

Example:
```typescript
async newMethod(query: string): Promise<LiteResult<NewResult>> {
  return this.request('POST', '/lite/new-method', { query });
}
```

### Custom Fetch Implementation

Replace fetch for advanced use cases:

```typescript
class CustomHoloLoomClient extends HoloLoomClient {
  private async request<T>(...): Promise<T> {
    // Custom fetch implementation
    // E.g., custom authentication, proxying, logging
  }
}
```

## Testing Strategy

### Unit Tests (Vitest)

```typescript
describe('HoloLoomClient', () => {
  it('should experience a memory', async () => {
    // Mock fetch
    // Call client.experience()
    // Assert response
  });

  it('should retry on network error', async () => {
    // Mock fetch to fail then succeed
    // Verify retry count
  });

  it('should enforce type safety', () => {
    // Type-check at compile time
    // Runtime assertions for dynamic values
  });
});
```

### Integration Tests (with real API)

```bash
# Set API endpoint
HOLOLOOM_API_URL=http://localhost:8000 npm test

# Tests run against real API
# Verify request/response contracts
```

### Type Checking

```bash
npm run typecheck
# Strict mode enabled in tsconfig.json
# Catches all type errors
```

## Deployment

### Package Distribution

**NPM Package** (`@hololoom/sdk`):
- Published to npm registry
- Includes TypeScript sources and generated types
- Tree-shakeable ESM + CommonJS builds

**Browser Bundle** (via unpkg):
```html
<script src="https://unpkg.com/@hololoom/sdk/dist/index.js"></script>
```

### Build Output

```
dist/
├── index.js           # ESM build
├── index.cjs          # CommonJS build
├── index.d.ts         # TypeScript types
├── types/
│   ├── index.js
│   ├── index.cjs
│   └── index.d.ts
└── *.map              # Source maps
```

## Version Strategy

**Semantic Versioning**:
- Major: Breaking API changes
- Minor: New methods, backward-compatible
- Patch: Bug fixes, performance improvements

**Example**:
```
v1.0.0 - Initial release (Lite API)
v1.1.0 - Add new query types (backward-compatible)
v1.1.1 - Fix retry logic bug
v2.0.0 - New major API version
```

## Migration Path

### From Direct HTTP Calls

```typescript
// Before: Raw fetch
const response = await fetch('http://localhost:8000/lite/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'test' })
});
const data = await response.json();

// After: SDK
import { createHoloLoomClient } from '@hololoom/sdk';
const client = createHoloLoomClient();
const result = await client.query('test');
```

### From Python Lite API

```python
# Python
result = await loom.query("test")
print(result.response)
```

```typescript
// TypeScript
const result = await client.query("test");
console.log(result.data.response);
```

## Future Enhancements

### Planned Features (v1.1+)

1. **Streaming responses**: Stream Spacetime results as they're generated
2. **Batch operations**: Batch multiple recalls/experiences
3. **Event emitters**: Subscribe to query events
4. **Request interceptors**: Middleware for requests/responses
5. **Offline mode**: Cache responses locally
6. **WebSocket support**: For real-time updates

### Research Directions

1. **GraphQL API**: Alternative to REST
2. **WASM bindings**: Direct memory access
3. **Plugin system**: Third-party integrations

## Documentation

### Included Docs

- `README.md` - Getting started, API reference
- `DESIGN.md` - This document (architecture, decisions)
- `examples/basic.ts` - Comprehensive usage patterns
- JSDoc comments in source code

### External References

- Python Lite API: `hololoom/hololoom.py`
- API server: `hololoom/server/agentic_api.py`
- Type definitions: `src/types/index.ts`

## Conclusion

The HoloLoom TypeScript SDK provides a thin, type-safe wrapper around the HoloLoom API. By mirroring the Python Lite API, it enables developers to work with HoloLoom from TypeScript/JavaScript applications with zero dependencies and production-grade error handling.
