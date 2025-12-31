# HoloLoom TypeScript SDK

A thin, zero-dependency TypeScript client for the HoloLoom API. Mirrors the Python Lite API for seamless integration across ecosystems.

## Features

- **Lite API Methods**: `experience()`, `recall()`, `reflect()`, `reason()`, `query()`
- **Type-Safe**: Complete TypeScript types with JSDoc documentation
- **Zero Dependencies**: Only uses native `fetch` (Node 18+, modern browsers)
- **Dual Environment**: Works in Node.js and browser
- **Retry Logic**: Automatic exponential backoff for resilience
- **Production-Ready**: Error handling, timeouts, configurable options

## Installation

```bash
npm install @hololoom/sdk
```

## Quick Start

```typescript
import { createHoloLoomClient } from '@hololoom/sdk';

const client = createHoloLoomClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key', // optional
});

// Store a memory
const stored = await client.experience({
  content: 'Thompson Sampling balances exploration and exploitation',
  metadata: { source: 'research' }
});

// Recall relevant memories
const memories = await client.recall(
  'What is Thompson Sampling?',
  { limit: 10, strategy: 'SIMILAR' }
);

// Multi-step reasoning with verification
const result = await client.reason(
  'Compare Thompson Sampling vs UCB',
  { mode: 'VERIFY', maxSteps: 5 }
);

// Full orchestrator query
const spacetime = await client.query(
  'Explain Thompson Sampling',
  { mode: 'RESEARCH', complexityLevel: 'COMPLEX' }
);

console.log(spacetime.data.response);
console.log(`Confidence: ${spacetime.data.confidence}`);
console.log(`Sources: ${spacetime.data.sources.length}`);
```

## API Reference

### HoloLoomClient

#### `experience(input: ExperienceInput): Promise<LiteResult<ExperienceResult>>`

Store a new memory in the knowledge graph.

```typescript
const result = await client.experience({
  content: 'Your memory content',
  metadata: { source: 'research', tags: ['ai', 'learning'] }
});

console.log(result.data.memoryId);
console.log(result.data.embedded); // true if embedding succeeded
```

#### `recall(query: string, options?: RecallOptions): Promise<LiteResult<RecallResult>>`

Retrieve relevant memories using intelligent search strategies.

```typescript
const memories = await client.recall(
  'Thompson Sampling',
  {
    strategy: 'SIMILAR',      // 'RECENT' | 'SIMILAR' | 'CONNECTED' | 'RESONANT' | 'BALANCED'
    limit: 10,
    minRelevance: 0.3,
    contextMemoryIds: ['mem_123'] // optional context
  }
);

// Use results
memories.data.memories.forEach(mem => {
  console.log(`${mem.content} (relevance: ${mem.relevance})`);
});
```

**Strategies**:
- **RECENT**: Most recently accessed memories
- **SIMILAR**: Semantic similarity search
- **CONNECTED**: Graph-connected memories
- **RESONANT**: Activation-based relevance
- **BALANCED**: Automatic strategy selection (default)

#### `reflect(input: ReflectInput): Promise<LiteResult<ReflectResult>>`

Provide feedback to improve memory learning.

```typescript
const result = await client.reflect({
  memoryIds: ['mem_123', 'mem_456'],
  feedback: {
    helpful: true,
    accurate: true,
    relevant: true,
    quality: 0.95 // 0.0-1.0
  },
  tags: ['important', 'verified']
});

console.log(`Updated ${result.data.updated} memories`);
```

#### `reason(query: string, options?: ReasoningOptions): Promise<LiteResult<ReasonResult>>`

Multi-step reasoning with optional verification and research modes.

```typescript
const result = await client.reason(
  'Analyze Thompson Sampling',
  {
    mode: 'RESEARCH',          // 'DIRECT' | 'VERIFY' | 'RESEARCH' | 'PLAN_EXECUTE'
    maxSteps: 5,
    verificationThreshold: 0.7
  }
);

console.log(result.data.response);
console.log(result.data.confidence); // 0.0-1.0

// View reasoning steps
result.data.stepsTaken.forEach((step, i) => {
  console.log(`${i + 1}. ${step.type}: ${step.query}`);
  console.log(`   Confidence: ${step.confidence}`);
});

// Check verification results (if mode='VERIFY')
if (result.data.verification?.verified) {
  console.log('✓ Verified');
}
```

**Modes**:
- **DIRECT**: Single-pass answer (~150ms)
- **VERIFY**: Answer + verification (~600ms)
- **RESEARCH**: Multi-query exploration (~900ms)
- **PLAN_EXECUTE**: Goal decomposition (~750ms)

#### `query(query: string, options?: QueryOptions): Promise<LiteResult<Spacetime>>`

Full orchestrator with weaving cycle, memory retrieval, and synthesis.

```typescript
const spacetime = await client.query(
  'Explain Thompson Sampling',
  {
    mode: 'RESEARCH',
    maxSteps: 5,
    complexityLevel: 'COMPLEX',
    context: 'Previous context about bandits',
    format: 'markdown',
    includeProvenance: true
  }
);

// Response
console.log(spacetime.data.response);
console.log(`Confidence: ${spacetime.data.confidence}`);

// Source attribution
spacetime.data.sources.forEach(src => {
  console.log(`- ${src.content}`);
  console.log(`  Relevance: ${src.relevance}`);
});

// Execution trace
spacetime.data.trace?.stages.forEach(stage => {
  console.log(`${stage.name}: ${stage.durationMs}ms [${stage.status}]`);
});

// Metadata
console.log(`Cache hit: ${spacetime.data.metadata?.cacheHit}`);
```

### Configuration

```typescript
// Initialize with config
const client = createHoloLoomClient({
  baseUrl: 'http://localhost:8000',      // Default: auto-detect
  apiKey: 'your-api-key',                // Optional
  timeout: 30000,                         // ms
  retries: 3,                             // retry attempts
  retryDelay: 100,                        // initial delay ms
  userAgent: 'my-app/1.0',                // custom agent
  headers: { 'X-Custom': 'value' }        // extra headers
});

// Runtime configuration
client.configure({
  timeout: 60000,
  baseUrl: 'https://api.hololoom.io'
});

// Get current config (redacted)
const config = client.getConfig();
```

### Environment Variables

```bash
# API configuration
HOLOLOOM_API_KEY=your-api-key
HOLOLOOM_API_URL=http://localhost:8000
```

### Error Handling

```typescript
import { HoloLoomClientError } from '@hololoom/sdk';

try {
  const result = await client.query('Your query');

  if (!result.success) {
    console.error(`API error: ${result.error}`);
    console.error(`Details:`, result.metadata);
  }
} catch (error) {
  if (error instanceof HoloLoomClientError) {
    console.error(`Client error: ${error.message}`);
    console.error(`Code: ${error.code}`);
    console.error(`Status: ${error.statusCode}`);
  }
}
```

## Type Definitions

All response types are fully typed:

```typescript
import type {
  Memory,
  RecallResult,
  ReasonResult,
  Spacetime,
  LiteResult
} from '@hololoom/sdk';

// Type-safe access
const memory: Memory = {
  id: 'mem_123',
  content: 'Your memory',
  timestamp: new Date().toISOString(),
  embedding: null,
  metadata: {}
};

const result: LiteResult<Spacetime> = await client.query('test');
const response: string = result.data.response;
const confidence: number = result.data.confidence; // 0.0-1.0
```

## Examples

See `examples/basic.ts` for comprehensive usage patterns:

```bash
# View examples
cat examples/basic.ts

# Run a specific example
npx ts-node examples/basic.ts
```

## Browser Usage

```html
<!-- In browser -->
<script type="module">
  import { createHoloLoomClient } from 'https://unpkg.com/@hololoom/sdk@1.0.0/dist/index.js';

  const client = createHoloLoomClient();

  async function query() {
    const result = await client.query('Explain AI');
    console.log(result.data.response);
  }

  query();
</script>
```

## Node.js Usage

```typescript
import { createHoloLoomClient } from '@hololoom/sdk';

const client = createHoloLoomClient({
  baseUrl: 'http://localhost:8000'
});

async function main() {
  const result = await client.query('Explain Thompson Sampling');
  console.log(result.data.response);
}

main();
```

## Performance Characteristics

| Operation | Typical Latency |
|-----------|-----------------|
| experience() | <50ms |
| recall() | <100ms |
| reason(DIRECT) | ~150ms |
| reason(VERIFY) | ~600ms |
| reason(RESEARCH) | ~900ms |
| query() | 150-1000ms+ |

*Actual latency depends on query complexity and backend performance.*

## Testing

```bash
# Run tests
npm test

# With coverage
npm run test:coverage

# Watch mode
npm test -- --watch
```

## Build

```bash
# Development
npm run build:watch

# Production
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint
```

## Migration from Python

### Python API → TypeScript API

```python
# Python
from hololoom import HoloLoom

async with HoloLoom() as loom:
    mem = await loom.experience("content")
    memories = await loom.recall("query")
    result = await loom.reflect(memories, feedback)
```

```typescript
// TypeScript
import { createHoloLoomClient } from '@hololoom/sdk';

const client = createHoloLoomClient();

const mem = await client.experience({ content: "content" });
const memories = await client.recall("query");
const result = await client.reflect({ memoryIds: [...], feedback: {...} });
```

### Response Handling

```python
# Python - direct access
response = await loom.query("test")
print(response.response)
print(response.confidence)
```

```typescript
// TypeScript - wrapped in LiteResult
const result = await client.query("test");
if (result.success) {
  console.log(result.data.response);
  console.log(result.data.confidence);
} else {
  console.error(result.error);
}
```

## Architecture

```
┌─────────────────────────┐
│  HoloLoomClient         │  Main class
├─────────────────────────┤
│ • experience()          │  Lite API methods
│ • recall()              │
│ • reflect()             │
│ • reason()              │
│ • query()               │
├─────────────────────────┤
│ • Fetch wrapper         │  HTTP transport
│ • Retry logic           │  Resilience
│ • Error handling        │  Safety
└─────────────────────────┘
       ↓
    HoloLoom API
```

## Development

### Project Structure

```
sdk/typescript/
├── src/
│   ├── index.ts           # Main exports
│   ├── client.ts          # HoloLoomClient class
│   └── types/
│       └── index.ts       # TypeScript types
├── examples/
│   └── basic.ts           # Usage examples
├── package.json
├── tsconfig.json
└── README.md
```

### Key Design Decisions

1. **Zero Dependencies**: Uses native `fetch` (available in Node 18+)
2. **Thin Client**: Minimal wrapping, delegates logic to server
3. **Type Safety**: Full TypeScript coverage with JSDoc docs
4. **Error Handling**: Automatic retries with exponential backoff
5. **Environment Support**: Works in Node.js and browsers
6. **Extensibility**: Easy to add new methods without breaking changes

## License

MIT

## Support

- **Issues**: https://github.com/blake/mythRL/issues
- **API Docs**: See `docs/` directory
- **Examples**: See `examples/` directory
