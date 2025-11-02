# Agentic System: LLM + Persistent Memory Integration

**Status**: ⚠️ Missing Links - Core orchestration ready, needs LLM + memory wiring
**Critical Gap**: The agentic system orchestrates reasoning but doesn't actually call LLMs or use persistent storage

---

## What's Missing

### 1. LLM Integration ❌

**Current State**: `WeavingOrchestrator._handle_answer()` returns **stub responses**:
```python
# HoloLoom/weaving_orchestrator.py:150
return {
    "result": f"Generated answer for: {query.text}",  # ❌ No actual LLM call!
    "confidence": 0.85,
}
```

**What Exists**:
✅ `HoloLoom/awareness/llm_integration.py` - LLM protocol + implementations
- `OllamaLLM` (local, free)
- `AnthropicLLM` (Claude)
- `OpenAILLM` (GPT)

**What's Missing**: Wiring the orchestrator to actually call LLMs.

---

### 2. Persistent Memory ❌

**Current State**: Agentic system uses **in-memory shards**:
```python
# HoloLoom/server/agentic_api.py:104
state.shards = _load_memory_shards()  # Returns empty list!
```

**What Exists**:
✅ `HoloLoom/memory/backend_factory.py` - Memory backends
- `INMEMORY`: NetworkX (development)
- `HYBRID`: Neo4j + Qdrant (production, persistent)
- `HYPERSPACE`: Advanced gated multipass (research)

**What's Missing**: Loading real data from persistent backends.

---

## How to Fix (Two Steps)

### Step 1: Connect LLM (30 minutes)

**Update `WeavingOrchestrator._handle_answer()`**:

```python
# HoloLoom/weaving_orchestrator.py

from HoloLoom.awareness.llm_integration import OllamaLLM, LLMProvider

class ToolExecutor:
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.tools = ["answer", "search", "notion_write", "calc"]
        self.logger = logging.getLogger(__name__)

        # Initialize LLM (lazy loading)
        self.llm = llm
        if self.llm is None:
            try:
                self.llm = OllamaLLM(model="llama3.2:3b")
                self.logger.info("Initialized Ollama LLM")
            except Exception as e:
                self.logger.warning(f"LLM unavailable, using fallback: {e}")
                self.llm = None

    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """Generate an answer based on context using actual LLM."""

        # Build context from retrieved shards
        if context and hasattr(context, 'shard_texts'):
            context_text = "\n\n".join(context.shard_texts[:5])
        else:
            context_text = "(No context available)"

        # Build prompt
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer based on the provided context."
        )

        user_prompt = f"""Context:
{context_text}

Question: {query.text}

Answer:"""

        # Call LLM
        if self.llm and self.llm.is_available():
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
                )

                return {
                    "tool": "answer",
                    "result": response.content,  # ✅ Actual LLM response!
                    "confidence": 0.85,
                    "sources": len(context.shards) if context else 0,
                    "llm_provider": response.provider.value,
                    "llm_model": response.model,
                    "usage": response.usage
                }
            except Exception as e:
                self.logger.error(f"LLM generation failed: {e}")
                # Fall through to fallback

        # Fallback (LLM unavailable)
        return {
            "tool": "answer",
            "result": f"[Fallback] Generated answer for: {query.text}\nContext: {context_text[:200]}...",
            "confidence": 0.5,
            "sources": 0
        }
```

**Alternative: Use Anthropic Claude**:
```python
from HoloLoom.awareness.llm_integration import AnthropicLLM
import os

llm = AnthropicLLM(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20241022"
)
```

---

### Step 2: Connect Persistent Memory (15 minutes)

**Update `HoloLoom/server/agentic_api.py`**:

```python
# HoloLoom/server/agentic_api.py

from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.config import MemoryBackend

class ServerState:
    """Global server state."""
    orchestrator: Optional[Any] = None
    audit_trail: Optional[AuditTrail] = None
    config: Optional[Config] = None
    shards: List[MemoryShard] = []
    memory_backend: Optional[Any] = None  # ✅ Add persistent backend

state = ServerState()

@app.on_event("startup")
async def startup():
    """Initialize server with persistent memory."""
    logger.info("Starting HoloLoom Agentic API server...")

    # Load config
    state.config = Config.fast()
    state.config.enable_agentic_reasoning = True
    state.config.memory_backend = MemoryBackend.HYBRID  # ✅ Use persistent storage

    # Initialize audit trail
    state.audit_trail = AuditTrail(persist_path="./alignment_logs")

    # ✅ Create persistent memory backend
    try:
        state.memory_backend = await create_memory_backend(state.config)
        logger.info(f"Memory backend: {state.config.memory_backend.value}")

        # Load existing memories from persistent storage
        state.shards = await _load_from_persistent_backend()
        logger.info(f"Loaded {len(state.shards)} memories from persistent storage")

    except Exception as e:
        logger.warning(f"Persistent backend unavailable: {e}")
        logger.info("Falling back to in-memory storage")
        state.shards = []

    logger.info("HoloLoom server ready!")

async def _load_from_persistent_backend() -> List[MemoryShard]:
    """
    Load memories from persistent backend (Neo4j/Qdrant).

    Returns:
        List of MemoryShard objects loaded from storage
    """
    if not state.memory_backend:
        return []

    try:
        # Query all memories (or filter by criteria)
        # This depends on your backend implementation
        from HoloLoom.memory.protocol import MemoryQuery

        query = MemoryQuery(
            text="",  # Empty query = retrieve all
            limit=1000  # Adjust based on your needs
        )

        result = await state.memory_backend.retrieve(query)

        # Convert Memory objects to MemoryShard objects
        shards = []
        for memory in result.memories:
            shard = MemoryShard(
                id=memory.id,
                text=memory.text,
                episode=memory.context.get("episode", "default"),
                entities=memory.context.get("entities", []),
                motifs=memory.context.get("motifs", []),
                metadata=memory.metadata
            )
            shards.append(shard)

        return shards

    except Exception as e:
        logger.error(f"Failed to load from persistent backend: {e}")
        return []

# ✅ Add endpoint to store new memories
@app.post("/memories/add")
async def add_memory(memory: Dict):
    """
    Add new memory to persistent storage.

    Example:
        POST /memories/add
        {
          "text": "Thompson Sampling is a Bayesian approach...",
          "entities": ["Thompson Sampling", "Bayesian"],
          "motifs": ["definition"]
        }
    """
    if not state.memory_backend:
        raise HTTPException(status_code=503, detail="Persistent memory unavailable")

    try:
        from HoloLoom.memory.protocol import Memory
        from datetime import datetime

        # Create memory object
        mem = Memory(
            id=f"mem_{int(datetime.now().timestamp())}",
            text=memory["text"],
            timestamp=datetime.now(),
            context={
                "entities": memory.get("entities", []),
                "motifs": memory.get("motifs", []),
                "episode": memory.get("episode", "default")
            },
            metadata=memory.get("metadata", {})
        )

        # Store in persistent backend
        await state.memory_backend.store([mem])

        # Also add to active shards
        shard = MemoryShard(
            id=mem.id,
            text=mem.text,
            episode=mem.context.get("episode", "default"),
            entities=mem.context.get("entities", []),
            motifs=mem.context.get("motifs", []),
            metadata=mem.metadata
        )
        state.shards.append(shard)

        return {"status": "success", "id": mem.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Complete Integration Example

Here's a full example showing LLM + persistent memory together:

```python
# examples/agentic_llm_memory_integration.py

import asyncio
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery
from HoloLoom.awareness.llm_integration import OllamaLLM
from datetime import datetime

async def main():
    print("="*80)
    print("Agentic System: LLM + Persistent Memory Integration")
    print("="*80)

    # 1. Setup persistent memory
    print("\n1. Setting up persistent memory...")
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant

    memory_backend = await create_memory_backend(config)
    print(f"   ✓ Memory backend: {config.memory_backend.value}")

    # 2. Add some knowledge to persistent storage
    print("\n2. Adding knowledge to persistent storage...")
    memories = [
        Memory(
            id="ts_1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits.",
            timestamp=datetime.now(),
            context={"entities": ["Thompson Sampling"], "motifs": ["definition"]},
            metadata={}
        ),
        Memory(
            id="ts_2",
            text="Thompson Sampling balances exploration and exploitation naturally.",
            timestamp=datetime.now(),
            context={"entities": ["Thompson Sampling"], "motifs": ["property"]},
            metadata={}
        )
    ]

    await memory_backend.store(memories)
    print(f"   ✓ Stored {len(memories)} memories")

    # 3. Load memories for agentic system
    print("\n3. Loading memories for agentic system...")
    query = MemoryQuery(text="Thompson Sampling", limit=10)
    result = await memory_backend.retrieve(query)

    shards = [
        MemoryShard(
            id=mem.id,
            text=mem.text,
            episode=mem.context.get("episode", "default"),
            entities=mem.context.get("entities", []),
            motifs=mem.context.get("motifs", [])
        )
        for mem in result.memories
    ]
    print(f"   ✓ Loaded {len(shards)} memories")

    # 4. Setup LLM
    print("\n4. Setting up LLM...")
    llm = OllamaLLM(model="llama3.2:3b")

    if llm.is_available():
        print(f"   ✓ LLM available: {llm.model}")
    else:
        print("   ⚠ LLM unavailable (install Ollama from https://ollama.ai)")
        return

    # 5. Create agentic orchestrator with LLM + memory
    print("\n5. Creating agentic orchestrator...")
    async with await create_agentic_orchestrator(config, shards) as agent:
        print("   ✓ Agentic orchestrator ready")

        # Note: You'd need to pass the LLM to the orchestrator
        # This requires modifying the orchestrator to accept an LLM parameter

        # 6. Test VERIFY mode with real LLM
        print("\n6. Testing VERIFY mode with LLM + persistent memory...")
        query = Query(text="What is Thompson Sampling?")

        result = await agent.reason(
            query,
            mode=ReasoningMode.VERIFY,
            max_steps=3
        )

        print(f"\n   Result:")
        print(f"   - Confidence: {result.spacetime.confidence:.3f}")
        print(f"   - Verified: {result.verification.verified if result.verification else 'N/A'}")
        print(f"   - Steps: {result.total_queries}")
        print(f"   - Duration: {result.total_duration_ms:.1f}ms")

        # The response would come from the LLM, using context from persistent memory
        # Right now this returns a stub because _handle_answer doesn't call LLM

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Quick Setup Guide

### Option 1: Local Development (Ollama)

```bash
# 1. Install Ollama
# Download from: https://ollama.ai

# 2. Pull a model
ollama pull llama3.2:3b

# 3. Test it works
ollama run llama3.2:3b "What is Thompson Sampling?"

# 4. Use in HoloLoom
# It will auto-connect to localhost:11434
```

### Option 2: Production (Anthropic Claude)

```bash
# 1. Get API key from https://console.anthropic.com

# 2. Set environment variable
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# 3. Use in code
from HoloLoom.awareness.llm_integration import AnthropicLLM
llm = AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

### Option 3: Persistent Memory (Docker)

```bash
# 1. Start Neo4j + Qdrant
docker-compose up -d

# 2. Verify they're running
curl http://localhost:6333/collections  # Qdrant
curl http://localhost:7474  # Neo4j web UI

# 3. Configure in code
config.memory_backend = MemoryBackend.HYBRID
```

---

## What You Get After Integration

| Feature | Before | After |
|---------|--------|-------|
| **LLM Responses** | ❌ Stub "Generated answer for..." | ✅ Actual Claude/Llama responses |
| **Memory** | ❌ Empty list | ✅ Persistent Neo4j/Qdrant storage |
| **Verification** | ✅ Orchestration only | ✅ Real contradiction checking |
| **Learning** | ✅ Pattern tracking | ✅ Persistent learning across sessions |

---

## Summary

**What I Built**:
✅ Agentic orchestration (4 reasoning modes, verification loops)
✅ Embedding integrity monitoring
✅ HTTP server
✅ VS Code integration

**What's Missing** (Your Question):
❌ **LLM calls** - orchestrator doesn't call actual LLMs
❌ **Persistent memory** - uses empty in-memory list

**Fix Time**:
- LLM integration: 30 minutes (update `_handle_answer`)
- Persistent memory: 15 minutes (update `startup()`)
- **Total**: 45 minutes

**Files to Modify**:
1. `HoloLoom/weaving_orchestrator.py` - Add LLM calls to `_handle_answer()`
2. `HoloLoom/server/agentic_api.py` - Load from persistent backend

**Then it's fully functional!** 🚀
