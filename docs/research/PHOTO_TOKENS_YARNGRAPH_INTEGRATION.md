# PhotoTokens → YarnGraph Integration

**Date**: November 7, 2025
**Status**: Phase 1 Complete, Integration Pending
**Goal**: Integrate visual tokens into HoloLoom's knowledge graph

---

## Overview

PhotoTokens are now implemented as first-class memory citizens. The next step is integrating them into YarnGraph (HoloLoom's knowledge graph) to enable:

1. **Multimodal nodes**: Photos alongside text memories
2. **Visual-text edges**: Link photos to related text memories
3. **Photo-photo edges**: Semantic relationships between images
4. **Unified retrieval**: Query across both text and visual memories

---

## Current Architecture

### PhotoToken (Implemented ✅)

```python
@dataclass
class PhotoToken:
    token_id: str
    timestamp: float
    image_data: bytes
    clip_embedding: np.ndarray  # 512D
    caption: Optional[str]
    tags: List[str]
    entities: List[str]
    metadata: Dict
```

**Key Features**:
- CLIP embeddings for text-image matching
- Structural features (color, brightness, layout)
- Deduplication via SHA256 hashing
- JPEG compression for storage efficiency

### PhotoTokenMemory (Implemented ✅)

```python
class PhotoTokenMemory:
    async def store(image, caption, tags) -> PhotoToken
    async def retrieve_by_text(query, k) -> List[Tuple[PhotoToken, float]]
    async def retrieve_by_image(query_image, k) -> List[Tuple[PhotoToken, float]]
    async def retrieve_by_tags(tags, k) -> List[PhotoToken]
```

**Performance**:
- Store: ~200ms (CLIP encoding + disk write)
- Retrieve by text: ~50ms (CLIP similarity search)
- Retrieve by image: ~100ms (CLIP embedding + search)

### YarnGraph (Existing)

Located in `HoloLoom/memory/graph.py`:

```python
class KG:  # Alias: YarnGraph
    def add_node(node_id, **attrs)
    def add_edge(source, target, edge_type, weight)
    def get_subgraph(node_ids) -> nx.MultiDiGraph
```

**Current Node Types**:
- Text entities (people, places, concepts)
- Relationships (IS_A, USES, MENTIONS, etc.)

---

## Integration Design

### 1. Extend YarnGraph Node Types

**File**: `HoloLoom/memory/graph.py` (lines ~50-100)

Add photo node type:

```python
# Current node types
NODE_TYPES = {
    'entity': {...},
    'concept': {...},
    # NEW: Add photo type
    'photo': {
        'required_attrs': ['token_id', 'timestamp'],
        'optional_attrs': ['caption', 'tags', 'clip_embedding']
    }
}
```

### 2. PhotoToken → YarnGraph Conversion

**Method**: `PhotoToken.to_yarn_node()` (Already implemented ✅)

```python
def to_yarn_node(self) -> Dict:
    return {
        'id': self.token_id,
        'type': 'photo_token',
        'timestamp': self.timestamp,
        'caption': self.caption,
        'tags': self.tags,
        'embeddings': {
            'clip': self.clip_embedding.tolist(),
            'caption': self.caption_embedding.tolist() if self.caption_embedding else None
        },
        'metadata': self.metadata
    }
```

### 3. Add Photos to Knowledge Graph

**New Method**: `YarnGraph.add_photo_node()`

```python
# Add to HoloLoom/memory/graph.py
class KG:
    def add_photo_node(self, photo_token: PhotoToken) -> str:
        """
        Add photo as node in knowledge graph.

        Args:
            photo_token: PhotoToken to add

        Returns:
            Node ID (token_id)
        """
        node_data = photo_token.to_yarn_node()

        # Add node
        self.graph.add_node(
            photo_token.token_id,
            **node_data
        )

        # Create edges to related entities (from caption/tags)
        if photo_token.caption:
            # Extract entities from caption (using existing motif extraction)
            from HoloLoom.motif.base import extract_entities
            entities = extract_entities(photo_token.caption)

            for entity in entities:
                # Add edge: photo -> entity
                self.add_edge(
                    photo_token.token_id,
                    entity,
                    edge_type='DEPICTS',
                    weight=1.0
                )

        # Tag-based edges
        for tag in photo_token.tags:
            self.add_edge(
                photo_token.token_id,
                tag,
                edge_type='TAGGED_AS',
                weight=1.0
            )

        return photo_token.token_id
```

### 4. Multimodal Retrieval

**New Method**: `YarnGraph.search_multimodal()`

```python
class KG:
    def search_multimodal(
        self,
        query: str,
        return_types: List[str] = ['text', 'photo'],
        k: int = 10
    ) -> List[Dict]:
        """
        Search across text and photo memories.

        Args:
            query: Text query
            return_types: Types to return ['text', 'photo', 'both']
            k: Number of results

        Returns:
            List of {type: 'text'|'photo', node_id: str, score: float, data: Dict}
        """
        results = []

        # Text search (existing)
        if 'text' in return_types:
            text_results = self.search_text(query, k=k//2)
            results.extend([
                {'type': 'text', 'node_id': nid, 'score': score, 'data': data}
                for nid, score, data in text_results
            ])

        # Photo search (new - requires PhotoTokenMemory)
        if 'photo' in return_types:
            # Get photo nodes
            photo_nodes = [
                (nid, attrs)
                for nid, attrs in self.graph.nodes(data=True)
                if attrs.get('type') == 'photo_token'
            ]

            # Score by caption similarity (simple baseline)
            from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
            embedder = MatryoshkaEmbeddings()

            query_emb = embedder.embed([query])[0]

            for node_id, attrs in photo_nodes:
                caption = attrs.get('caption', '')
                if caption:
                    caption_emb = embedder.embed([caption])[0]
                    score = np.dot(query_emb, caption_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(caption_emb) + 1e-8
                    )
                    results.append({
                        'type': 'photo',
                        'node_id': node_id,
                        'score': float(score),
                        'data': attrs
                    })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:k]
```

---

## Edge Types for Photos

### New Edge Types

| Edge Type | Source | Target | Meaning | Example |
|-----------|--------|--------|---------|---------|
| **DEPICTS** | Photo | Entity | Photo shows entity | `photo_abc → "laptop"` |
| **TAGGED_AS** | Photo | Tag | Photo has tag | `photo_abc → "diagram"` |
| **SIMILAR_TO** | Photo | Photo | Visually similar | `photo_abc → photo_def` |
| **ILLUSTRATES** | Photo | Concept | Photo explains concept | `photo_abc → "architecture"` |
| **REFERENCED_IN** | Photo | Text Memory | Photo mentioned in text | `photo_abc → memory_xyz` |

### Usage Example

```python
# Add photo to knowledge graph
kg = KG()
photo_token = await memory.store("diagram.png", caption="System architecture")

# Add as node
kg.add_photo_node(photo_token)

# Link to text memory
text_memory_id = "memory_discussion_123"
kg.add_edge(
    photo_token.token_id,
    text_memory_id,
    edge_type='REFERENCED_IN',
    weight=1.0
)

# Link to concept
kg.add_edge(
    photo_token.token_id,
    "system_architecture",
    edge_type='ILLUSTRATES',
    weight=0.9
)

# Query multimodal
results = kg.search_multimodal(
    "Show me the architecture diagram",
    return_types=['text', 'photo']
)
# Returns: [text_memory, photo_token]
```

---

## Integration Steps (Week 2)

### Step 1: Extend YarnGraph (2 hours)

**File**: `HoloLoom/memory/graph.py`

- [ ] Add `photo_token` to NODE_TYPES
- [ ] Implement `add_photo_node(photo_token)`
- [ ] Add new edge types (DEPICTS, TAGGED_AS, etc.)
- [ ] Test: Add photo node, verify edges created

### Step 2: Multimodal Search (3 hours)

**File**: `HoloLoom/memory/graph.py`

- [ ] Implement `search_multimodal(query, return_types, k)`
- [ ] CLIP-based photo scoring (if available)
- [ ] Caption-based fallback (if CLIP unavailable)
- [ ] Combine text + photo scores
- [ ] Test: Query returns both text and photos

### Step 3: HoloLoom API Integration (2 hours)

**File**: `HoloLoom/hololoom.py`

- [ ] Add `remember_photo(image, caption, tags)` method
- [ ] Add `recall(query, include_photos=True)` parameter
- [ ] Auto-link photos to related memories
- [ ] Test: Full experience → recall cycle with photos

### Step 4: Demo (1 hour)

**File**: `demos/demo_multimodal_memory.py`

- [ ] Store text memory: "We discussed the architecture"
- [ ] Store photo: architecture diagram
- [ ] Link photo to text memory
- [ ] Query: "What was the architecture?"
- [ ] Verify: Returns both text and photo

---

## API Preview

### User-Facing API (HoloLoom.py)

```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Store text memory
    text_mem = await loom.experience("We discussed the system architecture")

    # Store photo memory
    photo_mem = await loom.remember_photo(
        "whiteboard.jpg",
        caption="Architecture diagram from meeting",
        tags=["diagram", "architecture"]
    )

    # Link photo to text
    await loom.link_memories(text_mem, photo_mem)

    # Recall (multimodal)
    memories = await loom.recall(
        "What was the architecture?",
        include_photos=True
    )

    # Returns: [text_memory, photo_memory]
    for mem in memories:
        if mem.type == 'photo':
            print(f"Photo: {mem.caption}")
            display_image(mem.image_data)
        else:
            print(f"Text: {mem.content}")
```

---

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| **Add photo node** | <10ms | Graph insertion + edge creation |
| **Multimodal search** | <100ms | CLIP text encoding + graph search |
| **Photo → Text link** | <5ms | Simple edge addition |
| **Full recall (text + photo)** | <150ms | Combined search across modalities |

---

## Testing Strategy

### Unit Tests

**File**: `HoloLoom/tests/unit/test_photo_yarngraph.py`

```python
async def test_add_photo_node():
    """Test adding photo to YarnGraph."""
    kg = KG()
    photo = PhotoToken(...)

    node_id = kg.add_photo_node(photo)
    assert node_id in kg.graph.nodes
    assert kg.graph.nodes[node_id]['type'] == 'photo_token'

async def test_multimodal_search():
    """Test searching across text and photos."""
    kg = KG()
    # Add text memories
    # Add photo memories

    results = kg.search_multimodal("architecture diagram")
    assert len(results) > 0
    assert any(r['type'] == 'photo' for r in results)
```

### Integration Tests

**File**: `HoloLoom/tests/integration/test_multimodal_memory.py`

```python
async def test_experience_and_recall_with_photos():
    """Test full cycle with photos."""
    async with HoloLoom() as loom:
        # Experience text
        await loom.experience("Discussed architecture")

        # Remember photo
        await loom.remember_photo("diagram.png", caption="Architecture")

        # Recall multimodal
        memories = await loom.recall("architecture", include_photos=True)

        assert any(m.type == 'text' for m in memories)
        assert any(m.type == 'photo' for m in memories)
```

---

## Future Enhancements (Phase 2)

### 1. Auto-Captioning (BLIP)

Generate captions automatically:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def auto_caption(image: np.ndarray) -> str:
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
```

### 2. Visual Similarity Edges

Automatically link visually similar photos:

```python
async def link_similar_photos(memory: PhotoTokenMemory, kg: KG, threshold=0.8):
    """Create SIMILAR_TO edges between photos."""
    tokens = list(memory.tokens.values())

    for i, token1 in enumerate(tokens):
        for token2 in tokens[i+1:]:
            # CLIP similarity
            sim = np.dot(token1.clip_embedding, token2.clip_embedding)

            if sim > threshold:
                kg.add_edge(
                    token1.token_id,
                    token2.token_id,
                    edge_type='SIMILAR_TO',
                    weight=float(sim)
                )
```

### 3. OCR Integration

Extract text from images (DeepSeek-OCR):

```python
from HoloLoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner

spinner = DeepSeekOCRSpinner()
text = await spinner.extract_text(image)

# Link photo to extracted text entities
entities = extract_entities(text)
for entity in entities:
    kg.add_edge(photo.token_id, entity, edge_type='CONTAINS_TEXT', weight=1.0)
```

---

## Status Summary

**Phase 1 Complete** ✅:
- [x] PhotoToken dataclass
- [x] PhotoTokenMemory storage engine
- [x] MultimodalEncoder with CLIP
- [x] Structural feature extraction
- [x] Demo script

**Phase 2 (Week 2)** - YarnGraph Integration:
- [ ] Extend YarnGraph node types
- [ ] Multimodal search
- [ ] HoloLoom API integration
- [ ] Integration tests
- [ ] Multimodal demo

**Estimated Time**: 8 hours for Phase 2
**Status**: Ready to begin integration

---

**Next Steps**: Implement `add_photo_node()` and `search_multimodal()` in YarnGraph
