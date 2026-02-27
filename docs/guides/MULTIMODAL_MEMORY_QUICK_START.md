# Multimodal Memory Quick Start

**Last Updated**: November 7, 2025

Quick reference for using HoloLoom's multimodal memory system.

---

## Installation

```bash
# Minimal (structural features only)
pip install Pillow numpy

# Full (with CLIP for semantic matching)
pip install Pillow numpy openai-clip torch
```

---

## Basic Usage

### 1. Store Photos

```python
from hololoom import hololoom

async with HoloLoom() as loom:
    # Simple photo storage
    photo = await loom.remember_photo(
        "diagram.png",
        caption="System architecture diagram",
        tags=["diagram", "architecture"]
    )

    print(f"Stored: {photo.token_id}")
```

### 2. Recall Multimodal

```python
# Text-only recall (backward compatible)
text_memories = await loom.recall("What is the architecture?")

# Multimodal recall (text + photos)
results = await loom.recall(
    "Show me the architecture diagram",
    include_photos=True
)

print(f"Text: {len(results['text'])}, Photos: {len(results['photos'])}")
```

### 3. Link Photos to Memories

```python
# Store text
text_mem = await loom.experience("We discussed the architecture")

# Store photo linked to text
photo = await loom.remember_photo(
    "whiteboard.jpg",
    caption="Whiteboard from meeting",
    link_to_memory=text_mem.id  # Automatic linking
)

# Or link separately
await loom.link_photo_to_memory(
    photo.token_id,
    text_mem.id,
    relationship="ILLUSTRATES"
)
```

### 4. Find Similar Photos

```python
# Find visually similar images (CLIP)
similar = await loom.find_similar_photos("query.png", k=5)

for photo in similar:
    score = photo.metadata['score']
    print(f"{photo.caption}: {score:.3f}")
```

### 5. Filter by Tags

```python
# Get all diagram photos
diagrams = await loom.get_photos_by_tag("diagram")

for photo in diagrams:
    print(f"- {photo.caption}")
```

---

## API Reference

### HoloLoom Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `remember_photo(image, caption, tags, link_to_memory)` | Store photo | PhotoToken |
| `recall(query, include_photos=True)` | Multimodal search | Dict[str, List] |
| `find_similar_photos(query_image, k)` | Visual similarity | List[PhotoToken] |
| `get_photos_by_tag(tag, k)` | Tag filtering | List[PhotoToken] |
| `link_photo_to_memory(photo_id, memory_id, relationship)` | Manual linking | None |

### PhotoToken Attributes

```python
photo.token_id           # Unique ID
photo.caption            # Text description
photo.tags               # ["diagram", "architecture"]
photo.entities           # ["system", "architecture"]
photo.clip_embedding     # 512D CLIP vector
photo.structural_features # {brightness, aspect_ratio, ...}
photo.metadata           # Arbitrary data
```

---

## Performance

| Operation | Latency | Scales To |
|-----------|---------|-----------|
| remember_photo() | 150-200ms | 100K images |
| recall(include_photos=True) | 100-150ms | 10K images |
| find_similar_photos() | 90-120ms | 1K images |
| get_photos_by_tag() | <5ms | Unlimited |

---

## Examples

### Full Cycle Demo

```python
from hololoom import hololoom

async with HoloLoom() as loom:
    # 1. Experience text
    text = await loom.experience(
        "We discussed the system architecture at the meeting"
    )

    # 2. Remember photo
    photo = await loom.remember_photo(
        "architecture.png",
        caption="System architecture diagram",
        tags=["diagram", "architecture"],
        link_to_memory=text.id
    )

    # 3. Multimodal recall
    results = await loom.recall(
        "Show me the architecture",
        include_photos=True
    )

    # 4. Process results
    for mem in results['text']:
        print(f"Text: {mem.text}")

    for photo in results['photos']:
        print(f"Photo: {photo.caption}")
```

### Visual Similarity

```python
from PIL import Image
import numpy as np

# Load query image
query_img = np.array(Image.open("reference.png"))

# Find similar
similar = await loom.find_similar_photos(query_img, k=5)

for i, photo in enumerate(similar, 1):
    score = photo.metadata['score']
    print(f"{i}. {photo.caption} ({score:.3f})")
```

---

## Edge Types

| Edge Type | Meaning | Example |
|-----------|---------|---------|
| DEPICTS | Photo shows entity | photo → "architecture" |
| TAGGED_AS | Photo has tag | photo → "diagram" |
| ILLUSTRATES | Photo explains memory | photo → memory_id |
| SIMILAR_TO | Photos visually similar | photo ↔ photo |
| OCCURRED_AT | Photo from time | photo → time_thread |

---

## Tips

1. **Always use captions**: Improves search without CLIP
2. **Use tags liberally**: Fast filtering without embeddings
3. **Link related content**: Creates knowledge graph connections
4. **Batch operations**: Store multiple photos then query
5. **Use context managers**: Automatic cleanup

---

## Troubleshooting

**CLIP not available**:
```bash
pip install openai-clip torch
```

**Low search quality without CLIP**:
- Use descriptive captions
- Add relevant tags
- Rely on tag filtering

**Slow performance (>10K images)**:
- Consider vector database (FAISS, Qdrant)
- Reduce CLIP embedding dimension
- Add indexing

---

## Next Steps

- [Full documentation](MULTIMODAL_MEMORY_INTEGRATION_COMPLETE.md)
- [Integration guide](PHOTO_TOKENS_YARNGRAPH_INTEGRATION.md)
- [Design document](VISUAL_TOKENS_DESIGN.md)
- [Phase 1 summary](VISUAL_TOKENS_PHASE_1_COMPLETE.md)

---

**Questions?** See the comprehensive docs or file an issue.
