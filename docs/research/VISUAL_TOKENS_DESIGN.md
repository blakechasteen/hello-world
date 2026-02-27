# Visual Tokens: Multimodal Memory for HoloLoom

**Date**: November 7, 2025
**Status**: Design Phase
**Goal**: Enable HoloLoom to remember and reason about images

---

## 🎯 Vision

**"Memories aren't just text. They're photos, diagrams, screenshots, faces."**

Extend HoloLoom's memory system to store and retrieve visual information alongside text, creating a truly multimodal memory system.

---

## 🧠 Core Concept

### What Are Visual Tokens?

**Visual Tokens** are first-class memory citizens that represent visual information:
- **Photos**: User uploads, screenshots, diagrams
- **Structural Tokens**: Extracted visual features (edges, shapes, layouts)
- **Semantic Tokens**: High-level concepts (faces, objects, scenes)

### Why Visual Tokens?

```python
# Text-only memory (current):
"I saw a red ball" → stored as text embedding

# Visual token memory (new):
red_ball.jpg → stored as:
  - Image embedding (visual similarity)
  - Caption embedding (semantic

 similarity)
  - Structural features (color, shape, layout)
  - Metadata (timestamp, location, tags)
```

**Benefits**:
1. **Multimodal retrieval**: "Show me the diagram we discussed" → returns image
2. **Visual similarity**: "Find similar photos" → uses image embeddings
3. **Richer context**: Images provide context that text descriptions miss
4. **Screenshot memory**: Remember UI, diagrams, code screenshots

---

## 🏗️ Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Storage Layer (PhotoTokenMemory)                   │
│  - Store image bytes, embeddings, metadata                 │
│  - Efficient compression and retrieval                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Encoding Layer (MultimodalEncoder)                 │
│  - Image → CLIP embeddings (512D)                          │
│  - Image → Captions → text embeddings                      │
│  - Image → Structural features (color, layout)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Integration Layer (YarnGraph Extension)            │
│  - Visual tokens as graph nodes                            │
│  - Edges: text→image, image→image relationships            │
│  - Multimodal retrieval (text OR image queries)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Core Data Structures

### 1. PhotoToken

```python
@dataclass
class PhotoToken:
    """A visual memory token."""

    # Identity
    token_id: str  # Unique ID
    timestamp: float  # When created

    # Visual data
    image_data: bytes  # Compressed image (JPEG/PNG)
    image_hash: str  # SHA256 for deduplication
    dimensions: Tuple[int, int]  # (width, height)

    # Embeddings (multimodal)
    clip_embedding: np.ndarray  # CLIP visual embedding (512D)
    caption_embedding: Optional[np.ndarray]  # Text embedding of caption
    structural_features: Dict[str, float]  # color, edges, layout

    # Semantic info
    caption: Optional[str]  # Auto-generated or user-provided
    tags: List[str]  # ["diagram", "ui", "face", "document"]
    entities: List[str]  # Detected objects ["person", "laptop", "coffee"]

    # Context
    source: str  # "upload", "screenshot", "url"
    metadata: Dict[str, Any]  # Arbitrary metadata

    def to_yarn_node(self) -> Dict:
        """Convert to YarnGraph node format."""
        return {
            'id': self.token_id,
            'type': 'photo_token',
            'timestamp': self.timestamp,
            'caption': self.caption,
            'tags': self.tags,
            'embeddings': {
                'clip': self.clip_embedding.tolist(),
                'caption': self.caption_embedding.tolist() if self.caption_embedding is not None else None
            },
            'metadata': self.metadata
        }
```

### 2. PhotoTokenMemory

```python
class PhotoTokenMemory:
    """Storage and retrieval for visual tokens."""

    def __init__(
        self,
        storage_path: str = "./photo_memory",
        max_image_size: int = 2048,  # Max dimension
        compression_quality: int = 85  # JPEG quality
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Image storage
        self.images_dir = self.storage_path / "images"
        self.images_dir.mkdir(exist_ok=True)

        # Embeddings storage (fast retrieval)
        self.embeddings_file = self.storage_path / "embeddings.npz"
        self.metadata_file = self.storage_path / "metadata.json"

        # In-memory index
        self.tokens: Dict[str, PhotoToken] = {}
        self.clip_index: Optional[np.ndarray] = None  # Matrix of all CLIP embeddings

        # Load existing tokens
        self._load_tokens()

    async def store(
        self,
        image: Union[bytes, np.ndarray, str],  # bytes, array, or file path
        caption: Optional[str] = None,
        tags: List[str] = None,
        source: str = "upload",
        metadata: Dict = None
    ) -> PhotoToken:
        """Store a photo as a visual token."""
        pass

    async def retrieve_by_text(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[PhotoToken, float]]:
        """Retrieve photos matching text query (using CLIP)."""
        pass

    async def retrieve_by_image(
        self,
        query_image: Union[bytes, np.ndarray],
        k: int = 5
    ) -> List[Tuple[PhotoToken, float]]:
        """Retrieve similar photos (using CLIP embedding similarity)."""
        pass

    async def retrieve_by_tags(
        self,
        tags: List[str],
        k: int = 10
    ) -> List[PhotoToken]:
        """Retrieve photos by tags."""
        pass
```

### 3. MultimodalEncoder

```python
class MultimodalEncoder:
    """Encode images into multiple representations."""

    def __init__(self, use_clip: bool = True):
        self.use_clip = use_clip

        if use_clip:
            # CLIP for image-text matching
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")

        # Optional: Image captioning model
        self.caption_model = None  # Could add BLIP, etc.

    async def encode_image(
        self,
        image: np.ndarray  # RGB array
    ) -> Dict[str, np.ndarray]:
        """Encode image to multiple embeddings."""

        embeddings = {}

        # CLIP embedding (512D)
        if self.use_clip:
            clip_emb = self._encode_clip(image)
            embeddings['clip'] = clip_emb

        # Structural features (color, edges, layout)
        structural = self._extract_structural_features(image)
        embeddings['structural'] = structural

        return embeddings

    def _encode_clip(self, image: np.ndarray) -> np.ndarray:
        """Encode image with CLIP."""
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        image_input = self.clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)

        return image_features.cpu().numpy()[0]

    def _extract_structural_features(self, image: np.ndarray) -> np.ndarray:
        """Extract basic structural features (color, edges, etc.)."""
        features = []

        # Color histogram (RGB channels)
        for channel in range(3):
            hist, _ = np.histogram(image[:, :, channel], bins=16, range=(0, 256))
            features.extend(hist / hist.sum())  # Normalize

        # Mean brightness
        brightness = image.mean() / 255.0
        features.append(brightness)

        # Aspect ratio
        h, w = image.shape[:2]
        aspect = w / h
        features.append(aspect)

        return np.array(features, dtype=np.float32)
```

---

## 🔗 YarnGraph Integration

### Visual Nodes in Knowledge Graph

```python
# YarnGraph node types
NODE_TYPES = {
    'text': TextNode,        # Existing
    'photo': PhotoNode,      # NEW!
    'video': VideoNode,      # Future
    'audio': AudioNode       # Future
}

# Example YarnGraph with photos
kg = YarnGraph()

# Add text memory
text_node = kg.add_node(
    type='text',
    content="We discussed the system architecture",
    timestamp=now()
)

# Add photo memory
photo_node = kg.add_node(
    type='photo',
    token_id="photo_abc123",
    caption="System architecture diagram",
    timestamp=now()
)

# Link them
kg.add_edge(text_node, photo_node, edge_type="REFERENCES")

# Multimodal retrieval
results = kg.search_multimodal(
    query="Show me the architecture diagram",
    return_types=['text', 'photo']
)
# Returns: [text_node, photo_node]
```

---

## 🚀 Phase 1 Implementation (4-6 hours)

### Minimal Viable Product

**Goal**: Store and retrieve photos in memory

**Deliverables**:
1. `PhotoToken` dataclass
2. `PhotoTokenMemory` storage class
3. `MultimodalEncoder` (CLIP-based)
4. Basic YarnGraph integration
5. Demo script

**NOT in Phase 1**:
- Auto-captioning (add in Phase 2)
- DeepSeek-OCR (requires GPU, Phase 3)
- Video/audio tokens (future)

### File Structure

```
hololoom/
├── memory/
│   ├── photo_tokens.py (NEW - 400 lines)
│   │   - PhotoToken dataclass
│   │   - PhotoTokenMemory class
│   │   - Image compression/storage
│   │
│   ├── multimodal_encoder.py (NEW - 300 lines)
│   │   - MultimodalEncoder class
│   │   - CLIP integration
│   │   - Structural feature extraction
│   │
│   └── graph.py (MODIFY - add photo nodes)
│       - Add photo node type
│       - Multimodal search
│
├── spinningWheel/
│   └── image_spinner.py (NEW - 200 lines)
│       - Convert images → PhotoTokens
│       - Batch processing
│
demos/
└── demo_photo_memory.py (NEW - 150 lines)
    - Upload photo
    - Query by text
    - Query by similar image
    - Show results
```

---

## 📝 API Design

### User-Facing API

```python
from hololoom import hololoom
from hololoom.memory.photo_tokens import PhotoTokenMemory

async with HoloLoom() as loom:
    # Store a photo
    photo_token = await loom.remember_photo(
        "path/to/diagram.png",
        caption="System architecture diagram",
        tags=["diagram", "architecture"]
    )

    # Query by text
    results = await loom.recall("Show me the architecture diagram")
    # Returns: [text_memories, photo_token]

    # Query by similar image
    similar = await loom.find_similar_photos("path/to/query.png", k=5)
    # Returns: [photo_token1, photo_token2, ...]

    # Get photo
    image_bytes = photo_token.image_data
```

---

## 🔬 Technical Decisions

### 1. Image Embeddings: CLIP

**Why CLIP?**
- Pre-trained on 400M image-text pairs
- 512D embeddings (manageable size)
- Text-to-image and image-to-image similarity
- Fast inference (CPU-friendly)

**Alternatives considered**:
- ResNet: Visual only (no text matching)
- DINO: Self-supervised (no text matching)
- Custom: Too expensive to train

### 2. Storage: Local Files + NPZ

**Why local storage?**
- Simple (no database setup)
- Fast (memory-mapped NPZ)
- Portable (easy to backup/share)

**Alternatives**:
- Database (PostgreSQL): Overkill for MVP
- Object storage (S3): Network latency
- In-memory only: Not persistent

### 3. Compression: JPEG @ 85% Quality

**Why JPEG?**
- 5-10× compression
- Minimal quality loss at 85%
- Fast encoding/decoding
- Universal support

**Alternatives**:
- PNG: Lossless but 3× larger
- WebP: Better compression but less supported
- AVIF: Cutting-edge but slow encoding

---

## 📊 Performance Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| **Store photo** | <200ms | CLIP encoding + disk write |
| **Retrieve by text** | <50ms | CLIP text encoding + similarity search |
| **Retrieve by image** | <100ms | CLIP image encoding + similarity search |
| **Load existing tokens** | <1s | Load from disk on startup |

**Memory usage**: ~500 bytes per token (excluding image data)

---

## 🎬 Demo Scenario

```python
# Scenario: Remembering a meeting with diagrams

# 1. Store meeting notes (text)
await loom.experience("We discussed the new API architecture")

# 2. Store diagram (photo)
await loom.remember_photo(
    "whiteboard_photo.jpg",
    caption="API architecture sketch from meeting",
    tags=["meeting", "architecture", "api"]
)

# 3. Later: Recall by text
results = await loom.recall("What was the API architecture?")
# Returns: text memory + photo

# 4. Show the diagram
for result in results:
    if isinstance(result, PhotoToken):
        display_image(result.image_data)
```

---

## 🔮 Future Phases

### Phase 2: Advanced Features (8 hours)
- Auto-captioning (BLIP model)
- Face detection and recognition
- Object detection (YOLO)
- Semantic image segmentation

### Phase 3: DeepSeek-OCR (requires GPU, 12 hours)
- OCR for text in images
- Document understanding
- Table extraction
- Handwriting recognition

### Phase 4: Video Tokens (16 hours)
- Video → frame sequences
- Motion tracking
- Event detection
- Video search

---

## ✅ Success Criteria

**Phase 1 Complete When**:
- [ ] Can store photos as PhotoTokens
- [ ] Can retrieve photos by text query (CLIP)
- [ ] Can retrieve similar photos by image query
- [ ] Photos integrated with YarnGraph
- [ ] Demo works end-to-end
- [ ] Performance targets met

---

**Status**: Design complete, ready to implement
**Next**: Create PhotoToken and PhotoTokenMemory classes
**Estimated Time**: 4-6 hours for Phase 1
