# HoloLoom Multi-Modal Embeddings

**Phase 3A: Multi-Modal Understanding**

Version: 3.0.0
Status: Complete
Date: 2025-11-17

---

## Overview

HoloLoom Phase 3A extends the system from text-only to **multi-modal understanding**, enabling processing of images, audio, and video content. All modalities share a unified **768-dimensional embedding space**, enabling seamless cross-modal retrieval and reasoning.

### Key Capabilities

1. **Image Understanding** - CLIP-based vision-language embeddings with OCR
2. **Audio Processing** - Whisper transcription + acoustic fingerprinting
3. **Video Analysis** - Frame extraction, scene detection, temporal encoding
4. **Cross-Modal Fusion** - Combine embeddings across modalities
5. **Hybrid Search** - Query in any modality, retrieve from any modality

---

## Architecture

### Unified Embedding Space

All modalities project into a shared 768D space:

```
TEXT    → MatryoshkaEmbeddings(768D)
IMAGE   → CLIP Vision Encoder → Projection → 768D
AUDIO   → Whisper + Acoustic Features → 768D
VIDEO   → Frame Embeddings + Audio → Temporal Pool → 768D
```

This enables **cross-modal operations**:
- Text → Image: "find images of cats"
- Image → Text: "find documents about this image"
- Audio → Video: "find videos with similar audio"

### Module Structure

```
HoloLoom/multimodal/
├── base.py              # Protocols, enums, base types
├── image_encoder.py     # CLIP-based image encoding
├── audio_encoder.py     # Whisper + acoustic features
├── video_encoder.py     # Frame + audio processing
├── fusion.py            # Cross-modal fusion strategies
└── search.py            # Multi-modal search engine

HoloLoom/ingestion/parsers/
├── image.py             # Image file parser
├── audio.py             # Audio file parser
└── video.py             # Video file parser
```

---

## Installation

### Minimal Installation (Text + Basic Images)

```bash
pip install transformers torch pillow
```

### Recommended Installation (All Modalities)

```bash
pip install -r HoloLoom/requirements-phase3.txt
```

### System Requirements

**Software:**
- Python 3.8+ (3.10+ recommended)
- ffmpeg binary (for audio/video processing)
- tesseract binary (for OCR, optional)

**Hardware:**
- RAM: 8GB minimum, 16GB recommended
- Storage: 5GB for models
- GPU: Optional but recommended for video

**Models Downloaded:**
- CLIP ViT-B/32: ~350MB
- Whisper (base): ~74MB
- Total: ~500MB

### GPU Support

For faster encoding with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### 1. Image Encoding

```python
from HoloLoom.multimodal import ImageEncoder, Modality

# Create encoder
encoder = ImageEncoder()

# Encode image
embedding = await encoder.encode("cat.jpg")

print(f"Modality: {embedding.modality.value}")
print(f"Dimension: {len(embedding.embedding)}")
print(f"OCR text: {embedding.metadata.get('ocr_text')}")
print(f"Confidence: {embedding.confidence}")
```

### 2. Audio Encoding

```python
from HoloLoom.multimodal import AudioEncoder

# Create encoder
encoder = AudioEncoder()

# Encode audio
embedding = await encoder.encode("speech.mp3")

print(f"Transcription: {embedding.metadata.get('transcription')}")
print(f"Language: {embedding.metadata.get('language')}")
print(f"Duration: {embedding.metadata.get('duration'):.2f}s")
```

### 3. Video Encoding

```python
from HoloLoom.multimodal import VideoEncoder

# Create encoder
encoder = VideoEncoder(
    frame_rate=1.0,           # Extract 1 frame per second
    enable_scene_detection=True,
    enable_audio=True
)

# Encode video
embedding = await encoder.encode("video.mp4")

print(f"Duration: {embedding.metadata.get('duration'):.2f}s")
print(f"Frames: {embedding.metadata.get('extracted_frames')}")
print(f"Scenes: {embedding.metadata.get('scenes')}")
print(f"Transcription: {embedding.metadata.get('transcription')}")
```

### 4. Cross-Modal Fusion

```python
from HoloLoom.multimodal import MultiModalFuser, Modality

# Create fuser
fuser = MultiModalFuser(method="late")

# Fuse embeddings
fused = fuser.fuse_embeddings({
    Modality.TEXT: text_embedding,
    Modality.IMAGE: image_embedding,
    Modality.AUDIO: audio_embedding
})

print(f"Fused dimension: {len(fused.fused_embedding)}")
print(f"Method: {fused.metadata['fusion_method']}")
print(f"Modalities: {fused.metadata['modalities']}")
```

### 5. Multi-Modal Search

```python
from HoloLoom.multimodal import MultiModalSearch, Modality

# Create search engine
search = MultiModalSearch(
    collection_name="hololoom_multimodal",
    qdrant_url="http://localhost:6333"
)

# Index content
await search.index_content(
    content="cat.jpg",
    modality=Modality.IMAGE,
    metadata={"tags": ["animal", "pet"]}
)

# Search: Text → Image
results = await search.search(
    query="cats playing",
    query_modality=Modality.TEXT,
    target_modalities=[Modality.IMAGE, Modality.VIDEO],
    top_k=10
)

for result in results.results:
    print(f"ID: {result.id}")
    print(f"Modality: {result.modality.value}")
    print(f"Score: {result.score:.3f}")
    print(f"Cross-modal: {result.cross_modal}")
```

### 6. Hybrid Search

```python
# Multi-modal query
results = await search.hybrid_search(
    text_query="machine learning tutorial",
    image_query="diagram.jpg",
    weights={
        Modality.TEXT: 0.6,
        Modality.IMAGE: 0.4
    },
    top_k=10
)
```

---

## Components

### 1. Image Encoder

**Technology:** CLIP (Contrastive Language-Image Pre-training)

**Features:**
- Vision-language alignment (512D → 768D projection)
- OCR text extraction (pytesseract)
- Image similarity search
- Supports: PNG, JPG, JPEG, GIF, BMP, TIFF

**Configuration:**

```python
from HoloLoom.multimodal import ImageEncoderConfig, ImageEncoder

config = ImageEncoderConfig(
    model_name="openai/clip-vit-base-patch32",
    enable_ocr=True,
    ocr_lang="eng",
    device="cuda"  # or "cpu"
)

encoder = ImageEncoder(config)
```

**Graceful Degradation:**
- Without CLIP: Falls back to deterministic hash-based embeddings
- Without OCR: Skips text extraction

---

### 2. Audio Encoder

**Technology:** Whisper (speech-to-text) + librosa (acoustic features)

**Features:**
- Speech-to-text transcription
- Acoustic fingerprinting (MFCCs, chroma, spectral contrast)
- Hybrid encoding: 70% text + 30% acoustic
- Supports: MP3, WAV, FLAC, M4A, OGG

**Configuration:**

```python
from HoloLoom.multimodal import AudioEncoderConfig, AudioEncoder

config = AudioEncoderConfig(
    model_name="base",  # Whisper model size
    language=None,      # Auto-detect
    enable_diarization=False,
    device="cpu"
)

encoder = AudioEncoder(config)
```

**Whisper Models:**
- `tiny`: 39MB, fastest, lowest quality
- `base`: 74MB, good balance (recommended)
- `small`: 244MB
- `medium`: 769MB
- `large`: 2.9GB, best quality

**Acoustic Features:**
- MFCCs (20 coefficients)
- Chroma features (12 pitch classes)
- Spectral contrast (7 bands)
- Zero crossing rate
- Spectral rolloff

---

### 3. Video Encoder

**Technology:** OpenCV (frame extraction) + ImageEncoder + AudioEncoder

**Features:**
- Configurable frame sampling rate
- Scene detection (scenedetect)
- Audio track transcription
- Temporal pooling (mean, max, attention)
- Supports: MP4, AVI, MOV, MKV

**Configuration:**

```python
from HoloLoom.multimodal import VideoEncoderConfig, VideoEncoder

config = VideoEncoderConfig(
    frame_rate=1.0,              # FPS to extract
    enable_scene_detection=True,
    enable_audio=True,
    max_frames=300,
    device="cpu"
)

encoder = VideoEncoder(config)
```

**Processing Pipeline:**
1. Extract frames at specified rate
2. Detect scene changes (optional)
3. Encode each frame → 768D
4. Extract audio track → 768D
5. Temporal pooling → single 768D
6. Combine: 70% visual + 30% audio

**Temporal Pooling Strategies:**
- `mean`: Average pooling (default)
- `max`: Element-wise maximum
- `attention`: Weighted by embedding magnitude

---

### 4. Cross-Modal Fusion

**Strategies:**

#### Late Fusion (Default)
Weighted combination of modal embeddings:

```python
fuser = MultiModalFuser(method="late")
fused = fuser.fuse_embeddings(
    embeddings={
        Modality.TEXT: text_emb,
        Modality.IMAGE: image_emb
    },
    weights={
        Modality.TEXT: 0.6,
        Modality.IMAGE: 0.4
    }
)
```

#### Early Fusion
Concatenate features before projection:

```python
fuser = MultiModalFuser(method="early")
fused = fuser.fuse_embeddings(embeddings)
```

#### Hybrid Fusion
Cross-attention between modalities (learnable):

```python
fuser = MultiModalFuser(method="hybrid")
fused = fuser.fuse_embeddings(embeddings)
```

**Configuration:**

```python
from HoloLoom.multimodal import FusionConfig

config = FusionConfig(
    method="late",
    weights={
        Modality.TEXT: 0.4,
        Modality.IMAGE: 0.3,
        Modality.AUDIO: 0.2,
        Modality.VIDEO: 0.1
    },
    normalize=True  # L2 normalize output
)
```

---

### 5. Multi-Modal Search

**Features:**
- Cross-modal retrieval (any modality → any modality)
- Hybrid search with multi-modal queries
- Integration with Qdrant vector store
- Relevance ranking with modal fusion
- Metadata filtering

**Example: Text → Image Search**

```python
search = MultiModalSearch()

# Index images
await search.index_content(
    content="cat1.jpg",
    modality=Modality.IMAGE,
    metadata={"category": "animals"}
)

# Search with text
results = await search.search(
    query="cute cats",
    query_modality=Modality.TEXT,
    target_modalities=[Modality.IMAGE]
)
```

**Example: Image → Similar Content**

```python
results = await search.search(
    query="reference.jpg",
    query_modality=Modality.IMAGE,
    target_modalities=None,  # Search all
    top_k=5
)
```

**Example: Multi-Modal Query**

```python
results = await search.hybrid_search(
    text_query="tutorial on neural networks",
    image_query="diagram.jpg",
    weights={
        Modality.TEXT: 0.7,
        Modality.IMAGE: 0.3
    }
)
```

---

## File Parsers

### Image Parser

```python
from HoloLoom.ingestion.parsers.image import parse_image

text = await parse_image(
    file_path="photo.jpg",
    enable_ocr=True,
    ocr_lang="eng"
)
```

**Output:**
- Image metadata (format, size, resolution)
- EXIF data
- OCR-extracted text

### Audio Parser

```python
from HoloLoom.ingestion.parsers.audio import parse_audio

text = await parse_audio(
    file_path="speech.mp3",
    model_name="base",
    language=None  # Auto-detect
)
```

**Output:**
- Audio metadata (duration, sample rate)
- Whisper transcription
- Detected language

### Video Parser

```python
from HoloLoom.ingestion.parsers.video import parse_video

text = await parse_video(
    file_path="video.mp4",
    extract_audio=True,
    sample_frames=10,
    whisper_model="base"
)
```

**Output:**
- Video metadata (duration, FPS, resolution)
- Frame descriptions
- Audio transcription

---

## Integration with HoloLoom

### File Ingestion Pipeline

```python
from HoloLoom.ingestion.file_processor import FileProcessor
from HoloLoom.multimodal import MultiModalSearch, Modality

# Initialize
processor = FileProcessor()
search = MultiModalSearch()

# Process multi-modal document
chunks = await processor.process_file("document_with_images.pdf")

for chunk in chunks:
    # Detect modality
    if chunk.metadata.get("is_image"):
        modality = Modality.IMAGE
    else:
        modality = Modality.TEXT

    # Index in multi-modal store
    await search.index_content(
        content=chunk.text,
        modality=modality,
        metadata=chunk.metadata
    )
```

### Orchestrator Integration

```python
from HoloLoom.Orchestrator import Orchestrator
from HoloLoom.multimodal import MultiModalSearch, Modality

class MultiModalOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mm_search = MultiModalSearch()

    async def process_multimodal_query(
        self,
        query: str,
        image_query: Optional[str] = None
    ):
        # Hybrid search
        if image_query:
            results = await self.mm_search.hybrid_search(
                text_query=query,
                image_query=image_query
            )
        else:
            results = await self.mm_search.search(
                query=query,
                query_modality=Modality.TEXT
            )

        # Process results...
        return results
```

---

## Testing

Run comprehensive test suite:

```bash
# From repository root
PYTHONPATH=. python HoloLoom/test_multimodal.py
```

**Test Coverage:**
1. Base types and protocols
2. Image encoder (with/without CLIP)
3. Audio encoder (with/without Whisper)
4. Video encoder components
5. Fusion strategies (late, early, hybrid)
6. Cross-modal search
7. File parsers
8. End-to-end integration

**Expected Output:**
```
✓ Modality.TEXT enum
✓ ModalEmbedding creation
✓ LateFusion output dimension
✓ ImageEncoder creation
...
Test Summary: 45/45 passed
SUCCESS: All tests passed!
```

---

## Performance Considerations

### Encoding Speed

| Modality | Model     | Speed (CPU) | Speed (GPU) |
|----------|-----------|-------------|-------------|
| Text     | MiniLM    | ~100 ms     | ~20 ms      |
| Image    | CLIP      | ~200 ms     | ~50 ms      |
| Audio    | Whisper   | ~1-5s/min   | ~0.5s/min   |
| Video    | Full      | ~10s/min    | ~3s/min     |

### Memory Usage

| Operation          | RAM Usage  |
|-------------------|------------|
| CLIP model        | ~400 MB    |
| Whisper (base)    | ~1 GB      |
| Whisper (large)   | ~3 GB      |
| Video processing  | ~2-4 GB    |

### Optimization Tips

1. **Use GPU** for image/video processing
2. **Batch encode** images for efficiency
3. **Cache embeddings** to avoid re-encoding
4. **Use smaller Whisper models** for real-time
5. **Reduce frame rate** for long videos

---

## Troubleshooting

### Issue: CLIP model download fails

**Solution:**
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Download manually
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Issue: Whisper runs out of memory

**Solution:**
```python
# Use smaller model
encoder = AudioEncoder(AudioEncoderConfig(model_name="tiny"))

# Or process in chunks
```

### Issue: ffmpeg not found for video

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Issue: OCR not working

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Test
tesseract --version
```

---

## Future Enhancements

### Planned for Phase 3B

1. **Fine-tuning**: Domain-specific CLIP/Whisper models
2. **Streaming**: Real-time video/audio processing
3. **3D Content**: Point clouds, meshes
4. **Document Layout**: Preserve spatial structure
5. **Multi-lingual**: Improved language support

### Experimental Features

- **Few-shot learning**: Adapt to new modalities
- **Active learning**: Select informative samples
- **Explainability**: Visualize cross-modal alignments

---

## References

### Models

- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **Whisper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **Matryoshka**: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)

### Libraries

- [transformers](https://github.com/huggingface/transformers) - CLIP implementation
- [openai-whisper](https://github.com/openai/whisper) - Speech-to-text
- [librosa](https://librosa.org/) - Audio analysis
- [opencv-python](https://github.com/opencv/opencv-python) - Video processing

---

## License

HoloLoom Multi-Modal is part of the HoloLoom project.

See main repository LICENSE for details.

---

## Support

For issues, questions, or contributions:

1. Check [MULTIMODAL.md](./MULTIMODAL.md) (this document)
2. Run tests: `python HoloLoom/test_multimodal.py`
3. Review examples in module docstrings
4. Check [CLAUDE.md](../CLAUDE.md) for development guide

---

**Phase 3A Complete** ✓

Multi-modal embeddings enable HoloLoom to understand and reason across text, images, audio, and video in a unified framework.
