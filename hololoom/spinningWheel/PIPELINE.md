# HoloLoom SpinningWheel Pipeline Architecture

**Version**: 2.0
**Last Updated**: November 2025
**Philosophy**: *"If you need to configure it, we failed."*

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Data Flow Architecture](#data-flow-architecture)
3. [Stage Details](#stage-details)
4. [Modality Processing](#modality-processing)
5. [Fusion Strategies](#fusion-strategies)
6. [Memory Integration](#memory-integration)
7. [Performance Characteristics](#performance-characteristics)
8. [Examples by Source Type](#examples-by-source-type)

---

## Pipeline Overview

The SpinningWheel transforms **any input** into **queryable memory** through a 5-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPINNING WHEEL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Raw Input (Anything)
    │
    ├─ Text string
    ├─ File path (PDF, image, audio, CSV)
    ├─ URL (webpage, YouTube, API)
    ├─ Bytes (raw data)
    ├─ Dict (structured data)
    └─ List (multi-modal)
    │
    ▼
┌───────────────────────────────────┐
│  STAGE 1: Input Detection         │  InputRouter.detect_modality()
│  Auto-detect input type            │  - File extension analysis
│  No configuration required         │  - Magic number detection
└───────────────────────────────────┘  - Content inspection
    │
    ▼
┌───────────────────────────────────┐
│  STAGE 2: Modality Processing     │  Processor.process()
│  Route to specialized processor    │  - TEXT → TextProcessor
│  Extract modality-specific features│  - IMAGE → ImageProcessor
└───────────────────────────────────┘  - AUDIO → AudioProcessor
    │                                   - STRUCTURED → StructuredDataProcessor
    ▼
┌───────────────────────────────────┐
│  STAGE 3: Feature Extraction       │  ProcessedInput
│  Unified representation            │  - content: str (human-readable)
│  Cross-modal embeddings            │  - embedding: np.ndarray (384D)
└───────────────────────────────────┘  - features: Dict (modality-specific)
    │                                   - confidence: float (0.0-1.0)
    ▼
┌───────────────────────────────────┐
│  STAGE 4: Multi-Modal Fusion       │  MultiModalFusion.fuse()
│  (Optional - if multiple inputs)   │  - Attention-based fusion
│  Combine modalities into one       │  - Concatenation
└───────────────────────────────────┘  - Averaging/Max pooling
    │
    ▼
┌───────────────────────────────────┐
│  STAGE 5: Shard Creation           │  MultiModalSpinner._create_shard()
│  Convert to MemoryShard(s)         │  - Entity extraction
│  Ready for memory ingestion        │  - Motif detection
└───────────────────────────────────┘  - Metadata preservation
    │
    ▼
MemoryShard(s) → Memory Backend
    ├─ INMEMORY (NetworkX graph)
    ├─ HYBRID (Neo4j + Qdrant)
    └─ HYPERSPACE (Advanced gated multipass)
```

---

## Data Flow Architecture

### Phase 1: Input Detection (0-2ms)

**InputRouter** auto-detects input type using multiple signals:

```python
def detect_modality(input_data) -> ModalityType:
    """
    Detection Priority:
    1. Explicit modality key in dict
    2. File extension (.jpg, .mp3, .pdf)
    3. Magic numbers (PNG: \\x89PNG, JPEG: \\xff\\xd8)
    4. Content inspection (UTF-8 decodability)
    5. Default to TEXT
    """
```

**Detection Matrix**:

| Input Type | Detection Method | Confidence | Fallback |
|------------|------------------|------------|----------|
| `"hello world"` | String → TEXT | 100% | - |
| `"/path/image.jpg"` | Extension + exists() | 95% | TEXT if not found |
| `b'\x89PNG\r\n\x1a\n...'` | Magic number | 100% | TEXT if unknown |
| `{"text": "..."}` | Dict key inspection | 90% | STRUCTURED |
| `[text, image]` | List → MULTIMODAL | 100% | - |

### Phase 2: Modality Processing (5-500ms)

Each processor implements `InputProcessorProtocol`:

```python
class InputProcessorProtocol(Protocol):
    async def process(input_data, **kwargs) -> ProcessedInput
    def get_modality() -> ModalityType
    def is_available() -> bool  # Graceful degradation
```

**Processor Responsibilities**:

```
TextProcessor:
    Input: str, Path (to .txt/.md)
    Output: ProcessedInput
        - content: Full text
        - embedding: 384D semantic vector
        - features.entities: NER results (spaCy)
        - features.topics: Key topics
        - features.sentiment: Polarity/subjectivity

ImageProcessor:
    Input: Path (to .jpg/.png), bytes
    Output: ProcessedInput
        - content: Image caption/description
        - embedding: CLIP visual embedding
        - features.objects: Detected objects
        - features.ocr_text: Extracted text
        - features.colors: Dominant colors

AudioProcessor:
    Input: Path (to .wav/.mp3), bytes
    Output: ProcessedInput
        - content: Transcript
        - embedding: Audio features
        - features.transcript: Speech-to-text
        - features.emotion: Detected emotion
        - features.speaker_count: Number of speakers

StructuredDataProcessor:
    Input: Dict, Path (to .json/.csv)
    Output: ProcessedInput
        - content: Text summary of data
        - embedding: Schema embedding
        - features.schema: Column types
        - features.summary_stats: Mean, std, etc.
```

### Phase 3: Feature Extraction (Embedded in Phase 2)

**Unified `ProcessedInput` Structure**:

```python
@dataclass
class ProcessedInput:
    # Core fields (required)
    modality: ModalityType          # TEXT, IMAGE, AUDIO, STRUCTURED, MULTIMODAL
    content: str                     # Human-readable description
    embedding: np.ndarray            # 384D feature vector

    # Metadata
    confidence: float = 1.0          # Processing confidence (0.0-1.0)
    source: Optional[str] = None     # File path, URL, etc.

    # Modality-specific features
    features: Dict[str, Any]         # TextFeatures, ImageFeatures, etc.

    # Cross-modal alignment
    aligned_embeddings: Dict[ModalityType, np.ndarray]
```

**Feature Extraction Examples**:

```python
# Text Input
ProcessedInput(
    modality=ModalityType.TEXT,
    content="Thompson Sampling balances exploration and exploitation...",
    embedding=np.array([0.12, -0.34, ...]),  # 384D
    features={
        'entities': [
            {'text': 'Thompson Sampling', 'label': 'ALGORITHM'},
            {'text': 'exploration', 'label': 'CONCEPT'}
        ],
        'topics': ['reinforcement_learning', 'bandits'],
        'sentiment': {'polarity': 0.1, 'subjectivity': 0.3}
    }
)

# Image Input
ProcessedInput(
    modality=ModalityType.IMAGE,
    content="A diagram showing neural network architecture with labeled layers",
    embedding=np.array([0.45, 0.23, ...]),  # 384D CLIP
    features={
        'objects': [
            {'label': 'diagram', 'confidence': 0.95},
            {'label': 'neural network', 'confidence': 0.88}
        ],
        'ocr_text': 'Input Layer\nHidden Layer\nOutput Layer',
        'colors': ['#2E86AB', '#A23B72', '#F18F01']
    }
)
```

### Phase 4: Multi-Modal Fusion (10-50ms, optional)

**Fusion Strategies**:

#### 1. **Attention-Based Fusion** (Default, Highest Quality)

```python
async def fuse_attention(inputs: List[ProcessedInput]) -> ProcessedInput:
    """
    Cross-attention between modalities.

    Each modality attends to others:
    - Text attends to image (visual grounding)
    - Image attends to text (semantic context)
    - Audio attends to both (multi-sensory alignment)

    Output: Weighted combination based on relevance
    """
    embeddings = [inp.embedding for inp in inputs]

    # Compute attention weights (simplified)
    Q = embeddings[0]  # Query from first modality
    K = np.stack(embeddings)  # Keys from all modalities
    V = K  # Values = embeddings

    # Attention: softmax(Q @ K^T) @ V
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    fused_embedding = weights @ V

    return ProcessedInput(
        modality=ModalityType.MULTIMODAL,
        content=combine_contents(inputs),
        embedding=fused_embedding,
        features={'components': [inp.features for inp in inputs]}
    )
```

#### 2. **Concatenation Fusion** (Simple, Fast)

```python
async def fuse_concat(inputs: List[ProcessedInput]) -> ProcessedInput:
    """
    Concatenate embeddings along feature dimension.

    Input: [text_384D, image_384D, audio_384D]
    Output: 1152D fused embedding

    Pros: Preserves all information
    Cons: High dimensionality
    """
    fused_embedding = np.concatenate([inp.embedding for inp in inputs])
    # Optional: Project back to 384D with learned matrix
```

#### 3. **Average/Max Pooling** (Baseline)

```python
async def fuse_average(inputs: List[ProcessedInput]) -> ProcessedInput:
    """Element-wise average (or max) across embeddings."""
    fused_embedding = np.mean([inp.embedding for inp in inputs], axis=0)
```

**Fusion Decision Matrix**:

| Modalities | Strategy | Rationale | Latency |
|------------|----------|-----------|---------|
| Text + Image | Attention | Visual grounding improves retrieval | 30ms |
| Text + Audio | Attention | Semantic + acoustic features | 25ms |
| Image + Audio | Average | Weak cross-modal correlation | 5ms |
| Text + Image + Audio | Attention | Full multi-sensory context | 50ms |
| 5+ modalities | Concat → PCA | Preserve diversity, reduce dims | 40ms |

### Phase 5: Shard Creation (1-5ms)

**MultiModalSpinner** converts `ProcessedInput` → `MemoryShard`:

```python
def _create_shard(processed: ProcessedInput) -> MemoryShard:
    """
    Extract entities and motifs from features.

    Entity Extraction:
    - TEXT: Use NER results from features.entities
    - IMAGE: Use object labels from features.objects
    - AUDIO: Use transcript entities
    - STRUCTURED: Use column names, key values

    Motif Extraction:
    - TEXT: Topics from features.topics
    - IMAGE: Scene classification
    - AUDIO: Emotion, speaker count
    - STRUCTURED: Schema patterns
    """
    entities = extract_entities(processed.features)
    motifs = extract_motifs(processed.features)

    return MemoryShard(
        id=f"{processed.modality.value}_{hash(processed.content[:100])}",
        text=processed.content,
        episode=processed.source,
        entities=entities,
        motifs=motifs,
        metadata={
            'modality_type': processed.modality.name,
            'confidence': processed.confidence,
            'embedding': processed.embedding.tolist(),
            'features': processed.features
        }
    )
```

**Shard Metadata Preservation**:

```python
MemoryShard(
    id="text_1234567890",
    text="Thompson Sampling balances exploration...",
    episode="research_notes_2025_11_02",
    entities=["Thompson Sampling", "exploration", "exploitation"],
    motifs=["reinforcement_learning", "bandits", "decision_making"],
    metadata={
        'modality_type': 'TEXT',
        'confidence': 0.95,
        'embedding': [0.12, -0.34, ...],  # 384D
        'features': {
            'entities': [...],
            'topics': [...],
            'sentiment': {...}
        },
        'spinner': 'MultiModalSpinner',
        'processing_time_ms': 12.5
    }
)
```

---

## Stage Details

### Stage 1: Input Detection

**File**: `hololoom/input/router.py` (`InputRouter.detect_modality()`)

**Latency**: 0.5-2ms

**Logic**:

```python
def detect_modality(input_data: InputData) -> ModalityType:
    # Priority 1: Explicit modality in dict
    if isinstance(input_data, dict) and 'modality' in input_data:
        return ModalityType(input_data['modality'])

    # Priority 2: File path analysis
    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if path.exists():
            return _detect_from_file(path)  # Extension-based
        else:
            return ModalityType.TEXT  # Assume text content

    # Priority 3: Magic numbers (bytes)
    if isinstance(input_data, bytes):
        return _detect_from_bytes(input_data)

    # Priority 4: Default
    return ModalityType.TEXT
```

**Extension Mapping**:

```python
IMAGE: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp, .svg
AUDIO: .wav, .mp3, .ogg, .flac, .m4a, .aac, .wma
VIDEO: .mp4, .avi, .mov, .mkv, .webm, .flv
STRUCTURED: .json, .csv, .tsv, .xml, .yaml, .yml
TEXT: .txt, .md, .rst, .log, or unknown
```

**Magic Numbers**:

```python
PNG:  b'\x89PNG\r\n\x1a\n'
JPEG: b'\xff\xd8'
GIF:  b'GIF'
WAV:  b'RIFF' + b'WAVE' (offset 8)
MP3:  b'ID3' or b'\xff\xfb'
```

### Stage 2: Modality Processing

**Files**:
- `hololoom/input/text_processor.py` (TextProcessor)
- `hololoom/input/image_processor.py` (ImageProcessor)
- `hololoom/input/audio_processor.py` (AudioProcessor)
- `hololoom/input/structured_processor.py` (StructuredDataProcessor)

**Latency**:
- TEXT: 5-20ms (spaCy NER: +10ms)
- IMAGE: 50-200ms (CLIP: +150ms, OCR: +100ms)
- AUDIO: 200-500ms (Whisper STT: +300ms)
- STRUCTURED: 10-50ms (pandas parsing)

**Graceful Degradation**:

```python
class TextProcessor:
    def __init__(
        self,
        embedder=None,
        use_spacy=True,  # Optional: Disable if not available
        use_textblob=True  # Optional: Sentiment analysis
    ):
        self.embedder = embedder or SimpleEmbedder()

        # Try to load spaCy
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except (ImportError, OSError):
            print("Warning: spaCy not available, using simple entity extraction")
            self.nlp = None
            self.has_spacy = False

    async def process(self, text: str) -> ProcessedInput:
        # Always works: basic processing
        embedding = self.embedder.embed(text)

        # Optional: Advanced NER
        if self.has_spacy:
            doc = self.nlp(text)
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        else:
            # Fallback: Capitalized words
            entities = [{'text': w, 'label': 'UNKNOWN'} for w in re.findall(r'\b[A-Z][a-z]+\b', text)]

        return ProcessedInput(...)
```

### Stage 3: Feature Extraction

**Unified Feature Types**:

```python
@dataclass
class TextFeatures:
    entities: List[Dict[str, str]]   # [{'text': 'Python', 'label': 'LANGUAGE'}]
    sentiment: Dict[str, float]       # {'polarity': 0.2, 'subjectivity': 0.4}
    topics: List[str]                 # ['machine_learning', 'python']
    keyphrases: List[str]             # ['neural network', 'training loop']
    language: str = "en"

@dataclass
class ImageFeatures:
    objects: List[Dict]               # [{'label': 'cat', 'confidence': 0.95}]
    scene: Optional[str]              # 'indoor', 'outdoor', 'urban'
    caption: Optional[str]            # Generated image description
    ocr_text: Optional[str]           # Text extracted from image
    colors: List[str]                 # ['#FF5733', '#C70039']
    dimensions: tuple                 # (width, height)

@dataclass
class AudioFeatures:
    transcript: Optional[str]         # Speech-to-text output
    language: Optional[str]           # 'en', 'es', 'fr'
    speaker_count: int                # Number of speakers
    emotion: Optional[str]            # 'happy', 'neutral', 'sad'
    acoustic: Dict[str, Any]          # MFCC, pitch, energy
    duration: float                   # Seconds

@dataclass
class StructuredFeatures:
    schema: Dict[str, str]            # {'name': 'str', 'age': 'int'}
    row_count: int
    column_count: int
    relationships: List[Dict]         # Foreign keys, joins
    summary_stats: Dict[str, Any]     # Mean, std, min, max
```

### Stage 4: Multi-Modal Fusion

**File**: `hololoom/input/fusion.py` (`MultiModalFusion.fuse()`)

**When Fusion Occurs**:
1. User provides `List[inputs]` to `spin()`
2. Multiple files in `spin_directory()`
3. Cross-modal query (text + image + audio)

**Fusion Process**:

```python
async def fuse(
    inputs: List[ProcessedInput],
    strategy: str = "attention"
) -> ProcessedInput:
    """
    Fuse multiple modalities into unified representation.

    Steps:
    1. Align embeddings to same dimension (if needed)
    2. Apply fusion strategy (attention/concat/average)
    3. Combine content (concatenate descriptions)
    4. Merge features (preserve all modality-specific data)
    5. Set confidence (min of all inputs)
    """

    # Step 1: Align embeddings
    aligned = self.align_embeddings(inputs)

    # Step 2: Fuse
    if strategy == "attention":
        fused_emb = self._attention_fusion(aligned)
    elif strategy == "concat":
        fused_emb = np.concatenate([inp.embedding for inp in aligned])
    elif strategy == "average":
        fused_emb = np.mean([inp.embedding for inp in aligned], axis=0)
    elif strategy == "max":
        fused_emb = np.max([inp.embedding for inp in aligned], axis=0)

    # Step 3-5: Combine metadata
    return ProcessedInput(
        modality=ModalityType.MULTIMODAL,
        content="\n\n".join([inp.content for inp in inputs]),
        embedding=fused_emb,
        confidence=min([inp.confidence for inp in inputs]),
        features={
            'components': [inp.to_dict() for inp in inputs],
            'fusion_strategy': strategy,
            'component_count': len(inputs)
        }
    )
```

### Stage 5: Shard Creation

**File**: `hololoom/spinningWheel/multimodal_spinner.py` (`MultiModalSpinner._create_shard()`)

**Entity Extraction Logic**:

```python
def extract_entities(features: Dict) -> List[str]:
    """Extract entities from modality-specific features."""
    entities = []

    # TEXT modality
    if 'entities' in features:
        # From TextFeatures.entities (spaCy NER)
        entities.extend([e['text'] for e in features['entities']])

    # IMAGE modality
    if 'objects' in features:
        # From ImageFeatures.objects (object detection)
        entities.extend([obj['label'] for obj in features['objects']])

    # AUDIO modality
    if 'transcript' in features:
        # Extract entities from transcript (run through NER)
        transcript_entities = extract_from_text(features['transcript'])
        entities.extend(transcript_entities)

    # STRUCTURED modality
    if 'schema' in features:
        # Column names are entities
        entities.extend(features['schema'].keys())

    return list(set(entities[:20]))  # Deduplicate, limit to 20
```

**Motif Extraction Logic**:

```python
def extract_motifs(features: Dict) -> List[str]:
    """Extract motifs (topics, themes) from features."""
    motifs = []

    # TEXT modality
    if 'topics' in features:
        motifs.extend(features['topics'])

    # IMAGE modality
    if 'scene' in features:
        motifs.append(f"scene_{features['scene']}")

    # AUDIO modality
    if 'emotion' in features:
        motifs.append(f"emotion_{features['emotion']}")

    # STRUCTURED modality
    if 'schema' in features:
        # Data types as motifs
        motifs.extend([f"type_{dtype}" for dtype in set(features['schema'].values())])

    return list(set(motifs[:10]))  # Deduplicate, limit to 10
```

---

## Modality Processing

### TEXT Processing

**Input Types**:
- Plain string
- File path (`.txt`, `.md`, `.rst`, `.log`)
- URL (webpage, article)

**Pipeline**:

```
Text Input
    ↓
[Preprocessing]
    - Strip whitespace
    - Normalize unicode
    - Remove control characters
    ↓
[Embedding]
    - SimpleEmbedder (TF-IDF) OR
    - SentenceTransformer (384D)
    ↓
[NER] (Optional - spaCy)
    - Extract entities (PERSON, ORG, GPE, etc.)
    - Label with confidence
    ↓
[Topic Extraction]
    - TF-IDF top keywords
    - Noun phrases
    ↓
[Sentiment] (Optional - TextBlob)
    - Polarity: -1.0 (negative) to +1.0 (positive)
    - Subjectivity: 0.0 (objective) to 1.0 (subjective)
    ↓
ProcessedInput(
    modality=TEXT,
    content=text,
    embedding=384D vector,
    features=TextFeatures(...)
)
```

**Example**:

```python
await text_processor.process(
    "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem."
)

# Output:
ProcessedInput(
    modality=ModalityType.TEXT,
    content="Thompson Sampling is a Bayesian approach...",
    embedding=np.array([0.12, -0.34, ...]),
    confidence=1.0,
    features={
        'entities': [
            {'text': 'Thompson Sampling', 'label': 'ALGORITHM'},
            {'text': 'Bayesian', 'label': 'METHOD'}
        ],
        'topics': ['bandit_problem', 'bayesian_statistics'],
        'keyphrases': ['multi-armed bandit', 'Thompson Sampling'],
        'sentiment': {'polarity': 0.0, 'subjectivity': 0.2},
        'language': 'en'
    }
)
```

### IMAGE Processing

**Input Types**:
- File path (`.jpg`, `.png`, `.gif`, `.bmp`)
- Bytes (raw image data)
- URL (image URL)

**Pipeline**:

```
Image Input
    ↓
[Image Loading]
    - PIL/OpenCV
    - Resize to standard dimensions
    - Convert to RGB
    ↓
[Visual Embedding] (Optional - CLIP)
    - 512D visual-semantic space
    - Aligned with text embeddings
    ↓
[Object Detection] (Optional - YOLOv5)
    - Detect objects with bounding boxes
    - Confidence scores
    ↓
[Scene Classification]
    - Indoor/outdoor
    - Urban/nature
    ↓
[OCR] (Optional - Tesseract)
    - Extract text from image
    - Diagrams, slides, screenshots
    ↓
[Color Analysis]
    - Dominant colors (k-means)
    - Hex color codes
    ↓
ProcessedInput(
    modality=IMAGE,
    content="An image showing...",
    embedding=384D vector,
    features=ImageFeatures(...)
)
```

**Example**:

```python
await image_processor.process("/path/to/neural_network_diagram.png")

# Output:
ProcessedInput(
    modality=ModalityType.IMAGE,
    content="Diagram of neural network architecture with input, hidden, and output layers",
    embedding=np.array([0.45, 0.23, ...]),
    confidence=0.92,
    features={
        'objects': [
            {'label': 'diagram', 'confidence': 0.95, 'bbox': [10, 20, 300, 400]},
            {'label': 'flowchart', 'confidence': 0.78}
        ],
        'scene': 'diagram',
        'caption': 'Neural network architecture diagram',
        'ocr_text': 'Input Layer\n128 units\nHidden Layer\n64 units\nOutput Layer',
        'colors': ['#2E86AB', '#A23B72', '#F18F01'],
        'dimensions': (800, 600)
    }
)
```

### AUDIO Processing

**Input Types**:
- File path (`.wav`, `.mp3`, `.ogg`, `.flac`)
- Bytes (raw audio data)
- URL (audio file URL)

**Pipeline**:

```
Audio Input
    ↓
[Audio Loading]
    - librosa/pydub
    - Resample to 16kHz
    - Convert to mono
    ↓
[Speech-to-Text] (Optional - Whisper)
    - Transcribe audio
    - Detect language
    - Timestamp alignment
    ↓
[Speaker Diarization]
    - Detect number of speakers
    - Segment by speaker
    ↓
[Emotion Detection]
    - Acoustic features (MFCC, pitch)
    - Classify: happy, sad, neutral, angry
    ↓
[Audio Embedding]
    - Acoustic feature vector
    - 384D representation
    ↓
ProcessedInput(
    modality=AUDIO,
    content=transcript,
    embedding=384D vector,
    features=AudioFeatures(...)
)
```

**Example**:

```python
await audio_processor.process("/path/to/lecture.mp3")

# Output:
ProcessedInput(
    modality=ModalityType.AUDIO,
    content="Today we'll discuss Thompson Sampling, which is a powerful technique...",
    embedding=np.array([0.67, -0.12, ...]),
    confidence=0.88,
    features={
        'transcript': "Today we'll discuss Thompson Sampling...",
        'language': 'en',
        'speaker_count': 1,
        'emotion': 'neutral',
        'acoustic': {
            'mfcc_mean': [12.3, -8.5, ...],
            'pitch_mean': 120.5,
            'energy_mean': 0.42
        },
        'duration': 124.5,
        'sample_rate': 16000
    }
)
```

### STRUCTURED Data Processing

**Input Types**:
- Dict (JSON object)
- File path (`.json`, `.csv`, `.tsv`, `.yaml`)
- DataFrame (pandas)

**Pipeline**:

```
Structured Input
    ↓
[Data Loading]
    - pandas.read_csv()
    - json.load()
    - yaml.safe_load()
    ↓
[Schema Extraction]
    - Column names
    - Data types (int, float, str, datetime)
    - Nullable columns
    ↓
[Statistical Summary]
    - Row/column counts
    - Mean, std, min, max (numeric)
    - Value distributions (categorical)
    ↓
[Relationship Detection]
    - Foreign key candidates
    - Correlated columns
    ↓
[Text Conversion]
    - Generate human-readable summary
    - "A dataset with 1000 rows and 5 columns..."
    ↓
[Embedding]
    - Schema-based embedding
    - Column name + type embeddings
    ↓
ProcessedInput(
    modality=STRUCTURED,
    content=summary_text,
    embedding=384D vector,
    features=StructuredFeatures(...)
)
```

**Example**:

```python
await structured_processor.process("/path/to/data.csv")

# Input CSV:
# name,age,score
# Alice,25,95
# Bob,30,87

# Output:
ProcessedInput(
    modality=ModalityType.STRUCTURED,
    content="A dataset with 2 rows and 3 columns: name (string), age (integer), score (integer)",
    embedding=np.array([0.34, 0.56, ...]),
    confidence=1.0,
    features={
        'schema': {'name': 'str', 'age': 'int', 'score': 'int'},
        'row_count': 2,
        'column_count': 3,
        'relationships': [],
        'summary_stats': {
            'age': {'mean': 27.5, 'std': 2.5, 'min': 25, 'max': 30},
            'score': {'mean': 91.0, 'std': 4.0, 'min': 87, 'max': 95}
        }
    }
)
```

---

## Fusion Strategies

### 1. Attention-Based Fusion (Recommended)

**Use Case**: Text + Image, Text + Audio, or any combination where modalities have semantic correlation.

**Algorithm**:

```python
def attention_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Cross-attention fusion across modalities.

    Each modality attends to all others:
    - Text can "look at" image features
    - Image can "ground" in text semantics
    - Audio can align with both

    Math:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Where:
        Q = Query (from first modality)
        K = Keys (from all modalities)
        V = Values (embeddings)
    """
    n_modalities = len(embeddings)
    d = embeddings[0].shape[0]  # Embedding dimension

    # Stack embeddings
    E = np.stack(embeddings)  # Shape: (n_modalities, d)

    # Multi-head attention (simplified to single head)
    Q = E[0:1]  # Use first modality as query
    K = E  # All modalities as keys
    V = E  # All modalities as values

    # Compute attention scores
    scores = (Q @ K.T) / np.sqrt(d)  # Shape: (1, n_modalities)
    weights = softmax(scores, axis=-1)  # Normalize

    # Weighted sum of values
    fused = weights @ V  # Shape: (1, d)

    return fused.squeeze(0)
```

**Advantages**:
- Semantic alignment (text → image grounding)
- Automatic weighting (important modalities get higher weight)
- Preserves dimensionality (384D → 384D)

**Disadvantages**:
- Slower than averaging (~30ms vs ~2ms)
- Requires tuning (attention temperature)

### 2. Concatenation Fusion

**Use Case**: Preserve all modality information, downstream model handles fusion.

**Algorithm**:

```python
def concat_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate embeddings along feature dimension.

    Input: [text_384D, image_384D, audio_384D]
    Output: 1152D fused embedding

    Optional: Project back to 384D with learned matrix
    """
    fused = np.concatenate(embeddings, axis=0)

    # Optional: Dimensionality reduction
    if len(fused) > 384:
        fused = project_to_384D(fused)  # PCA or learned projection

    return fused
```

**Advantages**:
- No information loss
- Fast (~2ms)
- Simple

**Disadvantages**:
- High dimensionality (N × 384D)
- Requires projection layer for retrieval

### 3. Average/Max Pooling

**Use Case**: Baseline fusion, weak cross-modal correlation.

**Algorithm**:

```python
def average_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """Element-wise average across embeddings."""
    return np.mean(embeddings, axis=0)

def max_fusion(embeddings: List[np.ndarray]) -> np.ndarray:
    """Element-wise max (keeps strongest features)."""
    return np.max(embeddings, axis=0)
```

**Advantages**:
- Extremely fast (~2ms)
- Preserves dimensionality

**Disadvantages**:
- Treats all modalities equally (no weighting)
- Can dilute strong features

---

## Memory Integration

### Backend Storage

**MemoryShard** → **Memory Backend**:

```python
# Option 1: INMEMORY (NetworkX graph)
memory = await create_memory_backend(Config.bare())
await memory.add_shards(shards)

# Option 2: HYBRID (Neo4j + Qdrant)
memory = await create_memory_backend(Config.fused())
await memory.add_shards(shards)

# Option 3: HYPERSPACE (Advanced gated multipass)
config = Config.fused()
config.memory_backend = MemoryBackend.HYPERSPACE
memory = await create_memory_backend(config)
await memory.add_shards(shards)
```

### Shard → Graph Edges

**Knowledge Graph Integration**:

```python
async def ingest_shard_to_graph(kg: KG, shard: MemoryShard):
    """
    Convert MemoryShard to knowledge graph edges.

    Creates edges:
    - shard → entity (MENTIONS)
    - shard → motif (HAS_TOPIC)
    - entity → entity (CO_OCCURS)
    """
    shard_node = f"shard_{shard.id}"

    # Entity edges
    for entity in shard.entities:
        kg.add_edge(KGEdge(
            src=shard_node,
            dst=entity,
            type="MENTIONS",
            weight=1.0,
            metadata={'text': shard.text[:100]}
        ))

    # Motif edges
    for motif in shard.motifs:
        kg.add_edge(KGEdge(
            src=shard_node,
            dst=motif,
            type="HAS_TOPIC",
            weight=1.0
        ))

    # Co-occurrence edges
    for i, e1 in enumerate(shard.entities):
        for e2 in shard.entities[i+1:]:
            kg.add_edge(KGEdge(
                src=e1,
                dst=e2,
                type="CO_OCCURS",
                weight=0.5
            ))
```

### Vector Similarity Search

**Embedding Storage**:

```python
# Store embeddings for similarity search
embeddings_db = {}  # In practice: Qdrant, FAISS, etc.

for shard in shards:
    embedding = shard.metadata.get('embedding')
    if embedding:
        embeddings_db[shard.id] = np.array(embedding)

# Query by similarity
def query_similar(query_embedding: np.ndarray, k: int = 5) -> List[str]:
    """Find k most similar shards by cosine similarity."""
    scores = {}
    for shard_id, emb in embeddings_db.items():
        scores[shard_id] = cosine_similarity(query_embedding, emb)

    # Return top-k
    return sorted(scores, key=scores.get, reverse=True)[:k]
```

---

## Performance Characteristics

### Latency Breakdown

**Single TEXT Input** (Simple):
```
Detection:      0.5ms
Processing:     8ms (embedding + NER)
Shard Creation: 1ms
─────────────────────
Total:          ~10ms
```

**Single IMAGE Input** (with CLIP + OCR):
```
Detection:      1ms
Processing:     180ms (CLIP: 150ms, OCR: 30ms)
Shard Creation: 2ms
─────────────────────
Total:          ~183ms
```

**Multi-Modal** (Text + Image + Audio with Attention):
```
Detection:      1ms (×3 inputs)
Processing:
    - Text:     10ms
    - Image:    180ms
    - Audio:    350ms
Fusion:         30ms (attention)
Shard Creation: 5ms (×4 shards: 3 components + 1 fused)
─────────────────────
Total:          ~576ms
```

### Throughput

**Batch Processing** (10 text documents):
```
Sequential:     10 × 10ms = 100ms
Concurrent:     max(10 × 10ms) = 10ms (if I/O-bound)

With semaphore (max_concurrent=5):
    Batch 1: 5 docs × 10ms = 10ms
    Batch 2: 5 docs × 10ms = 10ms
    Total: 20ms
```

**Directory Ingestion** (100 mixed files):
```
Files:
    - 80 text files
    - 15 images
    - 5 PDFs

Latency:
    - Text: 80 × 10ms = 800ms
    - Images: 15 × 180ms = 2700ms
    - PDFs: 5 × 50ms = 250ms

Sequential Total: 3750ms

Concurrent (max=10):
    ~500ms (parallelizes I/O and processing)
```

### Memory Usage

**Per Shard**:
```
MemoryShard object:     ~500 bytes
    - id, text, episode, entities, motifs
Embedding (384D float32): 1536 bytes
Features dict:          ~200-1000 bytes
─────────────────────────────────────
Total per shard:        ~2-3KB
```

**1000 Shards**: ~2-3 MB
**100,000 Shards**: ~200-300 MB

---

## Examples by Source Type

### Example 1: Text String

```python
from hololoom.spinningWheel import spin

# Input
text = "Thompson Sampling is a Bayesian approach to multi-armed bandits."

# Process
memory = await spin(text)

# Output (MemoryShard)
MemoryShard(
    id="text_1234567890",
    text="Thompson Sampling is a Bayesian approach...",
    episode="unknown",
    entities=["Thompson Sampling", "Bayesian"],
    motifs=["multi_armed_bandits", "algorithms"],
    metadata={
        'modality_type': 'TEXT',
        'confidence': 1.0,
        'embedding': [0.12, -0.34, ...],  # 384D
        'processing_time_ms': 8.5
    }
)
```

### Example 2: Image File

```python
# Input
image_path = "/path/to/neural_network_diagram.png"

# Process
memory = await spin(image_path)

# Output
MemoryShard(
    id="image_9876543210",
    text="Diagram of neural network architecture with labeled layers",
    episode="/path/to/neural_network_diagram.png",
    entities=["neural_network", "diagram"],
    motifs=["machine_learning", "visualization"],
    metadata={
        'modality_type': 'IMAGE',
        'confidence': 0.92,
        'embedding': [0.45, 0.23, ...],
        'features': {
            'objects': [{'label': 'diagram', 'confidence': 0.95}],
            'ocr_text': 'Input Layer\nHidden Layer\nOutput Layer',
            'colors': ['#2E86AB', '#A23B72']
        }
    }
)
```

### Example 3: Audio File

```python
# Input
audio_path = "/path/to/lecture.mp3"

# Process
memory = await spin(audio_path)

# Output
MemoryShard(
    id="audio_5555555555",
    text="Today we'll discuss Thompson Sampling...",  # Transcript
    episode="/path/to/lecture.mp3",
    entities=["Thompson Sampling", "reinforcement learning"],
    motifs=["lecture", "algorithms"],
    metadata={
        'modality_type': 'AUDIO',
        'confidence': 0.88,
        'embedding': [0.67, -0.12, ...],
        'features': {
            'transcript': "Today we'll discuss...",
            'language': 'en',
            'duration': 124.5
        }
    }
)
```

### Example 4: Multi-Modal (Text + Image)

```python
# Input
inputs = [
    "This is a diagram of a neural network",
    "/path/to/neural_network_diagram.png"
]

# Process
memory = await spin(inputs)

# Output (4 shards: text component, image component, fused, metadata)
[
    MemoryShard(
        id="multimodal_component_0_...",
        text="This is a diagram of a neural network",
        episode="multimodal_input",
        entities=["neural network", "diagram"],
        motifs=["machine_learning"],
        metadata={
            'component_index': 0,
            'parent_modality': 'MULTIMODAL',
            'is_component': True,
            'modality': 'text'
        }
    ),
    MemoryShard(
        id="multimodal_component_1_...",
        text="Diagram of neural network architecture...",
        episode="multimodal_input",
        entities=["neural_network", "diagram"],
        motifs=["visualization"],
        metadata={
            'component_index': 1,
            'parent_modality': 'MULTIMODAL',
            'is_component': True,
            'modality': 'image'
        }
    ),
    MemoryShard(
        id="multimodal_fused_...",
        text="This is a diagram of a neural network\n\nDiagram of neural network architecture...",
        episode="multimodal_input",
        entities=["neural network", "diagram"],
        motifs=["machine_learning", "visualization"],
        metadata={
            'modality_type': 'MULTIMODAL',
            'component_count': 2,
            'is_fused': True,
            'confidence': 0.92,  # min(1.0, 0.92)
            'fusion_strategy': 'attention'
        }
    )
]
```

### Example 5: Batch Processing

```python
from hololoom.spinningWheel import spin_batch

# Input
sources = [
    "Text about Thompson Sampling",
    "/path/to/image.png",
    "/path/to/audio.mp3",
    "https://example.com/article.html"
]

# Process (concurrent, max 5 at a time)
memory = await spin_batch(sources, max_concurrent=5)

# Output
# - 4 shards (one per source)
# - All ingested into single memory backend
```

### Example 6: Directory Ingestion

```python
from hololoom.spinningWheel import spin_directory

# Input
directory = "/path/to/research_papers"

# Process (recursive, only PDFs)
memory = await spin_directory(
    directory,
    pattern="*.pdf",
    recursive=True
)

# Output
# - N shards (one per PDF)
# - Episodic structure: file paths
# - Entities: paper authors, citations
# - Motifs: research topics
```

### Example 7: URL Crawling

```python
from hololoom.spinningWheel import spin_url

# Input
url = "https://example.com/article.html"

# Process (with link following)
memory = await spin_url(
    url,
    follow_links=True,
    max_depth=2
)

# Output
# - Multiple shards (one per page crawled)
# - Cross-page entity linking
# - Site structure preserved
```

---

## Future Extensions

### 1. Video Processing

**Pipeline**:
```
Video File
    ↓
[Keyframe Extraction] (every 1 second)
    ↓
[Image Processing] (per frame)
    - Object detection
    - OCR on slides
    ↓
[Audio Extraction]
    - Speech-to-text
    - Speaker diarization
    ↓
[Temporal Alignment]
    - Match visuals to spoken words
    - "When the speaker says 'neural network', show diagram"
    ↓
[Shard Creation] (per scene)
    - Scene boundaries detected
    - Combined visual + audio shards
```

### 2. Real-Time Streaming

**Use Case**: Live audio transcription, webcam analysis

```python
async def spin_stream(source: AsyncIterator) -> AsyncIterator[MemoryShard]:
    """Process streaming input incrementally."""
    async for chunk in source:
        shard = await process_chunk(chunk)
        yield shard

        # Ingest immediately (don't wait for full stream)
        await memory.add_shard(shard)
```

### 3. Adaptive Quality

**Trade-off**: Speed vs. Quality

```python
class AdaptiveSpinner:
    async def spin(self, source: Any, quality: str = "auto") -> List[MemoryShard]:
        """
        Auto-adjust processing based on input size.

        Quality levels:
        - fast: No NER, no OCR, simple embeddings (5-10ms)
        - balanced: Basic NER, embeddings (20-50ms)
        - high: Full NER, OCR, CLIP, Whisper (200-500ms)
        - auto: Choose based on input size
        """
        if quality == "auto":
            quality = self._detect_quality(source)

        if quality == "fast":
            return await self._fast_process(source)
        elif quality == "balanced":
            return await self._balanced_process(source)
        else:
            return await self._high_quality_process(source)
```

---

## Pipeline Diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SPINNING WHEEL PIPELINE                            │
│                   "Anything → Queryable Memory"                          │
└──────────────────────────────────────────────────────────────────────────┘

                                  INPUT
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                  Text            Image           Audio
                 String           File            File
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   INPUT DETECTION     │
                        │   (InputRouter)       │
                        │ - Extension analysis  │
                        │ - Magic numbers       │
                        │ - Content inspection  │
                        └───────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │    TEXT      │ │   IMAGE     │ │   AUDIO     │
            │  PROCESSOR   │ │  PROCESSOR  │ │  PROCESSOR  │
            │              │ │             │ │             │
            │ - Embedding  │ │ - CLIP      │ │ - Whisper   │
            │ - NER        │ │ - OCR       │ │ - STT       │
            │ - Sentiment  │ │ - Objects   │ │ - Emotion   │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │ PROCESSED INPUT       │
                        │                       │
                        │ - modality: Enum      │
                        │ - content: str        │
                        │ - embedding: 384D     │
                        │ - features: Dict      │
                        │ - confidence: float   │
                        └───────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                   Single Input          Multiple Inputs
                        │                       │
                        │                       ▼
                        │           ┌───────────────────────┐
                        │           │  MULTI-MODAL FUSION   │
                        │           │                       │
                        │           │ Strategy:             │
                        │           │ - Attention           │
                        │           │ - Concatenation       │
                        │           │ - Average/Max         │
                        │           └───────────────────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  SHARD CREATION       │
                        │                       │
                        │ - Extract entities    │
                        │ - Extract motifs      │
                        │ - Preserve metadata   │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   MEMORY SHARD(S)     │
                        │                       │
                        │ - id, text, episode   │
                        │ - entities, motifs    │
                        │ - metadata (features) │
                        └───────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  INMEMORY   │ │   HYBRID    │ │ HYPERSPACE  │
            │ (NetworkX)  │ │ (Neo4j +    │ │ (Advanced)  │
            │             │ │  Qdrant)    │ │             │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   QUERYABLE MEMORY    │
                        │                       │
                        │ - Semantic search     │
                        │ - Graph traversal     │
                        │ - Multi-modal retrieval│
                        └───────────────────────┘
```

---

## Summary

The SpinningWheel pipeline transforms **any input** into **queryable memory** through:

1. **Universal Detection**: Auto-detect input type (text, image, audio, structured)
2. **Modality Processing**: Extract features with specialized processors
3. **Cross-Modal Fusion**: Combine multiple inputs intelligently
4. **Shard Creation**: Convert to unified `MemoryShard` representation
5. **Memory Integration**: Store in knowledge graph + vector database

**Key Principles**:
- **Zero Configuration**: `spin(anything)` just works
- **Graceful Degradation**: Falls back when optional dependencies unavailable
- **Protocol-Based**: All processors implement `InputProcessorProtocol`
- **Performance**: 10ms (text) to 500ms (multi-modal with attention)
- **Scalability**: Batch processing, concurrent execution, streaming support

**Result**: Everything you've ever experienced becomes queryable through a unified memory interface.

---

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Complete HoloLoom documentation
- [Input Protocol](../input/protocol.py) - ProcessedInput types
- [MultiModalSpinner](multimodal_spinner.py) - Core spinner implementation
- [ChatHistorySpinner](chat_history.py) - Conversation ingestion
- [Auto Functions](auto.py) - Convenience API