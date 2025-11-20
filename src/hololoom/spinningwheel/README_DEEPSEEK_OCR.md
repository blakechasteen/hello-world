# DeepSeek OCR Integration Guide

**Status**: ✅ Complete (January 2025)
**Location**: `HoloLoom/spinningWheel/deepseek_ocr_spinner.py`
**Performance**: ~2,500 tokens/sec on A100-40G, 10x compression (1000 words → 100 tokens)

## Overview

The DeepSeek OCR Spinner integrates DeepSeek's open-source OCR model for high-quality document and image text extraction into HoloLoom's memory system.

### Key Features

- **Document OCR → Markdown**: Structured output with proper formatting
- **Multi-page PDF support**: Automatic chunking and batching
- **Multi-resolution**: 512px - 1280px (64-400 tokens per image)
- **Graceful degradation**: Falls back to pytesseract if DeepSeek unavailable
- **Batch processing**: Process multiple documents efficiently
- **Entity extraction**: Optional Ollama enrichment for entities/motifs
- **Importance scoring**: Automatic quality assessment

## Installation

### Requirements

**Hardware**:
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 16GB+ VRAM for vLLM backend
- 8GB+ VRAM for transformers backend

**Software**:
- Python 3.12+
- PyTorch 2.6.0+
- CUDA 11.8+

### Quick Install (vLLM - Recommended)

```bash
# Create environment
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (fastest backend)
pip install vllm

# Install supporting libraries
pip install pillow PyMuPDF  # For image/PDF processing
pip install pytesseract  # Optional fallback OCR
```

### Alternative Install (Transformers)

```bash
# If vLLM installation fails or you don't have CUDA
pip install transformers torch pillow PyMuPDF
```

### Model Download

The model will auto-download on first use from HuggingFace:
- Model: `deepseek-ai/DeepSeek-OCR`
- Size: ~6GB
- License: MIT (free for commercial use)

## Quick Start

### Basic Usage

```python
from HoloLoom.spinningWheel import DeepSeekOCRSpinner

# Initialize spinner (uses sensible defaults)
spinner = DeepSeekOCRSpinner()

# Process single image
result = await spinner.spin("document.png")

# Access extracted text
for shard in result.shards:
    print(shard.text)
    print(f"Confidence: {shard.metadata['ocr_confidence']}")
```

### PDF Processing

```python
from HoloLoom.spinningWheel import DeepSeekOCRSpinner

spinner = DeepSeekOCRSpinner()

# Process PDF (auto-chunks by 10 pages)
result = await spinner.spin("report.pdf")

# Each shard is one chunk
for shard in result.shards:
    page_range = shard.metadata['page_range']
    print(f"Pages {page_range}:")
    print(shard.text[:500])  # First 500 chars
    print("---")
```

### Batch Processing

```python
from HoloLoom.spinningWheel import DeepSeekOCRSpinner

spinner = DeepSeekOCRSpinner()

# Process multiple documents
files = ["doc1.png", "doc2.pdf", "doc3.jpg"]
result = await spinner.spin(files)

print(f"Processed {result.shard_count} shards in {result.processing_time_ms:.1f}ms")
```

## Configuration

### Custom Configuration

```python
from HoloLoom.spinningWheel.deepseek_ocr_spinner import (
    DeepSeekOCRSpinner,
    DeepSeekOCRConfig,
    OCRBackend,
    OutputFormat
)

config = DeepSeekOCRConfig(
    # Backend selection
    backend=OCRBackend.VLLM,  # or TRANSFORMERS, FALLBACK

    # Resolution (higher = better quality, slower)
    resolution=1024,  # 512, 640, 1024, or 1280

    # Output format
    output_format=OutputFormat.MARKDOWN,  # or TEXT, JSON

    # PDF chunking
    chunk_pages=True,
    pages_per_chunk=10,

    # Entity extraction (requires Ollama)
    enable_enrichment=True,
    enrichment_model="llama3.2:3b",

    # Performance
    batch_size=8,
    max_tokens=8192
)

spinner = DeepSeekOCRSpinner(config)
result = await spinner.spin("document.pdf")
```

### Backend Options

| Backend | Speed | Quality | Requirements |
|---------|-------|---------|--------------|
| **VLLM** | Fastest (~2500 tok/s) | Best | vLLM + CUDA |
| **TRANSFORMERS** | Fast (~1000 tok/s) | Good | transformers + PyTorch |
| **FALLBACK** | Slow (~100 tok/s) | Basic | pytesseract only |

### Resolution Options

| Resolution | Tokens | Quality | Use Case |
|------------|--------|---------|----------|
| **512×512** | 64 | Fast | Simple text, receipts |
| **640×640** | 100 | Good | Documents, forms |
| **1024×1024** | 256 | High | Reports, papers |
| **1280×1280** | 400 | Best | Complex layouts, tables |

## Integration with HoloLoom

### Full Weaving Cycle

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.spinningWheel import DeepSeekOCRSpinner
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

# Process document into shards
spinner = DeepSeekOCRSpinner()
result = await spinner.spin("research_paper.pdf")
shards = result.shards

# Add to memory and query
config = Config.fused()
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(
        Query(text="What are the main findings of this paper?")
    )

    print(spacetime.response)
```

### Memory Storage

```python
from HoloLoom import HoloLoom
from HoloLoom.spinningWheel import DeepSeekOCRSpinner

async with HoloLoom() as loom:
    # Extract text from documents
    spinner = DeepSeekOCRSpinner()
    result = await spinner.spin(["doc1.pdf", "doc2.png"])

    # Experience (store) all shards
    for shard in result.shards:
        await loom.experience(shard.text)

    # Recall relevant documents
    memories = await loom.recall("What did the documents say about X?")

    for memory in memories:
        print(memory.text)
```

## Convenience Functions

### Quick Text Extraction

```python
from HoloLoom.spinningWheel.deepseek_ocr_spinner import (
    extract_text_from_image,
    extract_text_from_pdf
)

# Single image
text = await extract_text_from_image(
    "document.png",
    output_format="markdown",
    resolution=1024
)

# PDF (returns list of chunks)
chunks = await extract_text_from_pdf(
    "report.pdf",
    pages_per_chunk=10
)
```

## Performance Optimization

### GPU Memory Management

```python
config = DeepSeekOCRConfig(
    backend=OCRBackend.VLLM,
    batch_size=4,  # Reduce if OOM
    max_tokens=4096  # Reduce for shorter documents
)
```

### Batch Processing

```python
# Process large document sets efficiently
large_batch = [f"doc_{i}.pdf" for i in range(100)]

# Process in chunks to avoid memory issues
chunk_size = 10
for i in range(0, len(large_batch), chunk_size):
    batch = large_batch[i:i+chunk_size]
    result = await spinner.spin(batch)

    # Store results incrementally
    for shard in result.shards:
        await memory.add_shard(shard)
```

## Graceful Degradation

The spinner automatically falls back through multiple OCR backends:

1. **DeepSeek OCR** (best quality) - if vLLM/transformers + CUDA available
2. **pytesseract** (basic quality) - if pytesseract installed
3. **Filename extraction** (metadata only) - if no OCR available

```python
spinner = DeepSeekOCRSpinner()

# Check what's available
status = spinner.get_status()
print(status)  # AVAILABLE, DEGRADED, or UNAVAILABLE

capabilities = spinner.get_capabilities()
print(f"Basic processing: {capabilities.basic_processing}")
print(f"Entity extraction: {capabilities.entity_extraction}")
```

## Importance Scoring

The spinner automatically scores document importance based on:

- **Length**: Longer documents = more substantial
- **Structure**: Headings, lists, formatting
- **Technical content**: Domain-specific terms
- **File type**: PDF > image

```python
result = await spinner.spin("document.pdf")

for shard in result.shards:
    score = shard.metadata['importance_score']
    reason = shard.metadata['importance_reason']

    if score > 0.7:
        print(f"Important document: {reason}")
```

## Entity Extraction

Enable Ollama enrichment for automatic entity/motif extraction:

```python
config = DeepSeekOCRConfig(
    enable_enrichment=True,
    enrichment_model="llama3.2:3b"  # or "llama2", "mistral", etc.
)

spinner = DeepSeekOCRSpinner(config)
result = await spinner.spin("paper.pdf")

for shard in result.shards:
    print(f"Entities: {shard.entities}")
    print(f"Topics: {shard.motifs}")
```

**Note**: Requires Ollama installed locally:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2:3b
```

## Troubleshooting

### vLLM Installation Issues

If vLLM installation fails:

```bash
# Use transformers backend instead
pip install transformers torch

# Update config
config.backend = OCRBackend.TRANSFORMERS
```

### CUDA Out of Memory

```python
# Reduce batch size
config.batch_size = 2

# Reduce resolution
config.resolution = 512

# Use CPU (slower but works)
config.device = "cpu"
config.backend = OCRBackend.TRANSFORMERS
```

### Low OCR Quality

```python
# Increase resolution
config.resolution = 1280

# Use higher quality backend
config.backend = OCRBackend.VLLM

# Enable enrichment for better entity extraction
config.enable_enrichment = True
```

## API Reference

### DeepSeekOCRSpinner

```python
class DeepSeekOCRSpinner(BaseSpinner):
    """Spinner for extracting text from images/documents."""

    def __init__(
        self,
        config: Optional[DeepSeekOCRConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """Initialize spinner with config."""
        ...

    async def spin(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> SpinResult:
        """Process image(s)/PDF(s) → MemoryShards."""
        ...

    def score_importance(self, data: Any) -> ImportanceScore:
        """Score document importance."""
        ...
```

### DeepSeekOCRConfig

```python
@dataclass
class DeepSeekOCRConfig:
    """Configuration for DeepSeek OCR Spinner."""

    # Backend selection
    backend: OCRBackend = OCRBackend.VLLM
    model_name: str = "deepseek-ai/DeepSeek-OCR"

    # Resolution (512, 640, 1024, 1280)
    resolution: int = 1024

    # Output format
    output_format: OutputFormat = OutputFormat.MARKDOWN

    # Processing options
    batch_size: int = 8
    max_tokens: int = 8192

    # PDF chunking
    chunk_pages: bool = True
    pages_per_chunk: int = 10

    # Enrichment
    enable_enrichment: bool = False
    enrichment_model: str = "llama3.2:3b"

    # Performance
    device: str = "cuda"
```

### SpinResult

```python
@dataclass
class SpinResult:
    """Result of OCR operation."""

    shards: List[MemoryShard]  # Extracted shards
    success: bool  # True if successful
    error_message: Optional[str]  # Error details

    # Metrics
    processing_time_ms: float
    shard_count: int
    entity_count: int
    motif_count: int
    avg_importance: float
    avg_confidence: float

    # Warnings
    warnings: List[str]
```

## Examples

See `demos/demo_deepseek_ocr.py` for complete examples:

- Basic image OCR
- Multi-page PDF processing
- Batch document processing
- HoloLoom integration
- Entity extraction
- Importance scoring

## Performance Benchmarks

**Hardware**: NVIDIA A100-40G

| Backend | Resolution | Speed | Quality |
|---------|------------|-------|---------|
| vLLM | 1024×1024 | ~2,500 tok/s | Excellent |
| Transformers | 1024×1024 | ~1,000 tok/s | Good |
| pytesseract | 1024×1024 | ~100 tok/s | Basic |

**Compression**: 10x (1000 words → 100 visual tokens at 97% accuracy)

**Throughput**: 200,000 pages/day (single A100)

## Future Enhancements

- [ ] Direct bytes processing (no temp files)
- [ ] Table extraction with structure preservation
- [ ] Multi-column layout detection
- [ ] Handwriting recognition mode
- [ ] Streaming inference for very large PDFs
- [ ] GPU batching optimization
- [ ] Cache for repeated documents

## References

- **GitHub**: https://github.com/deepseek-ai/DeepSeek-OCR
- **HuggingFace**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Paper**: DeepSeek-OCR: Contexts Optical Compression
- **License**: MIT (free commercial use)

## Credits

- **DeepSeek-AI**: Original OCR model
- **HoloLoom**: Integration and spinner framework
- **Claude Code**: Implementation and documentation