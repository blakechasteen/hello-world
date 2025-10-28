# YouTube Spinner

YouTube input adapter for HoloLoom that extracts video transcripts and converts them into MemoryShards.

## Overview

The YouTube Spinner is part of HoloLoom's SpinningWheel module - a collection of lightweight input adapters that normalize raw data into MemoryShards for processing by the Orchestrator.

## Installation

```bash
pip install youtube-transcript-api
```

## Basic Usage

### Quick Transcription

```python
import asyncio
from HoloLoom.spinningWheel import transcribe_youtube

async def main():
    # Transcribe a video into MemoryShards
    shards = await transcribe_youtube('https://www.youtube.com/watch?v=VIDEO_ID')

    for shard in shards:
        print(f"Text: {shard.text[:100]}...")
        print(f"Entities: {shard.entities}")
        print(f"Metadata: {shard.metadata}")

asyncio.run(main())
```

### Using YouTubeSpinner Class

```python
import asyncio
from HoloLoom.spinningWheel import YouTubeSpinner, YouTubeSpinnerConfig

async def main():
    # Configure spinner
    config = YouTubeSpinnerConfig(
        chunk_duration=60.0,      # Split into 60-second chunks
        include_timestamps=True,   # Include timing info in metadata
        extract_entities=True,     # Extract basic entities
        enable_enrichment=False    # Optional: Ollama enrichment
    )

    spinner = YouTubeSpinner(config)

    # Spin video into shards
    shards = await spinner.spin({
        'url': 'https://www.youtube.com/watch?v=VIDEO_ID',
        'languages': ['en'],       # Preferred languages
        'episode': 'video_001',    # Episode identifier
        'metadata': {              # Additional metadata
            'source': 'tutorial',
            'date': '2025-10-23'
        }
    })

    print(f"Generated {len(shards)} shards")

asyncio.run(main())
```

## Configuration Options

### YouTubeSpinnerConfig

- **chunk_duration** (Optional[float]): Split transcript into time-based chunks.
  - `None` (default): Single shard for entire video
  - `60.0`: Create new shard every 60 seconds

- **include_timestamps** (bool): Include timestamp data in shard metadata
  - Default: `True`

- **extract_entities** (bool): Extract basic entities using simple heuristics
  - Default: `True`

- **enable_enrichment** (bool): Enable Ollama/Neo4j enrichment
  - Default: `False`
  - Requires Ollama running locally

## Input Format

The `spin()` method accepts a dictionary with:

```python
{
    'url': str,              # Required: YouTube URL or video ID
    'languages': List[str],  # Optional: Preferred languages ['en', 'es']
    'episode': str,          # Optional: Episode identifier
    'metadata': dict         # Optional: Additional metadata to include
}
```

## Output Format

Returns `List[MemoryShard]` with:

```python
MemoryShard(
    id='episode_chunk_000',           # Unique shard identifier
    text='transcript text...',        # Full text of this segment
    episode='youtube_VIDEO_ID',       # Episode/video identifier
    entities=['Entity1', 'Entity2'],  # Extracted entities
    motifs=[],                        # Motifs (filled by orchestrator)
    metadata={                        # Rich metadata
        'source': 'youtube',
        'video_id': 'VIDEO_ID',
        'language': 'en',
        'is_generated': True,
        'duration': 123.45,
        'chunk_start': 0.0,
        'chunk_end': 60.0,
        'url': 'https://www.youtube.com/watch?v=VIDEO_ID&t=0',
        'timestamps': [...]           # Detailed timing info
    }
)
```

## Examples

### Single Shard for Entire Video

```python
# Default behavior - one shard per video
shards = await transcribe_youtube('VIDEO_ID')
assert len(shards) == 1
```

### Chunked Transcription

```python
# Split into 2-minute chunks
shards = await transcribe_youtube('VIDEO_ID', chunk_duration=120.0)
# Long video will be split into multiple shards
```

### Multi-Language Support

```python
# Try Spanish first, fall back to English
shards = await transcribe_youtube('VIDEO_ID', languages=['es', 'en'])
print(f"Got transcript in: {shards[0].metadata['language']}")
```

### With Enrichment

```python
config = YouTubeSpinnerConfig(enable_enrichment=True)
spinner = YouTubeSpinner(config)

shards = await spinner.spin({'url': 'VIDEO_ID'})

# Enriched entities and motifs from Ollama
print(shards[0].entities)  # Enhanced entity list
print(shards[0].motifs)    # Detected motifs
```

## Integration with HoloLoom

### Adding to Orchestrator Memory

```python
from HoloLoom.spinningWheel import transcribe_youtube
from HoloLoom.orchestrator import Orchestrator
from HoloLoom.config import Config

async def ingest_video():
    # Step 1: Transcribe video
    shards = await transcribe_youtube('VIDEO_ID', chunk_duration=60.0)

    # Step 2: Initialize orchestrator
    config = Config.fast()
    orchestrator = Orchestrator(config)

    # Step 3: Add shards to memory
    for shard in shards:
        orchestrator.memory.add_shard(shard)

    # Step 4: Query the video content
    result = await orchestrator.process("What topics are discussed?")
    print(result.text)
```

### Factory Function

```python
from HoloLoom.spinningWheel import create_spinner

spinner = create_spinner('youtube')
shards = await spinner.spin({'url': 'VIDEO_ID'})
```

## Features

- **Multiple URL formats**: Full URL, youtu.be, video ID
- **Language selection**: Prefer specific languages with fallback
- **Time-based chunking**: Split long videos into manageable segments
- **Timestamp preservation**: Keep timing information for each segment
- **Entity extraction**: Basic entity detection with optional enrichment
- **Deep linking**: Metadata includes timestamped URLs for each chunk
- **Graceful degradation**: Works without optional dependencies

## Architecture

Follows HoloLoom's "warp thread" philosophy:

1. **Thin Adapter**: Normalizes YouTube data → MemoryShards
2. **Optional Enrichment**: Pre-processing with Ollama/Neo4j
3. **Orchestrator Processing**: Heavy lifting (embeddings, spectral features)
4. **Standardized Output**: Consistent MemoryShard format

The spinner is independent - it only depends on:
- `HoloLoom.Documentation.types.MemoryShard`
- `HoloLoom.spinningWheel.base.BaseSpinner`

## Troubleshooting

### 403 Forbidden Errors

YouTube may block requests from certain IPs:
- Try from a different network
- Use a VPN
- Reduce request frequency

### No Transcripts Available

- Video may have transcripts disabled
- Video may be age-restricted or private
- Try a different video to verify installation

### Enrichment Not Working

Ensure Ollama is running:
```bash
ollama serve
ollama pull llama3.2:3b
```

## Complete Example

See `HoloLoom/spinningWheel/examples/youtube_example.py` for comprehensive examples including:
- Basic transcription
- Chunked processing
- Multi-language support
- Enrichment
- Orchestrator integration

Run examples:
```bash
PYTHONPATH=. python HoloLoom/spinningWheel/examples/youtube_example.py
```

## API Reference

### transcribe_youtube()

```python
async def transcribe_youtube(
    url: str,
    languages: List[str] = None,
    chunk_duration: Optional[float] = None,
    enable_enrichment: bool = False
) -> List[MemoryShard]
```

Convenience function for quick transcription.

### YouTubeSpinner

```python
class YouTubeSpinner(BaseSpinner):
    def __init__(self, config: YouTubeSpinnerConfig = None)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]
```

Main spinner class following BaseSpinner protocol.

### YouTubeTranscriptExtractor

```python
class YouTubeTranscriptExtractor:
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]

    @staticmethod
    def get_transcript(video_id: str, languages: List[str] = None) -> Dict[str, Any]
```

Low-level extraction utility.

## Notes

- Respects YouTube's transcript availability
- Does not bypass restrictions
- For educational and accessibility purposes
- Follows HoloLoom's warp thread architecture
