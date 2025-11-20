# Bee Inspection Audio Ingestion System

Complete pipeline for processing bee inspection audio recordings into structured data with vector embeddings for semantic search.

## Features

✅ **Audio Transcription** - Whisper AI for accurate speech-to-text  
✅ **Structured Schema** - Follows SKILL.md bee inspection ontology  
✅ **Vector Embeddings** - HoloLoom's Nomic v1.5 embeddings (768d)  
✅ **Semantic Search** - Query inspections with natural language  
✅ **Graph Database** - Neo4j storage with relationships  
✅ **In-Memory Fallback** - Works without Neo4j for testing  

## Architecture

```
Audio File (.wav, .mp3)
    ↓
[Whisper Transcription]
    ↓
Raw Transcript + Timestamps
    ↓
[Schema Extraction]
    ↓
Structured Entities:
  - InspectionEvent
  - Hive entities
  - PopulationStock
  - BehaviorStock
  - Treatment
  - Observation
    ↓
[HoloLoom Embeddings]
    ↓
Vector Embeddings (768d)
    ↓
[Storage Layer]
    ↓
Neo4j (graph) + Vector Store (embeddings)
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/blakechasteen/mythrL/hello-world

# Install core dependencies
pip install torch numpy sentence-transformers

# Install audio processing
pip install openai-whisper

# Install Neo4j driver (optional)
pip install neo4j

# Or install all at once
pip install -r bee_inspection_requirements.txt
```

### 2. Run Demo

```bash
# Demo with sample transcript (no audio file needed)
python bee_inspection_demo.py
```

This will:
- Process a sample bee inspection transcript
- Generate embeddings using HoloLoom
- Store in memory (no database needed)
- Demonstrate semantic search

### 3. Process Real Audio

```bash
# Process audio file
python bee_inspection_ingest.py path/to/inspection.wav --date 2024-10-18

# With inspector name
python bee_inspection_ingest.py inspection.wav --date 2024-10-18 --inspector "John"

# Use different Whisper model (more accurate but slower)
python bee_inspection_ingest.py inspection.wav --date 2024-10-18 --whisper-model medium
```

### 4. Search Inspections

```bash
# Semantic search
python bee_inspection_ingest.py --search "aggressive behavior during inspection"

# Search for specific topics
python bee_inspection_ingest.py --search "thymol treatment dosage"
python bee_inspection_ingest.py --search "weak population frames of bees"
```

## Structured Schema

Follows the stocks/flows model from `SKILL.md`:

### Node Types

**Events:**
- `InspectionEvent` - The inspection itself (date, weather, duration, purpose)

**Stocks (State Measurements):**
- `Hive` - Physical hive entity (configuration, genetics, status)
- `PopulationStock` - Bee population (frames, strength, activity)
- `BehaviorStock` - Temperament (aggression, calmness, smoke response)
- `SmokerStock` - Equipment state (fuel, condition)
- `EquipmentStock` - Tools (hive tool, veil, etc.)

**Flows (Changes):**
- `Treatment` - Treatments applied (thymol, oxalic acid, dosage)
- `FeedingFlow` - Supplemental feeding (syrup, fondant)

**Processes:**
- `SmokingProcess` - Smoking technique (amount, timing, effectiveness)

**Knowledge:**
- `Observation` - Discoveries and insights
- `TechniqueKnowledge` - Procedural learning
- `Task` - Action items
- `Suggestion` - Recommendations
- `ResearchQuestion` - Questions to investigate

### Example Data

```python
InspectionEvent(
    eventId="inspect-2024-10-18-001",
    date="2024-10-18",
    weatherTemp=68.0,
    weatherCondition="clear",
    withVeil=True,
    primaryPurpose="routine_check"
)

PopulationStock(
    stockId="pop-hive-jodi-001-2024-10-18",
    framesOfBees=8.0,
    populationStrength="strong",
    entranceActivity="moderate",
    activityLevel="good"
)

Treatment(
    treatmentId="treat-hive-jodi-001-2024-10-18",
    treatmentType="thymol",
    treatmentRound=2,
    dosage=2.0,
    dosageUnit="cards",
    applicationMethod="card",
    placement="top_box"
)
```

## Vector Embeddings

### What Gets Embedded?

1. **Full Transcript** - Complete inspection recording
2. **Inspection Event** - Summary embedding
3. **Individual Entities** - Each hive, treatment, observation

### Embedding Model

Uses HoloLoom's **Nomic Embed v1.5**:
- Dimensions: 768
- Context length: 8192 tokens
- MTEB score: ~62 (high quality)
- Released: 2024

### Semantic Search Examples

```python
# Initialize pipeline
pipeline = BeeInspectionPipeline()

# Search for behavior patterns
results = pipeline.search("aggressive bees defensive smoke")

# Search for population issues
results = pipeline.search("weak colony low frames winter prep")

# Search for treatment protocols
results = pipeline.search("thymol dosage application method cards")
```

Returns ranked results with similarity scores (0-1).

## Storage Options

### Option 1: In-Memory (Default)

No setup required - perfect for testing:
```python
pipeline = BeeInspectionPipeline()
```

Data stored in memory, lost on exit.

### Option 2: Neo4j + In-Memory Vectors

Store graph in Neo4j, embeddings in memory:

```bash
# Start Neo4j (Docker)
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Use with pipeline
python bee_inspection_ingest.py audio.wav --date 2024-10-18 \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password
```

### Option 3: Neo4j + Qdrant (Full Production)

Store graph in Neo4j, vectors in Qdrant:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# TODO: Integrate Qdrant storage
# (currently in-memory only)
```

## Integration with HoloLoom

### As Memory Shards

```python
from HoloLoom.documentation.types import MemoryShard
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

# Process inspection
pipeline = BeeInspectionPipeline()
result = await pipeline.process_audio(audio_path, date="2024-10-18")

# Convert to memory shards
shards = [
    MemoryShard(
        text=result["transcript"],
        source=f"bee_inspection_{result['inspection'].eventId}",
        metadata={
            "date": result["inspection"].date,
            "inspector": result["inspection"].inspector,
            "type": "inspection_transcript"
        }
    )
]

# Use with HoloLoom
config = Config.fast()
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    response = await shuttle.weave(Query(text="What was the hive behavior?"))
    print(response.response)
```

### With Recursive Learning

```python
from HoloLoom.recursive import FullLearningEngine

# Enable full learning
async with FullLearningEngine(cfg=config, shards=shards) as engine:
    # Query about inspections
    result = await engine.weave(
        Query(text="Compare population trends across inspections"),
        enable_refinement=True
    )
    
    # System learns patterns and improves over time
    stats = engine.get_learning_statistics()
    print(f"Learning stats: {stats}")
```

## Advanced Usage

### Custom Schema Extraction with LLM

```python
# Edit bee_inspection_ingest.py:
class SchemaExtractor:
    async def _extract_with_llm(self, transcript: str, date: str) -> Dict:
        # Add OpenAI/Claude call here
        # Use SKILL.md schema in system prompt
        
        prompt = f"""
        Extract structured bee inspection data from this transcript.
        Follow the schema from SKILL.md:
        
        {transcript}
        
        Return JSON with:
        - InspectionEvent
        - Hive entities
        - Stocks (population, behavior)
        - Treatments
        - Observations
        """
        
        # Call LLM, parse response
        response = await openai.chat.completions.create(...)
        return parse_response(response)
```

### Batch Processing

```python
import asyncio
from pathlib import Path

async def process_batch(audio_dir: Path):
    pipeline = BeeInspectionPipeline()
    
    for audio_file in audio_dir.glob("*.wav"):
        # Extract date from filename: inspection-2024-10-18.wav
        date = audio_file.stem.split("-", 1)[1]
        
        result = await pipeline.process_audio(audio_file, date)
        print(f"✓ Processed {audio_file.name}")
    
    pipeline.close()

asyncio.run(process_batch(Path("inspections/")))
```

### Export to Cypher

```python
# Generate Neo4j import script
def export_to_cypher(result: Dict, output_path: Path):
    inspection = result["inspection"]
    
    cypher = f"""
    // Create inspection event
    CREATE (i:InspectionEvent {{
        eventId: '{inspection.eventId}',
        date: '{inspection.date}',
        inspector: '{inspection.inspector}'
    }})
    
    // Create hives and relationships
    // ... (see SKILL.md for full schema)
    """
    
    output_path.write_text(cypher)
```

## Performance

### Transcription Speed

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| tiny | ~32x realtime | Good | 39 MB |
| base | ~16x realtime | Better | 74 MB |
| small | ~6x realtime | Very Good | 244 MB |
| medium | ~3x realtime | Excellent | 769 MB |
| large | ~1x realtime | Best | 1550 MB |

Example: 10-minute audio → ~30 seconds with base model

### Embedding Speed

- Single inspection: ~50ms
- Batch of 10: ~200ms
- Full transcript (8000 chars): ~100ms

### Storage Size

Per inspection:
- Neo4j nodes: ~5-20 KB
- Vector embeddings: ~3 KB (768d × 4 bytes)
- Raw audio: 1-10 MB (depending on quality)
- Transcript: 5-50 KB

## Troubleshooting

### Whisper Not Found

```bash
pip install openai-whisper

# If ffmpeg error:
# macOS:
brew install ffmpeg

# Ubuntu:
sudo apt-get install ffmpeg
```

### Neo4j Connection Failed

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check connection
python -c "from neo4j import GraphDatabase; print('OK')"

# Use in-memory fallback (no Neo4j required)
python bee_inspection_demo.py
```

### Out of Memory

```bash
# Use smaller Whisper model
python bee_inspection_ingest.py audio.wav --date 2024-10-18 --whisper-model tiny

# Process shorter segments
# (split long audio files into 10-15 minute chunks)
```

## Examples

### Example 1: Single Inspection

```bash
# Record inspection audio (phone, recorder, etc.)
# Save as inspection-2024-10-18.wav

# Process
python bee_inspection_ingest.py inspection-2024-10-18.wav \
    --date 2024-10-18 \
    --inspector "Blake"

# Output:
# - inspection-2024-10-18.transcript.txt (transcript)
# - inspection-2024-10-18.json (structured data)
# - Stored in database with embeddings
```

### Example 2: Search Historical Data

```bash
# After processing multiple inspections...

# Find aggressive behavior
python bee_inspection_ingest.py --search "aggressive defensive bees"

# Find treatment issues
python bee_inspection_ingest.py --search "thymol residue foam"

# Find population trends
python bee_inspection_ingest.py --search "weak colony frames decreasing"
```

### Example 3: Comparative Analysis

```python
# Compare hives over time
pipeline = BeeInspectionPipeline()

# Query: "Compare Hive Jodi vs Hive Karen sealing behavior"
results = pipeline.search("sealing propolis tight loose comparison")

# Query: "Population trends across all hives"
results = pipeline.search("frames of bees population strong weak")

# Query: "Treatment effectiveness"
results = pipeline.search("thymol treatment effectiveness survival")
```

## Roadmap

### Phase 1: Core Pipeline ✅
- [x] Audio transcription (Whisper)
- [x] Schema extraction (rule-based)
- [x] Vector embeddings (HoloLoom)
- [x] In-memory storage
- [x] Semantic search

### Phase 2: Enhanced Extraction
- [ ] LLM-based schema extraction
- [ ] Entity linking (hive references)
- [ ] Timestamp alignment
- [ ] Speaker diarization (multiple inspectors)

### Phase 3: Production Storage
- [ ] Qdrant vector store integration
- [ ] Neo4j schema validation
- [ ] Batch import/export
- [ ] Migration scripts

### Phase 4: Analytics
- [ ] Time-series analysis
- [ ] Treatment correlation tracking
- [ ] Population forecasting
- [ ] Seasonal patterns

### Phase 5: Integration
- [ ] HoloLoom memory integration
- [ ] Recursive learning from inspections
- [ ] Multi-modal support (photos, videos)
- [ ] Mobile app connector

## References

- **SKILL.md** - Complete bee inspection schema
- **HoloLoom** - Vector embedding system
- **Whisper** - OpenAI speech recognition
- **Neo4j** - Graph database
- **Qdrant** - Vector database

## License

MIT License - See LICENSE file

## Support

Issues: https://github.com/blakechasteen/hello-world/issues

---

**Built with HoloLoom's self-improving AI for beekeepers who want to learn from their data.**
