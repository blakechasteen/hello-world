# 🐝 Bee Inspection Full Pipeline - Complete Guide

## ✅ All Features Implemented!

Your complete bee inspection audio processing system is ready with:

1. **✅ Whisper Audio Transcription** - Speech-to-text for audio files
2. **✅ LLM Schema Extraction** - OpenAI GPT-4 for structured data
3. **✅ Neo4j Graph Storage** - Store in graph database
4. **✅ Process Real Audio Files** - Handle .wav, .mp3, .m4a, etc.

---

## 🚀 Quick Start

### Demo Mode (No Setup Required)
```bash
cd /Users/blakechasteen/mythrL/hello-world
python3 bee_inspection_standalone.py
```

### Process Real Audio
```bash
# Basic usage
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18

# With inspector name
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --inspector "Blake"

# All features enabled
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm --neo4j
```

---

## 📦 Installation

### Core Dependencies (Required)
```bash
pip install numpy torch sentence-transformers einops
```

### Optional Features

**1. Whisper (Audio Transcription)**
```bash
pip install openai-whisper

# macOS: Install ffmpeg
brew install ffmpeg

# Ubuntu: Install ffmpeg
sudo apt-get install ffmpeg
```

**2. OpenAI (LLM Extraction)**
```bash
pip install openai

# Set API key
export OPENAI_API_KEY='your-openai-api-key'
```

**3. Neo4j (Graph Storage)**
```bash
pip install neo4j

# Start Neo4j with Docker
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -d neo4j:latest

# Or download from: https://neo4j.com/download/
```

---

## 🎯 Feature Details

### 1. Whisper Audio Transcription

**Automatically transcribes** bee inspection audio to text.

```bash
# Supported formats: .wav, .mp3, .m4a, .flac
python3 bee_inspection_standalone.py inspection.wav --date 2024-10-18
```

**Model Options:**
- `tiny` - Fastest (32x realtime), 39MB
- `base` - Balanced (16x realtime), 74MB ⭐ Default
- `small` - Better quality (6x realtime), 244MB
- `medium` - High quality (3x realtime), 769MB
- `large` - Best quality (1x realtime), 1550MB

**Output:**
- Full transcript text
- Timestamped segments
- Saved to `output_[filename]/transcript.txt`

### 2. LLM Schema Extraction

**Uses GPT-4** to extract structured bee inspection data following SKILL.md schema.

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='sk-...'

# Enable LLM extraction
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm
```

**Extracts:**
- InspectionEvent (date, weather, temperature, purpose)
- Hive entities (name, genetics, status)
- Population measurements
- Behavior observations
- Treatment details
- Key insights

**Schema Following SKILL.md:**
```json
{
  "inspection": {
    "eventId": "inspect-2024-10-18-001",
    "date": "2024-10-18",
    "weatherTemp": 68.0,
    "weatherCondition": "clear",
    "inspector": "Blake"
  },
  "hives": [
    {
      "hiveId": "hive-jodi-001",
      "commonName": "Jodi",
      "genetics": "dennis-line",
      "colonyStatus": "strong"
    }
  ],
  "observations": [
    "Very calm bees, minimal smoke needed",
    "Strong population - 8 frames of bees"
  ]
}
```

### 3. Neo4j Graph Storage

**Stores data in Neo4j** with relationships between inspections, hives, and observations.

```bash
# Start Neo4j
docker run --name neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# Enable Neo4j storage
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --neo4j
```

**Graph Structure:**
```
(InspectionEvent)-[:INSPECTED]->(Hive)
(InspectionEvent)-[:MEASURED_STOCK]->(PopulationStock)
(Treatment)-[:APPLIED_TO]->(Hive)
```

**Query Example:**
```cypher
// Find all inspections for a hive
MATCH (i:InspectionEvent)-[:INSPECTED]->(h:Hive {commonName: "Jodi"})
RETURN i.date, i.weatherTemp, h.colonyStatus
ORDER BY i.date DESC

// Find hives inspected on a date
MATCH (i:InspectionEvent {date: "2024-10-18"})-[:INSPECTED]->(h:Hive)
RETURN h.commonName, h.genetics, h.colonyStatus
```

### 4. Vector Embeddings + Semantic Search

**All data gets embedded** using Nomic v1.5 (768 dimensions) for semantic search.

```bash
# After processing inspections, search semantically
python3 bee_inspection_standalone.py --search "aggressive bees defensive"
```

**Search Examples:**
- "How did the bees behave?" → Finds behavior observations
- "Population frames of bees" → Finds population measurements
- "Treatments thymol dosage" → Finds treatment details
- "Weak colony winter prep" → Finds struggling hives

---

## 📝 Complete Usage Examples

### Example 1: Basic Audio Processing
```bash
# Record inspection on your phone, transfer audio file
# inspection-2024-10-18.wav

python3 bee_inspection_standalone.py inspection-2024-10-18.wav \
  --date 2024-10-18 \
  --inspector "Blake"

# Output:
# ✓ Transcribed with Whisper
# ✓ Extracted structured schema
# ✓ Generated embeddings (768d)
# ✓ Stored in memory
# ✓ Saved to output_inspection-2024-10-18/
```

### Example 2: Full Pipeline (All Features)
```bash
# Prerequisites:
# 1. Whisper installed: pip install openai-whisper
# 2. OpenAI key set: export OPENAI_API_KEY='sk-...'
# 3. Neo4j running: docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

python3 bee_inspection_standalone.py inspection.wav \
  --date 2024-10-18 \
  --inspector "Blake" \
  --use-llm \
  --neo4j

# Output:
# ✓ Transcribed 45 segments (Whisper)
# ✓ Extracted schema (GPT-4)
# ✓ Found 3 hives, 8 observations
# ✓ Generated embeddings
# ✓ Stored in Neo4j
# ✓ Files saved to output_inspection/
```

### Example 3: Batch Processing
```bash
# Process multiple inspections
for file in inspections/*.wav; do
  date=$(basename "$file" .wav | cut -d'-' -f2-4)
  python3 bee_inspection_standalone.py "$file" --date "$date" --use-llm --neo4j
done
```

### Example 4: From Text Transcript
```bash
# Already have transcript? Save as .txt
echo "Bee inspection on October 18..." > inspection.txt

python3 bee_inspection_standalone.py inspection.txt \
  --date 2024-10-18 \
  --use-llm
```

---

## 📊 Output Files

After processing, you get:

```
output_inspection-2024-10-18/
├── transcript.txt          # Full transcription
├── structured.json         # Extracted schema
└── embeddings.npy          # Vector embeddings (optional)
```

**structured.json example:**
```json
{
  "inspection": {
    "eventId": "inspect-2024-10-18-001",
    "date": "2024-10-18",
    "timestamp": 1729267200,
    "inspector": "Blake",
    "weatherTemp": 68.0,
    "weatherCondition": "clear",
    "primaryPurpose": "routine_check"
  },
  "hives": [
    {
      "hiveId": "hive-jodi-001",
      "commonName": "Jodi",
      "genetics": "dennis-line",
      "colonyStatus": "strong"
    }
  ],
  "observations": [
    "Very calm bees, minimal smoke needed",
    "8 frames of bees in top box - strong population",
    "Thymol treatment round 2 completed"
  ],
  "transcript_length": 1847,
  "segments_count": 12
}
```

---

## 🔧 Configuration Options

### Command Line Arguments

```bash
python3 bee_inspection_standalone.py [OPTIONS] [AUDIO_FILE]

Arguments:
  AUDIO_FILE              Path to audio or text file

Options:
  --date YYYY-MM-DD       Inspection date (required for audio files)
  --inspector NAME        Inspector name
  --use-llm              Enable LLM extraction (needs OPENAI_API_KEY)
  --neo4j                Enable Neo4j storage
  --neo4j-uri URI        Neo4j URI (default: bolt://localhost:7687)
  --neo4j-user USER      Neo4j username (default: neo4j)
  --neo4j-password PASS  Neo4j password (default: password)
```

### Environment Variables

```bash
# OpenAI API key for LLM extraction
export OPENAI_API_KEY='sk-...'

# Neo4j connection (alternative to CLI args)
export NEO4J_URI='bolt://localhost:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='your-password'
```

---

## 🎯 Integration with HoloLoom

### Use as Memory Shards
```python
from bee_inspection_standalone import BeeInspectionPipeline
from HoloLoom.documentation.types import MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

# Process inspection
pipeline = BeeInspectionPipeline(use_whisper=True)
result = await pipeline.process_audio(audio_path, date="2024-10-18")

# Convert to memory shards
shards = [
    MemoryShard(
        text=result["transcript"],
        source=f"bee_inspection_{result['inspection'].eventId}",
        metadata={
            "date": result["inspection"].date,
            "hives": [h.commonName for h in result["hives"]]
        }
    )
]

# Query with HoloLoom
config = Config.fast()
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    response = await shuttle.weave(Query(text="How was Hive Jodi's behavior?"))
```

### Recursive Learning
```python
from HoloLoom.recursive import FullLearningEngine

# Enable learning across multiple inspections
async with FullLearningEngine(cfg=config, shards=shards) as engine:
    # System learns patterns over time
    result = await engine.weave(
        Query(text="Compare population trends across all inspections"),
        enable_refinement=True
    )
```

---

## 🔍 Troubleshooting

### Whisper Not Found
```bash
pip install openai-whisper

# If error about ffmpeg:
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu
```

### OpenAI API Errors
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Or set it
export OPENAI_API_KEY='sk-...'

# Test connection
python3 -c "import openai; print('OK')"
```

### Neo4j Connection Failed
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Start Neo4j
docker run --name neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

# Test connection
python3 -c "from neo4j import GraphDatabase; print('OK')"
```

### Out of Memory
```bash
# Use smaller Whisper model
python3 bee_inspection_standalone.py audio.wav --date 2024-10-18
# (base model is default, or use tiny)

# Split long audio files into chunks
ffmpeg -i long.wav -f segment -segment_time 600 -c copy chunk%03d.wav
```

---

## ⚡ Performance

| Component | Speed | Memory |
|-----------|-------|--------|
| Whisper (base) | 16x realtime | ~500MB |
| LLM extraction | ~2-5 seconds | Minimal |
| Embeddings | ~100ms | ~200MB |
| Neo4j storage | <50ms | Varies |

**Example: 10-minute audio**
- Transcription: ~30 seconds
- LLM extraction: ~3 seconds
- Embeddings: ~100ms
- Total: **~35 seconds**

---

## 📈 Roadmap

### Completed ✅
- [x] Whisper audio transcription
- [x] LLM schema extraction
- [x] Neo4j graph storage
- [x] Vector embeddings
- [x] Semantic search
- [x] Process real audio files

### Future Enhancements
- [ ] Multi-speaker diarization
- [ ] Real-time transcription
- [ ] Photo/video integration
- [ ] Automated insights dashboard
- [ ] Mobile app integration
- [ ] Time-series analysis
- [ ] Treatment effectiveness tracking

---

## 💡 Tips & Best Practices

1. **Audio Quality**: Record in quiet environment, close to mic
2. **File Format**: WAV or FLAC preferred (lossless)
3. **Length**: Split into 10-15 min segments for better results
4. **Naming**: Use format `inspection-YYYY-MM-DD.wav`
5. **LLM Cost**: ~$0.01 per inspection (GPT-4o-mini)
6. **Storage**: Keep audio files for re-processing with better models

---

## 🎉 Success!

You now have a complete pipeline that:
1. ✅ Transcribes bee inspection audio with Whisper
2. ✅ Extracts structured schema with LLM
3. ✅ Stores in Neo4j graph database
4. ✅ Enables semantic search with embeddings

**Test it now:**
```bash
python3 bee_inspection_standalone.py
```

Then try with your real bee inspection audio files! 🐝
