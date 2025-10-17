# SpinningWheel

**Lightweight input adapters for HoloLoom**

SpinningWheel converts raw modality data (audio, text, code, etc.) into standardized `MemoryShard` objects that feed into the HoloLoom Orchestrator.

## Philosophy

- **Keep it simple:** Spinners are thin adapters, not full pipelines
- **Standardize output:** All spinners produce `MemoryShard` objects
- **Optional enrichment:** Add context with Ollama/Neo4j/mem0 before orchestration
- **Let Orchestrator handle complexity:** Embeddings, spectral features, and policy decisions happen downstream

## Available Spinners

### AudioSpinner

Converts audio transcripts and related metadata into MemoryShards.

**Supported formats:**
- Plain text transcripts
- Whisper JSON output
- `features.json` (utterance-based with entities/motifs)
- Task lists (CSV)
- Summary documents

**Usage:**

```python
from holoLoom.spinningWheel import AudioSpinner, SpinnerConfig
from holoLoom.orchestrator import HoloLoomOrchestrator
from holoLoom.config import Config
from holoLoom.documentation.types import Query

# Configure spinner
config = SpinnerConfig(
    enable_enrichment=False  # Optional: enable Ollama enrichment
)

# Create spinner
spinner = AudioSpinner(config)

# Spin raw data -> MemoryShards
raw_data = {
    'transcript': "Today I inspected the hives...",
    'tasks': [{'title': 'Order supplies', 'priority': 1}]
}

shards = await spinner.spin(raw_data)

# Feed to Orchestrator
orchestrator = HoloLoomOrchestrator(cfg=Config.fused(), shards=shards)
response = await orchestrator.process(Query(text="What's the hive status?"))
```

## Optional Enrichment

Enable lightweight context enrichment using Ollama:

```python
config = SpinnerConfig(
    enable_enrichment=True,
    ollama_model="llama3.2:3b"  # Fast, local model
)

spinner = AudioSpinner(config)
shards = await spinner.spin(raw_data)  # Now enriched with entities/motifs
```

**Requirements:**
```bash
# Install Ollama
brew install ollama  # macOS
# or: curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2:3b

# Install Python client
pip install ollama
```

## Architecture

```
Raw Audio Data
    ↓
AudioSpinner (parse, normalize, optional enrichment)
    ↓
MemoryShards (standardized format)
    ↓
HoloLoom Orchestrator (embeddings, spectral, policy)
    ↓
Response
```

## Future Spinners

- **TextSpinner:** Plain text/markdown notes
- **CodeSpinner:** Git commits, code diffs
- **VideoSpinner:** Video transcripts + visual features
- **SensorSpinner:** IoT/farm sensor data

## Development

### Adding a New Spinner

1. Inherit from `BaseSpinner`
2. Implement `async def spin(raw_data) -> List[MemoryShard]`
3. Add to factory in `__init__.py`

```python
from holoLoom.spinningWheel.base import BaseSpinner, SpinnerConfig, MemoryShard

class MySpinner(BaseSpinner):
    async def spin(self, raw_data):
        # Parse raw_data
        # Extract entities/motifs
        # Return MemoryShards
        return [MemoryShard(...)]
```

## License

MIT