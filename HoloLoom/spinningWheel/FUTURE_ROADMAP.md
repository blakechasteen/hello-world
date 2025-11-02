# SpinningWheel Future Roadmap

This document outlines planned spinners and enhancements for the HoloLoom data ingestion system.

## Phase 1: Core Spinners (✅ Complete - Nov 2025)

- ✅ WhisperSpinner - Local audio transcription with timecodes
- ✅ YouTubeSpinner - Video transcript ingestion
- ✅ PDFSpinner - Document extraction
- ✅ EmailSpinner - Email archive ingestion
- ✅ CodebaseSpinner - Source code parsing
- ✅ GitSpinner - Git repository history
- ✅ MatrixSpinner - Matrix.org chat rooms
- ✅ FileUploadSpinner - Universal file dispatcher
- ✅ URLSpinner - Web crawling with recursive discovery

## Phase 2: Advanced OCR & Vision (Q1-Q2 2026)

### DeepSeek OCR Integration
**Priority**: HIGH
**Status**: Planned

Integrate DeepSeek's vision models for advanced OCR capabilities:

**Features**:
- Scanned document OCR (better than Tesseract)
- Handwritten text recognition
- Mathematical formula extraction (LaTeX)
- Diagram/chart understanding
- Table structure preservation
- Multi-language support (100+ languages)

**Use Cases**:
- Historical documents
- Handwritten notes
- Scientific papers with equations
- Whiteboards and sketches
- Mixed media documents

**Implementation**:
```python
class DeepSeekOCRSpinner(BaseSpinner):
    """
    Advanced OCR using DeepSeek vision models.

    Capabilities:
    - Handwritten text recognition
    - Mathematical formula extraction
    - Diagram understanding
    - Table structure preservation
    """
    def __init__(
        self,
        model="deepseek-ocr-large",
        extract_formulas=True,
        extract_diagrams=True
    ):
        # ... implementation
```

**Dependencies**:
- DeepSeek API or local model
- PIL/OpenCV for image preprocessing
- Optional: GPU for local inference

### Other Vision Enhancements

- **ImageSpinner**: General image understanding and captioning
- **VideoSpinner**: Video frame analysis (not just audio)
- **DiagramSpinner**: Flowchart/architecture diagram parsing

## Phase 3: Real-Time Data Sources (Q2 2026)

### Live Streaming Spinners

- **TwitterSpinner**: Real-time tweet monitoring
- **RedditSpinner**: Subreddit monitoring
- **DiscordSpinner**: Discord channel ingestion
- **TelegramSpinner**: Telegram channel monitoring

### News & RSS

- **RSSSpinner**: RSS feed aggregation
- **NewsSpinner**: News article extraction with deduplication

## Phase 4: Structured Data (Q3 2026)

### Database Spinners

- **SQLSpinner**: SQL database schema and query ingestion
- **MongoSpinner**: MongoDB document ingestion
- **GraphQLSpinner**: GraphQL API introspection

### Spreadsheet & Documents

- **ExcelSpinner**: Excel workbook parsing
- **GoogleSheetsSpinner**: Google Sheets integration
- **NotionSpinner**: Notion database export
- **ObsidianSpinner**: Obsidian vault ingestion

## Phase 5: Collaboration Tools (Q4 2026)

### Messaging Platforms

- **SlackSpinner**: Slack workspace archival
- **TeamsSpinner**: Microsoft Teams chat
- **ZoomSpinner**: Meeting transcripts

### Project Management

- **JiraSpinner**: Issue tracking ingestion
- **GitHubSpinner**: GitHub issues, PRs, discussions
- **LinearSpinner**: Linear issue tracking

## Phase 6: Multimedia & Creative (2027)

### Audio/Music

- **MusicSpinner**: Music metadata and lyrics
- **PodcastSpinner**: Podcast feed ingestion with chapters

### Creative Tools

- **FigmaSpinner**: Design file comments/annotations
- **BlenderSpinner**: 3D model metadata
- **CADSpinner**: CAD file documentation

## Raw Data Preservation ("Wool" Storage)

**Philosophy**: Always save the original raw data before "spinning" it into processed shards.

### Current Implementation

- ✅ **Audio**: Save raw audio files to `./data/wool/audio/`
- ✅ **YouTube**: Save permalinks to `./data/wool/youtube/`
- 🚧 **PDF**: Save original PDFs (planned)
- 🚧 **Email**: Save mbox archives (planned)

### Future Enhancements

- **Versioning**: Track multiple versions of the same source
- **Deduplication**: Content-addressable storage (hash-based)
- **Compression**: Automatic compression for large files
- **Cloud Backup**: Optional S3/GCS backup
- **Metadata**: Store source URL, timestamp, checksums

### Storage Structure

```
data/wool/
├── audio/
│   ├── interview_2025-11-02.wav
│   └── meeting_notes.mp3
├── youtube/
│   ├── abc123_permalink.txt
│   └── def456_metadata.json
├── pdf/
│   ├── research_paper.pdf
│   └── technical_manual.pdf
├── email/
│   ├── inbox_2025-11.mbox
│   └── sent_2025-11.mbox
└── web/
    ├── example.com_snapshot.html
    └── blog.example.org_2025-11-02.html
```

## Performance & Scalability

### Parallel Processing

- **Multi-threaded spinning**: Process multiple files concurrently
- **GPU acceleration**: Use GPUs for Whisper, OCR, vision models
- **Distributed spinning**: Distribute across multiple machines

### Caching & Incremental Updates

- **Checkpointing**: Resume interrupted operations
- **Incremental sync**: Only process new/changed content
- **Result caching**: Cache processed shards for reuse

### Quality & Accuracy

- **Confidence scoring**: Better importance scoring algorithms
- **Error correction**: Auto-correction for transcription errors
- **Multilingual**: Better language detection and handling

## Integration & APIs

### Web Dashboard Integration

- ✅ **YouTube URL input**: WebSocket action for YouTube ingestion
- ✅ **Audio upload**: HTTP endpoint for audio files
- 🚧 **PDF upload**: Drag-and-drop PDF ingestion
- 🚧 **Bulk import**: Process entire directories

### External APIs

- **REST API**: Full REST API for spinner management
- **GraphQL**: GraphQL API for complex queries
- **Webhooks**: Real-time notifications on ingestion
- **MCP Server**: Model Context Protocol server

## Documentation Needs

- **Spinner Development Guide**: How to create new spinners
- **Protocol Documentation**: SpinnerProtocol specification
- **Performance Benchmarks**: Speed/accuracy comparisons
- **Migration Guides**: Upgrading between versions

## Community & Ecosystem

- **Plugin System**: Third-party spinner plugins
- **Spinner Marketplace**: Community-contributed spinners
- **Testing Framework**: Automated spinner testing
- **Quality Metrics**: Spinner quality dashboard

---

**Last Updated**: November 2, 2025
**Next Review**: Q1 2026
