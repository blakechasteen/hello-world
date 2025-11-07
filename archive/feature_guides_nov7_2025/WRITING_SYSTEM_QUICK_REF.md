# HoloLoom Writing System - Quick Reference

**Version**: 1.0.0 | **Status**: ✅ Production Ready | **Tests**: 21/21 passing

---

## 30-Second Start

```python
from HoloLoom.writing import write
from HoloLoom.documentation.types import MemoryShard

# Create memories
memories = [
    MemoryShard(id='m1', text="Your content here", metadata={'relevance': 0.95})
]

# Generate polished output
content = await write("Your query here", memories, refine=True)
```

---

## API Reference

### Simple API

```python
write(
    query: str,                    # User query
    memories: List[MemoryShard],   # Memory context
    mode: Optional[str] = None,    # 'narrative', 'technical', 'analysis', 'creative', or None (auto)
    style: Optional[str] = None,   # 'academic', 'casual', 'tufte', or None (auto)
    refine: bool = True,           # Enable multi-pass refinement
    format: str = 'markdown'       # Output format
) -> str
```

### Advanced API

```python
from HoloLoom.writing import Writer, create_default_writer
from HoloLoom.writing.core import WritingContext, WritingMode, StyleGuide

writer = create_default_writer()
context = WritingContext(query=query, memories=memories, mode=WritingMode.AUTO)
result = await writer.write(context, refine=True)
```

---

## 4 Writing Modes

| Mode | Use Case | Refiner | Structure |
|------|----------|---------|-----------|
| **NARRATIVE** | Explanations, teaching | ELEGANCE | Setup → Development → Conclusion |
| **TECHNICAL** | Documentation, how-tos | VERIFY | Overview → Implementation → Usage → Parameters |
| **ANALYSIS** | Reports, comparisons | VERIFY | Summary → Findings → Analysis → Conclusions |
| **CREATIVE** | Stories, poems, fiction | ELEGANCE | Story arc / Poetic structure |

---

## 2 Refinement Strategies

### ELEGANCE (for narrative/creative)

```
Pass 1: CLARITY    → Break long sentences, replace jargon
Pass 2: SIMPLICITY → Remove fillers, eliminate redundancy
Pass 3: BEAUTY     → Active voice, vary structure
```

**Cost**: +40-60ms | **Quality**: 0.67 → 0.91 typical

### VERIFY (for technical/analysis)

```
Pass 1: ACCURACY    → Qualify claims, add sources
Pass 2: COMPLETENESS → Expand acronyms, fill gaps
Pass 3: CONSISTENCY  → Standardize terms, fix contradictions
```

**Cost**: +60-80ms | **Quality**: 0.75 → 0.95 typical

---

## Mode Auto-Detection

| Query Pattern | Detected Mode |
|---------------|---------------|
| "What is X?" | NARRATIVE |
| "How to implement X?" | TECHNICAL |
| "Compare X and Y" | ANALYSIS |
| "Write a story about X" | CREATIVE |
| "Explain X" | NARRATIVE |
| "Analyze X" | ANALYSIS |

---

## Templates

### Email Template

```python
from HoloLoom.writing.templates import EmailTemplate
from HoloLoom.writing.templates.base import TemplateContext, TemplateType

template = EmailTemplate(style='professional', structure='standard')
context = TemplateContext(
    template_type=TemplateType.EMAIL,
    variables={'recipient': 'Blake', 'purpose': 'project_update'},
    memories=memories
)
result = await template.generate(context)
```

**54 variants**: 3 styles × 3 structures × 6 purposes

### Report Template

```python
from HoloLoom.writing.templates import ReportTemplate

template = ReportTemplate(
    report_type='executive',  # 'executive', 'technical', 'research'
    include_toc=True,
    include_executive_summary=True
)
result = await template.generate(context)
```

**12 configs**: 3 types × 2 TOC options × 2 summary options

---

## Export Formats

### Markdown

```python
from HoloLoom.writing.export import MarkdownExporter

exporter = MarkdownExporter(include_metadata=True, include_quality=True)
markdown = exporter.export(writing_result)
```

**Features**: YAML frontmatter, quality metrics as HTML comments

### HTML (Tufte CSS)

```python
from HoloLoom.writing.export import HTMLExporter

exporter = HTMLExporter(include_quality_sidebar=True, include_tufte_css=True)
html = exporter.export(writing_result)
```

**Features**: Tufte design, quality sidebar, responsive layout

---

## Integration Patterns

### With Weaving Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.writing import write

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(query)
    content = await write(query.text, spacetime.trace.retrieved_context, refine=True)
```

### With Recursive Learning

```python
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.writing import write

async with FullLearningEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query, enable_refinement=True)
    content = await write(query.text, spacetime.trace.retrieved_context)
```

---

## Performance

| Operation | Time |
|-----------|------|
| Simple generation (no refine) | 10-20ms |
| + ELEGANCE refinement | +40-60ms |
| + VERIFY refinement | +60-80ms |
| + Email template | +15-25ms |
| + Report template | +25-40ms |
| + Markdown export | +<5ms |
| + HTML export | +10-15ms |
| **Complete pipeline** | **100-150ms** |

---

## Quality Scoring (6 Dimensions)

```python
quality_scores = {
    'clarity': 0.85,       # Sentence structure, readability
    'completeness': 0.78,  # Content depth vs. query
    'coherence': 0.92,     # Logical flow, structure
    'accuracy': 0.88,      # Memory alignment
    'conciseness': 0.81,   # Signal-to-noise ratio
    'elegance': 0.74       # Stylistic quality
}
```

**Weights vary by mode**:
- Technical: accuracy (30%), completeness (25%)
- Creative: elegance (30%), coherence (25%)
- Analysis: accuracy (30%), clarity (30%)

---

## Common Use Cases

### 1. Explanation (Narrative)

```python
content = await write("What is Thompson Sampling?", memories, refine=True)
```

### 2. Documentation (Technical)

```python
content = await write("How to implement X?", memories, mode='technical', refine=True)
```

### 3. Comparison Report (Analysis)

```python
content = await write("Compare X and Y", memories, mode='analysis', refine=True)
```

### 4. Creative Writing

```python
content = await write("Write a story about X", memories, mode='creative', refine=True)
```

### 5. Professional Email

```python
from HoloLoom.writing.templates import EmailTemplate

template = EmailTemplate(style='professional', structure='brief')
email = await template.generate(context)
```

### 6. Executive Report

```python
from HoloLoom.writing.templates import ReportTemplate
from HoloLoom.writing.export import HTMLExporter

# Generate
template = ReportTemplate(report_type='executive', include_toc=True)
report = await template.generate(context)

# Export
exporter = HTMLExporter(include_quality_sidebar=True)
html = exporter.export(report)
```

---

## Testing

```bash
# Run all writing tests
PYTHONPATH=. python -m pytest HoloLoom/tests/unit/test_writing.py -v

# Run feature demos
PYTHONPATH=. python demos/demo_writing_system.py

# Run integration demos
PYTHONPATH=. python demos/demo_writing_orchestrator_integration.py
```

**Test Coverage**: 21/21 passing (100%)

---

## Troubleshooting

### Issue: Empty output

**Solution**: Check memory relevance scores (should be >0.5)

### Issue: Wrong mode detected

**Solution**: Explicitly specify mode:
```python
content = await write(query, memories, mode='technical', refine=True)
```

### Issue: Slow performance

**Solution**: Disable refinement or reduce memory count:
```python
content = await write(query, memories[:5], refine=False)
```

---

## File Locations

```
HoloLoom/writing/              # Main package
  ├── core/                    # Core engine
  ├── modes/                   # 4 writing modes
  ├── refinement/              # 2 refiners
  ├── templates/               # Email + Report templates
  └── export/                  # Markdown + HTML exporters

HoloLoom/tests/unit/test_writing.py  # Tests
demos/demo_writing_system.py          # Feature demos
demos/demo_writing_orchestrator_integration.py  # Integration demos
```

---

## Documentation

- **README**: [HoloLoom/writing/README.md](HoloLoom/writing/README.md) (605 lines)
- **Integration Guide**: [WRITING_SYSTEM_INTEGRATION_GUIDE.md](WRITING_SYSTEM_INTEGRATION_GUIDE.md)
- **Final Summary**: [WRITING_SYSTEM_FINAL_SUMMARY.md](WRITING_SYSTEM_FINAL_SUMMARY.md)
- **Phase 1-3 Docs**: WRITING_SYSTEM_COMPLETE.md, WRITING_PHASE_2_COMPLETE.md, WRITING_PHASE_3_COMPLETE.md
- **All Phases**: [WRITING_SYSTEM_ALL_PHASES_COMPLETE.md](WRITING_SYSTEM_ALL_PHASES_COMPLETE.md)

---

## Statistics

| Metric | Value |
|--------|-------|
| Total Lines | 7,780 |
| Core Code | 4,926 |
| Tests | 521 |
| Docs | 2,333 |
| Modes | 4 |
| Refiners | 2 |
| Templates | 2 |
| Exporters | 2 |
| Test Pass Rate | 100% (21/21) |

---

**Status**: ✅ Production Ready - All 3 Phases Complete

Quick questions? Check [WRITING_SYSTEM_INTEGRATION_GUIDE.md](WRITING_SYSTEM_INTEGRATION_GUIDE.md)

Deep dive? See [WRITING_SYSTEM_FINAL_SUMMARY.md](WRITING_SYSTEM_FINAL_SUMMARY.md)
