# Keep Elegant Design Patterns

This document describes the elegant design patterns implemented in Keep v0.2.0+

## Overview

Keep has been enhanced with sophisticated design patterns that provide:
- **Clean, expressive APIs** - Fluent builders and functional transformations
- **Testable, reusable components** - Protocol-based extensibility
- **Composable analytics** - Rich insights through functional composition
- **Narrative intelligence** - Temporal storytelling and pattern recognition
- **Seamless integration** - HoloLoom memory and reasoning adapters

## Pattern Catalog

### 1. Protocol-Based Extensibility

**File**: [`protocols.py`](protocols.py)

Defines extensible protocols for integrating custom implementations:

```python
from apps.keep.protocols import (
    InspectionDataSource,
    ColonyHealthAnalyzer,
    AlertGenerator,
    RecommendationEngine,
    WeatherDataProvider,
    JournalIntegration,
)
```

**Available Protocols**:
- `InspectionDataSource` - Custom data sources (mobile apps, IoT sensors, APIs)
- `ColonyHealthAnalyzer` - Pluggable health assessment algorithms
- `AlertGenerator` - Custom alerting strategies
- `RecommendationEngine` - Recommendation generation systems
- `ApiaryStateExporter` - Export to various formats
- `WeatherDataProvider` - Weather integration
- `JournalIntegration` - Narrative tracking systems

**Benefits**:
- Dependency injection ready
- Easy to test with mocks
- Extend without modifying core code
- Swap implementations at runtime

### 2. Fluent Builders

**File**: [`builders.py`](builders.py)

Provides chainable, readable object construction:

```python
from apps.keep import hive, colony, inspection, alert

# Fluent hive construction
hive1 = (hive("Alpha")
    .langstroth()
    .at("East meadow, near oak grove")
    .installed_on(datetime(2024, 3, 15))
    .notes("Excellent morning sun")
    .build())

# Fluent colony construction
colony1 = (colony()
    .in_hive(hive1.hive_id)
    .italian()
    .from_package()
    .excellent_health()
    .queen_laying()
    .population(60000)
    .queen_age(8)
    .build())

# Fluent inspection construction
inspection1 = (inspection()
    .for_hive(hive1.hive_id)
    .routine()
    .on(datetime.now())
    .weather("Sunny, 74°F")
    .queen_seen()
    .eggs_present()
    .brood_frames(8)
    .honey_frames(6)
    .no_pests()
    .action("Added honey super")
    .build())
```

**Benefits**:
- Highly readable
- IDE autocomplete friendly
- Sensible defaults
- Type-safe construction
- Self-documenting code

**Available Builders**:
- `HiveBuilder` / `hive()` - Build hives
- `ColonyBuilder` / `colony()` - Build colonies
- `InspectionBuilder` / `inspection()` - Build inspections
- `AlertBuilder` / `alert()` - Build alerts

### 3. Functional Transformations

**File**: [`transforms.py`](transforms.py)

Composable, functional data processing:

```python
from apps.keep import (
    filter_healthy,
    filter_concerning,
    filter_queenless,
    sort_by_health,
    sort_by_population,
    get_top_healthy_colonies,
    pipe,
    compose,
)

# Simple filters
healthy = filter_healthy(colonies)
concerning = filter_concerning(colonies)
queenless = filter_queenless(colonies)

# Composable transforms
top_5_healthy = get_top_healthy_colonies(5)(colonies)

# Custom pipelines
my_transform = pipe(
    filter_healthy,
    sort_by_population(reverse=True),
    take(3)
)
result = my_transform(colonies)
```

**Available Transforms**:

**Filters**:
- `filter_by_health(*statuses)` - Filter by health status
- `filter_healthy()` - Only healthy colonies
- `filter_concerning()` - Colonies needing attention
- `filter_by_queen_status(*statuses)` - Filter by queen status
- `filter_queenless()` - Potentially queenless colonies
- `filter_by_time_range(start, end)` - Filter by time
- `filter_recent(days)` - Recent inspections

**Sorts**:
- `sort_by_health(reverse)` - Sort by health status
- `sort_by_population(reverse)` - Sort by population
- `sort_by_timestamp(reverse)` - Sort by time
- `sort_alerts_by_priority(reverse)` - Sort alerts

**Aggregations**:
- `take(n)` - Take first n items
- `group_by(key_fn)` - Group items by key
- `count_by(key_fn)` - Count items by key
- `compute_health_distribution()` - Health status counts
- `compute_average_population()` - Average population
- `compute_inspection_frequency()` - Days between inspections
- `compute_harvest_totals()` - Harvest sums

**Composition**:
- `pipe(*funcs)` - Left-to-right composition
- `compose(*funcs)` - Right-to-left composition

**Benefits**:
- Pure functions, easy to test
- Reusable transformations
- Composable pipelines
- No side effects
- Lazy evaluation possible

### 4. Composable Analytics

**File**: [`analytics.py`](analytics.py)

Rich analysis through functional composition:

```python
from apps.keep import (
    ApiaryAnalytics,
    quick_health_check,
    productivity_summary,
)

# Quick health check
health = quick_health_check(apiary)
# Returns: {health_grade, health_score, risk_level, critical_actions}

# Detailed analytics
analytics = ApiaryAnalytics(apiary)

# Health analysis
health_score = analytics.compute_health_score()
trends = analytics.analyze_health_trends(days=90)

# Productivity analysis
metrics = analytics.analyze_productivity(days=365)
# Returns: ProductivityMetrics(total_harvest, avg_per_hive, top_producers, ...)

# Risk assessment
risk = analytics.assess_risk()
# Returns: RiskAssessment(overall_risk, risk_factors, at_risk_colonies, ...)

# Comparative analysis
comparisons = analytics.compare_colonies()

# Comprehensive report
report = analytics.generate_report()
```

**Available Analytics**:

**Health**:
- `compute_health_score()` - Overall health score (0-100)
- `analyze_health_trends()` - Trend analysis over time
- `compute_health_distribution()` - Status distribution

**Productivity**:
- `analyze_productivity()` - Harvest metrics
- `productivity_summary()` - Human-readable summary

**Risk**:
- `assess_risk()` - Comprehensive risk assessment
- `quick_health_check()` - Fast overview

**Comparisons**:
- `compare_colonies()` - Multi-dimensional comparison
- `generate_report()` - Complete analytics report

**Benefits**:
- Rich insights from simple data
- Trend detection
- Predictive capabilities
- Actionable recommendations
- Statistical rigor

### 5. Narrative Journaling

**File**: [`journal.py`](journal.py)

Temporal storytelling and pattern recognition:

```python
from apps.keep import (
    create_journal,
    EntryType,
    Sentiment,
)

# Create journal
journal = create_journal(apiary)

# Record entries
journal.observe(
    "Spring has arrived, bees are very active!",
    hive_ids=[hive1.hive_id],
    tags=["spring", "active"]
)

journal.celebrate(
    "First honey harvest! 45 lbs of beautiful amber honey.",
    hive_ids=[hive1.hive_id],
    tags=["harvest", "milestone"],
    quantity_lbs=45.0
)

journal.concern(
    "Queen not laying in Gamma hive, need to monitor closely.",
    sentiment=Sentiment.WORRIED,
    hive_ids=[hive3.hive_id],
    tags=["queen_issue"]
)

# Synthesize narrative
narrative = journal.synthesize_narrative(since=since, until=until)
# Returns: NarrativeSynthesis with summary, themes, sentiment_arc, ...

# Extract insights
insights = await journal.extract_insights()
# Returns: {insights, patterns, recommendations}

# Get timeline (merged journal + inspections)
timeline = journal.get_timeline(days=30)
```

**Entry Types**:
- `OBSERVATION` - What you saw
- `DECISION` - What you decided
- `REFLECTION` - What you learned
- `MILESTONE` - What you achieved
- `CONCERN` - What worries you
- `CELEBRATION` - What you're proud of

**Sentiments**:
- `POSITIVE` - Things are going well
- `NEUTRAL` - Normal observations
- `CONCERNED` - Minor worries
- `WORRIED` - Serious concerns

**Benefits**:
- Captures tacit knowledge
- Pattern recognition
- Sentiment tracking
- Temporal synthesis
- Learning from history
- Rich storytelling

### 6. HoloLoom Integration

**File**: [`hololoom_adapter.py`](hololoom_adapter.py)

Seamless AI reasoning integration:

```python
from apps.keep import (
    export_to_memory,
    create_hololoom_session,
    ApiaryMemoryAdapter,
)

# Export to memory shards
shards = export_to_memory(apiary)
# Returns list of MemoryShard objects for HoloLoom

# Query with HoloLoom
async with create_hololoom_session(apiary) as session:
    result = await session.query("What needs attention this week?")
    # Returns: {question, answer, confidence, sources, reasoning}

    insights = await session.get_insights()
    # Returns automated insights about apiary state

# Manual adapter usage
adapter = ApiaryMemoryAdapter(apiary)
shards = adapter.to_memory_shards()
```

**Exported Data**:
- Apiary overview (fleet status, production)
- Hive metadata (type, location, age)
- Colony status (health, queen, population)
- Recent inspections (last 20)
- Active alerts (top 10)

**Benefits**:
- Natural language queries
- AI-powered insights
- Context-aware reasoning
- Knowledge accumulation
- Graceful degradation (works without HoloLoom)

## Usage Examples

### Example 1: Elegant Data Pipeline

```python
from apps.keep import pipe, filter_healthy, sort_by_population, take

# Build a reusable transform
get_top_producers = pipe(
    filter_healthy,
    sort_by_population(reverse=True),
    take(5)
)

# Apply to data
top_colonies = get_top_producers(all_colonies)
```

### Example 2: Builder with Analytics

```python
from apps.keep import hive, colony, ApiaryAnalytics

# Build objects elegantly
h = hive("Alpha").langstroth().at("East field").build()
c = colony().in_hive(h.hive_id).italian().healthy().population(50000).build()

apiary.add_hive(h)
apiary.add_colony(c)

# Analyze immediately
analytics = ApiaryAnalytics(apiary)
health = analytics.compute_health_score()
```

### Example 3: Journal-Driven Insights

```python
from apps.keep import create_journal

journal = create_journal(apiary)

# Record throughout season
journal.observe("First drones spotted")
journal.celebrate("Spring buildup complete, 8 frames of brood!")
journal.concern("Seeing more mites than usual")
journal.decide("Starting mite treatment today")

# Extract patterns
insights = await journal.extract_insights()
# Detects: recurring concerns, sentiment trends, milestone frequency
```

### Example 4: Protocol-Based Extension

```python
from apps.keep.protocols import ColonyHealthAnalyzer

class MLHealthAnalyzer:
    """Custom ML-based health analyzer."""

    def analyze_health(self, colony, inspections, context):
        # Custom ML model
        features = extract_features(colony, inspections)
        prediction = self.model.predict(features)
        return HealthStatus(prediction)

    def get_health_factors(self, colony, inspections):
        return self.model.feature_importance()

# Use custom analyzer
analyzer = MLHealthAnalyzer()
health = analyzer.analyze_health(colony, recent_inspections, {})
```

## Design Principles

1. **Composition over Inheritance** - Build complex behavior from simple pieces
2. **Functional Core, Imperative Shell** - Pure functions wrapped in stateful APIs
3. **Protocol-Based Design** - Extend through protocols, not inheritance
4. **Fluent Interfaces** - Chainable, readable APIs
5. **Separation of Concerns** - Each module has single responsibility
6. **Graceful Degradation** - Works without optional dependencies
7. **Type Safety** - Full type hints for IDE support
8. **Testability** - Pure functions and protocols enable easy testing

## Performance Considerations

- **Lazy Evaluation**: Transforms are eager by default, but can be adapted for lazy evaluation
- **Caching**: Analytics results can be cached with `@lru_cache`
- **Batch Operations**: Functional transforms enable batch processing
- **Memory Efficiency**: Generators can replace lists for large datasets

## Testing Strategy

```python
# Test builders
def test_hive_builder():
    h = hive("Test").langstroth().at("Test location").build()
    assert h.name == "Test"
    assert h.hive_type == HiveType.LANGSTROTH

# Test transforms
def test_filter_healthy():
    colonies = [healthy_colony, sick_colony]
    result = filter_healthy(colonies)
    assert len(result) == 1

# Test composition
def test_pipe():
    transform = pipe(filter_healthy, take(1))
    result = transform(colonies)
    assert len(result) == 1
```

## Migration Guide

### From v0.1.0 to v0.2.0

**Old Style**:
```python
hive = Hive()
hive.name = "Alpha"
hive.hive_type = HiveType.LANGSTROTH
hive.location = "East field"
apiary.add_hive(hive)
```

**New Style**:
```python
hive = (hive("Alpha")
    .langstroth()
    .at("East field")
    .build())
apiary.add_hive(hive)
```

**Both styles work** - v0.2.0 is backward compatible!

## Future Enhancements

Planned for v0.3.0+:
- [ ] Async transforms for large datasets
- [ ] Stream processing utilities
- [ ] More statistical analytics
- [ ] Pattern matching DSL
- [ ] Query language for journal
- [ ] ML integration protocols
- [ ] Real-time event streaming

---

**Version**: 0.2.0
**Last Updated**: 2025-10-28
**Maintainer**: mythRL Team
