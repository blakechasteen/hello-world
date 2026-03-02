# Recursive Crawling with Matryoshka Gating

Intelligent web crawling that follows only high-quality links, using increasing importance thresholds at each depth.

## The Problem

Regular crawlers either stop at one page (too shallow) or crawl everything (too noisy). Matryoshka gating creates a natural funnel: broad at shallow depths, focused at deep depths.

```
Depth 0: 1 page (seed)
Depth 1: 50 links x 0.6 threshold = ~10 pages
Depth 2: 500 links x 0.75 threshold = ~15 pages
Depth 3: 750 links x 0.85 threshold = ~8 pages
Total: ~34 pages (manageable)
```

## Quick Start

```python
from hololoom.spinningWheel.recursive_crawler import crawl_recursive

pages = await crawl_recursive(
    seed_url='https://example.com/article',
    seed_topic='beekeeping hive management',
    max_depth=2,
    max_pages=20,
)

for page in pages:
    print(f"Depth {page['depth']}: {page['title']} (score: {page['link'].importance_score:.2f})")
```

## Custom Thresholds

```python
from hololoom.spinningWheel.recursive_crawler import RecursiveCrawler, CrawlConfig

config = CrawlConfig(
    max_depth=3,
    max_pages=30,
    importance_thresholds={
        0: 0.0,   # Seed: always crawl
        1: 0.7,   # Depth 1: 70%+ relevant
        2: 0.85,  # Depth 2: 85%+ relevant
        3: 0.95,  # Depth 3: exceptional only
    },
)

crawler = RecursiveCrawler(config)
pages = await crawler.crawl(seed_url, seed_topic)
```

## Link Scoring

Each link is scored 0-1 based on:

**Positive signals:**
- Anchor text contains topic words (+0.3)
- Surrounding context mentions topic (+0.2)
- Same domain as seed (+0.15)
- Descriptive anchor text (+0.1)

**Negative signals:**
- Social media links (-0.5)
- Navigation links (Home, About) (-0.3)
- Commercial patterns (/cart, /checkout) (-0.2)
- Skip entirely: /login, /signup, .pdf, .zip

## Threshold Profiles

| Profile | Depth 1 | Depth 2 | Depth 3 | Use Case |
|---------|---------|---------|---------|----------|
| Lenient | 0.4 | 0.5 | 0.6 | Discovery, mapping |
| Balanced | 0.6 | 0.75 | 0.85 | General research |
| Strict | 0.75 | 0.9 | 0.95 | Curated collections |

## Configuration

```python
@dataclass
class CrawlConfig:
    max_depth: int = 2
    max_pages: int = 50
    max_pages_per_domain: int = 10
    same_domain_only: bool = False
    min_content_length: int = 200
    rate_limit_seconds: float = 1.0
    extract_images: bool = True
    max_images_per_page: int = 5
```

## Integration with Memory

```python
memory = await create_unified_memory(user_id="blake")

for page in pages:
    memories = shards_to_memories(page['shards'])
    for mem in memories:
        mem.tags = ['web-crawl', f"depth-{page['depth']}"]
        mem.metadata['importance_score'] = page['link'].importance_score
    await memory.store_many(memories)

# Query across entire crawl
results = await memory.recall("specific question about topic")
```

## Performance

Typical crawl (20 pages, depth 2, balanced thresholds):
- ~3-5 seconds per page (scrape + process)
- Total: ~60-100 seconds
- Natural funnel: 1 -> 8 -> 11 pages per depth

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Only 1 page crawled | Lower depth-1 threshold (0.4-0.5) |
| Too many pages | Raise thresholds or reduce `max_pages` |
| Missing relevant content | Specify `seed_topic` explicitly |
| Slow crawling | Reduce `max_pages` or `max_depth` |
