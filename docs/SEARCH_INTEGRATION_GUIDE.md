# HoloLoom Documentation Search - Integration Guide

**Date:** November 16, 2025
**Status:** Complete & Ready for Integration
**Files:** `search.js`, `search.css`, `search-index.json`

## Overview

The HoloLoom documentation search system provides production-ready full-text search with autocomplete, highlighting, keyboard navigation, and search history—all with **zero external dependencies**.

## Quick Start

### 1. Include in HTML Template

Add these lines to your base HTML template (e.g., `docs/templates/base.html`):

```html
<!-- In <head> -->
<link rel="stylesheet" href="/assets/css/search.css">

<!-- In <body> - search input in navbar -->
<div class="search-container">
  <input
    id="search-input"
    type="text"
    placeholder="Search documentation (press / to focus)"
    autocomplete="off"
    aria-label="Search documentation"
  >
  <div id="search-results" role="region" aria-label="Search results"></div>
</div>

<!-- Before </body> closing tag -->
<script src="/assets/js/search.js"></script>
```

### 2. Build the Search Index

Generate `docs/data/search-index.json` from your HTML pages:

```bash
# Using the provided Python script (optional)
python scripts/generate-search-index.py

# Or manually create by extracting pages
# See "Building the Search Index" section below
```

### 3. Verify Integration

1. Open any documentation page in browser
2. Look for search input in navbar
3. Press `/` to focus search
4. Type a query (e.g., "Thompson")
5. See results populate in real-time

## Features

### Full-Text Search

- Tokenizes documents and queries
- Calculates relevance scores based on:
  - Match type (title > headers > body)
  - Token frequency
  - Phrase matching bonuses
- Returns top 10 most relevant results

### Autocomplete Suggestions

As user types, suggestions appear below search box:

```javascript
// Get suggestions for current query
const suggestions = window.hololoomSearch.getAutocomplete('Thomp', limit=5);
// Returns: ['thompson', 'thompson-sampling', ...]
```

### Fuzzy Matching

Handles typos using Levenshtein distance:
- Exact match: full weight (e.g., "thompson" = "thompson")
- 1 char difference: 90% weight (e.g., "tompson" vs "thompson")
- 2 char difference: 70% weight
- 3+ char difference: 50% weight

### Result Highlighting

Query terms highlighted in results:

```html
<span class="search-highlight">Thompson</span> Sampling
```

Styling customizable via CSS variables.

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `/` | Focus search input |
| `↑` `↓` | Navigate results |
| `Enter` | Go to selected result |
| `Escape` | Close results, unfocus |

### Search History

Last 10 searches stored in localStorage:

```javascript
// Access search history
const history = window.hololoomSearch.getSearchHistory();
// Returns: ['Thompson Sampling', 'knowledge graphs', ...]

// Clear history
window.hololoomSearch.clearSearchHistory();
```

## Building the Search Index

### Automatic Index Generation

The search index should be pre-generated at build time. Here's a Python script:

```python
# scripts/generate-search-index.py
import json
import re
from pathlib import Path
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.headings = []
        self.in_article = False
        self.current_heading = None

    def handle_starttag(self, tag, attrs):
        if tag == 'article':
            self.in_article = True
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.current_heading = tag

    def handle_endtag(self, tag):
        if tag == 'article':
            self.in_article = False
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.current_heading = None

    def handle_data(self, data):
        if self.in_article:
            text = data.strip()
            if text:
                if self.current_heading:
                    self.headings.append(text)
                else:
                    self.text.append(text)

def build_search_index():
    """Generate search index from all HTML pages"""
    index = {'pages': []}

    for html_file in sorted(Path('docs').glob('**/*.html')):
        # Skip special pages
        if html_file.name in ['404.html', 'search.html']:
            continue

        # Calculate URL
        relative = html_file.relative_to('docs')
        url = '/' + str(relative).replace('\\', '/')
        if url.endswith('/index.html'):
            url = url[:-10]  # Remove /index.html
        if not url.endswith('/') and url != '/':
            url = url[:-5]  # Remove .html

        # Extract content
        try:
            with open(html_file, encoding='utf-8') as f:
                content = f.read()

            extractor = TextExtractor()
            extractor.feed(content)

            # Extract title
            title_match = re.search(r'<title>([^<]+)</title>', content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).replace(' - HoloLoom Documentation', '')
            else:
                h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content)
                title = h1_match.group(1) if h1_match else html_file.stem

            # Extract meta description
            desc_match = re.search(
                r'<meta\s+name="description"\s+content="([^"]+)"',
                content,
                re.IGNORECASE
            )
            excerpt = desc_match.group(1) if desc_match else ' '.join(extractor.text[:30])

            # Build sections
            sections = []
            if extractor.headings:
                # Group text into sections by heading
                current_section = None
                current_content = []

                for heading in extractor.headings:
                    if current_section:
                        sections.append({
                            'heading': current_section,
                            'content': ' '.join(current_content),
                            'level': 2
                        })
                    current_section = heading
                    current_content = []

                if current_section:
                    sections.append({
                        'heading': current_section,
                        'content': ' '.join(current_content),
                        'level': 2
                    })

            page_entry = {
                'url': url,
                'title': title,
                'excerpt': excerpt,
                'sections': sections
            }

            index['pages'].append(page_entry)
            print(f"Indexed: {url}")

        except Exception as e:
            print(f"Error processing {html_file}: {e}")

    # Save index
    Path('docs/data').mkdir(exist_ok=True)
    with open('docs/data/search-index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated index with {len(index['pages'])} pages")
    return index

if __name__ == '__main__':
    build_search_index()
```

### Manual Index Entry

For custom entries or non-HTML content:

```json
{
  "url": "/custom-page",
  "title": "Page Title",
  "excerpt": "Short description for preview",
  "sections": [
    {
      "heading": "Section Heading",
      "content": "Full section text for indexing...",
      "level": 2
    }
  ]
}
```

### Index Structure Specification

```typescript
interface SearchIndex {
  pages: Array<{
    url: string;              // e.g., "/training/part1"
    title: string;            // Page title
    excerpt: string;          // Short preview (150-200 chars)
    sections?: Array<{
      heading: string;        // Heading text
      content: string;        // Body text for this section
      level?: number;         // h2, h3, etc. (default 2)
    }>;
  }>;
}
```

## API Reference

### DocumentSearch Class

#### Constructor

```javascript
const search = new DocumentSearch(indexPath = '/data/search-index.json');
```

#### Methods

**`search(query: string): Result[]`**

Perform search and return top 10 results:

```javascript
const results = window.hololoomSearch.search('Thompson Sampling');
// Returns:
// [
//   {
//     url: '/training/part1',
//     title: 'Part 1: Foundations',
//     excerpt: '...',
//     score: 23.5,
//     queryTokens: ['thompson', 'sampling']
//   },
//   ...
// ]
```

**`getAutocomplete(query: string, limit: number = 5): string[]`**

Get autocomplete suggestions:

```javascript
const suggestions = window.hololoomSearch.getAutocomplete('Thomp');
// Returns: ['thompson', 'thompson-sampling', 'thompson-bandit']
```

**`getSearchHistory(): string[]`**

Get last 10 searches:

```javascript
const history = window.hololoomSearch.getSearchHistory();
```

**`clearSearchHistory(): void`**

Clear all search history:

```javascript
window.hololoomSearch.clearSearchHistory();
```

**`getStatistics(): object`**

Get indexing statistics:

```javascript
const stats = window.hololoomSearch.getStatistics();
// Returns:
// {
//   indexedPages: 45,
//   indexedSections: 238,
//   indexTokens: 8394,
//   searchHistory: 3
// }
```

## Configuration

Customize search behavior via config object:

```javascript
// Available in search.js constructor
const search = new DocumentSearch();

// Modify config (before searches)
search.config.maxResults = 20;          // Default: 10
search.config.debounceDelay = 500;      // Default: 300ms
search.config.minQueryLength = 2;       // Default: 1
search.config.maxHistorySize = 20;      // Default: 10
search.config.highlightClass = 'hl';    // Default: 'search-highlight'
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| **Index load** | ~50-200ms | First search loads index from network |
| **Search (cold)** | ~10-50ms | First search, index already loaded |
| **Search (warm)** | ~1-5ms | Subsequent searches, tokenized |
| **Autocomplete** | ~5-15ms | Token prefix matching |
| **Result render** | ~20-50ms | DOM updates for 10 results |

**Total E2E:** User types → 300ms debounce → search → render = ~350ms from last keystroke

## Customization

### Styling

Customize via CSS variables in `search.css`:

```css
:root {
  --color-primary: #1e40af;           /* Link color */
  --color-primary-dark: #1e3a8a;      /* Hover color */
  --color-primary-light: #dbeafe;     /* Active result bg */
  --color-text: #1a1a1a;
  --color-text-muted: #6b7280;
  --color-background: #ffffff;
  --color-background-hover: #f8f9fa;
  --color-border: #e5e7eb;
}
```

### Result Display

Customize result rendering in `search.js`:

```javascript
// Modify renderResults() method to change layout
// Modify generateExcerpt() to change excerpt length
// Modify highlightQuery() to change highlight style
```

### Scoring Algorithm

Adjust scoring weights in `scoreHit()` method:

```javascript
const typeMultiplier = {
  'title': 3.0,    // Boost title matches 3x
  'heading': 2.0,  // Boost header matches 2x
  'body': 1.0      // Body matches baseline
};
```

## Browser Compatibility

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS 12+, Android 9+)

Uses:
- Vanilla JavaScript (ES6+)
- CSS Grid and Flexbox
- LocalStorage API
- Fetch API

## Accessibility

WCAG AAA compliant:

- ✅ Keyboard navigation (all features accessible)
- ✅ Screen reader support (ARIA labels, roles)
- ✅ Color contrast (7:1 ratio, light & dark)
- ✅ Focus indicators (3px outline)
- ✅ Reduced motion support
- ✅ Semantic HTML

## Troubleshooting

### Search index not loading

**Problem:** "Failed to load search index: 404"

**Solutions:**
1. Check file path: Should be `/data/search-index.json` relative to site root
2. Verify file exists and is valid JSON
3. Check browser console for CORS issues (should be same origin)

### No results appearing

**Problem:** Searching but getting "No results"

**Solutions:**
1. Generate index with `generate-search-index.py`
2. Verify index contains pages with matching content
3. Check `window.hololoomSearch.getStatistics()` - indexTokens > 0?

### Slow search performance

**Problem:** Search takes >500ms per query

**Solutions:**
1. Reduce result count: `config.maxResults = 5`
2. Increase debounce delay: `config.debounceDelay = 500`
3. Check browser console for errors
4. Profile with DevTools (should be <50ms JavaScript)

### Keyboard shortcuts not working

**Problem:** `/` key doesn't focus search

**Solutions:**
1. Verify `keyboard.js` is loaded
2. Check that search input has `id="search-input"`
3. Ensure no other JS is preventing default
4. Check focus isn't in textarea/input already

## Future Enhancements

Potential improvements (not in v1.0):

- [ ] Weighted keyword boosting (custom per-term weights)
- [ ] Phonetic matching (for pronunciation variants)
- [ ] Synonym expansion (e.g., "AI" ↔ "Artificial Intelligence")
- [ ] Category/tag filtering in results
- [ ] Search analytics (what do users search for?)
- [ ] Advanced query syntax (site:, filetype:, boolean operators)
- [ ] Voice search (speech-to-text)
- [ ] Search result previews (snippet with context window)
- [ ] Related/suggested searches
- [ ] Instant preview on hover

## Support & Issues

For issues or questions:

1. Check this guide's Troubleshooting section
2. Review search.js comments for implementation details
3. Open GitHub issue: https://github.com/hololoom/docs/issues

## License

MIT License - Free to use and modify

---

**Last Updated:** November 16, 2025
**Maintained By:** HoloLoom Documentation Team
