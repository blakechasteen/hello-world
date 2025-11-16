# HoloLoom Dashboard Pagination API - Quick Reference

## Endpoints with Pagination Support

### 1. Recent Executions
```
GET /api/v1/executions/recent
```

**Parameters**:
```
skip: int = 0      # Items to skip (offset)
limit: int = 20    # Items per page (max: 100)
```

**Example Requests**:
```bash
# Get first 20 executions (default)
curl http://localhost:8000/api/v1/executions/recent

# Get specific page
curl "http://localhost:8000/api/v1/executions/recent?skip=20&limit=10"

# Get last 5 executions
curl "http://localhost:8000/api/v1/executions/recent?skip=0&limit=5"

# Custom page size
curl "http://localhost:8000/api/v1/executions/recent?limit=50"
```

**Response**:
```json
{
  "executions": [ /* 20 records */ ],
  "pagination": {
    "skip": 0,
    "limit": 20,
    "total": 150,
    "count": 20,
    "has_more": true,
    "next_skip": 20
  }
}
```

---

### 2. Analytics Trends
```
GET /api/v1/analytics/trends
```

**Parameters**:
```
days: int = 7      # Days of history
skip: int = 0      # Items to skip (offset)
limit: int = 50    # Items per page (max: 100)
```

**Example Requests**:
```bash
# Get last 7 days of trends
curl http://localhost:8000/api/v1/analytics/trends

# Get 30 days, paginated
curl "http://localhost:8000/api/v1/analytics/trends?days=30&skip=0&limit=50"

# Get second page of 2-year trends
curl "http://localhost:8000/api/v1/analytics/trends?days=730&skip=50&limit=50"
```

**Response**:
```json
{
  "trends": [
    {
      "date": "2025-11-16",
      "avg_quality_gain": 0.125,
      "avg_final_quality": 0.856,
      "count": 8
    }
  ],
  "pagination": {
    "skip": 0,
    "limit": 50,
    "total": 7,
    "count": 7,
    "has_more": false,
    "next_skip": null
  }
}
```

---

## Client Implementation Examples

### JavaScript/Node.js
```javascript
class HoloLoomPaginatedClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async getExecutions(skip = 0, limit = 20) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/executions/recent?skip=${skip}&limit=${limit}`
    );
    return response.json();
  }

  async getExecutionsPageByPage(pageSize = 20, callback) {
    let skip = 0;
    let hasMore = true;

    while (hasMore) {
      const result = await this.getExecutions(skip, pageSize);
      callback(result.executions);
      hasMore = result.pagination.has_more;
      skip = result.pagination.next_skip;
    }
  }

  async getNextPage(pagination) {
    if (!pagination.has_more) {
      return null;
    }
    return this.getExecutions(pagination.next_skip, pagination.limit);
  }
}

// Usage
const client = new HoloLoomPaginatedClient();

// Get first page
const page1 = await client.getExecutions(0, 10);
console.log(`Got ${page1.pagination.count} items out of ${page1.pagination.total}`);

// Get next page
if (page1.pagination.has_more) {
  const page2 = await client.getExecutions(page1.pagination.next_skip, 10);
}

// Fetch all pages
await client.getExecutionsPageByPage(20, (executions) => {
  console.log(`Processing ${executions.length} items...`);
});
```

### Python
```python
import requests
from typing import List, Dict, Optional

class HoloLoomPaginatedClient:
    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url

    def get_executions(self, skip: int = 0, limit: int = 20) -> Dict:
        """Get paginated executions."""
        response = requests.get(
            f"{self.base_url}/api/v1/executions/recent",
            params={"skip": skip, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_all_executions(self, limit: int = 20) -> List[Dict]:
        """Get all executions, automatically handling pagination."""
        all_executions = []
        skip = 0

        while True:
            result = self.get_executions(skip, limit)
            all_executions.extend(result["executions"])

            if not result["pagination"]["has_more"]:
                break

            skip = result["pagination"]["next_skip"]

        return all_executions

    def get_trends(self, days: int = 7, skip: int = 0, limit: int = 50) -> Dict:
        """Get paginated trends."""
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/trends",
            params={"days": days, "skip": skip, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = HoloLoomPaginatedClient()

# Get single page
result = client.get_executions(skip=0, limit=10)
print(f"Page has {result['pagination']['count']} items")
print(f"Total items: {result['pagination']['total']}")

# Get all executions
all_execs = client.get_all_executions()
print(f"Downloaded {len(all_execs)} executions")

# Get trends
trends = client.get_trends(days=30, limit=100)
```

---

## Error Handling

### Invalid Parameters
```bash
# skip < 0 (returns 400)
curl "http://localhost:8000/api/v1/executions/recent?skip=-1"
# Response: {"detail": "skip must be >= 0"}

# limit < 1 (returns 400)
curl "http://localhost:8000/api/v1/executions/recent?limit=0"
# Response: {"detail": "limit must be >= 1"}
```

### Rate Limiting
```bash
# Too many requests (returns 429)
# After rate limit exceeded
curl http://localhost:8000/api/v1/executions/recent
# Response: 429 Too Many Requests
```

### Python Error Handling
```python
import requests
from requests.exceptions import RequestException

try:
    result = client.get_executions(skip=-1, limit=10)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        print(f"Invalid parameters: {e.response.json()['detail']}")
    elif e.response.status_code == 429:
        print("Rate limit exceeded, retrying in 60 seconds...")
```

---

## Pagination Patterns

### Pattern 1: Get Specific Page
```python
# Get page 3, 10 items per page
page_num = 3
items_per_page = 10
skip = (page_num - 1) * items_per_page

result = client.get_executions(skip=skip, limit=items_per_page)
```

### Pattern 2: Infinite Scroll
```javascript
async function infiniteScroll() {
  let skip = 0;
  const limit = 20;

  async function loadMore() {
    const result = await client.getExecutions(skip, limit);
    displayItems(result.executions);

    if (result.pagination.has_more) {
      skip = result.pagination.next_skip;
      return true; // More items available
    }
    return false; // All items loaded
  }

  // Keep loading until done
  while (await loadMore()) {
    // Continue loading
  }
}
```

### Pattern 3: Cached Pagination
```python
class CachedPaginationClient:
    def __init__(self, client):
        self.client = client
        self.cache = {}

    def get_page(self, page_num: int, limit: int = 20):
        """Get page with caching."""
        key = f"page_{page_num}_limit_{limit}"

        if key not in self.cache:
            skip = (page_num - 1) * limit
            self.cache[key] = self.client.get_executions(skip, limit)

        return self.cache[key]

    def clear_cache(self):
        self.cache.clear()

# Usage
cached_client = CachedPaginationClient(client)
page1 = cached_client.get_page(1, 20)
page2 = cached_client.get_page(2, 20)
page1_again = cached_client.get_page(1, 20)  # Returns cached result
```

---

## Performance Tips

### 1. Use Appropriate Page Sizes
```python
# Good: 20-50 items per page
result = client.get_executions(limit=30)

# Not optimal: Too small (many requests)
result = client.get_executions(limit=1)

# Not optimal: Too large (large payloads)
result = client.get_executions(limit=500)  # Capped at 100
```

### 2. Batch Operations
```python
# Better: Process pages
for skip in range(0, total, limit):
    result = client.get_executions(skip=skip, limit=50)
    process_batch(result['executions'])

# Not optimal: Fetch all at once
all_items = client.get_all_executions()  # Memory intensive
```

### 3. Avoid Large Offsets
```python
# Current implementation uses SQL OFFSET
# Offset becomes slow with very large skip values
# Good: skip < 10,000
result = client.get_executions(skip=5000, limit=50)

# Not optimal: skip > 100,000 (slow SQL query)
result = client.get_executions(skip=500000, limit=50)
```

---

## Pagination Metadata Interpretation

| Scenario | Pagination Metadata | What It Means |
|----------|-------------------|---------------|
| First page | `has_more: true`, `next_skip: 20` | More items available, go to page 2 |
| Middle page | `has_more: true`, `next_skip: 40` | More items available, continue |
| Last page | `has_more: false`, `next_skip: null` | No more items, stop pagination |
| Empty result | `count: 0`, `total: 0` | No items match criteria |
| Single page | `has_more: false`, `count: <total>` | All items fit on one page |

---

## Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Valid pagination request |
| 400 | Bad Request | Invalid skip/limit parameters |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Database error |

---

## Version History

- **2025-11-16**: Initial pagination implementation
  - Added `skip` parameter to `/api/v1/executions/recent`
  - Added `limit` capping at 100 items
  - Added pagination metadata to responses
  - Added pagination to `/api/v1/analytics/trends`

---

**Last Updated**: 2025-11-16
