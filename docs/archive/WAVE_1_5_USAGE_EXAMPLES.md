# Wave 1.5 API Endpoints - Usage Examples

## Quick Start

The workflow executor server provides 3 new ingestion endpoints for Wave 1.5:

### 1. File Upload (`POST /api/ingest/file`)
### 2. URL Ingestion (`POST /api/ingest/url`)
### 3. Entity Details (`GET /api/memory/entity/{entity_id}`)

---

## Examples by Language

### cURL (Command Line)

**File Upload**:
```bash
# Upload a PDF document
curl -X POST http://localhost:8001/api/ingest/file \
  -F "file=@research_paper.pdf"

# Expected response:
{
  "success": true,
  "job_id": "a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6",
  "filename": "research_paper.pdf",
  "shards_created": 12,
  "content_type": "application/pdf",
  "file_size": 256789,
  "warning": null
}
```

**URL Ingestion**:
```bash
# Ingest content from a URL
curl -X POST http://localhost:8001/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# With options:
curl -X POST http://localhost:8001/api/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/user/repo",
    "options": {
      "chunk_size": 2000,
      "languages": ["en"],
      "max_depth": 3
    }
  }'

# Expected response:
{
  "success": true,
  "job_id": "b2c3d4e5-f6g7-48h9-i0j1-k2l3m4n5o6p7",
  "url": "https://example.com/article",
  "shards_created": 8,
  "options_applied": {
    "chunk_size": 2000,
    "languages": ["en"],
    "max_depth": 3
  },
  "warning": null
}
```

**Entity Query**:
```bash
# Query entity details
curl http://localhost:8001/api/memory/entity/thompson_sampling

# Expected response:
{
  "success": true,
  "entity": {
    "id": "thompson_sampling",
    "exists": false,
    "relationships": [],
    "relationship_count": 0
  },
  "relationships": [],
  "relationship_count": 0,
  "note": "Entity data would be populated from memory backend if executor persisted across requests"
}
```

---

### Python (requests)

```python
import requests
import json

BASE_URL = "http://localhost:8001"

# 1. File Upload
print("=== FILE UPLOAD ===")
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/api/ingest/file", files=files)
    result = response.json()
    print(f"Job ID: {result['job_id']}")
    print(f"Shards created: {result['shards_created']}")
    if result['warning']:
        print(f"Warning: {result['warning']}")

# 2. URL Ingestion
print("\n=== URL INGESTION ===")
payload = {
    "url": "https://www.python.org/doc",
    "options": {
        "chunk_size": 1000,
        "languages": ["en"]
    }
}
response = requests.post(
    f"{BASE_URL}/api/ingest/url",
    json=payload
)
result = response.json()
print(f"Job ID: {result['job_id']}")
print(f"Shards created: {result['shards_created']}")
print(f"Options applied: {json.dumps(result['options_applied'], indent=2)}")

# 3. Entity Query
print("\n=== ENTITY QUERY ===")
entity_id = "thompson_sampling"
response = requests.get(f"{BASE_URL}/api/memory/entity/{entity_id}")
result = response.json()
entity = result['entity']
print(f"Entity: {entity['id']}")
print(f"Exists: {entity['exists']}")
print(f"Relationships: {entity['relationship_count']}")
```

---

### JavaScript/TypeScript (fetch API)

```javascript
const BASE_URL = "http://localhost:8001";

// 1. File Upload
async function uploadFile(file) {
    console.log("=== FILE UPLOAD ===");
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${BASE_URL}/api/ingest/file`, {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    console.log(`Job ID: ${result.job_id}`);
    console.log(`Shards created: ${result.shards_created}`);
    if (result.warning) {
        console.warn(`Warning: ${result.warning}`);
    }
    return result;
}

// 2. URL Ingestion
async function ingestUrl(url, options = {}) {
    console.log("=== URL INGESTION ===");
    const payload = {
        url: url,
        options: options
    };

    const response = await fetch(`${BASE_URL}/api/ingest/url`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });
    const result = await response.json();
    console.log(`Job ID: ${result.job_id}`);
    console.log(`Shards created: ${result.shards_created}`);
    console.log(`Options applied:`, result.options_applied);
    return result;
}

// 3. Entity Query
async function getEntity(entityId) {
    console.log("=== ENTITY QUERY ===");
    const response = await fetch(
        `${BASE_URL}/api/memory/entity/${entityId}`
    );
    const result = await response.json();
    const entity = result.entity;
    console.log(`Entity: ${entity.id}`);
    console.log(`Exists: ${entity.exists}`);
    console.log(`Relationships: ${entity.relationship_count}`);
    return result;
}

// Usage examples:
// uploadFile(fileInput.files[0]);
// ingestUrl("https://example.com", { chunk_size: 1000 });
// getEntity("thompson_sampling");
```

---

### React Component Example

```jsx
import React, { useState } from 'react';

function IngestionPanel() {
    const [file, setFile] = useState(null);
    const [url, setUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);

    const handleFileUpload = async (e) => {
        const selectedFile = e.target.files[0];
        if (!selectedFile) return;

        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(
                'http://localhost:8001/api/ingest/file',
                { method: 'POST', body: formData }
            );
            const data = await response.json();
            setResults(data);
        } catch (error) {
            setResults({ error: error.message });
        } finally {
            setLoading(false);
        }
    };

    const handleUrlIngestion = async () => {
        if (!url) return;

        setLoading(true);
        try {
            const response = await fetch(
                'http://localhost:8001/api/ingest/url',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: url,
                        options: { chunk_size: 1000 }
                    })
                }
            );
            const data = await response.json();
            setResults(data);
        } catch (error) {
            setResults({ error: error.message });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Document Ingestion</h2>

            <section>
                <h3>Upload File</h3>
                <input
                    type="file"
                    onChange={handleFileUpload}
                    disabled={loading}
                />
            </section>

            <section>
                <h3>Ingest URL</h3>
                <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com"
                    disabled={loading}
                />
                <button onClick={handleUrlIngestion} disabled={loading}>
                    {loading ? 'Processing...' : 'Ingest'}
                </button>
            </section>

            {results && (
                <section>
                    <h3>Results</h3>
                    <pre>{JSON.stringify(results, null, 2)}</pre>
                </section>
            )}
        </div>
    );
}

export default IngestionPanel;
```

---

## Error Handling Examples

### When SpinningWheel is not installed

**File Upload Response** (graceful degradation):
```json
{
  "success": true,
  "job_id": "c3d4e5f6-g7h8-49i0-j1k2-l3m4n5o6p7q8",
  "filename": "document.pdf",
  "shards_created": 0,
  "content_type": "application/pdf",
  "file_size": 256789,
  "warning": "SpinningWheel not available"
}
```

**URL Ingestion Response** (graceful degradation):
```json
{
  "success": true,
  "job_id": "d4e5f6g7-h8i9-4a0b-c1d2-e3f4g5h6i7j8",
  "url": "https://example.com/article",
  "shards_created": 0,
  "options_applied": {},
  "warning": "SpinningWheel not available"
}
```

### When SpinningWheel processing fails

**File Upload Response** (with error detail):
```json
{
  "success": true,
  "job_id": "e5f6g7h8-i9j0-4b1c-d2e3-f4g5h6i7j8k9",
  "filename": "corrupted.pdf",
  "shards_created": 0,
  "content_type": "application/pdf",
  "file_size": 12345,
  "warning": "PDF parsing failed: invalid header"
}
```

### Critical Error (file read fails)

**HTTP Response**: 500 Internal Server Error
```json
{
  "detail": "File ingestion failed: permission denied"
}
```

---

## Integration Workflow

### Typical Multi-Step Workflow

```python
import requests
import time

BASE_URL = "http://localhost:8001"

# Step 1: Ingest multiple sources
print("Step 1: Ingesting data sources...")

# Source 1: Document file
with open("whitepaper.pdf", "rb") as f:
    files = {"file": f}
    result1 = requests.post(f"{BASE_URL}/api/ingest/file", files=files).json()
    job1 = result1['job_id']
    print(f"  ✓ File ingested: {result1['shards_created']} shards (Job: {job1[:8]}...)")

# Source 2: Web article
result2 = requests.post(f"{BASE_URL}/api/ingest/url", json={
    "url": "https://example.com/research",
    "options": {"chunk_size": 2000}
}).json()
job2 = result2['job_id']
print(f"  ✓ URL ingested: {result2['shards_created']} shards (Job: {job2[:8]}...)")

# Source 3: Another URL
result3 = requests.post(f"{BASE_URL}/api/ingest/url", json={
    "url": "https://github.com/example/repo",
    "options": {"max_depth": 2}
}).json()
job3 = result3['job_id']
print(f"  ✓ URL ingested: {result3['shards_created']} shards (Job: {job3[:8]}...)")

# Step 2: Query entities from ingested content
print("\nStep 2: Querying entities...")

entity_ids = [
    "thompson_sampling",
    "bayesian_inference",
    "gradient_descent"
]

for entity_id in entity_ids:
    result = requests.get(f"{BASE_URL}/api/memory/entity/{entity_id}").json()
    entity = result['entity']
    print(f"  • {entity['id']}: exists={entity['exists']}, "
          f"relationships={entity['relationship_count']}")

print("\nWorkflow complete!")
```

---

## Performance Tips

1. **Batch Ingestion**: Process multiple files in parallel
2. **Caching Job IDs**: Store job IDs to poll for completion
3. **Error Recovery**: Check warning field before retrying
4. **Large Files**: Consider chunking before upload
5. **URL Options**: Adjust chunk_size based on content type

---

## Debugging

**Check API Health**:
```bash
curl http://localhost:8001/health
```

**Check Available Agents**:
```bash
curl http://localhost:8001/api/agents
```

**View Server Logs**:
```bash
# Watch STDOUT while running
python HoloLoom/web_dashboard/workflow_executor.py

# Or check logs if running via systemd/docker
journalctl -u hololoom-executor -f
```

---

## Next Steps

1. ✅ Test file uploads with different formats (PDF, DOCX, TXT)
2. ✅ Test URL ingestion with various websites
3. ✅ Implement persistent executor for entity queries
4. ✅ Add job polling endpoint (`GET /api/jobs/{job_id}`)
5. ✅ Implement memory storage of ingested shards
