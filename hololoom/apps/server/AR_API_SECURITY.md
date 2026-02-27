# AR API Security & Rate Limiting Documentation

**Created**: 2025-11-26
**Updated**: 2025-11-26
**Location**: `hololoom/server/`

## Overview

The HoloLoom AR API implements comprehensive security measures to protect against common attack vectors and ensure stable service availability. This document covers the security features implemented in the AR API server.

## Security Features

### 1. Rate Limiting

**Implementation**: Sliding window rate limiter with per-IP tracking

**Limits**:
- **Vision Endpoints**: 10 requests per 60 seconds per IP address
- **Other Endpoints**: No rate limiting (WebSocket connections handle their own flow control)

**Why These Limits?**
- Vision processing is computationally expensive (100-500ms per request)
- 10 requests/minute allows legitimate usage while preventing DoS attacks
- Averaging 1 request every 6 seconds is reasonable for AR applications

### 2. Input Validation

**File Upload Security**:
- **Max File Size**: 10MB per upload
- **Allowed Formats**: JPEG, PNG, WebP, GIF only
- **Content Verification**: Files are validated to ensure they're actual images
- **Memory Protection**: Files are read in chunks to prevent memory exhaustion

**Implementation Details**:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_FORMATS = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}
```

### 3. Rate Limit Headers

All vision endpoints return standard rate limit headers to help clients manage their request patterns:

```
X-RateLimit-Limit: 10        # Maximum requests per window
X-RateLimit-Remaining: 7     # Remaining requests in current window
X-RateLimit-Reset: 1701023400 # Unix timestamp when window resets
```

When rate limit is exceeded (HTTP 429), additional headers are returned:
```
Retry-After: 45  # Seconds to wait before retrying
```

## API Usage Examples

### Successful Request with Rate Limit Headers

```bash
curl -X POST http://localhost:8000/ar/vision/detect_objects \
  -F "file=@image.jpg" \
  -v

# Response Headers:
HTTP/1.1 200 OK
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1701023460
Content-Type: application/json

# Response Body:
{
  "objects": [...],
  "count": 5,
  "processing_time_ms": 125.4
}
```

### Rate Limit Exceeded Response

```bash
# After 10 requests within 60 seconds:
curl -X POST http://localhost:8000/ar/vision/detect_objects \
  -F "file=@image.jpg" \
  -v

# Response:
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1701023460
Retry-After: 42
Content-Type: application/json

{
  "detail": "Rate limit exceeded. Max 10 requests per 60s"
}
```

### File Validation Error

```bash
# Attempting to upload non-image file:
curl -X POST http://localhost:8000/ar/vision/detect_objects \
  -F "file=@document.pdf" \
  -v

# Response:
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "detail": "Invalid file format. Allowed: image/jpeg, image/jpg, image/png, image/webp, image/gif"
}
```

### File Size Error

```bash
# Attempting to upload file >10MB:
curl -X POST http://localhost:8000/ar/vision/detect_objects \
  -F "file=@large_image.jpg" \
  -v

# Response:
HTTP/1.1 413 Payload Too Large
Content-Type: application/json

{
  "detail": "File too large. Max size: 10MB"
}
```

## Client Implementation Best Practices

### JavaScript/TypeScript Example

```typescript
class ARVisionClient {
  private baseUrl = 'http://localhost:8000';
  private rateLimitRemaining = 10;
  private rateLimitReset = 0;

  async detectObjects(imageFile: File): Promise<DetectionResult> {
    // Check rate limit before sending
    if (this.rateLimitRemaining === 0) {
      const waitTime = this.rateLimitReset - Date.now() / 1000;
      if (waitTime > 0) {
        throw new Error(`Rate limit exceeded. Wait ${Math.ceil(waitTime)} seconds`);
      }
    }

    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${this.baseUrl}/ar/vision/detect_objects`, {
      method: 'POST',
      body: formData
    });

    // Update rate limit info from headers
    this.rateLimitRemaining = parseInt(response.headers.get('X-RateLimit-Remaining') || '10');
    this.rateLimitReset = parseInt(response.headers.get('X-RateLimit-Reset') || '0');

    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
      throw new Error(`Rate limit exceeded. Retry after ${retryAfter} seconds`);
    }

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Vision processing failed');
    }

    return await response.json();
  }
}
```

### Python Example

```python
import time
import requests
from typing import Optional

class ARVisionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.rate_limit_remaining = 10
        self.rate_limit_reset = 0

    def detect_objects(self, image_path: str) -> dict:
        """
        Detect objects with automatic rate limit handling.
        """
        # Check if we need to wait for rate limit reset
        if self.rate_limit_remaining == 0:
            wait_time = self.rate_limit_reset - time.time()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer

        # Prepare request
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/ar/vision/detect_objects",
                files=files
            )

        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 10))
        self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))

        # Handle rate limit error
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise Exception(f"Rate limit exceeded. Retry after {retry_after} seconds")

        # Handle other errors
        response.raise_for_status()

        return response.json()
```

## Vision Endpoints with Rate Limiting

All 8 vision endpoints implement the same rate limiting and security measures:

### Phase 2 Endpoints
1. **POST /ar/vision/detect_objects** - Object detection (YOLO/COCO-SSD)
2. **POST /ar/vision/analyze_scene** - Scene understanding
3. **POST /ar/vision/track_hands** - Hand tracking (MediaPipe)

### Phase 4 Endpoints
4. **POST /ar/vision/estimate_depth** - Depth estimation (MiDaS)
5. **POST /ar/vision/detect_markers** - ArUco/QR code detection

### Phase 5 Endpoints
6. **POST /ar/vision/segment_image** - Semantic segmentation (DeepLabV3)
7. **POST /ar/vision/estimate_pose** - Full-body pose estimation
8. **POST /ar/vision/track_camera** - SLAM camera tracking

## Security Recommendations

### For Production Deployment

1. **Use HTTPS**: Always deploy with TLS certificates in production
2. **API Keys**: Implement API key authentication for additional security
3. **CORS Configuration**: Restrict allowed origins to your AR client domains
4. **Monitoring**: Set up logging and alerting for rate limit violations
5. **DDoS Protection**: Consider using a CDN or cloud-based DDoS protection
6. **Resource Limits**: Set container/pod resource limits in Kubernetes
7. **Network Policies**: Restrict network access to necessary services only

### Rate Limit Tuning

The default rate limits (10 req/60s) can be adjusted based on your needs:

```python
# In ar_api.py, modify the rate limiter initialization:
vision_rate_limiter = RateLimiter(
    max_requests=20,      # Increase for higher throughput
    window_seconds=60     # Or use shorter windows (e.g., 30s)
)
```

Consider factors:
- **GPU/CPU capacity**: More powerful hardware can handle higher rates
- **Model complexity**: Simpler models (MobileNet) allow higher throughput
- **User patterns**: AR apps may burst during scanning phases
- **Cost constraints**: Cloud GPU costs scale with usage

## Monitoring & Alerting

### Prometheus Metrics (Future Enhancement)

```python
# Suggested metrics to track:
ar_vision_requests_total{endpoint, status}
ar_vision_rate_limit_rejections_total{endpoint}
ar_vision_processing_duration_seconds{endpoint}
ar_vision_file_size_bytes{endpoint}
```

### Logging

All security events are logged:
- Rate limit violations (WARNING level)
- Invalid file uploads (WARNING level)
- Processing errors (ERROR level)

Example log entries:
```
2025-11-26 10:30:45 WARNING Rate limit exceeded for IP 192.168.1.100
2025-11-26 10:31:12 WARNING Invalid file format uploaded: application/pdf
2025-11-26 10:32:01 ERROR Object detection failed: Model initialization error
```

## Testing Rate Limits

### Manual Testing Script

```bash
#!/bin/bash
# Test rate limiting behavior

echo "Testing AR Vision API rate limits..."

# Send 12 requests rapidly (should hit limit at 11th)
for i in {1..12}; do
  echo "Request $i:"
  curl -X POST http://localhost:8000/ar/vision/detect_objects \
    -F "file=@test_image.jpg" \
    -w "\nStatus: %{http_code}\n" \
    -H "X-Request-ID: test-$i" \
    -s | head -n 1
  sleep 1
done
```

### Automated Testing

```python
import pytest
import asyncio
from hololoom.server.ar_api import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter():
    """Test rate limiting behavior"""
    limiter = RateLimiter(max_requests=3, window_seconds=10)

    # Mock request object
    class MockRequest:
        class Client:
            host = "127.0.0.1"
        client = Client()

    request = MockRequest()

    # First 3 requests should succeed
    for i in range(3):
        info = await limiter.check_rate_limit(request)
        assert info["remaining"] == 3 - i - 1

    # 4th request should fail
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_rate_limit(request)

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
```

## Troubleshooting

### Common Issues

1. **"Rate limit exceeded" errors in development**
   - Solution: Increase limits for development environment
   - Or: Add IP whitelist for localhost/development IPs

2. **Legitimate users hitting rate limits**
   - Solution: Implement user-based (authenticated) rate limiting
   - Or: Increase limits based on usage patterns

3. **Memory issues with large files**
   - Current limit (10MB) should prevent most issues
   - For larger files, implement streaming uploads

4. **Slow vision processing causing timeouts**
   - Solution: Implement request queuing with background processing
   - Return job ID immediately, poll for results

## Future Enhancements

### Planned Security Features

1. **JWT Authentication** - User-specific rate limits
2. **Request Signing** - Prevent replay attacks
3. **IP Reputation** - Block known malicious IPs
4. **Adaptive Rate Limiting** - Adjust based on system load
5. **Request Queuing** - Queue requests when at capacity
6. **WebSocket Rate Limiting** - Per-message rate limits
7. **Content Security Policy** - For web-based AR clients
8. **Audit Logging** - Comprehensive security event logging

### Planned Monitoring

1. **Grafana Dashboards** - Real-time visualization
2. **AlertManager** - Automated alerting
3. **ELK Stack Integration** - Centralized logging
4. **Distributed Tracing** - End-to-end request tracking

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Rate Limiting Best Practices](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Image Upload Security](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)