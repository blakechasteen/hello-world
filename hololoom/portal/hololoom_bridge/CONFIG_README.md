# HoloLoom Bridge Configuration

Complete configuration guide for the HoloLoom Bridge component.

## Overview

The HoloLoom Bridge Configuration (`config.py`) provides a Pydantic-based configuration system for managing:
- HoloLoom server connection settings
- Request timeouts and retry behavior
- Local fallback options
- Default recall and weaving parameters

## Quick Start

### Default Configuration

```python
from hololoom.portal.hololoom_bridge import BridgeConfig

# Create config with defaults
config = BridgeConfig()
```

### Load from YAML File

```python
from hololoom.portal.hololoom_bridge import load_bridge_config

# Load from default location (configs/bridge.yaml)
config = load_bridge_config()

# Or specify custom path
config = load_bridge_config("path/to/config.yaml")
```

### Environment Variable Overrides

Set environment variables to override config values:

```bash
export HOLOLOOM_URL="http://hololoom.example.com:8000"
export HOLOLOOM_TIMEOUT="60"
export HOLOLOOM_RETRY_COUNT="5"
export HOLOLOOM_LOCAL_FALLBACK="false"
export HOLOLOOM_RECALL_K="10"
export HOLOLOOM_WEAVE_MODE="fused"
```

## Configuration Parameters

### `hololoom_url: str`

**Default**: `"http://localhost:8000"`

HoloLoom server URL for all bridge operations.

```python
config = BridgeConfig(hololoom_url="http://hololoom.internal:8000")
```

**Environment Variable**: `HOLOLOOM_URL`

### `timeout_seconds: float`

**Default**: `30.0`
**Constraints**: Must be > 0

Request timeout in seconds for all HTTP operations.

```python
config = BridgeConfig(timeout_seconds=60.0)
```

**Environment Variable**: `HOLOLOOM_TIMEOUT`

### `retry_count: int`

**Default**: `3`
**Constraints**: Must be >= 0

Number of retry attempts for failed requests (exponential backoff with retry_delay).

```python
config = BridgeConfig(retry_count=5)
```

**Environment Variable**: `HOLOLOOM_RETRY_COUNT`

### `retry_delay: float`

**Default**: `1.0`
**Constraints**: Must be >= 0

Delay in seconds between retry attempts.

```python
config = BridgeConfig(retry_delay=2.0)
```

**Note**: No environment variable override (use YAML config).

### `enable_local_fallback: bool`

**Default**: `True`

If `True`, uses local HoloLoom instance if server unavailable. If `False`, fails if server unavailable.

```python
config = BridgeConfig(enable_local_fallback=False)
```

**Environment Variable**: `HOLOLOOM_LOCAL_FALLBACK` (accepts "true", "1", "yes" as True)

### `default_recall_k: int`

**Default**: `5`
**Constraints**: Must be >= 1

Default number of results returned by recall operations.

```python
config = BridgeConfig(default_recall_k=10)
```

**Environment Variable**: `HOLOLOOM_RECALL_K`

### `default_weave_mode: str`

**Default**: `"fast"`
**Valid Values**: `"bare"`, `"fast"`, `"fused"`

Default reasoning mode for weaving operations:
- **bare**: Minimal processing, fastest (~50ms)
- **fast**: Balanced (default, ~150ms)
- **fused**: Full processing, highest quality (~300ms)

```python
config = BridgeConfig(default_weave_mode="fused")
```

**Environment Variable**: `HOLOLOOM_WEAVE_MODE`

## Configuration File (YAML)

**Location**: `configs/bridge.yaml`

```yaml
# HoloLoom server connection
hololoom_url: "http://localhost:8000"

# Request handling
timeout_seconds: 30.0
retry_count: 3
retry_delay: 1.0

# Fallback behavior
enable_local_fallback: true

# Default operations
default_recall_k: 5
default_weave_mode: "fast"
```

## Priority Order

Configuration is loaded in priority order (highest first):

1. **Environment Variables** (override everything)
2. **YAML File** (specified via parameter or `HOLOLOOM_BRIDGE_CONFIG` env var)
3. **Default File** (`configs/bridge.yaml` if exists)
4. **Hardcoded Defaults** (fallback)

Example:
```python
# All three override each other in this order
os.environ['HOLOLOOM_URL'] = 'http://override.example.com'
config = load_bridge_config('custom.yaml')  # 2nd priority
# 1st priority wins -> http://override.example.com
```

## Production Recommendations

### Connection Settings

For production deployments:

```yaml
hololoom_url: "http://hololoom.internal:8000"  # Use internal DNS
timeout_seconds: 45.0                          # Longer timeout
retry_count: 5                                  # More retries
enable_local_fallback: false                   # Require server
```

### Default Modes

- **Simple queries** (FAQs, lookups): Use `"bare"` for speed
- **Standard use** (general queries): Use `"fast"` (default)
- **Research** (complex analysis): Use `"fused"`

```python
# Per-operation override
result = await bridge.recall(query, k=20, mode="fused")
```

## Testing Configuration

```python
from hololoom.portal.hololoom_bridge import load_bridge_config

# Test loading
config = load_bridge_config()
assert config.hololoom_url == "http://localhost:8000"
assert config.timeout_seconds == 30.0
assert config.retry_count == 3

# Test with environment override
import os
os.environ['HOLOLOOM_URL'] = 'http://test.local'
config = load_bridge_config()
assert config.hololoom_url == 'http://test.local'
```

## Docker/Kubernetes Integration

### Environment Variables (Docker)

```dockerfile
ENV HOLOLOOM_URL="http://hololoom-service:8000"
ENV HOLOLOOM_TIMEOUT="45"
ENV HOLOLOOM_RETRY_COUNT="5"
ENV HOLOLOOM_LOCAL_FALLBACK="false"
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hololoom-bridge-config
data:
  bridge.yaml: |
    hololoom_url: "http://hololoom-service:8000"
    timeout_seconds: 45
    retry_count: 5
    enable_local_fallback: false
    default_recall_k: 10
    default_weave_mode: "fused"
```

### Kubernetes Secrets

For sensitive data (if needed):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hololoom-bridge-secrets
type: Opaque
stringData:
  hololoom_url: "http://hololoom-service:8000"
```

## API Reference

### BridgeConfig Class

**Pydantic BaseModel** with validation:

```python
class BridgeConfig(BaseModel):
    hololoom_url: str
    timeout_seconds: float  # > 0
    retry_count: int        # >= 0
    retry_delay: float      # >= 0
    enable_local_fallback: bool
    default_recall_k: int   # >= 1
    default_weave_mode: str  # bare|fast|fused
```

### load_bridge_config Function

```python
def load_bridge_config(path: Optional[str] = None) -> BridgeConfig:
    """Load bridge configuration.

    Priority order:
    1. Environment variables
    2. YAML file (if path provided or HOLOLOOM_BRIDGE_CONFIG env var)
    3. Default configs/bridge.yaml (if exists)
    4. Hardcoded defaults

    Args:
        path: Optional path to YAML config file

    Returns:
        BridgeConfig with all overrides applied
    """
```

## Troubleshooting

### Config Not Loading

**Problem**: `FileNotFoundError: configs/bridge.yaml not found`

**Solution**:
- Ensure `configs/bridge.yaml` exists in the portal directory
- Or provide explicit path: `load_bridge_config("path/to/bridge.yaml")`
- Or use environment variables: `HOLOLOOM_URL=...`

### Environment Variables Not Working

**Problem**: Environment variable override not taking effect

**Solution**:
- Check variable name: `HOLOLOOM_URL`, `HOLOLOOM_TIMEOUT`, etc.
- Boolean values: "true", "1", "yes" are True; others are False
- Export before running: `export HOLOLOOM_URL="..."`

### Invalid Configuration Values

**Problem**: `ValidationError: timeout_seconds must be > 0`

**Solution**:
- Check field constraints:
  - `timeout_seconds`: Must be > 0
  - `retry_count`: Must be >= 0
  - `default_recall_k`: Must be >= 1
  - `default_weave_mode`: Must be "bare", "fast", or "fused"

## See Also

- [Bridge API Documentation](./README.md)
- [HoloLoom Server Documentation](../../../CLAUDE.md)
- [Portal Architecture](../README.md)
