# Quick Start: Context Learning

Get started with learning-enabled routing in 3 simple steps.

## Minimal Example (Copy-Paste Ready)

```python
import asyncio
from HoloLoom.infrastructure.sql import SQLConfig, create_sql_backend, load_mock_data
from HoloLoom.infrastructure.mcp import create_mcp_server, generate_session_id
from HoloLoom.context import create_query_router


async def main():
    # 1. Setup backend
    sql_config = SQLConfig(sqlite_path="./data/my_app.db")
    mcp_server = await create_mcp_server(sql_config)
    await load_mock_data(mcp_server.sql_backend)  # Load sample data

    # 2. Create router with learning enabled
    router = await create_query_router(
        mcp_server,
        generate_session_id(),
        enable_learning=True,
        enable_calibration=True,
        enable_strategy_updates=True
    )

    # 3. Process queries - learning happens automatically!
    result = await router.route("What is the Varroa treatment policy?")

    print(f"Pattern: {result.pattern.value}")
    print(f"Backends: {', '.join(result.backends_used)}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Rows: {result.row_count}")
    print(f"Total events: {router.learning_tracker.total_events}")


if __name__ == "__main__":
    asyncio.run(main())
```

**That's it!** The system learns from every query.

---

## What Each Flag Does

```python
router = await create_query_router(
    mcp_server,
    session_id,
    enable_learning=True,        # Track all routing decisions
    enable_calibration=True,     # Adjust confidence predictions
    enable_strategy_updates=True # Adapt backend weights
)
```

- **`enable_learning`**: Records every routing decision (backend, confidence, latency, etc.)
- **`enable_calibration`**: Learns if the system is overconfident/underconfident and adjusts
- **`enable_strategy_updates`**: Adjusts backend weights based on performance (every 1 hour)

**All are optional** - set to `False` to disable learning (useful for testing).

---

## Viewing Learning Statistics

### Overall Performance

```python
overall = router.learning_tracker.get_overall_performance(window=100)
print(f"Avg confidence: {overall['avg_confidence']:.2f}")
print(f"Success rate: {overall['success_rate']*100:.1f}%")
```

### Per-Backend Performance

```python
backend_stats = router.learning_tracker.get_backend_comparison(window=100)
for backend, metrics in backend_stats.items():
    if metrics.count > 0:
        print(f"{backend}: {metrics.avg_confidence:.2f} confidence")
```

### Calibration Status

```python
curve = router.calibrator.get_calibration_curve()
if curve.calibrated:
    print(f"ECE: {curve.ece:.3f} (well-calibrated!)")
```

### Strategy Updates

```python
print(f"Updates: {router.strategy_updater.update_count}")
print(f"Rollbacks: {router.strategy_updater.rollback_count}")
```

---

## Running the Demo

**From any directory** (demo auto-configures paths):

```bash
# From context directory
cd HoloLoom/context
python demo_learning.py

# Or from repository root
cd c:/Users/blake/OneDrive/Documents/mythRL
python HoloLoom/context/demo_learning.py

# Or from anywhere
python path/to/HoloLoom/context/demo_learning.py
```

This runs 5 queries and shows learning statistics. **No PYTHONPATH setup required!**

---

## Configuration Options

### Calibration Settings

```python
from HoloLoom.context import ConfidenceCalibrator

calibrator = ConfidenceCalibrator(
    min_observations=100  # Min data before adjustments (default: 100)
)
```

### Strategy Update Settings

```python
from HoloLoom.context import StrategyUpdater

updater = StrategyUpdater(
    query_router=router,
    update_interval=3600.0,   # Seconds between updates (default: 1 hour)
    min_observations=100,     # Min data before first update (default: 100)
    max_adjustment=0.20       # Max weight change per update (default: 20%)
)
```

---

## Disabling Learning (for testing)

```python
# No learning - just routing
router = await create_query_router(
    mcp_server,
    session_id,
    enable_learning=False,
    enable_calibration=False,
    enable_strategy_updates=False
)
```

---

## Production Example

```python
import asyncio
from HoloLoom.infrastructure.sql import SQLConfig
from HoloLoom.infrastructure.mcp import create_mcp_server, generate_session_id
from HoloLoom.context import create_query_router


async def production_router():
    """Production-ready router with learning"""

    # Production database
    sql_config = SQLConfig(
        sqlite_path="./data/production.db",
        # Or use PostgreSQL:
        # postgres_url="postgresql://user:pass@localhost/db"
    )

    mcp_server = await create_mcp_server(sql_config)

    # Create router with learning
    router = await create_query_router(
        mcp_server,
        generate_session_id(),
        enable_learning=True,
        enable_calibration=True,
        enable_strategy_updates=True
    )

    return router


async def handle_query(router, query_text):
    """Handle a single query with error handling"""
    try:
        result = await router.route(query_text)

        return {
            "success": True,
            "pattern": result.pattern.value,
            "backends": result.backends_used,
            "confidence": result.confidence,
            "latency_ms": result.total_latency_ms,
            "rows": result.rows,
            "row_count": result.row_count
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    router = await production_router()

    # Handle queries
    result = await handle_query(router, "What is the policy?")

    if result["success"]:
        print(f"Success! Confidence: {result['confidence']:.2f}")
    else:
        print(f"Error: {result['error']}")

    # Periodically log learning statistics
    stats = router.learning_tracker.get_overall_performance(window=1000)
    print(f"Last 1000 queries: {stats['avg_confidence']:.2f} avg confidence")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Performance

- **Per-query overhead**: <2ms (calibration + tracking)
- **Strategy updates**: ~10ms every 1 hour
- **Memory**: ~100 bytes per tracked event

**Recommendation**: Keep learning enabled in production - the overhead is negligible and the benefits are significant.

---

## Next Steps

- **Full Documentation**: See [PART_4_LEARNING_COMPLETE.md](PART_4_LEARNING_COMPLETE.md)
- **Test Suite**: Run `python test_learning_routing.py` to see all tests
- **Advanced Usage**: See test suite for examples of all features

---

## Troubleshooting

### "Module not found: HoloLoom"

**For demo_learning.py**: The demo auto-configures paths, so this shouldn't happen. Just run:
```bash
python demo_learning.py
```

**For your own code**: Add the repository root to sys.path:
```python
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(repo_root))
```

Or set PYTHONPATH:
```bash
cd c:/Users/blake/OneDrive/Documents/mythRL
PYTHONPATH=. python your_script.py
```

### "Not enough observations for calibration"
- Calibrator needs 100 observations before adjusting
- Run more queries or lower `min_observations`

### "Strategy updates not happening"
- Updater needs 100 observations + 1 hour interval
- Use `force_update()` to trigger immediately for testing

---

## Questions?

See the test suite (`test_learning_routing.py`) for comprehensive examples of all features.
