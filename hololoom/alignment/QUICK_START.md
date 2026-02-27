# Quick Start: Production Monitoring

**Prerequisites**: Flask installed in your environment

---

## 1. Install Dependencies

```bash
# Install Flask for Prometheus server
pip install flask

# Or install all optional dependencies
pip install -r hololoom/alignment/requirements-optional.txt
```

---

## 2. Start Prometheus Server

### Option A: Using Wrapper Script (Recommended)

```bash
python run_prometheus_server.py
```

### Option B: Direct Run

```bash
# Windows PowerShell
$env:PYTHONPATH = "."
python hololoom/alignment/prometheus_server.py

# Linux/Mac
PYTHONPATH=. python hololoom/alignment/prometheus_server.py
```

---

## 3. Verify Server is Running

**Visit in browser:**
- Health check: http://localhost:9090/health
- Prometheus metrics: http://localhost:9090/metrics
- JSON stats: http://localhost:9090/stats

**Example curl:**
```bash
curl http://localhost:9090/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "components": 0,
  "total_samples": 0
}
```

---

## 4. Configure Grafana (Optional)

**Add Prometheus data source:**

1. Open Grafana (http://localhost:3000)
2. Go to Configuration → Data Sources
3. Add data source → Prometheus
4. Set URL: `http://localhost:9090`
5. Click "Save & Test"

**Example queries:**
```promql
# P99 latency for all components
alignment_latency_p99

# P99 for specific component
alignment_latency_p99{component="guardrails"}

# Alert rate
rate(alignment_alerts_total[5m])

# Sample throughput
rate(alignment_samples_total[1m])
```

---

## 5. Configure Matrix Alerts (Optional)

### Webhook Mode (Simple)

```python
from hololoom.alignment.matrix_chatops import send_matrix_webhook
from hololoom.alignment.monitoring import Alert, AlertLevel

alert = Alert(
    level=AlertLevel.WARNING,
    component="guardrails",
    message="P99 latency high",
    value=1.5,
    threshold=1.0,
    metric="p99"
)

send_matrix_webhook(
    alert,
    webhook_url="https://matrix.example.com/_matrix/client/r0/rooms/!abc:matrix.org/send/m.room.message"
)
```

### Bot Mode (Advanced)

```bash
# Install matrix-nio
pip install matrix-nio

# Set environment variables
export MATRIX_HOMESERVER=https://matrix.org
export MATRIX_USER_ID=@alignment-bot:matrix.org
export MATRIX_ACCESS_TOKEN=syt_...
export MATRIX_ROOM_ID=!alignment:matrix.org
```

```python
from hololoom.alignment.matrix_chatops import MatrixBot

bot = MatrixBot(
    homeserver="https://matrix.org",
    user_id="@bot:matrix.org",
    access_token="syt_..."
)

await bot.send_alert(alert)
await bot.close()
```

---

## 6. Run Integration Tests

```bash
# Run all integration tests
pytest hololoom/tests/integration/test_alignment_hololoom.py -v

# Run specific test
pytest hololoom/tests/integration/test_alignment_hololoom.py::test_alignment_basic_integration -v

# Run standalone demo
python demos/demo_alignment_orchestrator.py
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROMETHEUS_PORT` | Prometheus server port | 9090 |
| `PROMETHEUS_HOST` | Prometheus server host | 0.0.0.0 |
| `PROMETHEUS_AUTH_USER` | Basic auth username | None |
| `PROMETHEUS_AUTH_PASS` | Basic auth password | None |
| `MATRIX_HOMESERVER` | Matrix homeserver URL | - |
| `MATRIX_USER_ID` | Matrix bot user ID | - |
| `MATRIX_ACCESS_TOKEN` | Matrix bot access token | - |
| `MATRIX_ROOM_ID` | Default Matrix room ID | - |
| `MATRIX_WEBHOOK_URL` | Matrix webhook URL | - |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"

**Solution**: Install Flask
```bash
pip install flask
```

### "ModuleNotFoundError: No module named 'HoloLoom'"

**Solution**: Run from project root or use wrapper script
```bash
# Use wrapper (sets PYTHONPATH automatically)
python run_prometheus_server.py

# Or set PYTHONPATH manually
PYTHONPATH=. python hololoom/alignment/prometheus_server.py
```

### Port 9090 already in use

**Solution**: Use different port
```bash
export PROMETHEUS_PORT=8080
python run_prometheus_server.py
```

### No metrics showing up

**Solution**: Generate some metrics first
```bash
# Run demo to generate metrics
python demos/demo_production_deployment.py

# Then check metrics
curl http://localhost:9090/metrics
```

---

## Next Steps

1. ✅ Start Prometheus server
2. ✅ Verify endpoints working
3. ⏳ Configure Grafana (optional)
4. ⏳ Set up Matrix alerts (optional)
5. ⏳ Run integration tests
6. ⏳ Deploy to production

See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for full deployment guide.
