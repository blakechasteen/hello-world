# API Quick Start - 5 Minutes

**HoloLoom + DSPy Integration via REST API**

---

## ✅ Integration Status

```bash
curl http://localhost:8000/health | python -m json.tool
```

**Expected**:
```json
{
    "status": "healthy",
    "hololoom_initialized": true,
    "mode": "production"
}
```

✅ **Production mode** = Integration working!

---

## 🚀 Quick Tests

### 1. Root Endpoint
```bash
curl http://localhost:8000/ | python -m json.tool
```

### 2. List Workflows
```bash
curl http://localhost:8000/workflows | python -m json.tool
```

### 3. Interactive Docs
Open: **http://localhost:8000/docs**

---

## 🧪 Try Examples

### Code Explanation Optimization
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d @examples/optimize_code_explanation.json | python -m json.tool
```

### Customer Support Optimization
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d @examples/optimize_customer_support.json | python -m json.tool
```

### Q&A Workflow
```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d @examples/workflow_qa_thompson_sampling.json | python -m json.tool
```

**Note**: Without `OPENAI_API_KEY`, you'll get RateLimitError - this proves integration works!

---

## 🔑 Add OpenAI Key (Optional)

```bash
# Add to .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env

# Restart
docker-compose restart promptly-api

# Test again
curl -X POST http://localhost:8000/optimize \
  -d @examples/optimize_code_explanation.json | python -m json.tool
```

---

## 📚 Learn More

| File | Purpose |
|------|---------|
| **INTEGRATION_SUMMARY.md** | Complete overview (architecture, examples, troubleshooting) |
| **TESTING_GUIDE.md** | Comprehensive testing instructions |
| **examples/*.json** | Ready-to-use requests |

---

## 💡 What You Get

**HoloLoom**:
- 244D semantic space
- Thompson Sampling
- Knowledge graphs
- Multi-scale embeddings
- Recursive learning

**DSPy**:
- MIPROv2 optimizer
- BootstrapFewShot
- Systematic improvement

**API**:
- REST interface
- Swagger docs
- Docker deployment

---

## 🐛 Troubleshooting

**"stub" mode?**
```bash
docker logs promptly-api --tail 20
docker-compose restart promptly-api
```

**RateLimitError?**
- Add `OPENAI_API_KEY` to `.env`
- Restart: `docker-compose restart promptly-api`

---

**Status**: ✅ Ready to use
**Docs**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
**Logs**: `docker logs promptly-api`
