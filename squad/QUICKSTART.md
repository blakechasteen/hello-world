# Squad Quick Start

Get Squad running in **3 minutes**:

## Step 1: Install Dependencies (1 min)

```bash
# From squad/ directory
npm install

# Python deps (if not already installed)
pip install fastapi uvicorn pydantic
```

## Step 2: Start Server (30 sec)

```bash
# Terminal 1
python server.py
```

Wait for: `Squad server ready! 🚀`

## Step 3: Run Extension (1 min)

1. Open `squad/` folder in VS Code
2. Press `F5` (or Run → Start Debugging)
3. New VS Code window opens

## Step 4: Try It! (30 sec)

In the new window:

1. Press `Ctrl+Shift+Q` (or `Cmd+Shift+Q` on Mac)
2. Type: "What is Thompson Sampling?"
3. Watch Squad think and respond!

## What You Get

✅ **4 Reasoning Modes:**
- Direct (fast answers)
- Verify (with fact-checking)
- Research (multi-query exploration)
- Plan & Execute (goal decomposition)

✅ **Full Transparency:**
- See every reasoning step
- View verification results
- Track confidence scores

✅ **Code-Aware:**
- Explain selected code
- Suggest fixes for errors
- Context-aware responses

## Commands to Try

```
Ctrl+Shift+Q - Ask Squad anything
Ctrl+Shift+E - Explain selected code
```

Right-click on code:
- "Squad: Explain Selection"
- "Squad: Suggest Fix"

## Troubleshooting

**Server won't start?**
```bash
pip install fastapi uvicorn pydantic
```

**Extension won't compile?**
```bash
npm install
npm run compile
```

**Can't connect?**
- Check server is running on port 8000
- Look for errors in Output → "Squad"

---

**Next:** See [README.md](README.md) for full documentation
