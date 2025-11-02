# 🎮 Quick Reference: How to Fugg Wit It

## 🚀 1-Minute Start

```powershell
python ui/consciousness_ui_simple.py
```

Open browser → **http://localhost:7860**

## 🎯 What You Can Do

### Try Different Queries
```
"What are the applications of quantum computing?"
"Explain quantum entanglement" 
"How does quantum teleportation work?"
"What are the challenges in building quantum computers?"
```

### Adjust Complexity
- **LITE** → Fast & simple (1 pass)
- **FAST** → Balanced (2 passes)  
- **FULL** → Deep analysis (3 passes)
- **RESEARCH** → Maximum depth (4 passes)

### Toggle Fusion
- **ON** → Multipass graph crawling (discovers connected knowledge)
- **OFF** → Single retrieval (faster but less comprehensive)

### Control Memory
- **Max Memories** → 5-20 items to retrieve
- **Token Budget** → 1000-8000 context tokens

## 📊 What You'll See

### Tab 1: Awareness
```
Confidence: 87.3%
Domain: science/quantum_physics
Is Question: True
```

### Tab 2: Memory Fusion
```
Retrieved: 10 memories
Max Depth: 2 hops
Avg Score: 0.886
```

### Tab 3: Context Packing
```
Total Tokens: 487/2700
Compression: 25%
Avg Importance: 74%
```

### Tab 4: LLM Context
```
# Formatted markdown ready for LLM
# Shows exact context sent to AI
```

### Tab 5: Generation
```
Internal Reasoning: [how it thinks]
External Response: [what it says]
```

### Tab 6: Performance
```
Total Time: 4.73ms
- Awareness: <1ms
- Fusion: <2ms  
- Packing: 0.83ms
- Generation: 3.90ms
```

## 🎨 UI Layout

```
┌──────────────┬──────────────────────┐
│ CONTROLS     │ RESULTS (6 TABS)     │
│              │                      │
│ Query Box    │ 1️⃣ Awareness         │
│ Complexity   │ 2️⃣ Memory Fusion     │
│ Fusion ☑️    │ 3️⃣ Context Packing   │
│ Max Memories │ 4️⃣ LLM Context       │
│ Token Budget │ 5️⃣ Generation        │
│              │ ⚡ Performance       │
│ [PROCESS]    │                      │
│              │                      │
│ JSON Output  │                      │
└──────────────┴──────────────────────┘
```

## 💡 Try This

### Experiment #1: Compare Fusion
1. Ask: "What are quantum computing applications?"
2. Set FULL complexity, Fusion ON
3. Note: memories retrieved, depth reached
4. Toggle Fusion OFF
5. Process again
6. Compare: fewer memories, no graph traversal

### Experiment #2: Complexity Scaling
1. Same query
2. Try LITE → FAST → FULL → RESEARCH
3. Watch: passes increase, more memories, deeper graphs

### Experiment #3: Token Budget
1. Set budget to 2000 (small)
2. Process query
3. Note: compression increases
4. Set budget to 8000 (large)  
5. Process again
6. Note: less compression needed

### Experiment #4: Memory Limits
1. Set max memories to 5
2. Process query
3. Set max memories to 20
4. Process again
5. Compare: token usage, importance scores

## 🔧 Customize It

### Add Your Knowledge
Edit `consciousness_ui_simple.py`:

```python
self.knowledge_base = {
    'your_topic': {
        'content': 'Your knowledge...',
        'relevance': 0.95,
        'related': ['other_topic']
    }
}
```

### Change Theme
```python
demo = gr.Blocks(theme=gr.themes.Base())  # or Glass(), Monochrome()
```

### Enable Sharing
```python
demo.launch(share=True)  # Gets public URL
```

## 📈 Benchmark It

Track these metrics:
- **Time**: Should be <50ms (LITE) to <300ms (FULL)
- **Tokens**: Should use 15-25% of budget efficiently
- **Compression**: Should compress 20-40% of high-importance items
- **Quality**: Avg importance should be 65-85%

## 🎯 Use Cases

### Research Assistant
- High complexity (RESEARCH)
- Many memories (15-20)
- Large budget (6000-8000)
- Fusion ON

### Quick Answers
- Low complexity (LITE)
- Few memories (5-8)
- Small budget (2000-3000)
- Fusion OFF

### Balanced Exploration
- Medium complexity (FULL)
- Moderate memories (10-12)
- Medium budget (4000)
- Fusion ON

## 🐛 Quick Fixes

### Won't start?
```powershell
pip install gradio
```

### Port in use?
Change line in `consciousness_ui_simple.py`:
```python
demo.launch(server_port=7861)
```

### Slow?
- Use LITE complexity
- Reduce memories to 5
- Set budget to 2000

## 🎉 That's It!

You now have:
- ✅ Interactive web UI
- ✅ Real-time pipeline visualization
- ✅ Performance metrics
- ✅ Example queries
- ✅ Full customization

**Start exploring:**
```powershell
python ui/consciousness_ui_simple.py
```

**Then fugg wit it!** 🚀

---

**More Info**: See `WEB_UI_COMPLETE.md` and `ui/README.md`
