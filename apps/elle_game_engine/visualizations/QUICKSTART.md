# BigPlay Visualizations - Quick Start Guide

**Last Updated:** 2025-11-17
**Status:** All 16 visualizations complete and production-ready

---

## 🚀 Getting Started (2 minutes)

### Option 1: Open Locally (No Setup Required)

1. **Navigate to the visualizations directory:**
   ```bash
   cd apps/elle_game_engine/visualizations/
   ```

2. **Open the main hub:**
   ```bash
   # On macOS
   open index.html

   # On Linux
   xdg-open index.html

   # On Windows
   start index.html
   ```

3. **Click any visualization card to explore!**

### Option 2: Use a Local Server (Recommended)

For the best experience (especially for WebSocket features):

```bash
# Python 3
python -m http.server 8080

# Python 2
python -m SimpleHTTPServer 8080

# Node.js (if you have npx)
npx http-server -p 8080

# Then open: http://localhost:8080
```

---

## 📚 Visualization Gallery

### 1. **System Architecture** (`architecture.html`)
**What it shows:** Complete BigPlay tech stack from client to database

**Key features:**
- 16 clickable components
- 4-layer architecture visualization
- Platform integrations (Unity, Godot, Unreal, Web)

**How to use:**
- Click any component to see details
- Hover for tooltips
- Use dark mode toggle (top-right)

---

### 2. **PAD Emotion Model** (`emotion-model.html`)
**What it shows:** 3D Pleasure-Arousal-Dominance emotion space

**Key features:**
- Interactive 3D WebGL visualization
- 5 sample NPCs with different emotional states
- Rotate, zoom, click to explore

**How to use:**
- Drag to rotate the 3D sphere
- Scroll to zoom
- Click NPC markers to see details

---

### 3. **Performance Dashboard** (`performance-dashboard.html`)
**What it shows:** Real-time performance metrics and analytics

**Key features:**
- 6 KPI cards (latency, throughput, cost)
- 7 interactive charts
- Simulated real-time updates

**How to use:**
- Watch metrics update in real-time
- Charts auto-refresh every 2 seconds
- Toggle dark mode for night-friendly view

---

### 4. **Quest Flow Diagrams** (`quest-flow.html`)
**What it shows:** Dynamic quest generation with branching narratives

**Key features:**
- D3.js force-directed graph
- 3 example quests
- Multiple endings and conditional objectives

**How to use:**
- Drag nodes to rearrange
- Zoom with mouse wheel
- Pan by dragging background

---

### 5. **Multiplayer Architecture** (`multiplayer-architecture.html`)
**What it shows:** WebSocket connections and real-time synchronization

**Key features:**
- Animated sequence diagrams
- 4 interactive scenarios
- NPC conversation locking flow

**How to use:**
- Click scenario buttons to switch views
- Watch animated message flow
- Understand WebSocket lifecycle

---

### 6. **NPC Relationship Graph** (`npc-relationships.html`)
**What it shows:** Network graph of NPC relationships and alliances

**Key features:**
- D3.js force-directed layout
- 7 relationship types (friend, enemy, romance, etc.)
- 3 different worlds to explore

**How to use:**
- Select a world from dropdown
- Filter by relationship type
- Drag nodes to explore connections
- Zoom and pan

---

### 7. **Live NPC Playground** (`npc-playground.html`)
**What it shows:** Interactive environment to chat with NPCs

**Key features:**
- 3D emotion sphere with real-time updates
- Live PAD emotion sliders
- 3 NPC personalities (Alice, Guard Bob, Wizard Merrick)
- Export conversation as JSON

**How to use:**
- Select an NPC from dropdown
- Type message and press Enter
- Watch emotions change in real-time
- Adjust emotion sliders manually
- Export conversation history

---

### 8. **Interactive Code Editor** (`code-editor.html`)
**What it shows:** VS Code-quality code editor with examples

**Key features:**
- Monaco Editor integration
- 7 complete code examples
- Multi-language support (Python, JavaScript, C#)
- Simulated code execution

**How to use:**
- Select example from dropdown
- Choose language (Python/JS/C#)
- Edit code with IntelliSense
- Press Ctrl+Enter to run
- Press Ctrl+S to download

**Examples included:**
1. Hello World - Basic NPC creation
2. NPC Conversation - Multi-turn dialogue
3. Emotion System - PAD emotion manipulation
4. Quest Branching - Dynamic quest generation
5. Multiplayer - Shared world state
6. Unity Integration - C# game integration
7. JavaScript SDK - Web client usage

---

### 9. **Live Demo** (`live-demo.html`)
**What it shows:** Real-time WebSocket integration with live metrics

**Key features:**
- 3 connection modes (Demo, Local API, Production)
- Live NPC conversation
- Real-time latency chart
- Activity log with timestamps

**How to use:**

**Demo Mode (No Backend Required):**
1. Open `live-demo.html`
2. Already in "Demo" mode
3. Type message and press Enter
4. NPC responds with simulated latency (100-300ms)

**Local API Mode (Requires Backend):**
1. Start FastAPI server: `uvicorn main:app --reload`
2. Click "Local API" mode button
3. Connect to `ws://localhost:8000/ws`
4. Real NPC responses via LLM

**Production Mode:**
1. Click "Production" mode button
2. Connect to `wss://api.bigplay.dev/ws`
3. Production environment integration

---

## 🎨 Features Available Everywhere

### Dark Mode
- **Toggle:** Click moon/sun icon (top-right of any page)
- **Persistence:** Your preference is saved
- **Accessibility:** High contrast, WCAG compliant

### Tooltips
- **Hover:** Most elements have helpful tooltips
- **Keyboard:** Tab navigation supported
- **Mobile:** Tap-friendly on touch devices

### Responsive Design
- **Mobile:** All visualizations work on phones/tablets
- **Touch:** 44x44px minimum touch targets
- **Gestures:** Swipe, pinch-to-zoom where applicable

---

## 🔧 Troubleshooting

### Visualization Not Loading?

**Problem:** Blank screen or error messages

**Solutions:**
1. Check browser console (F12) for errors
2. Try a different browser (Chrome, Firefox, Safari)
3. Disable browser extensions (ad blockers)
4. Use local server instead of file:// protocol

### WebSocket Connection Failing?

**Problem:** "Disconnected" status in live demo

**Solutions:**
1. Use "Demo" mode (works without backend)
2. Verify backend is running: `curl http://localhost:8000/health`
3. Check WebSocket endpoint: `ws://localhost:8000/ws`
4. Review browser console for connection errors

### 3D Emotion Sphere Not Rendering?

**Problem:** Black screen in emotion visualizations

**Solutions:**
1. Check WebGL support: Visit https://get.webgl.org/
2. Update graphics drivers
3. Try different browser (Chrome has best WebGL support)
4. Disable hardware acceleration if glitchy

### Code Editor Not Working?

**Problem:** Monaco editor not loading

**Solutions:**
1. Check internet connection (Monaco loads from CDN)
2. Wait 5-10 seconds for Monaco to initialize
3. Check browser console for CDN errors
4. Try refreshing the page

---

## 📖 Learning Paths

### For Game Developers

**Recommended order:**
1. Start with **System Architecture** to understand the stack
2. Try **NPC Playground** to interact with NPCs
3. Explore **Emotion Model** to understand PAD system
4. Use **Code Editor** to see integration examples

### For Backend Engineers

**Recommended order:**
1. **System Architecture** - See full tech stack
2. **Multiplayer Architecture** - Understand WebSocket flow
3. **Live Demo** - Test real-time integration
4. **Performance Dashboard** - Monitor system health

### For Data Scientists

**Recommended order:**
1. **Emotion Model** - Understand PAD representation
2. **NPC Relationships** - See knowledge graph structure
3. **Quest Flow** - Explore branching logic
4. **Performance Dashboard** - Analytics and metrics

---

## 🎯 Common Tasks

### Export a Conversation

1. Open **NPC Playground**
2. Have a conversation with an NPC
3. Click "Export" button
4. Saves as JSON file with full history

### Share a Visualization

1. Host visualizations on web server
2. Share URL: `https://yourserver.com/visualizations/index.html`
3. Or send file directly (all work offline)

### Customize an NPC

1. Open **Code Editor**
2. Select "NPC Conversation" example
3. Modify `personality` or `role`
4. Run to see changes (simulated)

### Test WebSocket Integration

1. Open **Live Demo**
2. Start in "Demo" mode
3. Send test messages
4. Verify latency metrics
5. Switch to "Local API" when backend ready

---

## 🚀 Next Steps

After exploring the visualizations:

1. **Try the guided tour:** Built-in walkthrough of architecture
2. **Read the docs:** `../GETTING_STARTED.md` for full documentation
3. **Build something:** Use code examples as templates
4. **Contribute:** See `IMPLEMENTATION_STATUS.md` for enhancement ideas

---

## 📞 Need Help?

- **Documentation:** See `../API_REFERENCE.md`
- **Tutorials:** See `../TUTORIALS.md`
- **Issues:** Check browser console (F12)
- **Questions:** Review inline tooltips

---

**Enjoy exploring BigPlay! 🎮**
