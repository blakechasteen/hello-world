# 📱 Elle AR - iPhone Browser Demo

**Goal**: Test Elle AR assistant on iPhone Safari in **under 5 minutes**

**Status**: ✅ Ready to test!

---

## 🚀 3-Minute Quick Start (Easiest)

### Step 1: Start HoloLoom Backend

Open **Terminal 1** (PowerShell or Command Prompt):

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
python -m hololoom.server.agentic_api
```

**Expected output**:
```
✅ AR API server started on http://0.0.0.0:8000
✅ WebSocket endpoint: ws://0.0.0.0:8000/ws/ar
```

**Leave this running** ✅

---

### Step 2: Start Web Server

Open **Terminal 2**:

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL\elle\ar_web_client

# Option A: Run helper script (auto-configures IP)
start-demo.bat

# Option B: Manual start
python -m http.server 8080
```

**Expected output**:
```
Serving HTTP on 0.0.0.0 port 8080...
```

**Leave this running** ✅

---

### Step 3: Update Backend URL (First Time Only)

Edit `demo.html` line 246:

```javascript
// Before:
const BACKEND_URL = 'ws://192.168.1.100:8000/ws/ar';

// After (use YOUR PC's IP):
const BACKEND_URL = 'ws://10.0.0.231:8000/ws/ar';  // ⚠️ UPDATE THIS
```

**Find your IP**:
```powershell
ipconfig | findstr IPv4
# Output example: IPv4 Address. . . : 10.0.0.231
```

---

### Step 4: Open on iPhone

1. **Open Safari** on iPhone
2. Navigate to: `http://10.0.0.231:8080/demo.html` (use your PC's IP)
3. **Grant camera permission** when prompted
4. Tap **"Start AR Session"** button
5. Say **"Hey Elle"** (wake word)
6. Ask **"What is Thompson Sampling?"**
7. See **AR overlay** appear! 🎉

---

## 📋 Complete Checklist

**Before You Start**:
- [ ] PC and iPhone on **same WiFi network**
- [ ] Python installed on PC
- [ ] Firewall allows port 8000 and 8080

**Setup**:
- [ ] Start HoloLoom backend (Terminal 1)
- [ ] Start web server (Terminal 2)
- [ ] Update `demo.html` with your PC's IP (line 246)
- [ ] Note your PC's IP address

**iPhone**:
- [ ] Safari opened
- [ ] Navigate to `http://YOUR_PC_IP:8080/demo.html`
- [ ] Camera permission granted
- [ ] "Start AR Session" tapped

**Test**:
- [ ] Say "Hey Elle" (wake word detected)
- [ ] Say "What is Thompson Sampling?" (query sent)
- [ ] AR overlay appears with answer
- [ ] Voice response plays (optional)

---

## 🎯 What You'll See

**On iPhone Screen**:

```
┌─────────────────────────────┐
│ 🟢 Connected   Session: abc │  ← Status bar
├─────────────────────────────┤
│                             │
│   [CAMERA FEED]             │
│                             │
│   ┌───────────────────┐     │  ← AR Overlay (answer)
│   │ Thompson Sampling │     │
│   │ is a Bayesian...  │     │
│   └───────────────────┘     │
│                             │
├─────────────────────────────┤
│         🎤 Button           │  ← Voice button
│   Say "Hey Elle" to start   │  ← Transcript
└─────────────────────────────┘
```

**Features Working**:
- ✅ Camera feed (environment view)
- ✅ Voice recognition (Web Speech API)
- ✅ WebSocket connection to HoloLoom
- ✅ AR overlays (text answers)
- ✅ Voice output (text-to-speech)

---

## 🔧 Troubleshooting

### Issue 1: "Cannot connect to camera"

**Cause**: Camera permission denied

**Fix**:
1. iPhone Settings → Safari → Camera
2. Set to **"Ask"** or **"Allow"**
3. Reload page in Safari

---

### Issue 2: "Connection Failed"

**Causes & Fixes**:

**1. Backend not running**:
```bash
# Check if running:
curl http://localhost:8000/health

# Should return: {"status": "healthy"}
```

**2. Wrong IP address**:
```powershell
# Get correct IP:
ipconfig | findstr IPv4

# Update demo.html line 246 with this IP
```

**3. Firewall blocking**:
```powershell
# Run as Administrator:
netsh advfirewall firewall add rule name="Elle Demo" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="Elle Web" dir=in action=allow protocol=TCP localport=8080
```

**4. Different WiFi networks**:
- Check PC WiFi: `ipconfig`
- Check iPhone WiFi: Settings → WiFi
- Must be **same network**!

---

### Issue 3: "Voice not working"

**Cause**: Web Speech API not available

**Fix**:
1. iOS Settings → Safari → Advanced → Experimental Features
2. Enable **"WebXR Device API"**
3. Enable **"WebXR AR Module"**
4. Restart Safari

**Alternative**: Use **physical voice button** instead of wake word

---

### Issue 4: "AR overlays not appearing"

**Causes & Fixes**:

**1. Response from hololoom is empty**:
- Check Terminal 1 for errors
- Try different query: "What is machine learning?"

**2. Console errors**:
- Open Safari DevTools: Settings → Safari → Advanced → Web Inspector
- Connect iPhone to Mac
- Safari → Develop → iPhone → demo.html
- Check Console for errors

---

## 📊 Performance Tips

**For Best Experience**:

1. **Good lighting** - Camera needs clear view
2. **Close apps** - Free up iPhone memory
3. **Same WiFi** - PC and iPhone on same network
4. **Strong signal** - Stay close to WiFi router

**Expected Performance**:
- **Voice latency**: ~300-500ms
- **Query to overlay**: ~1-2 seconds
- **Frame rate**: 30-60 FPS

---

## 🎨 Customization

### Change Wake Word

Edit `demo.html` line 245:

```javascript
// Before:
const WAKE_WORD = 'hey elle';

// After (your custom wake word):
const WAKE_WORD = 'hey assistant';
```

### Change Overlay Style

Edit CSS in `demo.html` around line 63:

```css
.ar-overlay {
    background: rgba(0, 255, 136, 0.9);  /* Green */
    color: #000;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    font-size: 1rem;
}
```

### Add Text-to-Speech

Uncomment lines in `handleResponse()` function:

```javascript
// Speak response (optional)
if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(response);
    speechSynthesis.speak(utterance);
}
```

---

## 🚀 Next Steps

### Option 1: Full React App (More Features)

**If you want advanced features** (object detection, hand tracking, avatars):

```bash
cd elle/ar_web_client
npm install
npm run dev

# Then open: http://YOUR_PC_IP:5173 on iPhone
```

**See**: [IPHONE_QUICK_START.md](IPHONE_QUICK_START.md) for full guide

---

### Option 2: Deploy to Web

**For public URL** (access from anywhere):

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd elle/ar_web_client
vercel

# Get public URL like: https://elle-ar.vercel.app
```

**Pros**: HTTPS (required for some WebXR features), shareable URL

**Cons**: HoloLoom backend also needs public URL (use ngrok)

---

### Option 3: Unity Native App

**For Quest Store or native performance**:

See: `elle_unity/QUICK_START_GUIDE.md`

---

## 📞 Support

**Common Commands**:

```powershell
# Get your PC IP
ipconfig | findstr IPv4

# Start HoloLoom backend
python -m hololoom.server.agentic_api

# Start web server
python -m http.server 8080

# Test backend health
curl http://localhost:8000/health

# Add firewall rule
netsh advfirewall firewall add rule name="Elle" dir=in action=allow protocol=TCP localport=8000
```

**URLs**:
- Backend health: `http://localhost:8000/health`
- Demo page: `http://YOUR_PC_IP:8080/demo.html`
- Full app (if using npm): `http://YOUR_PC_IP:5173`

**Files to Edit**:
- `demo.html` - Standalone demo (line 246 for backend URL)
- `src/hooks/useElleConnection.ts` - React app backend URL
- `src/App.tsx` - Main React app

---

## 🎯 Testing Checklist

**Step-by-Step Test**:

1. **Backend Running**:
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status": "healthy"}
   ```

2. **Web Server Running**:
   ```bash
   # Terminal shows: Serving HTTP on 0.0.0.0 port 8080
   ```

3. **iPhone Connected**:
   ```
   Safari → http://10.0.0.231:8080/demo.html
   # Shows: Camera feed
   ```

4. **Voice Working**:
   ```
   Say: "Hey Elle"
   # Transcript shows: "hey elle"
   ```

5. **Query Working**:
   ```
   Say: "What is Thompson Sampling?"
   # AR overlay appears with answer
   ```

6. **Connection Status**:
   ```
   Top-left corner: 🟢 Connected
   ```

✅ **All 6 steps passed? Demo is working!** 🎉

---

**Created**: 2025-11-24
**Time to Working Demo**: **3 minutes**
**Platform**: iPhone Safari (iOS 15+)

🎉 **Your iPhone AR demo is ready!** 🎉
