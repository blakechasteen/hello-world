# Unity Elle Client - Quick Reference Card

**Print this page or keep it visible during setup**

---

## 🚀 5-Hour Timeline

```
Hour 1: Unity Setup     (Install Unity + packages)
Hour 2: Scene Creation  (Build ElleUnity scene)
Hour 3-4: First Build   (Build APK, deploy to Quest)
Hour 5: Test & Verify   (Voice → Query → Overlay)
```

---

## ⚡ Critical Commands

### Start HoloLoom Backend
```bash
cd c:\Users\blake\OneDrive\Documents\mythRL
python -m HoloLoom.server.ar_api
```

### Get Your PC's IP Address
```powershell
ipconfig | findstr IPv4
# Example output: 192.168.1.100
```

### Add Firewall Rule (Run as Admin)
```powershell
netsh advfirewall firewall add rule name="HoloLoom Unity" dir=in action=allow protocol=TCP localport=8000
```

### Connect Quest 3
```powershell
adb devices
# Should show: <DEVICE_ID>    device
```

### View Quest Logs
```bash
adb logcat -s Unity | findstr "HoloLoom"
```

---

## 📋 Setup Checklist

### Before Opening Unity
- [ ] HoloLoom backend running: `curl http://localhost:8000/health`
- [ ] PC IP address noted: `ipconfig | findstr IPv4`
- [ ] Firewall rule added (port 8000)
- [ ] Quest 3 Developer Mode enabled (Meta Quest app on phone)

### Unity Project Setup (1 hour)
- [ ] Unity Hub installed
- [ ] Unity 2022.3 LTS + Android Build Support (NOT Unity 6 - see note below)
- [ ] Project opened: `elle_unity` folder
- [ ] Packages auto-installed (wait 5-10 minutes)
- [ ] Manual packages: WebSocketSharp, Newtonsoft.Json (via NuGet)

**⚠️ Unity Version**: Use 2022.3 LTS (stable, Quest 3 tested). Unity 6 upgrade available later - see [UNITY_VERSION_GUIDE.md](UNITY_VERSION_GUIDE.md)

### Scene Creation (1 hour)
- [ ] New scene: ElleUnity.unity
- [ ] XR Origin added (GameObject → XR → XR Origin)
- [ ] Passthrough enabled (Camera background: transparent black)
- [ ] ElleManager GameObject created with components:
  - [ ] HoloLoomBridge (backend URL = your PC IP)
  - [ ] VoiceUI (references linked)
  - [ ] VisualizationManager
  - [ ] HandTracker (optional)
  - [ ] ObjectDetector (optional)
- [ ] Build Settings → Android + OpenXR enabled

### Build & Deploy (2 hours)
- [ ] Quest 3 connected via USB-C
- [ ] USB debugging allowed on Quest
- [ ] File → Build and Run (15-20 minute build)
- [ ] APK deployed to Quest
- [ ] Microphone permissions granted

### Testing (1 hour)
- [ ] Launch ElleUnity from Quest App Library → Unknown Sources
- [ ] Check Unity Console: "✅ Connected to HoloLoom backend"
- [ ] Say "Hey Elle" (wake word)
- [ ] Say "What is Thompson Sampling?" (query)
- [ ] See text overlay appear in AR
- [ ] Overlay auto-dismisses after 5 seconds

---

## 🎯 Success Criteria

**Prototype works when all 5 green checkmarks appear:**

✅ **Backend Connection**
```
Unity Console: ✅ Connected to HoloLoom backend
```

✅ **Voice Recognition**
```
Unity Console: 🎤 Wake word detected: Hey Elle
```

✅ **Query Sent**
```
Unity Console: 📤 Sending query: What is Thompson Sampling?
```

✅ **Response Received**
```
Unity Console: 📩 Received: {"response": "Thompson Sampling is...", ...}
```

✅ **Visualization Rendered**
```
Unity Console: Created overlay at (0.0, 1.5, 2.0)
Visual: Text overlay visible in Quest 3
```

**Total latency**: <1 second from "Hey Elle" to overlay visible

---

## ⚠️ Top 3 Issues & Fixes

### 1. "Connection refused"
```
❌ Cause: Backend not running or firewall blocking
✅ Fix: Start backend + add firewall rule + use PC IP (not localhost)
```

### 2. "Wake word not detected"
```
❌ Cause: No microphone permissions
✅ Fix: Quest Settings → Apps → ElleUnity → Permissions → Microphone
```

### 3. "Low FPS"
```
❌ Cause: ObjectDetector too heavy
✅ Fix: Disable ObjectDetector component or reduce detection frequency
```

---

## 📁 File Locations

```
elle_unity/
├── Assets/Scripts/
│   ├── Elle.Core/HoloLoomBridge.cs       ✅ WebSocket client
│   ├── Elle.UI/VoiceUI.cs                ✅ Voice recognition
│   ├── Elle.UI/VisualizationManager.cs   ✅ AR overlays
│   ├── Elle.Vision/HandTracker.cs        ✅ Hand gestures
│   └── Elle.Vision/ObjectDetector.cs     ✅ ML detection
├── QUICK_START_GUIDE.md                   ✅ Detailed steps
├── INTEGRATION_CHECKLIST.md               ✅ Full checklist
└── PROTOTYPE_READY.md                     ✅ Summary
```

---

## 🔧 Essential Unity Inspector Settings

### HoloLoomBridge Component
```
Backend URL: ws://192.168.1.100:8000/ws/ar  (CHANGE THIS!)
Auto Connect: ✅
Reconnect Delay: 5s
Log Messages: ✅
```

### VoiceUI Component
```
Wake Word: "Hey Elle"
Dictation Timeout: 5s
Show Visual Feedback: ✅
Bridge: (drag HoloLoomBridge)
Visualization Manager: (drag VisualizationManager)
```

### Camera (XR Origin/Camera Offset/Main Camera)
```
Background Type: Solid Color
Background Color: (0, 0, 0, 0)  ← Transparent for passthrough
Clear Flags: Solid Color
```

---

## 🎮 Quest 3 Controls

### Voice Commands
- **"Hey Elle"** → Wake word (start listening)
- **"What is [topic]?"** → Factual query
- **"Explain [concept]"** → Explanation query
- **"How do I [task]?"** → Procedural query

### Hand Gestures (if HandTracker enabled)
- **Pinch** (thumb + index) → Select object
- **Point** (index extended) → Indicate direction
- **Palm Up** → Open menu (future)
- **Grab** (fist) → Pick up object (future)

### Controller Buttons
- **Primary Button (A/X)** → Manual voice trigger
- **Grip** → Reset scene (future)
- **Trigger** → Confirm action (future)

---

## 📊 Performance Monitoring

### Frame Rate (check Unity Profiler)
- **Excellent**: 90 FPS (Quest 3 native)
- **Good**: 72 FPS (acceptable)
- **Poor**: <60 FPS (causes nausea)

### Latency (add timestamps in VoiceUI.cs)
```csharp
var startTime = Time.realtimeSinceStartup;
// ... query processing ...
var latency = (Time.realtimeSinceStartup - startTime) * 1000f;
Debug.Log($"⏱️ Latency: {latency:F1}ms");
```

**Target**: <320ms total (100ms voice + 10ms send + 150ms backend + 10ms receive + 50ms render)

### Memory (check Unity Profiler)
- **Baseline**: ~200 MB
- **Acceptable**: ~300 MB
- **Warning**: >400 MB
- **Critical**: >500 MB (likely crash)

---

## 🆘 Emergency Troubleshooting

### Unity Won't Open Project
```bash
# Delete Library folder (Unity will rebuild)
rmdir /s elle_unity\Library
```

### Build Fails
```bash
# Check Android SDK installed
# Unity Hub → Installs → Unity 2022.3 LTS → Add Modules
# Select: Android Build Support + Android SDK & NDK Tools
```

### Quest Won't Connect
```bash
# Reset ADB
adb kill-server
adb start-server
adb devices
```

### App Crashes on Launch
```bash
# Check logs
adb logcat -s Unity | findstr "Exception"
# Common: Missing permissions or invalid backend URL
```

---

## 📞 Support Resources

### Documentation (in order of detail)
1. **INTEGRATION_CHECKLIST.md** ← Most comprehensive
2. **QUICK_START_GUIDE.md** ← Step-by-step tutorial
3. **PROTOTYPE_READY.md** ← Overview + troubleshooting
4. **README.md** ← Architecture + controls

### External Resources
- **Unity XR**: https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@latest
- **Quest Developer**: https://developer.oculus.com/
- **HoloLoom Backend**: `HoloLoom/server/agentic_api.py`

### Common Unity Shortcuts
- **Ctrl+Shift+C** → Open Console
- **F** → Focus selected GameObject
- **Ctrl+P** → Play mode
- **Ctrl+Shift+B** → Build window

---

## 🎯 Phase 2 Roadmap (After Prototype Works)

**Add these features in order**:

1. **Spatial Memory** (1 day)
   - Anchor memories to locations
   - Re-visit location → recall memory

2. **Hand Gestures** (2 days)
   - Pinch to select
   - Point to indicate
   - Grab to manipulate

3. **Object Detection** (3 days)
   - YOLO model integration
   - Real-time labeling
   - Context-aware queries

4. **Multi-Query Research** (2 days)
   - Ask follow-up questions
   - Chain reasoning
   - Synthesis mode

5. **Polish** (2 days)
   - UX improvements
   - Performance optimization
   - Error handling

**Total Phase 2**: ~10 days → Production-ready app

---

## ✅ Final Checklist Before You Start

- [ ] Coffee/energy drink ready ☕
- [ ] Quest 3 charged (>50%)
- [ ] PC on same WiFi as Quest
- [ ] All documentation files open:
  - [ ] INTEGRATION_CHECKLIST.md (detailed steps)
  - [ ] QUICK_REFERENCE.md (this file - keep visible)
- [ ] Backend running and healthy
- [ ] 5 hours of uninterrupted time blocked off

**You're ready! Open Unity Hub and let's build this! 🚀**

---

**Created**: 2025-11-24
**Time to Prototype**: 5 hours from now
**Next**: Install Unity Hub (30 minutes)

**Success looks like**: Putting on Quest 3, saying "Hey Elle, what is Thompson Sampling?", and seeing a floating text answer in AR. Let's make it happen! 🎉
