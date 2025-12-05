# Unity Elle Client - Troubleshooting Flowchart

**Visual decision tree for debugging issues**

---

## 🌳 Main Diagnostic Tree

```
                    App Launched on Quest 3
                            |
                   Does it crash immediately?
                   /                        \
                YES                          NO
                 |                            |
         Check Unity Console          Check Unity Console
         for crash logs              for connection status
                |                            |
        [Jump to: Crash Tree]       [Jump to: Connection Tree]
```

---

## 💥 Crash Diagnostic Tree

```
App crashes immediately on Quest 3 launch
                    |
    What does adb logcat show?
                    |
    ┌───────────────┼───────────────┐
    |               |               |
    Missing        Native         Permission
    Libraries      Crash          Denied
    |               |               |
    v               v               v
```

### Crash Type 1: Missing Libraries
```
Error: DllNotFoundException: WebSocketSharp
Error: DllNotFoundException: Newtonsoft.Json

FIX:
1. Open Visual Studio from Unity
2. Tools → NuGet Package Manager → Package Manager Console
3. Run:
   Install-Package WebSocketSharp -Version 1.0.3-rc11
   Install-Package Newtonsoft.Json -Version 13.0.3
4. Rebuild in Unity
```

### Crash Type 2: Native Plugin Error
```
Error: Unable to load native library xyz.so
Error: JNI method not found

FIX:
1. File → Build Settings → Android
2. Player Settings → Other Settings
3. Check "IL2CPP" scripting backend (not Mono)
4. Check "ARM64" architecture
5. Rebuild
```

### Crash Type 3: Permission Denied
```
Error: SecurityException: android.permission.RECORD_AUDIO
Error: SecurityException: android.permission.CAMERA

FIX:
1. Player Settings → Publishing Settings
2. "Custom Main Manifest" → ✅
3. Add to AndroidManifest.xml:
   <uses-permission android:name="android.permission.RECORD_AUDIO"/>
   <uses-permission android:name="android.permission.CAMERA"/>
4. Rebuild
```

---

## 🔌 Connection Diagnostic Tree

```
App running, checking connection status...
                    |
        Is Unity Console showing:
        "✅ Connected to HoloLoom backend"?
                    |
            YES ──────────┐
            NO            |
             |            |
             v            v
    [Connection Failed]  [Connected Successfully]
             |            |
             |            └──> [Jump to: Voice Tree]
             |
    What error message?
             |
    ┌────────┼────────┐
    |        |        |
    Timeout  Refused  Host not found
    |        |        |
    v        v        v
```

### Connection Error 1: Timeout
```
Error: WebSocket connection timeout after 10 seconds

DIAGNOSIS:
Backend is not responding (server down or firewall blocking)

FIX:
1. On PC, check backend running:
   curl http://localhost:8000/health
   Should return: {"status": "healthy"}

2. If not running, start it:
   cd c:\Users\blake\OneDrive\Documents\mythRL
   python -m HoloLoom.server.ar_api

3. Check firewall (run as Admin):
   netsh advfirewall firewall show rule name="HoloLoom Unity"
   If not found, add rule:
   netsh advfirewall firewall add rule name="HoloLoom Unity" dir=in action=allow protocol=TCP localport=8000
```

### Connection Error 2: Connection Refused
```
Error: WebSocket error: Connection refused

DIAGNOSIS:
Backend URL is wrong or backend not listening on that port

FIX:
1. Check HoloLoomBridge.cs line 26:
   [SerializeField] private string backendUrl = "ws://???:8000/ws/ar";

2. Get your PC's actual IP:
   ipconfig | findstr IPv4
   Example: 192.168.1.100

3. Update backendUrl:
   ws://192.168.1.100:8000/ws/ar  (NOT localhost!)

4. Rebuild Unity project

5. Verify from Quest browser:
   http://192.168.1.100:8000/health
   Should show: {"status": "healthy"}
```

### Connection Error 3: Host Not Found
```
Error: WebSocket error: No such host is known

DIAGNOSIS:
Invalid IP address or DNS resolution failed

FIX:
1. Verify PC and Quest on same WiFi network
   Quest: Settings → WiFi → Check network name
   PC: ipconfig → Verify same subnet (192.168.1.x)

2. Ping PC from Quest:
   Quest Browser → http://YOUR_PC_IP:8000/health

3. If ping fails:
   - Check WiFi router (ensure client isolation disabled)
   - Try using PC's IPv4 address directly
   - Disable Windows Firewall temporarily (test only!)
```

---

## 🎤 Voice Diagnostic Tree

```
Connection successful, checking voice...
                    |
        Say "Hey Elle" out loud
                    |
        Does Unity Console show:
        "🎤 Wake word detected: Hey Elle"?
                    |
            YES ──────────┐
            NO            |
             |            |
             v            v
    [Wake Word Failed]   [Wake Word Working]
             |            |
             |            └──> [Jump to: Query Tree]
             |
    Check microphone status...
             |
    ┌────────┼────────┐
    |        |        |
    No       Muted    Wrong
    Permission        language
    |        |        |
    v        v        v
```

### Voice Issue 1: No Permission
```
Console: (Silent - no wake word detection started)

DIAGNOSIS:
App doesn't have microphone permission

FIX:
1. Quest: Settings → Apps → Unknown Sources → ElleUnity
2. Permissions → Microphone → ✅ Allow
3. Restart app
4. Console should show: "✅ Wake word recognition started: 'Hey Elle'"
```

### Voice Issue 2: Microphone Muted
```
Console: ✅ Wake word recognition started: 'Hey Elle'
Console: (But wake word never detected)

DIAGNOSIS:
Quest microphone is muted or voice too quiet

FIX:
1. Quest: Settings → Audio → Check microphone not muted
2. Test microphone:
   Quest Browser → https://webcammictest.com/check-mic.html
3. Speak louder and closer to Quest microphone (bottom edge)
4. Try manual trigger: Add UI button calling VoiceUI.TriggerVoiceInput()
```

### Voice Issue 3: Wrong Language
```
Console: ✅ Wake word recognition started: 'Hey Elle'
Console: Wake word detected: "hail" (instead of "Hey Elle")

DIAGNOSIS:
Unity speech recognition using wrong language

FIX:
1. Check VoiceUI.cs line 69:
   keywordRecognizer = new KeywordRecognizer(new[] { wakeWord });

2. Verify wakeWord is "Hey Elle" (case-sensitive)

3. If using non-English OS:
   Windows Settings → Time & Language → Language → Add English (US)
   Set as default for speech recognition
```

---

## 📤 Query Processing Tree

```
Wake word detected, checking query processing...
                    |
        Say "What is Thompson Sampling?"
                    |
        Does Unity Console show:
        "📝 Dictation result: '...' (confidence: High)"?
                    |
            YES ──────────┐
            NO            |
             |            |
             v            v
    [Dictation Failed]   [Dictation Working]
             |            |
             |            └──> [Jump to: Response Tree]
             |
    Check dictation status...
             |
    ┌────────┼────────┐
    |        |        |
    Timeout  Low      Silent
             Confidence
    |        |        |
    v        v        v
```

### Query Issue 1: Dictation Timeout
```
Console: 🎤 Dictation started - speak your query
Console: Dictation complete: Timeout

DIAGNOSIS:
Dictation timed out (default: 5 seconds)

FIX:
1. Speak immediately after "Hey Elle"
2. Speak clearly and continuously (don't pause mid-sentence)
3. Increase timeout in VoiceUI.cs line 28:
   [SerializeField] private float dictationTimeout = 10f;  (was 5f)
```

### Query Issue 2: Low Confidence
```
Console: 📝 Dictation result: 'what is tom sam link' (confidence: Low)

DIAGNOSIS:
Speech recognition misheard query

FIX:
1. Speak more clearly and slowly
2. Reduce background noise
3. Move to quieter environment
4. Use simpler queries:
   Good: "What is Python?"
   Bad: "What is the difference between Python 2 and Python 3?"
```

### Query Issue 3: Silent Dictation
```
Console: 🎤 Dictation started - speak your query
Console: (No dictation result, no error)

DIAGNOSIS:
Microphone input not reaching Unity

FIX:
1. Check Quest microphone working:
   Quest Browser → https://webcammictest.com/check-mic.html
2. Check Unity console for errors:
   "Dictation error: ..."
3. Restart app (sometimes Unity loses microphone access)
4. Worst case: Rebuild with clean project
```

---

## 📨 Response Diagnostic Tree

```
Query sent, checking backend response...
                    |
        Does Unity Console show:
        "📩 Received: {\"response\": \"...\", ...}"?
                    |
            YES ──────────┐
            NO            |
             |            |
             v            v
    [No Response]        [Response Received]
             |            |
             |            └──> [Jump to: Visualization Tree]
             |
    Check response status...
             |
    ┌────────┼────────┐
    |        |        |
    Backend  Empty    Malformed
    Error    Response JSON
    |        |        |
    v        v        v
```

### Response Issue 1: Backend Error
```
Console: 📤 Sending query: What is Thompson Sampling?
Console: ❌ HoloLoom error: 500 Internal Server Error

DIAGNOSIS:
Backend crashed or returned error

FIX:
1. Check backend terminal for Python errors
2. Common errors:
   - Model not loaded: Restart backend
   - Out of memory: Restart backend, reduce batch size
   - Import error: pip install missing_package
3. Test backend directly:
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"text":"test"}'
```

### Response Issue 2: Empty Response
```
Console: 📩 Received: {"response": "", "visualizations": []}

DIAGNOSIS:
Backend returned empty response (query failed internally)

FIX:
1. Check backend logs for errors
2. Try simpler query: "What is Python?"
3. Check if backend model loaded:
   Backend console should show: "✅ Model loaded"
4. Increase max_steps in HoloLoomBridge.cs:
   max_steps = 10  (was 5)
```

### Response Issue 3: Malformed JSON
```
Console: Failed to parse response: JsonReaderException
Console: Unexpected character encountered while parsing value

DIAGNOSIS:
Backend sent invalid JSON

FIX:
1. Check backend version matches Unity client
2. Update HoloLoom backend:
   cd c:\Users\blake\OneDrive\Documents\mythRL
   git pull
   pip install -r requirements.txt
3. Verify JSON structure:
   Backend should return:
   {"response": "...", "visualizations": [...], "confidence": 0.9}
```

---

## 👁️ Visualization Diagnostic Tree

```
Response received, checking visualization...
                    |
        Do you see text overlay in Quest 3?
                    |
            YES ──────────┐
            NO            |
             |            |
             v            v
    [No Visualization]   [✅ SUCCESS!]
             |            |
             |            └──> Prototype working!
             |
    Check visualization status...
             |
    ┌────────┼────────┬────────┐
    |        |        |        |
    Empty    Behind   TextMesh Not
    Array    Camera   Pro      Created
    |        |        |        |
    v        v        v        v
```

### Visualization Issue 1: Empty Array
```
Console: Rendering 0 visualization(s)

DIAGNOSIS:
Backend didn't return any visualizations

FIX:
1. Check response JSON:
   Console: 📩 Received: {..., "visualizations": []}
2. Ensure backend mode is "verify":
   HoloLoomBridge.cs line 141: mode = "verify"
3. Backend should return:
   "visualizations": [
     {"type": "overlay", "position": {...}, "data": {...}}
   ]
```

### Visualization Issue 2: Behind Camera
```
Console: Created overlay at (0.0, 1.5, 2.0)
Visual: Nothing visible in Quest

DIAGNOSIS:
Overlay spawned behind you or outside field of view

FIX:
1. Look around in Quest (overlay might be to side/behind)
2. Adjust spawn position in VisualizationManager.cs line 95:
   Vector3 spawnPos = Camera.main.transform.position
                     + Camera.main.transform.forward * 2.0f
                     + Camera.main.transform.up * 0.3f;  // Slightly above eye level
3. Reduce distance: * 1.5f (was * 2.0f) for closer overlay
```

### Visualization Issue 3: TextMeshPro Not Created
```
Console: Created overlay at (0.0, 1.5, 2.0)
Console: NullReferenceException: TMPro.TextMeshPro
Unity: Pink missing material

DIAGNOSIS:
TextMeshPro package not installed or not configured

FIX:
1. Window → Package Manager → TextMeshPro
2. If not installed: Click "Install"
3. If installed: Window → TextMeshPro → Import TMP Essential Resources
4. Rebuild scene (sometimes needs clean import)
```

### Visualization Issue 4: GameObject Not Created
```
Console: Rendering 1 visualization(s)
Console: (No "Created overlay" message)

DIAGNOSIS:
VisualizationManager.CreateOverlay() failed silently

FIX:
1. Add debug log at start of CreateOverlay():
   Debug.Log($"CreateOverlay called: {viz.type}, pos={viz.position}");
2. Check for exceptions in Console
3. Verify VisualizationManager attached to GameObject:
   Inspector → ElleManager → Components → VisualizationManager ✅
4. Check reference not null:
   Inspector → VisualizationManager → Overlay Prefab (should be set)
```

---

## ⚡ Performance Diagnostic Tree

```
Everything working but laggy/slow...
                    |
        Check Unity Profiler (Window → Analysis → Profiler)
                    |
    What's the bottleneck?
                    |
    ┌───────────────┼───────────────┬───────────────┐
    |               |               |               |
    CPU >16ms     GPU >11ms      Memory >400MB   Network lag
    |               |               |               |
    v               v               v               v
```

### Performance Issue 1: CPU Bottleneck
```
Profiler: CPU >16ms (target: 11ms for 90 FPS)
Symptom: Frame drops, stuttering

DIAGNOSIS:
Script execution too slow

FIX:
1. Disable ObjectDetector (heaviest component):
   Inspector → ObjectDetector → ✅ Uncheck
2. Reduce detection frequency:
   ObjectDetector.cs line 47: detectionInterval = 0.2f  (was 0.1f)
3. Reduce hand tracking updates:
   HandTracker.cs line 39: updateInterval = 0.033f  (was 0.011f)
```

### Performance Issue 2: GPU Bottleneck
```
Profiler: GPU >11ms (target: 11ms for 90 FPS)
Symptom: Rendering lag, low FPS

DIAGNOSIS:
Too many draw calls or heavy shaders

FIX:
1. Reduce overlay count:
   VisualizationManager.cs line 64: maxVisualizations = 5  (was 10)
2. Simplify materials: Use Unlit shader instead of Standard
3. Reduce particle effects (if any)
4. Lower Quest 3 resolution: Settings → Developer → Render Resolution → 0.8x
```

### Performance Issue 3: Memory Pressure
```
Profiler: Memory >400 MB
Symptom: Crashes after 5-10 minutes

DIAGNOSIS:
Memory leak or too many objects in scene

FIX:
1. Enable auto-cleanup in VisualizationManager.cs line 67:
   autoRemoveAfterSeconds = 5f  (was 0 = disabled)
2. Check for leaked GameObjects:
   Hierarchy → Filter by "Overlay" → Should be <10
3. Add explicit Destroy():
   VisualizationManager.cs line 145: Destroy(go, 5f);
```

### Performance Issue 4: Network Lag
```
Profiler: Network spikes >100ms
Symptom: Delay between speech and overlay

DIAGNOSIS:
Backend response slow or WiFi congested

FIX:
1. Check backend latency:
   Add timestamp in VoiceUI.cs:
   var start = Time.realtimeSinceStartup;
   var response = await bridge.SendQuery(text);
   Debug.Log($"Backend latency: {(Time.realtimeSinceStartup - start) * 1000f}ms");
2. Target: <150ms
3. If >150ms: Backend optimization needed (separate issue)
4. If WiFi congested: Move closer to router or use 5GHz band
```

---

## 🎯 Quick Decision Matrix

**Use this to quickly identify which tree to follow:**

| Symptom | Jump To |
|---------|---------|
| App crashes immediately | [Crash Tree] |
| App launches but black screen | [Connection Tree] |
| Connected but "Hey Elle" ignored | [Voice Tree] |
| Voice works but query not sent | [Query Tree] |
| Query sent but no response | [Response Tree] |
| Response received but nothing visible | [Visualization Tree] |
| Everything works but laggy | [Performance Tree] |

---

## 🆘 Nuclear Options (Last Resort)

### 1. Clean Unity Project
```bash
# Delete these folders (Unity will rebuild)
rmdir /s elle_unity\Library
rmdir /s elle_unity\Temp
rmdir /s elle_unity\Obj

# Reopen project in Unity Hub
```

### 2. Reinstall Packages
```bash
# Delete package cache
rmdir /s elle_unity\Library\PackageCache

# Reopen project (Unity will re-download)
```

### 3. Reset Quest 3 Developer Settings
```
Quest: Settings → System → Developer
- USB Connection Dialog: Allow
- Enable USB Debugging: ✅
- USB Debugging → Restart ADB on Quest
```

### 4. Fresh Build
```bash
# Delete Builds folder
rmdir /s elle_unity\Builds

# Unity: File → Build Settings → Build and Run
# Fresh APK build (~20 minutes)
```

### 5. Backend Reset
```bash
# Stop backend: Ctrl+C

# Clear cache
rm -rf .cache/

# Restart backend
python -m HoloLoom.server.ar_api
```

---

## ✅ Success Validation Checklist

**Run through this checklist to confirm prototype working:**

1. [ ] Backend running: `curl http://localhost:8000/health` → `{"status": "healthy"}`
2. [ ] Unity Console: `✅ Connected to HoloLoom backend`
3. [ ] Say "Hey Elle" → Console: `🎤 Wake word detected: Hey Elle`
4. [ ] Say "What is Thompson Sampling?" → Console: `📝 Dictation result: '...'`
5. [ ] Console: `📤 Sending query: What is Thompson Sampling?`
6. [ ] Console: `📩 Received: {"response": "Thompson Sampling is...", ...}`
7. [ ] Console: `Rendering 1 visualization(s)`
8. [ ] Console: `Created overlay at (0.0, 1.5, 2.0)`
9. [ ] Visual: Text overlay visible in Quest 3 AR
10. [ ] Visual: Overlay auto-dismisses after 5 seconds
11. [ ] Latency: <1 second from "Hey Elle" to overlay visible
12. [ ] FPS: 60-90 FPS (check Unity Profiler)

**All 12 checkmarks = Prototype working! 🎉**

---

## 📞 Emergency Contacts

### Documentation (in priority order)
1. **INTEGRATION_CHECKLIST.md** ← Start here (most comprehensive)
2. **QUICK_REFERENCE.md** ← Quick commands
3. **TROUBLESHOOTING_FLOWCHART.md** ← This file (visual debugging)
4. **QUICK_START_GUIDE.md** ← Detailed tutorial

### External Resources
- **Unity Forums**: https://forum.unity.com/
- **Quest Developer**: https://developer.oculus.com/
- **Stack Overflow**: [unity3d] + [quest] tags

### Diagnostic Commands (save these)
```bash
# Backend health
curl http://localhost:8000/health

# Quest logs
adb logcat -s Unity | findstr "HoloLoom"

# Network test
ipconfig | findstr IPv4

# Quest connection
adb devices

# Unity console (open in Unity)
Ctrl+Shift+C
```

---

**Created**: 2025-11-24
**Purpose**: Visual debugging aid
**Use**: Follow decision trees to diagnose issues

**Remember**: Most issues are fixable! Follow the flowchart systematically. ✅
