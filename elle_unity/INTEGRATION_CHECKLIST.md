# Unity Client Integration Checklist

**Created**: 2025-11-24
**Status**: All code written, ready for Unity import
**Time to Working Prototype**: ~5 hours

---

## Pre-Flight Checklist ✈️

Before opening Unity, verify these prerequisites:

### 1. Backend Running ✅

```bash
# Start HoloLoom backend
cd c:\Users\blake\OneDrive\Documents\mythRL
python -m HoloLoom.server.ar_api

# Expected output:
# ✅ AR API server started on http://0.0.0.0:8000
# ✅ WebSocket endpoint: ws://0.0.0.0:8000/ws/ar
```

**Test backend health**:
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### 2. Files Verified ✅

Confirm all Unity files exist:

```powershell
# Check C# scripts
dir elle_unity\Assets\Scripts\Elle.Core\*.cs
dir elle_unity\Assets\Scripts\Elle.UI\*.cs
dir elle_unity\Assets\Scripts\Elle.Vision\*.cs

# Check documentation
dir elle_unity\*.md

# Check Unity config
dir elle_unity\Packages\manifest.json
```

**Expected output**:
```
elle_unity/
├── Assets/Scripts/
│   ├── Elle.Core/HoloLoomBridge.cs       ✅ 380 lines
│   ├── Elle.UI/VoiceUI.cs                ✅ 280 lines
│   ├── Elle.UI/VisualizationManager.cs   ✅ 320 lines
│   ├── Elle.Vision/HandTracker.cs        ✅ 260 lines
│   └── Elle.Vision/ObjectDetector.cs     ✅ 280 lines
├── Packages/manifest.json                 ✅
├── QUICK_START_GUIDE.md                   ✅ 1,500 lines
├── README.md                              ✅ 600 lines
└── PROTOTYPE_READY.md                     ✅ 1,000 lines
```

### 3. Network Configuration ✅

**Find your PC's IP address** (for Quest 3 connection):
```powershell
ipconfig | findstr IPv4
# Example: IPv4 Address. . . . . . . . . : 192.168.1.100
```

**Update HoloLoomBridge.cs** (line 26):
```csharp
// Before (localhost - won't work from Quest)
[SerializeField] private string backendUrl = "ws://localhost:8000/ws/ar";

// After (your PC's IP)
[SerializeField] private string backendUrl = "ws://192.168.1.100:8000/ws/ar";
```

**Firewall rule** (allow port 8000):
```powershell
# Run as Administrator
netsh advfirewall firewall add rule name="HoloLoom Unity" dir=in action=allow protocol=TCP localport=8000
```

---

## Unity Setup (Step 1-3) ⏱️ 1 Hour

### Step 1: Install Unity Hub (15 minutes)

1. Download Unity Hub: https://unity.com/download
2. Install **Unity 2022.3 LTS** (NOT Unity 6) with modules:
   - ✅ Android Build Support
   - ✅ Android SDK & NDK Tools
   - ✅ OpenJDK

**⚠️ Important**: Use Unity 2022.3 LTS, not Unity 6
- Unity 6 is available but XR packages not fully stable yet
- 2022.3 LTS has proven Quest 3 compatibility
- See [UNITY_VERSION_GUIDE.md](UNITY_VERSION_GUIDE.md) for upgrade path later

### Step 2: Open Project (10 minutes)

1. Open Unity Hub
2. Click "Add" → "Add project from disk"
3. Navigate to: `c:\Users\blake\OneDrive\Documents\mythRL\elle_unity`
4. Click "Open"

**Expected**: Unity will:
- Import project (2-3 minutes)
- Auto-install packages from `manifest.json` (3-5 minutes)
- Show "elle_unity" project in Hub

### Step 3: Package Verification (35 minutes)

Unity auto-installs these packages (check Window → Package Manager):

**Core XR Packages**:
- ✅ XR Interaction Toolkit (2.5.2) - 5 min install
- ✅ OpenXR Plugin (1.9.1) - 3 min install
- ✅ XR Hands (1.3.0) - 2 min install
- ✅ Unity Barracuda (3.0.0) - 5 min install
- ✅ TextMeshPro (3.0.9) - 1 min install

**Manual Installs** (via NuGet in Visual Studio):
1. Open Visual Studio from Unity (Assets → Open C# Project)
2. Tools → NuGet Package Manager → Package Manager Console
3. Install packages:
   ```powershell
   Install-Package WebSocketSharp -Version 1.0.3-rc11
   Install-Package Newtonsoft.Json -Version 13.0.3
   ```
4. Close Visual Studio, return to Unity

**Verification**: No red errors in Unity Console

---

## Scene Creation (Step 4-6) ⏱️ 1 Hour

### Step 4: Create XR Origin (20 minutes)

Follow QUICK_START_GUIDE.md Section 5.1:

1. **Create new scene**: File → New Scene → "ElleUnity"
2. **Add XR Origin**:
   - GameObject → XR → XR Origin (Action-based)
   - Includes Main Camera, Left/Right Controllers
3. **Enable passthrough** (for Quest 3 AR):
   - Camera → Inspector → Background Type: Solid Color
   - Camera → Color: (0, 0, 0, 0) - Transparent
4. **Save scene**: Scenes/ElleUnity.unity

### Step 5: Add Elle Components (30 minutes)

Create empty GameObject "ElleManager" with components:

1. **HoloLoomBridge.cs**:
   - GameObject → Create Empty → "ElleManager"
   - Add Component → HoloLoomBridge
   - Inspector: Set Backend URL (your PC's IP)

2. **VoiceUI.cs**:
   - Add Component → VoiceUI
   - Inspector: Link HoloLoomBridge reference
   - Inspector: Link VisualizationManager reference

3. **VisualizationManager.cs**:
   - Add Component → VisualizationManager
   - Creates GameObjects for overlays/highlights

4. **HandTracker.cs** (optional):
   - Add Component → HandTracker
   - Inspector: Enable Quest 3 hand tracking

5. **ObjectDetector.cs** (optional):
   - Add Component → ObjectDetector
   - Inspector: Drop YOLO/SSD model (if available)

### Step 6: Build Settings (10 minutes)

Configure for Quest 3:

1. File → Build Settings → Android
2. Click "Switch Platform" (3-5 minute compile)
3. Player Settings → XR Plug-in Management → OpenXR
4. OpenXR Feature Groups → Enable:
   - ✅ Hand Tracking
   - ✅ Passthrough
   - ✅ Quest Support
5. Set minimum API level: 29 (Android 10)

---

## First Build & Test (Step 7-8) ⏱️ 2 Hours

### Step 7: Connect Quest 3 (15 minutes)

1. **Enable Developer Mode**:
   - Meta Quest app (phone) → Devices → Quest 3 → Settings
   - Developer Mode → Toggle ON
   - Restart Quest 3

2. **USB Connection**:
   - Connect Quest 3 to PC via USB-C
   - Put on headset → "Allow USB debugging" → Always allow

3. **Verify ADB**:
   ```powershell
   adb devices
   # Should show: <DEVICE_ID>    device
   ```

### Step 8: Build and Deploy (1 hour 45 minutes)

1. **Build APK**:
   - File → Build Settings → Build and Run
   - Save as: `Builds/ElleUnity.apk`
   - Build time: ~15-20 minutes (first build)

2. **Deploy to Quest**:
   - Unity auto-installs to Quest
   - Or manual: `adb install -r Builds/ElleUnity.apk`

3. **Launch on Quest**:
   - Quest 3 → App Library → Unknown Sources → ElleUnity
   - Grant microphone permissions

4. **Test workflow** (⏱️ 1 hour):
   - Say "Hey Elle" (wake word)
   - Say "What is Thompson Sampling?" (query)
   - See text overlay appear in AR (verification)

---

## Success Criteria ✅

Prototype is **working** when all 5 criteria met:

### 1. Backend Connection ✅
```
Unity Console output:
✅ Connected to HoloLoom backend
✅ Session ID: <DEVICE_ID>
```

### 2. Voice Recognition ✅
```
Unity Console output:
🎤 Wake word detected: Hey Elle
🎤 Dictation started - speak your query
📝 Dictation result: 'What is Thompson Sampling?' (confidence: High)
```

### 3. Query Processing ✅
```
Unity Console output:
📤 Sending query: What is Thompson Sampling?
📩 Received: {"response": "Thompson Sampling is...", ...}
```

### 4. Visualization Rendering ✅
```
Unity Console output:
✅ HoloLoom response: Thompson Sampling is...
Rendering 1 visualization(s)
Created overlay at (0.0, 1.5, 2.0)
```

### 5. End-to-End Latency ✅
```
Total time: < 1 second from "Hey Elle" to overlay visible
```

**Visual verification in Quest 3**:
- Text overlay appears at eye level
- Content matches HoloLoom response
- Auto-dismisses after 5 seconds

---

## Troubleshooting ⚠️

### Issue 1: "WebSocket error: Connection refused"

**Symptoms**:
```
❌ WebSocket error: Connection refused
Not connected to HoloLoom backend
```

**Solutions**:
1. ✅ Check HoloLoom server running: `curl http://localhost:8000/health`
2. ✅ Verify IP address in HoloLoomBridge.cs (not `localhost`)
3. ✅ Check firewall: `netsh advfirewall firewall show rule name="HoloLoom Unity"`
4. ✅ Ping PC from Quest: Quest Browser → http://YOUR_PC_IP:8000/health
5. ✅ Both PC and Quest on same WiFi network

### Issue 2: "Wake word not detected"

**Symptoms**: Saying "Hey Elle" does nothing

**Solutions**:
1. ✅ Grant microphone permissions: Quest Settings → Apps → ElleUnity → Permissions
2. ✅ Check Unity Console: `✅ Wake word recognition started: 'Hey Elle'`
3. ✅ Try manual trigger: Add UI button calling `VoiceUI.TriggerVoiceInput()`
4. ⚠️ Voice may not work in Unity Editor (test on Quest only)

### Issue 3: "Low FPS (<60)"

**Symptoms**: Stuttering, dropped frames

**Solutions**:
1. ✅ Disable ObjectDetector temporarily (heavy ML inference)
2. ✅ Reduce detection frequency: `detectionInterval = 0.2f` (was 0.1f)
3. ✅ Lower input resolution: `inputWidth = 320` (was 640)
4. ✅ Close background Quest apps
5. ✅ Check Unity Profiler: Window → Analysis → Profiler

### Issue 4: "APK won't install"

**Symptoms**: Build succeeds but won't install on Quest

**Solutions**:
1. ✅ Enable Developer Mode: Meta Quest app → Quest 3 → Developer Mode
2. ✅ Allow USB debugging: Put on Quest → "Allow USB debugging" prompt
3. ✅ Check USB cable supports data (not just charging)
4. ✅ Try SideQuest: https://sidequestvr.com/
5. ✅ Manual install: `adb install -r Builds/ElleUnity.apk`

### Issue 5: "Visualizations not appearing"

**Symptoms**: Query succeeds but no overlay visible

**Solutions**:
1. ✅ Check response JSON: Unity Console → `📩 Received: {...}`
2. ✅ Verify visualizations array not empty: `visualizations: [{...}]`
3. ✅ Check camera distance: Overlays spawn 2m in front (adjust position)
4. ✅ Enable billboard: Text should face camera automatically
5. ✅ Check TextMeshPro installed: Package Manager → TextMeshPro

---

## Performance Targets 🎯

### Frame Rate (Quest 3)
- **Target**: 90 FPS (Quest 3 native refresh rate)
- **Acceptable**: 72 FPS (reduced mode)
- **Minimum**: 60 FPS (below this = nausea)

**Measured**: Check Unity Profiler → CPU/GPU time

### Latency Budget (End-to-End)
| Stage | Target | Measured |
|-------|--------|----------|
| Voice recognition | <100ms | ❓ |
| WebSocket send | <10ms | ❓ |
| HoloLoom backend | <150ms | ❓ |
| WebSocket receive | <10ms | ❓ |
| Visualization render | <50ms | ❓ |
| **Total** | **<320ms** | **❓** |

**Measurement**: Add timestamps in VoiceUI.cs:
```csharp
var startTime = Time.realtimeSinceStartup;
var response = await bridge.SendQuery(text);
var latency = (Time.realtimeSinceStartup - startTime) * 1000f;
Debug.Log($"⏱️ Total latency: {latency:F1}ms");
```

### Memory Usage
- **Baseline**: ~200 MB (empty scene)
- **With Elle**: ~300 MB (acceptable)
- **Warning**: >400 MB (memory pressure)
- **Critical**: >500 MB (likely crash)

**Measurement**: Unity Profiler → Memory

---

## Next Steps After Prototype Works 🚀

### Phase 2 (Add Features)
Once basic workflow works, add:
1. **Hand gestures**: Pinch to select objects
2. **Spatial memory**: Anchor memories to locations
3. **Object detection**: YOLO model integration
4. **Multi-query**: Research mode (ask follow-ups)

### Phase 3 (Optimize)
Polish for production:
1. **Performance**: Target 90 FPS sustained
2. **UX**: Gesture tutorial, voice feedback
3. **Error handling**: Network loss, low battery
4. **Testing**: Playtesting with 10+ users

### Phase 4 (Launch)
Quest Store submission:
1. **Quest Store account**: https://developer.oculus.com/
2. **App Lab submission**: Soft launch to 100 users
3. **Iterate**: Feedback → fixes → resubmit
4. **Full launch**: Main Quest Store (target: 10k users)

---

## Development Tips 💡

### 1. Iteration Speed
- **Editor testing**: Test logic in Unity Editor (no build)
- **Quest testing**: Build only for voice/AR features
- **Logs**: Use `adb logcat -s Unity` for Quest logs

### 2. Debugging
- **Unity Console**: Primary debugging (Ctrl+Shift+C)
- **ADB Logcat**: Quest device logs
  ```bash
  adb logcat -s Unity | findstr "HoloLoom"
  ```
- **Network Sniffer**: Wireshark to debug WebSocket

### 3. Version Control
```bash
# Add Unity project to git
cd elle_unity
git init
git add .
git commit -m "feat: Unity client prototype complete"
```

**.gitignore** for Unity:
```
Library/
Temp/
Obj/
Build/
Builds/
*.csproj
*.unityproj
*.sln
*.user
*.userprefs
```

### 4. Collaboration
- **Share APK**: Upload to Google Drive, send link
- **Share build**: `adb install Builds/ElleUnity.apk` (remote)
- **Remote play**: Quest Remote Desktop (for demos)

---

## Resources 📚

### Unity XR Development
- **OpenXR Plugin**: https://docs.unity3d.com/Packages/com.unity.xr.openxr@latest
- **XR Interaction Toolkit**: https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@latest
- **Quest Developer Center**: https://developer.oculus.com/

### Quest 3 Specific
- **Hand Tracking Guide**: https://developer.oculus.com/documentation/unity/unity-handtracking/
- **Passthrough API**: https://developer.oculus.com/documentation/unity/unity-passthrough/
- **Performance Best Practices**: https://developer.oculus.com/documentation/unity/unity-perf/

### C# / Unity
- **Unity Scripting Reference**: https://docs.unity3d.com/ScriptReference/
- **C# Async/Await**: https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/async/

### HoloLoom Backend
- **API Documentation**: See `HoloLoom/server/agentic_api.py`
- **WebSocket Protocol**: See `UNITY_WEBXR_DUAL_STRATEGY.md` (lines 500-600)

---

## Timeline Summary ⏱️

| Phase | Original Estimate | With Pre-Written Scripts | Savings |
|-------|-------------------|-------------------------|---------|
| **Scripts** | 6 hours | ✅ **Done!** | **-6 hours** |
| Unity setup | 1 hour | 1 hour | 0 |
| Scene creation | 1 hour | 1 hour | 0 |
| Building | 1 hour | 1 hour | 0 |
| Testing | 2 hours | 2 hours | 0 |
| **Total** | **11 hours** | **5 hours** | **-6 hours** |

**Current Status**: Step 0 complete (all scripts written)
**Next**: Step 1 (Install Unity Hub)
**ETA to prototype**: 5 hours from now

---

## Questions? 🤔

**Check these first**:
1. ✅ QUICK_START_GUIDE.md (1,500 lines, step-by-step)
2. ✅ PROTOTYPE_READY.md (summary, troubleshooting)
3. ✅ README.md (architecture, controls)
4. ✅ UNITY_WEBXR_DUAL_STRATEGY.md (why Unity?)

**Still stuck?**
- Unity Console → Right-click error → Copy
- Paste error into Claude/GPT for diagnosis
- Check Unity forums: https://forum.unity.com/

---

**Created**: 2025-11-24
**Status**: ✅ Ready to Start
**Next Step**: Install Unity Hub → Open Project → Follow checklist

🎉 **Good luck with your prototype!** 🎉
