# Elle Unity Client - Quick Start Guide

**Goal**: Working Unity → HoloLoom prototype in 12 hours
**Status**: Ready to implement
**Created**: 2025-11-24

---

## ⏱️ 12-Hour Prototype Timeline

| Hours | Task | Deliverable |
|-------|------|-------------|
| **0-2** | Unity setup + packages | Project running |
| **2-4** | HoloLoom WebSocket connection | Console logs showing connection |
| **4-6** | Voice input implementation | Voice → console output |
| **6-8** | AR overlay rendering | Text overlays visible in headset |
| **8-10** | Quest 3 build + testing | APK running on Quest |
| **10-12** | End-to-end testing + polish | Voice → HoloLoom → AR overlay working |

---

## Step 1: Install Unity (30 minutes)

### Download Unity Hub

1. Go to https://unity.com/download
2. Download **Unity Hub** (launcher)
3. Install Unity Hub

### Install Unity Editor

1. Open Unity Hub
2. Click "Install Editor"
3. Select **Unity 2022.3 LTS** (Long Term Support)
   - ⚠️ **Not Unity 6** (too new, see note below)
4. Add modules:
   - ✅ Android Build Support
   - ✅ Android SDK & NDK Tools
   - ✅ OpenJDK
5. Click "Install" (~10 GB download)

**Why Unity 2022.3 LTS (Not Unity 6)?**
- ✅ **Stable**: 2+ years of bug fixes, no breaking changes
- ✅ **Quest 3 Tested**: Meta officially supports 2022.3 LTS
- ✅ **XR Packages**: OpenXR 1.9, XR Toolkit 2.5 fully compatible
- ✅ **Community**: 10,000+ answered questions, extensive tutorials
- 🎯 **Upgrade Later**: Migrate to Unity 6 LTS in Phase 2 (Q3 2025)

**Unity 6 Available**: Yes, but XR packages still maturing. See [UNITY_VERSION_GUIDE.md](UNITY_VERSION_GUIDE.md) for upgrade path.

**TL;DR**: 2022.3 LTS = 5 hours to prototype. Unity 6 = 8-12 hours (debugging compatibility issues). Stick with 2022.3 for now!

---

## Step 2: Create Unity Project (15 minutes)

### Create New Project

1. Open Unity Hub
2. Click "New Project"
3. Select **3D Core** template
4. Name: `ElleUnity`
5. Location: Your `elle_unity` folder
6. Click "Create Project"

### Configure for Android/Quest

1. File → Build Settings
2. Select **Android** platform
3. Click "Switch Platform" (takes 2-3 minutes)
4. Close Build Settings

### Enable XR Plugin

1. Edit → Project Settings
2. **XR Plug-in Management** (install if prompted)
3. Click **Android** tab (Android icon)
4. Check ✅ **OpenXR**
5. Click **OpenXR** settings
6. Under "Interaction Profiles":
   - Add "Meta Quest Touch Pro Controller Profile"
   - Add "Eye Gaze Interaction Profile"
7. Close Project Settings

---

## Step 3: Install Packages (20 minutes)

### Package Manager

Window → Package Manager

**Install these packages:**

1. **XR Interaction Toolkit**
   - Unity Registry → Search "XR Interaction Toolkit"
   - Install (v2.5.0 or newer)
   - Import sample: "Starter Assets"

2. **Unity Barracuda** (for ML inference)
   - Unity Registry → Search "Barracuda"
   - Install (v3.0.0 or newer)

3. **TextMeshPro**
   - Unity Registry → Search "TextMeshPro"
   - Install (should be pre-installed)
   - If prompted, import TMP Essentials

### Install via Package Manager (Packages via Git)

Window → Package Manager → + → Add package from git URL

**Add these URLs:**

```
com.unity.xr.hands@1.3.0
```

### NuGet Packages (via Visual Studio)

**After project is created**, open a C# script in Visual Studio:

1. Tools → NuGet Package Manager → Manage NuGet Packages for Solution
2. Search and install:
   - **WebSocketSharp** (sta.websocket-sharp)
   - **Newtonsoft.Json** (Json.NET)

**Alternative** (if above doesn't work):
- Download DLLs manually
- Place in `Assets/Plugins/` folder

---

## Step 4: Copy Scripts (5 minutes)

### Copy C# Files

Copy all scripts from your `elle_unity/Assets/Scripts/` folder to Unity project:

```
Assets/
├── Scripts/
│   ├── Elle.Core/
│   │   └── HoloLoomBridge.cs
│   ├── Elle.UI/
│   │   ├── VoiceUI.cs
│   │   └── VisualizationManager.cs
│   └── Elle.Vision/
│       ├── HandTracker.cs
│       └── ObjectDetector.cs
```

**Unity will automatically compile** when you copy files.

### Check for Errors

Look at Console (Window → General → Console)
- ❌ Red errors → missing packages (go back to Step 3)
- ⚠️ Yellow warnings → usually OK
- ✅ No errors → good to go!

---

## Step 5: Create Scene (30 minutes)

### 1. Create New Scene

1. File → New Scene
2. Select **Basic (Built-in)** template
3. Save as `Scenes/ARScene`

### 2. Delete Default Objects

- Delete "Main Camera"
- Delete "Directional Light"

### 3. Add XR Origin

1. Right-click Hierarchy → XR → XR Origin (Action-based)
2. This creates:
   - XR Origin
   - Camera Offset
   - Main Camera
   - Left/Right Controllers

### 4. Configure XR Origin

Select **XR Origin**, in Inspector:
- Camera Floor Offset Object: Drag "Camera Offset" here
- Camera Y Offset: 0

### 5. Add HoloLoom Bridge

1. Right-click Hierarchy → Create Empty
2. Rename to "HoloLoomBridge"
3. Add Component → Search "HoloLoomBridge"
4. In Inspector:
   - Backend URL: `ws://192.168.1.100:8000/ws/ar` (replace with your PC's IP)
   - Auto Connect: ✅ Checked
   - Log Messages: ✅ Checked

**Finding your PC IP:**
- Windows: `ipconfig` in cmd → look for "IPv4 Address"
- Mac/Linux: `ifconfig` → look for "inet"

### 6. Add Voice UI

1. Right-click Hierarchy → Create Empty
2. Rename to "VoiceUI"
3. Add Component → Search "VoiceUI"
4. In Inspector:
   - Wake Word: "Hey Elle"
   - Bridge: Drag "HoloLoomBridge" object here

### 7. Add Visualization Manager

1. Right-click Hierarchy → Create Empty
2. Rename to "VisualizationManager"
3. Add Component → Search "VisualizationManager"
4. Leave defaults for now (we'll create prefabs later)

### 8. Add Hand Tracker

1. Select "XR Origin"
2. Add Component → Search "HandTracker"
3. Enable Gesture Recognition: ✅ Checked

### 9. Add Object Detector (Optional - Phase 2)

1. Select "Main Camera"
2. Add Component → Search "ObjectDetector"
3. Leave model empty for now (requires trained model)

### 10. Add Lighting

1. Right-click Hierarchy → Light → Directional Light
2. Rotation: (50, -30, 0)
3. Intensity: 1.0

### Final Hierarchy Should Look Like:

```
Hierarchy:
├── XR Origin
│   ├── Camera Offset
│   │   └── Main Camera
│   ├── Left Controller
│   └── Right Controller
├── HoloLoomBridge
├── VoiceUI
├── VisualizationManager
└── Directional Light
```

---

## Step 6: Start HoloLoom Backend (5 minutes)

### Terminal 1: Start HoloLoom

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL\HoloLoom
python -m HoloLoom.server.ar_api
```

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     WebSocket endpoint: ws://0.0.0.0:8000/ws/ar
```

**Keep this terminal running!**

---

## Step 7: Test in Unity Editor (30 minutes)

### Play Mode Test

1. Click **Play** button (top center)
2. Check Console for:
   ```
   ✅ Connected to HoloLoom backend
   ✅ Wake word recognition started: 'Hey Elle'
   ```

### Test Voice Input (Windows only)

In Play mode:
1. Say "Hey Elle"
2. Should see: `🎤 Wake word detected`
3. Say "What is Thompson Sampling?"
4. Should see:
   ```
   📝 Dictation result: 'What is Thompson Sampling?'
   📤 Sending query: What is Thompson Sampling?
   📩 Received: {...}
   ✅ HoloLoom response: Thompson Sampling is...
   ```

**If voice doesn't work in Editor:**
- Normal! Voice recognition is limited in Unity Editor
- Test on Quest 3 instead (Step 8)

---

## Step 8: Build for Quest 3 (45 minutes)

### Configure Build Settings

1. File → Build Settings
2. Platform: **Android** (should already be selected)
3. Click "Add Open Scenes" (adds ARScene)
4. Click "Player Settings"

### Player Settings

**Other Settings:**
- Color Space: Linear
- Auto Graphics API: ✅ Unchecked
- Graphics APIs: OpenGLES3 only (remove Vulkan if present)
- Minimum API Level: Android 10.0 (API level 29)
- Target API Level: Android 12.0 (API level 31)
- Scripting Backend: IL2CPP
- Target Architectures: ✅ ARM64

**XR Plug-in Management:**
- OpenXR: ✅ Checked
- Depth Submission Mode: Depth 24-bit
- Stereo Rendering Mode: Multiview

**Publishing Settings:**
- Create new keystore (or use existing)
- Company Name: Your name
- Package: `com.yourname.elle`

### Build APK

1. File → Build Settings
2. Click "Build"
3. Save as `ElleUnity.apk`
4. Wait 5-10 minutes (first build is slow)

### Install on Quest 3

**Method 1: USB (Recommended)**

1. Enable Developer Mode on Quest 3:
   - Meta Quest app on phone → Devices → Quest 3 → Developer Mode → ON
2. Connect Quest 3 to PC via USB-C
3. Put on headset, allow USB debugging
4. In Unity: File → Build and Run
5. APK installs and launches automatically

**Method 2: SideQuest**

1. Install SideQuest: https://sidequestvr.com/setup-howto
2. Connect Quest 3 via USB
3. Open SideQuest
4. Drag `ElleUnity.apk` to SideQuest window
5. Install completes → Launch from "Unknown Sources" in Quest

---

## Step 9: Test on Quest 3 (60 minutes)

### First Launch

1. Put on Quest 3
2. Open app from Library → Unknown Sources → ElleUnity
3. Allow permissions:
   - ✅ Microphone
   - ✅ Camera (if using object detection)
4. Look around - should see empty AR space

### Test Voice → HoloLoom

1. Say "Hey Elle"
2. Wait for listening indicator (if implemented)
3. Say "What is this?"
4. Should see:
   - Text overlay appear in front of you
   - Elle's response visible in 3D space

### Check Console Logs (for debugging)

**Via USB:**
```bash
adb logcat -s Unity
```

Look for:
```
Unity: ✅ Connected to HoloLoom backend
Unity: 🎤 Wake word detected
Unity: 📤 Sending query: What is this?
Unity: ✅ HoloLoom response: ...
```

### Common Issues

**"Not connected to HoloLoom backend"**
- Check PC IP address is correct in HoloLoomBridge
- Check HoloLoom server is running
- Check Quest and PC are on same WiFi network
- Check firewall isn't blocking port 8000

**"Wake word not detected"**
- Voice recognition may need permissions
- Try manual trigger (add UI button)
- Check microphone permissions in Quest settings

**"No overlays visible"**
- Check VisualizationManager is attached
- Check HoloLoom response has visualizations
- Enable Gizmos in Scene view to see debug visualization

---

## Step 10: Iterate & Polish (60 minutes)

### Add Visual Feedback

Create listening indicator:

1. Create 3D Text:
   - Right-click Hierarchy → 3D Object → Text - TextMeshPro
   - Name: "ListeningIndicator"
   - Position: (0, 0.3, -0.5) (in front of user)
   - Text: "🎤 Listening..."
   - Font Size: 1.0
   - Alignment: Center

2. Disable by default:
   - Select ListeningIndicator
   - Uncheck checkbox at top of Inspector

3. Connect to VoiceUI:
   - Select VoiceUI object
   - Drag ListeningIndicator to "Listening Indicator" field

### Test Multiple Queries

Try different queries to test HoloLoom integration:

1. "What is Thompson Sampling?"
2. "Explain recursive learning"
3. "What's the difference between BARE and FUSED mode?"

### Check Performance

In Quest 3:
- Settings → Developer → Performance Overlay
- Target: 72 FPS minimum

If FPS < 72:
- Reduce detection frequency in ObjectDetector
- Disable object detection for prototype
- Simplify visualization prefabs

---

## ✅ Success Criteria

### Phase 1 Complete When:

- ✅ Unity connects to HoloLoom WebSocket
- ✅ Voice input works ("Hey Elle" → dictation)
- ✅ Query sent to HoloLoom successfully
- ✅ Response received from HoloLoom
- ✅ Text overlay visible in Quest 3
- ✅ End-to-end latency < 1 second

### Test Command

Say: **"Hey Elle, what is Thompson Sampling?"**

**Expected result:**
1. You say wake word → listening starts
2. You say query → sent to HoloLoom
3. ~500ms later → text overlay appears
4. Overlay says: "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff..."

---

## 📝 Next Steps (After Prototype)

### Phase 2: Feature Parity (Week 3-4)

- [ ] Hand tracking gestures (point, pinch)
- [ ] Object detection with Barracuda
- [ ] Spatial audio (Web Audio → Unity Audio)
- [ ] All visualization types (highlight, path)

### Phase 3: Optimization (Week 5-6)

- [ ] 90 FPS on Quest 3
- [ ] Battery optimization (<20%/hour)
- [ ] Native passthrough quality
- [ ] Gesture recognition (Circle to Search)

### Phase 4: Launch (Week 7-8)

- [ ] Quest Store submission
- [ ] Marketing assets (screenshots, video)
- [ ] User documentation
- [ ] Analytics integration

---

## 🐛 Troubleshooting

### Unity Won't Compile

**Error: "The type or namespace name 'WebSocketSharp' could not be found"**

Solution:
1. Download WebSocketSharp DLL: https://github.com/sta/websocket-sharp/releases
2. Place `websocket-sharp.dll` in `Assets/Plugins/`
3. Restart Unity

**Error: "The type or namespace name 'Newtonsoft' could not be found"**

Solution:
1. Window → Package Manager
2. Search "Newtonsoft Json"
3. Install from Unity Registry

### HoloLoom Connection Failed

**WebSocket error: Connection refused**

Check:
- ✅ HoloLoom server running?
- ✅ Correct IP address? (use `ipconfig` / `ifconfig`)
- ✅ Same WiFi network?
- ✅ Firewall allowing port 8000?

Test connection:
```bash
# From Quest (via adb shell)
adb shell
curl http://YOUR_PC_IP:8000/health
```

### Voice Not Working on Quest

**Wake word not detected**

1. Check microphone permissions (Quest Settings → Apps → ElleUnity → Permissions)
2. Test dictation manually (add UI button to trigger)
3. Check console logs for errors

**Dictation timeout**

Increase timeout in VoiceUI Inspector:
- Dictation Timeout: 10 seconds (was 5)

### Performance Issues

**FPS < 60**

1. Disable object detection temporarily
2. Reduce detection frequency: `detectionInterval = 0.2f` (5 FPS)
3. Lower resolution: `inputWidth = 320, inputHeight = 240`
4. Check background apps on Quest

---

## 📚 Resources

### Unity XR Documentation

- OpenXR Plugin: https://docs.unity3d.com/Packages/com.unity.xr.openxr@latest
- XR Interaction Toolkit: https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@latest
- XR Hands: https://docs.unity3d.com/Packages/com.unity.xr.hands@latest

### Quest Development

- Quest Developer Center: https://developer.oculus.com/
- Quest Setup Guide: https://developer.oculus.com/documentation/unity/unity-gs-overview/
- ADB Commands: https://developer.android.com/tools/adb

### Barracuda (ML)

- Barracuda Documentation: https://docs.unity3d.com/Packages/com.unity.barracuda@latest
- ONNX Models: https://github.com/onnx/models

---

## ⏱️ Estimated Timeline

| Task | Time | Cumulative |
|------|------|------------|
| Install Unity | 30 min | 0:30 |
| Create project | 15 min | 0:45 |
| Install packages | 20 min | 1:05 |
| Copy scripts | 5 min | 1:10 |
| Create scene | 30 min | 1:40 |
| Start backend | 5 min | 1:45 |
| Test in editor | 30 min | 2:15 |
| Build for Quest | 45 min | 3:00 |
| Test on Quest | 60 min | 4:00 |
| Iterate & polish | 60 min | 5:00 |
| **TOTAL** | **5 hours** | |

**Note**: Original estimate was 12 hours, but with scripts already written, you can complete in ~5 hours!

---

## 🎉 Completion

When you see this in your Quest 3:

```
You: "Hey Elle, what is Thompson Sampling?"
        ↓
Elle: [Text overlay appears in 3D space]
      "Thompson Sampling is a Bayesian approach to the
       exploration-exploitation tradeoff. It samples from
       the posterior distribution of expected rewards..."
```

**🎊 Congratulations! You've built a working Unity → HoloLoom AR client!**

Next: Continue to Phase 2 (feature parity) or Phase 3 (optimization).

---

**Created**: 2025-11-24
**Last Updated**: 2025-11-24
**Status**: Ready to use
