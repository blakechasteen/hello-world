# Chronos iOS App - Build Checklist

**Goal**: Build offline-first time tracking app for iPhone
**Time Estimate**: 1 hour for minimal version
**Status**: Core complete ✅, Views needed 🔨

---

## ✅ What You Already Have (Complete)

### Core Architecture (678 lines)

- [x] **ChronosEvent.swift** (200 lines) - Event model
  - [x] Codable struct with all event types
  - [x] JSON line serialization
  - [x] Factory methods for 5 verbs
  - [x] Display formatting
  - [x] Computed properties

- [x] **EventLog.swift** (236 lines) - Storage layer
  - [x] Async file I/O with FileActor
  - [x] Append-only .jsonl writer
  - [x] Load events on init
  - [x] Query methods (by date, task, type)
  - [x] Stats calculations
  - [x] Merge support for sync
  - [x] Backup on clear

- [x] **ChronosState.swift** (242 lines) - Business logic
  - [x] 5 verbs: start/stop/log/note/link
  - [x] Active task management
  - [x] Auto-stop on new start
  - [x] Observable state for SwiftUI
  - [x] Async + sync wrappers
  - [x] Confirmation messages
  - [x] Summary statistics

### Documentation (2000+ lines)

- [x] **README.md** - Complete reference
- [x] **NATIVE_APP.md** - Full architecture guide
- [x] **SHORTCUTS.md** - Siri Shortcuts guide
- [x] **This checklist** - Build instructions

---

## 🔨 What You Need to Build (1 Hour)

### Minimal Working App (Required)

#### 1. Create Xcode Project (5 minutes)
- [ ] Open Xcode
- [ ] File → New → Project → iOS App
- [ ] Name: "Chronos"
- [ ] Interface: SwiftUI
- [ ] Create

#### 2. Add Core Files (5 minutes)
- [ ] Create "Core" group in project navigator
- [ ] Right-click Core → Add Files
- [ ] Add `ChronosEvent.swift`
- [ ] Add `EventLog.swift`
- [ ] Add `ChronosState.swift`
- [ ] Verify they appear in target membership

#### 3. Create ContentView.swift (20 minutes)

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var chronos = ChronosState()
    @State private var showStartSheet = false

    var body: some View {
        NavigationStack {
            VStack {
                // Active task section
                if let active = chronos.activeTask {
                    ActiveTaskCard(task: active, chronos: chronos)
                        .padding()
                }

                // Events list
                List {
                    ForEach(chronos.todayEvents) { event in
                        EventRow(event: event)
                    }
                }
                .listStyle(.plain)

                // Stats footer
                Text("Today: \(chronos.totalTimeTodayFormatted)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 8)
            }
            .navigationTitle("Chronos")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showStartSheet = true
                    } label: {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                    }
                }
            }
            .sheet(isPresented: $showStartSheet) {
                StartTaskSheet(chronos: chronos, isPresented: $showStartSheet)
            }
        }
    }
}
```

**Checklist**:
- [ ] Copy code above
- [ ] Paste into ContentView.swift
- [ ] Replace entire file

#### 4. Create ActiveTaskCard.swift (10 minutes)

```swift
import SwiftUI

struct ActiveTaskCard: View {
    let task: ChronosEvent
    @ObservedObject var chronos: ChronosState
    @State private var currentTime = Date()

    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 12) {
            Text(task.task ?? "Unknown")
                .font(.title2)
                .fontWeight(.semibold)

            Text(chronos.activeTaskElapsedFormatted ?? "00:00")
                .font(.system(.largeTitle, design: .monospaced))
                .fontWeight(.bold)
                .foregroundStyle(.blue)

            Button {
                chronos.stop()
            } label: {
                Label("Stop", systemImage: "stop.circle.fill")
                    .font(.headline)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.blue.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .onReceive(timer) { _ in
            currentTime = Date()
        }
    }
}
```

**Checklist**:
- [ ] File → New → Swift File
- [ ] Name it "ActiveTaskCard.swift"
- [ ] Paste code above

#### 5. Create EventRow.swift (10 minutes)

```swift
import SwiftUI

struct EventRow: View {
    let event: ChronosEvent

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title3)
                .foregroundStyle(iconColor)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                Text(event.displayName)
                    .font(.body)

                if let duration = event.formattedDuration {
                    Text(duration)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            Text(event.formattedTime)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    var iconName: String {
        switch event.event {
        case .start: return "play.circle.fill"
        case .stop: return "stop.circle.fill"
        case .log: return "clock.fill"
        case .note: return "note.text"
        case .link: return "link.circle.fill"
        }
    }

    var iconColor: Color {
        switch event.event {
        case .start: return .green
        case .stop: return .red
        case .log: return .blue
        case .note: return .orange
        case .link: return .purple
        }
    }
}
```

**Checklist**:
- [ ] File → New → Swift File
- [ ] Name it "EventRow.swift"
- [ ] Paste code above

#### 6. Create StartTaskSheet.swift (10 minutes)

```swift
import SwiftUI

struct StartTaskSheet: View {
    @ObservedObject var chronos: ChronosState
    @Binding var isPresented: Bool
    @State private var taskName = ""
    @FocusState private var isTaskFieldFocused: Bool

    var body: some View {
        NavigationStack {
            Form {
                Section("Task Name") {
                    TextField("What are you working on?", text: $taskName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .focused($isTaskFieldFocused)
                }

                Section {
                    Text("Tap Start to begin tracking time.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Start Task")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Start") {
                        chronos.start(task: taskName)
                        isPresented = false
                    }
                    .disabled(taskName.isEmpty)
                    .bold()
                }
            }
            .onAppear {
                isTaskFieldFocused = true
            }
        }
        .presentationDetents([.medium])
    }
}
```

**Checklist**:
- [ ] File → New → Swift File
- [ ] Name it "StartTaskSheet.swift"
- [ ] Paste code above

#### 7. Configure Signing (5 minutes)

- [ ] Select project in navigator (blue icon)
- [ ] Select "Chronos" target
- [ ] "Signing & Capabilities" tab
- [ ] Check "Automatically manage signing"
- [ ] Select your Team (Apple ID)
- [ ] Verify Bundle Identifier is unique

#### 8. Build & Test (5 minutes)

- [ ] Connect iPhone via USB cable
- [ ] Select iPhone as destination (top toolbar)
- [ ] Click Play button ▶️
- [ ] On iPhone: Settings → General → VPN & Device Management
- [ ] Tap your email → Trust
- [ ] Return to Xcode, click Play ▶️ again
- [ ] App launches on iPhone! 🎉

---

## Testing Checklist (10 minutes)

### Basic Functionality
- [ ] App launches successfully
- [ ] Tap + button → sheet opens
- [ ] Enter task name → Start button enables
- [ ] Tap Start → sheet closes
- [ ] Active task card appears
- [ ] Timer counts up (1 second intervals)
- [ ] Tap Stop → task stops
- [ ] Event appears in list below
- [ ] Scroll through events
- [ ] See total time at bottom

### Data Persistence
- [ ] Force quit app (swipe up)
- [ ] Reopen app
- [ ] Events still visible ✅

### Edge Cases
- [ ] Start task while one is running → auto-stops first
- [ ] Empty task name → Start button disabled
- [ ] Cancel sheet → no task started

---

## Optional Enhancements (Later)

### Polish (30 minutes)
- [ ] Add pull-to-refresh on events list
- [ ] Add swipe actions (delete note, etc.)
- [ ] Add task history view (all time)
- [ ] Add search/filter
- [ ] Add dark mode refinements

### Siri Integration (1 hour)
- [ ] Create AppIntents framework
- [ ] Add StartTaskIntent
- [ ] Add StopTaskIntent
- [ ] Test "Hey Siri, start tracking work"

### Widgets (2 hours)
- [ ] Add WidgetExtension target
- [ ] Create ActiveTaskWidget
- [ ] Add small/medium widget sizes
- [ ] Test on home screen

### Apple Watch (3 hours)
- [ ] Add watchOS target
- [ ] Create WatchContentView
- [ ] Add complications
- [ ] Test on Watch

### Live Activities (2 hours)
- [ ] Add ActivityKit
- [ ] Create ChronosLiveActivity
- [ ] Show in Dynamic Island
- [ ] Test on iPhone 14 Pro+

---

## File Manifest

### ✅ Complete (In Repo)
```
chronos/ios/ChronosApp/Core/
├── ChronosEvent.swift        ✅ 200 lines
├── EventLog.swift            ✅ 236 lines
└── ChronosState.swift        ✅ 242 lines
```

### 🔨 To Create (This Weekend)
```
Chronos/
├── ChronosApp.swift          🔨 Auto-generated by Xcode
├── ContentView.swift         🔨 20 lines (see above)
├── Views/
│   ├── ActiveTaskCard.swift  🔨 30 lines (see above)
│   ├── EventRow.swift        🔨 40 lines (see above)
│   └── StartTaskSheet.swift  🔨 40 lines (see above)
└── Core/
    ├── ChronosEvent.swift    ✅ Copy from repo
    ├── EventLog.swift        ✅ Copy from repo
    └── ChronosState.swift    ✅ Copy from repo
```

**Total lines to write**: ~130 lines (mostly copy-paste)

---

## Success Criteria

### Minimum Viable App ✅
- [x] Core files work (already complete)
- [ ] Can start a task
- [ ] Timer counts up
- [ ] Can stop a task
- [ ] Events persist across launches

### You're Done When...
- [ ] App installs on your iPhone
- [ ] You track 3 tasks
- [ ] All data persists after app restart
- [ ] Timer updates smoothly (no jank)
- [ ] File size < 100 KB after 50 events

---

## Time Budget

| Task | Estimate | Actual |
|------|----------|--------|
| Create Xcode project | 5 min | |
| Add core files | 5 min | |
| ContentView | 20 min | |
| ActiveTaskCard | 10 min | |
| EventRow | 10 min | |
| StartTaskSheet | 10 min | |
| Configure signing | 5 min | |
| Build & test | 5 min | |
| **Total** | **70 min** | |

**Goal**: Working app in under 90 minutes

---

## Next Actions

1. **Right now**: Create Xcode project
2. **Next 5 min**: Add 3 core files
3. **Next 30 min**: Create 4 view files
4. **Next 5 min**: Configure signing
5. **Next 5 min**: Build & test
6. **Done!**: Track your first task 🎉

---

## Support

**Stuck?** Check:
- README.md - Complete reference
- NATIVE_APP.md - Full code examples
- Core files - Already complete and tested

**Questions?**
- All code is provided above
- Copy-paste exactly as shown
- Run and test

**Ready?** Start with step 1: Create Xcode project
