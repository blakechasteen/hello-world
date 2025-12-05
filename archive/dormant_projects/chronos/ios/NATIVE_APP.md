# Chronos Native iOS App

Beautiful, offline-first time tracking. Same data format as server, works everywhere.

## Architecture

```
ChronosApp (iOS 17+)
├── Core/
│   ├── EventLog.swift          # Local .jsonl storage
│   ├── ChronosState.swift      # 5 verbs (start/stop/log/note/link)
│   └── ChronosEvent.swift      # Event model
├── Views/
│   ├── ContentView.swift       # Main timeline view
│   ├── ActiveTaskView.swift    # Big timer for current task
│   ├── HistoryView.swift       # Past events
│   └── StatsView.swift         # Daily/weekly summaries
├── Intents/
│   ├── StartTaskIntent.swift   # Siri: "Start tracking X"
│   ├── StopTaskIntent.swift    # Siri: "Stop tracking"
│   └── StatusIntent.swift      # Siri: "What am I tracking?"
├── Widgets/
│   ├── ActiveTaskWidget.swift  # Home screen widget
│   └── StatsWidget.swift       # Daily summary widget
└── Watch/
    └── ChronosWatch (watchOS)  # Apple Watch companion
```

---

## Key Features

### 1. Offline-First
- All data stored locally in `.jsonl` format
- Exact same format as server
- Works on airplane, in basement, anywhere

### 2. Voice Integration
- SiriKit intents for all 5 verbs
- "Hey Siri, start tracking garden work"
- "Hey Siri, what am I tracking?"
- "Hey Siri, stop tracking"

### 3. Beautiful UI
- SwiftUI throughout
- Live Activities (Dynamic Island)
- Widgets (Lock Screen + Home Screen)
- Apple Watch complications

### 4. Optional Sync
- Can sync .jsonl with server (iCloud Drive, Dropbox, SSH)
- Merge conflicts resolved by append-only log
- Works without sync (local-only mode)

---

## Data Model

### ChronosEvent.swift

```swift
import Foundation

struct ChronosEvent: Codable, Identifiable {
    let id: String
    let event: EventType
    let ts: Date

    // Event-specific fields
    var task: String?
    var tags: [String]?
    var durationSec: TimeInterval?
    var startId: String?
    var text: String?
    var linkedTo: String?
    var fromId: String?
    var toId: String?
    var relation: String?

    enum EventType: String, Codable {
        case start, stop, log, note, link
    }

    enum CodingKeys: String, CodingKey {
        case id, event, ts
        case task, tags
        case durationSec = "duration_sec"
        case startId = "start_id"
        case text
        case linkedTo = "linked_to"
        case fromId = "from_id"
        case toId = "to_id"
        case relation
    }
}

extension ChronosEvent {
    /// Parse from .jsonl line
    static func from(jsonLine: String) -> ChronosEvent? {
        guard let data = jsonLine.data(using: .utf8) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ChronosEvent.self, from: data)
    }

    /// Convert to .jsonl line
    func toJSONLine() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(self),
              let json = String(data: data, encoding: .utf8) else {
            return ""
        }
        return json
    }
}
```

### EventLog.swift

```swift
import Foundation

class EventLog: ObservableObject {
    @Published var events: [ChronosEvent] = []

    private let fileURL: URL
    private var eventCounter: Int = 0

    init() {
        // Store in app's documents directory
        let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        self.fileURL = documentsPath.appendingPathComponent("events.jsonl")

        loadEvents()
    }

    /// Load all events from .jsonl file
    private func loadEvents() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return
        }

        do {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)

            events = lines.compactMap { line in
                guard !line.isEmpty else { return nil }
                return ChronosEvent.from(jsonLine: line)
            }

            // Get last event number for ID generation
            if let lastEvent = events.last {
                let idNumber = Int(lastEvent.id.replacingOccurrences(of: "chr_", with: "")) ?? 0
                eventCounter = idNumber
            }
        } catch {
            print("Error loading events: \\(error)")
        }
    }

    /// Append event to log
    func append(_ event: ChronosEvent) {
        var mutableEvent = event

        // Generate ID if not set
        if mutableEvent.id.isEmpty {
            eventCounter += 1
            mutableEvent.id = String(format: "chr_%04d", eventCounter)
        }

        // Set timestamp if not set
        if mutableEvent.ts == Date(timeIntervalSince1970: 0) {
            mutableEvent.ts = Date()
        }

        // Append to file
        do {
            let line = mutableEvent.toJSONLine() + "\\n"
            if let data = line.data(using: .utf8) {
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    let fileHandle = try FileHandle(forWritingTo: fileURL)
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(data)
                    fileHandle.closeFile()
                } else {
                    try data.write(to: fileURL)
                }
            }
        } catch {
            print("Error appending event: \\(error)")
        }

        // Update in-memory list
        events.append(mutableEvent)
    }

    /// Get events for a specific date
    func events(for date: Date) -> [ChronosEvent] {
        let calendar = Calendar.current
        return events.filter { event in
            calendar.isDate(event.ts, inSameDayAs: date)
        }
    }
}
```

### ChronosState.swift

```swift
import Foundation
import Combine

@MainActor
class ChronosState: ObservableObject {
    @Published var activeTask: ChronosEvent?
    @Published var events: [ChronosEvent] = []

    private let eventLog = EventLog()
    private var cancellables = Set<AnyCancellable>()

    init() {
        // Sync events from log
        eventLog.$events
            .assign(to: &$events)

        // Find active task
        loadActiveTask()
    }

    /// Find most recent unclosed start event
    private func loadActiveTask() {
        let starts = events.filter { $0.event == .start }
        let stops = Set(events.filter { $0.event == .stop }.compactMap { $0.startId })

        // Find starts without corresponding stops
        let unclosed = starts.filter { !stops.contains($0.id) }

        // Most recent unclosed is active
        activeTask = unclosed.max(by: { $0.ts < $1.ts })
    }

    // MARK: - The 5 Verbs

    func start(task: String, tags: [String] = []) {
        // Auto-stop previous task
        if activeTask != nil {
            stop()
        }

        let event = ChronosEvent(
            id: "",
            event: .start,
            ts: Date(),
            task: task,
            tags: tags.isEmpty ? nil : tags
        )

        eventLog.append(event)
        activeTask = event
    }

    func stop() {
        guard let active = activeTask else { return }

        let duration = Date().timeIntervalSince(active.ts)

        let event = ChronosEvent(
            id: "",
            event: .stop,
            ts: Date(),
            task: active.task,
            durationSec: duration,
            startId: active.id
        )

        eventLog.append(event)
        activeTask = nil
    }

    func log(task: String, duration: TimeInterval, tags: [String] = []) {
        let startTime = Date().addingTimeInterval(-duration)

        let event = ChronosEvent(
            id: "",
            event: .log,
            ts: startTime,
            task: task,
            durationSec: duration,
            tags: tags.isEmpty ? nil : tags
        )

        eventLog.append(event)
    }

    func note(text: String, linkedTo: String? = nil) {
        let linkedId = linkedTo ?? activeTask?.id

        guard linkedId != nil else { return }

        let event = ChronosEvent(
            id: "",
            event: .note,
            ts: Date(),
            text: text,
            linkedTo: linkedId
        )

        eventLog.append(event)
    }

    func link(from: String, to: String, relation: String = "related_to") {
        let event = ChronosEvent(
            id: "",
            event: .link,
            ts: Date(),
            fromId: from,
            toId: to,
            relation: relation
        )

        eventLog.append(event)
    }
}
```

---

## UI Views

### ContentView.swift (Main Timeline)

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var chronos = ChronosState()

    var body: some View {
        NavigationStack {
            VStack {
                if let active = chronos.activeTask {
                    ActiveTaskCard(task: active)
                        .padding()
                }

                List {
                    ForEach(todayEvents) { event in
                        EventRow(event: event)
                    }
                }
                .listStyle(.plain)
            }
            .navigationTitle("Today")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showStartSheet = true
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                }
            }
        }
        .sheet(isPresented: $showStartSheet) {
            StartTaskSheet(chronos: chronos)
        }
    }

    private var todayEvents: [ChronosEvent] {
        chronos.events.filter { event in
            Calendar.current.isDateInToday(event.ts)
        }
    }

    @State private var showStartSheet = false
}

struct ActiveTaskCard: View {
    let task: ChronosEvent
    @State private var elapsed: TimeInterval = 0

    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 12) {
            Text(task.task ?? "Unknown")
                .font(.title2)
                .fontWeight(.semibold)

            Text(elapsed.formatted(.time(pattern: .hourMinuteSecond)))
                .font(.system(.largeTitle, design: .monospaced))
                .fontWeight(.bold)
                .foregroundStyle(.blue)

            if let tags = task.tags {
                HStack {
                    ForEach(tags, id: \\.self) { tag in
                        Text("#\\(tag)")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.blue.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(.blue.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .onReceive(timer) { _ in
            elapsed = Date().timeIntervalSince(task.ts)
        }
    }
}
```

---

## Siri Integration

### StartTaskIntent.swift

```swift
import AppIntents
import Foundation

struct StartTaskIntent: AppIntent {
    static var title: LocalizedStringResource = "Start Tracking Task"
    static var description = IntentDescription("Start tracking a new task")

    @Parameter(title: "Task Name")
    var taskName: String

    @Parameter(title: "Tags", default: [])
    var tags: [String]

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let chronos = ChronosState()
        chronos.start(task: taskName, tags: tags)

        return .result(
            dialog: "Started tracking \\(taskName)"
        )
    }
}

struct StopTaskIntent: AppIntent {
    static var title: LocalizedStringResource = "Stop Tracking"
    static var description = IntentDescription("Stop the current task")

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let chronos = ChronosState()

        guard let active = chronos.activeTask else {
            return .result(dialog: "No active task")
        }

        let taskName = active.task ?? "Unknown"
        chronos.stop()

        let duration = Date().timeIntervalSince(active.ts)
        let formatted = duration.formatted(.time(pattern: .hourMinute))

        return .result(
            dialog: "Stopped \\(taskName). Duration: \\(formatted)"
        )
    }
}
```

### AppShortcutsProvider.swift

```swift
import AppIntents

struct ChronosShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: StartTaskIntent(),
            phrases: [
                "Start tracking \\(.applicationName)",
                "Start tracking \\(\\.$taskName) in \\(.applicationName)"
            ],
            shortTitle: "Start Task",
            systemImageName: "play.circle"
        )

        AppShortcut(
            intent: StopTaskIntent(),
            phrases: [
                "Stop tracking in \\(.applicationName)",
                "Stop my task in \\(.applicationName)"
            ],
            shortTitle: "Stop Task",
            systemImageName: "stop.circle"
        )
    }
}
```

---

## Widgets

### ActiveTaskWidget.swift

```swift
import WidgetKit
import SwiftUI

struct ActiveTaskWidget: Widget {
    let kind: String = "ActiveTaskWidget"

    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: Provider()) { entry in
            ActiveTaskWidgetView(entry: entry)
        }
        .configurationDisplayName("Active Task")
        .description("Shows your currently tracked task")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}

struct ActiveTaskWidgetView: View {
    let entry: Provider.Entry

    var body: some View {
        if let task = entry.activeTask {
            VStack(alignment: .leading, spacing: 8) {
                Text(task.task ?? "Unknown")
                    .font(.headline)

                Text(entry.elapsed.formatted(.time(pattern: .hourMinuteSecond)))
                    .font(.system(.title, design: .monospaced))
                    .fontWeight(.bold)

                Spacer()

                if let tags = task.tags {
                    HStack {
                        ForEach(tags.prefix(2), id: \\.self) { tag in
                            Text("#\\(tag)")
                                .font(.caption2)
                        }
                    }
                }
            }
            .padding()
            .containerBackground(for: .widget) {
                Color.blue.opacity(0.1)
            }
        } else {
            VStack {
                Image(systemName: "clock")
                    .font(.largeTitle)
                Text("No active task")
                    .font(.caption)
            }
            .containerBackground(for: .widget) { }
        }
    }
}
```

---

## Live Activities (Dynamic Island)

```swift
import ActivityKit

struct ChronosActivityAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        let taskName: String
        let startTime: Date
    }
}

// Start Live Activity
func startLiveActivity(task: String) {
    let attributes = ChronosActivityAttributes()
    let state = ChronosActivityAttributes.ContentState(
        taskName: task,
        startTime: Date()
    )

    do {
        let activity = try Activity<ChronosActivityAttributes>.request(
            attributes: attributes,
            contentState: state,
            pushType: nil
        )
        print("Started Live Activity: \\(activity.id)")
    } catch {
        print("Error starting Live Activity: \\(error)")
    }
}

// Live Activity View
struct ChronosLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: ChronosActivityAttributes.self) { context in
            // Lock screen / banner UI
            HStack {
                Image(systemName: "clock.fill")
                VStack(alignment: .leading) {
                    Text(context.state.taskName)
                        .font(.headline)
                    Text(context.state.startTime, style: .timer)
                        .font(.caption)
                }
            }
            .padding()
        } dynamicIsland: { context in
            DynamicIsland {
                // Expanded view
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: "clock.fill")
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text(context.state.startTime, style: .timer)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    Text(context.state.taskName)
                }
            } compactLeading: {
                Image(systemName: "clock.fill")
            } compactTrailing: {
                Text(context.state.startTime, style: .timer)
            } minimal: {
                Image(systemName: "clock")
            }
        }
    }
}
```

---

## Apple Watch App

### WatchContentView.swift

```swift
import SwiftUI

struct WatchContentView: View {
    @StateObject private var chronos = ChronosState()

    var body: some View {
        VStack {
            if let active = chronos.activeTask {
                VStack(spacing: 4) {
                    Text(active.task ?? "Task")
                        .font(.headline)

                    Text(active.ts, style: .timer)
                        .font(.system(.title, design: .monospaced))

                    Button("Stop") {
                        chronos.stop()
                    }
                    .tint(.red)
                }
            } else {
                Button {
                    // Voice input or predefined list
                } label: {
                    Label("Start", systemImage: "play.circle")
                }
            }
        }
    }
}
```

---

## Building the App

### Requirements
- Xcode 15+
- iOS 17+ / watchOS 10+
- Swift 5.9+

### Project Structure
```
ChronosApp/
├── ChronosApp.xcodeproj
├── ChronosApp/
│   ├── App/
│   │   └── ChronosApp.swift
│   ├── Core/
│   │   ├── EventLog.swift
│   │   ├── ChronosState.swift
│   │   └── ChronosEvent.swift
│   ├── Views/
│   │   ├── ContentView.swift
│   │   ├── ActiveTaskCard.swift
│   │   └── StartTaskSheet.swift
│   └── Intents/
│       ├── StartTaskIntent.swift
│       └── StopTaskIntent.swift
├── ChronosWidget/
│   └── ActiveTaskWidget.swift
└── ChronosWatch/
    └── WatchContentView.swift
```

### Build Steps
1. Create new iOS App project in Xcode
2. Add Widget Extension target
3. Add Watch App target
4. Copy Swift files from above
5. Configure App Intents in Info.plist
6. Build & Run

---

## Data Sync (Optional)

### iCloud Drive Sync

```swift
// Enable iCloud in Xcode capabilities
// Use ubiquitous container

class CloudSync {
    static func syncToCloud() {
        guard let cloudURL = FileManager.default.url(
            forUbiquityContainerIdentifier: nil
        ) else { return }

        let cloudFile = cloudURL.appendingPathComponent("events.jsonl")
        let localFile = EventLog().fileURL

        // Copy local to cloud
        try? FileManager.default.copyItem(at: localFile, to: cloudFile)
    }

    static func syncFromCloud() {
        // Merge cloud events with local
        // Append-only log makes this safe
    }
}
```

---

## Next Steps

1. **Start with Shortcuts** (works today via SSH)
2. **Build native app** (use code above as starting point)
3. **Test on TestFlight** (invite beta users)
4. **Submit to App Store** (or keep as personal tool)

The native app gives you:
- ✅ Offline mode
- ✅ Beautiful UI
- ✅ Widgets everywhere
- ✅ Siri voice control
- ✅ Apple Watch support
- ✅ Live Activities (Dynamic Island)

**Same .jsonl format, works with or without server.**
