//
//  EventLog.swift
//  Chronos
//
//  Append-only .jsonl event log with async I/O
//

import Foundation
import Combine

@MainActor
class EventLog: ObservableObject {
    @Published var events: [ChronosEvent] = []

    private let fileURL: URL
    private var eventCounter: Int = 0
    private let fileManager = FileManager.default

    // Background actor for file operations
    private let fileActor = FileActor()

    init(customPath: URL? = nil) {
        if let customPath = customPath {
            self.fileURL = customPath
        } else {
            let documentsPath = fileManager.urls(
                for: .documentDirectory,
                in: .userDomainMask
            )[0]
            self.fileURL = documentsPath.appendingPathComponent("events.jsonl")
        }

        // Load events asynchronously on init
        Task {
            await loadEvents()
        }
    }

    // MARK: - Load Events (Async)

    func loadEvents() async {
        guard fileManager.fileExists(atPath: fileURL.path) else {
            print("No existing log file, starting fresh")
            return
        }

        do {
            // Read file on background actor
            let content = try await fileActor.readFile(at: fileURL)
            let lines = content.components(separatedBy: .newlines)

            let loadedEvents = lines.compactMap { line -> ChronosEvent? in
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return nil
                }
                return ChronosEvent.from(jsonLine: line)
            }

            // Update on main actor
            self.events = loadedEvents

            // Get last event number for ID generation
            if let lastEvent = loadedEvents.last {
                let idString = lastEvent.id.replacingOccurrences(of: "chr_", with: "")
                eventCounter = Int(idString) ?? 0
            }

            print("Loaded \(loadedEvents.count) events")
        } catch {
            print("Error loading events: \(error)")
        }
    }

    // MARK: - Append Events (Async)

    func append(_ event: ChronosEvent) async {
        var mutableEvent = event

        // Generate ID if not set
        if mutableEvent.id.isEmpty {
            eventCounter += 1
            mutableEvent.id = String(format: "chr_%04d", eventCounter)
        }

        // Set timestamp if not set
        if mutableEvent.ts.timeIntervalSince1970 == 0 {
            mutableEvent.ts = Date()
        }

        do {
            // Write to file on background actor
            try await fileActor.appendLine(mutableEvent.toJSONLine(), to: fileURL)

            // Update in-memory list on main actor
            events.append(mutableEvent)

            print("Appended event: \(mutableEvent.id) - \(mutableEvent.event)")
        } catch {
            print("Error appending event: \(error)")
        }
    }

    // MARK: - Query Events (Synchronous - already in memory)

    func events(for date: Date) -> [ChronosEvent] {
        let calendar = Calendar.current
        return events.filter { event in
            calendar.isDate(event.ts, inSameDayAs: date)
        }
    }

    func events(from start: Date, to end: Date) -> [ChronosEvent] {
        events.filter { event in
            event.ts >= start && event.ts <= end
        }
    }

    func event(withId id: String) -> ChronosEvent? {
        events.first { $0.id == id }
    }

    func events(ofType type: ChronosEvent.EventType) -> [ChronosEvent] {
        events.filter { $0.event == type }
    }

    func events(forTask task: String) -> [ChronosEvent] {
        events.filter { $0.task == task }
    }

    // MARK: - Stats

    var totalEvents: Int {
        events.count
    }

    var eventsToday: [ChronosEvent] {
        events(for: Date())
    }

    var totalTimeToday: TimeInterval {
        eventsToday
            .filter { $0.event == .stop || $0.event == .log }
            .compactMap { $0.durationSec }
            .reduce(0, +)
    }

    // MARK: - File Management

    func exportLog() -> URL {
        fileURL
    }

    func clearAll() async {
        // Create backup
        let backupURL = fileURL.deletingLastPathComponent()
            .appendingPathComponent("events_backup_\(Int(Date().timeIntervalSince1970)).jsonl")

        try? await fileActor.copyFile(from: fileURL, to: backupURL)

        // Clear file
        try? await fileActor.writeFile("", to: fileURL)

        // Clear memory
        events.removeAll()
        eventCounter = 0

        print("Cleared all events (backup at: \(backupURL.lastPathComponent))")
    }
}

// MARK: - File Actor (Background File Operations)

actor FileActor {
    func readFile(at url: URL) throws -> String {
        try String(contentsOf: url, encoding: .utf8)
    }

    func appendLine(_ line: String, to url: URL) throws {
        let data = (line + "\n").data(using: .utf8)!

        if FileManager.default.fileExists(atPath: url.path) {
            let fileHandle = try FileHandle(forWritingTo: url)
            fileHandle.seekToEndOfFile()
            fileHandle.write(data)
            try fileHandle.close()
        } else {
            try data.write(to: url)
        }
    }

    func writeFile(_ content: String, to url: URL) throws {
        try content.write(to: url, atomically: true, encoding: .utf8)
    }

    func copyFile(from source: URL, to destination: URL) throws {
        try FileManager.default.copyItem(at: source, to: destination)
    }
}

// MARK: - Merge Support (Async)

extension EventLog {
    func merge(from otherURL: URL) async {
        do {
            let content = try await fileActor.readFile(at: otherURL)
            let lines = content.components(separatedBy: .newlines)

            var newEvents: [ChronosEvent] = []

            for line in lines {
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }

                if let event = ChronosEvent.from(jsonLine: line) {
                    // Only add if not already present
                    if !events.contains(where: { $0.id == event.id }) {
                        newEvents.append(event)
                    }
                }
            }

            // Sort by timestamp and append new events
            newEvents.sort { $0.ts < $1.ts }

            for event in newEvents {
                await append(event)
            }

            print("Merged \(newEvents.count) new events")
        } catch {
            print("Error merging events: \(error)")
        }
    }
}
