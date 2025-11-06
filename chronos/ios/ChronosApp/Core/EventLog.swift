//
//  EventLog.swift
//  Chronos
//
//  Append-only .jsonl event log storage
//

import Foundation
import Combine

@MainActor
class EventLog: ObservableObject {
    @Published var events: [ChronosEvent] = []

    private let fileURL: URL
    private var eventCounter: Int = 0
    private let fileManager = FileManager.default

    init(customPath: URL? = nil) {
        // Default: store in app's documents directory
        if let customPath = customPath {
            self.fileURL = customPath
        } else {
            let documentsPath = fileManager.urls(
                for: .documentDirectory,
                in: .userDomainMask
            )[0]
            self.fileURL = documentsPath.appendingPathComponent("events.jsonl")
        }

        loadEvents()
    }

    // MARK: - Load Events

    /// Load all events from .jsonl file
    private func loadEvents() {
        guard fileManager.fileExists(atPath: fileURL.path) else {
            print("No existing log file, starting fresh")
            return
        }

        do {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)

            events = lines.compactMap { line in
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return nil
                }
                return ChronosEvent.from(jsonLine: line)
            }

            // Get last event number for ID generation
            if let lastEvent = events.last {
                let idString = lastEvent.id.replacingOccurrences(of: "chr_", with: "")
                eventCounter = Int(idString) ?? 0
            }

            print("Loaded \(events.count) events")
        } catch {
            print("Error loading events: \(error)")
        }
    }

    // MARK: - Append Events

    /// Append event to log (generates ID and timestamp if missing)
    func append(_ event: ChronosEvent) {
        var mutableEvent = event

        // Generate ID if not set
        if mutableEvent.id.isEmpty {
            eventCounter += 1
            mutableEvent.id = String(format: "chr_%04d", eventCounter)
        }

        // Set timestamp if not set (zero date)
        if mutableEvent.ts.timeIntervalSince1970 == 0 {
            mutableEvent.ts = Date()
        }

        // Append to file
        do {
            let line = mutableEvent.toJSONLine() + "\n"
            guard let data = line.data(using: .utf8) else {
                print("Error converting event to data")
                return
            }

            if fileManager.fileExists(atPath: fileURL.path) {
                // Append to existing file
                let fileHandle = try FileHandle(forWritingTo: fileURL)
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            } else {
                // Create new file
                try data.write(to: fileURL)
            }

            // Update in-memory list
            events.append(mutableEvent)

            print("Appended event: \(mutableEvent.id) - \(mutableEvent.event)")
        } catch {
            print("Error appending event: \(error)")
        }
    }

    // MARK: - Query Events

    /// Get events for a specific date
    func events(for date: Date) -> [ChronosEvent] {
        let calendar = Calendar.current
        return events.filter { event in
            calendar.isDate(event.ts, inSameDayAs: date)
        }
    }

    /// Get events within a date range
    func events(from start: Date, to end: Date) -> [ChronosEvent] {
        events.filter { event in
            event.ts >= start && event.ts <= end
        }
    }

    /// Find event by ID
    func event(withId id: String) -> ChronosEvent? {
        events.first { $0.id == id }
    }

    /// Get events by type
    func events(ofType type: ChronosEvent.EventType) -> [ChronosEvent] {
        events.filter { $0.event == type }
    }

    /// Get events for a specific task name
    func events(forTask task: String) -> [ChronosEvent] {
        events.filter { $0.task == task }
    }

    // MARK: - Stats

    /// Total events count
    var totalEvents: Int {
        events.count
    }

    /// Events today
    var eventsToday: [ChronosEvent] {
        events(for: Date())
    }

    /// Total time tracked today (sum of stop/log durations)
    var totalTimeToday: TimeInterval {
        eventsToday
            .filter { $0.event == .stop || $0.event == .log }
            .compactMap { $0.durationSec }
            .reduce(0, +)
    }

    // MARK: - File Management

    /// Export log file to share
    func exportLog() -> URL {
        fileURL
    }

    /// Clear all events (dangerous - creates backup first)
    func clearAll() {
        // Create backup
        let backupURL = fileURL.deletingLastPathComponent()
            .appendingPathComponent("events_backup_\(Int(Date().timeIntervalSince1970)).jsonl")

        try? fileManager.copyItem(at: fileURL, to: backupURL)

        // Clear file
        try? "".write(to: fileURL, atomically: true, encoding: .utf8)

        // Clear memory
        events.removeAll()
        eventCounter = 0

        print("Cleared all events (backup at: \(backupURL.lastPathComponent))")
    }
}

// MARK: - Merge Support (for sync)

extension EventLog {
    /// Merge events from another log file (for iCloud sync)
    func merge(from otherURL: URL) {
        do {
            let content = try String(contentsOf: otherURL, encoding: .utf8)
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
                append(event)
            }

            print("Merged \(newEvents.count) new events")
        } catch {
            print("Error merging events: \(error)")
        }
    }
}
