//
//  ChronosEvent.swift
//  Chronos
//
//  Core event model matching Python .jsonl format
//

import Foundation

struct ChronosEvent: Codable, Identifiable, Hashable {
    var id: String
    let event: EventType
    var ts: Date

    // Event-specific fields (optional based on event type)
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

// MARK: - JSON Line Parsing

extension ChronosEvent {
    /// Parse from .jsonl line
    static func from(jsonLine: String) -> ChronosEvent? {
        guard let data = jsonLine.data(using: .utf8) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(ChronosEvent.self, from: data)
    }

    /// Convert to .jsonl line (without newline)
    func toJSONLine() -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]

        guard let data = try? encoder.encode(self),
              let json = String(data: data, encoding: .utf8) else {
            return ""
        }
        return json
    }
}

// MARK: - Computed Properties

extension ChronosEvent {
    /// Display name for the event
    var displayName: String {
        switch event {
        case .start: return "Started: \(task ?? "Unknown")"
        case .stop: return "Stopped: \(task ?? "Unknown")"
        case .log: return "Logged: \(task ?? "Unknown")"
        case .note: return "Note"
        case .link: return "Linked"
        }
    }

    /// Duration formatted as string (for stop/log events)
    var formattedDuration: String? {
        guard let duration = durationSec else { return nil }

        let hours = Int(duration / 3600)
        let minutes = Int((duration.truncatingRemainder(dividingBy: 3600)) / 60)
        let seconds = Int(duration.truncatingRemainder(dividingBy: 60))

        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        } else {
            return "\(seconds)s"
        }
    }

    /// Timestamp formatted for display
    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: ts)
    }
}

// MARK: - Factory Methods

extension ChronosEvent {
    /// Create a START event
    static func start(task: String, tags: [String] = []) -> ChronosEvent {
        ChronosEvent(
            id: "",  // Will be filled by EventLog
            event: .start,
            ts: Date(),
            task: task,
            tags: tags.isEmpty ? nil : tags
        )
    }

    /// Create a STOP event
    static func stop(task: String, duration: TimeInterval, startId: String) -> ChronosEvent {
        ChronosEvent(
            id: "",
            event: .stop,
            ts: Date(),
            task: task,
            durationSec: duration,
            startId: startId
        )
    }

    /// Create a LOG event
    static func log(task: String, duration: TimeInterval, tags: [String] = []) -> ChronosEvent {
        let startTime = Date().addingTimeInterval(-duration)
        return ChronosEvent(
            id: "",
            event: .log,
            ts: startTime,
            task: task,
            durationSec: duration,
            tags: tags.isEmpty ? nil : tags
        )
    }

    /// Create a NOTE event
    static func note(text: String, linkedTo: String) -> ChronosEvent {
        ChronosEvent(
            id: "",
            event: .note,
            ts: Date(),
            text: text,
            linkedTo: linkedTo
        )
    }

    /// Create a LINK event
    static func link(from: String, to: String, relation: String = "related_to") -> ChronosEvent {
        ChronosEvent(
            id: "",
            event: .link,
            ts: Date(),
            fromId: from,
            toId: to,
            relation: relation
        )
    }
}
