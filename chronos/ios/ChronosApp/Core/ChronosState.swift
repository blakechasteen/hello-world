//
//  ChronosState.swift
//  Chronos
//
//  State manager implementing the 5 verbs: start, stop, log, note, link
//

import Foundation
import Combine

@MainActor
class ChronosState: ObservableObject {
    @Published var activeTask: ChronosEvent?
    @Published var events: [ChronosEvent] = []

    private let eventLog: EventLog
    private var cancellables = Set<AnyCancellable>()

    init(eventLog: EventLog? = nil) {
        self.eventLog = eventLog ?? EventLog()

        // Sync events from log
        self.eventLog.$events
            .assign(to: &$events)

        // Find active task
        loadActiveTask()
    }

    // MARK: - Active Task Management

    /// Find most recent unclosed start event
    private func loadActiveTask() {
        let startEvents = events.filter { $0.event == .start }
        let stopEventStartIds = Set(events
            .filter { $0.event == .stop }
            .compactMap { $0.startId })

        // Find starts without corresponding stops
        let unclosedStarts = startEvents.filter { !stopEventStartIds.contains($0.id) }

        // Most recent unclosed is active
        activeTask = unclosedStarts.max(by: { $0.ts < $1.ts })

        if let active = activeTask {
            print("Active task: \(active.task ?? "Unknown") (started \(active.formattedTime))")
        } else {
            print("No active task")
        }
    }

    /// Elapsed time for active task
    var activeTaskElapsed: TimeInterval? {
        guard let active = activeTask else { return nil }
        return Date().timeIntervalSince(active.ts)
    }

    /// Formatted elapsed time
    var activeTaskElapsedFormatted: String? {
        guard let elapsed = activeTaskElapsed else { return nil }

        let hours = Int(elapsed / 3600)
        let minutes = Int((elapsed.truncatingRemainder(dividingBy: 3600)) / 60)
        let seconds = Int(elapsed.truncatingRemainder(dividingBy: 60))

        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        } else {
            return String(format: "%02d:%02d", minutes, seconds)
        }
    }

    // MARK: - The 5 Verbs

    /// START - Begin tracking a task
    @discardableResult
    func start(task: String, tags: [String] = []) -> String {
        // Auto-stop previous task if exists
        if activeTask != nil {
            stop()
        }

        let event = ChronosEvent.start(task: task, tags: tags)
        eventLog.append(event)

        // Reload to get the event with generated ID
        loadActiveTask()

        let tagsStr = tags.isEmpty ? "" : " — tagged \(tags.map { "#\($0)" }.joined(separator: " "))"
        return "✅ Started \(task) at \(event.formattedTime)\(tagsStr)"
    }

    /// STOP - End current task
    @discardableResult
    func stop() -> String {
        guard let active = activeTask else {
            return "❌ No active task to stop"
        }

        let duration = Date().timeIntervalSince(active.ts)
        let event = ChronosEvent.stop(
            task: active.task ?? "Unknown",
            duration: duration,
            startId: active.id
        )

        eventLog.append(event)
        loadActiveTask()  // This will clear activeTask

        return "✅ Stopped \(active.task ?? "Unknown") — duration \(event.formattedDuration ?? "")"
    }

    /// LOG - Record a completed task retroactively
    @discardableResult
    func log(task: String, duration: TimeInterval, tags: [String] = []) -> String {
        let event = ChronosEvent.log(task: task, duration: duration, tags: tags)
        eventLog.append(event)

        let tagsStr = tags.isEmpty ? "" : " — tagged \(tags.map { "#\($0)" }.joined(separator: " "))"
        return "✅ Logged \(task), \(event.formattedDuration ?? "")\(tagsStr)"
    }

    /// NOTE - Add context to a task
    @discardableResult
    func note(text: String, linkedTo: String? = nil) -> String {
        let linkedId = linkedTo ?? activeTask?.id

        guard let linkedId = linkedId else {
            return "❌ No active task and no event ID specified"
        }

        // Verify linked event exists
        guard let linkedEvent = eventLog.event(withId: linkedId) else {
            return "❌ Event \(linkedId) not found"
        }

        let event = ChronosEvent.note(text: text, linkedTo: linkedId)
        eventLog.append(event)

        let taskName = linkedEvent.task ?? linkedId
        return "✅ Note added to \(taskName)"
    }

    /// LINK - Connect events to external entities
    @discardableResult
    func link(from fromId: String, to toId: String, relation: String = "related_to") -> String {
        // Verify from event exists
        guard eventLog.event(withId: fromId) != nil else {
            return "❌ Event \(fromId) not found"
        }

        let event = ChronosEvent.link(from: fromId, to: toId, relation: relation)
        eventLog.append(event)

        return "✅ Linked \(fromId) → \(toId) (\(relation))"
    }

    // MARK: - Query Helpers

    /// Get today's events
    var todayEvents: [ChronosEvent] {
        eventLog.eventsToday
    }

    /// Get events for date
    func events(for date: Date) -> [ChronosEvent] {
        eventLog.events(for: date)
    }

    /// Get total time tracked today
    var totalTimeToday: TimeInterval {
        eventLog.totalTimeToday
    }

    /// Get total time tracked today (formatted)
    var totalTimeTodayFormatted: String {
        let total = totalTimeToday
        let hours = Int(total / 3600)
        let minutes = Int((total.truncatingRemainder(dividingBy: 3600)) / 60)

        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }

    /// Status message
    var statusMessage: String {
        if let active = activeTask,
           let elapsed = activeTaskElapsedFormatted {
            return "⏱ Active: \(active.task ?? "Unknown") (elapsed \(elapsed))"
        } else {
            return "⏸ No active task"
        }
    }

    // MARK: - Tasks Summary

    /// Get unique task names from events
    var taskNames: [String] {
        Array(Set(events.compactMap { $0.task })).sorted()
    }

    /// Get unique tags from events
    var allTags: [String] {
        let tagSets = events.compactMap { $0.tags }
        return Array(Set(tagSets.flatMap { $0 })).sorted()
    }

    /// Get total time for a specific task
    func totalTime(for task: String) -> TimeInterval {
        eventLog.events(forTask: task)
            .filter { $0.event == .stop || $0.event == .log }
            .compactMap { $0.durationSec }
            .reduce(0, +)
    }
}

// MARK: - Export

extension ChronosState {
    /// Export log file URL (for sharing)
    var exportURL: URL {
        eventLog.exportLog()
    }
}
