//
//  ChronosState.swift
//  Chronos
//
//  State manager implementing the 5 verbs: start, stop, log, note, link
//  Fully async with proper concurrency handling
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

        // Observe events changes to update active task
        self.eventLog.$events
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.loadActiveTask()
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Active Task Management

    private func loadActiveTask() {
        let startEvents = events.filter { $0.event == .start }
        let stopEventStartIds = Set(events
            .filter { $0.event == .stop }
            .compactMap { $0.startId })

        let unclosedStarts = startEvents.filter { !stopEventStartIds.contains($0.id) }
        activeTask = unclosedStarts.max(by: { $0.ts < $1.ts })

        if let active = activeTask {
            print("Active task: \(active.task ?? "Unknown") (started \(active.formattedTime))")
        }
    }

    var activeTaskElapsed: TimeInterval? {
        guard let active = activeTask else { return nil }
        return Date().timeIntervalSince(active.ts)
    }

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

    // MARK: - The 5 Verbs (All Async)

    /// START - Begin tracking a task
    @discardableResult
    func start(task: String, tags: [String] = []) async -> String {
        // Auto-stop previous task if exists
        if activeTask != nil {
            await stop()
        }

        let event = ChronosEvent.start(task: task, tags: tags)
        await eventLog.append(event)

        let tagsStr = tags.isEmpty ? "" : " — tagged \(tags.map { "#\($0)" }.joined(separator: " "))"
        let formattedTime = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .short)
        return "✅ Started \(task) at \(formattedTime)\(tagsStr)"
    }

    /// STOP - End current task
    @discardableResult
    func stop() async -> String {
        guard let active = activeTask else {
            return "❌ No active task to stop"
        }

        let duration = Date().timeIntervalSince(active.ts)
        let event = ChronosEvent.stop(
            task: active.task ?? "Unknown",
            duration: duration,
            startId: active.id
        )

        await eventLog.append(event)

        return "✅ Stopped \(active.task ?? "Unknown") — duration \(event.formattedDuration ?? "")"
    }

    /// LOG - Record a completed task retroactively
    @discardableResult
    func log(task: String, duration: TimeInterval, tags: [String] = []) async -> String {
        let event = ChronosEvent.log(task: task, duration: duration, tags: tags)
        await eventLog.append(event)

        let tagsStr = tags.isEmpty ? "" : " — tagged \(tags.map { "#\($0)" }.joined(separator: " "))"
        return "✅ Logged \(task), \(event.formattedDuration ?? "")\(tagsStr)"
    }

    /// NOTE - Add context to a task
    @discardableResult
    func note(text: String, linkedTo: String? = nil) async -> String {
        let linkedId = linkedTo ?? activeTask?.id

        guard let linkedId = linkedId else {
            return "❌ No active task and no event ID specified"
        }

        guard let linkedEvent = eventLog.event(withId: linkedId) else {
            return "❌ Event \(linkedId) not found"
        }

        let event = ChronosEvent.note(text: text, linkedTo: linkedId)
        await eventLog.append(event)

        let taskName = linkedEvent.task ?? linkedId
        return "✅ Note added to \(taskName)"
    }

    /// LINK - Connect events to external entities
    @discardableResult
    func link(from fromId: String, to toId: String, relation: String = "related_to") async -> String {
        guard eventLog.event(withId: fromId) != nil else {
            return "❌ Event \(fromId) not found"
        }

        let event = ChronosEvent.link(from: fromId, to: toId, relation: relation)
        await eventLog.append(event)

        return "✅ Linked \(fromId) → \(toId) (\(relation))"
    }

    // MARK: - Query Helpers (Synchronous - already in memory)

    var todayEvents: [ChronosEvent] {
        eventLog.eventsToday
    }

    func events(for date: Date) -> [ChronosEvent] {
        eventLog.events(for: date)
    }

    var totalTimeToday: TimeInterval {
        eventLog.totalTimeToday
    }

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

    var statusMessage: String {
        if let active = activeTask,
           let elapsed = activeTaskElapsedFormatted {
            return "⏱ Active: \(active.task ?? "Unknown") (elapsed \(elapsed))"
        } else {
            return "⏸ No active task"
        }
    }

    // MARK: - Tasks Summary

    var taskNames: [String] {
        Array(Set(events.compactMap { $0.task })).sorted()
    }

    var allTags: [String] {
        let tagSets = events.compactMap { $0.tags }
        return Array(Set(tagSets.flatMap { $0 })).sorted()
    }

    func totalTime(for task: String) -> TimeInterval {
        eventLog.events(forTask: task)
            .filter { $0.event == .stop || $0.event == .log }
            .compactMap { $0.durationSec }
            .reduce(0, +)
    }

    // MARK: - Export

    var exportURL: URL {
        eventLog.exportLog()
    }
}

// MARK: - Convenience (Non-async wrappers for SwiftUI)

extension ChronosState {
    /// Start task (SwiftUI-friendly - fire and forget)
    func start(task: String, tags: [String] = []) {
        Task {
            _ = await start(task: task, tags: tags)
        }
    }

    /// Stop task (SwiftUI-friendly)
    func stop() {
        Task {
            _ = await stop()
        }
    }

    /// Log task (SwiftUI-friendly)
    func log(task: String, duration: TimeInterval, tags: [String] = []) {
        Task {
            _ = await log(task: task, duration: duration, tags: tags)
        }
    }

    /// Note (SwiftUI-friendly)
    func note(text: String, linkedTo: String? = nil) {
        Task {
            _ = await note(text: text, linkedTo: linkedTo)
        }
    }
}
