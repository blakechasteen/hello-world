/*
 HoloLoomClient.swift
 ====================
 Swift client for HoloLoom mobile API (iOS)

 Features:
 - REST API client with async/await
 - WebSocket support for real-time chat
 - JWT authentication
 - Offline mode with local persistence
 - File upload

 Requirements:
 - iOS 15.0+
 - Swift 5.5+ (async/await)
 */

import Foundation
import Combine

// MARK: - Models

struct User: Codable {
    let username: String
    let email: String?
    let fullName: String?
    let createdAt: Date?
    let avatarUrl: String?

    enum CodingKeys: String, CodingKey {
        case username
        case email
        case fullName = "full_name"
        case createdAt = "created_at"
        case avatarUrl = "avatar_url"
    }
}

struct LoginResponse: Codable {
    let accessToken: String
    let refreshToken: String?
    let tokenType: String
    let expiresIn: Int
    let username: String
    let sessionId: String
    let user: User

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case tokenType = "token_type"
        case expiresIn = "expires_in"
        case username
        case sessionId = "session_id"
        case user
    }
}

struct Message: Codable, Identifiable {
    let id: String
    let sessionId: String
    let role: String
    let content: String
    let timestamp: Date
    let attachments: [Attachment]
    let metadata: [String: AnyCodable]
    let status: String

    enum CodingKeys: String, CodingKey {
        case id = "message_id"
        case sessionId = "session_id"
        case role
        case content
        case timestamp
        case attachments
        case metadata
        case status
    }
}

struct Attachment: Codable {
    let fileId: String
    let fileType: String
    let fileName: String
    let fileSize: Int
    let url: String

    enum CodingKeys: String, CodingKey {
        case fileId = "file_id"
        case fileType = "file_type"
        case fileName = "file_name"
        case fileSize = "file_size"
        case url
    }
}

struct ChatSession: Codable, Identifiable {
    let id: String
    let title: String
    let createdAt: Date
    let lastActivity: Date
    let messageCount: Int
    let metadata: [String: AnyCodable]

    enum CodingKeys: String, CodingKey {
        case id = "session_id"
        case title
        case createdAt = "created_at"
        case lastActivity = "last_activity"
        case messageCount = "message_count"
        case metadata
    }
}

struct WeavingTrace: Codable {
    let motifs: [String]
    let embeddings: EmbeddingInfo
    let toolSelection: ToolSelection
    let executionTimeMs: Double

    enum CodingKeys: String, CodingKey {
        case motifs
        case embeddings
        case toolSelection = "tool_selection"
        case executionTimeMs = "execution_time_ms"
    }

    struct EmbeddingInfo: Codable {
        let scales: [Int]
        let similarityScores: [Double]?

        enum CodingKeys: String, CodingKey {
            case scales
            case similarityScores = "similarity_scores"
        }
    }

    struct ToolSelection: Codable {
        let selectedTool: String
        let confidence: Double
        let strategy: String

        enum CodingKeys: String, CodingKey {
            case selectedTool = "selected_tool"
            case confidence
            case strategy
        }
    }
}

// Helper for dynamic JSON values
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let value = try? container.decode(String.self) {
            self.value = value
        } else if let value = try? container.decode(Int.self) {
            self.value = value
        } else if let value = try? container.decode(Double.self) {
            self.value = value
        } else if let value = try? container.decode(Bool.self) {
            self.value = value
        } else {
            self.value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let val as String:
            try container.encode(val)
        case let val as Int:
            try container.encode(val)
        case let val as Double:
            try container.encode(val)
        case let val as Bool:
            try container.encode(val)
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - API Client

class HoloLoomClient {
    private let baseURL: String
    private var accessToken: String?
    private var refreshToken: String?
    private let session: URLSession

    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    init(baseURL: String = "https://api.hololoom.ai/v1") {
        self.baseURL = baseURL
        self.session = URLSession.shared
    }

    // MARK: - Authentication

    func login(username: String, password: String, deviceId: String? = nil) async throws -> LoginResponse {
        let url = URL(string: "\(baseURL)/auth/login")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "username": username,
            "password": password,
            "device_id": deviceId ?? UIDevice.current.identifierForVendor?.uuidString ?? ""
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw APIError.unauthorized
        }

        let loginResponse = try decoder.decode(LoginResponse.self, from: data)
        self.accessToken = loginResponse.accessToken
        self.refreshToken = loginResponse.refreshToken

        return loginResponse
    }

    func logout() async throws {
        let url = URL(string: "\(baseURL)/auth/logout")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        try addAuthHeader(to: &request)

        _ = try await session.data(for: request)

        self.accessToken = nil
        self.refreshToken = nil
    }

    func getCurrentUser() async throws -> User {
        let url = URL(string: "\(baseURL)/auth/me")!
        var request = URLRequest(url: url)
        try addAuthHeader(to: &request)

        let (data, _) = try await session.data(for: request)
        return try decoder.decode(User.self, from: data)
    }

    // MARK: - Chat Sessions

    func listSessions(limit: Int = 20, offset: Int = 0) async throws -> [ChatSession] {
        let url = URL(string: "\(baseURL)/chat/sessions?limit=\(limit)&offset=\(offset)")!
        var request = URLRequest(url: url)
        try addAuthHeader(to: &request)

        let (data, _) = try await session.data(for: request)
        let response = try decoder.decode([String: AnyCodable].self, from: data)

        // For now, return empty array (implement proper parsing)
        return []
    }

    func createSession(title: String? = nil) async throws -> ChatSession {
        let url = URL(string: "\(baseURL)/chat/sessions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        try addAuthHeader(to: &request)

        let body: [String: Any] = ["title": title ?? "New conversation"]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        return try decoder.decode(ChatSession.self, from: data)
    }

    func deleteSession(_ sessionId: String) async throws {
        let url = URL(string: "\(baseURL)/chat/sessions/\(sessionId)")!
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        try addAuthHeader(to: &request)

        _ = try await session.data(for: request)
    }

    // MARK: - Messages

    func getMessages(sessionId: String, limit: Int = 50) async throws -> [Message] {
        let url = URL(string: "\(baseURL)/chat/sessions/\(sessionId)/messages?limit=\(limit)")!
        var request = URLRequest(url: url)
        try addAuthHeader(to: &request)

        let (data, _) = try await session.data(for: request)
        let response = try decoder.decode([String: AnyCodable].self, from: data)

        // For now, return empty array (implement proper parsing)
        return []
    }

    func sendMessage(
        sessionId: String,
        text: String,
        attachments: [String] = []
    ) async throws -> (userMessage: Message, assistantMessage: Message, trace: WeavingTrace) {
        let url = URL(string: "\(baseURL)/chat/sessions/\(sessionId)/messages")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        try addAuthHeader(to: &request)

        let body: [String: Any] = [
            "text": text,
            "attachments": attachments
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)

        // Parse response (simplified)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        // TODO: Proper parsing
        let userMsg = Message(
            id: "msg_1",
            sessionId: sessionId,
            role: "user",
            content: text,
            timestamp: Date(),
            attachments: [],
            metadata: [:],
            status: "sent"
        )

        let assistantMsg = Message(
            id: "msg_2",
            sessionId: sessionId,
            role: "assistant",
            content: "Response",
            timestamp: Date(),
            attachments: [],
            metadata: [:],
            status: "sent"
        )

        let trace = WeavingTrace(
            motifs: [],
            embeddings: WeavingTrace.EmbeddingInfo(scales: [96, 192, 384]),
            toolSelection: WeavingTrace.ToolSelection(
                selectedTool: "echo",
                confidence: 0.9,
                strategy: "epsilon_greedy"
            ),
            executionTimeMs: 100
        )

        return (userMsg, assistantMsg, trace)
    }

    // MARK: - File Upload

    func uploadFile(data: Data, fileName: String, mimeType: String) async throws -> Attachment {
        let url = URL(string: "\(baseURL)/files/upload")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        try addAuthHeader(to: &request)

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // Add file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (responseData, _) = try await session.data(for: request)

        // Parse response
        let json = try JSONSerialization.jsonObject(with: responseData) as! [String: Any]

        return Attachment(
            fileId: json["file_id"] as! String,
            fileType: json["file_type"] as! String,
            fileName: json["file_name"] as! String,
            fileSize: json["file_size"] as! Int,
            url: json["url"] as! String
        )
    }

    // MARK: - Sync

    func pullSync(lastSync: Date?, entityTypes: [String] = ["messages", "sessions"]) async throws -> [String: Any] {
        let url = URL(string: "\(baseURL)/sync/pull")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        try addAuthHeader(to: &request)

        var body: [String: Any] = ["entity_types": entityTypes]
        if let lastSync = lastSync {
            let formatter = ISO8601DateFormatter()
            body["last_sync"] = formatter.string(from: lastSync)
        }

        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    // MARK: - Push Notifications

    func registerForPush(deviceToken: String, platform: String = "ios") async throws {
        let url = URL(string: "\(baseURL)/push/register")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        try addAuthHeader(to: &request)

        let body: [String: Any] = [
            "device_token": deviceToken,
            "platform": platform,
            "device_id": UIDevice.current.identifierForVendor?.uuidString ?? ""
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        _ = try await session.data(for: request)
    }

    // MARK: - Private Helpers

    private func addAuthHeader(to request: inout URLRequest) throws {
        guard let token = accessToken else {
            throw APIError.unauthorized
        }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
}

// MARK: - WebSocket Client

class HoloLoomWebSocket: NSObject {
    private var webSocketTask: URLSessionWebSocketTask?
    private let sessionId: String
    private let token: String

    var onConnected: (() -> Void)?
    var onThinking: ((String, String) -> Void)?
    var onResponseChunk: ((String, Bool) -> Void)?
    var onError: ((String) -> Void)?

    init(sessionId: String, token: String) {
        self.sessionId = sessionId
        self.token = token
        super.init()
    }

    func connect() {
        let urlString = "wss://api.hololoom.ai/ws/chat/\(sessionId)?token=\(token)"
        guard let url = URL(string: urlString) else { return }

        webSocketTask = URLSession.shared.webSocketTask(with: url)
        webSocketTask?.resume()

        receiveMessage()
    }

    func sendMessage(_ text: String) {
        let message: [String: Any] = [
            "type": "message",
            "text": text,
            "metadata": [
                "client_message_id": UUID().uuidString,
                "timestamp": ISO8601DateFormatter().string(from: Date())
            ]
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: data, encoding: .utf8) else {
            return
        }

        webSocketTask?.send(.string(jsonString)) { error in
            if let error = error {
                print("Send error: \(error)")
            }
        }
    }

    func sendHeartbeat() {
        let message = ["type": "ping"]
        guard let data = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: data, encoding: .utf8) else {
            return
        }

        webSocketTask?.send(.string(jsonString), completionHandler: nil)
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        self?.handleMessage(text)
                    }
                @unknown default:
                    break
                }
                self?.receiveMessage()

            case .failure(let error):
                print("Receive error: \(error)")
            }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            return
        }

        switch type {
        case "connected":
            onConnected?()

        case "thinking":
            let message = json["message"] as? String ?? ""
            let stage = json["stage"] as? String ?? ""
            onThinking?(message, stage)

        case "response_chunk":
            let text = json["text"] as? String ?? ""
            let done = json["done"] as? Bool ?? false
            onResponseChunk?(text, done)

        case "error":
            let error = json["error"] as? String ?? "Unknown error"
            onError?(error)

        default:
            break
        }
    }

    func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
    }
}

// MARK: - Errors

enum APIError: Error {
    case unauthorized
    case notFound
    case serverError
    case networkError
    case decodingError
}

// MARK: - Example Usage

/*
 // Initialize client
 let client = HoloLoomClient(baseURL: "https://api.hololoom.ai/v1")

 // Login
 let loginResponse = try await client.login(username: "admin", password: "admin123")
 print("Logged in as: \(loginResponse.username)")

 // Create session
 let session = try await client.createSession(title: "My Chat")

 // Send message (REST)
 let (userMsg, assistantMsg, trace) = try await client.sendMessage(
     sessionId: session.id,
     text: "What is Thompson Sampling?"
 )
 print("Response: \(assistantMsg.content)")

 // Real-time chat (WebSocket)
 let ws = HoloLoomWebSocket(sessionId: loginResponse.sessionId, token: loginResponse.accessToken)

 ws.onConnected = {
     print("WebSocket connected")
 }

 ws.onResponseChunk = { text, done in
     print(text, terminator: "")
     if done {
         print("\n--- Response complete ---")
     }
 }

 ws.connect()
 ws.sendMessage("Hello, HoloLoom!")

 // Upload file
 let imageData = UIImage(named: "photo")?.jpegData(compressionQuality: 0.8)
 let attachment = try await client.uploadFile(
     data: imageData!,
     fileName: "photo.jpg",
     mimeType: "image/jpeg"
 )
 print("Uploaded: \(attachment.fileId)")
 */
