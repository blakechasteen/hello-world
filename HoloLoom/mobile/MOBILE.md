# HoloLoom Mobile Integration Guide

Complete guide for building iOS and Android apps with the HoloLoom API.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Authentication](#authentication)
5. [REST API](#rest-api)
6. [WebSocket Protocol](#websocket-protocol)
7. [Offline Support](#offline-support)
8. [Push Notifications](#push-notifications)
9. [File Uploads](#file-uploads)
10. [iOS Implementation](#ios-implementation)
11. [Android Implementation](#android-implementation)
12. [Best Practices](#best-practices)
13. [Testing](#testing)
14. [Deployment](#deployment)

---

## Overview

HoloLoom provides a comprehensive mobile API for building native iOS and Android applications. The system uses:

- **REST API** for standard CRUD operations
- **WebSocket** for real-time chat with streaming responses
- **JWT authentication** for secure access
- **Offline sync** for mobile-first experience
- **Push notifications** for engagement

### Key Features

- Real-time chat with AI-powered responses
- Multi-scale embeddings and knowledge graph memory
- Tool execution via MCP protocol
- File upload (images, audio, documents)
- Offline message queueing and sync
- Push notifications (FCM & APNs)
- Dark mode support

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Mobile Apps                         │
│  ┌──────────────────┐   ┌──────────────────┐       │
│  │   iOS (Swift)    │   │  Android (Kotlin)│       │
│  │   ┌──────────┐   │   │   ┌──────────┐   │       │
│  │   │ SwiftUI  │   │   │   │ Compose  │   │       │
│  │   │  Views   │   │   │   │  Screens │   │       │
│  │   └────┬─────┘   │   │   └────┬─────┘   │       │
│  │        │         │   │        │         │       │
│  │   ┌────▼─────┐   │   │   ┌────▼─────┐   │       │
│  │   │ViewModels│   │   │   │ViewModels│   │       │
│  │   └────┬─────┘   │   │   └────┬─────┘   │       │
│  │        │         │   │        │         │       │
│  │   ┌────▼─────────────────────▼─────┐   │       │
│  │   │   HoloLoom Client SDK          │   │       │
│  │   │  • REST API • WebSocket        │   │       │
│  │   │  • Auth • Sync • Push          │   │       │
│  │   └────────────┬───────────────────┘   │       │
│  └────────────────┼───────────────────────┘       │
│                   │                               │
│  ┌────────────────▼───────────────────┐           │
│  │   Local Storage                    │           │
│  │  • CoreData/Room • Keychain        │           │
│  └────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
                      │
                      │ HTTPS/WSS
                      ▼
┌─────────────────────────────────────────────────────┐
│               HoloLoom Backend                      │
│  ┌──────────────────────────────────────┐           │
│  │   FastAPI Web Server                 │           │
│  │   ┌──────────┐    ┌──────────┐       │           │
│  │   │   REST   │    │WebSocket │       │           │
│  │   │    API   │    │  Server  │       │           │
│  │   └────┬─────┘    └────┬─────┘       │           │
│  │        └──────┬─────────┘             │           │
│  │               │                       │           │
│  │        ┌──────▼────────┐              │           │
│  │        │  Orchestrator │              │           │
│  │        │  • Features   │              │           │
│  │        │  • Memory     │              │           │
│  │        │  • Policy     │              │           │
│  │        └──────┬────────┘              │           │
│  │               │                       │           │
│  │        ┌──────▼────────┐              │           │
│  │        │  MCP Server   │              │           │
│  │        │  Tool Exec    │              │           │
│  │        └───────────────┘              │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input** → Mobile UI (SwiftUI/Compose)
2. **ViewModel** → Process user action
3. **Client SDK** → Send REST/WebSocket request
4. **Backend API** → Authenticate, validate
5. **Orchestrator** → Process through HoloLoom pipeline
6. **MCP Tools** → Execute selected tools
7. **Response** → Stream back via WebSocket
8. **ViewModel** → Update UI state
9. **Local Storage** → Persist for offline

---

## Getting Started

### Prerequisites

**iOS:**
- Xcode 14+
- iOS 15.0+
- Swift 5.5+

**Android:**
- Android Studio Arctic Fox+
- Android SDK 24+
- Kotlin 1.9+

**Backend:**
- Python 3.9+
- FastAPI
- Running HoloLoom server

### Quick Start (iOS)

1. **Add HoloLoomClient to your project:**

```swift
// Copy HoloLoomClient.swift to your project
// Or install via Swift Package Manager (if published)
```

2. **Initialize the client:**

```swift
import HoloLoomClient

let client = HoloLoomClient(baseURL: "https://api.hololoom.ai/v1")
```

3. **Login and start chatting:**

```swift
Task {
    // Login
    let response = try await client.login(
        username: "demo",
        password: "demo123"
    )

    // Create session
    let session = try await client.createSession(title: "My Chat")

    // Connect WebSocket
    let ws = HoloLoomWebSocket(
        sessionId: response.sessionId,
        token: response.accessToken
    )

    ws.onResponseChunk = { text, done in
        print(text, terminator: "")
    }

    ws.connect()
    ws.sendMessage("Hello, HoloLoom!")
}
```

### Quick Start (Android)

1. **Add dependencies to `build.gradle.kts`:**

```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

2. **Copy HoloLoomClient.kt to your project**

3. **Initialize and use:**

```kotlin
import com.hololoom.client.*

// In your ViewModel
class ChatViewModel : ViewModel() {
    private val client = HoloLoomClient(
        baseURL = "https://api.hololoom.ai/v1"
    )

    fun login(username: String, password: String) {
        viewModelScope.launch {
            val response = client.login(username, password)
            connectWebSocket(response.sessionId, response.accessToken)
        }
    }

    private fun connectWebSocket(sessionId: String, token: String) {
        val ws = HoloLoomWebSocket(sessionId, token, listener)
        ws.connect()
    }
}
```

---

## Authentication

HoloLoom uses **JWT (JSON Web Tokens)** for authentication.

### Login Flow

```mermaid
sequenceDiagram
    participant App
    participant API
    participant Server

    App->>API: POST /auth/login {username, password}
    API->>Server: Verify credentials
    Server->>API: User authenticated
    API->>App: {access_token, refresh_token, session_id}
    App->>App: Store tokens securely
    App->>API: Subsequent requests with Bearer token
```

### iOS Implementation

```swift
// Login
let response = try await client.login(
    username: "admin",
    password: "admin123",
    deviceId: UIDevice.current.identifierForVendor?.uuidString
)

// Store tokens securely in Keychain
KeychainHelper.save(token: response.accessToken, key: "access_token")
KeychainHelper.save(token: response.refreshToken, key: "refresh_token")

// Use token in requests (automatic in HoloLoomClient)
```

### Android Implementation

```kotlin
// Login
val response = client.login(
    username = "admin",
    password = "admin123",
    deviceId = Settings.Secure.getString(
        context.contentResolver,
        Settings.Secure.ANDROID_ID
    )
)

// Store tokens securely in EncryptedSharedPreferences
val encryptedPrefs = EncryptedSharedPreferences.create(
    "secure_prefs",
    MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC),
    context,
    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
)

encryptedPrefs.edit {
    putString("access_token", response.accessToken)
    putString("refresh_token", response.refreshToken)
}
```

### Token Refresh

When `access_token` expires (default: 24 hours), use `refresh_token`:

```swift
// iOS
let newToken = try await client.refreshToken(response.refreshToken)
```

```kotlin
// Android
val newToken = client.refreshToken(response.refreshToken)
```

### Logout

```swift
// iOS
try await client.logout()
KeychainHelper.delete(key: "access_token")
```

```kotlin
// Android
client.logout()
encryptedPrefs.edit { clear() }
```

---

## REST API

Full API reference: See [`openapi.yaml`](./openapi.yaml)

### Base URL

- **Production:** `https://api.hololoom.ai/v1`
- **Staging:** `https://staging.hololoom.ai/v1`
- **Local:** `http://localhost:8000`

### Common Endpoints

#### Authentication

```
POST   /auth/register      Register new user
POST   /auth/login         Login
POST   /auth/logout        Logout
POST   /auth/refresh       Refresh access token
GET    /auth/me            Get current user
```

#### Chat Sessions

```
GET    /chat/sessions                   List sessions
POST   /chat/sessions                   Create session
GET    /chat/sessions/{id}              Get session
DELETE /chat/sessions/{id}              Delete session
GET    /chat/sessions/{id}/messages     Get messages
POST   /chat/sessions/{id}/messages     Send message
```

#### Files

```
POST   /files/upload       Upload file
GET    /files/{id}         Download file
DELETE /files/{id}         Delete file
```

#### Sync

```
POST   /sync/pull          Pull server updates
POST   /sync/push          Push local changes
```

#### Push Notifications

```
POST   /push/register      Register device
POST   /push/unregister    Unregister device
GET    /push/preferences   Get preferences
PUT    /push/preferences   Update preferences
```

### Request Format

All requests use JSON:

```http
POST /chat/sessions HTTP/1.1
Host: api.hololoom.ai
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "title": "My Conversation"
}
```

### Response Format

Successful responses return 2xx status codes:

```json
{
  "session_id": "admin_1234567890.123",
  "title": "My Conversation",
  "created_at": "2025-01-15T10:30:00Z",
  "last_activity": "2025-01-15T10:30:00Z",
  "message_count": 0
}
```

Error responses return 4xx/5xx:

```json
{
  "error": "Unauthorized",
  "error_code": "AUTH_FAILED",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## WebSocket Protocol

For real-time chat with streaming responses.

**Full documentation:** See [`WEBSOCKET_PROTOCOL.md`](./WEBSOCKET_PROTOCOL.md)

### Connection

```
wss://api.hololoom.ai/ws/chat/{session_id}?token={jwt_token}
```

### Message Types

**Client → Server:**
- `message` - Send chat message
- `ping` - Heartbeat
- `typing` - Typing indicator
- `stop_generation` - Stop AI response

**Server → Client:**
- `connected` - Connection established
- `thinking` - Processing status
- `response_chunk` - Streaming response
- `response_complete` - Full response with trace
- `error` - Error occurred

### Example Chat Flow

```javascript
// Connect
ws = new WebSocket("wss://api.hololoom.ai/ws/chat/session_123?token=...")

// Send message
ws.send(JSON.stringify({
  type: "message",
  text: "What is Thompson Sampling?"
}))

// Receive streaming response
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data)

  switch (msg.type) {
    case "thinking":
      showStatus(msg.message)
      break

    case "response_chunk":
      appendText(msg.text)
      if (msg.done) {
        responseComplete()
      }
      break

    case "response_complete":
      showTrace(msg.trace)
      break
  }
}
```

---

## Offline Support

Mobile apps should work offline and sync when online.

### Strategy

1. **Local Storage:** SQLite/CoreData/Room
2. **Message Queue:** Queue messages when offline
3. **Background Sync:** Sync when connection restored
4. **Conflict Resolution:** Server wins, client wins, or merge

### iOS Implementation

```swift
// Local storage with CoreData
class MessageStore {
    func saveMessage(_ message: Message) {
        // Save to CoreData
    }

    func queueOfflineMessage(_ text: String, sessionId: String) {
        // Save with pending status
    }

    func getPendingMessages() -> [Message] {
        // Fetch messages with pending status
    }
}

// Sync when online
func syncWhenOnline() async {
    guard isOnline else { return }

    let pending = messageStore.getPendingMessages()

    for message in pending {
        do {
            try await client.sendMessage(
                sessionId: message.sessionId,
                text: message.content
            )
            messageStore.markSynced(message.id)
        } catch {
            print("Sync failed: \(error)")
        }
    }

    // Pull server updates
    let syncData = try await client.pullSync(
        lastSync: lastSyncTimestamp
    )

    // Merge into local database
    mergeSyncData(syncData)
}
```

### Android Implementation

```kotlin
// Local storage with Room
@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String,
    val sessionId: String,
    val content: String,
    val role: String,
    val timestamp: Long,
    val syncStatus: SyncStatus // PENDING, SYNCED, FAILED
)

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE syncStatus = 'PENDING'")
    suspend fun getPendingMessages(): List<MessageEntity>

    @Update
    suspend fun update(message: MessageEntity)
}

// Background sync with WorkManager
class SyncWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val client = HoloLoomClient()
        val dao = AppDatabase.getInstance(applicationContext).messageDao()

        val pending = dao.getPendingMessages()

        for (message in pending) {
            try {
                client.sendMessage(message.sessionId, message.content)
                dao.update(message.copy(syncStatus = SyncStatus.SYNCED))
            } catch (e: Exception) {
                return Result.retry()
            }
        }

        return Result.success()
    }
}

// Schedule periodic sync
val syncRequest = PeriodicWorkRequestBuilder<SyncWorker>(15, TimeUnit.MINUTES)
    .setConstraints(Constraints.Builder()
        .setRequiredNetworkType(NetworkType.CONNECTED)
        .build())
    .build()

WorkManager.getInstance(context).enqueueUniquePeriodicWork(
    "sync",
    ExistingPeriodicWorkPolicy.KEEP,
    syncRequest
)
```

---

## Push Notifications

Receive notifications when new messages arrive.

### iOS - APNs Setup

1. **Enable Push Notifications in Xcode:**
   - Target → Signing & Capabilities → + Capability → Push Notifications

2. **Register for notifications:**

```swift
import UserNotifications

func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
) -> Bool {
    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
        if granted {
            DispatchQueue.main.async {
                application.registerForRemoteNotifications()
            }
        }
    }
    return true
}

func application(
    _ application: UIApplication,
    didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
) {
    let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()

    Task {
        try await client.registerForPush(deviceToken: token, platform: "ios")
    }
}
```

3. **Handle notifications:**

```swift
func userNotificationCenter(
    _ center: UNUserNotificationCenter,
    didReceive response: UNNotificationResponse,
    withCompletionHandler completionHandler: @escaping () -> Void
) {
    let userInfo = response.notification.request.content.userInfo

    if let sessionId = userInfo["session_id"] as? String {
        // Navigate to chat session
        navigateToChat(sessionId: sessionId)
    }

    completionHandler()
}
```

### Android - FCM Setup

1. **Add Firebase to project:**
   - Download `google-services.json`
   - Add to `app/` directory

2. **Add dependencies:**

```kotlin
dependencies {
    implementation(platform("com.google.firebase:firebase-bom:32.7.0"))
    implementation("com.google.firebase:firebase-messaging-ktx")
}
```

3. **Create FCM service:**

```kotlin
class HoloLoomFirebaseMessagingService : FirebaseMessagingService() {
    override fun onNewToken(token: String) {
        super.onNewToken(token)

        // Register with backend
        CoroutineScope(Dispatchers.IO).launch {
            try {
                client.registerForPush(deviceToken = token, platform = "android")
            } catch (e: Exception) {
                Log.e("FCM", "Failed to register: ${e.message}")
            }
        }
    }

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)

        val notification = message.notification
        val data = message.data

        // Show notification
        showNotification(
            title = notification?.title ?: "New message",
            body = notification?.body ?: "",
            sessionId = data["session_id"]
        )
    }

    private fun showNotification(title: String, body: String, sessionId: String?) {
        val intent = Intent(this, MainActivity::class.java).apply {
            putExtra("session_id", sessionId)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }

        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent, PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, "chat_channel")
            .setContentTitle(title)
            .setContentText(body)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(Random.nextInt(), notification)
    }
}
```

4. **Add to AndroidManifest.xml:**

```xml
<service
    android:name=".HoloLoomFirebaseMessagingService"
    android:exported="false">
    <intent-filter>
        <action android:name="com.google.firebase.MESSAGING_EVENT" />
    </intent-filter>
</service>
```

---

## File Uploads

Upload images, audio, and documents.

### iOS

```swift
// Upload image from camera/gallery
func uploadImage(_ image: UIImage) async throws -> Attachment {
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        throw UploadError.invalidImage
    }

    let attachment = try await client.uploadFile(
        data: imageData,
        fileName: "photo.jpg",
        mimeType: "image/jpeg"
    )

    return attachment
}

// Use in message
let attachment = try await uploadImage(selectedImage)
let (userMsg, assistantMsg, trace) = try await client.sendMessage(
    sessionId: sessionId,
    text: "What's in this image?",
    attachments: [attachment.fileId]
)
```

### Android

```kotlin
// Upload image from gallery
fun uploadImage(uri: Uri) = viewModelScope.launch {
    val file = uriToFile(uri, context)

    try {
        val attachment = client.uploadFile(file, "image/jpeg")

        // Use in message
        client.sendMessage(
            sessionId = sessionId,
            text = "What's in this image?",
            attachments = listOf(attachment.fileId)
        )
    } catch (e: Exception) {
        Log.e("Upload", "Failed: ${e.message}")
    }
}

private fun uriToFile(uri: Uri, context: Context): File {
    val inputStream = context.contentResolver.openInputStream(uri)
    val file = File(context.cacheDir, "upload_${System.currentTimeMillis()}.jpg")

    inputStream?.use { input ->
        file.outputStream().use { output ->
            input.copyTo(output)
        }
    }

    return file
}
```

---

## iOS Implementation

### Project Structure

```
HoloLoom/
├── App/
│   └── HoloLoomApp.swift
├── Views/
│   ├── ChatView.swift
│   ├── HistoryView.swift
│   ├── SettingsView.swift
│   └── LoginView.swift
├── ViewModels/
│   ├── ChatViewModel.swift
│   └── AuthViewModel.swift
├── Services/
│   ├── HoloLoomClient.swift
│   ├── WebSocketService.swift
│   └── StorageService.swift
├── Models/
│   ├── Message.swift
│   ├── User.swift
│   └── Session.swift
└── Resources/
    └── Assets.xcassets
```

### SwiftUI Chat View

```swift
import SwiftUI

struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var messageText = ""

    var body: some View {
        VStack {
            // Messages list
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.messages) { message in
                        MessageRow(message: message)
                    }
                }
                .padding()
            }

            // Input field
            HStack {
                TextField("Type a message...", text: $messageText)
                    .textFieldStyle(.roundedBorder)

                Button(action: sendMessage) {
                    Image(systemName: "paperplane.fill")
                }
                .disabled(messageText.isEmpty)
            }
            .padding()
        }
        .navigationTitle("Chat")
        .onAppear {
            viewModel.connect()
        }
    }

    private func sendMessage() {
        viewModel.sendMessage(messageText)
        messageText = ""
    }
}

struct MessageRow: View {
    let message: Message

    var body: some View {
        HStack {
            if message.role == "user" {
                Spacer()
            }

            VStack(alignment: message.role == "user" ? .trailing : .leading) {
                Text(message.content)
                    .padding(12)
                    .background(
                        message.role == "user" ? Color.blue : Color.gray.opacity(0.2)
                    )
                    .foregroundColor(
                        message.role == "user" ? .white : .primary
                    )
                    .cornerRadius(16)

                Text(message.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if message.role == "assistant" {
                Spacer()
            }
        }
    }
}
```

### ViewModel

```swift
import Foundation
import Combine

@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var isLoading = false
    @Published var error: String?

    private let client = HoloLoomClient()
    private var webSocket: HoloLoomWebSocket?
    private var sessionId: String?

    func connect() {
        guard let sessionId = sessionId,
              let token = KeychainHelper.get(key: "access_token") else {
            return
        }

        webSocket = HoloLoomWebSocket(sessionId: sessionId, token: token)

        webSocket?.onResponseChunk = { [weak self] text, done in
            // Append to last message
            if !text.isEmpty {
                self?.appendToLastMessage(text)
            }
        }

        webSocket?.connect()
    }

    func sendMessage(_ text: String) {
        let userMessage = Message(
            id: UUID().uuidString,
            sessionId: sessionId ?? "",
            role: "user",
            content: text,
            timestamp: Date(),
            attachments: [],
            metadata: [:],
            status: "sending"
        )

        messages.append(userMessage)

        webSocket?.sendMessage(text)

        // Add placeholder for assistant response
        let assistantMessage = Message(
            id: UUID().uuidString,
            sessionId: sessionId ?? "",
            role: "assistant",
            content: "",
            timestamp: Date(),
            attachments: [],
            metadata: [:],
            status: "generating"
        )

        messages.append(assistantMessage)
    }

    private func appendToLastMessage(_ text: String) {
        guard let lastIndex = messages.lastIndex(where: { $0.role == "assistant" }) else {
            return
        }

        var message = messages[lastIndex]
        message.content += text
        messages[lastIndex] = message
    }
}
```

---

## Android Implementation

### Project Structure

```
app/src/main/kotlin/com/hololoom/
├── ui/
│   ├── screens/
│   │   ├── ChatScreen.kt
│   │   ├── HistoryScreen.kt
│   │   ├── SettingsScreen.kt
│   │   └── LoginScreen.kt
│   └── theme/
│       └── Theme.kt
├── viewmodels/
│   ├── ChatViewModel.kt
│   └── AuthViewModel.kt
├── data/
│   ├── local/
│   │   ├── dao/
│   │   │   └── MessageDao.kt
│   │   └── database/
│   │       └── AppDatabase.kt
│   ├── remote/
│   │   └── HoloLoomClient.kt
│   └── repository/
│       └── MessageRepository.kt
└── models/
    ├── Message.kt
    ├── User.kt
    └── Session.kt
```

### Jetpack Compose Chat Screen

```kotlin
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ChatScreen(
    viewModel: ChatViewModel = viewModel()
) {
    val messages by viewModel.messages.collectAsState()
    var messageText by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Chat") })
        },
        bottomBar = {
            ChatInputField(
                text = messageText,
                onTextChange = { messageText = it },
                onSend = {
                    viewModel.sendMessage(messageText)
                    messageText = ""
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(messages) { message ->
                MessageRow(message = message)
            }
        }
    }

    LaunchedEffect(Unit) {
        viewModel.connect()
    }
}

@Composable
fun MessageRow(message: Message) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.role == "user") {
            Arrangement.End
        } else {
            Arrangement.Start
        }
    ) {
        Surface(
            color = if (message.role == "user") {
                MaterialTheme.colorScheme.primary
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            },
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.widthIn(max = 280.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                Text(
                    text = message.content,
                    color = if (message.role == "user") {
                        MaterialTheme.colorScheme.onPrimary
                    } else {
                        MaterialTheme.colorScheme.onSurface
                    }
                )

                Text(
                    text = message.timestamp,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}

@Composable
fun ChatInputField(
    text: String,
    onTextChange: (String) -> Unit,
    onSend: () -> Unit
) {
    Surface(
        tonalElevation = 3.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = text,
                onValueChange = onTextChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message...") }
            )

            IconButton(
                onClick = onSend,
                enabled = text.isNotBlank()
            ) {
                Icon(
                    imageVector = Icons.Default.Send,
                    contentDescription = "Send"
                )
            }
        }
    }
}
```

### ViewModel

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class ChatViewModel : ViewModel() {
    private val client = HoloLoomClient()
    private var webSocket: HoloLoomWebSocket? = null

    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading

    fun connect() {
        val sessionId = // Get from storage
        val token = // Get from storage

        webSocket = HoloLoomWebSocket(
            sessionId = sessionId,
            token = token,
            listener = object : HoloLoomWebSocket.ChatListener {
                override fun onConnected(sessionId: String) {
                    // Connected
                }

                override fun onThinking(message: String, stage: String) {
                    // Show status
                }

                override fun onResponseChunk(text: String, done: Boolean) {
                    if (text.isNotEmpty()) {
                        appendToLastMessage(text)
                    }
                }

                override fun onError(error: String) {
                    // Handle error
                }

                override fun onClosed() {
                    // Reconnect
                }
            }
        )

        webSocket?.connect()
    }

    fun sendMessage(text: String) {
        val userMessage = Message(
            id = UUID.randomUUID().toString(),
            sessionId = sessionId,
            role = "user",
            content = text,
            timestamp = Instant.now().toString(),
            status = "sending"
        )

        _messages.value = _messages.value + userMessage

        webSocket?.sendMessage(text)

        // Add placeholder
        val assistantMessage = Message(
            id = UUID.randomUUID().toString(),
            sessionId = sessionId,
            role = "assistant",
            content = "",
            timestamp = Instant.now().toString(),
            status = "generating"
        )

        _messages.value = _messages.value + assistantMessage
    }

    private fun appendToLastMessage(text: String) {
        val messages = _messages.value.toMutableList()
        val lastIndex = messages.indexOfLast { it.role == "assistant" }

        if (lastIndex != -1) {
            val message = messages[lastIndex]
            messages[lastIndex] = message.copy(content = message.content + text)
            _messages.value = messages
        }
    }

    override fun onCleared() {
        super.onCleared()
        webSocket?.disconnect()
    }
}
```

---

## Best Practices

### Security

1. **Never store tokens in plain text**
   - iOS: Use Keychain
   - Android: Use EncryptedSharedPreferences

2. **Validate SSL certificates**
   - Don't disable SSL pinning in production

3. **Sanitize user input**
   - Validate before sending to server

4. **Use app transport security (iOS)**
   - Require HTTPS connections

### Performance

1. **Implement pagination**
   - Load messages in batches

2. **Use lazy loading**
   - LazyVStack (iOS) / LazyColumn (Android)

3. **Cache images and files**
   - Use URLCache (iOS) / Coil/Glide (Android)

4. **Debounce typing indicators**
   - Avoid excessive WebSocket messages

5. **Background processing**
   - Use BackgroundTasks (iOS) / WorkManager (Android)

### UX

1. **Show loading states**
   - ProgressView, SkeletonView

2. **Handle errors gracefully**
   - Show retry buttons

3. **Offline indicators**
   - Visual cue when offline

4. **Optimistic UI updates**
   - Show message immediately, sync later

5. **Accessibility**
   - VoiceOver (iOS) / TalkBack (Android) support

---

## Testing

### Unit Tests

**iOS:**

```swift
import XCTest
@testable import HoloLoom

class HoloLoomClientTests: XCTestCase {
    var client: HoloLoomClient!

    override func setUp() {
        client = HoloLoomClient(baseURL: "http://localhost:8000")
    }

    func testLogin() async throws {
        let response = try await client.login(
            username: "admin",
            password: "admin123"
        )

        XCTAssertEqual(response.username, "admin")
        XCTAssertFalse(response.accessToken.isEmpty)
    }

    func testCreateSession() async throws {
        // Login first
        _ = try await client.login(username: "admin", password: "admin123")

        let session = try await client.createSession(title: "Test")
        XCTAssertEqual(session.title, "Test")
    }
}
```

**Android:**

```kotlin
import org.junit.Test
import org.junit.Assert.*
import kotlinx.coroutines.runBlocking

class HoloLoomClientTest {
    private val client = HoloLoomClient(baseURL = "http://localhost:8000")

    @Test
    fun testLogin() = runBlocking {
        val response = client.login("admin", "admin123")

        assertEquals("admin", response.username)
        assertFalse(response.accessToken.isEmpty())
    }

    @Test
    fun testCreateSession() = runBlocking {
        // Login first
        client.login("admin", "admin123")

        val session = client.createSession("Test")
        assertEquals("Test", session.title)
    }
}
```

### Integration Tests

Test with local server:

```bash
# Start local server
cd HoloLoom
python -m uvicorn HoloLoom.web.app:create_app --reload

# Run mobile tests
# iOS: Cmd+U in Xcode
# Android: ./gradlew test
```

---

## Deployment

### iOS App Store

1. **Configure signing:**
   - Xcode → Signing & Capabilities
   - Select team and provisioning profile

2. **Increment version:**
   - Update `CFBundleShortVersionString` in Info.plist

3. **Archive and submit:**
   - Product → Archive
   - Upload to App Store Connect
   - Submit for review

### Google Play Store

1. **Generate signed APK/AAB:**

```bash
./gradlew bundleRelease
```

2. **Upload to Play Console:**
   - Create app listing
   - Upload AAB
   - Submit for review

### Backend Deployment

**Using Docker:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY HoloLoom HoloLoom/

CMD ["uvicorn", "HoloLoom.web.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

**Deploy to cloud:**

```bash
# AWS, GCP, Azure, etc.
docker build -t hololoom-api .
docker push your-registry/hololoom-api:latest
```

---

## Additional Resources

- **OpenAPI Specification:** [`openapi.yaml`](./openapi.yaml)
- **WebSocket Protocol:** [`WEBSOCKET_PROTOCOL.md`](./WEBSOCKET_PROTOCOL.md)
- **Backend API:** [`api.py`](./api.py)
- **iOS Client:** [`examples/ios/HoloLoomClient.swift`](./examples/ios/HoloLoomClient.swift)
- **Android Client:** [`examples/android/HoloLoomClient.kt`](./examples/android/HoloLoomClient.kt)

### Support

- **Issues:** GitHub Issues
- **Email:** support@hololoom.ai
- **Documentation:** https://docs.hololoom.ai

---

## License

MIT License - See LICENSE file for details
