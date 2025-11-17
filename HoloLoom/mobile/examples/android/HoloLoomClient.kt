/*
 * HoloLoomClient.kt
 * ==================
 * Kotlin client for HoloLoom mobile API (Android)
 *
 * Features:
 * - REST API client with coroutines
 * - WebSocket support for real-time chat
 * - JWT authentication
 * - Offline mode with Room database
 * - File upload
 *
 * Requirements:
 * - Android SDK 24+
 * - Kotlin 1.9+
 * - OkHttp 4.x
 * - kotlinx.serialization
 */

package com.hololoom.client

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.time.Instant
import java.util.*
import java.util.concurrent.TimeUnit

// MARK: - Models

@Serializable
data class User(
    val username: String,
    val email: String? = null,
    @SerialName("full_name") val fullName: String? = null,
    @SerialName("created_at") val createdAt: String? = null,
    @SerialName("avatar_url") val avatarUrl: String? = null
)

@Serializable
data class LoginResponse(
    @SerialName("access_token") val accessToken: String,
    @SerialName("refresh_token") val refreshToken: String? = null,
    @SerialName("token_type") val tokenType: String,
    @SerialName("expires_in") val expiresIn: Int,
    val username: String,
    @SerialName("session_id") val sessionId: String,
    val user: User
)

@Serializable
data class Message(
    @SerialName("message_id") val id: String,
    @SerialName("session_id") val sessionId: String,
    val role: String,
    val content: String,
    val timestamp: String,
    val attachments: List<Attachment> = emptyList(),
    val metadata: JsonObject = JsonObject(emptyMap()),
    val status: String
)

@Serializable
data class Attachment(
    @SerialName("file_id") val fileId: String,
    @SerialName("file_type") val fileType: String,
    @SerialName("file_name") val fileName: String,
    @SerialName("file_size") val fileSize: Int,
    val url: String
)

@Serializable
data class ChatSession(
    @SerialName("session_id") val id: String,
    val title: String,
    @SerialName("created_at") val createdAt: String,
    @SerialName("last_activity") val lastActivity: String,
    @SerialName("message_count") val messageCount: Int,
    val metadata: JsonObject = JsonObject(emptyMap())
)

@Serializable
data class WeavingTrace(
    val motifs: List<String>,
    val embeddings: EmbeddingInfo,
    @SerialName("tool_selection") val toolSelection: ToolSelection,
    @SerialName("execution_time_ms") val executionTimeMs: Double
) {
    @Serializable
    data class EmbeddingInfo(
        val scales: List<Int>,
        @SerialName("similarity_scores") val similarityScores: List<Double>? = null
    )

    @Serializable
    data class ToolSelection(
        @SerialName("selected_tool") val selectedTool: String,
        val confidence: Double,
        val strategy: String
    )
}

@Serializable
data class SendMessageResponse(
    @SerialName("user_message") val userMessage: Message,
    @SerialName("assistant_message") val assistantMessage: Message,
    val trace: WeavingTrace
)

// MARK: - API Client

class HoloLoomClient(
    private val baseURL: String = "https://api.hololoom.ai/v1"
) {
    private var accessToken: String? = null
    private var refreshToken: String? = null

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        isLenient = true
    }

    // MARK: - Authentication

    suspend fun login(
        username: String,
        password: String,
        deviceId: String? = null
    ): LoginResponse = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            put("username", username)
            put("password", password)
            put("device_id", deviceId ?: UUID.randomUUID().toString())
        }

        val request = Request.Builder()
            .url("$baseURL/auth/login")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.Unauthorized()
        }

        val loginResponse = json.decodeFromString<LoginResponse>(response.body!!.string())
        accessToken = loginResponse.accessToken
        refreshToken = loginResponse.refreshToken

        loginResponse
    }

    suspend fun logout() = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$baseURL/auth/logout")
            .post("".toRequestBody())
            .addAuthHeader()
            .build()

        client.newCall(request).execute()

        accessToken = null
        refreshToken = null
    }

    suspend fun getCurrentUser(): User = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$baseURL/auth/me")
            .get()
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.Unauthorized()
        }

        json.decodeFromString(response.body!!.string())
    }

    // MARK: - Chat Sessions

    suspend fun listSessions(
        limit: Int = 20,
        offset: Int = 0
    ): List<ChatSession> = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$baseURL/chat/sessions?limit=$limit&offset=$offset")
            .get()
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        val jsonResponse = json.parseToJsonElement(response.body!!.string()).jsonObject
        val sessions = jsonResponse["sessions"]?.jsonArray ?: JsonArray(emptyList())

        sessions.map { json.decodeFromJsonElement<ChatSession>(it) }
    }

    suspend fun createSession(title: String? = null): ChatSession = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            put("title", title ?: "New conversation")
        }

        val request = Request.Builder()
            .url("$baseURL/chat/sessions")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        json.decodeFromString(response.body!!.string())
    }

    suspend fun deleteSession(sessionId: String) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$baseURL/chat/sessions/$sessionId")
            .delete()
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.NotFound()
        }
    }

    // MARK: - Messages

    suspend fun getMessages(
        sessionId: String,
        limit: Int = 50
    ): List<Message> = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$baseURL/chat/sessions/$sessionId/messages?limit=$limit")
            .get()
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        val jsonResponse = json.parseToJsonElement(response.body!!.string()).jsonObject
        val messages = jsonResponse["messages"]?.jsonArray ?: JsonArray(emptyList())

        messages.map { json.decodeFromJsonElement<Message>(it) }
    }

    suspend fun sendMessage(
        sessionId: String,
        text: String,
        attachments: List<String> = emptyList()
    ): SendMessageResponse = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            put("text", text)
            put("attachments", JsonArray(attachments.map { JsonPrimitive(it) }))
        }

        val request = Request.Builder()
            .url("$baseURL/chat/sessions/$sessionId/messages")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        json.decodeFromString(response.body!!.string())
    }

    // MARK: - File Upload

    suspend fun uploadFile(
        file: File,
        mimeType: String
    ): Attachment = withContext(Dispatchers.IO) {
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                file.name,
                file.asRequestBody(mimeType.toMediaType())
            )
            .build()

        val request = Request.Builder()
            .url("$baseURL/files/upload")
            .post(requestBody)
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        json.decodeFromString(response.body!!.string())
    }

    // MARK: - Sync

    suspend fun pullSync(
        lastSync: Instant? = null,
        entityTypes: List<String> = listOf("messages", "sessions")
    ): JsonObject = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            if (lastSync != null) {
                put("last_sync", lastSync.toString())
            }
            put("entity_types", JsonArray(entityTypes.map { JsonPrimitive(it) }))
        }

        val request = Request.Builder()
            .url("$baseURL/sync/pull")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        json.parseToJsonElement(response.body!!.string()).jsonObject
    }

    suspend fun pushSync(
        messages: List<Message> = emptyList(),
        sessions: List<ChatSession> = emptyList()
    ): JsonObject = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            put("messages", JsonArray(messages.map { json.encodeToJsonElement(it) }))
            put("sessions", JsonArray(sessions.map { json.encodeToJsonElement(it) }))
        }

        val request = Request.Builder()
            .url("$baseURL/sync/push")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }

        json.parseToJsonElement(response.body!!.string()).jsonObject
    }

    // MARK: - Push Notifications

    suspend fun registerForPush(
        deviceToken: String,
        platform: String = "android"
    ) = withContext(Dispatchers.IO) {
        val body = buildJsonObject {
            put("device_token", deviceToken)
            put("platform", platform)
            put("device_id", UUID.randomUUID().toString())
        }

        val request = Request.Builder()
            .url("$baseURL/push/register")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .addAuthHeader()
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            throw APIException.ServerError()
        }
    }

    // MARK: - Private Helpers

    private fun Request.Builder.addAuthHeader(): Request.Builder {
        accessToken?.let {
            header("Authorization", "Bearer $it")
        }
        return this
    }
}

// MARK: - WebSocket Client

class HoloLoomWebSocket(
    private val sessionId: String,
    private val token: String,
    private val listener: ChatListener
) {
    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    interface ChatListener {
        fun onConnected(sessionId: String)
        fun onThinking(message: String, stage: String)
        fun onResponseChunk(text: String, done: Boolean)
        fun onError(error: String)
        fun onClosed()
    }

    fun connect() {
        val url = "wss://api.hololoom.ai/ws/chat/$sessionId?token=$token"
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                println("WebSocket connected")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                println("WebSocket error: ${t.message}")
                listener.onError(t.message ?: "Unknown error")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                println("WebSocket closed: $code - $reason")
                listener.onClosed()
            }
        })
    }

    fun sendMessage(text: String) {
        val message = buildJsonObject {
            put("type", "message")
            put("text", text)
            putJsonObject("metadata") {
                put("client_message_id", UUID.randomUUID().toString())
                put("timestamp", Instant.now().toString())
            }
        }

        webSocket?.send(message.toString())
    }

    fun sendHeartbeat() {
        val message = buildJsonObject {
            put("type", "ping")
        }

        webSocket?.send(message.toString())
    }

    private fun handleMessage(text: String) {
        try {
            val jsonMsg = json.parseToJsonElement(text).jsonObject
            val type = jsonMsg["type"]?.jsonPrimitive?.content ?: return

            when (type) {
                "connected" -> {
                    val sessionId = jsonMsg["session_id"]?.jsonPrimitive?.content ?: ""
                    listener.onConnected(sessionId)
                }

                "thinking" -> {
                    val message = jsonMsg["message"]?.jsonPrimitive?.content ?: ""
                    val stage = jsonMsg["stage"]?.jsonPrimitive?.content ?: ""
                    listener.onThinking(message, stage)
                }

                "response_chunk" -> {
                    val chunkText = jsonMsg["text"]?.jsonPrimitive?.content ?: ""
                    val done = jsonMsg["done"]?.jsonPrimitive?.boolean ?: false
                    listener.onResponseChunk(chunkText, done)
                }

                "error" -> {
                    val error = jsonMsg["error"]?.jsonPrimitive?.content ?: "Unknown error"
                    listener.onError(error)
                }

                "pong" -> {
                    // Heartbeat response
                }
            }
        } catch (e: Exception) {
            println("Error parsing message: ${e.message}")
        }
    }

    fun disconnect() {
        webSocket?.close(1000, "Client disconnect")
    }
}

// MARK: - Errors

sealed class APIException : Exception() {
    class Unauthorized : APIException()
    class NotFound : APIException()
    class ServerError : APIException()
    class NetworkError : APIException()
}

// MARK: - Example Usage

/*
fun main() = runBlocking {
    // Initialize client
    val client = HoloLoomClient(baseURL = "https://api.hololoom.ai/v1")

    // Login
    val loginResponse = client.login(username = "admin", password = "admin123")
    println("Logged in as: ${loginResponse.username}")

    // Create session
    val session = client.createSession(title = "My Chat")
    println("Created session: ${session.id}")

    // Send message (REST)
    val response = client.sendMessage(
        sessionId = session.id,
        text = "What is Thompson Sampling?"
    )
    println("Response: ${response.assistantMessage.content}")
    println("Tool used: ${response.trace.toolSelection.selectedTool}")

    // Real-time chat (WebSocket)
    val ws = HoloLoomWebSocket(
        sessionId = loginResponse.sessionId,
        token = loginResponse.accessToken,
        listener = object : HoloLoomWebSocket.ChatListener {
            override fun onConnected(sessionId: String) {
                println("WebSocket connected: $sessionId")
            }

            override fun onThinking(message: String, stage: String) {
                println("Status: $message ($stage)")
            }

            override fun onResponseChunk(text: String, done: Boolean) {
                if (text.isNotEmpty()) {
                    print(text)
                }
                if (done) {
                    println("\n--- Response complete ---")
                }
            }

            override fun onError(error: String) {
                println("Error: $error")
            }

            override fun onClosed() {
                println("WebSocket closed")
            }
        }
    )

    ws.connect()
    delay(1000) // Wait for connection

    ws.sendMessage("Hello, HoloLoom!")

    // Wait for response
    delay(5000)

    ws.disconnect()

    // Upload file
    val file = File("/path/to/image.jpg")
    val attachment = client.uploadFile(file, "image/jpeg")
    println("Uploaded: ${attachment.fileId}")

    // Sync
    val syncData = client.pullSync()
    println("Synced: ${syncData["sync_timestamp"]}")
}
*/

// MARK: - ViewModel Integration (Android)

/*
class ChatViewModel : ViewModel() {
    private val client = HoloLoomClient()
    private var webSocket: HoloLoomWebSocket? = null

    private val _messages = MutableLiveData<List<Message>>()
    val messages: LiveData<List<Message>> = _messages

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    fun login(username: String, password: String) {
        viewModelScope.launch {
            try {
                _isLoading.value = true
                val response = client.login(username, password)
                connectWebSocket(response.sessionId, response.accessToken)
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun sendMessage(sessionId: String, text: String) {
        webSocket?.sendMessage(text)
    }

    private fun connectWebSocket(sessionId: String, token: String) {
        webSocket = HoloLoomWebSocket(sessionId, token, object : HoloLoomWebSocket.ChatListener {
            override fun onConnected(sessionId: String) {
                // Handle connection
            }

            override fun onThinking(message: String, stage: String) {
                // Update UI with thinking status
            }

            override fun onResponseChunk(text: String, done: Boolean) {
                // Append to message content
            }

            override fun onError(error: String) {
                // Handle error
            }

            override fun onClosed() {
                // Handle disconnection
            }
        })

        webSocket?.connect()
    }

    override fun onCleared() {
        super.onCleared()
        webSocket?.disconnect()
    }
}
*/
