# HoloLoom Mobile Integration

Native iOS and Android client support for HoloLoom.

## Quick Links

- **[MOBILE.md](./MOBILE.md)** - Complete integration guide
- **[openapi.yaml](./openapi.yaml)** - REST API specification
- **[WEBSOCKET_PROTOCOL.md](./WEBSOCKET_PROTOCOL.md)** - WebSocket protocol
- **[api.py](./api.py)** - Backend mobile endpoints

## Example Clients

- **iOS (Swift):** [`examples/ios/HoloLoomClient.swift`](./examples/ios/HoloLoomClient.swift)
- **Android (Kotlin):** [`examples/android/HoloLoomClient.kt`](./examples/android/HoloLoomClient.kt)

## Quick Start

### 1. Start Mock Server (for testing)

```bash
cd HoloLoom/mobile
python mock_server.py --port 8000
```

### 2. Start Real Server (with full HoloLoom)

```bash
cd HoloLoom
python -m HoloLoom.web.app

# Or with mobile endpoints
python -c "
from HoloLoom.web.app import create_app
from HoloLoom.mobile.api import add_mobile_routes
import uvicorn

app = create_app()
add_mobile_routes(app)
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

### 3. Test API

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login_json \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Create session
curl -X POST http://localhost:8000/api/chat/sessions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Chat"}'

# Send message
curl -X POST http://localhost:8000/api/chat/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is Thompson Sampling?"}'
```

### 4. Test WebSocket

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c "ws://localhost:8000/ws/chat/SESSION_ID?token=YOUR_TOKEN"

# Send message
> {"type": "message", "text": "Hello!"}

# Receive streaming response
< {"type": "thinking", "message": "Processing..."}
< {"type": "response_chunk", "text": "Hello! ", "done": false}
< {"type": "response_chunk", "text": "How can I help?", "done": false}
< {"type": "response_chunk", "text": "", "done": true}
```

## Directory Structure

```
mobile/
├── README.md                          # This file
├── MOBILE.md                          # Complete integration guide
├── openapi.yaml                       # OpenAPI 3.0 specification
├── WEBSOCKET_PROTOCOL.md              # WebSocket protocol docs
├── api.py                             # Backend mobile endpoints
├── mock_server.py                     # Mock server for testing
│
└── examples/
    ├── ios/
    │   └── HoloLoomClient.swift       # iOS client SDK
    │
    └── android/
        └── HoloLoomClient.kt          # Android client SDK
```

## Features

### Authentication
- JWT-based authentication
- Refresh token support
- Secure token storage (Keychain/EncryptedSharedPreferences)

### Real-Time Chat
- WebSocket connection with auto-reconnect
- Streaming responses
- Typing indicators
- Heartbeat/ping-pong

### Offline Support
- Local message queue
- Background sync
- Conflict resolution
- CoreData (iOS) / Room (Android)

### Push Notifications
- APNs (iOS)
- FCM (Android)
- Notification preferences
- Quiet hours

### File Uploads
- Images, audio, documents
- Multipart form data
- Progress tracking
- File attachments in messages

## API Endpoints

### Authentication
```
POST   /api/auth/register      Register new user
POST   /api/auth/login         Login (returns JWT)
POST   /api/auth/logout        Logout
POST   /api/auth/refresh       Refresh access token
GET    /api/auth/me            Get current user
```

### Chat
```
GET    /api/chat/sessions                   List sessions
POST   /api/chat/sessions                   Create session
GET    /api/chat/sessions/{id}              Get session
DELETE /api/chat/sessions/{id}              Delete session
GET    /api/chat/sessions/{id}/messages     Get messages
POST   /api/chat/sessions/{id}/messages     Send message
```

### Files
```
POST   /api/files/upload       Upload file
GET    /api/files/{id}         Download file
DELETE /api/files/{id}         Delete file
```

### Sync
```
POST   /api/sync/pull          Pull server updates
POST   /api/sync/push          Push local changes
```

### Push Notifications
```
POST   /api/push/register      Register device
POST   /api/push/unregister    Unregister device
GET    /api/push/preferences   Get preferences
PUT    /api/push/preferences   Update preferences
```

### WebSocket
```
WS     /ws/chat/{session_id}?token={jwt}    Real-time chat
```

## Implementation Checklist

### iOS
- [ ] Copy `HoloLoomClient.swift` to project
- [ ] Enable Push Notifications capability
- [ ] Add Keychain for token storage
- [ ] Implement CoreData for offline storage
- [ ] Configure APNs certificates
- [ ] Add background modes (fetch, remote-notification)
- [ ] Implement UI with SwiftUI
- [ ] Add unit tests

### Android
- [ ] Add dependencies (OkHttp, kotlinx.serialization)
- [ ] Copy `HoloLoomClient.kt` to project
- [ ] Add Firebase for FCM
- [ ] Configure EncryptedSharedPreferences
- [ ] Implement Room database for offline
- [ ] Set up WorkManager for background sync
- [ ] Implement UI with Jetpack Compose
- [ ] Add unit tests

## Testing

### Unit Tests
```bash
# iOS
# Run in Xcode: Cmd+U

# Android
./gradlew test
```

### Integration Tests
```bash
# Start mock server
python mock_server.py --port 8000

# Run app against mock server
# iOS: Set baseURL to http://localhost:8000
# Android: Set baseURL to http://10.0.2.2:8000 (emulator)
```

### API Docs
```bash
# Start server
python mock_server.py

# Open browser
open http://localhost:8000/docs
```

## Deployment

### iOS
1. Configure signing in Xcode
2. Archive build (Product → Archive)
3. Upload to App Store Connect
4. Submit for review

### Android
1. Generate signed APK/AAB: `./gradlew bundleRelease`
2. Upload to Google Play Console
3. Submit for review

### Backend
```bash
# Docker
docker build -t hololoom-api .
docker run -p 8000:8000 hololoom-api

# Cloud (AWS/GCP/Azure)
# Deploy container with environment variables
```

## Environment Variables

```bash
# Backend
JWT_SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:pass@host/db
REDIS_URL=redis://host:6379
UPLOAD_DIR=/var/uploads

# iOS (Info.plist)
API_BASE_URL=https://api.hololoom.ai/v1

# Android (build.gradle.kts)
buildConfigField("String", "API_BASE_URL", "\"https://api.hololoom.ai/v1\"")
```

## Support

- **Documentation:** [MOBILE.md](./MOBILE.md)
- **Issues:** GitHub Issues
- **Email:** support@hololoom.ai

## License

MIT License
