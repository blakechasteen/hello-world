# Community Platform - Quick Start Guide

**Date**: 2025-11-18
**Status**: Architecture & Schema Complete

This guide shows you how to build the complete community platform based on the architecture and database schema.

---

## What We Have

✅ **Complete Architecture** ([COMMUNITY_PLATFORM_ARCHITECTURE.md](COMMUNITY_PLATFORM_ARCHITECTURE.md))
- Hybrid design (Forum + Social + Chat + Q&A)
- Plugin system architecture
- Real-time features
- Moderation & gamification
- Technology stack defined

✅ **Complete Database Schema** ([COMMUNITY_DATABASE_SCHEMA.md](COMMUNITY_DATABASE_SCHEMA.md))
- 22 PostgreSQL tables
- Neo4j social graph schema
- Redis caching strategy
- Elasticsearch indexes

---

## Implementation Roadmap

### Phase 1: Core Backend (Week 1-2)

#### 1.1 Setup Project Structure
```bash
mkdir -p community/backend/{models,api,auth,plugins,websocket}
cd community/backend
python -m venv .venv
source .venv/bin/activate
```

#### 1.2 Install Dependencies
```bash
pip install fastapi uvicorn[standard] sqlalchemy alembic asyncpg
pip install python-jose passlib bcrypt pydantic-settings
pip install redis neo4j elasticsearch
pip install python-socketio celery
```

#### 1.3 Create Database Models
Based on `COMMUNITY_DATABASE_SCHEMA.md`, create SQLAlchemy models:

**`backend/models/user.py`**:
```python
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import uuid

class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    # Profile
    display_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(Text)

    # Stats
    reputation_score = Column(Integer, default=0)
    trust_level = Column(Integer, default=0)

    # ... (see schema for complete columns)
```

Create models for:
- [x] User
- [x] Community
- [x] CommunityMember
- [x] Post
- [x] Comment
- [x] Vote
- [x] Message
- [x] Conversation
- [x] Notification
- [x] Plugin

#### 1.4 Create Alembic Migration
```bash
alembic init alembic
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

#### 1.5 Build Core API

**`backend/main.py`** - FastAPI app:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Community Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running", "version": "1.0.0"}
```

**`backend/api/users.py`** - User endpoints:
```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{username}")
async def get_user(username: str, db: AsyncSession = Depends(get_db)):
    # Implementation
    pass

@router.get("/{username}/posts")
async def get_user_posts(username: str):
    # Implementation
    pass
```

Create API routers for:
- [x] `/api/auth` - Login, register, OAuth
- [x] `/api/users` - User profiles, followers
- [x] `/api/communities` - CRUD, join/leave
- [x] `/api/posts` - Create, vote, delete
- [x] `/api/comments` - Threaded comments
- [x] `/api/messages` - Direct messages, conversations
- [x] `/api/notifications` - Get, mark read
- [x] `/api/search` - Full-text search

### Phase 2: Real-Time Features (Week 3)

#### 2.1 Setup Socket.io
```bash
pip install python-socketio aioredis
```

**`backend/websocket/server.py`**:
```python
import socketio
from aioredis import Redis

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    client_manager=socketio.AsyncRedisManager('redis://localhost')
)

@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")

@sio.event
async def join_room(sid, data):
    room = data['room']
    sio.enter_room(sid, room)
    await sio.emit('user_joined', {'user': data['user']}, room=room)

@sio.event
async def send_message(sid, data):
    room = data['room']
    await sio.emit('message_received', data, room=room)
```

#### 2.2 Integrate with FastAPI
```python
from fastapi import FastAPI
import socketio

sio = socketio.AsyncServer(async_mode='asgi')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)
```

#### 2.3 Implement Real-Time Features
- [x] Online presence
- [x] Live notifications
- [x] Chat messages
- [x] Post updates
- [x] Typing indicators

### Phase 3: Plugin System (Week 4)

#### 3.1 Create Plugin Base Class

**`backend/plugins/base.py`**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class PluginBase(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize plugin resources"""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Cleanup plugin resources"""
        pass

    def register_hook(self, hook_name: str, handler: callable):
        """Register a hook handler"""
        pass
```

#### 3.2 Create Plugin Manager

**`backend/plugins/manager.py`**:
```python
class PluginManager:
    def __init__(self):
        self._plugins = {}
        self._hooks = {}

    async def load_plugin(self, plugin_id: str, config: Dict):
        # Load and initialize plugin
        pass

    async def trigger_hook(self, hook_name: str, context: Dict):
        # Execute all registered handlers for hook
        pass
```

#### 3.3 Create Example Plugins

Create plugins for:
- [x] **Polls** - Create and vote on polls
- [x] **Events** - Community events calendar
- [x] **Marketplace** - Buy/sell within community
- [x] **Gamification** - Points, badges, levels
- [x] **Moderation** - Auto-mod, spam detection
- [x] **Analytics** - User behavior tracking

### Phase 4: Frontend (Week 5-6)

#### 4.1 Setup React Project
```bash
cd community/frontend
npm create vite@latest . -- --template react-ts
npm install
```

#### 4.2 Install Dependencies
```bash
npm install react-router-dom @tanstack/react-query
npm install @headlessui/react @heroicons/react
npm install tailwindcss autoprefixer postcss
npm install axios socket.io-client
npm install zustand react-hook-form zod
npm install @tiptap/react @tiptap/starter-kit
```

#### 4.3 Create Component Structure
```
frontend/src/
├── components/
│   ├── layout/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── Footer.tsx
│   ├── posts/
│   │   ├── PostCard.tsx
│   │   ├── PostList.tsx
│   │   ├── CreatePost.tsx
│   │   └── CommentThread.tsx
│   ├── communities/
│   │   ├── CommunityCard.tsx
│   │   ├── CommunityHeader.tsx
│   │   └── MemberList.tsx
│   └── users/
│       ├── UserProfile.tsx
│       ├── UserCard.tsx
│       └── FollowerList.tsx
├── pages/
│   ├── Home.tsx
│   ├── Community.tsx
│   ├── Post.tsx
│   ├── Profile.tsx
│   └── Messages.tsx
├── hooks/
│   ├── useAuth.ts
│   ├── usePosts.ts
│   ├── useWebSocket.ts
│   └── useNotifications.ts
├── services/
│   ├── api.ts
│   ├── websocket.ts
│   └── auth.ts
└── store/
    ├── authStore.ts
    ├── uiStore.ts
    └── notificationStore.ts
```

#### 4.4 Implement Key Pages

**Home Feed**:
- Algorithmic or chronological feed
- Filter by communities
- Infinite scroll
- Real-time updates

**Community Page**:
- Community header
- Post list
- Sidebar (rules, mods, stats)
- Join/leave button

**Post Detail**:
- Post content
- Voting buttons
- Threaded comments
- Share buttons

**User Profile**:
- Profile header
- Posts tab
- Comments tab
- Followers/following

**Messages**:
- Conversation list
- Chat interface
- Real-time updates

### Phase 5: Search & Discovery (Week 7)

#### 5.1 Setup Elasticsearch
```bash
docker run -d -p 9200:9200 elasticsearch:8.11.0
```

#### 5.2 Create Indexing Service
```python
from elasticsearch import AsyncElasticsearch

class SearchService:
    def __init__(self):
        self.es = AsyncElasticsearch(['http://localhost:9200'])

    async def index_post(self, post):
        await self.es.index(
            index='posts',
            id=post.post_id,
            document={
                'title': post.title,
                'content': post.content,
                'author': post.author.username,
                'created_at': post.created_at
            }
        )

    async def search_posts(self, query: str):
        result = await self.es.search(
            index='posts',
            body={
                'query': {
                    'multi_match': {
                        'query': query,
                        'fields': ['title^2', 'content']
                    }
                }
            }
        )
        return result['hits']['hits']
```

#### 5.3 Implement Search UI
- Global search bar
- Search results page
- Filters (posts, users, communities)
- Autocomplete suggestions

### Phase 6: Moderation (Week 8)

#### 6.1 Create Moderation Tools
- Report queue
- Ban users
- Remove content
- Mod log
- Auto-mod rules

#### 6.2 Implement AI Moderation
```python
from transformers import pipeline

toxicity_classifier = pipeline("text-classification",
                               model="unitary/toxic-bert")

async def check_toxicity(text: str) -> float:
    result = toxicity_classifier(text)[0]
    return result['score'] if result['label'] == 'toxic' else 0
```

### Phase 7: Deployment (Week 9)

#### 7.1 Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:pass@postgres/community
    depends_on:
      - postgres
      - redis
      - neo4j

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  neo4j:
    image: neo4j:5.12
    environment:
      NEO4J_AUTH: neo4j/password

  elasticsearch:
    image: elasticsearch:8.11.0
```

#### 7.2 Kubernetes (Production)
- Deployment manifests
- Services & Ingress
- ConfigMaps & Secrets
- Horizontal Pod Autoscaler
- Persistent volumes

---

## Development Workflow

### 1. Local Development
```bash
# Start databases
docker-compose up postgres redis neo4j elasticsearch

# Backend
cd community/backend
source .venv/bin/activate
uvicorn main:app --reload

# Frontend
cd community/frontend
npm run dev
```

### 2. Testing
```bash
# Backend tests
pytest backend/tests/ -v --cov

# Frontend tests
npm test
```

### 3. CI/CD
- GitHub Actions for automated testing
- Docker builds on push to main
- Auto-deploy to staging
- Manual deploy to production

---

## Next Steps

1. **Week 1-2**: Build core backend (users, communities, posts)
2. **Week 3**: Add real-time features (WebSocket, notifications)
3. **Week 4**: Build plugin system with examples
4. **Week 5-6**: Build React frontend (all core pages)
5. **Week 7**: Add search & discovery
6. **Week 8**: Implement moderation tools
7. **Week 9**: Production deployment

---

## Resources

- **Architecture**: [COMMUNITY_PLATFORM_ARCHITECTURE.md](COMMUNITY_PLATFORM_ARCHITECTURE.md)
- **Database Schema**: [COMMUNITY_DATABASE_SCHEMA.md](COMMUNITY_DATABASE_SCHEMA.md)
- **LMS Reference**: Use the LMS backend as a reference (`backend/` directory)

---

**Estimated Timeline**: 9 weeks for MVP
**Team Size**: 2-3 developers
**Budget**: Open-source (core) + Premium plugins (marketplace)

---

**Author**: Claude Code
**Date**: 2025-11-18
