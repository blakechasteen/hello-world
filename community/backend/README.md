# Community Platform - Backend API

FastAPI-based backend for the Community Platform with plugin support, real-time features, and multi-database architecture.

## Features

- ✅ **User Authentication** - JWT-based auth with access & refresh tokens
- ✅ **Communities** - Create, join, and manage communities
- ✅ **Posts & Comments** - Threaded discussions with voting
- ✅ **Plugin System** - Extensible architecture with 100+ hooks
- ✅ **Real-time** - WebSocket support for live updates (coming soon)
- ✅ **Multi-Database** - PostgreSQL + Neo4j + Redis + Elasticsearch

## Architecture

See [../COMMUNITY_PLATFORM_ARCHITECTURE.md](../COMMUNITY_PLATFORM_ARCHITECTURE.md) for complete architecture documentation.

## Requirements

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Neo4j 5+ (optional, for social graph)
- Elasticsearch 8+ (optional, for search)

## Quick Start

### 1. Install Dependencies

```bash
cd community/backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Start Database Services

Using Docker Compose (recommended):

```bash
cd ../..  # Return to project root
docker-compose up -d postgres redis
```

Or install manually:
- PostgreSQL: https://www.postgresql.org/download/
- Redis: https://redis.io/download/

### 4. Initialize Database

```bash
# Create database
createdb community

# Run migrations (once Alembic is set up)
alembic upgrade head
```

### 5. Run Server

```bash
# Development mode (with auto-reload)
uvicorn main:app --reload --port 8000

# Or use the script
python main.py
```

API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication (`/api/v1/auth`)

- `POST /auth/register` - Register new user
- `POST /auth/login` - Login with username/password
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Get current user info
- `POST /auth/logout` - Logout (invalidate token)

### Users (`/api/v1/users`)

- `GET /users/{username}` - Get user profile
- `PATCH /users/me` - Update current user profile
- `GET /users/{username}/posts` - Get user's posts
- `GET /users/{username}/comments` - Get user's comments

### Communities (`/api/v1/communities`)

- `POST /communities` - Create community
- `GET /communities` - List communities
- `GET /communities/{name}` - Get community details
- `POST /communities/{name}/join` - Join community
- `POST /communities/{name}/leave` - Leave community

### Posts (`/api/v1/posts`)

- `POST /posts` - Create post
- `GET /posts` - List posts (with filters)
- `GET /posts/{id}` - Get post details
- `POST /posts/{id}/vote` - Vote on post (up/down)

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
│
├── core/                  # Core utilities
│   ├── config.py          # Configuration
│   └── database.py        # Database connection
│
├── models/                # SQLAlchemy models
│   ├── user.py            # User model
│   ├── community.py       # Community models
│   ├── post.py            # Post model
│   ├── comment.py         # Comment model
│   ├── vote.py            # Vote model
│   ├── message.py         # Message models
│   ├── notification.py    # Notification model
│   └── plugin.py          # Plugin models
│
├── auth/                  # Authentication
│   ├── security.py        # JWT & password hashing
│   ├── dependencies.py    # Auth dependencies
│   └── schemas.py         # Pydantic schemas
│
├── api/                   # API endpoints
│   ├── auth.py            # Auth endpoints
│   ├── users.py           # User endpoints
│   ├── communities.py     # Community endpoints
│   └── posts.py           # Post endpoints
│
└── plugins/               # Plugin system
    ├── base.py            # Plugin base class
    ├── manager.py         # Plugin manager
    └── examples/          # Example plugins
        └── polls_plugin.py
```

## Plugin System

### Creating a Plugin

```python
from plugins.base import PluginBase, PluginType, HookType, PluginContext

class MyPlugin(PluginBase):
    @property
    def name(self) -> str:
        return "My Plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Plugin description"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.CONTENT

    async def initialize(self) -> bool:
        # Register hooks
        self.register_hook(HookType.AFTER_POST_CREATE, self._on_post_created)
        return True

    async def shutdown(self) -> bool:
        # Cleanup
        return True

    async def _on_post_created(self, context: PluginContext):
        post_id = context.get("post_id")
        # Do something with the post
        return context
```

### Loading a Plugin

```python
from plugins.manager import plugin_manager

await plugin_manager.load_plugin(
    plugin_id="my_plugin",
    plugin_module="plugins.examples.my_plugin",
    plugin_class="MyPlugin",
    config={"setting": "value"},
)
```

### Available Hooks

**Content Hooks:**
- `BEFORE_POST_CREATE`, `AFTER_POST_CREATE`
- `BEFORE_POST_UPDATE`, `AFTER_POST_UPDATE`
- `POST_RENDER`, `POST_VOTE`

**User Hooks:**
- `USER_REGISTER`, `USER_LOGIN`, `USER_LOGOUT`
- `USER_REPUTATION_CHANGE`

**Community Hooks:**
- `COMMUNITY_CREATE`, `COMMUNITY_JOIN`, `COMMUNITY_LEAVE`

**Moderation Hooks:**
- `CONTENT_REPORT`, `CONTENT_APPROVE`, `CONTENT_REMOVE`
- `USER_BAN`, `USER_UNBAN`

See `plugins/base.py` for complete list.

## Database Models

### User
- Authentication (username, email, password)
- Profile (bio, avatar, reputation, trust level)
- Stats (posts, comments, followers, following)

### Community
- Basic info (name, description, icon, banner)
- Type (public, private, restricted)
- Stats (members, posts)

### Post
- Content (title, body, media)
- Type (text, link, image, video, poll, question)
- Voting (upvotes, downvotes, score)
- Engagement (comments, views, shares)

### Comment
- Threaded structure (parent, path, depth)
- Content (text, HTML)
- Voting

See [../COMMUNITY_DATABASE_SCHEMA.md](../COMMUNITY_DATABASE_SCHEMA.md) for complete schema.

## Testing

```bash
# Run tests (once test suite is created)
pytest

# With coverage
pytest --cov=. --cov-report=html
```

## Deployment

See [../COMMUNITY_QUICKSTART.md](../COMMUNITY_QUICKSTART.md) for deployment guide.

## License

MIT License - See LICENSE file

## Author

Built with Claude Code - 2025-11-18
