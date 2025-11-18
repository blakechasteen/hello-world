# LMS Orchestration Backend

FastAPI-based backend for the LMS orchestration ecosystem.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Database Services

```bash
make up
make health
```

### 4. Run Migrations

```bash
make migrate
make seed
```

### 5. Start Backend

```bash
# Development (with auto-reload)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python -m backend.main
```

### 6. Access API

- **API Documentation**: http://localhost:8000/docs
- **API Base**: http://localhost:8000/api
- **Health Check**: http://localhost:8000/health

## Project Structure

```
backend/
├── __init__.py
├── main.py                # FastAPI application
├── config.py              # Settings and configuration
├── database.py            # Database connections
│
├── models/                # SQLAlchemy models
│   ├── user.py
│   ├── institution.py
│   ├── course.py
│   ├── lesson.py
│   ├── assessment.py
│   └── plugin.py
│
├── api/                   # API routes
│   ├── __init__.py
│   ├── auth.py           # Login, register
│   ├── courses.py        # Course CRUD
│   ├── lessons.py        # Lesson CRUD
│   ├── assessments.py    # Assessment CRUD
│   └── submissions.py    # Student submissions
│
├── auth/                  # Authentication
│   ├── jwt.py            # JWT token handling
│   └── password.py       # Password hashing
│
├── plugins/               # Plugin system
│   ├── base.py           # Base plugin class
│   └── manager.py        # Plugin manager
│
├── services/              # Business logic
│   ├── knowledge_graph.py # Neo4j integration
│   ├── embeddings.py     # Qdrant integration
│   └── orchestrator.py   # Theme orchestration
│
└── README.md             # This file
```

## API Endpoints

### Authentication

```bash
# Register
POST /api/auth/register
{
  "email": "student@demo.edu",
  "password": "password123",
  "first_name": "John",
  "last_name": "Doe",
  "institution_id": "<uuid>"
}

# Login
POST /api/auth/login
{
  "email": "student@demo.edu",
  "password": "password123"
}
```

### Courses

```bash
# List courses
GET /api/courses/
Headers: Authorization: Bearer <token>

# Get course
GET /api/courses/{course_id}
Headers: Authorization: Bearer <token>
```

### Lessons

```bash
# List lessons for course
GET /api/lessons/course/{course_id}
Headers: Authorization: Bearer <token>
```

### Assessments

```bash
# List assessments for course
GET /api/assessments/course/{course_id}
Headers: Authorization: Bearer <token>
```

### Submissions

```bash
# Submit work
POST /api/submissions/
Headers: Authorization: Bearer <token>
{
  "assessment_id": "<uuid>",
  "content": "My essay content...",
  "attachments": []
}

# Get my submissions
GET /api/submissions/student/me
Headers: Authorization: Bearer <token>
```

## Configuration

Edit `.env` to configure:

```env
# Database
DATABASE_URL=postgresql://lms_user:lms_dev_password@localhost:5432/lms_dev

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=lms_dev_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
```

## Development

### Testing with curl

```bash
# Login and get token
TOKEN=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@demo.edu","password":"admin123"}' \
  | jq -r '.access_token')

# Use token to access API
curl http://localhost:8000/api/courses/ \
  -H "Authorization: Bearer $TOKEN"
```

### Testing with httpie

```bash
# Login
http POST :8000/api/auth/login email=admin@demo.edu password=admin123

# Use token
http :8000/api/courses/ "Authorization: Bearer <token>"
```

## Plugin System

### Creating a Plugin

1. Create plugin class:

```python
from backend.plugins import PluginBase, HookContext, HookResponse

class MyPlugin(PluginBase):
    async def initialize(self):
        # Setup resources
        self.register_hook("after_assessment_submit", self.on_submit)
        return True

    async def shutdown(self):
        # Cleanup
        return True

    async def on_submit(self, context: HookContext) -> HookResponse:
        # Handle event
        return HookResponse(success=True, data={"processed": True})
```

2. Load plugin:

```python
from backend.plugins import plugin_manager

await plugin_manager.load_plugin(
    plugin_id="my-plugin",
    plugin_path="./plugins/my_plugin/plugin.py",
    config={"setting": "value"}
)
```

3. Trigger hooks:

```python
from backend.plugins import plugin_manager, HookContext

context = HookContext(
    user_id="<uuid>",
    data={"submission_id": "<uuid>"}
)

responses = await plugin_manager.trigger_hook("after_assessment_submit", context)
```

## Database

### Models

All models use SQLAlchemy with async support:
- `Institution` - Educational institutions
- `User` - Students, instructors, admins
- `Course` - Course offerings
- `CourseEnrollment` - Enrollments
- `Lesson` - Individual lessons
- `Assessment` - Quizzes, exams, assignments
- `Submission` - Student work
- `Plugin` - Installed plugins
- `PluginConfiguration` - Plugin settings

### Migrations

```bash
# Create migration
make migrate-create
# Name: add_new_field

# Apply migrations
make migrate

# Rollback
make migrate-down
```

## Testing

```bash
# Run all tests
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Specific test file
pytest backend/tests/test_auth.py -v
```

## Production Deployment

### Docker

```bash
# Build
docker build -t lms-backend .

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  lms-backend
```

### Environment Variables

Production environment variables to set:
- `DATABASE_URL` - Production database
- `SECRET_KEY` - Strong random key
- `DEBUG=false`
- `ENVIRONMENT=production`
- `SENTRY_DSN` - Error tracking (optional)

### Security Checklist

- [ ] Change `SECRET_KEY` from default
- [ ] Use HTTPS in production
- [ ] Enable CORS only for trusted origins
- [ ] Set strong database passwords
- [ ] Enable rate limiting
- [ ] Configure Sentry for error tracking
- [ ] Regular security updates

## Contributing

See [PLUGIN_DEVELOPMENT_GUIDE.md](../PLUGIN_DEVELOPMENT_GUIDE.md) for plugin development.

## License

MIT
