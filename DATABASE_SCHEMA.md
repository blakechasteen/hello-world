# LMS Orchestration Database Schema
**Date**: 2025-11-17
**Version**: 1.0

Complete database schema for the LMS orchestration ecosystem across three database types:
- **PostgreSQL**: Relational data (users, courses, submissions)
- **Neo4j**: Knowledge graphs (student learning, concepts)
- **Qdrant**: Vector embeddings (semantic search)

---

## Table of Contents

1. [PostgreSQL Schema](#postgresql-schema)
2. [Neo4j Schema](#neo4j-schema)
3. [Qdrant Collections](#qdrant-collections)
4. [Migration Strategy](#migration-strategy)
5. [Indexing Strategy](#indexing-strategy)

---

## PostgreSQL Schema

### Core Tables

#### institutions

```sql
CREATE TABLE institutions (
    institution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_institutions_domain ON institutions(domain);
```

#### users

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id UUID NOT NULL REFERENCES institutions(institution_id),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('student', 'instructor', 'admin', 'ta')),
    avatar_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP
);

CREATE INDEX idx_users_institution ON users(institution_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
```

#### courses

```sql
CREATE TABLE courses (
    course_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id UUID NOT NULL REFERENCES institutions(institution_id),
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    semester VARCHAR(50),
    year INTEGER,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'draft')),
    settings JSONB DEFAULT '{}',
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(institution_id, code, semester, year)
);

CREATE INDEX idx_courses_institution ON courses(institution_id);
CREATE INDEX idx_courses_status ON courses(status);
CREATE INDEX idx_courses_created_by ON courses(created_by);
```

#### course_enrollments

```sql
CREATE TABLE course_enrollments (
    enrollment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('student', 'instructor', 'ta')),
    enrolled_at TIMESTAMP DEFAULT NOW(),
    dropped_at TIMESTAMP,
    grade VARCHAR(10),

    UNIQUE(course_id, user_id)
);

CREATE INDEX idx_enrollments_course ON course_enrollments(course_id);
CREATE INDEX idx_enrollments_user ON course_enrollments(user_id);
CREATE INDEX idx_enrollments_role ON course_enrollments(role);
```

#### lessons

```sql
CREATE TABLE lessons (
    lesson_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    content TEXT,
    theme VARCHAR(50) NOT NULL CHECK (theme IN ('lecture', 'flipped', 'project', 'socratic')),
    concept VARCHAR(255),  -- Primary concept covered
    prerequisites TEXT[],  -- Array of prerequisite concepts
    difficulty VARCHAR(50) DEFAULT 'intermediate',
    duration_minutes INTEGER,
    plugins TEXT[],  -- Array of required plugin IDs
    settings JSONB DEFAULT '{}',
    order_index INTEGER,
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP
);

CREATE INDEX idx_lessons_course ON lessons(course_id);
CREATE INDEX idx_lessons_concept ON lessons(concept);
CREATE INDEX idx_lessons_status ON lessons(status);
CREATE INDEX idx_lessons_order ON lessons(course_id, order_index);
```

#### assessments

```sql
CREATE TABLE assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES lessons(lesson_id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    instructions TEXT,
    assessment_type VARCHAR(50) NOT NULL,  -- quiz, exam, assignment, peer_review, etc.
    plugin_id VARCHAR(100),  -- Plugin handling this assessment
    concept VARCHAR(255),
    max_score FLOAT NOT NULL,
    passing_score FLOAT,
    time_limit_minutes INTEGER,
    attempts_allowed INTEGER DEFAULT 1,
    settings JSONB DEFAULT '{}',
    due_date TIMESTAMP,
    available_from TIMESTAMP,
    available_until TIMESTAMP,
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'closed')),
    created_by UUID NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_assessments_course ON assessments(course_id);
CREATE INDEX idx_assessments_lesson ON assessments(lesson_id);
CREATE INDEX idx_assessments_plugin ON assessments(plugin_id);
CREATE INDEX idx_assessments_due_date ON assessments(due_date);
```

#### submissions

```sql
CREATE TABLE submissions (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID NOT NULL REFERENCES assessments(assessment_id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    content TEXT,
    attachments TEXT[],
    word_count INTEGER,
    submitted_at TIMESTAMP DEFAULT NOW(),
    late BOOLEAN DEFAULT FALSE,
    score FLOAT,
    graded_at TIMESTAMP,
    graded_by UUID REFERENCES users(user_id),
    feedback TEXT,
    metadata JSONB DEFAULT '{}',

    UNIQUE(assessment_id, student_id, attempt_number)
);

CREATE INDEX idx_submissions_assessment ON submissions(assessment_id);
CREATE INDEX idx_submissions_student ON submissions(student_id);
CREATE INDEX idx_submissions_graded ON submissions(graded_at);
```

### Plugin-Specific Tables

#### peer_review_assignments

```sql
CREATE TABLE peer_review_assignments (
    assignment_id UUID PRIMARY KEY REFERENCES assessments(assessment_id) ON DELETE CASCADE,
    review_type VARCHAR(50) NOT NULL CHECK (review_type IN ('single_blind', 'double_blind', 'open')),
    rubric JSONB NOT NULL,
    reviews_per_submission INTEGER DEFAULT 3,
    enable_calibration BOOLEAN DEFAULT TRUE,
    calibration_threshold FLOAT DEFAULT 0.8,
    review_deadline TIMESTAMP,
    dispute_deadline TIMESTAMP,
    quality_weight FLOAT DEFAULT 0.7,
    completion_weight FLOAT DEFAULT 0.3,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### peer_reviews

```sql
CREATE TABLE peer_reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL REFERENCES submissions(submission_id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    scores JSONB NOT NULL,  -- Array of {criterion_id, score, feedback}
    total_score FLOAT NOT NULL,
    overall_feedback TEXT,
    quality_score FLOAT,  -- Quality of this review
    helpfulness_score FLOAT,
    timeliness_score FLOAT,
    status VARCHAR(50) DEFAULT 'assigned' CHECK (status IN ('assigned', 'in_progress', 'submitted', 'disputed', 'resolved')),
    submitted_at TIMESTAMP,
    time_spent_minutes INTEGER,
    metadata JSONB DEFAULT '{}',

    UNIQUE(submission_id, reviewer_id)
);

CREATE INDEX idx_peer_reviews_submission ON peer_reviews(submission_id);
CREATE INDEX idx_peer_reviews_reviewer ON peer_reviews(reviewer_id);
CREATE INDEX idx_peer_reviews_status ON peer_reviews(status);
```

#### calibration_scores

```sql
CREATE TABLE calibration_scores (
    calibration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    assignment_id UUID NOT NULL REFERENCES peer_review_assignments(assignment_id) ON DELETE CASCADE,
    calibration_score FLOAT NOT NULL,  -- 0.0-1.0
    attempts_completed INTEGER DEFAULT 0,
    passed BOOLEAN DEFAULT FALSE,
    last_attempt_at TIMESTAMP,

    UNIQUE(student_id, assignment_id)
);

CREATE INDEX idx_calibration_student ON calibration_scores(student_id);
CREATE INDEX idx_calibration_assignment ON calibration_scores(assignment_id);
```

### Analytics Tables

#### engagement_events

```sql
CREATE TABLE engagement_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    course_id UUID REFERENCES courses(course_id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES lessons(lesson_id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,  -- view, click, video_progress, etc.
    event_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW(),
    session_id UUID,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_engagement_student ON engagement_events(student_id);
CREATE INDEX idx_engagement_course ON engagement_events(course_id);
CREATE INDEX idx_engagement_type ON engagement_events(event_type);
CREATE INDEX idx_engagement_timestamp ON engagement_events(timestamp DESC);

-- Partition by month for performance
CREATE TABLE engagement_events_2025_11 PARTITION OF engagement_events
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
```

#### learning_analytics

```sql
CREATE TABLE learning_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    course_id UUID NOT NULL REFERENCES courses(course_id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    calculated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',

    UNIQUE(student_id, course_id, metric_name, calculated_at)
);

CREATE INDEX idx_analytics_student_course ON learning_analytics(student_id, course_id);
CREATE INDEX idx_analytics_metric ON learning_analytics(metric_name);
CREATE INDEX idx_analytics_calculated ON learning_analytics(calculated_at DESC);
```

### Plugin System Tables

#### plugins

```sql
CREATE TABLE plugins (
    plugin_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    category VARCHAR(50) NOT NULL,
    description TEXT,
    author VARCHAR(255),
    permissions TEXT[],
    hooks TEXT[],
    api_routes JSONB,
    ui_components JSONB,
    config_schema JSONB,
    status VARCHAR(50) DEFAULT 'inactive' CHECK (status IN ('active', 'inactive', 'deprecated')),
    installed_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(plugin_id, version)
);

CREATE INDEX idx_plugins_category ON plugins(category);
CREATE INDEX idx_plugins_status ON plugins(status);
```

#### plugin_configurations

```sql
CREATE TABLE plugin_configurations (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id UUID NOT NULL REFERENCES institutions(institution_id) ON DELETE CASCADE,
    plugin_id VARCHAR(100) NOT NULL REFERENCES plugins(plugin_id) ON DELETE CASCADE,
    config JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(institution_id, plugin_id)
);

CREATE INDEX idx_plugin_config_institution ON plugin_configurations(institution_id);
CREATE INDEX idx_plugin_config_plugin ON plugin_configurations(plugin_id);
```

---

## Neo4j Schema

### Node Labels

#### Student

```cypher
CREATE CONSTRAINT student_unique IF NOT EXISTS
FOR (s:Student) REQUIRE s.student_id IS UNIQUE;

CREATE INDEX student_institution IF NOT EXISTS
FOR (s:Student) ON (s.institution_id);

// Properties:
// - student_id: UUID
// - name: String
// - email: String
// - institution_id: UUID
// - created_at: DateTime
// - metadata: Map
```

#### Concept

```cypher
CREATE CONSTRAINT concept_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

CREATE INDEX concept_name IF NOT EXISTS
FOR (c:Concept) ON (c.name);

CREATE INDEX concept_domain IF NOT EXISTS
FOR (c:Concept) ON (c.domain);

// Properties:
// - concept_id: UUID
// - name: String
// - description: String
// - domain: String (e.g., "machine_learning", "statistics")
// - difficulty: String
// - created_at: DateTime
```

#### Resource

```cypher
CREATE CONSTRAINT resource_unique IF NOT EXISTS
FOR (r:Resource) REQUIRE r.resource_id IS UNIQUE;

// Properties:
// - resource_id: UUID
// - title: String
// - type: String (video, text, quiz, etc.)
// - url: String
// - created_at: DateTime
```

#### Instructor

```cypher
CREATE CONSTRAINT instructor_unique IF NOT EXISTS
FOR (i:Instructor) REQUIRE i.instructor_id IS UNIQUE;

// Properties:
// - instructor_id: UUID
// - name: String
// - email: String
// - institution_id: UUID
```

### Relationship Types

#### MASTERED

```cypher
// (Student)-[:MASTERED]->(Concept)
// Properties:
// - confidence: Float (0.0-1.0)
// - evidence: String
// - timestamp: DateTime
// - source: String (quiz, assignment, etc.)

CREATE INDEX mastered_confidence IF NOT EXISTS
FOR ()-[r:MASTERED]->() ON (r.confidence);
```

#### STRUGGLING_WITH

```cypher
// (Student)-[:STRUGGLING_WITH]->(Concept)
// Properties:
// - confidence: Float (0.0-1.0)
// - attempts: Integer
// - last_attempt: DateTime
// - error_patterns: List<String>
```

#### PREREQUISITE

```cypher
// (Concept)-[:PREREQUISITE]->(Concept)
// Properties:
// - importance: Float (0.0-1.0)
// - optional: Boolean
```

#### LEARNED_FROM

```cypher
// (Concept)-[:LEARNED_FROM]->(Resource)
// Properties:
// - timestamp: DateTime
// - effectiveness: Float
```

#### TAUGHT_BY

```cypher
// (Concept)-[:TAUGHT_BY]->(Instructor)
// Properties:
// - course_id: UUID
// - semester: String
// - timestamp: DateTime
```

#### STUDY_WITH

```cypher
// (Student)-[:STUDY_WITH]->(Student)
// Properties:
// - context: String (study_group, project, peer_review)
// - started_at: DateTime
// - effectiveness: Float
```

#### SIMILAR_TO

```cypher
// (Concept)-[:SIMILAR_TO]->(Concept)
// Properties:
// - similarity: Float (0.0-1.0)
// - type: String (semantic, structural, pedagogical)
```

### Example Queries

#### Find Learning Path

```cypher
// Find shortest learning path between two concepts
MATCH path = shortestPath(
    (start:Concept {name: "Linear Algebra"})-[:PREREQUISITE*]->(end:Concept {name: "Deep Learning"})
)
RETURN [node IN nodes(path) | node.name] AS learning_path
```

#### Get Student Mastery

```cypher
// Get all concepts mastered by student
MATCH (s:Student {student_id: $student_id})-[m:MASTERED]->(c:Concept)
WHERE m.confidence >= 0.7
RETURN c.name, c.domain, m.confidence, m.timestamp
ORDER BY m.timestamp DESC
```

#### Recommend Next Concepts

```cypher
// Recommend next concepts based on what student has mastered
MATCH (s:Student {student_id: $student_id})-[:MASTERED]->(mastered:Concept)
MATCH (mastered)-[:PREREQUISITE*0..1]->(next:Concept)
WHERE NOT (s)-[:MASTERED]->(next)
  AND NOT (s)-[:STRUGGLING_WITH]->(next)
WITH next, COUNT(DISTINCT mastered) AS prerequisites_met
ORDER BY prerequisites_met DESC
LIMIT 5
RETURN next.name, next.description, prerequisites_met
```

#### Find Struggling Students

```cypher
// Find students struggling with a concept
MATCH (s:Student)-[sw:STRUGGLING_WITH]->(c:Concept {name: "Backpropagation"})
WHERE sw.attempts >= 3
RETURN s.student_id, s.name, sw.attempts, sw.last_attempt
ORDER BY sw.attempts DESC
```

#### Collaborative Learning Network

```cypher
// Find students who can help each other
MATCH (student:Student {student_id: $student_id})-[:STRUGGLING_WITH]->(concept:Concept)
MATCH (helper:Student)-[:MASTERED]->(concept)
WHERE helper.institution_id = student.institution_id
  AND helper.student_id <> student.student_id
WITH student, concept, helper
OPTIONAL MATCH (student)-[:STUDY_WITH]-(helper)
WITH student, concept, helper, COUNT(*) AS already_connected
WHERE already_connected = 0
RETURN concept.name, helper.student_id, helper.name
LIMIT 5
```

---

## Qdrant Collections

### lessons_embeddings

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="lessons_embeddings",
    vectors_config=VectorParams(
        size=384,  # Sentence-transformers dimension
        distance=Distance.COSINE
    )
)

# Payload schema:
{
    "lesson_id": "uuid",
    "course_id": "uuid",
    "title": "string",
    "description": "string",
    "concept": "string",
    "theme": "string",
    "difficulty": "string",
    "keywords": ["string"],
    "created_at": "timestamp"
}
```

### submissions_embeddings

```python
client.create_collection(
    collection_name="submissions_embeddings",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Payload schema:
{
    "submission_id": "uuid",
    "assessment_id": "uuid",
    "student_id": "uuid",
    "excerpt": "string",  # First 500 characters
    "word_count": "integer",
    "submitted_at": "timestamp"
}
```

### feedback_embeddings

```python
client.create_collection(
    collection_name="feedback_embeddings",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Payload schema:
{
    "feedback_id": "uuid",
    "source_type": "string",  # peer_review, instructor, auto
    "source_id": "uuid",
    "sentiment": "float",  # -1.0 to 1.0
    "constructiveness": "float",  # 0.0 to 1.0
    "timestamp": "timestamp"
}
```

### concepts_embeddings

```python
client.create_collection(
    collection_name="concepts_embeddings",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Payload schema:
{
    "concept_id": "uuid",
    "name": "string",
    "description": "string",
    "domain": "string",
    "difficulty": "string",
    "related_concepts": ["string"]
}
```

### Example Searches

#### Find Similar Lessons

```python
# Find lessons similar to a query
results = client.search(
    collection_name="lessons_embeddings",
    query_vector=query_embedding,
    limit=10,
    query_filter={
        "must": [
            {"key": "difficulty", "match": {"value": "intermediate"}}
        ]
    }
)
```

#### Find Similar Student Work

```python
# Find similar submissions (for plagiarism detection or examples)
results = client.search(
    collection_name="submissions_embeddings",
    query_vector=submission_embedding,
    limit=5,
    query_filter={
        "must": [
            {"key": "assessment_id", "match": {"value": assessment_id}}
        ],
        "must_not": [
            {"key": "student_id", "match": {"value": current_student_id}}
        ]
    }
)
```

---

## Migration Strategy

### Initial Setup

```bash
# Run migrations
alembic upgrade head

# Seed initial data
python scripts/seed_institutions.py
python scripts/seed_concepts.py
```

### Version Control

```python
# migrations/versions/001_initial_schema.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    # Create tables
    op.create_table('institutions', ...)
    op.create_table('users', ...)
    # ...

def downgrade():
    # Drop tables in reverse order
    op.drop_table('users')
    op.drop_table('institutions')
```

### Data Migration

```python
# For major schema changes, use data migrations
def migrate_lessons_to_new_theme_system():
    """Migrate lessons from old single theme to new multi-theme"""
    connection = op.get_bind()
    connection.execute("""
        UPDATE lessons
        SET theme = CASE
            WHEN old_theme = 'traditional' THEN 'lecture'
            WHEN old_theme = 'active' THEN 'flipped'
            ELSE theme
        END
    """)
```

---

## Indexing Strategy

### PostgreSQL Indexes

```sql
-- Composite indexes for common queries
CREATE INDEX idx_submissions_student_assessment
ON submissions(student_id, assessment_id);

CREATE INDEX idx_engagement_student_timestamp
ON engagement_events(student_id, timestamp DESC);

CREATE INDEX idx_reviews_submission_status
ON peer_reviews(submission_id, status);

-- Partial indexes for active records
CREATE INDEX idx_active_courses
ON courses(institution_id)
WHERE status = 'active';

CREATE INDEX idx_published_lessons
ON lessons(course_id, order_index)
WHERE status = 'published';

-- GIN indexes for JSONB
CREATE INDEX idx_plugin_config_gin
ON plugin_configurations USING GIN (config);

CREATE INDEX idx_lesson_settings_gin
ON lessons USING GIN (settings);
```

### Neo4j Indexes

```cypher
// Composite indexes for relationship queries
CREATE INDEX student_mastery_composite IF NOT EXISTS
FOR (s:Student) ON (s.student_id, s.institution_id);

// Full-text search indexes
CREATE FULLTEXT INDEX concept_search IF NOT EXISTS
FOR (c:Concept) ON EACH [c.name, c.description];

// Range indexes for confidence scores
CREATE RANGE INDEX mastered_confidence_range IF NOT EXISTS
FOR ()-[m:MASTERED]->() ON (m.confidence);
```

### Qdrant Indexes

```python
# Qdrant automatically indexes payloads
# Create additional indexes for frequent filters
client.create_payload_index(
    collection_name="lessons_embeddings",
    field_name="difficulty",
    field_schema="keyword"
)

client.create_payload_index(
    collection_name="submissions_embeddings",
    field_name="submitted_at",
    field_schema="datetime"
)
```

---

## Backup Strategy

### PostgreSQL

```bash
# Daily backups
pg_dump -Fc lms_production > backups/lms_$(date +%Y%m%d).dump

# Point-in-time recovery
# Enable WAL archiving in postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/archive/%f'
```

### Neo4j

```bash
# Daily backups
neo4j-admin dump --database=neo4j --to=/backup/neo4j_$(date +%Y%m%d).dump

# Incremental backups with Enterprise Edition
neo4j-admin backup --backup-dir=/backup/incremental --name=daily
```

### Qdrant

```bash
# Snapshot backups
curl -X POST 'http://localhost:6333/collections/lessons_embeddings/snapshots'

# Download snapshot
curl 'http://localhost:6333/collections/lessons_embeddings/snapshots/snapshot.zip' \
  -o backup/qdrant_lessons_$(date +%Y%m%d).zip
```

---

## Performance Optimization

### Connection Pooling

```python
# PostgreSQL connection pool
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/lms',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)
```

### Query Optimization

```sql
-- Use EXPLAIN ANALYZE to identify slow queries
EXPLAIN ANALYZE
SELECT * FROM submissions
WHERE student_id = 'uuid'
AND assessment_id = 'uuid';

-- Add indexes based on query patterns
CREATE INDEX idx_custom ON table(column1, column2)
WHERE condition;
```

### Caching Strategy

```python
# Redis for query result caching
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_student_analytics(student_id):
    # Try cache first
    cached = cache.get(f"analytics:{student_id}")
    if cached:
        return json.loads(cached)

    # Query database
    analytics = query_database(student_id)

    # Cache for 1 hour
    cache.setex(
        f"analytics:{student_id}",
        3600,
        json.dumps(analytics)
    )

    return analytics
```

---

**Author**: Claude Code
**Date**: 2025-11-17
**Version**: 1.0
