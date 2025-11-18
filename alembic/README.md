# Database Migrations

This directory contains Alembic database migrations for the LMS Orchestration system.

## Quick Start

### 1. Start Database Services

```bash
make up
```

Wait for PostgreSQL to be ready:
```bash
make health
```

### 2. Run Migrations

```bash
make migrate
```

This will create all tables defined in `001_initial_schema.py`.

### 3. Seed Initial Data

```bash
make seed
```

This will populate the database with:
- Sample institution (Demo University)
- Admin user (admin@demo.edu / admin123)
- Instructor (instructor@demo.edu / instructor123)
- 5 sample students (student1@demo.edu / student1123, etc.)
- 4 plugins (simple-quiz, peer-review-system, video-player, analytics-dashboard)
- Sample course (CS101) with 4 lessons
- Sample assessments (quiz and peer review)

## Migration Commands

### Run all pending migrations
```bash
make migrate
# or directly:
alembic upgrade head
```

### Rollback last migration
```bash
make migrate-down
# or directly:
alembic downgrade -1
```

### Create new migration
```bash
make migrate-create
# Enter migration name when prompted

# or directly:
alembic revision --autogenerate -m "description"
```

### View migration history
```bash
alembic history
```

### View current migration version
```bash
alembic current
```

## Database Schema

The initial migration (`001_initial_schema.py`) creates:

### Core Tables (8)
- `institutions` - Educational institutions
- `users` - All users (students, instructors, admins, TAs)
- `courses` - Courses offered
- `course_enrollments` - Student/instructor enrollments
- `lessons` - Individual lessons with pedagogical themes
- `assessments` - Assignments, quizzes, exams
- `submissions` - Student work submitted

### Plugin Tables (3)
- `peer_review_assignments` - Peer review configuration
- `peer_reviews` - Individual peer reviews
- `calibration_scores` - Reviewer calibration tracking

### Analytics Tables (2)
- `engagement_events` - Student interaction events (partitioned)
- `learning_analytics` - Calculated learning metrics

### Plugin System Tables (2)
- `plugins` - Installed plugins
- `plugin_configurations` - Institution-specific plugin settings

**Total**: 15 tables with comprehensive indexing

## Environment Variables

Configure database connection in `.env`:

```env
DATABASE_URL=postgresql://lms_user:lms_dev_password@localhost:5432/lms_dev
```

Or override via environment:
```bash
export DATABASE_URL="postgresql://user:pass@host:port/db"
alembic upgrade head
```

## Backup & Restore

### Backup all databases
```bash
make db-backup
```

This creates backups in `./backups/`:
- `postgres_YYYYMMDD_HHMMSS.sql` - PostgreSQL dump
- `neo4j_YYYYMMDD_HHMMSS.dump` - Neo4j dump

### Restore PostgreSQL
```bash
docker-compose exec -T postgres psql -U lms_user -d lms_dev < backups/postgres_YYYYMMDD_HHMMSS.sql
```

## Development Workflow

### Making Schema Changes

1. **Edit models** (when SQLAlchemy models exist):
   ```python
   # backend/models/user.py
   class User(Base):
       # ... add new column
   ```

2. **Generate migration**:
   ```bash
   make migrate-create
   # Name: "add_user_preferences"
   ```

3. **Review generated migration**:
   ```bash
   cat alembic/versions/YYYYMMDD_HHMM_*_add_user_preferences.py
   ```

4. **Apply migration**:
   ```bash
   make migrate
   ```

5. **Test rollback** (optional):
   ```bash
   make migrate-down
   make migrate  # Re-apply
   ```

### Manual Migrations

For complex changes, create manual migrations:

```bash
alembic revision -m "complex_data_migration"
```

Then edit the generated file:
```python
def upgrade():
    # Custom SQL or Python logic
    op.execute("""
        UPDATE lessons
        SET theme = 'flipped'
        WHERE difficulty = 'beginner'
    """)

def downgrade():
    # Reverse the change
    op.execute("""
        UPDATE lessons
        SET theme = 'lecture'
        WHERE theme = 'flipped' AND difficulty = 'beginner'
    """)
```

## Best Practices

### DO
✅ Test migrations on dev before production
✅ Backup before migrating production
✅ Write reversible migrations (downgrade should work)
✅ Use transactions for data migrations
✅ Add indexes for foreign keys and frequently queried columns

### DON'T
❌ Delete migrations after they've been applied to production
❌ Edit migrations after they've been committed
❌ Skip writing downgrade logic
❌ Commit migrations without testing
❌ Run migrations directly in production without backup

## Troubleshooting

### Migration fails with "relation already exists"
The table was created manually or by a previous migration. Either:
1. Drop the table and re-run
2. Mark migration as complete: `alembic stamp head`

### Can't connect to database
1. Check database is running: `make health`
2. Verify credentials in `.env`
3. Check Docker network: `docker network ls`

### Rollback fails
If downgrade logic is missing or broken:
1. Manually revert changes in database
2. Update migration's `downgrade()` function
3. Commit fix for future use

### Clean slate (DANGER - deletes all data)
```bash
make reset
make up
make migrate
make seed
```

## Files

- `alembic.ini` - Alembic configuration
- `env.py` - Migration environment setup
- `versions/001_initial_schema.py` - Initial schema migration
- `README.md` - This file

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
