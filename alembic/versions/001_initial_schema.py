"""Initial schema

Revision ID: 001
Revises:
Create Date: 2025-11-18

Creates all core tables for LMS Orchestration:
- Core tables (institutions, users, courses, enrollments, lessons, assessments, submissions)
- Plugin-specific tables (peer_review_assignments, peer_reviews, calibration_scores)
- Analytics tables (engagement_events, learning_analytics)
- Plugin system tables (plugins, plugin_configurations)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY


# revision identifiers, used by Alembic
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all initial tables"""

    # ============================================================================
    # Core Tables
    # ============================================================================

    # institutions
    op.create_table(
        'institutions',
        sa.Column('institution_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False, unique=True),
        sa.Column('settings', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
    )
    op.create_index('idx_institutions_domain', 'institutions', ['domain'])

    # users
    op.create_table(
        'users',
        sa.Column('user_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('institution_id', UUID, sa.ForeignKey('institutions.institution_id'), nullable=False),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('avatar_url', sa.Text),
        sa.Column('metadata', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('last_login_at', sa.TIMESTAMP),
        sa.CheckConstraint("role IN ('student', 'instructor', 'admin', 'ta')", name='users_role_check'),
    )
    op.create_index('idx_users_institution', 'users', ['institution_id'])
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_role', 'users', ['role'])

    # courses
    op.create_table(
        'courses',
        sa.Column('course_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('institution_id', UUID, sa.ForeignKey('institutions.institution_id'), nullable=False),
        sa.Column('code', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('semester', sa.String(50)),
        sa.Column('year', sa.Integer),
        sa.Column('status', sa.String(50), server_default='active'),
        sa.Column('settings', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('created_by', UUID, sa.ForeignKey('users.user_id'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('active', 'archived', 'draft')", name='courses_status_check'),
        sa.UniqueConstraint('institution_id', 'code', 'semester', 'year', name='unique_course'),
    )
    op.create_index('idx_courses_institution', 'courses', ['institution_id'])
    op.create_index('idx_courses_status', 'courses', ['status'])
    op.create_index('idx_courses_created_by', 'courses', ['created_by'])

    # course_enrollments
    op.create_table(
        'course_enrollments',
        sa.Column('enrollment_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.course_id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('enrolled_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('dropped_at', sa.TIMESTAMP),
        sa.Column('grade', sa.String(10)),
        sa.CheckConstraint("role IN ('student', 'instructor', 'ta')", name='enrollments_role_check'),
        sa.UniqueConstraint('course_id', 'user_id', name='unique_enrollment'),
    )
    op.create_index('idx_enrollments_course', 'course_enrollments', ['course_id'])
    op.create_index('idx_enrollments_user', 'course_enrollments', ['user_id'])
    op.create_index('idx_enrollments_role', 'course_enrollments', ['role'])

    # lessons
    op.create_table(
        'lessons',
        sa.Column('lesson_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.course_id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('content', sa.Text),
        sa.Column('theme', sa.String(50), nullable=False),
        sa.Column('concept', sa.String(255)),
        sa.Column('prerequisites', ARRAY(sa.Text)),
        sa.Column('difficulty', sa.String(50), server_default='intermediate'),
        sa.Column('duration_minutes', sa.Integer),
        sa.Column('plugins', ARRAY(sa.Text)),
        sa.Column('settings', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('order_index', sa.Integer),
        sa.Column('status', sa.String(50), server_default='draft'),
        sa.Column('created_by', UUID, sa.ForeignKey('users.user_id'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('published_at', sa.TIMESTAMP),
        sa.CheckConstraint("theme IN ('lecture', 'flipped', 'project', 'socratic')", name='lessons_theme_check'),
        sa.CheckConstraint("status IN ('draft', 'published', 'archived')", name='lessons_status_check'),
    )
    op.create_index('idx_lessons_course', 'lessons', ['course_id'])
    op.create_index('idx_lessons_concept', 'lessons', ['concept'])
    op.create_index('idx_lessons_status', 'lessons', ['status'])
    op.create_index('idx_lessons_order', 'lessons', ['course_id', 'order_index'])

    # assessments
    op.create_table(
        'assessments',
        sa.Column('assessment_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.course_id', ondelete='CASCADE'), nullable=False),
        sa.Column('lesson_id', UUID, sa.ForeignKey('lessons.lesson_id', ondelete='SET NULL')),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('instructions', sa.Text),
        sa.Column('assessment_type', sa.String(50), nullable=False),
        sa.Column('plugin_id', sa.String(100)),
        sa.Column('concept', sa.String(255)),
        sa.Column('max_score', sa.Float, nullable=False),
        sa.Column('passing_score', sa.Float),
        sa.Column('time_limit_minutes', sa.Integer),
        sa.Column('attempts_allowed', sa.Integer, server_default='1'),
        sa.Column('settings', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('due_date', sa.TIMESTAMP),
        sa.Column('available_from', sa.TIMESTAMP),
        sa.Column('available_until', sa.TIMESTAMP),
        sa.Column('status', sa.String(50), server_default='draft'),
        sa.Column('created_by', UUID, sa.ForeignKey('users.user_id'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('draft', 'published', 'closed')", name='assessments_status_check'),
    )
    op.create_index('idx_assessments_course', 'assessments', ['course_id'])
    op.create_index('idx_assessments_lesson', 'assessments', ['lesson_id'])
    op.create_index('idx_assessments_plugin', 'assessments', ['plugin_id'])
    op.create_index('idx_assessments_due_date', 'assessments', ['due_date'])

    # submissions
    op.create_table(
        'submissions',
        sa.Column('submission_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('assessment_id', UUID, sa.ForeignKey('assessments.assessment_id', ondelete='CASCADE'), nullable=False),
        sa.Column('student_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('attempt_number', sa.Integer, nullable=False, server_default='1'),
        sa.Column('content', sa.Text),
        sa.Column('attachments', ARRAY(sa.Text)),
        sa.Column('word_count', sa.Integer),
        sa.Column('submitted_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('late', sa.Boolean, server_default='false'),
        sa.Column('score', sa.Float),
        sa.Column('graded_at', sa.TIMESTAMP),
        sa.Column('graded_by', UUID, sa.ForeignKey('users.user_id')),
        sa.Column('feedback', sa.Text),
        sa.Column('metadata', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint('assessment_id', 'student_id', 'attempt_number', name='unique_submission'),
    )
    op.create_index('idx_submissions_assessment', 'submissions', ['assessment_id'])
    op.create_index('idx_submissions_student', 'submissions', ['student_id'])
    op.create_index('idx_submissions_graded', 'submissions', ['graded_at'])

    # ============================================================================
    # Plugin-Specific Tables
    # ============================================================================

    # peer_review_assignments
    op.create_table(
        'peer_review_assignments',
        sa.Column('assignment_id', UUID, sa.ForeignKey('assessments.assessment_id', ondelete='CASCADE'), primary_key=True),
        sa.Column('review_type', sa.String(50), nullable=False),
        sa.Column('rubric', JSONB, nullable=False),
        sa.Column('reviews_per_submission', sa.Integer, server_default='3'),
        sa.Column('enable_calibration', sa.Boolean, server_default='true'),
        sa.Column('calibration_threshold', sa.Float, server_default='0.8'),
        sa.Column('review_deadline', sa.TIMESTAMP),
        sa.Column('dispute_deadline', sa.TIMESTAMP),
        sa.Column('quality_weight', sa.Float, server_default='0.7'),
        sa.Column('completion_weight', sa.Float, server_default='0.3'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.CheckConstraint("review_type IN ('single_blind', 'double_blind', 'open')", name='review_type_check'),
    )

    # peer_reviews
    op.create_table(
        'peer_reviews',
        sa.Column('review_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('submission_id', UUID, sa.ForeignKey('submissions.submission_id', ondelete='CASCADE'), nullable=False),
        sa.Column('reviewer_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('scores', JSONB, nullable=False),
        sa.Column('total_score', sa.Float, nullable=False),
        sa.Column('overall_feedback', sa.Text),
        sa.Column('quality_score', sa.Float),
        sa.Column('helpfulness_score', sa.Float),
        sa.Column('timeliness_score', sa.Float),
        sa.Column('status', sa.String(50), server_default='assigned'),
        sa.Column('submitted_at', sa.TIMESTAMP),
        sa.Column('time_spent_minutes', sa.Integer),
        sa.Column('metadata', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint(
            "status IN ('assigned', 'in_progress', 'submitted', 'disputed', 'resolved')",
            name='peer_reviews_status_check'
        ),
        sa.UniqueConstraint('submission_id', 'reviewer_id', name='unique_peer_review'),
    )
    op.create_index('idx_peer_reviews_submission', 'peer_reviews', ['submission_id'])
    op.create_index('idx_peer_reviews_reviewer', 'peer_reviews', ['reviewer_id'])
    op.create_index('idx_peer_reviews_status', 'peer_reviews', ['status'])

    # calibration_scores
    op.create_table(
        'calibration_scores',
        sa.Column('calibration_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('student_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('assignment_id', UUID, sa.ForeignKey('peer_review_assignments.assignment_id', ondelete='CASCADE'), nullable=False),
        sa.Column('calibration_score', sa.Float, nullable=False),
        sa.Column('attempts_completed', sa.Integer, server_default='0'),
        sa.Column('passed', sa.Boolean, server_default='false'),
        sa.Column('last_attempt_at', sa.TIMESTAMP),
        sa.UniqueConstraint('student_id', 'assignment_id', name='unique_calibration'),
    )
    op.create_index('idx_calibration_student', 'calibration_scores', ['student_id'])
    op.create_index('idx_calibration_assignment', 'calibration_scores', ['assignment_id'])

    # ============================================================================
    # Analytics Tables
    # ============================================================================

    # engagement_events (partitioned table - main table only, partitions created separately)
    op.create_table(
        'engagement_events',
        sa.Column('event_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('student_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.course_id', ondelete='CASCADE')),
        sa.Column('lesson_id', UUID, sa.ForeignKey('lessons.lesson_id', ondelete='CASCADE')),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_data', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column('timestamp', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('session_id', UUID),
        sa.Column('metadata', JSONB, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index('idx_engagement_student', 'engagement_events', ['student_id'])
    op.create_index('idx_engagement_course', 'engagement_events', ['course_id'])
    op.create_index('idx_engagement_type', 'engagement_events', ['event_type'])
    op.create_index('idx_engagement_timestamp', 'engagement_events', [sa.text('timestamp DESC')])

    # learning_analytics
    op.create_table(
        'learning_analytics',
        sa.Column('analytics_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('student_id', UUID, sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
        sa.Column('course_id', UUID, sa.ForeignKey('courses.course_id', ondelete='CASCADE'), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('calculated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('metadata', JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint('student_id', 'course_id', 'metric_name', 'calculated_at', name='unique_analytics'),
    )
    op.create_index('idx_analytics_student_course', 'learning_analytics', ['student_id', 'course_id'])
    op.create_index('idx_analytics_metric', 'learning_analytics', ['metric_name'])
    op.create_index('idx_analytics_calculated', 'learning_analytics', [sa.text('calculated_at DESC')])

    # ============================================================================
    # Plugin System Tables
    # ============================================================================

    # plugins
    op.create_table(
        'plugins',
        sa.Column('plugin_id', sa.String(100), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('author', sa.String(255)),
        sa.Column('permissions', ARRAY(sa.Text)),
        sa.Column('hooks', ARRAY(sa.Text)),
        sa.Column('api_routes', JSONB),
        sa.Column('ui_components', JSONB),
        sa.Column('config_schema', JSONB),
        sa.Column('status', sa.String(50), server_default='inactive'),
        sa.Column('installed_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.CheckConstraint("status IN ('active', 'inactive', 'deprecated')", name='plugins_status_check'),
        sa.UniqueConstraint('plugin_id', 'version', name='unique_plugin_version'),
    )
    op.create_index('idx_plugins_category', 'plugins', ['category'])
    op.create_index('idx_plugins_status', 'plugins', ['status'])

    # plugin_configurations
    op.create_table(
        'plugin_configurations',
        sa.Column('config_id', UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('institution_id', UUID, sa.ForeignKey('institutions.institution_id', ondelete='CASCADE'), nullable=False),
        sa.Column('plugin_id', sa.String(100), sa.ForeignKey('plugins.plugin_id', ondelete='CASCADE'), nullable=False),
        sa.Column('config', JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column('enabled', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.UniqueConstraint('institution_id', 'plugin_id', name='unique_plugin_config'),
    )
    op.create_index('idx_plugin_config_institution', 'plugin_configurations', ['institution_id'])
    op.create_index('idx_plugin_config_plugin', 'plugin_configurations', ['plugin_id'])

    # ============================================================================
    # Additional Indexes
    # ============================================================================

    # Composite indexes for common queries
    op.create_index('idx_submissions_student_assessment', 'submissions', ['student_id', 'assessment_id'])
    op.create_index('idx_reviews_submission_status', 'peer_reviews', ['submission_id', 'status'])

    # Partial indexes for active records
    op.execute("""
        CREATE INDEX idx_active_courses
        ON courses(institution_id)
        WHERE status = 'active'
    """)

    op.execute("""
        CREATE INDEX idx_published_lessons
        ON lessons(course_id, order_index)
        WHERE status = 'published'
    """)

    # GIN indexes for JSONB columns
    op.create_index('idx_plugin_config_gin', 'plugin_configurations', ['config'], postgresql_using='gin')
    op.create_index('idx_lesson_settings_gin', 'lessons', ['settings'], postgresql_using='gin')


def downgrade() -> None:
    """Drop all tables in reverse order"""

    # Plugin system
    op.drop_table('plugin_configurations')
    op.drop_table('plugins')

    # Analytics
    op.drop_table('learning_analytics')
    op.drop_table('engagement_events')

    # Plugin-specific (peer review)
    op.drop_table('calibration_scores')
    op.drop_table('peer_reviews')
    op.drop_table('peer_review_assignments')

    # Core tables (reverse dependency order)
    op.drop_table('submissions')
    op.drop_table('assessments')
    op.drop_table('lessons')
    op.drop_table('course_enrollments')
    op.drop_table('courses')
    op.drop_table('users')
    op.drop_table('institutions')
