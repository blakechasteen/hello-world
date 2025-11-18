"""Initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-11-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

# revision identifiers, used by Alembic.
revision: str = '0001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('user_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('email_verified', sa.Boolean, default=False),
        sa.Column('password_hash', sa.String(255), nullable=False),

        # Profile
        sa.Column('display_name', sa.String(100)),
        sa.Column('bio', sa.Text),
        sa.Column('avatar_url', sa.Text),
        sa.Column('cover_image_url', sa.Text),
        sa.Column('location', sa.String(100)),
        sa.Column('website', sa.String(255)),

        # Social links
        sa.Column('social_links', JSONB, server_default='{}'),

        # Stats
        sa.Column('reputation_score', sa.Integer, server_default='0'),
        sa.Column('trust_level', sa.Integer, server_default='0'),
        sa.Column('post_count', sa.Integer, server_default='0'),
        sa.Column('comment_count', sa.Integer, server_default='0'),
        sa.Column('follower_count', sa.Integer, server_default='0'),
        sa.Column('following_count', sa.Integer, server_default='0'),

        # Status
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('role', sa.String(20), server_default='member'),

        # Settings
        sa.Column('preferences', JSONB, server_default='{}'),
        sa.Column('notification_settings', JSONB, server_default='{}'),
        sa.Column('privacy_settings', JSONB, server_default='{}'),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_seen_at', sa.TIMESTAMP),
        sa.Column('last_login_at', sa.TIMESTAMP),

        # Metadata
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.CheckConstraint("trust_level >= 0 AND trust_level <= 5"),
        sa.CheckConstraint("status IN ('active', 'suspended', 'banned', 'deleted')"),
        sa.CheckConstraint("role IN ('guest', 'member', 'moderator', 'admin')"),
    )

    # Create indexes for users
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_reputation', 'users', ['reputation_score'], postgresql_using='btree')

    # Create communities table
    op.create_table(
        'communities',
        sa.Column('community_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('icon_url', sa.Text),
        sa.Column('banner_url', sa.Text),
        sa.Column('community_type', sa.String(20), server_default='public'),
        sa.Column('member_count', sa.Integer, server_default='0'),
        sa.Column('post_count', sa.Integer, server_default='0'),
        sa.Column('rules', JSONB, server_default='[]'),
        sa.Column('settings', JSONB, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.CheckConstraint("community_type IN ('public', 'private', 'restricted')"),
    )

    op.create_index('idx_communities_name', 'communities', ['name'])

    # Create community_members table
    op.create_table(
        'community_members',
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),
        sa.Column('community_id', UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(20), server_default='member'),
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('notification_level', sa.String(20), server_default='all'),
        sa.Column('post_count', sa.Integer, server_default='0'),
        sa.Column('comment_count', sa.Integer, server_default='0'),
        sa.Column('joined_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('last_visited_at', sa.TIMESTAMP),

        sa.PrimaryKeyConstraint('user_id', 'community_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['community_id'], ['communities.community_id'], ondelete='CASCADE'),

        sa.CheckConstraint("role IN ('member', 'moderator', 'admin', 'owner')"),
        sa.CheckConstraint("status IN ('active', 'muted', 'banned')"),
        sa.CheckConstraint("notification_level IN ('all', 'mentions', 'none')"),
    )

    # Create posts table
    op.create_table(
        'posts',
        sa.Column('post_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('author_id', UUID(as_uuid=True)),
        sa.Column('community_id', UUID(as_uuid=True)),
        sa.Column('title', sa.String(300), nullable=False),
        sa.Column('content', sa.Text),
        sa.Column('content_html', sa.Text),
        sa.Column('post_type', sa.String(20), server_default='text'),
        sa.Column('url', sa.Text),
        sa.Column('url_title', sa.String(300)),
        sa.Column('url_description', sa.Text),
        sa.Column('url_thumbnail', sa.Text),
        sa.Column('media_urls', ARRAY(sa.Text), server_default='{}'),
        sa.Column('tags', ARRAY(sa.String), server_default='{}'),
        sa.Column('upvote_count', sa.Integer, server_default='0'),
        sa.Column('downvote_count', sa.Integer, server_default='0'),
        sa.Column('vote_score', sa.Integer, server_default='0'),
        sa.Column('hot_score', sa.Float, server_default='0'),
        sa.Column('controversy_score', sa.Float, server_default='0'),
        sa.Column('wilson_score', sa.Float, server_default='0'),
        sa.Column('comment_count', sa.Integer, server_default='0'),
        sa.Column('view_count', sa.Integer, server_default='0'),
        sa.Column('share_count', sa.Integer, server_default='0'),
        sa.Column('status', sa.String(20), server_default='published'),
        sa.Column('pinned', sa.Boolean, server_default=False),
        sa.Column('locked', sa.Boolean, server_default=False),
        sa.Column('nsfw', sa.Boolean, server_default=False),
        sa.Column('spoiler', sa.Boolean, server_default=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('published_at', sa.TIMESTAMP),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.ForeignKeyConstraint(['author_id'], ['users.user_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['community_id'], ['communities.community_id'], ondelete='CASCADE'),

        sa.CheckConstraint("post_type IN ('text', 'link', 'image', 'video', 'poll', 'question')"),
        sa.CheckConstraint("status IN ('draft', 'published', 'removed', 'deleted')"),
    )

    op.create_index('idx_posts_hot_score', 'posts', ['hot_score'], postgresql_using='btree')
    op.create_index('idx_posts_vote_score', 'posts', ['vote_score'], postgresql_using='btree')
    op.create_index('idx_posts_created_at', 'posts', ['created_at'], postgresql_using='btree')
    op.create_index('idx_posts_tags', 'posts', ['tags'], postgresql_using='gin')

    # Create comments table
    op.create_table(
        'comments',
        sa.Column('comment_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('post_id', UUID(as_uuid=True), nullable=False),
        sa.Column('author_id', UUID(as_uuid=True)),
        sa.Column('parent_id', UUID(as_uuid=True)),
        sa.Column('path', sa.Text, nullable=False),
        sa.Column('depth', sa.Integer, server_default='0'),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('content_html', sa.Text),
        sa.Column('upvote_count', sa.Integer, server_default='0'),
        sa.Column('downvote_count', sa.Integer, server_default='0'),
        sa.Column('vote_score', sa.Integer, server_default='0'),
        sa.Column('status', sa.String(20), server_default='published'),
        sa.Column('edited', sa.Boolean, server_default=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.ForeignKeyConstraint(['post_id'], ['posts.post_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['author_id'], ['users.user_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['parent_id'], ['comments.comment_id'], ondelete='CASCADE'),

        sa.CheckConstraint("status IN ('published', 'removed', 'deleted')"),
    )

    op.create_index('idx_comments_post_path', 'comments', ['post_id', 'path'])

    # Create votes table
    op.create_table(
        'votes',
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),
        sa.Column('target_type', sa.String(20), nullable=False),
        sa.Column('target_id', UUID(as_uuid=True), nullable=False),
        sa.Column('vote_value', sa.SmallInteger, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),

        sa.PrimaryKeyConstraint('user_id', 'target_type', 'target_id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),

        sa.CheckConstraint("target_type IN ('post', 'comment')"),
        sa.CheckConstraint("vote_value IN (-1, 1)"),
    )

    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('conversation_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_type', sa.String(20), server_default='direct'),
        sa.Column('title', sa.String(200)),
        sa.Column('creator_id', UUID(as_uuid=True)),
        sa.Column('participant_count', sa.Integer, server_default='0'),
        sa.Column('message_count', sa.Integer, server_default='0'),
        sa.Column('is_archived', sa.Boolean, server_default=False),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_message_at', sa.TIMESTAMP),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.ForeignKeyConstraint(['creator_id'], ['users.user_id'], ondelete='SET NULL'),

        sa.CheckConstraint("conversation_type IN ('direct', 'group')"),
    )

    # Create conversation_participants table
    op.create_table(
        'conversation_participants',
        sa.Column('conversation_id', UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),
        sa.Column('is_muted', sa.Boolean, server_default=False),
        sa.Column('notification_level', sa.String(20), server_default='all'),
        sa.Column('last_read_at', sa.TIMESTAMP),
        sa.Column('unread_count', sa.Integer, server_default='0'),
        sa.Column('joined_at', sa.TIMESTAMP, server_default=sa.func.now()),

        sa.PrimaryKeyConstraint('conversation_id', 'user_id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.conversation_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),

        sa.CheckConstraint("notification_level IN ('all', 'mentions', 'none')"),
    )

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('message_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', UUID(as_uuid=True), nullable=False),
        sa.Column('sender_id', UUID(as_uuid=True)),
        sa.Column('reply_to_id', UUID(as_uuid=True)),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('content_type', sa.String(20), server_default='text'),
        sa.Column('attachments', ARRAY(sa.Text), server_default='{}'),
        sa.Column('is_edited', sa.Boolean, server_default=False),
        sa.Column('is_deleted', sa.Boolean, server_default=False),
        sa.Column('read_by', ARRAY(UUID), server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.conversation_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['sender_id'], ['users.user_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reply_to_id'], ['messages.message_id'], ondelete='SET NULL'),

        sa.CheckConstraint("content_type IN ('text', 'image', 'file', 'system')"),
    )

    op.create_index('idx_messages_conversation', 'messages', ['conversation_id'])

    # Create notifications table
    op.create_table(
        'notifications',
        sa.Column('notification_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),
        sa.Column('notification_type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('message', sa.Text),
        sa.Column('link_type', sa.String(20)),
        sa.Column('link_id', UUID(as_uuid=True)),
        sa.Column('actor_id', UUID(as_uuid=True)),
        sa.Column('is_read', sa.Boolean, server_default=False),
        sa.Column('is_delivered', sa.Boolean, server_default=False),
        sa.Column('channels', JSONB, server_default='{"in_app": true}'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('read_at', sa.TIMESTAMP),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['actor_id'], ['users.user_id'], ondelete='SET NULL'),

        sa.CheckConstraint("""notification_type IN (
            'post_reply', 'comment_reply', 'mention', 'upvote',
            'award', 'follow', 'message', 'community_invite',
            'moderation', 'system', 'achievement'
        )"""),
        sa.CheckConstraint("link_type IN ('post', 'comment', 'user', 'community', 'message')"),
    )

    op.create_index('idx_notifications_user_read', 'notifications', ['user_id', 'is_read'])

    # Create plugins table
    op.create_table(
        'plugins',
        sa.Column('plugin_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('author', sa.String(100)),
        sa.Column('plugin_type', sa.String(50), nullable=False),
        sa.Column('main_file', sa.String(255), nullable=False),
        sa.Column('config_schema', JSONB, server_default='{}'),
        sa.Column('required_permissions', ARRAY(sa.String), server_default='{}'),
        sa.Column('hooks', ARRAY(sa.String), server_default='{}'),
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('is_official', sa.Boolean, server_default=False),
        sa.Column('downloads', sa.Integer, server_default='0'),
        sa.Column('rating', sa.Integer, server_default='0'),
        sa.Column('review_count', sa.Integer, server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('published_at', sa.TIMESTAMP),
        sa.Column('metadata', JSONB, server_default='{}'),

        sa.CheckConstraint("""plugin_type IN (
            'content', 'integration', 'moderation', 'engagement',
            'communication', 'analytics', 'customization', 'admin'
        )"""),
        sa.CheckConstraint("status IN ('active', 'disabled', 'deprecated')"),
    )

    op.create_index('idx_plugins_name', 'plugins', ['name'])

    # Create plugin_instances table
    op.create_table(
        'plugin_instances',
        sa.Column('instance_id', UUID(as_uuid=True), primary_key=True),
        sa.Column('plugin_id', UUID(as_uuid=True), nullable=False),
        sa.Column('community_id', UUID(as_uuid=True)),
        sa.Column('config', JSONB, server_default='{}'),
        sa.Column('enabled', sa.Boolean, server_default=True),
        sa.Column('installed_at', sa.TIMESTAMP, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, server_default=sa.func.now(), onupdate=sa.func.now()),

        sa.ForeignKeyConstraint(['plugin_id'], ['plugins.plugin_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['community_id'], ['communities.community_id'], ondelete='CASCADE'),
    )


def downgrade() -> None:
    op.drop_table('plugin_instances')
    op.drop_table('plugins')
    op.drop_table('notifications')
    op.drop_table('messages')
    op.drop_table('conversation_participants')
    op.drop_table('conversations')
    op.drop_table('votes')
    op.drop_table('comments')
    op.drop_table('posts')
    op.drop_table('community_members')
    op.drop_table('communities')
    op.drop_table('users')
