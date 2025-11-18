# Community Platform Database Schema

**Date**: 2025-11-18
**Version**: 1.0

Complete database schema for hybrid community platform supporting forum, social network, chat, and Q&A features.

---

## Table of Contents

1. [PostgreSQL Schema](#postgresql-schema)
2. [Neo4j Graph Schema](#neo4j-graph-schema)
3. [Redis Data Structures](#redis-data-structures)
4. [Elasticsearch Indexes](#elasticsearch-indexes)

---

## PostgreSQL Schema

### Core Tables

#### users

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,

    -- Profile
    display_name VARCHAR(100),
    bio TEXT,
    avatar_url TEXT,
    cover_image_url TEXT,
    location VARCHAR(100),
    website VARCHAR(255),

    -- Social links
    social_links JSONB DEFAULT '{}',

    -- Stats
    reputation_score INTEGER DEFAULT 0,
    trust_level INTEGER DEFAULT 0 CHECK (trust_level BETWEEN 0 AND 5),
    post_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    follower_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'banned', 'deleted')),
    role VARCHAR(20) DEFAULT 'member' CHECK (role IN ('guest', 'member', 'moderator', 'admin')),

    -- Settings
    preferences JSONB DEFAULT '{}',
    notification_settings JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_seen_at TIMESTAMP,
    last_login_at TIMESTAMP,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_reputation ON users(reputation_score DESC);
CREATE INDEX idx_users_created ON users(created_at DESC);
CREATE INDEX idx_users_status ON users(status) WHERE status = 'active';
```

#### communities

```sql
CREATE TABLE communities (
    community_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    rules TEXT,

    -- Appearance
    icon_url TEXT,
    banner_url TEXT,
    theme JSONB DEFAULT '{}',

    -- Type
    type VARCHAR(20) DEFAULT 'public' CHECK (type IN ('public', 'private', 'restricted')),
    category VARCHAR(50),

    -- Stats
    member_count INTEGER DEFAULT 0,
    post_count INTEGER DEFAULT 0,
    active_user_count INTEGER DEFAULT 0,

    -- Settings
    settings JSONB DEFAULT '{}',
    allowed_post_types TEXT[] DEFAULT ARRAY['text', 'link', 'image', 'video'],

    -- Creator
    created_by UUID NOT NULL REFERENCES users(user_id),

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'private')),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_communities_slug ON communities(slug);
CREATE INDEX idx_communities_category ON communities(category);
CREATE INDEX idx_communities_member_count ON communities(member_count DESC);
CREATE INDEX idx_communities_created ON communities(created_at DESC);
```

#### community_members

```sql
CREATE TABLE community_members (
    membership_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID NOT NULL REFERENCES communities(community_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Role in community
    role VARCHAR(20) DEFAULT 'member' CHECK (role IN ('member', 'moderator', 'admin')),

    -- Custom flair
    flair_text VARCHAR(100),
    flair_color VARCHAR(7),

    -- Permissions
    permissions JSONB DEFAULT '{}',

    -- Activity
    post_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    last_active_at TIMESTAMP,

    -- Timestamps
    joined_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(community_id, user_id)
);

CREATE INDEX idx_community_members_user ON community_members(user_id);
CREATE INDEX idx_community_members_community ON community_members(community_id);
CREATE INDEX idx_community_members_role ON community_members(role);
```

#### posts

```sql
CREATE TABLE posts (
    post_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Author & Community
    author_id UUID NOT NULL REFERENCES users(user_id),
    community_id UUID REFERENCES communities(community_id) ON DELETE CASCADE,

    -- Content
    title VARCHAR(300) NOT NULL,
    content TEXT,
    content_html TEXT,
    post_type VARCHAR(20) DEFAULT 'text' CHECK (post_type IN ('text', 'link', 'image', 'video', 'poll', 'question')),

    -- Link posts
    url TEXT,

    -- Media
    media_urls TEXT[],
    thumbnail_url TEXT,

    -- Tags
    tags TEXT[],

    -- Voting
    upvote_count INTEGER DEFAULT 0,
    downvote_count INTEGER DEFAULT 0,
    vote_score INTEGER DEFAULT 0,
    hot_score FLOAT DEFAULT 0,
    controversy_score FLOAT DEFAULT 0,

    -- Engagement
    comment_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'published' CHECK (status IN ('draft', 'published', 'removed', 'deleted')),
    pinned BOOLEAN DEFAULT FALSE,
    locked BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE,
    nsfw BOOLEAN DEFAULT FALSE,
    spoiler BOOLEAN DEFAULT FALSE,

    -- Moderation
    removed_by UUID REFERENCES users(user_id),
    removal_reason TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP,
    last_activity_at TIMESTAMP,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_posts_author ON posts(author_id);
CREATE INDEX idx_posts_community ON posts(community_id);
CREATE INDEX idx_posts_created ON posts(created_at DESC);
CREATE INDEX idx_posts_hot_score ON posts(hot_score DESC) WHERE status = 'published';
CREATE INDEX idx_posts_vote_score ON posts(vote_score DESC) WHERE status = 'published';
CREATE INDEX idx_posts_tags ON posts USING GIN(tags);
CREATE INDEX idx_posts_status ON posts(status);
```

#### comments

```sql
CREATE TABLE comments (
    comment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Post & Author
    post_id UUID NOT NULL REFERENCES posts(post_id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES users(user_id),

    -- Threading
    parent_id UUID REFERENCES comments(comment_id) ON DELETE CASCADE,
    path TEXT NOT NULL,  -- Materialized path for tree structure
    depth INTEGER DEFAULT 0,

    -- Content
    content TEXT NOT NULL,
    content_html TEXT,

    -- Voting
    upvote_count INTEGER DEFAULT 0,
    downvote_count INTEGER DEFAULT 0,
    vote_score INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'published' CHECK (status IN ('published', 'removed', 'deleted')),
    edited BOOLEAN DEFAULT FALSE,

    -- Moderation
    removed_by UUID REFERENCES users(user_id),
    removal_reason TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_comments_post ON comments(post_id, path);
CREATE INDEX idx_comments_author ON comments(author_id);
CREATE INDEX idx_comments_parent ON comments(parent_id);
CREATE INDEX idx_comments_created ON comments(created_at DESC);
CREATE INDEX idx_comments_vote_score ON comments(vote_score DESC);
```

#### votes

```sql
CREATE TABLE votes (
    vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Voteable (polymorphic)
    voteable_type VARCHAR(20) NOT NULL CHECK (voteable_type IN ('post', 'comment')),
    voteable_id UUID NOT NULL,

    -- Vote direction
    direction INTEGER NOT NULL CHECK (direction IN (-1, 1)),

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(user_id, voteable_type, voteable_id)
);

CREATE INDEX idx_votes_user ON votes(user_id);
CREATE INDEX idx_votes_voteable ON votes(voteable_type, voteable_id);
```

#### messages

```sql
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Conversation
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,

    -- Sender
    sender_id UUID NOT NULL REFERENCES users(user_id),

    -- Content
    content TEXT NOT NULL,
    content_html TEXT,

    -- Attachments
    attachments JSONB DEFAULT '[]',

    -- Status
    edited BOOLEAN DEFAULT FALSE,
    deleted BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at DESC);
CREATE INDEX idx_messages_sender ON messages(sender_id);
```

#### conversations

```sql
CREATE TABLE conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Type
    type VARCHAR(20) DEFAULT 'direct' CHECK (type IN ('direct', 'group', 'channel')),

    -- Group/channel details
    name VARCHAR(100),
    description TEXT,
    icon_url TEXT,

    -- Creator
    created_by UUID REFERENCES users(user_id),

    -- Last message
    last_message_id UUID REFERENCES messages(message_id),
    last_message_at TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_conversations_type ON conversations(type);
CREATE INDEX idx_conversations_last_message ON conversations(last_message_at DESC);
```

#### conversation_participants

```sql
CREATE TABLE conversation_participants (
    participant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Status
    joined_at TIMESTAMP DEFAULT NOW(),
    left_at TIMESTAMP,
    last_read_message_id UUID REFERENCES messages(message_id),
    last_read_at TIMESTAMP,

    -- Notifications
    muted BOOLEAN DEFAULT FALSE,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    UNIQUE(conversation_id, user_id)
);

CREATE INDEX idx_conversation_participants_user ON conversation_participants(user_id);
CREATE INDEX idx_conversation_participants_conversation ON conversation_participants(conversation_id);
```

#### notifications

```sql
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Recipient
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Type
    type VARCHAR(50) NOT NULL,

    -- Content
    title VARCHAR(255) NOT NULL,
    message TEXT,

    -- Action
    action_url TEXT,

    -- Related entity (polymorphic)
    entity_type VARCHAR(50),
    entity_id UUID,

    -- Actor (who triggered the notification)
    actor_id UUID REFERENCES users(user_id),

    -- Status
    read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP,
    delivered BOOLEAN DEFAULT FALSE,

    -- Channels
    channels TEXT[] DEFAULT ARRAY['in_app'],

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_notifications_user ON notifications(user_id, read, created_at DESC);
CREATE INDEX idx_notifications_type ON notifications(type);
CREATE INDEX idx_notifications_created ON notifications(created_at DESC);
```

#### moderation_logs

```sql
CREATE TABLE moderation_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Moderator
    moderator_id UUID NOT NULL REFERENCES users(user_id),

    -- Action
    action VARCHAR(50) NOT NULL,

    -- Target (polymorphic)
    target_type VARCHAR(20) NOT NULL CHECK (target_type IN ('user', 'post', 'comment', 'community')),
    target_id UUID NOT NULL,

    -- Details
    reason TEXT,
    details JSONB DEFAULT '{}',

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_moderation_logs_moderator ON moderation_logs(moderator_id);
CREATE INDEX idx_moderation_logs_target ON moderation_logs(target_type, target_id);
CREATE INDEX idx_moderation_logs_created ON moderation_logs(created_at DESC);
```

#### achievements

```sql
CREATE TABLE achievements (
    achievement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Details
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon_url TEXT,
    category VARCHAR(50),

    -- Requirements
    criteria JSONB NOT NULL,
    points INTEGER DEFAULT 0,

    -- Rarity
    rarity VARCHAR(20) DEFAULT 'common' CHECK (rarity IN ('common', 'rare', 'epic', 'legendary')),

    -- Visibility
    hidden BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_achievements_category ON achievements(category);
CREATE INDEX idx_achievements_rarity ON achievements(rarity);
```

#### user_achievements

```sql
CREATE TABLE user_achievements (
    user_achievement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    achievement_id UUID NOT NULL REFERENCES achievements(achievement_id),

    -- Progress
    progress INTEGER DEFAULT 0,
    completed BOOLEAN DEFAULT FALSE,

    -- Timestamps
    unlocked_at TIMESTAMP,

    UNIQUE(user_id, achievement_id)
);

CREATE INDEX idx_user_achievements_user ON user_achievements(user_id);
CREATE INDEX idx_user_achievements_completed ON user_achievements(completed, unlocked_at DESC);
```

#### plugins

```sql
CREATE TABLE plugins (
    plugin_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    category VARCHAR(50) NOT NULL,
    description TEXT,
    author VARCHAR(255),

    -- Configuration
    config_schema JSONB,
    default_config JSONB DEFAULT '{}',

    -- Permissions
    permissions TEXT[],

    -- Hooks
    hooks TEXT[],

    -- Status
    status VARCHAR(20) DEFAULT 'inactive' CHECK (status IN ('active', 'inactive', 'deprecated')),

    -- Marketplace
    price DECIMAL(10,2) DEFAULT 0,
    downloads INTEGER DEFAULT 0,
    rating FLOAT,

    -- Timestamps
    installed_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(plugin_id, version)
);

CREATE INDEX idx_plugins_category ON plugins(category);
CREATE INDEX idx_plugins_status ON plugins(status);
```

#### plugin_instances

```sql
CREATE TABLE plugin_instances (
    instance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plugin_id VARCHAR(100) NOT NULL REFERENCES plugins(plugin_id),

    -- Scope
    scope_type VARCHAR(20) CHECK (scope_type IN ('global', 'community')),
    scope_id UUID,  -- community_id if scope_type = 'community'

    -- Configuration
    config JSONB DEFAULT '{}',

    -- Status
    enabled BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(plugin_id, scope_type, scope_id)
);

CREATE INDEX idx_plugin_instances_plugin ON plugin_instances(plugin_id);
CREATE INDEX idx_plugin_instances_scope ON plugin_instances(scope_type, scope_id);
```

---

## Neo4j Graph Schema

### Node Labels

#### :User
```cypher
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.user_id IS UNIQUE;

// Properties: user_id, username, reputation, trust_level
```

#### :Community
```cypher
CREATE CONSTRAINT community_id_unique IF NOT EXISTS
FOR (c:Community) REQUIRE c.community_id IS UNIQUE;

// Properties: community_id, slug, name, member_count
```

#### :Post
```cypher
CREATE CONSTRAINT post_id_unique IF NOT EXISTS
FOR (p:Post) REQUIRE p.post_id IS UNIQUE;

// Properties: post_id, title, vote_score, created_at
```

#### :Tag
```cypher
CREATE CONSTRAINT tag_name_unique IF NOT EXISTS
FOR (t:Tag) REQUIRE t.name IS UNIQUE;

// Properties: name, usage_count
```

### Relationship Types

#### User Relationships

```cypher
// Friendship (bidirectional)
(:User)-[:FRIENDS_WITH {since: DateTime}]->(:User)

// Following (unidirectional)
(:User)-[:FOLLOWS {since: DateTime}]->(:User)

// Blocking
(:User)-[:BLOCKS {since: DateTime}]->(:User)

// Community membership
(:User)-[:MEMBER_OF {role: String, joined_at: DateTime}]->(:Community)

// Post authorship
(:User)-[:AUTHORED {created_at: DateTime}]->(:Post)

// Voting
(:User)-[:VOTED {direction: Integer, created_at: DateTime}]->(:Post)
(:User)-[:VOTED {direction: Integer, created_at: DateTime}]->(:Comment)

// Achievements
(:User)-[:EARNED {unlocked_at: DateTime}]->(:Achievement)
```

#### Content Relationships

```cypher
// Post in community
(:Post)-[:POSTED_IN {created_at: DateTime}]->(:Community)

// Post tags
(:Post)-[:TAGGED_WITH]->(:Tag)

// Similar posts (recommendation)
(:Post)-[:SIMILAR_TO {score: Float}]->(:Post)

// User interests (inferred from activity)
(:User)-[:INTERESTED_IN {score: Float}]->(:Tag)
```

### Example Queries

#### Find mutual friends
```cypher
MATCH (me:User {user_id: $my_id})-[:FRIENDS_WITH]-(friend)-[:FRIENDS_WITH]-(mutual)
WHERE mutual.user_id <> $my_id
  AND NOT (me)-[:FRIENDS_WITH]-(mutual)
RETURN mutual, COUNT(*) as mutual_count
ORDER BY mutual_count DESC
LIMIT 10
```

#### Recommend communities
```cypher
MATCH (me:User {user_id: $my_id})-[:INTERESTED_IN]->(tag)<-[:TAGGED_WITH]-(post)-[:POSTED_IN]->(community)
WHERE NOT (me)-[:MEMBER_OF]->(community)
WITH community, COUNT(*) as relevance
RETURN community
ORDER BY relevance DESC
LIMIT 5
```

#### Find trending posts
```cypher
MATCH (post:Post)-[:POSTED_IN]->(community:Community)
WHERE post.created_at > datetime() - duration('PT24H')
WITH post, post.vote_score / (duration.inSeconds(datetime() - post.created_at) / 3600 + 2)^1.5 AS trending_score
RETURN post
ORDER BY trending_score DESC
LIMIT 50
```

---

## Redis Data Structures

### Session Storage
```
Key: session:{session_id}
Type: Hash
TTL: 1 hour
Fields: user_id, username, role, created_at
```

### Online Presence
```
Key: online:users
Type: Sorted Set
Score: timestamp
Members: user_id
TTL: 5 minutes (sliding)
```

### Rate Limiting
```
Key: ratelimit:{user_id}:{endpoint}
Type: String (counter)
TTL: 1 minute
```

### Hot Posts Cache
```
Key: hot:posts:{community_id}
Type: List
TTL: 5 minutes
Values: post_id (ordered by hot score)
```

### User Feed Cache
```
Key: feed:{user_id}
Type: List
TTL: 10 minutes
Values: post_id (chronological)
```

### Real-Time Events (Pub/Sub)
```
Channel: notifications:{user_id}
Channel: community:{community_id}:posts
Channel: conversation:{conversation_id}:messages
```

---

## Elasticsearch Indexes

### users_index
```json
{
  "mappings": {
    "properties": {
      "user_id": {"type": "keyword"},
      "username": {"type": "keyword"},
      "display_name": {"type": "text"},
      "bio": {"type": "text"},
      "reputation_score": {"type": "integer"},
      "created_at": {"type": "date"}
    }
  }
}
```

### posts_index
```json
{
  "mappings": {
    "properties": {
      "post_id": {"type": "keyword"},
      "title": {"type": "text"},
      "content": {"type": "text"},
      "author_username": {"type": "keyword"},
      "community_slug": {"type": "keyword"},
      "tags": {"type": "keyword"},
      "vote_score": {"type": "integer"},
      "created_at": {"type": "date"}
    }
  }
}
```

### communities_index
```json
{
  "mappings": {
    "properties": {
      "community_id": {"type": "keyword"},
      "slug": {"type": "keyword"},
      "name": {"type": "text"},
      "description": {"type": "text"},
      "member_count": {"type": "integer"},
      "created_at": {"type": "date"}
    }
  }
}
```

---

## Migration Strategy

### Initial Setup
```bash
# PostgreSQL
alembic upgrade head

# Neo4j
cypher-shell < setup_neo4j.cypher

# Elasticsearch
python setup_elasticsearch.py

# Redis
# Auto-configured (no schema)
```

### Data Sync
- PostgreSQL → Neo4j: Periodic sync (every 5 minutes)
- PostgreSQL → Elasticsearch: Real-time indexing (on create/update)
- PostgreSQL → Redis: Cache on read (lazy loading)

---

**Total Tables**: 22 PostgreSQL tables
**Total Node Types**: 5 Neo4j labels
**Total Relationship Types**: 12 Neo4j relationships
**Total ES Indexes**: 3 Elasticsearch indexes

**Author**: Claude Code
**Date**: 2025-11-18
