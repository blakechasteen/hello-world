# Community Platform Architecture

**Date**: 2025-11-18
**Version**: 1.0
**Type**: Hybrid Community Platform

A comprehensive online community platform combining Forum, Social Network, Chat, and Knowledge Base features with an extensible plugin architecture.

---

## Table of Contents

1. [Vision & Philosophy](#vision--philosophy)
2. [System Architecture](#system-architecture)
3. [Core Features](#core-features)
4. [Plugin System](#plugin-system)
5. [Technology Stack](#technology-stack)
6. [Database Design](#database-design)
7. [API Architecture](#api-architecture)
8. [Real-Time System](#real-time-system)
9. [Security & Moderation](#security--moderation)
10. [Deployment Strategy](#deployment-strategy)

---

## Vision & Philosophy

**"WordPress for Online Communities"**

Build a platform that is:
- **Hybrid**: Combines forum, social network, chat, and Q&A in one unified experience
- **Extensible**: Plugin architecture allows infinite customization
- **Scalable**: Supports communities from 10 to 10 million users
- **Open**: Open-source core with marketplace for premium plugins
- **Modern**: Real-time, mobile-first, AI-powered

**Target Users**:
- 🏢 **Organizations**: Internal communities, customer forums
- 🎓 **Education**: Student communities, alumni networks
- 🎮 **Gaming**: Guild forums, esports communities
- 💼 **Professional**: Industry networks, special interest groups
- 🌍 **General**: Hobby communities, local groups

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  Web App (React)  │  Mobile App  │  Desktop App (Electron) │
├─────────────────────────────────────────────────────────────┤
│                      API Gateway (Kong)                      │
├─────────────────────────────────────────────────────────────┤
│                     Backend Services                         │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   REST API  │  │  WebSocket  │  │   GraphQL   │         │
│  │  (FastAPI)  │  │  (Socket.io)│  │ (Strawberry)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Plugin    │  │  Moderation │  │  Search     │         │
│  │   Engine    │  │   Engine    │  │  (Elastic)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │PostgreSQL│ │   Neo4j  │ │  Redis   │ │Elasticsearch│     │
│  │(Primary) │ │ (Social) │ │ (Cache)  │ │ (Search)  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │   S3     │ │RabbitMQ  │ │Prometheus│                     │
│  │ (Media)  │ │ (Queue)  │ │(Metrics) │                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Frontend Applications
- **Web App** (React + TypeScript + Vite)
  - Server-side rendering (Next.js)
  - Progressive Web App (PWA)
  - Offline support

- **Mobile Apps** (React Native)
  - iOS and Android
  - Push notifications
  - Native features (camera, location)

- **Desktop App** (Electron - optional)
  - Native experience
  - System tray integration

#### 2. Backend Services

**Core API** (FastAPI):
- RESTful endpoints for CRUD operations
- JWT authentication + OAuth2
- Rate limiting and throttling
- API versioning (`/api/v1/`)

**Real-Time Service** (Socket.io):
- WebSocket connections
- Presence tracking (online/offline)
- Live notifications
- Chat messages
- Live updates (likes, comments)

**GraphQL API** (Strawberry):
- Flexible data fetching
- Subscriptions for real-time
- Batch requests
- Introspection

**Search Service** (Elasticsearch):
- Full-text search
- Faceted filtering
- Autocomplete
- Relevance ranking

**Plugin Engine**:
- Plugin loading and lifecycle
- Hook system
- Sandboxed execution
- Marketplace integration

**Moderation Engine**:
- Auto-moderation (spam, toxicity)
- Report queue
- Mod tools
- Content filters

#### 3. Data Stores

**PostgreSQL** (Primary relational data):
- Users, posts, comments
- Communities, groups
- Permissions, roles
- Plugin configurations

**Neo4j** (Social graph):
- User connections (friends, followers)
- Community membership
- Content relationships
- Recommendation engine

**Redis** (Caching & real-time):
- Session storage
- Real-time presence
- Rate limiting counters
- Hot data cache
- Pub/sub for events

**Elasticsearch** (Search):
- Full-text search index
- User search
- Content search
- Tag search

**S3 / Object Storage** (Media):
- User avatars
- Post images/videos
- File attachments
- CDN integration

**RabbitMQ** (Message queue):
- Async task processing
- Email notifications
- Webhook deliveries
- Background jobs

---

## Core Features

### 1. User System

**Authentication**:
- Email/password
- OAuth2 (Google, GitHub, Discord)
- Two-factor authentication (2FA)
- Magic link login
- SSO (SAML, LDAP)

**User Profiles**:
- Bio, avatar, cover image
- Social links
- Badges and achievements
- Activity history
- Reputation score

**User Roles**:
- Guest (read-only)
- Member (basic permissions)
- Moderator (community-specific)
- Admin (platform-wide)
- Custom roles (plugin-defined)

### 2. Content Types

**Posts** (Forum-style):
- Title, body (Markdown/rich text)
- Images, videos, embeds
- Tags, categories
- Voting (upvote/downvote)
- Pin, lock, archive

**Comments** (Threaded):
- Nested replies (unlimited depth)
- Voting
- Edit history
- Sorting (best, new, controversial)

**Messages** (Chat):
- Direct messages (1-to-1)
- Group messages
- Channels (Discord-style)
- Message reactions
- File attachments

**Questions** (Q&A):
- Question + answers
- Accepted answer
- Bounties
- Expert tagging

**Social Posts** (Feed):
- Short updates (Twitter-style)
- Photo posts (Instagram-style)
- Link sharing
- Polls

### 3. Communities & Groups

**Communities** (Subreddit-style):
- Public, private, restricted
- Custom themes
- Rules and guidelines
- Moderator team
- Flairs and tags

**Groups** (Facebook-style):
- Join/invite-only
- Events and announcements
- File sharing
- Group chat

**Channels** (Discord-style):
- Text channels
- Voice channels (future)
- Categories
- Permissions per channel

### 4. Social Graph

**Connections**:
- Friends (bidirectional)
- Followers (unidirectional)
- Blocked users
- Muted users

**Feed Algorithm**:
- Chronological
- Algorithmic (engagement-based)
- Filtered by interests
- Personalized recommendations

### 5. Gamification

**Reputation System**:
- Karma points
- Trust level (0-5)
- Decay over time
- Activity-based growth

**Achievements**:
- Badges (custom per community)
- Titles and flairs
- Leaderboards
- Streaks

**Rewards**:
- Community currency
- Premium features
- Special access
- Physical rewards (via plugins)

### 6. Moderation

**Content Moderation**:
- Report system
- Mod queue
- Auto-mod rules
- Content filters (spam, NSFW)
- Shadow banning

**User Moderation**:
- Warnings
- Temporary bans
- Permanent bans
- IP bans
- Mute/restrict

**AI Moderation**:
- Toxicity detection
- Spam detection
- Duplicate detection
- NSFW image detection

### 7. Search & Discovery

**Full-Text Search**:
- Posts, comments, users
- Communities, groups
- Real-time indexing
- Advanced filters

**Discovery**:
- Trending posts
- Popular communities
- Recommended users
- Related content

### 8. Notifications

**Types**:
- Mentions (@username)
- Replies to posts/comments
- Direct messages
- System announcements
- Custom (plugin-defined)

**Channels**:
- In-app notifications
- Email
- Push notifications (mobile)
- Browser notifications
- SMS (via plugin)
- Webhook (via plugin)

### 9. Real-Time Features

**Live Updates**:
- New posts in feed
- Comment count updates
- Vote count updates
- Online presence

**Live Chat**:
- Instant messaging
- Typing indicators
- Read receipts
- Message reactions

**Live Collaboration**:
- Collaborative editing
- Live polls
- Live events

---

## Plugin System

### Plugin Architecture

```
┌────────────────────────────────────────────┐
│           Plugin Marketplace                │
│  (Discovery, ratings, reviews, purchases)  │
├────────────────────────────────────────────┤
│             Plugin Manager                  │
│  (Install, update, enable, configure)      │
├────────────────────────────────────────────┤
│              Plugin Engine                  │
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ Hook System  │  │   Sandbox    │        │
│  │ (100+ hooks) │  │ (Isolation)  │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ Permission   │  │   Database   │        │
│  │   System     │  │    Access    │        │
│  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────┘
```

### Plugin Categories

**1. Content Plugins** (Extend content types):
- Polls
- Events & calendars
- Marketplace (buy/sell)
- Job board
- Wiki pages
- Galleries
- Forms & surveys

**2. Integration Plugins** (External services):
- Payment (Stripe, PayPal)
- Email (SendGrid, Mailgun)
- Analytics (Google Analytics, Mixpanel)
- CRM (Salesforce, HubSpot)
- Social (Twitter, LinkedIn)
- Storage (Dropbox, Google Drive)

**3. Moderation Plugins** (Safety & quality):
- Spam detection (Akismet)
- Profanity filter
- Auto-moderation rules
- Verification system
- Content reporting
- Shadowban tools

**4. Engagement Plugins** (Gamification):
- Leaderboards
- Contests
- Referral system
- Daily challenges
- Loyalty programs
- Virtual currency

**5. Communication Plugins**:
- Email templates
- Webhooks
- Bots (chatbots, moderation bots)
- RSS feeds
- Newsletter integration

**6. Analytics Plugins**:
- User behavior tracking
- A/B testing
- Heatmaps
- Conversion funnels
- Cohort analysis

**7. Customization Plugins**:
- Themes
- Custom CSS/JS
- Widgets
- Custom fields
- Branding

**8. Admin Plugins**:
- Backup & restore
- Import/export
- Database tools
- Performance monitoring
- Security audits

### Plugin Hooks

**100+ Hook Points**:

**Content Hooks**:
- `before_post_create`, `after_post_create`
- `before_post_update`, `after_post_update`
- `before_post_delete`, `after_post_delete`
- `post_render`, `post_vote`

**User Hooks**:
- `user_register`, `user_login`, `user_logout`
- `user_profile_update`, `user_avatar_change`
- `user_reputation_change`

**Community Hooks**:
- `community_create`, `community_join`, `community_leave`
- `moderator_add`, `moderator_remove`

**System Hooks**:
- `notification_send`, `email_send`
- `search_index`, `cache_clear`
- `webhook_trigger`

---

## Technology Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **ORM**: SQLAlchemy (async)
- **Real-Time**: Socket.io
- **GraphQL**: Strawberry
- **Task Queue**: Celery + RabbitMQ
- **Testing**: pytest, pytest-asyncio

### Frontend
- **Framework**: React 18 + TypeScript
- **Build**: Vite
- **State**: Redux Toolkit + RTK Query
- **Styling**: Tailwind CSS + Headless UI
- **Forms**: React Hook Form + Zod
- **Rich Text**: TipTap or Lexical
- **Real-Time**: Socket.io-client
- **Testing**: Vitest, React Testing Library

### Databases
- **Primary**: PostgreSQL 15
- **Social Graph**: Neo4j 5
- **Cache**: Redis 7
- **Search**: Elasticsearch 8
- **Media**: S3-compatible (MinIO, AWS S3)

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (production)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack
- **CDN**: Cloudflare

---

## Database Design

See [COMMUNITY_DATABASE_SCHEMA.md](COMMUNITY_DATABASE_SCHEMA.md) for complete schema.

**Key Tables**:
- `users` (20+ columns)
- `posts` (content, votes, status)
- `comments` (threaded, nested)
- `communities` (subreddit-style)
- `groups` (private communities)
- `messages` (DMs and channels)
- `notifications` (multi-channel)
- `moderation_logs` (audit trail)
- `plugins` (installed plugins)
- `achievements` (badges, titles)

**Total**: 40+ tables

---

## API Architecture

### REST API (`/api/v1/`)

**Authentication**:
- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/logout`
- `POST /auth/refresh`
- `POST /auth/oauth/{provider}`

**Users**:
- `GET /users/{id}`
- `PATCH /users/{id}`
- `GET /users/{id}/posts`
- `GET /users/{id}/followers`
- `POST /users/{id}/follow`

**Posts**:
- `GET /posts` (list with filters)
- `POST /posts` (create)
- `GET /posts/{id}`
- `PATCH /posts/{id}`
- `DELETE /posts/{id}`
- `POST /posts/{id}/vote`
- `GET /posts/{id}/comments`

**Communities**:
- `GET /communities`
- `POST /communities`
- `GET /communities/{id}`
- `POST /communities/{id}/join`
- `GET /communities/{id}/posts`

**Messages**:
- `GET /messages/conversations`
- `GET /messages/conversations/{id}`
- `POST /messages/conversations/{id}/messages`

### WebSocket Events

**Client → Server**:
- `join_room` (community, channel, DM)
- `leave_room`
- `send_message`
- `typing_start`, `typing_stop`
- `presence_update`

**Server → Client**:
- `message_received`
- `user_joined`, `user_left`
- `user_typing`
- `post_updated`
- `notification_received`

### GraphQL Schema

```graphql
type User {
  id: ID!
  username: String!
  avatar: String
  reputation: Int!
  posts(limit: Int): [Post!]!
  followers: [User!]!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  community: Community
  votes: Int!
  comments: [Comment!]!
  createdAt: DateTime!
}

type Query {
  user(id: ID!): User
  post(id: ID!): Post
  feed(cursor: String, limit: Int): PostConnection!
}

type Mutation {
  createPost(input: CreatePostInput!): Post!
  votePost(id: ID!, direction: VoteDirection!): Post!
}

type Subscription {
  postAdded(communityId: ID): Post!
  messageReceived(conversationId: ID!): Message!
}
```

---

## Real-Time System

### WebSocket Architecture

```
Client                WebSocket Server           Redis Pub/Sub
  │                         │                         │
  ├─ Connect ──────────────>│                         │
  │                         ├─ Store connection       │
  │                         ├─ Join room ────────────>│
  │                         │                         │
  ├─ Send message ─────────>│                         │
  │                         ├─ Publish ──────────────>│
  │                         │                         │
  │                         │<─ Receive ──────────────┤
  │<─ Broadcast message ────┤                         │
  │                         │                         │
```

**Scaling Strategy**:
- Sticky sessions (Nginx)
- Redis adapter for Socket.io
- Horizontal scaling (multiple instances)
- Load balancer (HAProxy or ALB)

---

## Security & Moderation

### Security Measures

**Authentication**:
- Bcrypt password hashing
- JWT with short expiry (15 min)
- Refresh token rotation
- Rate limiting (100 req/min per IP)

**Authorization**:
- Role-based access control (RBAC)
- Permission checks on every request
- Scoped API tokens

**Data Protection**:
- HTTPS only (TLS 1.3)
- CORS configuration
- XSS prevention (sanitization)
- CSRF tokens
- SQL injection prevention (ORM)

**Privacy**:
- GDPR compliance (data export, deletion)
- Privacy settings (profile visibility)
- Anonymous posting (optional)
- IP logging (optional)

### Moderation Tools

**Auto-Moderation**:
- Spam detection (Bayesian filtering)
- Toxicity detection (Perspective API)
- Link validation
- Rate limiting (posting frequency)

**Manual Moderation**:
- Report queue
- Mod log (all actions logged)
- Bulk actions
- Custom rules (regex, keywords)

---

## Deployment Strategy

### Development
```bash
docker-compose up
```

### Staging
- Kubernetes cluster
- Auto-deploy on push to `develop`
- Smoke tests after deploy

### Production
- Multi-region Kubernetes
- Blue-green deployment
- Canary releases
- Auto-scaling (HPA)

### Monitoring
- Prometheus metrics
- Grafana dashboards
- Sentry error tracking
- Uptime monitoring

---

## Roadmap

### Phase 1: Core Platform (Months 1-3)
- ✅ Architecture design
- ✅ Database schema
- ✅ User authentication
- ✅ Posts & comments
- ✅ Basic moderation

### Phase 2: Social Features (Months 4-6)
- Communities & groups
- Social graph (friends, followers)
- News feed algorithm
- Notifications
- Real-time chat

### Phase 3: Advanced Features (Months 7-9)
- Plugin system
- Gamification
- Search & discovery
- Mobile apps
- Analytics

### Phase 4: Scale & Polish (Months 10-12)
- Performance optimization
- Advanced moderation
- AI features
- Marketplace
- Enterprise features

---

**Author**: Claude Code
**Date**: 2025-11-18
**Version**: 1.0
