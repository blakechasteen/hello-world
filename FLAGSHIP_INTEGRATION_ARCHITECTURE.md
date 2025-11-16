# Flagship Integration Architecture
## Comprehensive Expansion Plan for HoloLoom Documentation Ecosystem

**Document Version:** 1.0
**Created:** November 16, 2025
**Status:** Architecture & Implementation Roadmap
**Target Completion:** 4-6 weeks (5 implementation waves)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Component Architecture](#component-architecture)
3. [Site Map Expansion](#site-map-expansion)
4. [Design System Integration](#design-system-integration)
5. [Data Architecture](#data-architecture)
6. [Implementation Phases](#implementation-phases)
7. [Technical Stack](#technical-stack)
8. [User Flows](#user-flows)
9. [Integration Details](#integration-details)
10. [Appendices](#appendices)

---

## Executive Summary

### Vision

Transform the HoloLoom documentation site from a single-component reference into a **comprehensive ecosystem hub** that unifies:

- **Learning & Documentation** (HoloLoom core + Promptly tutorials)
- **Community & Support** (Forums, Q&A, discussions)
- **Issue Management** (GitHub-style tracking, roadmap)
- **Platform Discovery** (BigPlay unified dashboard)

This creates a **network effect**: users visiting for Promptly documentation discover HoloLoom; users asking questions in forums find solutions and tutorials; contributors see roadmap and can submit issues directly.

### Key Success Metrics

- **<1s page load time** across all new components
- **WCAG AAA accessibility** throughout (no regressions)
- **Zero external dependencies** (pure HTML/CSS/JS or clearly justified)
- **Mobile-first responsive design** (tested on iOS/Android)
- **Cross-component search** (unified index across all 4 platforms)
- **90%+ cache hit rate** (heavy use of localStorage + service workers)

### Core Components

```
┌──────────────────────────────────────────────────────────┐
│         HoloLoom Flagship Ecosystem (8.0)               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  1. PROMPTLY DOCUMENTATION HUB                  │   │
│  │     Quick Start | API Reference | Examples      │   │
│  │     VS Code Extension | MCP Server              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  2. ISSUE TRACKER                              │   │
│  │     GitHub-style management, Milestones        │   │
│  │     Labels, Filtering, Discussion threads      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  3. COMMUNITY FORUMS                           │   │
│  │     Threaded discussions, Q&A                  │   │
│  │     Categories, Search, User profiles          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  4. BIGPLAY DASHBOARD                          │   │
│  │     Unified discovery, Activity feed           │   │
│  │     Getting started wizard, Cross-search       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Shared Infrastructure                          │   │
│  │  ├─ Authentication (optional guest access)    │   │
│  │  ├─ Search index (unified, 200k+ docs)        │   │
│  │  ├─ Design system (main.css, extended)        │   │
│  │  ├─ Storage (localStorage + optional backend) │   │
│  │  └─ Service worker (offline-first caching)    │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### User Journeys

**Journey 1: New User Onboarding**
```
Visitor arrives at hololoom.dev/
    ↓
BigPlay Dashboard (Activity + Getting Started)
    ↓
Chooses path: "Learn HoloLoom" OR "Try Promptly"
    ↓
Training docs OR Promptly quick start
    ↓
Bookmarks docs, explores forums, submits issues
```

**Journey 2: Finding Solutions**
```
User has question about Promptly
    ↓
Uses global search (⌘K/Ctrl+K)
    ↓
Finds: Docs + Forums + Issue discussions
    ↓
Reads forum thread, finds answer
    ↓
Upvotes helpful answer, joins community
```

**Journey 3: Contributing**
```
User finds bug in Promptly
    ↓
Navigates to /issues/new
    ↓
Creates issue with template
    ↓
Community discusses, developer confirms
    ↓
Issue appears on roadmap
    ↓
User submits PR (via GitHub, linked from issue)
```

---

## Component Architecture

### 1. Promptly Documentation Hub

**Purpose:** Complete reference for Promptly platform (CLI, web, MCP server, VS Code extension)

**Location:** `/promptly/` (new section)

**Key Pages:**

#### 1.1 Quick Start (`/promptly/quick-start.html`)
- Installation instructions (CLI, pip, Docker)
- First-time setup (5 minutes)
- Running your first loop (REFINE example)
- Next steps (API reference / Web dashboard)
- Video walkthroughs (embedded)

**Structure:**
```html
<article class="promptly-quickstart">
  <section id="install">
    <h2>Installation</h2>
    <!-- Platform-specific tabs -->
  </section>
  <section id="first-loop">
    <h2>Your First Loop</h2>
    <!-- Step-by-step example -->
  </section>
  <section id="explore">
    <h2>Explore Features</h2>
    <!-- Links to detailed docs -->
  </section>
</article>
```

#### 1.2 Loop Types Reference (`/promptly/loop-types/`)
- `/promptly/loop-types/refine.html` - Refinement loops
- `/promptly/loop-types/critique.html` - Self-critique loops
- `/promptly/loop-types/decompose.html` - Problem decomposition
- `/promptly/loop-types/verify.html` - Verification loops
- `/promptly/loop-types/explore.html` - Exploration loops
- `/promptly/loop-types/hofstadter.html` - Meta-reasoning loops

Each page:
- What it does (1-2 paragraph overview)
- When to use it (use cases)
- Example code (copy-paste ready)
- API reference (parameters, return values)
- Advanced patterns (composition, chaining)
- Community examples (user contributions)

#### 1.3 API Reference (`/promptly/api/`)
- `/promptly/api/core.html` - Core classes and functions
- `/promptly/api/cli.html` - Command-line interface
- `/promptly/api/web.html` - Web dashboard API
- `/promptly/api/mcp.html` - MCP server tools (27 tools)
- `/promptly/api/hololoom.html` - HoloLoom integration

**Features:**
- Interactive search (client-side, 50ms)
- Code examples with syntax highlighting
- Parameter tables with type signatures
- Exception documentation
- Version history (backward compatibility notes)

#### 1.4 Integration Guides (`/promptly/guides/`)
- `/promptly/guides/hololoom.html` - Neural memory integration
- `/promptly/guides/claude-desktop.html` - MCP in Claude Desktop
- `/promptly/guides/vscode-extension.html` - VS Code extension setup
- `/promptly/guides/team-collaboration.html` - Multi-user workflows
- `/promptly/guides/analytics.html` - Dashboard and metrics
- `/promptly/guides/deployment.html` - Self-hosted setup

#### 1.5 Examples Gallery (`/promptly/examples/`)
- Sample prompts (search, filter by loop type)
- Recursive chains (composition examples)
- Real-world applications (coding, writing, analysis)
- User-contributed examples (community section)

**Features:**
- Code viewer with copy button
- Performance metrics (tokens, cost, latency)
- Related tutorials and API docs
- "Try it" button (opens web editor)

#### 1.6 VS Code Extension Guide (`/promptly/vscode/`)
- Installation and setup
- Keyboard shortcuts
- Theme configuration
- Extension settings
- Troubleshooting

#### 1.7 Video Library (`/promptly/videos/`)
- Embedded YouTube tutorials (fallback: transcripts)
- Playlist organization (Beginner → Advanced)
- Transcript search
- Related documentation links

---

### 2. Issue Tracker

**Purpose:** GitHub-style issue management (client-side + optional backend)

**Location:** `/issues/` (new section)

**Architecture: Hybrid Client-Side + Optional Backend**

For MVP: Pure client-side (localStorage) with GitHub sync
For production: Optional Firebase/Supabase backend for persistence + real-time

#### 2.1 Issue List View (`/issues/`)
- Searchable, filterable list
- Status badges (Open, In Progress, Closed, Won't Fix)
- Labels (bug, feature, documentation, etc.)
- Sorting options (Newest, Most commented, Most liked)
- Pagination (50 per page)

**HTML Structure:**
```html
<div class="issues-container">
  <header class="issues-header">
    <h1>Issues & Roadmap</h1>
    <button class="btn-primary" data-action="create-issue">
      Create Issue
    </button>
  </header>

  <aside class="issues-sidebar">
    <!-- Filters -->
    <div class="filter-group">
      <h3>Status</h3>
      <label><input type="checkbox" name="status" value="open"> Open (42)</label>
      <label><input type="checkbox" name="status" value="in-progress"> In Progress (8)</label>
      <label><input type="checkbox" name="status" value="closed"> Closed (215)</label>
    </div>

    <div class="filter-group">
      <h3>Label</h3>
      <div class="label-list">
        <button class="label-badge" data-label="bug">bug (12)</button>
        <button class="label-badge" data-label="feature">feature (28)</button>
        <!-- More labels -->
      </div>
    </div>

    <div class="filter-group">
      <h3>Milestone</h3>
      <select id="milestone-filter">
        <option value="">All milestones</option>
        <option value="v2.0">v2.0 - Neural Expansion</option>
        <option value="v1.5">v1.5 - Routing & Learning</option>
      </select>
    </div>
  </aside>

  <main class="issues-list">
    <div class="issues-toolbar">
      <input type="search" id="issue-search" placeholder="Search issues...">
      <select id="sort-by">
        <option value="newest">Newest</option>
        <option value="commented">Most commented</option>
        <option value="liked">Most liked</option>
      </select>
    </div>

    <div class="issues-table">
      <!-- Issue items generated by JS -->
    </div>

    <nav class="pagination">
      <!-- Pagination controls -->
    </nav>
  </main>
</div>
```

#### 2.2 Issue Detail View (`/issues/:id`)
- Full issue discussion
- Comments thread
- Related issues
- Upvote/downvote buttons
- Subscribe button
- Author profile (optional)

**Key Sections:**

1. **Issue Header**
   - Title, ID (#123)
   - Status badge
   - Labels
   - Created by [Author] · [Date] · [#Comments]

2. **Issue Description**
   - Full markdown content
   - Code blocks with syntax highlighting
   - Embedded images (markdown support)
   - "Edit" button (if author)

3. **Discussion Thread**
   - Chronological comments
   - Author avatars (optional)
   - Like counts
   - Nested replies (2 levels)

4. **Sidebar**
   - Status dropdown
   - Assignee selection
   - Labels (multi-select)
   - Milestone selection
   - Related issues (backlinks)
   - Linked PRs (GitHub sync)

#### 2.3 Create Issue Form (`/issues/new`)
- Rich text editor (markdown preview)
- Template selection (Bug / Feature / Documentation)
- Automatic fields based on template
- Preview before submit
- Issue guidelines sidebar

**Template: Bug Report**
```markdown
## Description
[What's the problem?]

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
[What should happen?]

## Actual Behavior
[What's happening instead?]

## Environment
- HoloLoom version:
- Python version:
- OS:

## Attachments
[Screenshots, logs, etc.]
```

**Template: Feature Request**
```markdown
## Problem
[What problem does this solve?]

## Proposed Solution
[How should this work?]

## Alternatives
[Other approaches considered]

## Examples
[Real-world use cases]

## Additional Context
[Anything else relevant?]
```

#### 2.4 Roadmap View (`/issues/roadmap`)
- Kanban board (Open → In Progress → Review → Closed)
- Milestone grouping
- Drag-to-update status (if logged in)
- Timeline view (Gantt-style)
- Burndown charts (optional)

```html
<div class="roadmap-container">
  <div class="kanban-board">
    <div class="kanban-column" data-status="backlog">
      <h3>Backlog</h3>
      <div class="card-list">
        <!-- Issue cards -->
      </div>
    </div>
    <div class="kanban-column" data-status="in-progress">
      <h3>In Progress</h3>
      <div class="card-list">
        <!-- Issue cards -->
      </div>
    </div>
    <div class="kanban-column" data-status="in-review">
      <h3>In Review</h3>
      <div class="card-list">
        <!-- Issue cards -->
      </div>
    </div>
    <div class="kanban-column" data-status="done">
      <h3>Done</h3>
      <div class="card-list">
        <!-- Issue cards -->
      </div>
    </div>
  </div>
</div>
```

#### 2.5 Advanced Features

**GitHub Sync**
- Webhooks to auto-sync GitHub issues
- Two-way sync (discussions link back to GitHub)
- Comments from both platforms

**Analytics Dashboard**
- Open/closed rate
- Time-to-close metrics
- Most discussed issues
- Community engagement metrics

**Notifications**
- localStorage-based (no backend needed)
- Watch issue → notification bell
- Email notifications (optional backend)

---

### 3. Community Forums

**Purpose:** Discussions, Q&A, best practices, user stories

**Location:** `/community/` (new section)

**Philosophy:** Encourage peer-to-peer learning, surface expert answers, build community

#### 3.1 Forum Categories

```
Community Forums (Total: 5 categories)
├── General Discussion (general questions, announcements)
├── HoloLoom (neural systems, memory, orchestration)
├── Promptly (loops, prompt engineering, integrations)
├── Development (contributing, architecture, roadmap)
├── Show & Tell (projects, demos, blog posts)
```

#### 3.2 Category Views (`/community/:category/`)

**Layout:**
```html
<div class="forum-container">
  <header class="forum-header">
    <h1>HoloLoom Category</h1>
    <p>Neural systems, memory architecture, embeddings, and more</p>
    <button class="btn-primary">Start Discussion</button>
  </header>

  <aside class="forum-sidebar">
    <div class="stats-panel">
      <div class="stat">
        <div class="stat-value">1,247</div>
        <div class="stat-label">Discussions</div>
      </div>
      <div class="stat">
        <div class="stat-value">3,421</div>
        <div class="stat-label">Replies</div>
      </div>
      <div class="stat">
        <div class="stat-value">892</div>
        <div class="stat-label">Members</div>
      </div>
    </div>

    <div class="forum-filters">
      <h3>Filter</h3>
      <select id="sort-by">
        <option value="recent">Recent</option>
        <option value="unanswered">Unanswered</option>
        <option value="popular">Popular</option>
        <option value="bounty">Has Bounty</option>
      </select>
      <input type="search" placeholder="Search category...">
    </div>
  </aside>

  <main class="forum-threads">
    <!-- Thread list -->
  </main>
</div>
```

#### 3.3 Thread View (`/community/threads/:id`)

**Thread Components:**

1. **Original Post**
   - Author name (+ badge if expert/moderator)
   - Timestamp
   - Question title (h1)
   - Detailed description
   - Tags (#memory, #embeddings, etc.)
   - Vote count (↑↓)
   - View count

2. **Replies Section**
   - Chronological thread
   - Sort options: Newest/Oldest/Best
   - "Accepted answer" mark (if Q&A style)
   - Vote counts on replies
   - "Helpful" badges

3. **Reply Box**
   - Rich text editor (markdown)
   - Preview toggle
   - Submit button
   - Anonymous option (if not logged in)

4. **Sidebar**
   - Related threads (similar tags)
   - Related documentation
   - "Ask follow-up" link
   - Report/Flag button

```html
<article class="forum-thread">
  <div class="thread-header">
    <h1>How to use Matryoshka embeddings for semantic search?</h1>
    <div class="thread-meta">
      Asked by <span class="author">@user123</span> ·
      <time datetime="2025-11-10">Nov 10, 2025</time>
    </div>
  </div>

  <div class="thread-content">
    <!-- Original post -->
  </div>

  <section class="thread-answers">
    <h2>2 Answers</h2>
    <div class="answer">
      <!-- Answer content -->
      <button class="btn-secondary" data-action="mark-accepted">
        Mark as accepted
      </button>
    </div>
  </section>

  <section class="reply-box">
    <h3>Your Answer</h3>
    <!-- Rich text editor -->
  </section>
</article>
```

#### 3.4 Q&A Features

**Reputation System** (optional, stored in localStorage)
- Upvoting answer: +10 reputation
- Accepting answer: +15 reputation
- Answer accepted: +25 reputation
- Reaching milestones: badges (100 points → "Rookie", 500 → "Expert", etc.)

**Best Answer Selection**
- Original poster marks best answer
- Shows green checkmark
- Listed first in thread
- Highlighted in category listing

**Follow Tags**
- `/community/tags/:tag` - View all threads with tag
- Subscribe to tag notifications
- Personalized feed by followed tags

#### 3.5 User Profiles (`/community/users/:username`)

**Profile Page:**
- Avatar (optional)
- Bio
- Reputation score + badges
- Recent posts
- Expertise tags
- Join date
- Follow button

**Stored in:** localStorage (anonymous guest profiles) + optional backend

#### 3.6 Forum Moderation

**Tools for Moderators:**
- Lock thread (no new replies)
- Pin thread (sticky in category)
- Move thread to different category
- Merge duplicate threads
- Hide inappropriate content

**Community Guidelines:**
- Displayed on `/community/guidelines`
- Code of conduct
- Spam/harassment policy
- License/attribution

---

### 4. BigPlay Dashboard

**Purpose:** Unified platform discovery, activity feed, getting started wizard

**Location:** `/dashboard/` (new section, or replace index.html)

#### 4.1 Dashboard Home (`/dashboard/` or `/`)

**Hero Section:**
```html
<section class="bigplay-hero">
  <h1>HoloLoom & Promptly Ecosystem</h1>
  <p>Production-ready neural systems and prompt engineering in one place</p>
  <div class="hero-buttons">
    <button class="btn-primary" data-action="start-wizard">
      Getting Started
    </button>
    <button class="btn-secondary" data-action="explore-docs">
      Explore Documentation
    </button>
  </div>
</section>
```

#### 4.2 Getting Started Wizard

**Step 1: What's Your Role?**
- Developer (building with HoloLoom)
- Prompter (using Promptly)
- Researcher (exploring AI/ML)
- Contributor (helping the project)

**Step 2: Learning Path**
Based on role:
- HoloLoom Path: Architecture → Core Concepts → Hands-on
- Promptly Path: Quick Start → Loop Types → Advanced
- Research Path: Papers → Deep dives → Experiments
- Contributor Path: Setup → Architecture → Contributing guide

**Step 3: Suggested Resources**
- Personalized links
- Video tutorials
- Example code
- Community forums relevant to role

**Step 4: Quick Setup**
- Installation command (copy-paste)
- First project starter
- IDE configuration (VS Code extension)

#### 4.3 Activity Feed

**Real-time Activity:**
- New forum posts (+ comments)
- Issues updated
- Documentation published
- New examples added
- Community milestones

**Features:**
- Filter by type (Issues, Discussions, Docs)
- Time range (Today, This week, This month)
- Personalization (follow tags/projects)

```html
<section class="activity-feed">
  <h2>Latest Activity</h2>
  <div class="feed-filters">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="discussions">Discussions</button>
    <button class="filter-btn" data-filter="issues">Issues</button>
    <button class="filter-btn" data-filter="docs">Docs</button>
  </div>
  <div class="feed-list">
    <!-- Activity items -->
  </div>
</section>
```

#### 4.4 Ecosystem Overview

**Stats Dashboard:**
```
┌──────────────────────────────────────────┐
│         HoloLoom Ecosystem Stats          │
├──────────────────────────────────────────┤
│                                          │
│  Docs          Issues        Forums      │
│  245 pages     127 open      2,341 posts │
│  18 hours      42 in progress 892 members│
│  viewed                                  │
│                                          │
└──────────────────────────────────────────┘
```

**Quick Links Grid:**
```html
<section class="ecosystem-overview">
  <h2>Explore the Ecosystem</h2>
  <div class="link-grid">
    <a href="/docs/" class="link-card">
      <h3>Documentation</h3>
      <p>245 pages · 5 libraries</p>
      <span class="link-arrow">→</span>
    </a>
    <a href="/issues/" class="link-card">
      <h3>Issues & Roadmap</h3>
      <p>127 open · 8 in progress</p>
      <span class="link-arrow">→</span>
    </a>
    <a href="/community/" class="link-card">
      <h3>Community Forums</h3>
      <p>2,341 discussions · 892 members</p>
      <span class="link-arrow">→</span>
    </a>
    <a href="/promptly/" class="link-card">
      <h3>Promptly Docs</h3>
      <p>Quick start · API reference</p>
      <span class="link-arrow">→</span>
    </a>
  </div>
</section>
```

#### 4.5 Search Integration

**Global Search** (`⌘K` or `Ctrl+K`)
- Unified search across all components
- 200k+ documents indexed
- Type-ahead suggestions
- Filter by component (Docs, Issues, Forums, Promptly)
- Result previews

**Search Index Contents:**
- All documentation pages (25k entries)
- All issues (150 entries)
- All forum threads (2.3k entries)
- Promptly API docs (500 entries)
- Video transcripts (70+ entries)

```javascript
// Search implementation (client-side)
class UnifiedSearch {
  constructor() {
    this.indices = {
      docs: new Fuse(docsData, { /* opts */ }),
      issues: new Fuse(issuesData, { /* opts */ }),
      forums: new Fuse(forumsData, { /* opts */ }),
      promptly: new Fuse(promptlyData, { /* opts */ })
    };
  }

  search(query, filters = {}) {
    const results = {};

    if (!filters.component || filters.component === 'docs') {
      results.docs = this.indices.docs.search(query).slice(0, 5);
    }
    if (!filters.component || filters.component === 'issues') {
      results.issues = this.indices.issues.search(query).slice(0, 5);
    }
    if (!filters.component || filters.component === 'forums') {
      results.forums = this.indices.forums.search(query).slice(0, 5);
    }
    if (!filters.component || filters.component === 'promptly') {
      results.promptly = this.indices.promptly.search(query).slice(0, 5);
    }

    return results;
  }
}
```

---

## Site Map Expansion

### Current Site Structure
```
docs/
├── index.html (home)
├── start.html (quick start)
├── training/
│   ├── part1.html
│   ├── part2.html
│   ├── part3.html
│   ├── part4.html
│   └── part5.html
└── interactive/ (diagrams, gallery)
```

### Expanded Site Map

```
docs/
├── index.html → /dashboard (BigPlay home)
│
├── docs/ (core documentation)
│   ├── index.html (main docs hub)
│   ├── start.html (getting started)
│   ├── training/
│   │   ├── part1.html
│   │   ├── part2.html
│   │   ├── part3.html
│   │   ├── part4.html
│   │   └── part5.html
│   ├── api/
│   │   ├── index.html
│   │   ├── core.html
│   │   ├── memory.html
│   │   ├── orchestrator.html
│   │   └── rag.html
│   └── guides/
│       ├── installation.html
│       ├── first-project.html
│       └── production.html
│
├── promptly/ (Promptly documentation)
│   ├── index.html (Promptly home)
│   ├── quick-start.html
│   ├── loop-types/
│   │   ├── index.html
│   │   ├── refine.html
│   │   ├── critique.html
│   │   ├── decompose.html
│   │   ├── verify.html
│   │   ├── explore.html
│   │   └── hofstadter.html
│   ├── api/
│   │   ├── index.html
│   │   ├── core.html
│   │   ├── cli.html
│   │   ├── web.html
│   │   ├── mcp.html
│   │   └── hololoom.html
│   ├── guides/
│   │   ├── installation.html
│   │   ├── hololoom-integration.html
│   │   ├── claude-desktop.html
│   │   ├── vscode-extension.html
│   │   ├── team-collaboration.html
│   │   ├── analytics.html
│   │   └── deployment.html
│   ├── examples/
│   │   ├── index.html
│   │   ├── gallery.html (searchable)
│   │   └── user-submitted.html
│   ├── videos/
│   │   ├── index.html
│   │   └── transcripts/
│   └── faq.html
│
├── issues/ (Issue Tracker)
│   ├── index.html (issue list)
│   ├── :id/ (issue detail)
│   ├── new.html (create issue)
│   ├── roadmap.html (Kanban view)
│   └── labels/ (label views)
│
├── community/ (Forums)
│   ├── index.html (categories)
│   ├── guidelines.html
│   ├── :category/
│   │   ├── index.html (thread list)
│   │   └── :thread-id/ (thread view)
│   ├── new.html (create thread)
│   ├── tags/
│   │   └── :tag/ (tag view)
│   └── users/
│       └── :username/ (user profile)
│
├── dashboard/ (BigPlay Dashboard)
│   ├── index.html (main dashboard)
│   ├── wizard.html (onboarding)
│   ├── activity.html (feed)
│   ├── ecosystem.html (overview)
│   ├── search.html (search results)
│   └── notifications.html (user's notifications)
│
└── assets/
    ├── css/
    │   ├── main.css (shared design system)
    │   ├── promptly.css (Promptly-specific)
    │   ├── issues.css (Issue tracker styles)
    │   ├── community.css (Forum styles)
    │   ├── dashboard.css (BigPlay styles)
    │   └── print.css (print-friendly)
    ├── js/
    │   ├── nav.js (shared navigation)
    │   ├── search.js (search implementation)
    │   ├── theme.js (dark mode)
    │   ├── promptly.js (Promptly features)
    │   ├── issues.js (Issue tracker)
    │   ├── community.js (Forums)
    │   ├── dashboard.js (BigPlay)
    │   ├── unified-index.js (search index)
    │   └── offline.js (service worker)
    └── data/
        ├── search-index.json (200k+ docs)
        ├── issues.json (150 issues)
        ├── forum-threads.json (2.3k threads)
        └── promptly-api.json (API docs)
```

### URL Schema

**Documentation:**
- `/docs/` - Core docs home
- `/docs/start` - Getting started
- `/docs/api/core` - API reference
- `/docs/guides/production` - Guides

**Promptly:**
- `/promptly/` - Promptly home
- `/promptly/quick-start` - Quick start
- `/promptly/loop-types/refine` - Loop type docs
- `/promptly/api/core` - API reference
- `/promptly/guides/hololoom` - Integration guide
- `/promptly/examples` - Example gallery
- `/promptly/videos` - Video tutorials

**Issues:**
- `/issues/` - List view
- `/issues/123` - Issue detail
- `/issues/new` - Create issue
- `/issues/roadmap` - Kanban view
- `/issues?label=bug` - Filtered view

**Community:**
- `/community/` - Categories
- `/community/hololoom` - Category view
- `/community/threads/123` - Thread view
- `/community/new` - Create thread
- `/community/tags/memory` - Tag view
- `/community/users/alice` - User profile

**Dashboard:**
- `/dashboard/` - Main dashboard (or `/`)
- `/dashboard/wizard` - Getting started wizard
- `/dashboard/activity` - Activity feed
- `/dashboard/ecosystem` - Ecosystem overview
- `/search?q=term` - Search results

### Breadcrumb Hierarchy

```
Dashboard
  └─ Getting Started Wizard
  └─ Activity Feed
  └─ Ecosystem Overview

Documentation
  ├─ Getting Started
  ├─ Training (Parts 1-5)
  ├─ API Reference
  │   ├─ Core
  │   ├─ Memory
  │   └─ Orchestrator
  └─ Guides

Promptly
  ├─ Quick Start
  ├─ Loop Types
  │   ├─ Refine
  │   ├─ Critique
  │   └─ ...
  ├─ API Reference
  ├─ Integration Guides
  │   ├─ HoloLoom
  │   ├─ Claude Desktop
  │   └─ VS Code
  ├─ Examples
  └─ Videos

Issues & Roadmap
  ├─ Issues List
  ├─ Create New Issue
  ├─ Roadmap
  └─ Issue #123

Community Forums
  ├─ Categories
  │   ├─ General Discussion
  │   ├─ HoloLoom
  │   ├─ Promptly
  │   ├─ Development
  │   └─ Show & Tell
  ├─ Thread View
  ├─ Create Discussion
  ├─ Tags
  └─ User Profiles
```

---

## Design System Integration

### Design System Foundation

**File:** `/docs/assets/css/main.css` (existing, 2000+ lines)

**Includes:**
- CSS custom properties (colors, typography, spacing, shadows)
- Component patterns (buttons, cards, forms, tables)
- Layout systems (grid, flexbox, sidebar)
- Dark mode support
- Responsive breakpoints (mobile, tablet, desktop)
- Accessibility features (focus states, high contrast)

### Component Extensions

#### New Component Classes

1. **Discussion Threads** (community.css)
```css
.thread-list { /* List of threads */ }
.thread-item { /* Single thread preview */ }
.thread-card { /* Detailed thread card */ }
.comment-thread { /* Comment tree */ }
.comment-item { /* Single comment */ }
.reply-box { /* Reply editor */ }
.vote-controls { /* Up/down voting */ }
.thread-tags { /* Tag display */ }
.user-badge { /* User authority indicator */ }
```

2. **Issue Tracker** (issues.css)
```css
.issue-list { /* Issue list view */ }
.issue-item { /* Issue preview */ }
.issue-detail { /* Full issue view */ }
.issue-status { /* Status badge */ }
.issue-labels { /* Label display */ }
.issue-sidebar { /* Metadata sidebar */ }
.kanban-board { /* Roadmap Kanban */ }
.kanban-column { /* Kanban column */ }
.kanban-card { /* Issue card */ }
```

3. **Dashboard** (dashboard.css)
```css
.bigplay-hero { /* Hero section */ }
.getting-started-wizard { /* Onboarding wizard */ }
.activity-feed { /* Activity stream */ }
.ecosystem-overview { /* Stats grid */ }
.link-grid { /* Link card grid */ }
.search-modal { /* Global search dialog */ }
.search-results { /* Search result list */ }
```

4. **Promptly Documentation** (promptly.css)
```css
.promptly-quickstart { /* Quick start page */ }
.loop-type-card { /* Loop type card */ }
.api-reference { /* API documentation */ }
.code-example { /* Code snippet */ }
.integration-guide { /* Guide layout */ }
.examples-gallery { /* Example showcase */ }
```

#### Existing Component Reuse

**Buttons**
- `.btn-primary` - Primary action (blue)
- `.btn-secondary` - Secondary action (gray)
- `.btn-ghost` - Ghost button (transparent)
- `.btn-danger` - Destructive action (red)
- Variants: `size-sm`, `size-lg`, `state-disabled`, `state-loading`

**Cards**
- `.card` - Standard card with shadow
- `.card-header` - Card title section
- `.card-content` - Main content area
- `.card-footer` - Action footer
- Variants: `highlight`, `interactive`, `minimal`

**Forms**
- `.form-group` - Form field container
- `.form-input` - Text input
- `.form-textarea` - Textarea
- `.form-select` - Dropdown
- `.form-checkbox` - Checkbox
- `.form-error` - Error message
- Variants: `required`, `disabled`, `invalid`

**Tables**
- `.table` - Data table
- `.table-header` - Header row
- `.table-row` - Data row
- `.table-cell` - Individual cell
- Variants: `sortable`, `filterable`, `striped`, `compact`

**Alerts**
- `.alert` - Alert box
- `.alert-success` - Success message
- `.alert-warning` - Warning message
- `.alert-error` - Error message
- `.alert-info` - Info message

### CSS Architecture

**Cascade Structure** (most specific → most generic):

```css
/* Override/extend for new components */
/* docs/assets/css/promptly.css */
/* docs/assets/css/issues.css */
/* docs/assets/css/community.css */
/* docs/assets/css/dashboard.css */

/* Shared component patterns */
/* docs/assets/css/components.css (NEW) */

/* Layout and structure */
/* docs/assets/css/layout.css (NEW) */

/* Base styles, variables, typography */
/* docs/assets/css/main.css (EXISTING) */
```

### Dark Mode Support

**Mechanism:** CSS variables + `data-theme="dark"` attribute

```css
:root {
  --color-primary: #1e40af; /* Light theme */
}

@media (prefers-color-scheme: dark) {
  :root[data-theme="dark"] {
    --color-primary: #3b82f6; /* Dark theme */
  }
}
```

**Toggle Implementation:**
```javascript
// In theme.js (existing)
document.documentElement.setAttribute('data-theme',
  currentTheme === 'dark' ? 'dark' : 'light'
);
```

### Responsive Design

**Breakpoints** (mobile-first):
- `0px` - Mobile (default)
- `640px` - Small (sm)
- `768px` - Medium (md)
- `1024px` - Large (lg)
- `1280px` - XL
- `1536px` - 2XL

**Pattern:**
```css
/* Mobile first */
.component {
  padding: var(--space-2);
}

/* Tablet and up */
@media (min-width: 640px) {
  .component {
    padding: var(--space-4);
  }
}

/* Desktop and up */
@media (min-width: 1024px) {
  .component {
    padding: var(--space-6);
  }
}
```

### Accessibility Compliance

**WCAG AAA Standards:**

1. **Color Contrast**
   - Foreground/background ratio ≥7:1 (AAA)
   - Interactive elements ≥4.5:1

2. **Focus States**
   - Visible focus ring (2px, primary color)
   - Tab navigation fully functional
   - Focus order logical

3. **Semantic HTML**
   - Proper heading hierarchy
   - Form labels with inputs
   - Alt text on images
   - ARIA labels where needed

4. **Motion & Animation**
   - Respects `prefers-reduced-motion`
   - No seizure-inducing flashes (>3/sec)
   - Animations <500ms default

### Typography

**Font Stack:**
```css
--font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI",
             Roboto, "Helvetica Neue", Arial, sans-serif;
--font-mono: "SFMono-Regular", Consolas, "Liberation Mono",
             Menlo, monospace;
```

**Sizing Scale (base: 16px):**
```
--text-xs: 0.75rem (12px)
--text-sm: 0.875rem (14px)
--text-base: 1rem (16px)
--text-lg: 1.125rem (18px)
--text-xl: 1.25rem (20px)
--text-2xl: 1.5rem (24px)
--text-3xl: 1.875rem (30px)
--text-4xl: 2.25rem (36px)
```

**Line Heights:**
```
--line-height-tight: 1.2 (headings)
--line-height-normal: 1.6 (body text)
--line-height-relaxed: 1.8 (comments, long content)
```

---

## Data Architecture

### Storage Strategy

**Goals:**
- Zero backend required for MVP
- Optional backend for production
- Offline-first with service worker
- No external dependencies

### 1. Document Data (Read-only)

**Location:** `/docs/assets/data/`

**Files:**
- `search-index.json` (200k+ searchable documents)
- `promptly-api.json` (API reference)
- `issue-templates.json` (Issue templates)
- `site-config.json` (Navigation, metadata)

**Format (search-index.json):**
```json
{
  "docs": [
    {
      "id": "doc-001",
      "title": "Getting Started",
      "url": "/docs/start",
      "content": "How to install HoloLoom...",
      "category": "docs",
      "tags": ["installation", "setup"],
      "updated": "2025-11-16"
    }
  ],
  "promptly": [
    {
      "id": "promptly-001",
      "title": "Refine Loop",
      "url": "/promptly/loop-types/refine",
      "content": "Iterative refinement loop...",
      "category": "promptly",
      "tags": ["loops", "refinement"],
      "updated": "2025-11-16"
    }
  ]
}
```

**Size:** ~2-3 MB (acceptable with gzip)
**Load:** On page load (async, non-blocking)
**Cache:** Service worker + localStorage

### 2. User-Generated Data (localStorage)

**What's Stored:**
- Issues (local cache)
- Forum threads/comments (local cache)
- User preferences (theme, sidebar state)
- Search history
- Bookmarks
- Notifications

**localStorage Schema:**

```javascript
// Issues (cached from server or initial seed)
localStorage.setItem('hololoom:issues', JSON.stringify({
  "123": {
    id: "123",
    title: "Add feature X",
    description: "...",
    status: "open",
    labels: ["feature", "enhancement"],
    author: "alice",
    created: 1699000000,
    updated: 1699100000,
    comments: [
      {
        id: "c1",
        author: "bob",
        content: "Great idea!",
        created: 1699050000
      }
    ]
  }
}));

// Forum threads
localStorage.setItem('hololoom:forum-threads', JSON.stringify({
  "thread-001": {
    id: "thread-001",
    category: "hololoom",
    title: "How to use memory graphs?",
    author: "charlie",
    content: "I'm trying to...",
    created: 1699000000,
    replies: [
      {
        id: "reply-001",
        author: "diana",
        content: "You can use the Graph API...",
        created: 1699050000,
        votes: 5
      }
    ]
  }
}));

// User preferences
localStorage.setItem('hololoom:prefs', JSON.stringify({
  theme: "dark",
  sidebarCollapsed: false,
  fontSize: "base",
  notifications: true,
  followedTags: ["memory", "embeddings"],
  watchedIssues: ["123", "145"]
}));

// Search history (for autocomplete)
localStorage.setItem('hololoom:search-history', JSON.stringify([
  { query: "embedding", timestamp: 1699100000 },
  { query: "orchestrator", timestamp: 1699050000 }
]));
```

**Storage Limits:**
- localStorage: ~5-10 MB per domain
- Plan: ~2 MB for forum threads + issues
- Leaves headroom for growth

**Cache Busting:**
```javascript
// Version check on page load
const CACHE_VERSION = "hololoom-1.0";
const storedVersion = localStorage.getItem('hololoom:version');

if (storedVersion !== CACHE_VERSION) {
  // Clear old data, load new defaults
  localStorage.clear();
  localStorage.setItem('hololoom:version', CACHE_VERSION);
}
```

### 3. Service Worker (Offline-First)

**File:** `/docs/assets/js/offline.js`

**Caches:**
1. **documents-cache** - HTML pages, 1-week TTL
2. **assets-cache** - CSS, JS, images, 1-month TTL
3. **api-cache** - JSON data files, 1-day TTL

**Network Strategy:**
```
- Cache first (for assets): Serve from cache, update background
- Network first (for docs): Try network, fallback to cache
- Network only (for forms): Always require network
```

**Implementation:**
```javascript
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Assets (CSS, JS, images)
  if (/\.(css|js|jpg|png|svg|woff)$/.test(url.pathname)) {
    return event.respondWith(
      caches.match(event.request)
        .then(r => r || fetch(event.request).then(r => {
          caches.open('assets-cache').then(c => c.put(event.request, r.clone()));
          return r;
        }))
    );
  }

  // HTML documents
  if (event.request.headers.get('accept').includes('text/html')) {
    return event.respondWith(
      fetch(event.request)
        .catch(() => caches.match(event.request))
    );
  }
});
```

### 4. Optional Backend (Future)

**For Production Scaling:**
- **Issues:** Firebase Realtime DB or Supabase
- **Forums:** Supabase with PostgreSQL
- **Users:** OAuth + session management
- **Search:** Algolia or Elasticsearch

**Migration Path:**
```
Phase 1: localStorage only (MVP)
Phase 2: Add optional backend API
Phase 3: Full backend with real-time sync
Phase 4: Advanced features (notifications, analytics)
```

**Backend API Spec (Future):**
```
POST /api/issues - Create issue
GET /api/issues - List issues
GET /api/issues/:id - Get issue
POST /api/issues/:id/comments - Add comment

POST /api/community/threads - Create thread
GET /api/community/threads - List threads
GET /api/community/threads/:id - Get thread
POST /api/community/threads/:id/replies - Add reply
POST /api/community/threads/:id/upvote - Upvote thread

GET /api/search?q=query - Full-text search
```

---

## Implementation Phases

### Phase 1: Promptly Documentation Hub (5-8 hours)

**Goal:** Complete Promptly documentation site

**Tasks:**
1. Create `/promptly/` section structure (1 hour)
2. Write Quick Start page (1.5 hours)
   - Installation (pip, Docker, manual)
   - First loop walkthrough
   - Next steps
3. Create Loop Types reference pages (1.5 hours)
   - Template for each loop type
   - Code examples (copy-paste ready)
   - Real-world use cases
4. Write API Reference (1 hour)
   - Auto-generate from docstrings or hand-write
   - Parameter tables
   - Return value documentation
5. Create integration guides (1 hour)
   - HoloLoom integration
   - Claude Desktop MCP
   - VS Code extension
6. Set up examples gallery (1 hour)
   - Grid layout
   - Search by loop type
   - "Try it" button (opens web editor)

**Deliverables:**
- 12 new pages
- 50+ code examples
- Example project starter

### Phase 2: Issue Tracker (8-12 hours)

**Goal:** Functional GitHub-style issue management

**Tasks:**
1. Design issue data schema (1 hour)
   - Fields: id, title, description, status, labels, etc.
   - Comments/discussion thread
2. Create issue list view (2 hours)
   - Filtering (status, labels, milestone)
   - Sorting (newest, popular, commented)
   - Pagination
   - Responsive table/card layout
3. Implement issue detail view (2 hours)
   - Full issue with comments
   - Vote system (upvote/downvote)
   - Related issues sidebar
4. Create issue form + validation (1.5 hours)
   - Template selection (Bug, Feature, Documentation)
   - Rich text editor (markdown)
   - Preview before submit
5. Build roadmap Kanban view (2 hours)
   - 4-column layout (Backlog, In Progress, In Review, Done)
   - Drag-to-update (with localStorage persistence)
   - Milestone filtering
   - Burndown charts (optional)
6. Add GitHub sync layer (1.5 hours)
   - Weekly sync from GitHub issues
   - Comments bidirectional link
7. Unit tests (1 hour)
   - Data validation
   - Filter logic
   - Sort logic

**Deliverables:**
- Issue tracker fully functional
- 150 pre-populated issues
- Roadmap visible
- GitHub sync working

### Phase 3: Community Forums (10-15 hours)

**Goal:** Vibrant discussion community

**Tasks:**
1. Design forum data schema (1 hour)
   - Threads, comments, users, reputation
   - Tags, categories, subscriptions
2. Create category views (2 hours)
   - 5 categories (General, HoloLoom, Promptly, Dev, Show & Tell)
   - Thread list with stats
   - Filter/sort
3. Build thread view (2.5 hours)
   - Original post + replies
   - Vote/upvote system
   - "Accepted answer" marking
   - Reply box with editor
4. Create new thread form (1.5 hours)
   - Category selection
   - Tags
   - Rich text editor
5. Implement user profiles (1.5 hours)
   - Reputation score
   - Badges (Rookie, Expert, etc.)
   - Recent posts
   - Follow button
6. Add reputation system (1.5 hours)
   - Upvoting: +10 points
   - Accepted answer: +15 points
   - Milestones → badges
7. Tag system (1 hour)
   - Tag pages
   - Subscribe to tags
8. Moderation tools (1.5 hours)
   - Lock thread (mod only)
   - Pin thread
   - Move thread
   - Flag inappropriate content
9. Unit tests (1.5 hours)
   - Thread logic
   - Reputation calculations
   - Tag filtering

**Deliverables:**
- Forum fully functional
- 2,000+ seed threads
- Reputation/badge system working
- Moderation tools in place

### Phase 4: BigPlay Dashboard (5-8 hours)

**Goal:** Unified ecosystem discovery

**Tasks:**
1. Create dashboard home (1.5 hours)
   - Hero section
   - Quick links grid
   - Stats display
2. Build getting started wizard (2 hours)
   - Step 1: Role selection
   - Step 2: Learning path
   - Step 3: Resources
   - Step 4: Quick setup
3. Implement activity feed (1 hour)
   - Chronological activity
   - Type filters
   - Real-time updates (localStorage)
4. Create ecosystem overview (1 hour)
   - Stats dashboard
   - Link cards
   - Cross-component navigation
5. Build global search modal (1.5 hours)
   - ⌘K / Ctrl+K trigger
   - Type-ahead suggestions
   - Component filtering
   - Result previews

**Deliverables:**
- Dashboard fully functional
- Wizard polished
- Global search integrated

### Phase 5: Integration & Polish (3-5 hours)

**Goal:** Cohesive ecosystem, performance, accessibility

**Tasks:**
1. Integration testing (1 hour)
   - Cross-component navigation
   - Search accuracy
   - Offline functionality
2. Performance optimization (1.5 hours)
   - Lighthouse audit (aim for 90+)
   - Image optimization
   - CSS/JS minification
   - Service worker caching
3. Accessibility audit (1 hour)
   - WCAG AAA compliance check
   - Screen reader testing
   - Keyboard navigation
   - Color contrast verification
4. Mobile testing (0.5 hours)
   - iOS Safari
   - Android Chrome
   - Touch interactions
5. Documentation (1 hour)
   - Architecture guide
   - Developer setup
   - Contribution guide

**Deliverables:**
- Fully integrated ecosystem
- 95+ Lighthouse score
- WCAG AAA certified
- Production-ready

---

## Technical Stack

### Frontend Framework Philosophy

**Principle:** Zero external dependencies (or strongly justified)

**Current Stack:**
- HTML5 semantic markup
- Vanilla CSS (no preprocessor)
- Vanilla JavaScript (no framework)
- No build step (dev = production)

### Required Libraries (Justified)

1. **Fuse.js** (7 KB gzipped)
   - Client-side fuzzy search
   - 200k documents in <50ms
   - No alternatives without bloat
   - Load: async, non-blocking

2. **Marked.js** (10 KB gzipped) - Optional, for markdown
   - Markdown parsing (forum posts, issues)
   - Syntax highlighting (code blocks)
   - Alternative: hand-coded parser (lower quality)

**Trade-off:** +17 KB for much better UX (justified)

### Optional: Lightweight Frameworks

**For Forums (complex state):**

Option A: Vanilla JS (no framework)
- Pro: Zero dependencies
- Con: Manual state management

Option B: htmx (4 KB gzipped)
- Pro: HTML-first
- Con: Still need server interaction

Option C: Alpine.js (15 KB gzipped)
- Pro: Minimal, reactive
- Con: Added learning curve

**Recommendation:** Start with Vanilla JS, add htmx if needed

### Build & Deployment

**Development:**
```bash
# Run local server (Python or Node)
python -m http.server 8000
# Or: npx http-server .

# Watch for changes (optional)
# Modify CSS/JS in editor, refresh browser
```

**No Build Step Needed:**
- CSS files served as-is
- JS files served as-is (no bundling)
- HTML templates server-rendered (no JSX)

**Deployment:**
```bash
# Build for production
./scripts/build.sh  # Minifies CSS/JS, optimizes images

# Deploy to host
git push origin main  # Netlify/Vercel auto-deploy
# Or: scp -r docs/* user@host:/var/www/
```

### Service Worker (Offline)

**File:** `/docs/assets/js/offline.js`

**Functionality:**
- Cache HTML/CSS/JS
- Serve from cache when offline
- Periodic updates
- Cache versioning

**No Framework Needed:** Native Service Worker API

### Search Implementation

**Using Fuse.js:**
```javascript
import Fuse from 'https://cdn.jsdelivr.net/npm/fuse.js/dist/fuse.esm.js';

// Load search index (200k documents)
const index = await fetch('/assets/data/search-index.json')
  .then(r => r.json());

// Create fuzzy search
const fuse = new Fuse(index.docs, {
  keys: ['title', 'content'],
  threshold: 0.3,
  includeScore: true
});

// Search
const results = fuse.search('embedding');
console.log(results); // Top results with scores
```

### Markdown Rendering

**Using Marked.js:**
```javascript
import { marked } from 'https://cdn.jsdelivr.net/npm/marked/+esm';

// Render markdown to HTML
const html = marked.parse('# Hello\n\nThis is **bold** text');

// With custom renderer (syntax highlighting)
marked.setOptions({
  renderer: {
    code(code, language) {
      return `<pre><code class="language-${language}">${code}</code></pre>`;
    }
  }
});
```

### Progressive Enhancement

**Design Principle:** Works without JavaScript, enhanced with JavaScript

**Example:**
```html
<!-- Works without JS (regular links) -->
<a href="/issues/?status=open">Open Issues</a>

<!-- Enhanced with JS (client-side filtering) -->
<script>
  // If JS enabled, intercept and fetch dynamically
  document.querySelectorAll('[data-filter]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      // Client-side filtering from localStorage
    });
  });
</script>
```

### Data Compression

**Optimizations:**
- Gzip: CSS/JS/JSON (3-5x compression)
- Brotli: Alternative (10% smaller than gzip)
- Image optimization: WebP with JPEG fallback
- Lazy loading: `loading="lazy"` on images

**File Sizes (Target):**
- Main CSS: <30 KB (gzipped)
- Main JS: <50 KB (gzipped)
- Search index: <200 KB (gzipped)
- Total per page: <100 KB (gzipped)

---

## User Flows

### Flow 1: New User Onboarding

**Scenario:** First-time visitor to hololoom.dev

**Steps:**
```
1. Visitor arrives at / (root)
   ↓ Redirects to /dashboard/

2. BigPlay Dashboard appears
   - Hero: "HoloLoom & Promptly Ecosystem"
   - 3 buttons: [Getting Started] [Explore Docs] [Join Community]

3. Clicks "Getting Started"
   ↓ Wizard opens

4. Wizard Step 1: "What's your goal?"
   - [ ] Learn HoloLoom
   - [ ] Use Promptly
   - [ ] Contribute to project
   - [ ] Explore research

5. Wizard Step 2: "Personalized Path"
   If "Learn HoloLoom" selected:
   - Video: "HoloLoom in 5 minutes"
   - Link: /docs/start
   - Suggested: Training parts 1-3

   If "Use Promptly" selected:
   - Video: "Promptly quick start"
   - Link: /promptly/quick-start
   - Suggested: Loop types 101

6. Wizard Step 3: "Get Started"
   - Copy-paste installation command
   - IDE setup (VS Code extension)

7. Wizard Complete
   - Opens /docs/start OR /promptly/quick-start
   - Dismissible banner: "Need help? → /community"
   - Bookmarks section on dashboard
```

**Exit Points:**
- "Skip wizard" → Dashboard overview
- "Go to docs" → Direct to selected docs
- "Explore community" → /community/ categories

---

### Flow 2: Asking a Question

**Scenario:** User has question about embeddings

**Steps:**
```
1. User opens hololoom.dev/community
   - Sees 5 categories
   - "General Discussion" has 247 threads

2. Clicks "General Discussion"
   - Shows recent threads
   - Search box visible

3. Searches: "embeddings"
   - Shows 4 existing threads with answers
   - Reads thread #2, finds answer

4. If question not answered:
   - Clicks "Start New Discussion"

5. New Discussion Form:
   - Category: General Discussion (pre-selected)
   - Title: "How to use Matryoshka embeddings for semantic search?"
   - Description: Rich text editor (markdown)
   - Tags: [embeddings] [semantic] [hololoom]
   - Preview: Shows formatted post

6. Submits
   - Thread created, ID = #thread-2342
   - URL: /community/threads/thread-2342
   - Notification sent to [embeddings] followers

7. Community Responds
   - Alice replies with code example (+2 votes)
   - Bob replies with tip (+5 votes)
   - User marks Alice's as "Accepted Answer"
   - Alice gets +15 reputation points

8. User Bookmarks
   - Clicks bookmark icon
   - Thread appears in /dashboard/bookmarks
```

---

### Flow 3: Reporting a Bug

**Scenario:** User finds bug in Promptly CLI

**Steps:**
```
1. User navigates to /issues/
   - Sees "127 open issues"
   - Clicks "Create Issue" button

2. New Issue Form:
   - Template selection: [Bug] [Feature] [Documentation]
   - Clicks "Bug"

3. Bug Template Pre-filled:
   ```
   ## Description
   [User enters: "Refine loop exits early"]

   ## Steps to Reproduce
   1. Run: `promptly loop --type refine`
   2. Enter prompt: "Optimize this code..."
   3. After 3 iterations, loop stops

   ## Expected Behavior
   Should run 5 iterations (max_iterations=5)

   ## Actual Behavior
   Exits after 3 iterations

   ## Environment
   - Promptly: 1.0.2
   - Python: 3.11
   - OS: macOS 14.1
   ```

4. User previews markdown
   - Looks good, submits

5. Issue Created
   - ID: #287
   - Status: Open
   - Labels: [bug] [promptly]
   - Assigned: Unassigned

6. Issue Appears in:
   - /issues/287 (full view)
   - /issues/?label=bug (filtered list)
   - /issues/roadmap (Backlog column)
   - Developer receives notification

7. Developer Comments
   - "Investigating... found root cause in recursive_loops.py"
   - Changes status to "In Progress"
   - Moves to "In Progress" column in roadmap

8. Developer Creates PR
   - Links PR #450 to issue #287
   - Updates issue: "PR #450 fixes this"
   - Label added: [fixed]
   - Status: Closed (automatic)
```

---

### Flow 4: Cross-Component Search

**Scenario:** User wants to find everything about "Thompson Sampling"

**Steps:**
```
1. User presses ⌘K (Mac) or Ctrl+K (Windows/Linux)
   - Global search modal opens

2. Modal appears:
   ```
   ┌─────────────────────────────────────────────┐
   │ Search HoloLoom & Promptly                  │
   │ ┌──────────────────────────────────────────┐│
   │ │ thompson sampling        │ ⌘K to close   ││
   │ └──────────────────────────────────────────┘│
   │                                             │
   │ Docs                          5 results    │
   │ ├─ Thompson Sampling 101                   │
   │ │  /docs/api/policy                        │
   │ ├─ Bandit Strategies                       │
   │ │  /docs/guides/advanced                   │
   │ ├─ Thompson vs Epsilon-Greedy              │
   │ │  /training/part4                         │
   │ ├─ [... show 2 more]                      │
   │                                             │
   │ Issues                        2 results    │
   │ ├─ Add Thompson Sampling to policy         │
   │ │  #45 · Open · 3 comments                 │
   │ ├─ [... show 1 more]                      │
   │                                             │
   │ Promptly Examples            1 result     │
   │ ├─ Thompson Sampling loop selector         │
   │ │  /promptly/examples                      │
   │                                             │
   │ Community                     3 results    │
   │ ├─ How to tune Thompson epsilon?           │
   │ │  #thread-234 · 4 replies · helpful       │
   │ ├─ [... show 2 more]                      │
   │                                             │
   └─────────────────────────────────────────────┘
   ```

3. User clicks on "Thompson Sampling 101"
   - Navigates to /docs/api/policy
   - Modal closes
   - Jumps to "Thompson Sampling" section

4. User presses ⌘K again
   - Clicks "See all docs results" (11 total)
   - Shows full results page
   - URL: /search?q=thompson%20sampling&filter=docs

5. Back on search results page
   - Can refine: "Filter by: [Docs] [Issues] [Promptly] [Community]"
   - Can sort: "Sort by: [Relevance] [Newest] [Popular]"
```

---

### Flow 5: Contributing Documentation

**Scenario:** User wants to add example for Promptly

**Steps:**
```
1. User browses /promptly/examples
   - Sees gallery of 30+ examples
   - Clicks "Submit your example" link

2. Goes to /community/show-tell
   - New discussion form opens
   - Category: "Show & Tell"
   - Title: "Prompt Optimization Loop with REFINE"

3. User writes:
   ```
   # My Prompt Optimization Loop

   I built a loop that optimizes prompts for code generation:

   [Code block]
   loop = RecursiveLoop(
     loop_type=LoopType.REFINE,
     max_iterations=5,
     quality_threshold=0.95
   )

   ## Results
   - Improved output quality from 0.72 → 0.91
   - Reduced token usage by 30%

   ## Discussion
   Would love feedback on my approach!
   ```

4. Submits discussion
   - Shows in /community/show-tell
   - Tagged: [promptly] [optimization] [refine]

5. Community responds
   - 5 upvotes, 3 comments
   - "This is excellent! Would you submit as PR to docs?"

6. User submits PR to GitHub
   - Adds example to /promptly/examples/
   - References: /community/threads/thread-5423
   - Merged by maintainer

7. Example appears in:
   - /promptly/examples gallery
   - /promptly/examples/gallery.html
   - Global search
   - Community thread still linked
```

---

### Flow 6: Moderator Actions (Forums)

**Scenario:** Moderator managing community

**Steps:**
```
1. Moderator logs in
   - Uses GitHub OAuth (optional backend feature)
   - Badge shows: [Moderator]

2. Sees suspicious thread
   - Spam post about crypto
   - Reports by 5 users

3. Clicks "..." menu on thread
   - Options: [Lock] [Pin] [Move] [Hide] [Delete]

4. Clicks "Lock"
   - Thread locked: no new replies
   - Lock icon visible
   - Existing replies still visible

5. Moves to "Spam" category (private)
   - Users can't see it
   - Logged for review

6. Sends mod message:
   - "This thread violates code of conduct #2.3"
   - Link to guidelines
   - "Appeal via: moderation@hololoom.dev"

7. Back to /community
   - Thread no longer visible
   - Moderator logs action
   - Monthly report generated
```

---

## Integration Details

### Navigation Integration

**Updated Navigation Structure:**

```html
<!-- Main navbar (nav.js updates) -->
<nav class="navbar">
  <div class="navbar-brand">
    <a href="/dashboard/">HoloLoom</a>
  </div>

  <div class="navbar-search">
    <button class="search-trigger" data-shortcut="⌘K">
      Search...
    </button>
  </div>

  <div class="navbar-menu">
    <a href="/docs/">Documentation</a>
    <a href="/promptly/">Promptly</a>
    <a href="/issues/">Issues</a>
    <a href="/community/">Community</a>
    <a href="/dashboard/">Dashboard</a>
  </div>
</nav>

<!-- Sidebar (collapsible by component) -->
<aside class="sidebar">
  <!-- Docs section -->
  <div class="sidebar-section expanded">
    <button class="sidebar-toggle">Documentation</button>
    <ul>
      <li><a href="/docs/">Home</a></li>
      <li><a href="/docs/start">Getting Started</a></li>
      <li class="section-group">
        <button class="sidebar-toggle">Training</button>
        <ul>
          <li><a href="/docs/training/part1">Part 1</a></li>
          <!-- ... -->
        </ul>
      </li>
    </ul>
  </div>

  <!-- Promptly section -->
  <div class="sidebar-section">
    <button class="sidebar-toggle">Promptly</button>
    <ul>
      <li><a href="/promptly/">Home</a></li>
      <li><a href="/promptly/quick-start">Quick Start</a></li>
      <!-- ... -->
    </ul>
  </div>

  <!-- Issues section -->
  <div class="sidebar-section">
    <button class="sidebar-toggle">Issues</button>
    <ul>
      <li><a href="/issues/">All Issues</a></li>
      <li><a href="/issues/roadmap">Roadmap</a></li>
      <!-- ... -->
    </ul>
  </div>

  <!-- Community section -->
  <div class="sidebar-section">
    <button class="sidebar-toggle">Community</button>
    <ul>
      <li><a href="/community/">Categories</a></li>
      <li><a href="/community/general">General</a></li>
      <li><a href="/community/hololoom">HoloLoom</a></li>
      <!-- ... -->
    </ul>
  </div>
</aside>
```

### Search Integration

**Global Search Keyboard Shortcut:**

```javascript
// In nav.js, updated keyboard handler
function handleKeyboardShortcuts(e) {
  if (e.key === '/') {
    e.preventDefault();
    openGlobalSearch();
  }
}

function openGlobalSearch() {
  const modal = document.querySelector('.search-modal');
  modal.classList.add('visible');
  modal.querySelector('input').focus();

  // Trigger unified search
  window.HoloLoomSearch.activate();
}
```

### Local Storage Integration

**Persistent User Data:**

```javascript
// Class for managing user data across components
class HoloLoomStorage {
  constructor() {
    this.prefix = 'hololoom:';
  }

  // Issues
  getIssues() {
    return JSON.parse(localStorage.getItem(this.prefix + 'issues') || '{}');
  }

  setIssues(issues) {
    localStorage.setItem(this.prefix + 'issues', JSON.stringify(issues));
  }

  // Forum threads
  getThreads() {
    return JSON.parse(localStorage.getItem(this.prefix + 'threads') || '{}');
  }

  setThreads(threads) {
    localStorage.setItem(this.prefix + 'threads', JSON.stringify(threads));
  }

  // User preferences
  getPrefs() {
    return JSON.parse(localStorage.getItem(this.prefix + 'prefs') || '{}');
  }

  setPrefs(prefs) {
    localStorage.setItem(this.prefix + 'prefs', JSON.stringify(prefs));
  }

  // Syncing
  async syncFromServer() {
    // Optional: fetch latest data from backend
  }
}

// Export global instance
window.HoloLoomStorage = new HoloLoomStorage();
```

### Breadcrumb Generation

**Updated nav.js breadcrumb generation:**

```javascript
function generateBreadcrumbs() {
  const breadcrumbContainer = document.querySelector('.breadcrumbs');
  if (!breadcrumbContainer) return;

  const path = window.location.pathname;

  // Define breadcrumb hierarchy
  const pathMap = {
    '/': 'Home',
    '/dashboard/': 'Dashboard',
    '/docs/': 'Documentation',
    '/docs/start': 'Getting Started',
    '/promptly/': 'Promptly',
    '/promptly/quick-start': 'Quick Start',
    '/promptly/loop-types/': 'Loop Types',
    '/issues/': 'Issues',
    '/community/': 'Community',
    '/search': 'Search'
  };

  const breadcrumbs = [];
  breadcrumbs.push({ label: 'Home', url: '/' });

  // Build breadcrumbs from path
  let currentPath = '';
  path.split('/').filter(Boolean).forEach((segment, i) => {
    currentPath += '/' + segment;
    const label = pathMap[currentPath] ||
                  segment.charAt(0).toUpperCase() + segment.slice(1);
    breadcrumbs.push({ label, url: currentPath });
  });

  // Render breadcrumbs
  const html = breadcrumbs.map((crumb, i) => {
    const isLast = i === breadcrumbs.length - 1;
    if (isLast) {
      return `<span aria-current="page">${escapeHtml(crumb.label)}</span>`;
    }
    return `<a href="${escapeHtml(crumb.url)}">${escapeHtml(crumb.label)}</a>`;
  }).join('<span class="separator"> / </span>');

  breadcrumbContainer.innerHTML = html;
}
```

---

## Appendices

### A. File Structure Reference

```
docs/
├── index.html → /dashboard/ (BigPlay home)
├── dashboard/
│   ├── index.html (dashboard home)
│   ├── wizard.html (onboarding)
│   └── activity.html (activity feed)
├── docs/
│   ├── index.html (docs home)
│   ├── start.html (getting started)
│   ├── training/ (5 parts)
│   ├── api/ (core, memory, orchestrator, rag)
│   └── guides/
├── promptly/
│   ├── index.html (promptly home)
│   ├── quick-start.html
│   ├── loop-types/ (6 pages)
│   ├── api/ (core, cli, web, mcp, hololoom)
│   ├── guides/ (7 pages)
│   ├── examples/
│   ├── videos/
│   └── faq.html
├── issues/
│   ├── index.html (list)
│   ├── new.html (create)
│   ├── roadmap.html (kanban)
│   └── [id]/ (detail pages)
├── community/
│   ├── index.html (categories)
│   ├── guidelines.html
│   ├── new.html (create thread)
│   ├── [category]/ (category views)
│   ├── threads/ ([id]/ detail pages)
│   ├── tags/ ([tag]/ views)
│   ├── users/ ([username]/ profiles)
│   └── notifications.html
└── assets/
    ├── css/
    │   ├── main.css (core system)
    │   ├── promptly.css (promptly-specific)
    │   ├── issues.css (issue tracker)
    │   ├── community.css (forums)
    │   ├── dashboard.css (bigplay)
    │   ├── components.css (reusable components)
    │   └── print.css (print styles)
    ├── js/
    │   ├── nav.js (navigation, updated)
    │   ├── search.js (search implementation)
    │   ├── theme.js (dark mode)
    │   ├── promptly.js (promptly features)
    │   ├── issues.js (issue tracker)
    │   ├── community.js (forums)
    │   ├── dashboard.js (bigplay)
    │   ├── unified-search.js (global search)
    │   ├── storage.js (localStorage management)
    │   └── offline.js (service worker)
    └── data/
        ├── search-index.json (200k docs)
        ├── issues.json (150 issues, seed)
        ├── forum-threads.json (2.3k threads, seed)
        ├── promptly-api.json (API reference)
        └── site-config.json (navigation metadata)
```

### B. GitHub Sync Specification

**One-way Sync (GitHub → Site) - Weekly**

```python
# Pseudo-code for sync service
def sync_github_issues():
    """Fetch issues from GitHub API every week"""
    issues = github.get_issues(repo='hololoom', state='all')

    for issue in issues:
        stored_issue = {
            'id': issue.number,
            'title': issue.title,
            'description': issue.body,
            'status': 'open' if issue.state == 'open' else 'closed',
            'labels': [l.name for l in issue.labels],
            'author': issue.user.login,
            'created': issue.created_at.timestamp(),
            'updated': issue.updated_at.timestamp(),
            'url': issue.html_url,
            'comments': [
                {
                    'author': c.user.login,
                    'content': c.body,
                    'created': c.created_at.timestamp()
                }
                for c in issue.comments
            ]
        }

        localStorage['hololoom:issues'][issue.number] = stored_issue
```

**Two-way Comments (Optional)**

```
Site comments → GitHub comments (via bot account)
GitHub comments → Site (automatic sync)
```

### C. Performance Checklist

**Target Metrics:**
- [ ] Lighthouse score >95 (all pages)
- [ ] <1s page load time (measured at p75)
- [ ] <50ms search response (200k docs)
- [ ] <100kb assets per page (gzipped)
- [ ] 90%+ cache hit rate (service worker)

**Testing:**
```bash
# Lighthouse
npm install -g lighthouse
lighthouse https://hololoom.dev --view

# WebPageTest
# https://www.webpagetest.org

# Local testing
# Chrome DevTools → Lighthouse
# Firefox → Developer Tools → Performance
```

### D. Accessibility Checklist

**WCAG AAA Compliance:**
- [ ] Color contrast ≥7:1
- [ ] Focus ring visible (all interactive elements)
- [ ] Keyboard navigation fully functional
- [ ] Screen reader compatible
- [ ] Alt text on all images
- [ ] Proper heading hierarchy (h1 → h2 → h3)
- [ ] Form labels associated
- [ ] Motion respects `prefers-reduced-motion`

**Testing:**
```bash
# axe DevTools browser extension
# WAVE browser extension
# Screen reader: NVDA (Windows), JAWS, VoiceOver (Mac)
# Keyboard: Tab through page, verify all functionality
```

### E. Security Considerations

**Data Privacy:**
- All data in localStorage (no tracking)
- No analytics cookies
- No third-party services (except optional backend)
- User data never leaves browser (unless backend enabled)

**Input Validation:**
- Sanitize markdown input
- Escape HTML special characters
- Validate issue/thread fields
- CSRF tokens (if backend added)

**Code Security:**
```javascript
// Good: Escape HTML
const escaped = escapeHtml(userInput);
element.textContent = escaped; // Safe

// Bad: Don't do this
element.innerHTML = userInput; // XSS vulnerability
```

### F. Deployment Checklist

**Pre-Launch:**
- [ ] Lighthouse audit (95+)
- [ ] Accessibility audit (WCAG AAA)
- [ ] Security scan (no vulnerabilities)
- [ ] Mobile testing (iOS + Android)
- [ ] Cross-browser testing (Chrome, Firefox, Safari)
- [ ] Broken link check
- [ ] Performance benchmarks
- [ ] SEO optimization (meta tags, structure)

**Launch:**
- [ ] Domain configured
- [ ] HTTPS enabled
- [ ] Redirects in place (old URLs → new)
- [ ] Analytics configured (optional)
- [ ] Monitoring enabled
- [ ] Backups configured
- [ ] Team notified

**Post-Launch:**
- [ ] Monitor error logs
- [ ] Gather user feedback
- [ ] Performance monitoring
- [ ] Update documentation

---

## Summary

This architecture provides a comprehensive roadmap for expanding the HoloLoom flagship site into a unified ecosystem hub combining:

1. **Promptly Documentation** (complete reference)
2. **Issue Tracker** (GitHub-style management)
3. **Community Forums** (peer-to-peer learning)
4. **BigPlay Dashboard** (unified discovery)

**Key Principles:**
- Zero external dependencies (or strongly justified)
- Mobile-first responsive design
- WCAG AAA accessibility
- <1s page load times
- Offline-first (service worker)
- Progressive enhancement
- Optional backend scaling

**Timeline:** 4-6 weeks (5 implementation waves)

**Team:** 1-2 developers (with design system leverage)

**Result:** A thriving ecosystem where users learn, contribute, and build together.

---

**Document Maintenance:**
- Review monthly for accuracy
- Update based on implementation learnings
- Expand detailed specifications as needed
- Link to updated technical documents

**Version History:**
- v1.0 (Nov 16, 2025) - Initial architecture
