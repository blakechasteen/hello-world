# Future SpinningWheel Adapters Roadmap

> **Created**: 2025-12-23
> **Status**: Planning Document
> **Philosophy**: "If you need to configure it, we failed" - Universal data ingestion

This document catalogs potential future SpinningWheel adapters for HoloLoom. Each spinner follows the BaseSpinner protocol and aims to provide zero-configuration data ingestion.

---

## Table of Contents

1. [Knowledge Management](#1-knowledge-management)
2. [Productivity & Task Management](#2-productivity--task-management)
3. [Communication Platforms](#3-communication-platforms)
4. [Financial & Commerce](#4-financial--commerce)
5. [Health & Fitness](#5-health--fitness)
6. [Design & Creative Tools](#6-design--creative-tools)
7. [Location & Geospatial](#7-location--geospatial)
8. [DevOps & Infrastructure](#8-devops--infrastructure)
9. [Creative Writing & Media](#9-creative-writing--media)
10. [Analytics & Monitoring](#10-analytics--monitoring)
11. [Research & Academia](#11-research--academia)
12. [Social Media](#12-social-media)
13. [Gaming & Entertainment](#13-gaming--entertainment)
14. [Legal & Compliance](#14-legal--compliance)
15. [Priority Matrix](#priority-matrix)
16. [Implementation Guidelines](#implementation-guidelines)

---

## 1. Knowledge Management

Personal knowledge management tools and second-brain applications.

### NotionSpinner
**Purpose**: Ingest Notion workspace exports (pages, databases, blocks)
**Input Formats**: `.zip` (Notion export), JSON API responses
**Complexity**: Medium
**Dependencies**: `notion-client` (optional, for API access)
**Value**: High - Notion is widely used for personal/team knowledge bases
**Key Features**:
- Parse Notion's block structure (paragraphs, headers, lists, toggles)
- Handle database entries with properties
- Preserve page hierarchy and links
- Extract embedded content (images, files, embeds)

**Importance Signals**:
- Page depth in hierarchy
- Last edited timestamp
- Backlink count
- Database property richness

---

### ObsidianSpinner
**Purpose**: Ingest Obsidian vault exports (markdown with wikilinks)
**Input Formats**: `.md` files, `.obsidian` config, entire vault directories
**Complexity**: Medium
**Dependencies**: None (pure markdown parsing)
**Value**: High - Popular among researchers, developers, PKM enthusiasts
**Key Features**:
- Parse wikilinks (`[[page]]`, `[[page|alias]]`)
- Handle frontmatter YAML metadata
- Process Dataview queries (extract, don't execute)
- Preserve folder structure as hierarchy
- Extract tags (`#tag`, nested `#tag/subtag`)

**Importance Signals**:
- Backlink count (most-linked notes)
- Tag frequency
- Daily note vs evergreen content
- Frontmatter metadata richness

---

### ConfluenceSpinner
**Purpose**: Ingest Atlassian Confluence spaces and pages
**Input Formats**: HTML exports, XML exports, REST API JSON
**Complexity**: Medium-High
**Dependencies**: `atlassian-python-api` (optional)
**Value**: High - Enterprise documentation standard
**Key Features**:
- Parse Confluence storage format (XHTML-based)
- Handle macros (code blocks, panels, expand)
- Preserve space/page hierarchy
- Extract attachments metadata
- Handle page templates

**Importance Signals**:
- View count
- Comment count
- Last modified by (authority)
- Space importance
- Child page count

---

### RoamSpinner
**Purpose**: Ingest Roam Research graph exports
**Input Formats**: JSON export, EDN format
**Complexity**: Medium
**Dependencies**: None
**Value**: Medium - Niche but influential in PKM community
**Key Features**:
- Parse block references (`((block-uid))`)
- Handle page references (`[[page]]`)
- Preserve daily notes structure
- Extract queries (don't execute)
- Parse attributes (`attribute:: value`)

**Importance Signals**:
- Block reference count
- Page reference count
- Daily note proximity
- Attribute richness

---

### AnkiSpinner
**Purpose**: Ingest Anki flashcard decks
**Input Formats**: `.apkg` (Anki package), `.txt` (tab-separated)
**Complexity**: Low-Medium
**Dependencies**: `genanki` or direct SQLite parsing
**Value**: Medium - Popular for spaced repetition learning
**Key Features**:
- Extract card front/back content
- Parse cloze deletions
- Handle media (images, audio)
- Preserve deck hierarchy
- Extract review statistics

**Importance Signals**:
- Review difficulty (ease factor)
- Lapse count (frequently forgotten = important)
- Interval length (well-learned content)
- Creation date

---

### LogseqSpinner
**Purpose**: Ingest Logseq graph exports
**Input Formats**: `.md` files with Logseq conventions, JSON exports
**Complexity**: Medium
**Dependencies**: None
**Value**: Medium - Growing PKM tool, open-source
**Key Features**:
- Parse block-based structure
- Handle page/block references
- Extract properties
- Preserve journal structure
- Parse queries (don't execute)

**Importance Signals**:
- Reference count
- Property richness
- Journal vs page content
- Block depth

---

## 2. Productivity & Task Management

Task tracking, project management, and productivity tools.

### CalendarSpinner
**Purpose**: Ingest calendar events from various formats
**Input Formats**: `.ics` (iCal), Google Calendar API, Outlook API
**Complexity**: Low-Medium
**Dependencies**: `icalendar`, `google-api-python-client` (optional)
**Value**: High - Universal need for time-based context
**Key Features**:
- Parse recurring events
- Handle timezones correctly
- Extract attendees and organizers
- Preserve event descriptions and attachments
- Handle all-day vs timed events

**Importance Signals**:
- Attendee count
- Recurring frequency
- Duration
- Response status (accepted = higher importance)
- Calendar source (work > personal typically)

---

### TodoistSpinner
**Purpose**: Ingest Todoist tasks and projects
**Input Formats**: JSON API, CSV exports
**Complexity**: Low
**Dependencies**: `todoist-api-python` (optional)
**Value**: Medium-High - Popular task manager
**Key Features**:
- Parse task hierarchy (projects, sections, subtasks)
- Handle labels and filters
- Extract due dates and priorities
- Preserve comments and attachments
- Handle recurring tasks

**Importance Signals**:
- Priority level (p1-p4)
- Due date proximity
- Project depth
- Comment count
- Recurring frequency

---

### LinearSpinner
**Purpose**: Ingest Linear issues and projects
**Input Formats**: GraphQL API responses, CSV exports
**Complexity**: Medium
**Dependencies**: `gql` (optional)
**Value**: Medium-High - Popular among tech teams
**Key Features**:
- Parse issue relationships (blocks, blocked by, duplicates)
- Handle cycles and projects
- Extract estimates and actual time
- Preserve comments and activity
- Handle custom fields

**Importance Signals**:
- Priority/urgency
- Cycle assignment
- Blocker status
- Comment activity
- Estimate size

---

### AsanaSpinner
**Purpose**: Ingest Asana tasks, projects, and portfolios
**Input Formats**: JSON API, CSV exports
**Complexity**: Medium
**Dependencies**: `asana` (official SDK, optional)
**Value**: Medium - Enterprise project management
**Key Features**:
- Parse task dependencies
- Handle sections and columns
- Extract custom fields
- Preserve attachments and comments
- Handle portfolios and goals

**Importance Signals**:
- Due date
- Assignee authority
- Project importance
- Dependency chain position
- Custom field values

---

### GitHubIssuesSpinner
**Purpose**: Ingest GitHub Issues and Discussions (extends existing GitSpinner)
**Input Formats**: REST API, GraphQL API
**Complexity**: Low-Medium
**Dependencies**: `PyGithub` or `gql` (optional)
**Value**: High - Universal developer tool
**Key Features**:
- Parse issue bodies and comments
- Handle labels and milestones
- Extract reactions and participants
- Preserve issue templates
- Handle linked PRs

**Importance Signals**:
- Reaction count
- Comment count
- Label importance (bug > enhancement)
- Milestone assignment
- Author authority (maintainer > first-time contributor)

---

### JiraSpinner
**Purpose**: Ingest Jira issues, epics, and sprints
**Input Formats**: REST API, XML exports
**Complexity**: Medium-High
**Dependencies**: `jira` (official SDK, optional)
**Value**: High - Enterprise standard
**Key Features**:
- Parse issue types (epic, story, task, bug)
- Handle custom workflows
- Extract sprint information
- Preserve comments and attachments
- Handle linked issues

**Importance Signals**:
- Priority level
- Story points
- Sprint assignment
- Epic importance
- Blocker status

---

### TrelloSpinner
**Purpose**: Ingest Trello boards, lists, and cards
**Input Formats**: JSON export, REST API
**Complexity**: Low
**Dependencies**: `py-trello` (optional)
**Value**: Medium - Popular for simple project tracking
**Key Features**:
- Parse card descriptions and checklists
- Handle labels and due dates
- Extract attachments and comments
- Preserve board structure
- Handle card actions (activity log)

**Importance Signals**:
- List position (Done < In Progress < To Do typically)
- Label priority
- Due date proximity
- Checklist completion
- Comment activity

---

## 3. Communication Platforms

Messaging, chat, and communication tools.

### TelegramSpinner
**Purpose**: Ingest Telegram chat exports
**Input Formats**: JSON export (from Telegram Desktop)
**Complexity**: Medium
**Dependencies**: None (or `telethon` for live API)
**Value**: Medium - Popular messaging platform
**Key Features**:
- Parse message types (text, media, stickers, polls)
- Handle forwarded messages
- Extract reply chains
- Preserve channel/group metadata
- Handle bots and commands

**Importance Signals**:
- Forward count
- Reply depth
- Reaction count
- Message length
- Media presence

---

### SignalSpinner
**Purpose**: Ingest Signal message exports
**Input Formats**: Signal Desktop backup (encrypted SQLite)
**Complexity**: High
**Dependencies**: `signal-backup-decode` or similar
**Value**: Medium - Privacy-focused users
**Key Features**:
- Decrypt backup with passphrase
- Parse message types
- Handle attachments
- Preserve group information
- Extract reactions

**Importance Signals**:
- Conversation frequency
- Message length
- Reaction count
- Group vs individual

---

### iMessageSpinner
**Purpose**: Ingest iMessage/SMS from macOS/iOS
**Input Formats**: `chat.db` SQLite database
**Complexity**: Medium
**Dependencies**: None (SQLite)
**Value**: Medium-High - iOS users
**Key Features**:
- Parse message types (text, tapback, effect)
- Handle group chats
- Extract attachments metadata
- Preserve thread structure
- Handle read receipts

**Importance Signals**:
- Tapback/reaction count
- Thread length
- Reply latency patterns
- Attachment presence

---

### ZoomSpinner
**Purpose**: Ingest Zoom meeting transcripts and recordings
**Input Formats**: VTT transcripts, chat logs, recording metadata
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: High - Universal video conferencing
**Key Features**:
- Parse speaker-attributed transcripts
- Handle chat messages with timestamps
- Extract participant lists
- Preserve meeting metadata
- Handle recording chapters

**Importance Signals**:
- Meeting duration
- Participant count
- Speaker time distribution
- Chat activity
- Recording availability

---

### TeamsSpinner
**Purpose**: Ingest Microsoft Teams messages and meetings
**Input Formats**: Graph API, export files
**Complexity**: Medium-High
**Dependencies**: `msal` for auth (optional)
**Value**: High - Enterprise standard
**Key Features**:
- Parse channel messages
- Handle 1:1 and group chats
- Extract meeting transcripts
- Preserve reactions and replies
- Handle files and tabs

**Importance Signals**:
- Channel importance
- Reaction count
- Reply thread depth
- Mention presence (@mentions)
- Meeting attendance

---

### WhatsAppSpinner
**Purpose**: Ingest WhatsApp chat exports
**Input Formats**: `.txt` export, `.zip` with media
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: High - Most popular messaging app globally
**Key Features**:
- Parse timestamp formats (varies by locale)
- Handle media references
- Extract group membership
- Preserve reply quotes
- Handle status messages (calls, member changes)

**Importance Signals**:
- Message frequency
- Media count
- Group size
- Reply presence
- Link sharing

---

## 4. Financial & Commerce

Banking, invoicing, and financial tools.

### BankStatementSpinner
**Purpose**: Ingest bank statements and transactions
**Input Formats**: PDF statements, CSV exports, OFX/QFX
**Complexity**: Medium-High
**Dependencies**: `pdfplumber`, `ofxparse`
**Value**: High - Personal finance tracking
**Key Features**:
- Parse transaction tables from PDFs
- Handle various bank formats
- Extract merchant names
- Categorize transactions
- Preserve account metadata

**Importance Signals**:
- Transaction amount
- Recurring patterns
- Merchant category
- Account type

---

### InvoiceSpinner
**Purpose**: Ingest invoices and receipts (structured)
**Input Formats**: PDF, JSON (from invoice APIs), XML (UBL)
**Complexity**: Medium
**Dependencies**: `pdfplumber`, XML parser
**Value**: Medium-High - Business expense tracking
**Key Features**:
- Extract line items
- Parse tax information
- Handle multiple currencies
- Preserve vendor information
- Extract payment terms

**Importance Signals**:
- Invoice amount
- Due date proximity
- Vendor relationship
- Line item count

---

### CryptoWalletSpinner
**Purpose**: Ingest cryptocurrency wallet transactions
**Input Formats**: Etherscan CSV, blockchain API responses
**Complexity**: Medium
**Dependencies**: `web3` (optional)
**Value**: Medium - Crypto users
**Key Features**:
- Parse transaction history
- Handle multiple chains (ETH, BTC, etc.)
- Extract token transfers
- Preserve gas/fee information
- Handle NFT transactions

**Importance Signals**:
- Transaction value
- Gas price (network congestion indicator)
- Token type
- Contract interaction type

---

### MintSpinner (or similar aggregator)
**Purpose**: Ingest financial aggregator exports
**Input Formats**: CSV exports
**Complexity**: Low
**Dependencies**: None
**Value**: Medium - Personal finance overview
**Key Features**:
- Parse categorized transactions
- Handle multiple accounts
- Extract budget information
- Preserve trends data
- Handle investment accounts

**Importance Signals**:
- Category importance
- Budget adherence
- Account type
- Trend direction

---

### QuickBooksSpinner
**Purpose**: Ingest QuickBooks accounting data
**Input Formats**: QBO files, API exports
**Complexity**: Medium-High
**Dependencies**: `intuit-oauth` (optional)
**Value**: High - Small business standard
**Key Features**:
- Parse chart of accounts
- Handle invoices and bills
- Extract customer/vendor data
- Preserve journal entries
- Handle payroll data

**Importance Signals**:
- Account balance
- Transaction amount
- Customer/vendor volume
- Aging status

---

## 5. Health & Fitness

Health tracking, fitness, and medical data.

### AppleHealthSpinner
**Purpose**: Ingest Apple Health exports
**Input Formats**: `export.xml` from Apple Health
**Complexity**: Medium
**Dependencies**: None (XML parsing)
**Value**: High - iOS health ecosystem
**Key Features**:
- Parse workout data
- Handle vital signs (heart rate, blood pressure)
- Extract sleep data
- Preserve source device information
- Handle clinical records (if exported)

**Importance Signals**:
- Data recency
- Measurement type (vitals > steps typically)
- Source reliability (medical device > phone)
- Anomaly detection (unusual readings)

---

### StravaSpinner
**Purpose**: Ingest Strava activities and routes
**Input Formats**: GPX exports, FIT files, API responses
**Complexity**: Medium
**Dependencies**: `fitparse`, `gpxpy`
**Value**: Medium - Athletes and fitness enthusiasts
**Key Features**:
- Parse activity streams (GPS, heart rate, power)
- Handle segments and achievements
- Extract route information
- Preserve gear/equipment data
- Handle club activities

**Importance Signals**:
- Activity type
- Personal records
- Kudos count
- Segment performance
- Training load

---

### FHIRSpinner
**Purpose**: Ingest FHIR (Fast Healthcare Interoperability Resources) data
**Input Formats**: FHIR JSON/XML bundles
**Complexity**: High
**Dependencies**: `fhirclient`
**Value**: High - Healthcare interoperability standard
**Key Features**:
- Parse FHIR resources (Patient, Observation, Medication, etc.)
- Handle references between resources
- Extract coded values (SNOMED, LOINC, ICD)
- Preserve provenance
- Handle bundles and transactions

**Importance Signals**:
- Resource type (Condition > Observation typically)
- Clinical status
- Effective date
- Code criticality

---

### MyFitnessPalSpinner
**Purpose**: Ingest nutrition and food logging data
**Input Formats**: CSV exports, API responses
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: Medium - Nutrition tracking
**Key Features**:
- Parse food diary entries
- Handle macro/micronutrient data
- Extract exercise logs
- Preserve goals and targets
- Handle custom foods

**Importance Signals**:
- Calorie significance
- Macro balance
- Meal timing
- Goal adherence

---

### GarminSpinner
**Purpose**: Ingest Garmin Connect data
**Input Formats**: FIT files, Garmin export ZIP
**Complexity**: Medium
**Dependencies**: `fitparse`
**Value**: Medium-High - Popular fitness devices
**Key Features**:
- Parse activity data
- Handle device metrics (VO2 max, training status)
- Extract sleep and stress data
- Preserve course/route data
- Handle multi-sport activities

**Importance Signals**:
- Activity type
- Training effect
- Device type
- Performance metrics

---

## 6. Design & Creative Tools

Design software and creative tool exports.

### FigmaSpinner
**Purpose**: Ingest Figma design files and components
**Input Formats**: REST API JSON, `.fig` files (limited)
**Complexity**: Medium-High
**Dependencies**: None (API-based)
**Value**: High - Design industry standard
**Key Features**:
- Parse frame/component hierarchy
- Extract text content from designs
- Handle design tokens (colors, typography)
- Preserve comments and annotations
- Handle component variants

**Importance Signals**:
- Page/frame depth
- Component usage count
- Comment count
- Last modified date
- Version history

---

### DesignTokensSpinner
**Purpose**: Ingest design token files
**Input Formats**: JSON (Style Dictionary), YAML, Figma Tokens
**Complexity**: Low
**Dependencies**: None
**Value**: Medium - Design systems
**Key Features**:
- Parse token hierarchies
- Handle aliases and references
- Extract color, typography, spacing tokens
- Preserve token descriptions
- Handle platform-specific values

**Importance Signals**:
- Token usage frequency
- Reference count
- Category (color > spacing typically)
- Description presence

---

### SketchSpinner
**Purpose**: Ingest Sketch design files
**Input Formats**: `.sketch` (ZIP with JSON)
**Complexity**: Medium
**Dependencies**: None (ZIP + JSON parsing)
**Value**: Medium - macOS design tool
**Key Features**:
- Parse artboard hierarchy
- Extract text layers
- Handle symbols and styles
- Preserve layer metadata
- Handle prototyping links

**Importance Signals**:
- Artboard importance
- Symbol usage count
- Layer depth
- Export settings

---

### AdobeXDSpinner
**Purpose**: Ingest Adobe XD design files
**Input Formats**: `.xd` files (proprietary, limited), API
**Complexity**: High
**Dependencies**: Adobe API access
**Value**: Medium - Enterprise design tool
**Key Features**:
- Parse artboard structure
- Extract text and assets
- Handle components
- Preserve prototyping flows
- Handle comments

**Importance Signals**:
- Artboard hierarchy
- Component usage
- Prototype complexity
- Comment count

---

## 7. Location & Geospatial

GPS tracks, location history, and geographic data.

### GPXSpinner
**Purpose**: Ingest GPS Exchange Format files
**Input Formats**: `.gpx`
**Complexity**: Low
**Dependencies**: `gpxpy`
**Value**: Medium - Outdoor activities, travel
**Key Features**:
- Parse tracks, routes, and waypoints
- Handle elevation data
- Extract timestamps and speeds
- Preserve metadata (name, description)
- Handle track segments

**Importance Signals**:
- Track length
- Elevation gain
- Waypoint count
- Named locations

---

### GoogleTimelineSpinner
**Purpose**: Ingest Google Location History/Timeline
**Input Formats**: JSON (Google Takeout), KML
**Complexity**: Medium
**Dependencies**: None
**Value**: High - Android users with location history
**Key Features**:
- Parse location points with timestamps
- Handle place visits
- Extract activity segments (driving, walking)
- Preserve confidence levels
- Handle semantic locations

**Importance Signals**:
- Visit duration
- Place category
- Frequency of visits
- Confidence level

---

### GeoJSONSpinner
**Purpose**: Ingest GeoJSON geographic data
**Input Formats**: `.geojson`, `.json`
**Complexity**: Low
**Dependencies**: None
**Value**: Medium - Geographic data interchange
**Key Features**:
- Parse geometry types (Point, LineString, Polygon, etc.)
- Handle feature properties
- Extract coordinate systems
- Preserve feature collections
- Handle topology

**Importance Signals**:
- Geometry complexity
- Property richness
- Feature count
- Named features

---

### KMLSpinner
**Purpose**: Ingest Keyhole Markup Language files
**Input Formats**: `.kml`, `.kmz`
**Complexity**: Low-Medium
**Dependencies**: None (XML parsing)
**Value**: Medium - Google Earth, mapping
**Key Features**:
- Parse placemarks and paths
- Handle styles and icons
- Extract descriptions (HTML)
- Preserve folder structure
- Handle time-based features

**Importance Signals**:
- Folder depth
- Description richness
- Style presence
- Time span

---

## 8. DevOps & Infrastructure

Infrastructure as code, CI/CD, and operations.

### TerraformSpinner
**Purpose**: Ingest Terraform configurations and state
**Input Formats**: `.tf` files, `.tfstate` JSON
**Complexity**: Medium
**Dependencies**: `python-hcl2`
**Value**: High - Infrastructure as Code standard
**Key Features**:
- Parse HCL2 resource definitions
- Handle modules and dependencies
- Extract variable definitions
- Preserve provider configurations
- Handle state file resources

**Importance Signals**:
- Resource type (compute > storage typically)
- Dependency depth
- Provider criticality
- State drift detection

---

### KubernetesSpinner
**Purpose**: Ingest Kubernetes manifests and cluster state
**Input Formats**: YAML manifests, `kubectl` JSON output
**Complexity**: Medium
**Dependencies**: `kubernetes` (optional for live cluster)
**Value**: High - Container orchestration standard
**Key Features**:
- Parse resource manifests (Deployment, Service, etc.)
- Handle Helm charts
- Extract labels and annotations
- Preserve namespace structure
- Handle CRDs

**Importance Signals**:
- Resource kind (Pod < Deployment < StatefulSet)
- Replica count
- Resource requests/limits
- Label importance

---

### GitHubActionsSpinner
**Purpose**: Ingest GitHub Actions workflows and runs
**Input Formats**: YAML workflow files, API responses
**Complexity**: Low-Medium
**Dependencies**: None (YAML parsing)
**Value**: High - CI/CD for GitHub projects
**Key Features**:
- Parse workflow triggers and jobs
- Handle matrix builds
- Extract secrets and variables
- Preserve run history
- Handle reusable workflows

**Importance Signals**:
- Workflow criticality (deploy > test > lint)
- Run frequency
- Success rate
- Job duration

---

### DockerfileSpinner
**Purpose**: Ingest Dockerfiles and docker-compose files
**Input Formats**: `Dockerfile`, `docker-compose.yml`
**Complexity**: Low
**Dependencies**: None
**Value**: Medium-High - Container standard
**Key Features**:
- Parse Dockerfile instructions
- Handle multi-stage builds
- Extract service definitions
- Preserve environment variables
- Handle build args

**Importance Signals**:
- Image size impact
- Layer count
- Security (FROM base)
- Exposed ports

---

### AnsibleSpinner
**Purpose**: Ingest Ansible playbooks and roles
**Input Formats**: YAML playbooks, roles
**Complexity**: Medium
**Dependencies**: None (YAML parsing)
**Value**: Medium - Configuration management
**Key Features**:
- Parse playbook tasks
- Handle role structure
- Extract variables and templates
- Preserve inventory information
- Handle handlers

**Importance Signals**:
- Task criticality
- Role reuse
- Handler importance
- Variable scope

---

### PrometheusSpinner
**Purpose**: Ingest Prometheus alerting rules and dashboards
**Input Formats**: YAML rules, Grafana JSON
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: Medium-High - Monitoring standard
**Key Features**:
- Parse alert definitions
- Handle recording rules
- Extract PromQL queries
- Preserve dashboard panels
- Handle silences

**Importance Signals**:
- Alert severity
- Query complexity
- Dashboard usage
- Alert frequency

---

## 9. Creative Writing & Media

Writing tools, screenplays, and media production.

### ScrivenerSpinner
**Purpose**: Ingest Scrivener writing projects
**Input Formats**: `.scriv` (directory), `.scrivx` XML
**Complexity**: Medium
**Dependencies**: None (XML parsing)
**Value**: Medium - Writers and authors
**Key Features**:
- Parse binder structure
- Handle manuscript vs research
- Extract character/setting sheets
- Preserve snapshots
- Handle compilation settings

**Importance Signals**:
- Document status (draft, final)
- Label/status
- Target word count
- Last modified

---

### FountainSpinner
**Purpose**: Ingest Fountain screenplay format
**Input Formats**: `.fountain` plaintext
**Complexity**: Low
**Dependencies**: None (regex parsing)
**Value**: Medium - Screenwriters
**Key Features**:
- Parse scene headings
- Handle character dialogue
- Extract action lines
- Preserve transitions
- Handle dual dialogue

**Importance Signals**:
- Scene count
- Character frequency
- Dialogue density
- Scene importance (INT/EXT)

---

### SubtitleSpinner
**Purpose**: Ingest subtitle files
**Input Formats**: `.srt`, `.vtt`, `.ass`, `.ssa`
**Complexity**: Low
**Dependencies**: `pysrt` or `webvtt-py`
**Value**: Medium - Video content
**Key Features**:
- Parse timing information
- Handle multiple speakers
- Extract styling (ASS/SSA)
- Preserve line breaks
- Handle multiple tracks

**Importance Signals**:
- Timing precision
- Speaker identification
- Styling presence
- Duration

---

### PodcastRSSSpinner
**Purpose**: Ingest podcast RSS feeds
**Input Formats**: RSS XML
**Complexity**: Low
**Dependencies**: `feedparser`
**Value**: Medium - Podcast metadata
**Key Features**:
- Parse episode metadata
- Handle show notes
- Extract duration and file info
- Preserve chapter markers
- Handle podcast categories

**Importance Signals**:
- Episode duration
- Download count (if available)
- Description length
- Chapter presence

---

### MuseScoreSpinner
**Purpose**: Ingest MuseScore music notation
**Input Formats**: `.mscz` (ZIP with XML)
**Complexity**: Medium
**Dependencies**: None (XML parsing)
**Value**: Low-Medium - Musicians
**Key Features**:
- Parse musical notation
- Handle instrument parts
- Extract lyrics
- Preserve dynamics and articulations
- Handle multiple voices

**Importance Signals**:
- Part count
- Complexity (tempo changes, key changes)
- Lyrics presence
- Duration

---

## 10. Analytics & Monitoring

Analytics platforms and monitoring tools.

### GoogleAnalyticsSpinner
**Purpose**: Ingest Google Analytics reports
**Input Formats**: CSV exports, API responses
**Complexity**: Medium
**Dependencies**: `google-api-python-client` (optional)
**Value**: High - Web analytics standard
**Key Features**:
- Parse dimension/metric reports
- Handle segments
- Extract user flows
- Preserve date ranges
- Handle custom dimensions

**Importance Signals**:
- Metric value
- Trend direction
- Segment size
- Goal completion

---

### MixpanelSpinner
**Purpose**: Ingest Mixpanel analytics exports
**Input Formats**: JSON exports, API responses
**Complexity**: Medium
**Dependencies**: `mixpanel` (optional)
**Value**: Medium-High - Product analytics
**Key Features**:
- Parse event data
- Handle user properties
- Extract funnel definitions
- Preserve cohort information
- Handle formulas

**Importance Signals**:
- Event frequency
- Funnel step importance
- User segment size
- Property completeness

---

### DatadogSpinner
**Purpose**: Ingest Datadog metrics, logs, and dashboards
**Input Formats**: JSON exports, API responses
**Complexity**: Medium-High
**Dependencies**: `datadog-api-client` (optional)
**Value**: High - Enterprise monitoring
**Key Features**:
- Parse metric definitions
- Handle log patterns
- Extract dashboard configurations
- Preserve monitors and alerts
- Handle APM traces

**Importance Signals**:
- Alert severity
- Metric criticality
- Log level
- Dashboard usage

---

### SentrySpinner
**Purpose**: Ingest Sentry error tracking exports
**Input Formats**: JSON exports, API responses
**Complexity**: Medium
**Dependencies**: `sentry-sdk` (optional)
**Value**: High - Error tracking standard
**Key Features**:
- Parse error stack traces
- Handle issue grouping
- Extract breadcrumbs
- Preserve user context
- Handle release information

**Importance Signals**:
- Error frequency
- User impact
- Error severity
- Stack trace depth

---

## 11. Research & Academia

Academic papers, citations, and research tools.

### ZoteroSpinner
**Purpose**: Ingest Zotero library exports
**Input Formats**: RDF, BibTeX, JSON export
**Complexity**: Medium
**Dependencies**: None
**Value**: High - Academic researchers
**Key Features**:
- Parse citation metadata
- Handle attachments (PDFs)
- Extract tags and collections
- Preserve notes
- Handle related items

**Importance Signals**:
- Citation count
- Item type (article > webpage)
- Collection membership
- Annotation count

---

### MendeleySpinner
**Purpose**: Ingest Mendeley reference exports
**Input Formats**: BibTeX, RIS, JSON API
**Complexity**: Medium
**Dependencies**: None
**Value**: Medium - Reference management
**Key Features**:
- Parse reference metadata
- Handle group libraries
- Extract annotations
- Preserve folders
- Handle author disambiguation

**Importance Signals**:
- Read status
- Star rating
- Annotation count
- Folder importance

---

### ArXivSpinner
**Purpose**: Ingest arXiv paper metadata
**Input Formats**: OAI-PMH XML, API responses
**Complexity**: Low-Medium
**Dependencies**: `arxiv` package (optional)
**Value**: Medium-High - ML/Physics researchers
**Key Features**:
- Parse paper metadata
- Handle categories and subjects
- Extract abstract text
- Preserve version history
- Handle author affiliations

**Importance Signals**:
- Citation count (external)
- Category relevance
- Publication date
- Author h-index

---

### JupyterNotebookSpinner
**Purpose**: Enhanced Jupyter notebook processing (extends existing)
**Input Formats**: `.ipynb`
**Complexity**: Low (already exists, enhance)
**Dependencies**: None
**Value**: High - Data science standard
**Key Features**:
- Better cell type handling
- Execute output parsing
- Widget state extraction
- Kernel metadata
- Magic command handling

**Importance Signals**:
- Output presence
- Code cell complexity
- Markdown documentation
- Widget state

---

## 12. Social Media

Social platform exports and APIs.

### TwitterSpinner
**Purpose**: Ingest Twitter/X archive exports
**Input Formats**: Twitter archive ZIP (JSON)
**Complexity**: Medium
**Dependencies**: None
**Value**: Medium-High - Social media presence
**Key Features**:
- Parse tweets and threads
- Handle likes and bookmarks
- Extract media references
- Preserve DMs (if exported)
- Handle lists

**Importance Signals**:
- Engagement (likes, retweets)
- Thread length
- Media presence
- Follower interaction

---

### LinkedInSpinner
**Purpose**: Ingest LinkedIn data exports
**Input Formats**: ZIP export (CSV files)
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: Medium-High - Professional network
**Key Features**:
- Parse profile information
- Handle connections
- Extract messages
- Preserve posts and articles
- Handle endorsements/recommendations

**Importance Signals**:
- Connection count
- Endorsement count
- Post engagement
- Profile completeness

---

### RedditSpinner
**Purpose**: Ingest Reddit account data
**Input Formats**: GDPR export, API responses
**Complexity**: Medium
**Dependencies**: `praw` (optional)
**Value**: Medium - Reddit users
**Key Features**:
- Parse posts and comments
- Handle saved content
- Extract upvotes/karma
- Preserve subreddit context
- Handle awards

**Importance Signals**:
- Karma score
- Comment depth
- Subreddit authority
- Award count

---

### MastodonSpinner
**Purpose**: Ingest Mastodon/ActivityPub exports
**Input Formats**: ActivityPub JSON, CSV exports
**Complexity**: Medium
**Dependencies**: `Mastodon.py` (optional)
**Value**: Medium - Fediverse users
**Key Features**:
- Parse toots and boosts
- Handle followers/following
- Extract media
- Preserve content warnings
- Handle instance context

**Importance Signals**:
- Boost count
- Reply depth
- Instance authority
- Favorite count

---

## 13. Gaming & Entertainment

Gaming platforms and entertainment services.

### SteamSpinner
**Purpose**: Ingest Steam library and activity
**Input Formats**: Steam API responses, profile exports
**Complexity**: Medium
**Dependencies**: None (API-based)
**Value**: Medium - PC gamers
**Key Features**:
- Parse game library
- Handle playtime statistics
- Extract achievements
- Preserve review content
- Handle friend activity

**Importance Signals**:
- Playtime hours
- Achievement completion
- Review score
- Last played

---

### GoodreadsSpinner
**Purpose**: Ingest Goodreads library exports
**Input Formats**: CSV export
**Complexity**: Low
**Dependencies**: None
**Value**: Medium - Book readers
**Key Features**:
- Parse book metadata
- Handle shelves and tags
- Extract ratings and reviews
- Preserve reading dates
- Handle reading challenges

**Importance Signals**:
- Personal rating
- Average rating
- Review length
- Read date

---

### SpotifySpinner
**Purpose**: Ingest Spotify listening history
**Input Formats**: JSON (Extended streaming history), API
**Complexity**: Medium
**Dependencies**: `spotipy` (optional)
**Value**: Medium - Music listeners
**Key Features**:
- Parse listening history
- Handle playlists
- Extract audio features
- Preserve saved albums/tracks
- Handle podcast listens

**Importance Signals**:
- Play count
- Skip rate
- Playlist inclusion
- Save status

---

### YouTubeHistorySpinner
**Purpose**: Ingest YouTube watch history
**Input Formats**: Google Takeout JSON/HTML
**Complexity**: Low-Medium
**Dependencies**: None
**Value**: Medium-High - YouTube users
**Key Features**:
- Parse watch history
- Handle search history
- Extract liked videos
- Preserve comments
- Handle subscriptions

**Importance Signals**:
- Watch percentage
- Like status
- Comment presence
- Channel subscription

---

## 14. Legal & Compliance

Legal documents and compliance frameworks.

### ContractSpinner
**Purpose**: Ingest legal contracts and agreements
**Input Formats**: PDF, DOCX with contract structure
**Complexity**: High
**Dependencies**: `pdfplumber`, `python-docx`
**Value**: Medium-High - Legal/business use
**Key Features**:
- Parse contract sections
- Handle signature blocks
- Extract defined terms
- Preserve exhibit references
- Handle amendment tracking

**Importance Signals**:
- Contract value
- Expiration proximity
- Amendment count
- Clause criticality

---

### PolicyDocumentSpinner
**Purpose**: Ingest compliance policies (SOC2, HIPAA, etc.)
**Input Formats**: PDF, DOCX, Markdown
**Complexity**: Medium
**Dependencies**: Document parsers
**Value**: Medium - Compliance teams
**Key Features**:
- Parse policy sections
- Handle control mappings
- Extract requirements
- Preserve version history
- Handle exception tracking

**Importance Signals**:
- Control criticality
- Compliance framework
- Last review date
- Exception count

---

### PatentSpinner
**Purpose**: Ingest patent documents
**Input Formats**: USPTO XML, EPO XML, PDF
**Complexity**: High
**Dependencies**: XML parsers
**Value**: Medium - IP management
**Key Features**:
- Parse claims structure
- Handle drawing references
- Extract prior art citations
- Preserve classification codes
- Handle family relationships

**Importance Signals**:
- Claim count
- Citation count
- Priority date
- Classification relevance

---

## Priority Matrix

### Tier 1: High Value, Medium Complexity (Recommended First)
| Spinner | Value | Complexity | Dependencies |
|---------|-------|------------|--------------|
| NotionSpinner | High | Medium | Optional API |
| ObsidianSpinner | High | Medium | None |
| CalendarSpinner | High | Low-Medium | `icalendar` |
| GoogleAnalyticsSpinner | High | Medium | Optional API |
| AppleHealthSpinner | High | Medium | None |
| TerraformSpinner | High | Medium | `python-hcl2` |
| KubernetesSpinner | High | Medium | Optional API |
| ZoteroSpinner | High | Medium | None |

### Tier 2: High Value, High Complexity (Investment Required)
| Spinner | Value | Complexity | Dependencies |
|---------|-------|------------|--------------|
| FigmaSpinner | High | Medium-High | API access |
| ConfluenceSpinner | High | Medium-High | Optional API |
| JiraSpinner | High | Medium-High | `jira` SDK |
| FHIRSpinner | High | High | `fhirclient` |
| DatadogSpinner | High | Medium-High | API client |

### Tier 3: Medium Value, Low Complexity (Quick Wins)
| Spinner | Value | Complexity | Dependencies |
|---------|-------|------------|--------------|
| GPXSpinner | Medium | Low | `gpxpy` |
| SubtitleSpinner | Medium | Low | `pysrt` |
| DesignTokensSpinner | Medium | Low | None |
| TodoistSpinner | Medium | Low | Optional API |
| GoodreadsSpinner | Medium | Low | None |

### Tier 4: Niche but Valuable
| Spinner | Value | Complexity | Target Audience |
|---------|-------|------------|-----------------|
| RoamSpinner | Medium | Medium | PKM enthusiasts |
| AnkiSpinner | Medium | Low-Medium | Students/Learners |
| FountainSpinner | Medium | Low | Screenwriters |
| MuseScoreSpinner | Low-Medium | Medium | Musicians |
| PatentSpinner | Medium | High | IP professionals |

---

## Implementation Guidelines

### BaseSpinner Protocol Requirements
Every new spinner must implement:

```python
class NewSpinner(BaseSpinner):
    def __init__(self, config: Optional[SpinnerConfig] = None):
        super().__init__(name="new_spinner", config=config)

    async def _spin_impl(self, data: Dict[str, Any]) -> SpinResult:
        """Core implementation - transform data to MemoryShards"""
        pass

    def get_capabilities(self) -> SpinnerCapabilities:
        """Declare what this spinner can do"""
        pass

    def is_available(self) -> bool:
        """Check if dependencies are available"""
        pass
```

### 9-Signal Importance Scoring
Each spinner should implement importance scoring using available signals:

1. **length_score**: Content length (logarithmic)
2. **technical_score**: Technical terminology density
3. **structural_score**: Heading/section presence
4. **authority_score**: Source authority signals
5. **recency_score**: Temporal decay
6. **engagement_score**: User interaction metrics
7. **reference_score**: Citation/link density
8. **noise_penalty**: Boilerplate/noise detection
9. **custom_signals**: Domain-specific signals

### Testing Requirements
- Unit tests with mock data
- Integration tests with sample files
- Edge case handling (empty, malformed, large files)
- Performance benchmarks for large inputs
- Optional dependency graceful degradation

### Documentation Template
```markdown
## [SpinnerName]

### Overview
Brief description of what this spinner does.

### Input Formats
- Format 1: Description
- Format 2: Description

### Usage
\`\`\`python
from hololoom.spinningWheel import SpinnerName
spinner = SpinnerName()
result = await spinner.spin(data)
\`\`\`

### Dependencies
- Required: None
- Optional: package_name (for feature X)

### Importance Signals
Description of how importance is calculated.

### Examples
Sample input/output examples.
```

---

## Contributing

When implementing a new spinner:

1. **Open an issue** describing the spinner and use case
2. **Check existing spinners** for similar patterns to follow
3. **Implement with tests** in `hololoom/spinningWheel/`
4. **Update this document** to move from "Future" to implemented
5. **Submit PR** with example usage in demos/

---

## Version History

- **2025-12-23**: Initial document creation with 50+ potential spinners
- Categories: Knowledge Management, Productivity, Communication, Financial, Health, Design, Location, DevOps, Creative, Analytics, Research, Social, Gaming, Legal

---

*"If you need to configure it, we failed." - SpinningWheel Philosophy*
