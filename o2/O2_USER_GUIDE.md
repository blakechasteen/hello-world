# O2 User Guide

**Platform Anarchism for Everyone**

Welcome to O2! This guide will help you get started with democratic, user-owned collaboration.

---

## Table of Contents

1. [What is O2?](#what-is-o2)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Commands Reference](#commands-reference)
5. [Governance](#governance)
6. [Memory & Data](#memory--data)
7. [Agentic Swarms](#agentic-swarms)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)
10. [FAQs](#faqs)

---

## What is O2?

**O2** is a platform that combines:
- **Matrix.org** - Decentralized, federated messaging
- **HoloLoom** - Agentic AI reasoning and memory
- **Platform Anarchism** - User data ownership, democratic governance

Unlike traditional platforms (Slack, Discord, Teams), O2:
- ✅ **You own your data** (export anytime, zero friction)
- ✅ **Community governs democratically** (voting, not admin fiat)
- ✅ **AI serves you** (transparent reasoning, no manipulation)
- ✅ **Federated** (run your own instance or choose any community)
- ✅ **Open source** (MIT license, inspect and modify everything)

---

## Quick Start

### 1. Join an O2 Instance

**Option A: Use Public Instance** (easiest):
1. Create Matrix account at https://app.element.io
2. Join community room (e.g., `#o2-community:matrix.org`)
3. Bot auto-joins when room created or invited
4. Send: `@o2 help`

**Option B: Self-Host** (full control):
1. Follow [O2_PLATFORM_ARCHITECTURE.md](O2_PLATFORM_ARCHITECTURE.md#deployment-architecture)
2. Run `./setup.sh` (takes ~5 minutes)
3. Register bot user
4. Invite bot to your room

### 2. First Commands

Try these to get started:

```
@o2 help
```
Shows all available commands

```
@o2 what is platform anarchism?
```
Ask HoloLoom any question

```
@o2 propose: welcome new members policy
```
Create your first governance proposal

### 3. Understanding Responses

O2 responds with **structured messages**:

```
💭 Query: What is Thompson Sampling?

1. Thompson Sampling is a Bayesian approach to balancing exploration and exploitation...
2. It uses Beta distributions to model uncertainty...
3. Applications include A/B testing, reinforcement learning...
```

Icons guide you:
- 💭 - Query/answer
- 📋 - Governance proposal
- 🤖 - Swarm activity
- ✅ - Success
- ❌ - Error
- 🚧 - Feature in development

---

## Core Concepts

### Platform Anarchism

**Power to users, not platforms.**

Traditional platforms:
- Company owns your data
- Admins make all decisions
- Opaque algorithms control what you see
- Vendor lock-in (can't leave easily)

O2's approach:
- **You own your data** - Export anytime, import anywhere
- **Community decides** - Vote on policies, no dictators
- **Transparent AI** - Full reasoning audit trails
- **Federation** - Run your own instance or migrate freely

### Federated Memory

Each user has their own **isolated HoloLoom instance**:

```
You (@alice:matrix.org)
  ↓
Your Private Memory Graph
  ├── Knowledge you've learned
  ├── Conversations you've had
  ├── Photos and files you've stored
  └── Complete audit trail
```

**Key features**:
- 🔒 **Private by default** - Others can't see your memories
- 📤 **Export anytime** - Full data portability
- 🤝 **Selective sharing** - Share specific memories with explicit consent
- 🔑 **You control access** - Revoke anytime

### Democratic Governance

Communities govern via **transparent voting**:

**Proposal → Vote → Execute**

Anyone can propose changes. Community votes. Approved proposals auto-execute.

**Example**:
```
Alice: @o2 propose: require code review for deployments
[Community votes via reactions]
Result: PASSED (12 yes, 3 no)
[Policy automatically updated]
```

### Agentic Swarms

Multiple specialized AI agents collaborate on complex tasks:

```
You: @o2 run swarm: deploy authentication service

O2 Swarm:
  ✅ QA Agent: Tests passing (124/124)
  ✅ Security Agent: No vulnerabilities
  🔄 Deployment Agent: Rolling out...
  ⏳ Monitoring Agent: Watching metrics...
  📝 Documentation Agent: Updating docs...
```

**Agent types**:
- **QA** - Code review, testing (trough/xTerminator integration)
- **Security** - Vulnerability scanning
- **Deployment** - Automated rollouts
- **Monitoring** - Health checks, metrics
- **Documentation** - Changelog, README generation
- **Research** - Analysis, synthesis

---

## Commands Reference

### Basic Commands

**Help**:
```
@o2 help
```
Show all commands and features

**Ask Questions**:
```
@o2 [your question]
@o2 what is platform anarchism?
@o2 explain Thompson Sampling
```
HoloLoom answers using its knowledge graph

**Research Mode**:
```
@o2 research: [topic]
@o2 research: decentralized governance
```
Deep research with multi-query exploration

**Verify Claims**:
```
@o2 verify: [claim]
@o2 verify: blockchain is energy-efficient
```
Fact-check claims for contradictions

### Governance Commands

**Create Proposal**:
```
@o2 propose: [title]
@o2 propose: ban spam accounts
@o2 propose: require 2/3 majority for critical decisions
```

**Vote**:
```
@o2 vote yes
@o2 vote no
@o2 vote abstain
```
Or use reactions: ✅ (yes), ❌ (no), 🤔 (abstain)

**Check Results**:
```
@o2 tally votes
```
Shows current vote counts and outcome

### Memory Commands

**Export Data**:
```
@o2 export my data
```
Creates encrypted archive of your entire memory graph

**Share Memory** (coming soon):
```
@o2 share memory "[memory_id]" with @bob:matrix.org
```
Share specific memory with another user

**Revoke Access** (coming soon):
```
@o2 revoke access from @charlie:matrix.org
```
Revoke previously granted access

### Swarm Commands

**Run Swarm**:
```
@o2 run swarm: [task description]
@o2 run swarm: deploy new feature
@o2 run swarm: analyze security of codebase
```
Activates multi-agent collaboration

---

## Governance

### How Governance Works

**1. Proposal Creation**

Anyone can create a proposal:
```
@o2 propose: [title]
```

**Proposal includes**:
- Title (short description)
- Author (who proposed it)
- Type (policy, feature, governance, technical)
- Threshold (approval % needed, default 67%)
- Duration (voting period, default 72 hours)

**2. Voting Period**

Community members vote:
- React with ✅ (approve)
- React with ❌ (reject)
- React with 🤔 (abstain)

Or use text commands:
```
@o2 vote yes
```

**3. Tallying**

After voting period or manually:
```
@o2 tally votes
```

**Result calculation**:
- Yes %: yes votes / (yes + no votes)
- Abstentions don't count toward total
- Passes if yes % ≥ threshold

**4. Execution**

Approved proposals **auto-execute**:
- Policy changes: Updated room settings
- Feature changes: Enabled/disabled features
- Governance changes: New voting rules
- Technical changes: Configuration updates

### Governance Best Practices

**Clear Proposals**:
```
❌ Bad: "Change things"
✅ Good: "Require code review for all production deployments"
```

**Appropriate Thresholds**:
- **Low-risk**: 50% (simple majority)
- **Medium-risk**: 67% (2/3 supermajority)
- **High-risk**: 75% (3/4 supermajority)
- **Critical**: 90% (near-consensus)

**Discussion Before Voting**:
- Discuss proposal in room first
- Address concerns and objections
- Consider amendments
- Then create formal proposal

**Reversibility**:
- Bad decisions can be undone
- Create new proposal to reverse previous one
- Complete audit trail maintained

---

## Memory & Data

### Your Memory Graph

Your HoloLoom instance stores:

**Knowledge Graph**:
- Entities (concepts, people, events)
- Relationships (uses, mentions, leads to)
- Temporal data (when you learned it)

**Embeddings**:
- 228D semantic projections
- 16 interpretable axes (sentiment, formality, etc.)
- Multi-scale representations (96, 192, 384 dimensions)

**Photos & Media**:
- Images with CLIP embeddings
- Visual compression (5-20x token savings)
- Captions and alt-text

**Audit Trail**:
- Complete provenance of all data
- When added, by whom, why
- All access grants/revokes

### Data Export

**Export Process**:
```
@o2 export my data
```

**You receive**:
```
alice_2025-11-17_14-30-00.o2.json
```

**Archive contains**:
- `graph.json` - Knowledge graph (entities, relationships)
- `embeddings.npy` - Vector embeddings
- `photos/` - All images and metadata
- `audit_trail.jsonl` - Complete provenance
- `metadata.json` - Config, version, timestamp

**File is**:
- Encrypted with your password
- Portable across O2 instances
- Standard JSON format (human-readable)
- Includes everything (zero data loss)

### Data Import

**Moving to new instance**:
1. Export from old instance: `@o2 export my data`
2. Download archive
3. Join new instance
4. Upload archive: `@o2 import data from alice_backup.o2.json`
5. Done! All memories restored

**No vendor lock-in. Full portability.**

### Memory Sharing

**Coming soon**: Selective memory sharing

**How it will work**:
1. You choose specific memory to share
2. Grant permissions (read-only or read-write)
3. Set expiration (7 days, 30 days, forever)
4. Recipient can access via their HoloLoom
5. You can revoke anytime

**Privacy guarantees**:
- Explicit consent required (no implicit sharing)
- Granular permissions (memory-level, not graph-level)
- Time-limited access (auto-expiration)
- Complete audit trail (who accessed when)

---

## Agentic Swarms

### What Are Swarms?

**Swarms** = Multiple specialized AI agents collaborating on complex tasks.

**Instead of**:
- Single AI agent doing everything
- User manually coordinating steps
- Sequential, slow execution

**Swarms provide**:
- Specialized agents (QA, security, deployment, etc.)
- Parallel execution (faster)
- Expert knowledge (each agent has domain expertise)
- Consensus (agents vote on ambiguous decisions)

### Using Swarms

**Basic syntax**:
```
@o2 run swarm: [task description]
```

**Examples**:

**Deploy Feature**:
```
@o2 run swarm: deploy authentication service

O2: 🤖 Swarm Activated

QA Agent: Running tests... ✅ 124/124 passing
Security Agent: Scanning... ✅ 0 critical vulnerabilities
Deployment Agent: Rolling out... 🔄 33% complete
Monitoring Agent: Waiting for deployment...
Documentation Agent: Generating changelog...

Estimated completion: 5 minutes
```

**Security Analysis**:
```
@o2 run swarm: analyze security of authentication module

O2: 🤖 Swarm Activated

Security Agent: Scanning for vulnerabilities...
QA Agent: Reviewing test coverage...
Research Agent: Analyzing best practices...

[Results after completion]
```

### Swarm Workflow

**1. Task Decomposition**:
- O2 breaks task into subtasks
- Assigns agents based on capabilities
- Determines execution order (parallel vs sequential)

**2. Agent Selection**:
- QA Agent: Testing, code review
- Security Agent: Vulnerability scanning
- Deployment Agent: Rollouts, monitoring
- Monitoring Agent: Health checks, metrics
- Documentation Agent: Changelog, README
- Research Agent: Analysis, synthesis

**3. Parallel Execution**:
- Independent tasks run concurrently
- Dependencies handled automatically
- Progress reported in real-time

**4. Consensus & Escalation**:
- Agents vote on ambiguous decisions
- Critical decisions escalate to humans
- Complete audit trail maintained

**5. Result Aggregation**:
- Results from all agents combined
- Summary and recommendations provided
- Next steps suggested

### Human-in-Loop

**Critical decisions require human approval**:

```
@o2 run swarm: deploy to production

O2: 🤖 Swarm Complete

Summary:
  ✅ QA Agent: All tests passing
  ⚠️ Security Agent: 2 medium vulnerabilities detected
  ⏸️ Deployment Agent: Waiting for approval

⚠️ Human approval required:
  - 2 medium vulnerabilities detected
  - Production deployment is high-risk

Approve: `@o2 approve deployment`
Reject: `@o2 reject deployment`
```

**You stay in control. AI assists, doesn't decide.**

---

## Advanced Topics

### Running Your Own Instance

See [O2_PLATFORM_ARCHITECTURE.md](O2_PLATFORM_ARCHITECTURE.md) for complete deployment guide.

**Quick overview**:
1. Install Docker + Docker Compose
2. Clone O2 repository
3. Run `./setup.sh`
4. Configure domain (production) or use localhost (dev)
5. Register bot user
6. Invite bot to rooms

**Benefits of self-hosting**:
- Full data control
- Custom policies
- Instance-level features
- Federation with other instances

### Federation

**O2 instances federate** via Matrix protocol:

```
Your Instance (matrix.example.com)
  ↔ Federation ↔
Friend's Instance (matrix.friend.org)
  ↔ Federation ↔
Community Instance (matrix.community.org)
```

**What federation means**:
- Users on different instances can chat
- Instances remain independent
- Each instance has own policies
- Data stays with home instance

**Cross-instance memory sharing** (coming soon):
- Share memories with users on other instances
- Encrypted end-to-end
- Explicit consent required
- Complete audit trail

### Customization

**Per-Room Settings**:
- Governance thresholds (majority, 2/3, consensus)
- Voting durations (24h, 72h, 1 week)
- Swarm agents enabled (QA only, full swarm, etc.)
- Memory sharing policies (allowed, restricted, disabled)

**Instance-Level Settings**:
- LLM backend (Ollama local, Claude, GPT)
- Memory backend (in-memory, Neo4j + Qdrant)
- Alignment framework (safety level)
- Performance tuning (BARE/FAST/FUSED modes)

### Integration

**O2 integrates with**:
- HoloLoom (agentic AI, knowledge graphs)
- Matrix (federated messaging)
- Trough/xTerminator (code QA)
- Neo4j (production knowledge graphs)
- Qdrant (production vector store)
- Ollama (local LLMs)

**API access** (coming soon):
- REST API for programmatic access
- Webhooks for automation
- Plugin system for extensions

---

## Troubleshooting

### Bot Doesn't Respond

**Check**:
1. Bot invited to room: `/invite @o2:your-server.org`
2. Bot has joined: Look for join message
3. Mention bot correctly: `@o2 help` (lowercase)

**Debug**:
```bash
# Check bot logs
docker-compose logs -f o2-bot

# Restart bot
docker-compose restart o2-bot
```

### Voting Not Working

**Check**:
1. Proposal is active (not expired)
2. Vote syntax correct: `@o2 vote yes`
3. Or use reactions: ✅ ❌ 🤔

**Debug**:
```
@o2 tally votes
```
Shows current vote counts

### Memory Export Fails

**Check**:
1. You have memories (try `@o2 what is platform anarchism?` first)
2. Disk space available
3. Permissions correct

**Debug**:
```bash
# Check disk space
docker exec o2-bot df -h

# Check permissions
docker exec o2-bot ls -la /app/memories
```

### Swarm Stuck

**Check**:
1. Task description clear
2. Required agents available
3. Network connectivity

**Debug**:
```bash
# Check swarm logs
docker-compose logs -f o2-bot | grep swarm

# Restart swarm
docker-compose restart o2-bot
```

---

## FAQs

### General

**Q: What makes O2 different from Slack/Discord?**

A: Three key differences:
1. **Data ownership** - You own your data (export anytime)
2. **Democratic governance** - Community votes, no admin dictators
3. **Federated** - Run your own instance or choose any community

**Q: Is O2 open source?**

A: Yes! MIT license. Code at https://github.com/your-org/o2-platform

**Q: Can I use O2 without self-hosting?**

A: Yes! Join any public O2 instance (like email - you can use Gmail or run your own server).

### Privacy & Security

**Q: Who can see my data?**

A: Only you. Your HoloLoom instance is isolated and encrypted. Others can't access without explicit permission.

**Q: Is Matrix secure?**

A: Yes. Matrix supports E2E encryption (like Signal). Your messages are encrypted end-to-end in encrypted rooms.

**Q: Can instance admins see my memories?**

A: No. Memories encrypted with keys derived from your user ID + password. Admins can't decrypt without your password.

### Governance

**Q: Who can create proposals?**

A: Anyone in the community. Platform anarchism means no gatekeepers.

**Q: Can proposals be vetoed?**

A: No. If a proposal passes the threshold, it executes automatically. That's the point - no admin veto power.

**Q: What if a bad proposal passes?**

A: Create a new proposal to reverse it. All changes auditable and reversible.

### Technical

**Q: What LLMs does O2 support?**

A: Ollama (local, privacy-first), Claude (Anthropic), GPT (OpenAI). Configurable per instance.

**Q: How much does it cost to run?**

A: Self-hosted: Server costs only (~$20-100/month depending on size). No per-user fees. Public instances: Usually free (community-funded).

**Q: Can I migrate between instances?**

A: Yes! Export your data (`@o2 export my data`), join new instance, import. Zero friction.

---

## What's Next?

Now that you understand O2, try:

1. **Ask questions** - `@o2 what is Thompson Sampling?`
2. **Create proposal** - `@o2 propose: welcome message for new members`
3. **Export data** - `@o2 export my data` (see what you get!)
4. **Run swarm** - `@o2 run swarm: research platform anarchism`

**Remember**: Platform Anarchism means YOU are in control. Own your data. Govern democratically. Build commons together.

---

**Questions? Join #o2-community:matrix.org**

**Made with ❤️ by the Platform Anarchism community**

🚀 **Decentralize. Democratize. Own your future.**
